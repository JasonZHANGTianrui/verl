"""窗口RL核心契约：实际可见前缀、原概率下标、监督一次和整轨迹梯度归一化。"""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import torch
from image_window import half_window_inputs, omitted_turns

from verl.experimental.physical_ai.image_context import SamplingWindow, context_views


class Tokenizer:
    """仅为真实模态边界提供固定ID，文字中的同ID仍必须保留。"""

    def encode(self, value, **_):
        return [90 if value == "<|vision_start|>" else 91]


def fixture():
    """20轮双图、state、不同长度输出；跨12/18边界，刻意加入文字视觉标记。"""
    ids, kinds, generated, pixels, calls = [7], [6], [False], {}, []
    for turn in range(20):
        for camera in range(2):
            ids.extend([90, 100, 100, 91])
            kinds.extend([4, 1, 1, 4])
            generated.extend([False] * 4)
            pixels[f"image_{2 * turn + camera:04d}"] = torch.tensor([turn, camera])
        ids.append(9)
        kinds.append(2)
        generated.append(False)
        start = len(ids)
        tokens = [90] + [200 + turn] * (turn % 5 + 1)
        ids.extend(tokens)
        kinds.extend([6] * len(tokens))
        generated.extend([True] * len(tokens))
        calls.append(
            dict(
                original_input_tokens=start, ids=tokens, visual_history=dict(omitted_turns=omitted_turns(turn + 1, 12))
            )
        )
    inputs = dict(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.tensor([kinds]),
        pixel_values=pixels,
        proprio=torch.arange(40).reshape(1, 20, 2),
    )
    artifact = dict(context_window=SamplingWindow(12, Tokenizer()).certificate(inputs, calls))
    return inputs, artifact, torch.tensor([generated[1:]]), calls


class ImageContextTests(unittest.TestCase):
    def test_action_qa_join_retains_all_action_gradients(self):
        """普通GRPO必须用两阶段完整证书；QA独有证书会遗漏动作生成目标。"""
        from image_window import image_layout, project_prefix

        from verl.experimental.physical_ai.image_context import join_phase_windows

        inputs, _, mask, calls = fixture()
        split = 13  # 动作已跨12轮删图，QA再跨18轮，覆盖两个不同边界。
        end = calls[split - 1]["original_input_tokens"] + len(calls[split - 1]["ids"])
        prefix, _ = project_prefix(inputs, None, image_layout(inputs, Tokenizer()), end=end, removed_turns=0)
        action = SamplingWindow(12, Tokenizer()).certificate(prefix, calls[:split])
        qa = SamplingWindow(12, Tokenizer()).certificate(inputs, calls[split:])
        joined = dict(context_window=join_phase_windows(action, qa))
        seen = [i for _, idx, active in context_views(inputs, joined, mask) for i in idx[active[0]].tolist()]
        self.assertEqual(sorted(seen), torch.where(mask[0])[0].tolist())
        self.assertEqual(len(seen), len(set(seen)))
        with self.assertRaises(ValueError):
            list(context_views(inputs, dict(context_window=qa), mask))
        wrong = deepcopy(qa)
        wrong["image_keys"] = ["other-image", *wrong["image_keys"][1:]]
        with self.assertRaises(ValueError):
            join_phase_windows(action, wrong)

    def test_seggrpo_window2_qa_masks_parent_actions(self):
        """SegGRPO的QA证书只含新生成；父动作仍在前缀中，但不重复形成QA梯度。"""
        inputs, _, mask, calls = fixture()
        for turn, call in enumerate(calls):
            call["visual_history"]["omitted_turns"] = omitted_turns(turn + 1, 2)
        split = 13
        boundary = calls[split]["original_input_tokens"]
        qa_mask = mask.clone()
        qa_mask[:, : boundary - 1] = False
        original_ids = inputs["input_ids"].clone()
        certificate = SamplingWindow(2, Tokenizer()).certificate(inputs, calls[split:])
        seen = []
        for actual, indices, active in context_views(inputs, dict(context_window=certificate), qa_mask):
            seen.extend(indices[active[0]].tolist())
            self.assertEqual(len(actual["pixel_values"]), 2)
            self.assertTrue(torch.equal(actual["proprio"], inputs["proprio"][:, : actual["proprio"].shape[1]]))
        self.assertEqual(sorted(seen), torch.where(qa_mask[0])[0].tolist())
        self.assertEqual(len(seen), len(set(seen)))
        self.assertTrue(all(index >= boundary - 1 for index in seen))
        self.assertTrue(torch.equal(inputs["input_ids"], original_ids))
        # 误把父动作也交给QA证书，必须拒绝，不能默默漏掉或重复监督。
        with self.assertRaises(ValueError):
            list(context_views(inputs, dict(context_window=certificate), mask))

    def test_window_views_cover_original_tokens_once(self):
        """动作下标不能随删图偏移；state/文字逐值保留，不把图片标记误删。"""
        inputs, artifact, mask, _ = fixture()
        seen = []
        for actual, indices, active in context_views(inputs, artifact, mask):
            seen.extend(indices[active[0]].tolist())
            self.assertTrue(torch.equal(actual["input_ids"][:, 1:], inputs["input_ids"][:, indices + 1]))
            self.assertEqual(int(active.sum()), int((actual["attention_mask"][:, 1:][active] == 6).sum()))
        self.assertEqual(sorted(seen), torch.where(mask[0])[0].tolist())
        self.assertEqual(len(seen), len(set(seen)))

    def test_bad_certificate_and_missing_segment_rejected(self):
        """漏掉早期监督或把后来的窗口用于早期动作，都必须拒绝更新。"""
        inputs, artifact, mask, _ = fixture()
        for change in ("wrong_window", "missing"):
            bad = deepcopy(artifact)
            if change == "wrong_window":
                bad["context_window"]["segments"][0]["removed_turns"] = 6
            else:
                bad["context_window"]["segments"].pop()
            with self.assertRaises(ValueError):
                list(context_views(inputs, bad, mask))

    def test_segment_weighting_matches_per_message_causal_gradient(self):
        """独立逐消息参考与合并窗口段的标量loss、梯度一致，不能对各段简单平均。"""
        inputs, artifact, mask, calls = fixture()
        theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        total = int(mask.sum())
        grouped = theta * 0
        for actual, indices, active in context_views(inputs, artifact, mask):
            features = actual["input_ids"].double().cumsum(1)[:, :-1] / 10000
            values = -torch.exp(theta * features)[active]
            grouped = grouped + values.mean() * int(active.sum()) / total
        grad = torch.autograd.grad(grouped, theta)[0]
        reference = theta * 0
        # 按实际每次请求独立投影，不读取分段证书。
        from image_window import image_layout, project_prefix

        layout = image_layout(inputs, Tokenizer())
        for call in calls:
            start = call["original_input_tokens"]
            end = start + len(call["ids"])
            prefix, _ = project_prefix(inputs, None, layout, end=end, removed_turns=0)
            actual, _ = half_window_inputs(prefix, Tokenizer(), max_image_turn=12)
            features = actual["input_ids"].double().cumsum(1)[0, -len(call["ids"]) - 1 : -1] / 10000
            reference = reference - torch.exp(theta * features).sum() / total
        self.assertTrue(torch.allclose(grouped, reference, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(grad, torch.autograd.grad(reference, theta)[0], atol=1e-12, rtol=1e-12))

    def test_eviction_replaces_whole_cache(self):
        """首次淘汰必须创建新原生会话，不能沿用含旧图的递归/卷积状态。"""
        inputs, _, _, _ = fixture()
        engine = SimpleNamespace(
            ids=inputs["input_ids"][:, :100],
            types=inputs["attention_mask"][:, :100],
            image_keys=list(inputs["pixel_values"])[:10],
            model=object(),
        )
        created = []

        def factory(model):
            fresh = SimpleNamespace(model=model)
            created.append(fresh)
            return fresh

        actual, new, audit = SamplingWindow(12, Tokenizer()).project(inputs, engine, factory)
        self.assertIs(new, created[0])
        self.assertTrue(audit["cache_rebuilt"])
        self.assertEqual(audit["omitted_turns"], 12)
        self.assertEqual(len(actual["pixel_values"]), 16)


if __name__ == "__main__":
    unittest.main()
