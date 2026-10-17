# Copyright 2026 Physical AI contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""原生G0.5采样、前缀及分支隔离；需Physical AI集成工作区。"""

import unittest
from types import SimpleNamespace

import torch

from verl.experimental.physical_ai.batched_sampling import pack_caches, run_phases, sample_responses, unpack_cache
from verl.experimental.physical_ai.sampling import fork_chat, sample_response
from verl.experimental.physical_ai.workspace import configure_workspace


class SamplingTests(unittest.TestCase):
    """核对原生采样、批量缓存隔离、停止机制和环境渲染精度。"""

    def test_egl_mapping_is_explicit_and_complete(self):
        """显式非同序映射按local_rank读取，旧默认保留，缺设备不能自动回退软件渲染。"""
        from verl.experimental.physical_ai.libero_session import egl_device_for_rank

        self.assertEqual(egl_device_for_rank({}, 1, 2), 0)
        self.assertEqual(egl_device_for_rank(dict(egl_device_id=3), 0, 2), 3)
        self.assertEqual(egl_device_for_rank(dict(egl_device_ids=[3, 1]), 0, 2), 3)
        self.assertEqual(egl_device_for_rank(dict(egl_device_ids=[3, 1]), 1, 2), 1)
        for bad in ([0], [0, True], [0, -1], "0,1"):
            with self.assertRaises(ValueError):
                egl_device_for_rank(dict(egl_device_ids=bad), 0, 2)

    def test_fork_copies_registered_visual_state_without_hiding_model_changes(self):
        """设备alpha/标志独立复制；质量、RGB、未登记site或身份变化仍拒绝。"""
        import json

        import numpy as np

        from verl.experimental.physical_ai.libero_session import copy_visual_runtime

        def environment(alpha, visible):
            """构造含一个设备site和一个普通site的模型，不依赖真实模拟器。"""
            model = SimpleNamespace(
                site_rgba=np.array([[0.9, 0.05, 0.05, alpha], [1., 1., 1., 0.]]), mass=1., geom_size=0.1
            )
            model.site_name2id = lambda name: {"stove_burner": 0, "ordinary": 1}[name]
            model.get_xml = lambda: json.dumps(
                dict(rgba=model.site_rgba.tolist(), mass=model.mass, geom_size=model.geom_size)
            )
            obj = SimpleNamespace(object_properties={"vis_site_names": {"burner": ("stove_burner", visible)}})
            return SimpleNamespace(
                sim=SimpleNamespace(model=model), visualization_sites_list=["stove"], get_object=lambda name: obj
            )

        parent, child = environment(2., True), environment(-1., False)
        before = parent.sim.model.get_xml()
        receipt = copy_visual_runtime(parent, child)
        self.assertEqual(parent.sim.model.get_xml(), before)
        self.assertEqual(child.sim.model.get_xml(), before)
        self.assertEqual(receipt["sites"][0]["alpha"], 2.)
        child.get_object("stove").object_properties["vis_site_names"]["burner"] = ("stove_burner", False)
        self.assertTrue(parent.get_object("stove").object_properties["vis_site_names"]["burner"][1])
        for change in ("mass", "geometry", "rgb", "unregistered_alpha", "site_identity", "registration"):
            with self.subTest(change=change):
                child = environment(-1., False)
                if change == "mass":
                    child.sim.model.mass = 2.
                elif change == "geometry":
                    child.sim.model.geom_size = 0.2
                elif change == "rgb":
                    child.sim.model.site_rgba[0, 0] = 0.1
                elif change == "unregistered_alpha":
                    child.sim.model.site_rgba[1, 3] = 1.
                elif change == "site_identity":
                    child.get_object("stove").object_properties["vis_site_names"]["burner"] = ("ordinary", False)
                else:
                    child.visualization_sites_list = []
                with self.assertRaises(ValueError):
                    copy_visual_runtime(parent, child)
        parent.sim.model.site_rgba[0, 3] = np.nan
        with self.assertRaises(ValueError):
            copy_visual_runtime(parent, environment(-1., False))

    def test_interleaved_environments_activate_own_render_context(self):
        """交错读图必须先切本实例上下文；替换offscreen context后不继续使用旧对象。"""
        from verl.experimental.physical_ai.libero_session import bind_render_context

        active = []

        def context(name):
            """模拟EGL当前上下文，渲染结果取决于最近一次激活。"""
            return SimpleNamespace(gl_ctx=SimpleNamespace(make_current=lambda: active.append(name)))

        first = SimpleNamespace(_render_context_offscreen=context("a"), render=lambda **kw: (active[-1], kw))
        second = SimpleNamespace(_render_context_offscreen=context("b"), render=lambda **kw: (active[-1], kw))
        bind_render_context(first)
        bind_render_context(second)
        self.assertEqual(first.render(camera="wrist"), ("a", {"camera": "wrist"}))
        self.assertEqual(second.render(), ("b", {}))
        self.assertEqual(first.render(), ("a", {}))
        first._render_context_offscreen = context("a-new")
        bind_render_context(first)
        self.assertEqual(first.render(), ("a-new", {}))
        self.assertEqual(active, ["a", "b", "a", "a-new"])

    def test_fork_image_quantization_is_bounded_and_does_not_edit_input(self):
        """允许一个uint8量化级；两级、尺寸变化与错误dtype必须拒绝，原图不修改。"""
        import numpy as np

        from verl.experimental.physical_ai.libero_session import compare_fork_image

        parent = np.full((4, 4, 3), 128, dtype=np.uint8)
        child = parent.copy()
        child[0, 0, 0] += 1
        before = parent.copy()
        self.assertEqual(compare_fork_image(parent, child)["changed_pixels"], 1)
        np.testing.assert_array_equal(parent, before)
        child[0, 0, 0] += 1
        with self.assertRaises(ValueError):
            compare_fork_image(parent, child)
        with self.assertRaises(ValueError):
            compare_fork_image(parent, parent.astype(float))
        with self.assertRaises(ValueError):
            compare_fork_image(parent, parent[:2])

    def test_packed_cache_keeps_real_history_and_branch_isolation(self):
        """不同历史长度仅补KV，不向递推状态加虚构步骤；拆出的分支拥有独立存储。"""
        configure_workspace()
        from g05.models.kv_cache import SparseKVCache

        caches = []
        for length in (2, 5):
            c = SparseKVCache(last_linear_layer=1)
            c.key_cache[0] = torch.randn(1, 2, length, 3)
            c.value_cache[0] = torch.randn(1, 2, length, 3)
            c.conv_states[1] = torch.randn(1, 2, 4)
            c.recurrent_states[1] = torch.randn(1, 2, 3, 3)
            c.split_recurrent_states[1] = torch.randn(1, 2, 3, 3)
            caches.append(c)
        packed, pads = pack_caches(caches)
        self.assertEqual(pads, [3, 0])
        self.assertFalse(bool(packed.key_cache[0][0, :, :3].any()))
        for i, original in enumerate(caches):
            restored = unpack_cache(packed, i, pads[i])
            for field in ("key_cache", "value_cache", "conv_states", "recurrent_states", "split_recurrent_states"):
                for key, expected in getattr(original, field).items():
                    torch.testing.assert_close(getattr(restored, field)[key], expected)
            restored.conv_states[1].zero_()
            self.assertTrue(bool(original.conv_states[1].abs().sum()))
            self.assertTrue(bool(packed.conv_states[1][i].abs().sum()))

    def test_batch_early_eos_does_not_advance_finished_branch(self):
        """短历史分支先EOS退出，另一条继续；KV padding不泄漏且EOS本身被缓存。"""
        configure_workspace()
        from g05.models.kv_cache import SparseKVCache

        owner = self

        class VLM:
            def decode(self, hidden):
                logits = torch.full((hidden.shape[0], 1, 4), -100.0)
                for i in range(hidden.shape[0]):
                    logits[i, 0, 1 if hidden[i, 0, 0] == 0 else 2] = 100
                return logits

            def embed(self, ids):
                return ids.float().unsqueeze(-1)

            def __call__(self, **kw):
                cache = kw["kv_cache"]
                size = kw["inputs_embeds"].shape[0]
                if size == 2:
                    owner.assertTrue(bool((kw["attention_mask"][0, :, :, :3] < 0).all()))
                    owner.assertTrue(bool((kw["attention_mask"][1] == 0).all()))
                new = kw["inputs_embeds"].unsqueeze(1)
                cache.update(new, new, 0)
                cache.conv_states[1].add_(1)
                return torch.ones(size, 1, 1), cache

        model = SimpleNamespace(
            training=False,
            vlm=VLM(),
            attn_implementation="test",
            ar_helper=SimpleNamespace(action_token_range=(2, 4)),
            cfg=SimpleNamespace(vlm=SimpleNamespace(max_position_embeddings=16)),
            mask_helper=SimpleNamespace(_sample_step=lambda shape, device: torch.ones(shape, device=device)),
        )

        class Engine:
            def __init__(self, length, marker):
                self.model = model
                self.ids = torch.zeros(1, length, dtype=torch.long)
                self.types = torch.full_like(self.ids, 4)
                self.hidden = torch.tensor([[[float(marker)]]])
                self.last_position = torch.tensor([[length - 1.0]])
                self.cache = SparseKVCache(last_linear_layer=1)
                self.cache.key_cache[0] = torch.zeros(1, 1, length, 1)
                self.cache.value_cache[0] = torch.zeros(1, 1, length, 1)
                self.cache.conv_states[1] = torch.zeros(1, 1, 1)

            def append(self, inputs, **kwargs):
                owner.assertTrue(torch.equal(inputs["input_ids"], self.ids))

        engines = [Engine(2, 0), Engine(5, 1)]
        requests = [
            dict(
                engine=e,
                inputs={"input_ids": e.ids},
                max_new_tokens=3,
                stop_ids=[1],
                temperature=1.0,
                generator=torch.Generator().manual_seed(i),
                mode="cached",
                compute_dtype="float32",
            )
            for i, e in enumerate(engines)
        ]
        results = sample_responses(requests)
        self.assertEqual([r["ids"] for r in results], [[1], [2, 2, 2]])
        self.assertEqual([e.cache.num_items() for e in engines], [3, 8])
        self.assertEqual([float(e.cache.conv_states[1].item()) for e in engines], [1.0, 3.0])
        self.assertEqual([r["termination"] for r in results], ["message_end", "token_budget"])

    def test_phase_scheduler_batches_and_closes_all_on_error(self):
        """调度保持返回顺序；不同结束长度能汇成batch，失败时所有视频/环境生成器都收尾。"""
        closed, batches = [], []

        def phase(identity, count):
            """可变长度的回合，用finally模拟视频资源释放。"""
            try:
                for step in range(count):
                    response = yield dict(identity=identity, step=step)
                    self.assertEqual(response, identity)
                return identity
            finally:
                closed.append(identity)

        def sample(requests):
            """记录一次真正收到的batch宽度，避免串行循环伪装batch。"""
            batches.append(len(requests))
            return [r["identity"] for r in requests]

        self.assertEqual(run_phases([phase("a", 1), phase("b", 3)], 8, sample_batch=sample), ["a", "b"])
        self.assertEqual(batches, [2, 1, 1])
        self.assertCountEqual(closed, ["a", "b"])
        closed.clear()

        def broken(_):
            """批次生成失败不得留下其他回合的未关闭资源。"""
            raise RuntimeError("injected")

        with self.assertRaisesRegex(RuntimeError, "injected"):
            run_phases([phase("a", 1), phase("b", 3)], 8, sample_batch=broken)
        self.assertCountEqual(closed, ["a", "b"])

    def test_native_observation_import_with_verl_scripts(self):
        """verl普通scripts包存在时仍加载准确G0.5函数，不覆盖框架自己的包。"""
        import inspect
        import sys
        from pathlib import Path

        root = configure_workspace()
        sys.path[:0] = [
            str(root / "training-pipeline/ours/g05"),
            str(root / "training-pipeline/g05/src"),
            str(root / "training-pipeline/g05"),
        ]
        import scripts

        scripts_source = scripts.__file__
        from g05_native_api import observation_builder

        self.assertEqual(
            Path(inspect.getfile(observation_builder())).resolve(),
            (root / "training-pipeline/g05/scripts/serve_policy.py").resolve(),
        )
        self.assertEqual(scripts.__file__, scripts_source)

    def test_sampler_uses_fp32_head_and_consumes_final_token(self):
        """输出头必须在禁用autocast下算概率，不能先BF16舍入再转float。"""
        import sys

        root = configure_workspace()
        sys.path.insert(0, str(root / "training-pipeline/ours/g05"))
        owner = self

        class Head:
            def decode(self, hidden):
                owner.assertFalse(torch.is_autocast_enabled())
                return torch.tensor([[[0.0, 100.0, 0.0, 0.0]]])

        class Engine:
            def __init__(self):
                self.model = SimpleNamespace(
                    vlm=Head(),
                    cfg=SimpleNamespace(vlm=SimpleNamespace(max_position_embeddings=16)),
                    ar_helper=SimpleNamespace(action_token_range=(2, 4)),
                )
                self.calls = []
                self.hidden = torch.zeros(1, 1, 2)

            def append(self, inputs, *, return_logits=True):
                self.calls.append((inputs["input_ids"].clone(), return_logits))
                return None

        engine = Engine()
        inputs = dict(input_ids=torch.tensor([[0, 0]]), attention_mask=torch.tensor([[4, 4]]))
        result = sample_response(
            engine, inputs, max_new_tokens=4, stop_ids=[1], temperature=1.0, generator=torch.Generator().manual_seed(1)
        )
        self.assertEqual(result["ids"], [1])
        self.assertEqual(result["termination"], "message_end")
        self.assertEqual(result["logprobs"], [0.0])
        self.assertEqual(engine.calls[-1][0].tolist(), [[0, 0, 1]])
        self.assertTrue(all(not flag for _, flag in engine.calls))
        self.assertEqual(inputs["input_ids"].tolist(), [[0, 0]])

    def test_recompute_sampling_keeps_full_prefix(self):
        """参考采样每步完整前向，静态观察embedding仅计算一次，不进入KV递推。"""
        import sys

        root = configure_workspace()
        sys.path.insert(0, str(root / "training-pipeline/ours/g05"))
        lengths = []
        embeddings = []
        owner = self

        class VLM:
            def __call__(self, **kwargs):
                owner.assertIsNone(kwargs["kv_cache"])
                owner.assertFalse(kwargs["return_kv_cache"])
                lengths.append(kwargs["inputs_embeds"].shape[1])
                return kwargs["inputs_embeds"]

            def embed(self, ids):
                return torch.nn.functional.one_hot(ids, 4).float()

            def decode(self, hidden):
                owner.assertFalse(torch.is_autocast_enabled())
                return torch.tensor([[[0.0, 100.0, 0.0, 0.0]]])

        vlm = VLM()

        def make_embeddings(ids, types, pixels, proprio):
            embeddings.append(ids.clone())
            return vlm.embed(ids)

        model = SimpleNamespace(
            vlm=vlm,
            _forward_embed=make_embeddings,
            build_causal_mask_and_position_ids=lambda *a, **k: (None, None),
            attn_implementation="test",
            cfg=SimpleNamespace(vlm=SimpleNamespace(max_position_embeddings=16)),
            ar_helper=SimpleNamespace(action_token_range=(2, 4)),
        )
        inputs = dict(
            input_ids=torch.tensor([[0, 0]]),
            attention_mask=torch.tensor([[4, 4]]),
            pixel_values={"image": torch.ones(1)},
            proprio=torch.ones(1, 1, 27),
        )
        result = sample_response(
            SimpleNamespace(model=model),
            inputs,
            max_new_tokens=2,
            stop_ids=[],
            temperature=1.0,
            generator=torch.Generator().manual_seed(1),
            mode="recompute",
        )
        self.assertEqual(result["ids"], [1, 1])
        self.assertEqual(lengths, [2, 3])
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(inputs["input_ids"].tolist(), [[0, 0]])

    def test_fork_keeps_model_shared_and_context_independent(self):
        """分支独立修改token/图像/state；模型权重不因复制会话翻倍。"""
        policy = object()
        parent = SimpleNamespace(
            policy=policy,
            ids=[1, 2],
            types=[4, 6],
            pixels={"image": torch.ones(1, 3, 2, 2)},
            states=[torch.ones(1, 27)],
        )
        child = fork_chat(parent)
        child.ids.append(3)
        child.types[0] = 6
        child.pixels["image"].zero_()
        child.states[0].zero_()
        self.assertIs(child.policy, policy)
        self.assertEqual(parent.ids, [1, 2])
        self.assertEqual(parent.types, [4, 6])
        self.assertTrue(bool(parent.pixels["image"].all()))
        self.assertTrue(bool(parent.states[0].all()))
