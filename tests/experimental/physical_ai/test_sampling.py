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

from verl.experimental.physical_ai.sampling import fork_chat, sample_response
from verl.experimental.physical_ai.workspace import configure_workspace


class SamplingTests(unittest.TestCase):
    """保留原测试条件，只变更模块导入与工作区定位。"""

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
