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

"""概率与梯度契约：分块投影不能改变next-token对齐、分布或反向结果。"""

import unittest

import torch

from verl.experimental.physical_ai.g05_logprobs import next_token_logprobs
from verl.experimental.physical_ai.workspace import configure_workspace


class LogProbabilityTests(unittest.TestCase):
    """使用小型完整词表作独立参照，不依赖真实checkpoint或GPU。"""

    def test_values_and_gradients_match_full_projection(self):
        """两个batch、稀疏监督及非单位温度下，分块结果和梯度均须一致。"""
        torch.manual_seed(0)
        hidden = torch.randn(2, 6, 4, requires_grad=True)
        head = torch.nn.Linear(4, 9, bias=False)
        ids = torch.randint(0, 9, (2, 6))
        mask = torch.tensor([[0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1]], dtype=torch.bool)
        actual = next_token_logprobs(hidden, ids, mask, head, chunk_size=2, temperature=0.8, checkpoint_projection=True)
        reference = torch.log_softmax(head(hidden[:, :-1]) / 0.8, -1)
        reference = reference.gather(-1, ids[:, 1:, None]).squeeze(-1) * mask[:, 1:]
        torch.testing.assert_close(actual, reference)
        actual_grad = torch.autograd.grad(actual.sum(), (hidden, head.weight), retain_graph=True)
        expected_grad = torch.autograd.grad(reference.sum(), (hidden, head.weight))
        for actual_value, expected_value in zip(actual_grad, expected_grad, strict=False):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertTrue(torch.equal(actual[~mask[:, 1:]], torch.zeros_like(actual[~mask[:, 1:]])))

    def test_rejects_modality_mask_and_unpredictable_first_token(self):
        """G0.5模态类型mask不能混成RL生成mask；首token没有next-token预测来源。"""
        hidden = torch.zeros(1, 3, 4)
        ids = torch.zeros(1, 3, dtype=torch.long)
        head = torch.nn.Linear(4, 9)
        with self.assertRaises(ValueError):
            next_token_logprobs(hidden, ids, torch.tensor([[1, 3, 6]]), head)
        with self.assertRaises(ValueError):
            next_token_logprobs(hidden, ids, torch.ones(1, 3, dtype=torch.bool), head)

    def test_delta_fp32_wrapper_matches_recurrent_reference(self):
        """fallback归一化/递推使用FP32，返回dtype和梯度不丢失，重复安装不叠包。"""
        import sys
        from types import SimpleNamespace

        root = configure_workspace()
        sys.path.insert(0, str(root / "training-pipeline/g05/src"))
        from g05.models.g05.qwen35.gated_deltanet import torch_chunk_gated_delta_rule as chunk
        from g05.models.g05.qwen35.gated_deltanet import torch_recurrent_gated_delta_rule as recurrent

        from verl.experimental.physical_ai.precision import stable_delta_precision

        module = SimpleNamespace(layer_idx=0, chunk_gated_delta_rule=chunk, recurrent_gated_delta_rule=recurrent)
        model = SimpleNamespace(
            vlm=SimpleNamespace(layers=[SimpleNamespace(layer_type="linear_attention", linear_attn=module)])
        )
        receipt = stable_delta_precision(model)
        wrapper = module.chunk_gated_delta_rule
        stable_delta_precision(model)
        self.assertIs(module.chunk_gated_delta_rule, wrapper)
        self.assertTrue(all(x["protected"] for x in receipt["functions"]))
        torch.manual_seed(0)
        q = torch.randn(1, 17, 2, 8, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        g = -torch.rand(1, 17, 2)
        beta = torch.rand(1, 17, 2)
        kwargs = dict(initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
        result, state = wrapper(q, k, v, g, beta, chunk_size=16, **kwargs)
        reference, ref_state = recurrent(q.float(), k.float(), v.float(), g, beta, **kwargs)
        torch.testing.assert_close(result, reference.to(q.dtype))
        torch.testing.assert_close(state, ref_state, atol=1e-6, rtol=1e-5)
        result.float().square().sum().backward()
        self.assertTrue(bool(torch.isfinite(q.grad).all()))
        self.assertGreater(float(q.grad.float().norm()), 0)

    def test_empty_generated_set_has_zero_gradient(self):
        """纯观察片段没有策略目标，不能产生虚假概率或非零梯度。"""
        hidden = torch.randn(1, 3, 4, requires_grad=True)
        ids = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        result = next_token_logprobs(hidden, ids, mask, torch.nn.Linear(4, 9))
        result.sum().backward()
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertTrue(torch.equal(hidden.grad, torch.zeros_like(hidden)))


if __name__ == "__main__":
    unittest.main()
