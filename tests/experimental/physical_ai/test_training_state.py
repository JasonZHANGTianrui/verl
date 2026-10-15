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

"""组边界恢复的核心回归：优化器连续性、数据契约与损坏文件拒绝。"""

import tempfile
import unittest
from pathlib import Path

import torch

from verl.experimental.physical_ai.training_state import load_boundary, save_boundary


class TrainingStateTests(unittest.TestCase):
    """用有动量的小模型核对真实优化器恢复，不用无状态SGD掩盖漏存状态。"""

    def test_resume_matches_uninterrupted_optimizer(self):
        """连续两步与保存/重载后第二步应逐参数一致，任务/数据变化必须拒绝。"""
        torch.manual_seed(3)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def step(m, o):
            """使用固定输入和真实反向更新，检查优化器动量是否接续。"""
            o.zero_grad()
            m(torch.ones(1, 3)).square().sum().backward()
            o.step()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            contract = dict(data_sha256="fixed-source", seed=42)
            save_boundary(
                model,
                optimizer,
                Path(directory) / "saved",
                completed_groups=1,
                optimizer_steps=1,
                contract=contract,
                rng_states=[dict(torch=torch.get_rng_state())],
            )
            manifest = Path(directory) / "saved/checkpoint.json"
            saved, receipt = load_boundary(manifest, contract)
            restored = torch.nn.Linear(3, 1)
            restored.load_state_dict(saved["model_state_dict"])
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01, momentum=0.9)
            restored_optimizer.load_state_dict(saved["optimizer_state_dict"])
            step(model, optimizer)
            step(restored, restored_optimizer)
            for a, b in zip(model.parameters(), restored.parameters(), strict=False):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(saved["completed_groups"], 1)
            with self.assertRaisesRegex(ValueError, "配置/数据"):
                load_boundary(manifest, dict(data_sha256="changed", seed=42))
            with Path(receipt["path"]).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_boundary(manifest, contract)
