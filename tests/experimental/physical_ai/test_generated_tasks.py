"""新来源接入契约：明确类型/预算、消费者验收绑定；不把未知成功当作假标签。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import verl.experimental.physical_ai.generated_tasks as module
from verl.experimental.physical_ai.generated_tasks import validate_task, verify_acceptance
from verl.experimental.physical_ai.training_state import content_hash


class GeneratedTasksTest(unittest.TestCase):
    def test_source_and_budget_dispatch(self):
        """生成任务不可借旧HDF5类型进入；单目标按300而非suite90的520。"""
        task = dict(
            schema=module.SCHEMA,
            split="train",
            environment={"initial_condition": {"kind": "libero-bddl-seeded-reset-preview-v1"}},
            objective={"goals": [["On", "x", "y"]]},
            execution_budget_steps=300,
            followup={"question_spec": {}},
        )
        with patch("benchmark.libero.questions.registry.default_registry"):
            self.assertEqual(validate_task(task), 1)
            with self.assertRaises(ValueError):
                validate_task(dict(task, execution_budget_steps=520))
            with self.assertRaises(ValueError):
                validate_task(dict(task, schema="sft"))

    def test_acceptance_binds_real_files_and_complete_scopes(self):
        """仅标签passed不够：发布内容、三类样例和真实消费者文件必须全对应。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            folder = Path(module.__file__).parent
            r = dict(
                status="passed",
                manifest_sha256=content_hash(root / "manifest.json"),
                scopes=["single", "combination", "replacement"],
                records=160,
                consumer_files={
                    n: content_hash(folder / n)
                    for n in ["generated_tasks.py", "task_data.py", "libero_session.py", "numerical_entry.py"]
                },
            )
            p = root / "consumer-acceptance.json"
            p.write_text(json.dumps(r))
            verify_acceptance(root, {"records": 160})
            for key, value in [
                ("scopes", ["single"]),
                ("records", 159),
                ("manifest_sha256", "bad"),
                ("consumer_files", {}),
            ]:
                p.write_text(json.dumps(dict(r, **{key: value})))
                with self.assertRaises(ValueError):
                    verify_acceptance(root, {"records": 160})


if __name__ == "__main__":
    unittest.main()
