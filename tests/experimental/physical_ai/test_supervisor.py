"""本地故障日志与共享盘失效的核心回归；不运行GPU或真实训练。"""

import errno
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "supervise", Path(__file__).resolve().parents[3] / "scripts/physical_ai/supervise.py"
)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class SupervisorTests(unittest.TestCase):
    def test_shared_disk_full_still_records_child_failure_locally(self):
        """共享盘ENOSPC不再吞掉子进程真正的错误与退出码，也不误标running。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "local"
            mirror = root / "shared/state.json"
            real = supervisor.write_json

            def write(path, value):
                if Path(path) == mirror:
                    raise OSError(errno.ENOSPC, "No space left on device")
                return real(path, value)

            with patch.object(supervisor, "write_json", side_effect=write):
                code = supervisor.supervise(
                    [sys.executable, "-c", "import sys; print('ORIGINAL_FAILURE', flush=True); sys.exit(7)"],
                    local,
                    mirror,
                    root,
                    minimum_bytes=0,
                    interval=0.02,
                )
            state = json.loads((local / "status.json").read_text())
            self.assertEqual(code, 7)
            self.assertEqual(state["phase"], "failed")
            self.assertEqual(state["mirror_error"]["errno"], errno.ENOSPC)
            self.assertIn("ORIGINAL_FAILURE", (local / "train.log").read_text())
            self.assertFalse(state["child_alive"])

    def test_startup_write_failure_never_launches_training(self):
        """真实写入失败必须拦截启动，不能只看statvfs声称空间正常。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(supervisor, "check_storage", side_effect=OSError(errno.EDQUOT, "quota exceeded")),
                patch.object(supervisor.subprocess, "Popen") as launch,
            ):
                with self.assertRaises(OSError):
                    supervisor.supervise(["never-run"], root / "local", root / "shared.json", root, minimum_bytes=0)
                launch.assert_not_called()
            self.assertEqual(json.loads((root / "local/status.json").read_text())["phase"], "supervisor_failed")

    def test_runtime_storage_failure_stops_only_own_child(self):
        """运行中磁盘门槛失败留本地原因并停止子进程，不能继续写到磁盘全满。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(
                supervisor,
                "check_storage",
                side_effect=[dict(available_bytes=100), OSError(errno.ENOSPC, "storage exhausted")],
            ):
                code = supervisor.supervise(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    root / "local",
                    root / "mirror.json",
                    root,
                    minimum_bytes=0,
                    interval=0.01,
                    storage_interval=0.01,
                )
            state = json.loads((root / "local/status.json").read_text())
            self.assertNotEqual(code, 0)
            self.assertEqual(state["phase"], "storage_blocked")
            self.assertEqual(state["storage_error"]["errno"], errno.ENOSPC)
            self.assertFalse(state["child_alive"])

    def test_normal_exit_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = supervisor.supervise(
                [sys.executable, "-c", 'print("ok")'],
                root / "local",
                root / "mirror.json",
                root,
                minimum_bytes=0,
                interval=0.02,
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((root / "mirror.json").read_text())["phase"], "complete")

    def test_reject_shared_local_log_directory(self):
        with self.assertRaises(ValueError):
            supervisor.supervise(["none"], "/mnt/afs/fake-log", "/tmp/mirror", "/tmp")


if __name__ == "__main__":
    unittest.main()
