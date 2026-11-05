"""节点本地RL监督：共享盘写满时仍保留训练输出、退出码与真实存活状态。

本程序不改变训练算法或自动重跑。源码置于冻结训练模块之外，恢复契约不因新增
控制器而改变。local_dir必须在节点本地盘，mirror与storage_root可以位于共享AFS。
"""

import argparse
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path


def write_json(path, value):
    """原子写状态并fsync；本地写失败直接抛错，不伪称已记录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def publish(local, mirror, state):
    """先写本地真值，再尽力同步共享盘；同步错误回写本地，不吞掉训练失败。"""
    write_json(local, state)
    try:
        write_json(mirror, state)
    except OSError as exc:
        state["mirror_error"] = dict(errno=exc.errno, message=str(exc), at=time.time())
        write_json(local, state)


def check_storage(root, minimum_bytes):
    """检查空间并真实写入fsync，发现ENOSPC/配额失败时不启动子进程。

    字节阈值仅是启动门槛，不承诺statvfs能覆盖共享盘未来消费或所有用户配额。
    测试文件在finally清理，不修改既有数据；free=0时不会绕过空间保护。
    """
    root = Path(root)
    stat = os.statvfs(root)
    free = stat.f_bavail * stat.f_frsize
    if free < minimum_bytes:
        raise OSError(28, f"available bytes {free} below launch reserve {minimum_bytes}")
    probe = root / (".rl-write-probe-" + uuid.uuid4().hex)
    try:
        with probe.open("xb") as stream:
            stream.write(b"rl-write-probe\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        probe.unlink(missing_ok=True)
    return dict(available_bytes=free, minimum_bytes=minimum_bytes, write_fsync_passed=True)


def supervise(command, local_dir, mirror, storage_root, *, minimum_bytes=128 * 2**30, interval=10, storage_interval=60):
    """监督一份固定命令；真实returncode决定完成/失败，不用进度文件推断存活。

    子进程stdout/stderr只写本地，不经AFS管道。SIGTERM/INT只终止本次子进程组，
    不扫描杀其他实验。没有自动重启；非零退出保留日志尾部供恢复前定位。
    """
    local_dir = Path(local_dir).resolve()
    if str(local_dir).startswith("/mnt/afs/"):
        raise ValueError("local_dir不能位于共享AFS")
    local_dir.mkdir(parents=True, exist_ok=False)
    state_path = local_dir / "status.json"
    state = dict(
        schema="physical-ai.local-supervisor.v1",
        phase="preparing",
        supervisor_pid=os.getpid(),
        command=command,
        local_dir=str(local_dir),
        storage_root=str(storage_root),
        started_at=time.time(),
    )
    process = None
    requested_signal = None
    previous = {}

    def stop(number, _frame):
        """记录外部停止请求，并只向本控制器创建的进程组传递信号。"""
        nonlocal requested_signal
        requested_signal = number
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, stop)
    try:
        state["storage_check"] = check_storage(storage_root, minimum_bytes)
        publish(state_path, mirror, state)
        with (local_dir / "train.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            state.update(phase="running", child_pid=process.pid)
            next_storage_check = time.monotonic() + storage_interval
            while process.poll() is None:
                if time.monotonic() >= next_storage_check:
                    try:
                        state["storage_check"] = check_storage(storage_root, minimum_bytes)
                    except OSError as exc:
                        state["storage_error"] = dict(errno=exc.errno, message=str(exc), at=time.time())
                        # 不尝试在资源不足时强存半批；停止本次进程组，保留原完整checkpoint。
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
                    next_storage_check = time.monotonic() + storage_interval
                state.update(observed_at=time.time(), child_alive=True)
                publish(state_path, mirror, state)
                try:
                    process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
            code = process.returncode
        state.update(
            phase="storage_blocked" if "storage_error" in state else "complete" if code == 0 else "failed",
            returncode=code,
            child_alive=False,
            finished_at=time.time(),
            requested_signal=requested_signal,
        )
        with (local_dir / "train.log").open("rb") as stream:
            stream.seek(max(0, (local_dir / "train.log").stat().st_size - 4000))
            state["log_tail"] = stream.read().decode(errors="replace")
        publish(state_path, mirror, state)
        return code
    except BaseException as exc:
        state.update(phase="supervisor_failed", error=repr(exc), finished_at=time.time())
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state["child_alive"] = False
        publish(state_path, mirror, state)
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    """显式指定本地/共享控制位置和原训练命令；不加载模型或注入算法参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=128)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.min_free_gib < 0:
        parser.error("须给定命令和非负空间阈值")
    code = supervise(
        command, args.local_dir, args.mirror, args.storage_root, minimum_bytes=int(args.min_free_gib * 2**30)
    )
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
