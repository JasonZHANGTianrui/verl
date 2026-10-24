"""进程启动时固定NumPy/SciPy各自的BLAS内核；不改环境、资产、容差或全局安装。

必须按文件路径执行，早于verl/torch导入。两个库可绑定不同内核以匹配数据制作机；
设置只在该库首次加载期间有效，随后恢复原环境变量，避免影响其他库/外部进程。
"""

import argparse
import ctypes
import json
import os
import runpy
import sys
from pathlib import Path


def configure(numpy_core, scipy_core):
    """加载并核对实际DSO内核，返回数值运行身份；晚于库导入或不支持的内核直接拒绝。"""
    if "numpy" in sys.modules or "scipy" in sys.modules:
        raise RuntimeError("数值内核须在numpy/scipy首次导入之前配置")
    previous = os.environ.get("OPENBLAS_CORETYPE")
    try:
        os.environ["OPENBLAS_CORETYPE"] = numpy_core
        import numpy as np

        np.linalg.svd(np.eye(3))
        os.environ["OPENBLAS_CORETYPE"] = scipy_core
        import scipy
        import scipy.linalg

        scipy.linalg.svd(np.eye(3))
    finally:
        if previous is None:
            os.environ.pop("OPENBLAS_CORETYPE", None)
        else:
            os.environ["OPENBLAS_CORETYPE"] = previous
    actual = {}
    for label, module, symbols, expected in [
        ("numpy", np, ["openblas_get_corename64_", "openblas_get_corename"], numpy_core),
        ("scipy", scipy, ["scipy_openblas_get_corename", "openblas_get_corename"], scipy_core),
    ]:
        libs = list((Path(module.__file__).parent.parent / (label + ".libs")).glob("*openblas*.so"))
        if len(libs) != 1:
            raise RuntimeError("无法唯一识别" + label + "的OpenBLAS库")
        lib = ctypes.CDLL(str(libs[0]))
        name = None
        for symbol in symbols:
            if hasattr(lib, symbol):
                function = getattr(lib, symbol)
                function.restype = ctypes.c_char_p
                name = function().decode()
                break
        if name != expected:
            raise RuntimeError(f"{label}内核不匹配：{name} != {expected}")
        actual[label + "_core"] = name
    actual.update(numpy_version=np.__version__, scipy_version=scipy.__version__)
    os.environ["PHYSICAL_AI_NUMERICS_IDENTITY"] = json.dumps(actual, sort_keys=True)
    return actual


def main():
    """以原argv启动目标模块或脚本；训练worker/机械验收/策略小批共用此入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy-core", required=True)
    parser.add_argument("--scipy-core", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module")
    target.add_argument("--script")
    args, remaining = parser.parse_known_args()
    identity = configure(args.numpy_core, args.scipy_core)
    print("NUMERICS_IDENTITY " + json.dumps(identity, sort_keys=True), flush=True)
    sys.argv = [args.module or args.script, *remaining]
    if args.module:
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
