"""新旧官方观察导入名必须是同一模块/函数，不用另一实现替代训练交接。"""

import sys, types, unittest
from unittest.mock import patch
from verl.experimental.physical_ai.native_runtime import bind_native_api_alias


class NativeRuntimeTest(unittest.TestCase):
    def test_alias_preserves_function_and_rejects_collision(self):
        """别名幂等、函数对象完全一致；既有异源模块不会被覆盖。"""
        module = types.ModuleType("official")
        module.build_obs_dict = lambda value: value
        with patch.dict(sys.modules):
            sys.modules.pop("g05_native_api", None)
            bind_native_api_alias(module)
            bind_native_api_alias(module)
            from g05_native_api import build_obs_dict

            self.assertIs(build_obs_dict, module.build_obs_dict)
            other = types.ModuleType("foreign")
            with self.assertRaises(ValueError):
                bind_native_api_alias(other)
            self.assertIs(sys.modules["g05_native_api"], module)


if __name__ == "__main__":
    unittest.main()
