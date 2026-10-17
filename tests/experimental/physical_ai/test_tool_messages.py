"""回复中的合法/非法调用独立处理；动作ID与公开原文不能被修复或重分词替换。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from verl.experimental.physical_ai.tool_messages import message_requests


class Tokenizer:
    """只为动作ID字符定位提供可逆测试词表，不模拟模型生成。"""

    def decode(self, ids, **kwargs):
        return "".join({1000: "[A]", 1001: "[B]"}.get(i, chr(i)) for i in ids)


def ids(text):
    """按测试词表保留特殊动作ID，其他文本逐字符编码。"""
    out = []
    while text:
        special = next((s for s in ("[A]", "[B]") if text.startswith(s)), None)
        if special:
            out.append(1000 if special == "[A]" else 1001)
            text = text[3:]
        else:
            out.append(ord(text[0]))
            text = text[1:]
    return out


def call(action, parameter="action"):
    """生成一个测试外壳；真实语法校验继续调用公共工具解析器。"""
    return f"<tool_call><function=robot.execute><parameter={parameter}>{action}</parameter></function></tool_call>"


class ToolMessageTests(unittest.TestCase):
    def parse(self, text):
        return message_requests(
            text, ids(text), Tokenizer(), SimpleNamespace(action_token_begin_idx=1000, action_token_end_idx=1002)
        )

    def test_mixed_calls_keep_valid_order_and_invalid_as_text(self):
        bad = call("[A]", "invalid")
        for text in (bad + " middle " + call("[B]"), call("[B]") + " middle " + bad):
            good, ordinary, rejected = self.parse(text)
            self.assertEqual(len(good), 1)
            self.assertEqual(good[0]["segments"][0]["native_ids"], [1001])
            self.assertIn(bad, ordinary)
            self.assertIn("middle", ordinary)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(good[0]["raw_text"], text)

    def test_two_valid_calls_and_no_valid_calls(self):
        good, ordinary, _ = self.parse("before " + call("[A]") + " between " + call("[B]") + " after")
        self.assertEqual([r["segments"][0]["native_ids"] for r in good], [[1000], [1001]])
        self.assertEqual(ordinary, "before  between  after")
        text = call("fake-code") + call("[A]", "invalid")
        self.assertEqual(self.parse(text)[:2], ([], text))
        self.assertEqual(self.parse("plain answer")[:2], ([], "plain answer"))

    def test_private_thought_is_not_a_tool_request(self):
        text = "<think>Example: " + call("[A]") + "</think> answer"
        self.assertEqual(self.parse(text)[:2], ([], text))

    def test_infrastructure_exception_is_not_model_text(self):
        with patch("verl.experimental.physical_ai.tool_messages.parse_request", side_effect=RuntimeError("internal")):
            with self.assertRaises(RuntimeError):
                self.parse(call("[A]"))


if __name__ == "__main__":
    unittest.main()
