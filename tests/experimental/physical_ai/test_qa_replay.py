"""离线QA独立算法回归：无动作采样、历史mask、unknown排除、完整组与旧算法隔离。"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from verl.experimental.physical_ai import qa_replay
from verl.experimental.physical_ai.seggrpo import Segment, UNSCORED_QA


def example():
    """历史含5个token，新回答含2个token；两个不同奖励来自同一冻结经历。"""
    history = dict(kind='fixed_sft', rollout_id='sft/a', case_id='case',
                   manifest_sha256='a'*64, prefix_sha256='b'*64, source_record_sha256='c'*64)
    rows = [Segment(f'qa-{i}', 'qa', 'q', 'condition', 'case', 'model/update-0', 2, 4, 6,
                    float(i), 'scored', parent_rollout_id='sft/a', history_parent=history) for i in range(2)]
    arts = [dict(prefix_length=5, generated=[False]*5+[True]*2, old_logprobs=[0.]*5+[-1.,-2.]) for _ in rows]
    return rows, arts


class QAReplayTests(unittest.TestCase):
    def test_only_new_answer_receives_advantage(self):
        rows, arts = example()
        result = qa_replay.prepare_batch(rows, arts)
        self.assertFalse(result['active'][:, :4].any())
        self.assertEqual(result['active'].sum().item(), 4)
        self.assertTrue((result['advantages'][0, 4:] < 0).all())
        self.assertTrue((result['advantages'][1, 4:] > 0).all())
        torch.testing.assert_close(result['old'][0, 4:], torch.tensor([-1., -2.]))

    def test_unknown_preserves_samples_without_loss(self):
        rows, arts = example()
        rows = [replace(r, reward=None, reward_status=UNSCORED_QA) for r in rows]
        result = qa_replay.prepare_batch(rows, arts)
        self.assertEqual(result['sampled'].sum().item(), 4)
        self.assertFalse(result['active'].any())
        self.assertFalse(result['advantages'].any())

    def test_history_leak_and_incomplete_group_rejected(self):
        rows, arts = example()
        with self.assertRaises(ValueError):qa_replay.task_groups(rows[:1])
        arts[0]['generated'][2] = True
        with self.assertRaises(ValueError):qa_replay.prepare_batch(rows, arts)

    def test_mixed_prefix_and_online_pair_rejected(self):
        rows, _ = example()
        with self.assertRaises(ValueError):qa_replay.task_groups([rows[0], replace(rows[1], condition_id='different')])
        with self.assertRaises(ValueError):qa_replay.task_groups([replace(r, paired_execution_group_id='execution') for r in rows])

    def test_namespace_keeps_parent_and_minibatches_keep_group(self):
        rows, _ = example()
        rows = qa_replay.namespace_group(rows, 'task0')+qa_replay.namespace_group(rows, 'task1')
        self.assertEqual(qa_replay.minibatch_indices(rows, 2), [[0, 1], [2, 3]])
        self.assertEqual({r.parent_rollout_id for r in rows}, {'sft/a'})
        with self.assertRaises(ValueError):qa_replay.minibatch_indices(rows, 3)

    def test_collect_dispatches_only_replay(self):
        rows, arts = example()
        old = [replace(r, paired_execution_group_id='execution-group') for r in rows]
        with TemporaryDirectory() as tmp, patch('verl.experimental.physical_ai.sft_replay.collect_qa', return_value=(old, arts)) as collect:
            result, _ = qa_replay.collect_group(None, None, {'sampling_mode':'cached'}, Path(tmp), policy_version='model/update-0', task={})
            collect.assert_called_once()
            self.assertTrue(all(r.phase=='qa' and r.paired_execution_group_id is None for r in result))

if __name__=='__main__':unittest.main()
