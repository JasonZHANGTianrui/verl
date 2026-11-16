"""动作专用组必须使用原执行器，屏蔽followup，不混入QA奖励或改变调用者配置。"""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from verl.experimental.physical_ai import action_grpo
from verl.experimental.physical_ai.algorithms import resolve_algorithm
from verl.experimental.physical_ai.seggrpo import Segment


def config():
    """最小合法动作专用配置，故意携带原题，验证采样前会禁用followup。"""
    return dict(advantage_estimator='action_grpo', group_size=2, qa_steps=0,
                question={'question':'Where is the bowl?'})


class ActionGRPOTests(TestCase):
    def test_same_executor_receives_no_qa_and_config_is_unchanged(self):
        cfg=config();original=deepcopy(cfg)
        records=[Segment(f't{i}','trajectory','g','c','case','policy',2,0,2,float(i),'scored') for i in range(2)]
        artifacts=[dict(qa_reward=None,qa_status='no_question',task_reward=i) for i in range(2)]
        with patch('verl.experimental.physical_ai.grpo_rollout.collect_group',return_value=(records,artifacts)) as collect:
            got,_=action_grpo.collect_group(None,None,cfg,'output',policy_version='policy',task={},resources={})
            self.assertEqual(got,records)
            self.assertIsNone(collect.call_args.args[2]['question'])
            self.assertEqual(collect.call_args.args[2]['qa_steps'],0)
        self.assertEqual(cfg,original)

    def test_qa_reward_is_rejected(self):
        r=Segment('t','trajectory','g','c','case','policy',2,0,2,1.,'scored')
        with patch('verl.experimental.physical_ai.grpo_rollout.collect_group',return_value=([r],[dict(qa_reward=1,qa_status='scored',task_reward=1)])):
            with self.assertRaises(ValueError):action_grpo.collect_group(None,None,config(),'out',policy_version='policy')

    def test_configuration_rejects_replay_and_qa_groups(self):
        for extra in [dict(qa_steps=32),dict(qa_history_mode='fixed_sft'),dict(qa_group_size=8),dict(qa_replay_release='pool')]:
            with self.assertRaises(ValueError):action_grpo.validate_config({**config(),**extra})

    def test_dispatch_keeps_other_algorithms_separate(self):
        selected=resolve_algorithm(config())
        self.assertEqual((selected.name,selected.width),('action_grpo',2))
        self.assertIs(selected.collect_group,action_grpo.collect_group)
        from verl.experimental.physical_ai.grpo_rollout import collect_group
        self.assertIs(resolve_algorithm(dict(advantage_estimator='grpo',group_size=2)).collect_group,collect_group)
