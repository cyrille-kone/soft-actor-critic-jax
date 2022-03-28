# coding=utf-8
r"""
PyCharm Editor
Author @git Team
"""
import utils
import unittest
import yaml

class TestUtils(unittest.TestCase):
    def test_load_config(self):
        self.assertEqual(utils.load_config(("TEST", "KEY")), ["VALUE"], "Load config value checking failed")

    def test_load_yaml(self):
        with open('../configs/example.yaml', 'r') as f:
            config_args = yaml.safe_load(f.read())
        eval_args = {
            'env': 'pendulum',
            'agent': 'random',
            'max_thread': -1,
            'actor_lr': '3e-4',
            'critic_lr': '3e-4',
            'temp_lr': '3e-4',
            'hidden_dims': [256, 256],
            'discount': 0.99,
            'tau': 0.005,
            'target_period_update': 1,
            'init_temperature': 1.0,
            'target_entropy': None,
            'backup_entropy': True,
            'replay_buffer_size': None}
        self.assertDictEqual(eval_args, config_args)

# TODO

if __name__ == "__main__":
    unittest.main()
