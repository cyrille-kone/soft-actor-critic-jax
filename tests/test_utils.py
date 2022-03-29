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

# TODO

if __name__ == "__main__":
    unittest.main()
