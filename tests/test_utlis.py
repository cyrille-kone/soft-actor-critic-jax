# coding=utf-8
r"""
PyCharm Editor
Author @git Team
"""
import utils
import unittest

class TestUtils(unittest.TestCase):
    def test_load_config(self):
        self.assertEqual(utils.load_config(("TEST", "KEY")), ["VALUE"], "Load config value checking failed")


# TODO

if __name__ == "__main__":
    unittest.main()
