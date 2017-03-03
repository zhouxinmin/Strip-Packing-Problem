# coding:utf-8

import unittest
from homeworkclub import *


class PrimesTestCase(unittest.TestCase):
    """Tests for `homeworkclub.py`."""

    def test_is_summary(self):
        """Is five successfully determined to be prime?"""
        self.assertTrue(summary("2016-06-02 20:00~22:00 7\n2016-06-03 09:00~12:00 14\n2016-06-04 14:00~17:00 22\n"))
        self.assertTrue(disassembly("2016-06-02 20:00~22:00 7\n2016-06-03 09:00~12:00 14\n2016-06-04 14:00~17:00 22\n"))
        self.assertTrue(pro("2016-06-02 20:00~22:00 7"))
        self.assertTrue(counting(7))

if __name__ == '__main__':
    unittest.main()