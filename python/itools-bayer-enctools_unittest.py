#!/usr/bin/env python3

"""itools-bayer-enctool_unittest.py: itools bayer enctools unittest.

# runme
# $ ./itools-bayer-enctool_unittest.py
"""

import importlib
import math
import os
import shlex
import string
import tempfile
import numpy as np
import unittest

itools_bayer = importlib.import_module("itools-bayer")
itools_bayer_enctools = importlib.import_module("itools-bayer-enctools")


class MainTest(unittest.TestCase):
    def testNothing(self):
        pass


if __name__ == "__main__":
    unittest.main()
