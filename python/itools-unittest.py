#!/usr/bin/env python3

"""itools-unittest.py: itools bayer unittest.

Operation:
$ ./*.unittest.py  # run all the tests
$ ./*.unittest.py --list_tests  # list all available tests
$ ./*.unittest.py --filter <filter>
where:
  <filter> := <test_filter_item> [":" <test_filter_item>]*
  <test_filter_item> := <test_function>.<test_name>
  <test_function> := <string> | "*"
  <test_name> := <string> | "*"

Examples:
```
$ ./*.unittest.py --filter TestFunction:TestFunction2.t*
$ ./*.unittest.py --filter *.TestName:*.TestName2*
```
"""

import argparse
import fnmatch
import numpy as np
import sys
import unittest


class TestCase(unittest.TestCase):
    def getTestCases(self, function_name, test_case_list):
        global LIST_TESTS
        global FILTER

        list_tests = LIST_TESTS
        filter_string = FILTER

        if list_tests:
            print(f" {function_name}.")
            for test_case in test_case_list:
                print(f"  {function_name}.{test_case['name']}")
            self.skipTest("list test")

        return self.filterTestCases(function_name, test_case_list, filter_string)

    def filterTestCases(self, function_name, test_case_list, filter_string):
        if not filter_string:
            return test_case_list

        # each filter item is of the form TestFunction.TestName, supports '*'
        filters = filter_string.split(":")
        matched_test_case_name_list = set()
        for filt in filters:
            try:
                func_pat, case_pat = filt.split(".", 1)
            except ValueError:
                continue  # skip invalid filters
            if fnmatch.fnmatch(function_name, func_pat):
                for test_case in test_case_list:
                    if fnmatch.fnmatch(test_case["name"], case_pat):
                        matched_test_case_name_list.add(test_case["name"])
        matched_test_case_list = []
        for test_case in test_case_list:
            if test_case["name"] in matched_test_case_name_list:
                matched_test_case_list.append(test_case)
        return list(matched_test_case_list)

    # function to compare 2 planar representations
    @classmethod
    def comparePlanar(cls, planar, expected_planar, absolute_tolerance, label):
        assert set(expected_planar.keys()) == set(planar.keys()), "Broken planar output"
        for key in expected_planar:
            np.testing.assert_allclose(
                planar[key],
                expected_planar[key],
                atol=absolute_tolerance,
                err_msg=f"error on {label} case {key=}",
            )

    # function to compare 2 buffer representations
    @classmethod
    def compareBuffer(cls, buffer, expected_buffer, dtype, absolute_tolerance, label):
        assert len(buffer) == len(expected_buffer), f"error on {label} case: wrong size"
        plane = np.frombuffer(buffer, dtype=dtype)
        expected_plane = np.frombuffer(expected_buffer, dtype=dtype)
        np.testing.assert_allclose(
            plane,
            expected_plane,
            atol=absolute_tolerance,
            err_msg=f"error on {label} case",
        )


def main(argv):
    global FILTER
    global LIST_TESTS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_tests",
        action="store_true",
        dest="list_tests",
        default=False,
        help="List Tests",
    )
    parser.add_argument(
        "--filter",
        dest="filter",
        default=None,
        metavar="filter",
        help="Filter String",
    )

    options, unknown_options = parser.parse_known_args()
    FILTER = options.filter
    LIST_TESTS = options.list_tests
    # clean sys.argv before passing to unittest
    sys.argv = [sys.argv[0]] + unknown_options
    unittest.main()
