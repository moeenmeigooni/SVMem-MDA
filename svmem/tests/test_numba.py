"""
Unit and regression test for the svmem package.
"""

# Import package, test suite, and other packages as needed
import svmem
import pytest

def default_test_settings():
    structure = ''
    trajectory = ''

    return (structure, trajectory)


def test_numpy():
    structure, trajectory = default_test_settings()
    svm = svmem()
    svm.run()
    assert svm.return_status


def test_numba():
    structure, trajectory = default_test_settings()
    svm = svmem()
    svm.run()
    assert svm.return_status


def test_jax():
    structure, trajectory = default_test_settings()
    svm = svmem()
    svm.run()
    assert svm.return_status


def test_trajectory():
    structure, trajectory = default_test_settings()
    svm = svmem()
    svm.run()
    assert svm.return_status
