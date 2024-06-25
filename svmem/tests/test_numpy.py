"""
Unit and regression test for the svmem package.
"""

# Import package, test suite, and other packages as needed
import MDAnalysis as mda
import numpy as np
import svmem
from svmem.backends.SVMem_numpy import *
import pytest

@pytest.fixture(scope='module')
def default_test_settings():
    periodic = [True, True, False]
    gamma = 0.1
    learning_rate = .01
    max_iter = 500
    tolerance = 0.001
    train_labels = None

    return periodic, gamma, learning_rate, max_iter, tolerance, train_labels

@pytest.fixture(scope='module')
def default_vectors():
    v1 = np.array([1, 1, 0])
    v2 = np.array([0, 0, -1])
    
    return v1, v2

@pytest.mark.parametrize(
    'structure,trajectory,forcefield', [
        ('',
         None,
         'martini'),
        ('',
         '',
         'martini'),
        ('',
         '',
         'charmm')
    ],
)
def test_svmem(structure, trajectory, forcefield, default_test_settings):
    periodic, gamma, learning_rate, max_iter, tolerance, train_labels = default_test_settings
    
    if trajectory is not None:
        u = mda.Universe(structure, trajectory)
    else:
        u = mda.Universe(structure)
        
    svm = svmem.SVMem(
        u, 
        backend='numpy', 
        forcefield=forcefield,
        periodic=periodic, 
        gamma=gamma,
        learning_rate=learning_rate,
        max_iter=max_iter, 
        tolerance=tolerance, 
        train_labels=train_labels
    )
    
    svm.run()
    assert svm.return_status
    
def test_ndot(default_vectors):
    a, b = default_vectors
    assert ndot(a, b) == 0.
    
def test_nsign():
    int1 = 1
    int2 = -1
    float1 = 1.
    float2 = -1.
    
    assert nsign(int1) == 1.
    assert nsign(int2) == -1.
    assert nsign_int(float1) == 1
    assert nsign_int(float2) == -1
    
def test_vector_operations(default_vectors):
    v1, v2 = default_vectors
    
    def magnitude(vec):
        n = len(vec)
        mag = 0.
        for i in range(n):
            mag += vec[i] ** 2.
        return np.sqrt(mag)
    
    assert vec_mag(v1) == magnitude(v1)
    assert vec_mag(v2) == magnitude(v2)
    
    assert vec_mags(np.vstack(v1, v2)) == [magnitude(v1), magnitude(v2)]
    
    assert vec_norm(v1) == v1 / magnitude(v1)
    assert vec_norms(np.vstack(v1, v2)) == np.vstack(v1 / magnitude(v1), 
                                                     v2 / magnitude(v2))
    
    
    
    
    