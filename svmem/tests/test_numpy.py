"""
Unit and regression test for the svmem package.
"""

# Import package, test suite, and other packages as needed
import MDAnalysis as mda
import numpy as np
from svmem.analysis import SVMem
from svmem.analysis.backends import SVMem_numpy as snp
import pytest

@pytest.fixture(scope='module')
def default_sim_settings():
    box_dims = np.array([12., 12., 12.])
    periodic = [True, True, False]
    return box_dims, periodic

@pytest.fixture(scope='module')
def default_ml_settings():
    gamma = 0.1
    learning_rate = .01
    max_iter = 500
    tolerance = 0.001
    train_labels = None

    return gamma, learning_rate, max_iter, tolerance, train_labels

@pytest.fixture(scope='module')
def default_vectors():
    v1 = np.array([1, 1, 0])
    v2 = np.array([0, 0, -1])
    
    return v1, v2

@pytest.mark.parametrize(
    'structure,trajectory,forcefield', [
        ('martini_membrane.pdb',
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
def test_svmem(structure, trajectory, forcefield, 
               default_sim_settings, default_ml_settings):
    _, periodic = default_sim_settings
    gamma, learning_rate, max_iter, tolerance, labels = default_ml_settings
    
    if trajectory is not None:
        u = mda.Universe(structure, trajectory)
    else:
        u = mda.Universe(structure)
        
    svm = SVMem(
        u, 
        backend='numpy', 
        forcefield=forcefield,
        periodic=periodic, 
        gamma=gamma,
        learning_rate=learning_rate,
        max_iter=max_iter, 
        tolerance=tolerance, 
        train_labels=labels
    )
    
    svm.run()
    assert svm.return_status
    
def test_ndot(default_vectors):
    a, b = default_vectors
    assert ndot(a, b) == 0.
    
@pytest.mark.parametrize(
    'func,value,result', [
        (nsign, 1, 1.),
        (nsign, -1, -1.),
        (nsign_int, 1., 1),
        (nsign_int, -1., -1)
    ]
)
def test_nsign(func, value, result):
    assert func(value) == result
    
def test_vector_operations(default_vectors):
    v1, v2 = default_vectors
    
    def magnitude(vec: np.ndarray):
        n = len(vec)
        mag = 0.
        for i in range(n):
            mag += vec[i] ** 2.
        return np.sqrt(mag)
    
    assert snp.vec_mag(v1) == magnitude(v1)
    assert snp.vec_mag(v2) == magnitude(v2)
    
    assert snp.vec_mags(np.vstack(v1, v2)) == [magnitude(v1), magnitude(v2)]
    
    assert snp.vec_norm(v1) == v1 / magnitude(v1)
    assert snp.vec_norms(np.vstack(v1, v2)) == np.vstack(v1 / magnitude(v1),
                                                         v2 / magnitude(v2))
    
def test_unravel_upper_triangle_index():
    n = 5
    a_ref = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    b_ref = np.array([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])
    a, b = snp.unravel_upper_triangle_index(n)
    assert ((a == a_ref).all() and (b == b_ref).all())
    
def test_sym_dist_mat():
    matrix = np.arange(9).reshape(3, 3)
    #correct_result = np.array([5.19615242, 10.39230485, 5.19615242])
    assert snp._sym_dist_mat(matrix) == np.triu_indices(matrix, k=1)
    
def test_dist_mat(default_sim_settings):
    box_dims, periodic = default_sim_settings
    xyz1 = np.arange(9).reshape(3, 3)
    xyz2 = np.arange(9, 18).reshape(3, 3)
    result = np.array([15.58845727, 16.4924225, 16.58312395, 10.39230485, 
                       15.58845727, 16.4924225,  5.19615242, 10.39230485, 
                       15.58845727])
    
    assert snp.dist_mat(xyz1, xyz2, box_dims, periodic) == result
    
def test_dist_vec(default_sim_settings):
    box_dims, periodic = default_sim_settings
    xyz = np.arange(3)
    xyzs = np.arange(3, 12).reshape(3, 3)
    result = np.array([5.19615242, 10.39230485, 15.58845727])
    
    assert snp.dist_vec(xyz, xyzs, box_dims, periodic) == result
    
def test_disp(default_sim_settings):
    box_dims, periodic = default_sim_settings
    xyz = np.arange(3)
    xyzs = np.arange(3, 6)
    result = np.array([-3., -3., -3.])
    
    assert snp.disp_vec(xyz, xyzs, box_dims, periodic) == result 
    
def test_disp_vec(default_sim_settings):
    box_dims, periodic = default_sim_settings
    xyz = np.arange(3)
    xyzs = np.arange(3, 12).reshape(3, 3)
    result = np.array([[-3., -3., -3.],
                       [-6., -6., -6.],
                       [-9., -9., -9.]])
    
    assert snp.disp_vec(xyz, xyzs, box_dims, periodic) == result
    
def test_gaussian_transform_vec(default_ml_settings):
    gamma, _ = default_ml_settings
    arr = np.arange(3, dtype=np.float32)
    result = np.array([1., 0.9048374, 0.67032003])
    
    assert snp.gaussian_transform_vec(arr, gamma) == result
    
def test_gaussian_transform_mat(default_sim_settings):
    gamma, _ = default_sim_settings
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = np.array([[1., 0.9048374, 0.67032003],
                       [0.40656966, 0.20189652, 0.082085],
                       [0.02732372, 0.00744658, 0.00166156]])
    
    assert snp.gaussian_transform_mat(matrix, gamma) == result

def test_predict():
    vector = np.array([])
    weights = np.array([])
    intercept = np.array([])
    result = 1
    
    #assert snp.predict(vector, weights, intercept) == result

def test_update_disps():
    pass

def test_gradient():
    pass

def test_gradient_descent():
    pass

def test_coordinate_descent():
    pass

def test_descend_to_boundary():
    pass

def test_analytical_derivative():
    pass

def test_gaussian_curvature():
    pass

def test_mean_curvature():
    pass

def test_curvatures():
    pass