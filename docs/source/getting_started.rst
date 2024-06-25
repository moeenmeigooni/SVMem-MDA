Getting Started
===============

Python environment requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* MDAnalysis >= version
* numba >= 0.45.1
* scikit-learn >= 0.21.2
* jax >= version

Basic usage
~~~~~~~~~~~
.. code-block:: python
  :linenos:

  import MDAnalysis as mda
  from svmem import svmem

  # load structure into MDAnalysis Universe object
  u = mda.Universe('membrane.pdb') 

  # select membrane only
  lipid = u.select_atoms('segid MEMB')

  # define various hyperparameters
  periodic = np.array([True, True, False]) 
  gamma = 0.01 
  learning_rate = 0.1
  max_iter = 500
  tolerance = 0.0001

  # instance svmem class
  svmem = svmem(lipid,
                periodic=periodic, 
                gamma=gamma,
                learning_rate=learning_rate,
                max_iter=max_iter,
                tolerance=tolerance) 

  # run analysis
  svmem.run()

  # curvature and normal vectors are stored in the svmem object
  svmem.results.mean_curvature
  svmem.results.gaussian_curvature
  svmem.results.normal_vectors

Troubleshooting
~~~~~~~~~~~~~~~
Having problems running SVMem? Documentation will be limited until the release of version 0.1.
Until then, email me at meigoon2@illinois.edu with questions/comments/concerns, and I'll get back to you as soon as I can.
