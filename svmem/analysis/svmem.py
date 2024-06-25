"""
SVMem --- :mod:`svmem.analysis.SVMem`
===========================================================

This module contains the :class:`SVMem` class.

"""
from typing import Union, TYPE_CHECKING

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
import numpy as np
from typing import List

if TYPE_CHECKING:
    from MDAnalysis.core.universe import Universe, AtomGroup


class SVMem(AnalysisBase):
    """SVMem class.

    This class is used to perform analysis on a trajectory.

    Parameters
    ----------
    universe_or_atomgroup: :class:`~MDAnalysis.core.universe.Universe` or :class:`~MDAnalysis.core.groups.AtomGroup`
        Universe or group of atoms to apply this analysis to.
        If a trajectory is associated with the atoms,
        then the computation iterates over the trajectory.
    select: str
        Selection string for atoms to extract from the input Universe or
        AtomGroup

    Attributes
    ----------
    universe: :class:`~MDAnalysis.core.universe.Universe`
        The universe to which this analysis is applied
    atomgroup: :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to which this analysis is applied
    results: :class:`~MDAnalysis.analysis.base.Results`
        results of calculation are stored here, after calling
        :meth:`SVMem.run`
    start: Optional[int]
        The first frame of the trajectory used to compute the analysis
    stop: Optional[int]
        The frame to stop at for the analysis
    step: Optional[int]
        Number of frames to skip between each analyzed frame
    n_frames: int
        Number of frames analysed in the trajectory
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`SVMem.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`SVMem.run`
    """

    def __init__(
        self,
        membrane: mda.AtomGroup,
        method: str='numba', forcefield: str='martini',
        periodic: List[bool]=[True, True, False], gamma: float=1.,
        learning_rate: float=0.01, max_iter: int=500, 
        tolerance: float=0.0001, train_labels: None=None,
        **kwargs
    ):
        super().__init__(membrane.universe.trajectory, **kwargs)
        self.u = membrane.universe
        self.memb = membrane
        resids = ' '.join([str(x) for x in np.unique(self.memb.residues.resids)])

        print(self.memb.resids)
        # Defined headgroup atom selections by forcefield
        match forcefield.lower():
            case 'martini':
                head_sel = f'name GL0 PO4 and resid {resids}'
            case 'charmm':
                head_sel = f'name OG12 P and resid {resids}'
            case _:
                raise ValueError(f'{forcefield=} is not a valid forcefield! \
                        Please specify either martini or charmm!')

        self.train_points = self.u.select_atoms(head_sel)
        self.n_train_points = self.train_points.n_atoms
        self.n_frames = self.u.trajectory.n_frames

        # Backend switch
        match method.lower():
            case 'jax':
                raise NotImplementedError('Not yet implemented, coming soon(TM).')
                from backends import SVMem_jax as backend
            case 'numba':
                from backends import SVMem_numba as backend
            case 'numpy':
                from backends import SVMem_numpy as backend
            case _:
                raise ValueError(f'{method=} is not a valid backend choice! \
                        Please specify from jax, numba or python!')

        atom_ids_per_lipid = [residue.atoms.indices for residue in self.memb.residues]
        self.backend = backend.Backend(periodic, train_labels, gamma, learning_rate,
                                       max_iter, tolerance, atom_ids_per_lipid)

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        self.center_shift = np.mean(self.memb.positions, axis=0)
        self.memb.positions -= self.center_shift
        self.results.weights_list = []
        self.results.intercept_list = []
        self.results.support_indices_list = []
        self.results.mean_curvature = np.empty((self.n_frames, self.n_train_points))
        self.results.gaussian_curvature = np.empty_like(self.results.mean_curvature)
        self.results.normal_vectors = np.empty((self.n_frames, self.n_train_points, 3))

    def _single_frame(self):
        """Calculate data from a single frame of trajectory"""
        fr = self._frame_index

        mean, gaussian, normals, weights, intercept, support_indices = self.backend.calculate_curvature(
                self.train_points.positions, 
                self.u.dimensions[:3], 
                self.memb
                )

        self.results.mean_curvature[fr] = mean
        self.results.gaussian_curvature[fr] = gaussian
        self.results.normal_vectors[fr] = normals
        self.results.weights_list.append(weights)
        self.results.intercept_list.append(intercept)
        self.results.support_indices_list.append(support_indices)

    def _conclude(self):
        """Calculate the final results of the analysis"""
        self.memb.positions += self.center_shift
