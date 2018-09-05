"""
Classes for manipulating pencils.

"""

from functools import partial
from collections import defaultdict
import numpy as np
from scipy import sparse
from mpi4py import MPI
import uuid

from ..tools.array import zeros_with_pattern
from ..tools.array import expand_pattern
from ..tools.progress import log_progress

import logging
logger = logging.getLogger(__name__.split('.')[-1])


def build_pencils(domain):
    """
    Create the set of pencils over a domain.

    Parameters
    ----------
    domain : domain object
        Problem domain

    Returns
    -------
    pencils : list
        Pencil objects

    """

    # Get transverse indeces in fastest sequence
    trans_shape = domain.local_coeff_shape[:-1]
    indices = np.ndindex(*trans_shape)

    # Construct corresponding trans diff consts and build pencils
    pencils = []
    scales = domain.remedy_scales(1)
    start = domain.distributor.coeff_layout.start(scales)[:-1]
    for index in indices:
        pencils.append(Pencil(domain, index, start+index))

    return pencils


def build_matrices(pencils, problem, matrices):
    """Build pencil matrices."""
    # Build new cachid for NCC expansions
    cacheid = uuid.uuid4()
    # Build test operator dicts to synchronously expand all NCCs
    for eq in problem.eqs+problem.bcs:
        for matrix in matrices:
            expr, vars = eq[matrix]
            if expr != 0:
                test_index = [0] * problem.domain.dim
                expr.operator_dict(test_index, vars, cacheid=cacheid, **problem.ncc_kw)
    # Build matrices
    for pencil in log_progress(pencils, logger, 'info', desc='Building pencil matrix', iter=np.inf, frac=0.1, dt=10):
        pencil.build_matrices(problem, matrices, cacheid=cacheid)


class Pencil:
    """
    Object holding problem matrices for a given transverse wavevector.

    Parameters
    ----------
    index : tuple of ints
        Transverse indeces for retrieving pencil from system data

    """

    def __init__(self, domain, local_index, global_index):
        self.domain = domain
        self.local_index = tuple(local_index)
        self.global_index = tuple(global_index)
        if domain.bases[-1].coupled:
            self.build_matrices = self._build_coupled_matrices
        else:
            self.build_matrices = self._build_uncoupled_matrices

    def _build_uncoupled_matrices(self, problem, names, cacheid=None):

        matrices = {name: [] for name in (names+['select'])}

        zbasis = self.domain.bases[-1]
        dtype = zbasis.coeff_dtype
        for last_index in range(zbasis.coeff_size):
            submatrices = self._build_uncoupled_submatrices(problem, names, last_index, cacheid=cacheid)
            for name in matrices:
                matrices[name].append(submatrices[name])

        for name in matrices:
            blocks = matrices[name]
            matrix = sparse.block_diag(blocks, format='csr', dtype=dtype)
            matrix.eliminate_zeros()
            matrices[name] = matrix

        # Store minimal CSR matrices for fast dot products
        for name in names:
            matrix = matrices[name]
            # Store full matrix
            matrix.eliminate_zeros()
            setattr(self, name+'_full', matrix.tocsr().copy())
            # Store truncated matrix
            matrix.data[np.abs(matrix.data) < problem.entry_cutoff] = 0
            matrix.eliminate_zeros()
            setattr(self, name, matrix.tocsr().copy())


        # Store expanded CSR matrices for fast combination
        self.LHS = zeros_with_pattern(*[matrices[name] for name in names]).tocsr()
        for name in names:
            matrix = matrices[name]
            matrix = expand_pattern(matrix, self.LHS)
            setattr(self, name+'_exp', matrix.tocsr().copy())

        # Store operators for RHS
        self.G_eq = matrices['select']
        self.G_bc = None

        # no Dirichlet
        self.dirichlet = None

    def _build_uncoupled_submatrices(self, problem, names, last_index, cacheid=None):

        index = list(self.global_index) + [last_index]
        index_dict = {}
        for axis, basis in enumerate(self.domain.bases):
            index_dict['n'+basis.name] = index[axis]

        # Find applicable equations
        selected_eqs = [eq for eq in problem.eqs if eval(eq['raw_condition'], index_dict)]
        # Check selections
        nvars = problem.nvars
        neqs = len(selected_eqs)
        Neqs = len(problem.eqs)
        if neqs != nvars:
            raise ValueError("Pencil {} has {} equations for {} variables.".format(index, neqs, nvars))

        # Build matrices
        dtype = self.domain.bases[-1].coeff_dtype
        matrices = {name: np.zeros((nvars, nvars), dtype=dtype) for name in names}
        matrices['select'] = np.zeros((nvars, Neqs))
        for i, eq in enumerate(selected_eqs):
            j = problem.eqs.index(eq)
            matrices['select'][i,j] = 1
            for name in names:
                expr, vars = eq[name]
                if expr != 0:
                    op_dict = expr.operator_dict(index, vars, cacheid=cacheid, **problem.ncc_kw)
                    matrix = matrices[name]
                    for j in range(nvars):
                        matrix[i,j] = op_dict[vars[j]]

        return matrices

    def _build_coupled_matrices(self, problem, names, cacheid=None):

        global_index = self.global_index
        # Check basic solvability conditions
        # Find applicable equations
        index_dict = {}
        for axis, basis in enumerate(self.domain.bases):
            if basis.separable:
                index_dict['n'+basis.name] = global_index[axis]
        selected_eqs = [eq for eq in problem.eqs if eval(eq['raw_condition'], index_dict)]
        selected_bcs = [bc for bc in problem.bcs if eval(bc['raw_condition'], index_dict)]
        # Check selections
        nvars = problem.nvars
        neqs = len(selected_eqs)
        nbcs = len(selected_bcs)
        ndiff = sum(eq['differential'] for eq in selected_eqs)
        if neqs != nvars:
            raise ValueError("Pencil {} has {} equations for {} variables.".format(global_index, neqs, nvars))
        if nbcs != ndiff:
            raise ValueError("Pencil {} has {} boundary conditions for {} differential equations.".format(global_index, nbcs, ndiff))

        zbasis = self.domain.bases[-1]
        zsize = zbasis.coeff_size
        zdtype = zbasis.coeff_dtype
        compound = hasattr(zbasis, 'subbases')
        Zero_Nz = sparse.csr_matrix((zsize, zsize), dtype=zdtype)
        Identity_1 = sparse.identity(1, dtype=zdtype, format='csr')
        Identity_Nz = sparse.identity(zsize, dtype=zdtype, format='csr')
        Drop_Nz = sparse.eye(0, zsize, dtype=zdtype, format='csr')

        # Build right preconditioner (block diagonal)
        pre_right_diags = []
        for var in problem.variables:
            if problem.meta[var][zbasis.name]['constant']:
                #PR = Identity_1
                PR = Identity_Nz
            elif problem.meta[var][zbasis.name].get('dirichlet'):
                PR = zbasis.Dirichlet
            else:
                PR = Identity_Nz
            pre_right_diags.append(PR)

        # Build matrices
        LHS_blocks = {name: [] for name in names}
        pre_left_diags = []
        for eq in (problem.bcs + problem.eqs):

            # Drop non-selected equations
            if eq not in (selected_eqs + selected_bcs):
                pre_left_diags.append(Drop_Nz)
                continue
            # Build left preconditioner block
            if eq in problem.bcs:
                PL = zbasis.DropNonconstantRows
            elif eq['differential'] and compound:
                PL = zbasis.DropLastRow @ zbasis.FilterMatchRows @ zbasis.Precondition
            elif eq['differential']:
                PL = zbasis.DropLastRow @ zbasis.Precondition
            elif compound:
                PL = zbasis.FilterMatchRows
            else:
                PL = Identity_Nz
            pre_left_diags.append(PL)

            # Build left-preconditioned LHS matrix blocks
            PL_Zero = sparse.csr_matrix(PL.shape, dtype=zdtype)
            PL_Zero_coo = sparse.coo_matrix(PL.shape, dtype=zdtype)
            PL_coo = PL.tocoo()
            for name in names:
                eq_expr, eq_vars = eq[name]
                if eq_expr != 0:
                    Ei = eq_expr.operator_dict(global_index, eq_vars, cacheid=cacheid, **problem.ncc_kw)
                else:
                    Ei = defaultdict(int)
                eq_blocks = []
                for j in range(nvars):
                    # Build equation terms
                    Eij = Ei[eq_vars[j]]
                    if np.isscalar(Eij):
                        if Eij == 0:
                            Eij = PL_Zero_coo
                        elif Eij == 1:
                            Eij = PL_coo
                        else:
                            Eij = PL_coo * Eij
                    elif PL is Identity_Nz:
                        Eij = Eij.tocoo()
                    else:
                        Eij = (PL @ Eij).tocoo()
                    eq_blocks.append(Eij)
                LHS_blocks[name].append(eq_blocks)

        # Add match terms
        if compound and ('L' in names):
            for i, eq in enumerate(selected_eqs):
                if eq['differential']:
                    Match = zbasis.DropLastRow @ zbasis.Match
                else:
                    Match = zbasis.Match
                LHS_blocks['L'][nbcs+i][i] = (LHS_blocks['L'][nbcs+i][i] + Match).tocoo()

        # Combine blocks
        self.pre_left = sparse.block_diag(pre_left_diags, format='csr', dtype=zdtype)
        self.pre_right = sparse.block_diag(pre_right_diags, format='csr', dtype=zdtype)
        LHS_matrices = {name: fastblock(LHS_blocks[name]).tocsr() for name in names}

        # Store minimal-entry matrices for fast dot products
        for name, matrix in LHS_matrices.items():
            # Store full matrix
            matrix.eliminate_zeros()
            setattr(self, name+'_full', matrix.tocsr().copy())
            # Truncate entries
            matrix.data[np.abs(matrix.data) < problem.entry_cutoff] = 0
            matrix.eliminate_zeros()
            # Store truncated matrix
            setattr(self, name, matrix.tocsr().copy())

        # Store expanded right-preconditioned matrices
        # Apply right preconditioning
        if self.pre_right is not None:
            for name in names:
                LHS_matrices[name] = LHS_matrices[name] * self.pre_right
        # Build expanded LHS matrix to store matrix combinations
        self.LHS = zeros_with_pattern(*LHS_matrices.values()).tocsr()
        # Store expanded matrices for fast combination
        for name, matrix in LHS_matrices.items():
            matrix = expand_pattern(matrix, self.LHS)
            setattr(self, name+'_exp', matrix.tocsr().copy())


def fastblock(blocks):
    M = len(blocks)
    N = len(blocks[0])
    data, row, col = [], [], []
    i0 = 0
    for m in range(M):
        j0 = 0
        bm = blocks[m]
        for n in range(N):
            bmn = bm[n]
            data.extend(bmn.data)
            row.extend(bmn.row + i0)
            col.extend(bmn.col + j0)
            j0 += bmn.shape[1]
        i0 += bmn.shape[0]
    return sparse.coo_matrix((data, (row, col)), shape=(i0, j0))