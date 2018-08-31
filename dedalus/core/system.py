"""
Classes for systems of coefficients/fields.

"""

import numpy as np

from ..tools.general import unify


class CoeffSystem:
    """
    Representation of a collection of fields that don't need to be transformed,
    and are therefore stored as a contigous set of coefficient data, joined
    along the last axis, for efficient pencil and group manipulation.

    Parameters
    ----------
    pencil_length : int
        Number of coefficients in a single pencil
    domain : domain object
        Problem domain

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field coefficients

    """

    def __init__(self, pencil_length, domain):
        # Allocate data for joined coefficients
        shape = domain.local_coeff_shape.copy()
        shape[-1] = pencil_length
        dtype = domain.dist.coeff_layout.dtype
        self.data = np.zeros(shape, dtype=dtype)

    def get_pencil(self, pencil):
        """Return pencil view from system buffer."""
        return self.data[pencil.local_index]

    def set_pencil(self, pencil, data):
        """Set pencil data in system buffer."""
        np.copyto(self.data[pencil.local_index], data)


class FieldSystem(CoeffSystem):
    """
    Collection of fields alongside a CoeffSystem buffer for efficient pencil
    and group manipulation.

    Parameters
    ----------
    fields : list of field objets
        Fields to join into system

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field coefficients
    fields : list
        Field objects
    nfields : int
        Number of fields in system
    field_dict : dict
        Dictionary of fields
    slices : dict
        Dictionary of last-axis slice objects connecting field and system data

    """

    def __init__(self, fields):
        domain = unify(field.domain for field in fields)
        # Reorder to put constant fields first
        zbasis = domain.bases[-1]
        const_fields = [f for f in fields if f.meta[zbasis.name]['constant']]
        nonconst_fields = [f for f in fields if not f.meta[zbasis.name]['constant']]
        # Allocate data for joined coefficients
        pencil_length = len(const_fields) + len(nonconst_fields) * zbasis.coeff_size
        super().__init__(pencil_length, domain)
        # Create slices for each field's data
        # Group modes, with constant fields first so nonconst fields have fixed stride
        self.slices = {}
        for i, f in enumerate(const_fields):
            self.slices[f] = slice(i, i+1)
        offset = len(const_fields)
        stride = len(nonconst_fields)
        for i, f in enumerate(nonconst_fields):
            self.slices[f] = slice(offset+i, None, stride)
        # Attributes
        self.domain = domain
        self.fields = const_fields + nonconst_fields
        self.field_names = [f.name for f in self.fields]
        self.nfields = len(self.fields)
        self.field_dict = dict(zip(self.field_names, self.fields))

    def __getitem__(self, name):
        """Return field corresponding to specified name."""
        return self.field_dict[name]

    def gather(self):
        """Copy fields into system buffer."""
        data = self.data
        slices = self.slices
        for field in self.fields:
            field.require_coeff_space()
            np.copyto(data[..., slices[field]], field.data)

    def scatter(self):
        """Extract fields from system buffer."""
        data = self.data
        slices = self.slices
        coeff_layout = self.domain.dist.coeff_layout
        for field in self.fields:
            field.layout = coeff_layout
            np.copyto(field.data, data[..., slices[field]])

