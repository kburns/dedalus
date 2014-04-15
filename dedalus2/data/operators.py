"""
Abstract and built-in classes defining deferred operations on fields.

"""

from functools import reduce, partial
import numpy as np

from .future import Future
from .field import Field
from ..tools.general import OrderedSet
from ..tools.cache import CachedClass
from ..tools.dispatch import MultiClass, CachedMultiClass


class Operator(Future, metaclass=CachedClass):
    """
    Base class for deferred operations on fields.

    Parameters
    ----------
    *args : fields, operators, and numeric types
        Operands. Number must match class attribute `arity`, if present.
    out : field, optional
        Output field.  If not specified, a new field will be used.

    Notes
    -----
    Operators are stacked (i.e. provided as arguments to other operators) to
    construct trees that represent compound expressions.  Nodes are evaluated
    by first recursively evaluating their subtrees, and then calling the
    `operate` method.

    """

    name = 'Op'
    arity = None
    store_last = False

    def __init__(self, *args, out=None):

        # Check arity
        if self.arity is not None:
            if len(args) != self.arity:
                raise ValueError("Wrong number of arguments.")

        # Required attributes
        self.args = list(args)
        self.original_args = list(args)
        self.domain = unique_domain(out, *args)
        self.out = out
        self.last_id = None
        self.build_metadata()

    def __repr__(self):
        repr_args = map(repr, self.args)
        return '%s(%s)' %(self.name, ', '.join(repr_args))

    def __str__(self):
        str_args = map(str, self.args)
        return '%s(%s)' %(self.name, ', '.join(str_args))

    def __getattr__(self, attr):
        # Intercept numpy ufunc calls
        if attr in UfuncWrapper.supported:
            ufunc = UfuncWrapper.supported[attr]
            return partial(UfuncWrapper, ufunc, self)
        else:
            raise AttributeError("%r object has no attribute %r" %(self.__class__.__name__, attr))

    def reset(self):
        """Restore original arguments."""

        self.args = list(self.original_args)

    def field_set(self, include_out=False):
        """Get set of field leaves."""

        # Recursively collect field arguments
        fields = OrderedSet()
        for a in self.args:
            if isinstance(a, Field):
                fields.add(a)
            elif isinstance(a, Operator):
                fields.update(a.field_set(include_out=include_out))

        # Add output field as directed
        if include_out:
            if self.out:
                fields.add(self.out)

        return fields

    def evaluate(self, id=None, force=True):
        """Recursively evaluate operation."""

        # Check storage
        if self.store_last and (id is not None):
            if id == self.last_id:
                return self.last_out
            else:
                # Clear cache to free output field
                self.last_id = None
                self.last_out = None

        # Recursively attempt evaluation of all operator arguments
        # Track evaluation success with flag
        all_eval = True
        for i, a in enumerate(self.args):
            if isinstance(a, Operator):
                a_eval = a.evaluate(id=id, force=force)
                # If evaluation succeeds, substitute result
                if a_eval is not None:
                    self.args[i] = a_eval
                # Otherwise change flag
                else:
                    all_eval = False

        # Return None if any arguments are not evaluable
        if not all_eval:
            return None

        # Check conditions unless forcing evaluation
        if not force:
            # Return None if operator conditions are not satisfied
            if not self.check_conditions():
                return None

        # Allocate output field if necessary
        if self.out:
            out = self.out
        else:
            out = self.domain.new_field()
        out.set_scales(self.domain.dealias, keep_data=False)

        # Perform operation
        self.operate(out)

        # Update metadata
        np.copyto(out.constant, self.constant)

        # Reset to free temporary field arguments
        self.reset()

        # Update storage
        if self.store_last and (id is not None):
            self.last_id = id
            self.last_out = out

        return out

    def attempt(self, id=None):
        """Recursively attempt to evaluate operation."""

        return self.evaluate(id=id, force=False)

    def build_metadata(self):

        raise NotImplementedError()

    def check_conditions(self):
        """Check that all argument fields are in proper layouts."""

        # This method must be implemented in derived classes and should return
        # a boolean indicating whether the operation can be computed without
        # changing the layout of any of the field arguments.

        raise NotImplementedError()

    def operate(self, out):
        """Perform operation."""

        # This method must be implemented in derived classes, take an output
        # field as its only argument, and evaluate the operation into this
        # field without modifying the data of the arguments.

        raise NotImplementedError()

    @staticmethod
    def from_string(string, vars, domain):
        """Build operator tree from string expression."""

        expression = eval(string, vars)
        if isinstance(expression, Operator):
            return expression
        elif isinstance(expression, Field):
            return Cast(expression)
        else:
            return Cast(expression, domain)


class Cast(Operator, metaclass=CachedMultiClass):

    name = 'Cast'

    def __init__(self, arg0, domain, out=None):
        # Required attributes
        self.args = [arg0]
        self.original_args = [arg0]
        self.domain = domain
        self.out = out
        self.last_id = None
        self.build_metadata()


class CastField(Cast):

    @staticmethod
    def _check_args(*args, **kw):
        return is_field(args[0])

    def __init__(self, arg0, out=None):
        # Initialize using field domain
        Cast.__init__(self, arg0, arg0.domain, out=out)

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, = self.args
        # Copy in current layout
        layout = arg0.layout
        out[layout] = arg0[layout]


class CastNumeric(Cast):

    @staticmethod
    def _check_args(*args, **kw):
        return is_numeric(args[0])

    def __init__(self, *args, **kw):
        Cast.__init__(self, *args, **kw)

    def build_metadata(self):
        self.constant = numeric_constant(self.args[0], self.domain)

    def check_conditions(self):
        return True

    def operate(self, out):
        # Copy in grid layout
        out['g'] = self.args[0]


class Integrate(Operator):

    store_last = True

    def __init__(self, arg0, *bases, out=None):
        # No bases: integrate over whole domain
        if len(bases) == 0:
            bases = list(arg0.domain.bases)
        # Multiple bases: recursively integrate
        if len(bases) > 1:
            arg0 = Integrate(arg0, *bases[:-1])
        # Required attributes
        self.args = [arg0]
        self.original_args = [arg0]
        self.domain = arg0.domain
        self.out = out
        self.last_id = None
        # Additional attributes
        self.basis = self.domain.get_basis_object(bases[-1])
        self.axis = arg0.domain.bases.index(self.basis)
        self.build_metadata()

    def __repr__(self):
        return 'Integ(%r, %r)' %(self.args[0], self.basis)

    def __str__(self):
        return 'Integ(%s, %s)' %(self.args[0], self.basis)

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)
        self.constant[self.axis] = True

    def check_conditions(self):
        # References
        arg0, = self.args
        axis = self.axis
        # Must be in ceoff+local layout
        is_coeff = not arg0.layout.grid_space[axis]
        is_local = arg0.layout.local[axis]

        return (is_coeff and is_local)

    def operate(self, out):
        # References
        arg0, = self.args
        axis = self.axis
        # Integrate in coeff+local layout
        arg0.require_coeff_space(axis)
        arg0.require_local(axis)
        out.layout = arg0.layout
        # Use basis integration method
        self.basis.integrate(arg0.data, out.data, axis=axis)


class Interpolate(Operator):

    store_last = True

    def __init__(self, arg0, basis, position, out=None):

        # Required attributes
        self.args = [arg0]
        self.original_args = [arg0]
        self.domain = arg0.domain
        self.out = out
        self.last_id = None
        # Additional attributes
        self.basis = self.domain.get_basis_object(basis)
        self.axis = arg0.domain.bases.index(self.basis)
        self.position = position
        self.build_metadata()

    def __repr__(self):
        return 'Interp(%r, %r, %r)' %(self.args[0], self.basis, self.position)

    def __str__(self):
        return 'Interp(%s, %s, %s)' %(self.args[0], self.basis, self.position)

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)
        self.constant[self.axis] = True

    def check_conditions(self):
        # References
        arg0, = self.args
        axis = self.axis
        # Must be in ceoff+local layout
        is_coeff = not arg0.layout.grid_space[axis]
        is_local = arg0.layout.local[axis]

        return (is_coeff and is_local)

    def operate(self, out):
        # References
        arg0, = self.args
        axis = self.axis
        # Integrate in coeff+local layout
        arg0.require_coeff_space(axis)
        arg0.require_local(axis)
        out.layout = arg0.layout
        # Use basis integration method
        self.basis.interpolate(arg0.data, out.data, self.position, axis=axis)


class GeneralFunction(Operator):

    def __init__(self, domain, layout, func, args=[], kw={}, out=None,):

        # Required attributes
        self.args = list(args)
        self.original_args = list(args)
        self.domain = domain
        self.out = out
        self.last_id = None
        # Additional attributes
        self.layout = domain.distributor.get_layout_object(layout)
        self.func = func
        self.kw = kw
        self._field_arg_indices = [i for (i,arg) in enumerate(self.args) if is_fieldlike(arg)]
        try:
            self.name = func.__name__
        except AttributeError:
            self.name = str(func)
        self.build_metadata()

    def build_metadata(self):
        self.constant = np.array([False] * self.domain.dim)

    def check_conditions(self):
        # Fields must be in proper layout
        for i in self._field_arg_indices:
            if self.args[i].layout is not self.layout:
                return False
        return True

    def operate(self, out):
        # Apply func in proper layout
        for i in self._field_arg_indices:
            self.args[i].require_layout(self.layout)
        out.layout = self.layout
        np.copyto(out.data, self.func(*self.args, **self.kw))


class UfuncWrapper(Operator):

    supported = {ufunc.__name__: ufunc for ufunc in
        (np.sign, np.conj, np.exp, np.exp2, np.log, np.log2, np.log10, np.sqrt,
         np.square, np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan,
         np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh)}

    def __init__(self, ufunc, arg0, out=None):

        # Required Attributes
        self.args = [arg0]
        self.original_args = [arg0]
        self.domain = arg0.domain
        self.out = out
        self.last_id = None
        # Additional attributes
        self.ufunc = ufunc
        self.name = ufunc.__name__
        self._grid_layout = self.domain.distributor.grid_layout
        self.build_metadata()

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, = self.args
        # Apply ufunc in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        self.ufunc(arg0.data, out=out.data)


class Absolute(Operator):

    name = 'Abs'
    arity = 1

    def __init__(self, *args, **kw):
        Operator.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, = self.args
        # Rectify in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.abs(arg0.data, out.data)


class Negate(Operator):

    name = 'Neg'
    arity = 1

    def __str__(self):
        return '(-%s)' %self.args[0]

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, = self.args
        # Negate in current layout
        out.layout = arg0.layout
        np.negative(arg0.data, out.data)


class Arithmetic(Operator):

    arity = 2

    def __str__(self):
        str_args = map(str, self.args)
        return '(%s)' %self.str_op.join(str_args)


class Add(Arithmetic, metaclass=CachedMultiClass):

    name = 'Add'
    str_op = ' + '


class AddFieldField(Add):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def build_metadata(self):
        self.constant = self.args[0].constant & self.args[1].constant

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Add in arg0 layout (arbitrary choice)
        arg1.require_layout(arg0.layout)
        out.layout = arg0.layout
        np.add(arg0.data, arg1.data, out.data)


class AddFieldNumeric(Add):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_numeric(arg1))

    def __init__(self, *args, **kw):
        Add.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = self.args[0].constant & numeric_constant(self.args[1], self.domain)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1, out.data)


class AddNumericField(Add):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_numeric(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Add.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Add in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0, arg1.data, out.data)


class Subtract(Arithmetic, metaclass=CachedMultiClass):

    name = 'Sub'
    str_op = ' - '


class SubFieldField(Subtract):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def build_metadata(self):
        self.constant = self.args[0].constant & self.args[1].constant

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Subtract in arg0 layout (arbitrary choice)
        arg1.require_layout(arg0.layout)
        out.layout = arg0.layout
        np.subtract(arg0.data, arg1.data, out.data)


class SubFieldNumeric(Subtract):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_numeric(arg1))

    def __init__(self, *args, **kw):
        Subtract.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = self.args[0].constant & numeric_constant(self.args[1], self.domain)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Subtract in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.subtract(arg0.data, arg1, out.data)


class SubNumericField(Subtract):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_numeric(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Subtract.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Subtract in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.subtract(arg0, arg1.data, out.data)


class Multiply(Arithmetic, metaclass=CachedMultiClass):

    name = 'Mult'
    str_op = ' * '


class MultFieldField(Multiply):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Multiply.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = self.args[0].constant & self.args[1].constant

    def check_conditions(self):
        # Must be in grid layout
        return ((self.args[0].layout is self._grid_layout) and
                (self.args[1].layout is self._grid_layout))

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1.data, out.data)


class MultFieldScalar(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_scalar(arg1))

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg0.layout
        np.multiply(arg0.data, arg1, out.data)


class MultScalarField(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_scalar(arg0) and is_fieldlike(arg1))

    def build_metadata(self):
        self.constant = np.copy(self.args[1].constant)

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg1.layout
        np.multiply(arg0, arg1.data, out.data)


class MultFieldArray(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_array(arg1))

    def __init__(self, *args, **kw):
        Multiply.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = self.args[0].constant & numeric_constant(self.args[1], self.domain)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1, out.data)


class MultArrayField(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_array(arg0) and is_fieldlike(arg1))

    def __init__(self, *args, **kw):
        Multiply.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0, arg1.data, out.data)


class Divide(Arithmetic, metaclass=CachedMultiClass):

    name = 'Div'
    str_op = ' / '


class DivFieldField(Divide):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Divide.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = self.args[0].constant & self.args[1].constant

    def check_conditions(self):
        # Must be in grid layout
        return ((self.args[0].layout is self._grid_layout) and
                (self.args[1].layout is self._grid_layout))

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in grid layout
        arg0.require_grid_space()
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.divide(arg0.data, arg1.data, out.data)


class DivFieldScalar(Divide):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_scalar(arg1))

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in current layout
        out.layout = arg0.layout
        np.divide(arg0.data, arg1, out.data)


class DivFieldArray(Divide):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_array(arg1))

    def __init__(self, *args, **kw):
        Divide.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = self.args[0].constant & numeric_constant(self.args[1], self.domain)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.divide(arg0.data, arg1, out.data)


class DivNumericField(Divide):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_numeric(arg0) and is_fieldlike(arg1))

    def __init__(self, *args, **kw):
        Divide.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.divide(arg0, arg1.data, out.data)


class Power(Operator):

    name = 'Power'

    def __init__(self, arg0, power, out=None):

        # Required Attributes
        self.args = [arg0]
        self.original_args = [arg0]
        self.domain = arg0.domain
        self.out = out
        self.last_id = None
        # Additional attributes
        self.power = power
        self._grid_layout = self.domain.distributor.grid_layout
        self.build_metadata()

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, = self.args
        # Raise in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.power(arg0.data, self.power, out.data)


class MagSquared(Operator):

    name = 'MagSq'
    arity = 1

    def __init__(self, *args, **kw):
        Operator.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, = self.args
        # Multiply by complex conjugate in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg0.data.conj(), out.data)


class Differentiate(Operator):

    arity = 1

    def check_conditions(self):
        # References
        arg0, = self.args
        axis = self.axis
        # Must be in ceoff+local layout
        is_coeff = not arg0.layout.grid_space[axis]
        is_local = arg0.layout.local[axis]

        return (is_coeff and is_local)

    def build_metadata(self):
        self.constant = np.copy(self.args[0].constant)

    def operate(self, out):
        # References
        arg0, = self.args
        axis = self.axis
        # Differentiate in coeff+local space
        arg0.require_coeff_space(axis)
        arg0.require_local(axis)
        out.layout = arg0.layout
        # Use basis differentiation method
        self.basis.differentiate(arg0.data, out.data, axis=axis)


# Collect operators to expose to parser
parsable_ops = {'Integrate': Integrate,
                'Interpolate': Interpolate,
                'Absolute': Absolute,
                'Negate': Negate,
                'Add': Add,
                'Subtract': Subtract,
                'Multiply': Multiply,
                'Divide': Divide,
                'Power': Power,
                'MagSquared': MagSquared}
parsable_ops.update(UfuncWrapper.supported)


# Type tests
def is_scalar(arg):
    return np.isscalar(arg)

def is_array(arg):
    return isinstance(arg, np.ndarray)

def is_numeric(arg):
    return (is_scalar(arg) or is_array(arg))

def is_field(arg):
    return isinstance(arg, Field)

def is_fieldlike(arg):
    return isinstance(arg, (Field, Operator))


# Convenience functions
def create_diff_operator(basis_, axis_):
    """Create differentiation operator for a basis+axis."""

    if basis_.name is not None:
        name_ = 'd' + basis_.name
    else:
        name_ = 'd' + str(axis_)

    class d_(Differentiate):
        name = name_
        basis = basis_
        axis = axis_

    return d_

def numeric_constant(arg0, domain):
    """
    Determine constant array for numeric types. This may give false negatives
    for determining constancy along directions with a block size of 1.

    """

    if is_scalar(arg0):
        return np.array([True]*domain.dim)
    elif is_array(arg0):
        # Determine local constant array by comparing to grid shape
        comm = domain.distributor.comm_cart
        local_constant = np.less(arg0.shape, domain.distributor.grid_layout.local_shape(domain.dealias))
        # Communicate to eliminate edge cases where grid_layout.shape[i] == 1
        gathered_constant = comm.gather(local_constant, root=0)
        if domain.distributor.rank == 0:
            global_constant = reduce(np.bitwise_or, gathered_constant)
        else:
            global_constant = None
        global_constant = comm.bcast(global_constant, root=0)

        return global_constant

def unique_domain(*args):
    """Return unique domain from a set of fields."""

    # Get set of domains
    domains = []
    for arg in args:
        if is_fieldlike(arg):
            domains.append(arg.domain)
    domain_set = set(domains)

    if len(domain_set) > 1:
        raise ValueError("Non-unique domains")
    else:
        return list(domain_set)[0]

