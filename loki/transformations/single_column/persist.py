# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

from loki.batch import Transformation
from loki.expression import symbols as sym
from loki.ir import nodes as ir, FindNodes, Transformer
from loki.logging import info
from loki.tools import as_tuple
from loki.types import BasicType, DerivedType, SymbolAttributes


__all__ = ['SCCPersistentDescriptorsTransformation']


class SCCPersistentDescriptorsTransformation(Transformation):
    """
    Keep the device descriptors of the field pointers of a driver mapped across calls.

    The driver obtains its full-field pointers from FIELD_API with
    ``GET_DEVICE_DATA_RDONLY/RDWR/WRONLY`` (method form ``X%F%GET_DEVICE_DATA_RDWR(P)``
    or ``SGET_DEVICE_DATA_RDWR(P, X%F)``). The data is already on the device, but every
    target region that uses ``P`` maps (allocates, copies and frees) its descriptor again.
    This transformation makes each such local pointer ``SAVE`` and inserts, after the last
    ``GET_DEVICE_DATA_*`` call, a remap of its descriptor that only happens when the
    descriptor would change:

    .. code-block:: fortran

        IF (.NOT. ASSOCIATED(P_PREV, P)) THEN
          IF (ASSOCIATED(P_PREV)) THEN
        !$loki exit unstructured-data detach(P)
          END IF
        !$loki unstructured-data in(P)
          P_PREV => P
        END IF

    ``ASSOCIATED(P_PREV, P)`` compares the host target and shape. A field that is deleted and
    recreated can get the same host address but a different device allocation. The OpenMP
    runtime (afar 24.3.0) handles that case itself: when a target region maps the present
    pointer and its pointee now has a different device address, it copies a corrected
    descriptor (one 48-byte copy, only when the pointee moved). The optional device check
    (``CONTIGUOUS`` pointers) remaps such descriptors in the driver instead, at the cost of one
    ``OMP_GET_MAPPED_PTR`` lookup per pointer and call.
    The target regions then find the descriptor present. Data freshness is unchanged:
    ``GET_DEVICE_DATA_*`` is still called on every call.

    Parameters
    ----------
    persistent_descriptors : bool or list of str
        Apply the transformation to all driver routines, or only to the named ones.
    persistent_descriptors_device_check : bool
        Also remap when the device address of the target changed (OpenMP offload only;
        off by default, see above).
    """

    _get_device_data = re.compile(r'^s?get_device_data_(rdonly|rdwr|wronly)$', re.IGNORECASE)

    def __init__(self, persistent_descriptors=False, persistent_descriptors_device_check=False):
        self.persistent_descriptors = persistent_descriptors
        self.device_check = persistent_descriptors_device_check

    def transform_subroutine(self, routine, **kwargs):
        if not self.persistent_descriptors or kwargs.get('role') != 'driver':
            return
        if not isinstance(self.persistent_descriptors, bool):
            if routine.name.lower() not in {str(n).lower() for n in as_tuple(self.persistent_descriptors)}:
                return

        args = {str(a.name).lower() for a in routine.arguments}
        pointers = {}
        last = None
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if not self._get_device_data.match(str(call.name).split('%')[-1]):
                continue
            if not call.arguments:
                continue
            last = call
            ptr = call.arguments[0]
            if not isinstance(ptr, sym.Array) or ptr.dimensions or getattr(ptr, 'parent', None):
                continue
            if str(ptr.name).lower() in args or ptr.name not in routine.variable_map:
                continue
            if not ptr.type.pointer:
                continue
            pointers.setdefault(str(ptr.name).lower(), ptr)
        if not pointers:
            return

        anchor = self._top_level_node(routine.body.body, last)
        if anchor is None:
            return

        null = sym.InlineCall(sym.DeferredTypeSymbol('NULL'), parameters=())
        associated = sym.DeferredTypeSymbol('ASSOCIATED')
        c_associated = sym.DeferredTypeSymbol('C_ASSOCIATED')
        c_loc = sym.DeferredTypeSymbol('C_LOC')
        mapped_ptr = sym.DeferredTypeSymbol('OMP_GET_MAPPED_PTR')
        device = sym.InlineCall(sym.DeferredTypeSymbol('OMP_GET_DEFAULT_DEVICE'), parameters=())
        remap = sym.Variable(name='LL_PERSIST_REMAP', type=SymbolAttributes(BasicType.LOGICAL), scope=routine)
        c_ptr_type = SymbolAttributes(DerivedType('C_PTR'), save=True)

        new_vars = []
        checks = [ir.Comment(text=''),
                  ir.Comment(text='! descriptors stay mapped across calls; remap only when the descriptor changes')]
        for ptr in pointers.values():
            routine.symbol_attrs[ptr.name] = ptr.type.clone(save=True, initial=null)
            var = routine.variable_map[ptr.name].clone(dimensions=None)
            prev = routine.variable_map[ptr.name].clone(name=f'{ptr.name}_PREV', scope=routine,
                                                        type=routine.symbol_attrs[ptr.name])
            new_vars.append(prev)
            prev = prev.clone(dimensions=None)

            check_device = self.device_check and ptr.type.contiguous
            if check_device:
                devptr = sym.Variable(name=f'{ptr.name}_DEVPTR', type=c_ptr_type, scope=routine)
                new_vars.append(devptr)
                current = sym.InlineCall(mapped_ptr, parameters=(sym.InlineCall(c_loc, parameters=(var,)), device))
                checks += [
                    ir.Assignment(lhs=remap, rhs=sym.LogicalNot(sym.InlineCall(associated, parameters=(prev, var)))),
                    ir.Conditional(
                        condition=sym.LogicalNot(remap), inline=True,
                        body=(ir.Assignment(lhs=remap, rhs=sym.LogicalNot(
                            sym.InlineCall(c_associated, parameters=(devptr, current)))),)
                    )
                ]
                condition = remap
            else:
                condition = sym.LogicalNot(sym.InlineCall(associated, parameters=(prev, var)))

            release = ir.Conditional(
                condition=sym.InlineCall(associated, parameters=(prev,)),
                body=(ir.Pragma(keyword='loki', content=f'exit unstructured-data detach({ptr.name})'),)
            )
            body = (release,
                    ir.Pragma(keyword='loki', content=f'unstructured-data in({ptr.name})'),
                    ir.Assignment(lhs=prev, rhs=var, ptr=True))
            if check_device:
                body += (ir.Conditional(condition=sym.InlineCall(associated, parameters=(var,)), inline=True,
                                        body=(ir.Assignment(lhs=devptr, rhs=current),)),)
            checks.append(ir.Conditional(condition=condition, body=body))

        if any(v.name.endswith('_DEVPTR') for v in new_vars):
            new_vars.append(remap)
            routine.spec.prepend((
                ir.Import(module='ISO_C_BINDING', symbols=tuple(
                    sym.Variable(name=n, scope=routine) for n in ('C_PTR', 'C_LOC', 'C_ASSOCIATED'))),
                ir.Import(module='OMP_LIB', symbols=tuple(
                    sym.Variable(name=n, scope=routine) for n in ('OMP_GET_MAPPED_PTR', 'OMP_GET_DEFAULT_DEVICE')))
            ))
        routine.variables += as_tuple(new_vars)

        routine.body = Transformer({anchor: (anchor,) + tuple(checks)}).visit(routine.body)
        info(f'[Loki-SCC::Persist] {routine.name}: {len(pointers)} persistent descriptors')

    @staticmethod
    def _top_level_node(body, node):
        """The node of ``body`` that is or contains ``node``."""
        for n in body:
            if n is node or node in FindNodes(type(node)).visit(n):
                return n
        return None
