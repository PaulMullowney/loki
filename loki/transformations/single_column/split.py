# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer,
    get_pragma_command_and_parameters, pragmas_attached
)
from loki.logging import info
from loki.tools import as_tuple
from loki.transformations.utilities import find_driver_loops


__all__ = ['SCCSplitDriverLoopsTransformation']


class _Unsupported(Exception):
    """Raised when a driver loop cannot be split; the loop is left unchanged."""


class SCCSplitDriverLoopsTransformation(Transformation):
    """
    Split annotated SCC driver loops into one perfectly nested gang/vector pair per vector loop.

    A driver loop ``DO JKGLO=1,NGPTOT,NPROMA`` (``!$loki loop gang``) whose body is a prelude of
    scalar block assignments (``KIDIA = ...``, ``IBL = ...``) followed by vector loops
    (``!$loki loop vector``, ``DO JL=KIDIA,KFDIA``) becomes one driver loop per vector loop:

    .. code-block:: fortran

        !$loki loop gang private(KIDIA, KFDIA, IBL, ...)
        DO JKGLO=1,NGPTOT,NPROMA
          !$loki loop vector
          DO JL=1,NPROMA
            KIDIA = ...; KFDIA = ...; IBL = ...
            IF (JL >= KIDIA .AND. JL <= KFDIA) THEN
              <vector-loop body>
            END IF
          END DO
        END DO

    so that :any:`SCCCollapseTransformation` can collapse each pair. The inner loop runs over
    the driver-loop step, the prelude is recomputed in every iteration and its scalars, as well
    as scalars assigned in the vector-loop body, become private. Block-invariant code between
    the vector loops (conditionals, loops containing vector loops, assignments and calls that
    do not use the block scalars) moves out of the driver loop, around the new nests. A
    sequential loop nest without a vector loop becomes its own nest over its horizontal loop,
    without the guard. Any other structure leaves the driver loop unchanged.

    Run it after :any:`SCCAnnotateTransformation` and before :any:`SCCCollapseTransformation`.

    Parameters
    ----------
    split_driver_loops : bool, optional
        Enable splitting, by default False.
    horizontal : :any:`Dimension`, optional
        The horizontal dimension; its sizes identify the horizontal loop of sequential nests.
    """

    def __init__(self, split_driver_loops=False, horizontal=None):
        self.split_driver_loops = split_driver_loops
        self.horizontal = horizontal

    @staticmethod
    def _loki_params(pragmas, command):
        for pragma in as_tuple(pragmas):
            if pragma.keyword.lower() == 'loki':
                cmd, params = get_pragma_command_and_parameters(pragma)
                if cmd == command:
                    return params
        return None

    @classmethod
    def _is_vector_loop(cls, node):
        if not isinstance(node, ir.Loop):
            return False
        params = cls._loki_params(node.pragma, 'loop')
        return params is not None and 'vector' in params and 'gang' not in params

    @classmethod
    def _contains_vector_loop(cls, nodes):
        return any(cls._is_vector_loop(loop) for loop in FindNodes(ir.Loop).visit(as_tuple(nodes)))

    @staticmethod
    def _names(node):
        return {v.name_parts[0].lower() for v in FindVariables().visit(node)}

    @staticmethod
    def _is_prelude_assignment(node):
        if not isinstance(node, ir.Assignment) or node.ptr:
            return False
        return not (isinstance(node.lhs, sym.Array) and node.lhs.dimensions)

    @staticmethod
    def _clause_list(params, key):
        value = params.get(key)
        return [v.strip() for v in value.split(',')] if value else []

    @staticmethod
    def _pragma_content(params, command, extra_private):
        params = dict(params)
        private = [v.strip() for v in (params.get('private') or '').split(',') if v.strip()]
        fprivate = {v.strip().lower() for v in (params.get('firstprivate') or '').split(',') if v.strip()}
        for v in extra_private:
            if v.lower() not in {p.lower() for p in private} and v.lower() not in fprivate:
                private.append(v)
        if private:
            params['private'] = ', '.join(private)
        clauses = ''.join(f' {key}({value})' if value is not None else f' {key}'
                          for key, value in params.items() if key != 'gang')
        return f'{command} gang{clauses}'

    def _horizontal_sizes(self):
        if self.horizontal is None:
            return set()
        return {str(s).lower() for s in as_tuple(getattr(self.horizontal, 'size_expressions', self.horizontal.size))}

    def _horizontal_loop(self, nest, block_size):
        """Return the innermost loop of a sequential nest that runs over the full horizontal range."""
        sizes = self._horizontal_sizes() | {str(block_size).lower()}
        for loop in reversed(FindNodes(ir.Loop).visit(nest)):
            lower, upper = loop.bounds.lower, loop.bounds.upper
            if str(lower) == '1' and str(upper).lower() in sizes and loop.bounds.step in (None, 1):
                return loop
        return None

    def transform_subroutine(self, routine, **kwargs):
        if not self.split_driver_loops or kwargs.get('role') != 'driver':
            return
        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            mapper = {}
            for outer in find_driver_loops(routine.body, targets=()):
                params = self._loki_params(outer.pragma, 'loop')
                if params is None or 'gang' not in params or 'vector' in params:
                    continue
                try:
                    mapper[outer] = self.split_loop(outer, params)
                except _Unsupported as exc:
                    info(f'[Loki-SCC::Split] Not splitting {routine.name}/{outer.variable}: {exc}')
            if mapper:
                routine.body = Transformer(mapper).visit(routine.body)

    def split_loop(self, outer, params):
        """Return the host-side sequence of nodes that replaces the driver loop ``outer``."""
        block_size = outer.bounds.step
        if block_size is None or str(block_size) == '1':
            raise _Unsupported('driver loop without a block-size step')
        body = [n for n in outer.body if not isinstance(n, ir.Pragma)]
        prelude = []
        while body and (self._is_prelude_assignment(body[0]) or isinstance(body[0], (ir.Comment, ir.CommentBlock))):
            node = body.pop(0)
            if isinstance(node, ir.Assignment):
                prelude.append(node)
        if not self._contains_vector_loop(body) and not FindNodes(ir.Loop).visit(tuple(body)):
            raise _Unsupported('no vector loop or sequential nest')
        block_names = {outer.variable.name.lower()} | {a.lhs.name_parts[0].lower() for a in prelude}
        prelude_private = list(dict.fromkeys(a.lhs.name_parts[0] for a in prelude))
        self._kernels = []
        ctx = {'outer': outer, 'params': params, 'prelude': prelude, 'block_names': block_names,
               'prelude_private': prelude_private, 'block_size': block_size}
        nodes = self._emit(body, ctx)
        self._check_cross_kernel_scalars()
        # Pragmas attached to the driver loop that are not its gang annotation (e.g. the
        # end of an enclosing data region) stay once, around the new sequence.
        before = tuple(p for p in as_tuple(outer.pragma) if not self._is_gang_pragma(p, 'loop'))
        after = tuple(p for p in as_tuple(outer.pragma_post) if not self._is_gang_pragma(p, 'end-loop'))
        blank = (ir.Comment(text=''),)
        return before + (blank if before else ()) + tuple(nodes) + (blank if after else ()) + after

    @staticmethod
    def _is_gang_pragma(pragma, command):
        if pragma.keyword.lower() != 'loki':
            return False
        cmd, params = get_pragma_command_and_parameters(pragma)
        return cmd == command and 'gang' in params

    def _emit(self, nodes, ctx):
        out = self._Separated()
        for node in nodes:
            if isinstance(node, (ir.Comment, ir.CommentBlock)):
                out.append(node)
            elif self._is_vector_loop(node):
                lower, upper = node.bounds.lower, node.bounds.upper
                hoisted, body = self._invariant_prefix(node, ctx)
                out.extend(hoisted)
                out.append(self._kernel(node.variable, body, ctx, guard=(lower, upper)))
            elif isinstance(node, ir.Conditional):
                if self._names(node.condition) & ctx['block_names']:
                    raise _Unsupported(f'block-dependent condition {node.condition}')
                if self._contains_vector_loop(node) or FindNodes(ir.Loop).visit(node):
                    out.append(node.clone(body=tuple(self._emit(node.body, ctx)),
                                          else_body=tuple(self._emit(node.else_body, ctx))))
                elif self._names(node) & ctx['block_names']:
                    raise _Unsupported('block-dependent conditional without a vector loop')
                else:
                    out.append(node)
            elif isinstance(node, ir.Loop):
                if self._contains_vector_loop(node.body):
                    if self._names(node.bounds) & ctx['block_names']:
                        raise _Unsupported(f'block-dependent bounds of loop {node.variable}')
                    out.append(node.clone(body=tuple(self._emit(node.body, ctx)), pragma=None, pragma_post=None))
                else:
                    out.append(self._sequential_nest_kernel(node, ctx))
            elif isinstance(node, (ir.Assignment, ir.CallStatement)):
                if self._names(node) & ctx['block_names']:
                    raise _Unsupported(f'block-dependent statement outside the prelude: {node}')
                out.append(node)
            else:
                raise _Unsupported(f'unsupported node {type(node).__name__}')
        return list(out)

    class _Separated(list):
        """Node list that keeps a blank line between adjacent loops, so that the end pragma of
        one nest is not attached to the next nest when pragmas are re-attached."""

        def append(self, node):
            if self and isinstance(self[-1], ir.Loop) and isinstance(node, ir.Loop):
                super().append(ir.Comment(text=''))
            super().append(node)

    def _is_invariant_setup(self, node, excluded):
        """Scalar assignments (optionally under a conditional) that use no arrays, calls,
        block scalars or the vector index: they can run once on the host."""
        if isinstance(node, ir.Assignment):
            if node.ptr or isinstance(node.lhs, sym.Array):
                return False
        elif isinstance(node, ir.Conditional):
            if not all(isinstance(n, (ir.Assignment, ir.Conditional, ir.Comment)) for n in node.body + node.else_body):
                return False
            if not all(self._is_invariant_setup(n, excluded) for n in node.body + node.else_body
                       if not isinstance(n, ir.Comment)):
                return False
        else:
            return False
        if FindNodes(ir.CallStatement).visit(node):
            return False
        variables = FindVariables().visit(node)
        if any(isinstance(v, sym.Array) for v in variables):
            return False
        return not ({v.name_parts[0].lower() for v in variables} & excluded)

    def _invariant_prefix(self, loop, ctx):
        """Split the leading block- and column-invariant scalar setup off a vector-loop body."""
        excluded = ctx['block_names'] | {loop.variable.name.lower()}
        body = list(loop.body)
        hoisted = []
        while body and (isinstance(body[0], (ir.Comment, ir.CommentBlock))
                        or self._is_invariant_setup(body[0], excluded)):
            node = body.pop(0)
            if not isinstance(node, (ir.Comment, ir.CommentBlock)):
                hoisted.append(node)
        # Scalars set by the hoisted code must not be modified later in the loop body
        targets = {a.lhs.name.lower() for a in FindNodes(ir.Assignment).visit(tuple(hoisted))}
        rewritten = {a.lhs.name_parts[0].lower() for a in FindNodes(ir.Assignment).visit(tuple(body))}
        if targets & rewritten:
            return [], loop.body
        return hoisted, tuple(body)

    def _sequential_nest_kernel(self, nest, ctx):
        """A sequential nest that contains its horizontal loop becomes a nest over that loop."""
        hloop = self._horizontal_loop(nest, ctx['block_size'])
        if hloop is None:
            raise _Unsupported(f'sequential nest {nest.variable} without a full horizontal loop')
        if FindNodes((ir.CallStatement, ir.Conditional)).visit(nest):
            raise _Unsupported(f'sequential nest {nest.variable} with calls or conditionals')
        if hloop is nest:
            body = nest.body
        else:
            body = (Transformer({hloop: hloop.body}).visit(nest),)
        return self._kernel(hloop.variable, body, ctx, guard=None, upper=hloop.bounds.upper)

    def _kernel(self, variable, body, ctx, guard, upper=None):
        outer = ctx['outer']
        inner_body = tuple(a.clone() for a in ctx['prelude'])
        if guard is not None:
            condition = sym.LogicalAnd((sym.Comparison(variable, '>=', guard[0]),
                                        sym.Comparison(variable, '<=', guard[1])))
            inner_body += (ir.Conditional(condition=condition, body=as_tuple(body), else_body=()),)
        else:
            inner_body += as_tuple(body)
        loop_vars = {l.variable.name.lower() for l in FindNodes(ir.Loop).visit(inner_body)} | {variable.name.lower()}
        assigned = [a.lhs for a in FindNodes(ir.Assignment).visit(as_tuple(body))
                    if isinstance(a.lhs, sym.Scalar) and not a.lhs.parent and not a.ptr
                    and a.lhs.name.lower() not in loop_vars]
        body_private = list(dict.fromkeys(v.name for v in assigned))
        self._kernels.append({'private': {v.lower() for v in body_private}, 'body': as_tuple(body)})
        inner = ir.Loop(variable=variable, bounds=sym.LoopRange((sym.IntLiteral(1), upper or ctx['block_size'])),
                        body=inner_body, pragma=(ir.Pragma(keyword='loki', content='loop vector'),),
                        pragma_post=(ir.Pragma(keyword='loki', content='end loop vector'),))
        content = self._pragma_content(ctx['params'], 'loop', ctx['prelude_private'] + body_private)
        pragma_post = tuple(p for p in as_tuple(outer.pragma_post) if self._is_gang_pragma(p, 'end-loop'))
        return outer.clone(body=(inner,), pragma=(ir.Pragma(keyword='loki', content=content),),
                           pragma_post=pragma_post)

    def _check_cross_kernel_scalars(self):
        """A scalar set in one new nest and read in a later one would see the host value there."""
        for i, kernel in enumerate(self._kernels):
            for later in self._kernels[i + 1:]:
                carried = (self._names(later['body']) & kernel['private']) - later['private']
                if carried:
                    raise _Unsupported(f'scalars {sorted(carried)} are set in one nest and read in a later one')
