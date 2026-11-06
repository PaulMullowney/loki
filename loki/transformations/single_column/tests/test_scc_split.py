# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Dimension, fgen
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes, get_pragma_parameters, pragmas_attached
from loki.transformations import PragmaModelTransformation
from loki.transformations.single_column import (
    SCCSplitDriverLoopsTransformation, SCCCollapseTransformation
)

HORIZONTAL = Dimension(name='horizontal', size=('KLON', 'NPROMA'), index='JL', bounds=('KIDIA', 'KFDIA'))

DRIVER = '''
subroutine driver(ngptot, nproma, nlev, klon, ldflag, a, b, z)
  integer, intent(in) :: ngptot, nproma, nlev, klon
  logical, intent(in) :: ldflag
  real, intent(inout) :: a(nproma, nlev, *), b(nproma, *), z(klon, nlev, *)
  integer :: jkglo, kidia, kfdia, ibl, jl, jk
  real :: ztmp
  !$loki loop gang firstprivate(nlev)
  do jkglo = 1, ngptot, nproma
    kidia = 1
    kfdia = min(nproma, ngptot - jkglo + 1)
    ibl = (jkglo - 1) / nproma + 1
    !$loki loop seq
    do jk = 1, nlev
      do jl = 1, klon
        z(jl, jk, ibl) = 0.
      end do
    end do
    !$loki loop vector
    do jl = kidia, kfdia
      ztmp = 2. * b(jl, ibl)
      a(jl, 1, ibl) = ztmp
    end do
    !$loki end loop vector
    if (ldflag) then
      call abor1('flag')
    end if
    !$loki loop vector
    do jl = kidia, kfdia
      b(jl, ibl) = b(jl, ibl) + a(jl, 1, ibl)
    end do
    !$loki end loop vector
  end do
  !$loki end loop gang
end subroutine driver
'''


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_split_driver_loops(frontend, tmp_path):
    """Each vector loop and the zeroing nest become perfect nests that then collapse."""
    routine = Subroutine.from_source(DRIVER, frontend=frontend, xmods=[tmp_path])
    SCCSplitDriverLoopsTransformation(split_driver_loops=True, horizontal=HORIZONTAL).apply(routine, role='driver')
    with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
        outers = [l for l in FindNodes(ir.Loop).visit(routine.body) if l.variable == 'jkglo']
        assert len(outers) == 3
        for outer in outers:
            params = get_pragma_parameters(outer.pragma, starts_with='loop')
            assert params['firstprivate'] == 'nlev'
            assert {v.strip().lower() for v in params['private'].split(',')} >= {'kidia', 'kfdia', 'ibl'}
            inner = outer.body[0]
            assert len(outer.body) == 1 and inner.variable == 'jl'
            assert str(inner.bounds.lower) == '1'
        # zeroing nest: over KLON, no guard; vector loops: over NPROMA, guarded, ztmp private
        assert str(outers[0].body[0].bounds.upper).lower() == 'klon'
        assert not FindNodes(ir.Conditional).visit(outers[0].body[0].body)
        assert str(outers[1].body[0].bounds.upper).lower() == 'nproma'
        assert 'ztmp' in get_pragma_parameters(outers[1].pragma, starts_with='loop')['private'].lower()
        assert len(FindNodes(ir.Conditional).visit(outers[1].body[0].body)) == 1
    # the abort stays on the host between the nests
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert [str(c.name).lower() for c in calls] == ['abor1']

    SCCCollapseTransformation(collapse_gang_vector=True, collapse_data_sharing=True, collapse_strided_gang=True).apply(routine, role='driver')
    PragmaModelTransformation(directive='omp-gpu', emit_thread_limit=False).apply(routine)
    code = ' '.join(fgen(routine).lower().replace('( ', '(').replace(' )', ')').split())
    assert code.count('!$omp target teams distribute parallel do collapse(2)') == 3
    assert 'if (jl >= kidia .and. jl <= kfdia) then' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_split_carried_scalar(frontend, tmp_path):
    """A scalar set in one vector loop and read in a later one leaves the driver loop unchanged."""
    source = DRIVER.replace('b(jl, ibl) = b(jl, ibl) + a(jl, 1, ibl)', 'b(jl, ibl) = ztmp')
    routine = Subroutine.from_source(source, frontend=frontend, xmods=[tmp_path])
    before = fgen(routine)
    SCCSplitDriverLoopsTransformation(split_driver_loops=True, horizontal=HORIZONTAL).apply(routine, role='driver')
    assert fgen(routine) == before


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_split_disabled(frontend, tmp_path):
    """Disabled by default and for kernels."""
    routine = Subroutine.from_source(DRIVER, frontend=frontend, xmods=[tmp_path])
    before = fgen(routine)
    SCCSplitDriverLoopsTransformation(horizontal=HORIZONTAL).apply(routine, role='driver')
    SCCSplitDriverLoopsTransformation(split_driver_loops=True, horizontal=HORIZONTAL).apply(routine, role='kernel')
    assert fgen(routine) == before


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_split_enclosing_data_region(frontend, tmp_path):
    """Pragmas of an enclosing data region stay once, around the split nests."""
    source = DRIVER.replace('  !$loki loop gang firstprivate(nlev)', '  !$loki data present(a, b, z)\n  !$loki loop gang firstprivate(nlev)')
    source = source.replace('  !$loki end loop gang\n', '  !$loki end loop gang\n  !$loki end data\n')
    routine = Subroutine.from_source(source, frontend=frontend, xmods=[tmp_path])
    SCCSplitDriverLoopsTransformation(split_driver_loops=True, horizontal=HORIZONTAL).apply(routine, role='driver')
    SCCCollapseTransformation(collapse_gang_vector=True, collapse_data_sharing=True,
                              collapse_strided_gang=True).apply(routine, role='driver')
    PragmaModelTransformation(directive='omp-gpu', emit_thread_limit=False).apply(routine)
    code = ' '.join(fgen(routine).lower().replace('( ', '(').replace(' )', ')').split())
    assert code.count('target teams distribute parallel do collapse(2)') == 3
    assert code.count('!$loki data present') == 1 and code.count('!$loki end data') == 1
    assert code.index('!$loki data present') < code.index('collapse(2)') < code.rindex('collapse(2)') < code.index('!$loki end data')


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_split_hoist_invariant_setup(frontend, tmp_path):
    """Invariant scalar setup at the top of a vector loop moves to the host (postphy ILEVOML)."""
    source = DRIVER.replace('      ztmp = 2. * b(jl, ibl)\n', '      if (ldflag) then\n        ilev = 0\n      else\n        ilev = nlev\n      end if\n      ztmp = 2. * b(jl, ibl)\n')
    source = source.replace('b(jl, ibl) = b(jl, ibl) + a(jl, 1, ibl)', 'b(jl, ibl) = b(jl, ibl) + a(jl, ilev, ibl)')
    source = source.replace('  real :: ztmp\n', '  real :: ztmp\n  integer :: ilev\n')
    routine = Subroutine.from_source(source, frontend=frontend, xmods=[tmp_path])
    SCCSplitDriverLoopsTransformation(split_driver_loops=True, horizontal=HORIZONTAL).apply(routine, role='driver')
    with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
        outers = [l for l in FindNodes(ir.Loop).visit(routine.body) if l.variable == 'jkglo']
        assert len(outers) == 3
        assert not any('ilev' in fgen(o).lower().split('if (jl')[0] and 'ilev =' in fgen(o).lower() for o in outers)
    host = [n for n in routine.body.body if isinstance(n, ir.Conditional)]
    assert any('ilev = 0' in fgen(c).lower() for c in host)
