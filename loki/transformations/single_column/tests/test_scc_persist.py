# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, fgen
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes
from loki.transformations import PragmaModelTransformation
from loki.transformations.single_column import SCCPersistentDescriptorsTransformation


FCODE = """
subroutine driver(ngptot, nproma, nlev, state, f_q, darg)
  use field_access_module, only: sget_device_data_rdonly
  implicit none
  integer, intent(in) :: ngptot, nproma, nlev
  type(state_type), intent(inout) :: state
  type(field_3rb), intent(in) :: f_q
  real, pointer, contiguous, intent(inout) :: darg(:,:,:)
  real, pointer, contiguous :: t(:,:,:)
  real, pointer, contiguous :: q(:,:,:)
  real, allocatable :: tmp(:,:,:)
  integer :: jkglo, ibl, jl, jk

  call state%f_t%get_device_data_rdwr(t)
  call sget_device_data_rdonly(q, f_q)
  call state%f_x%get_device_data_rdonly(darg)
  call state%f_y%get_device_data_rdonly(state%y)

!$loki loop gang
  do jkglo=1,ngptot,nproma
    ibl = (jkglo - 1) / nproma + 1
!$loki loop vector
    do jl=1,nproma
      do jk=1,nlev
        t(jl,jk,ibl) = t(jl,jk,ibl) + q(jl,jk,ibl) + darg(jl,jk,ibl)
      end do
    end do
!$loki end loop vector
  end do
!$loki end loop gang
end subroutine driver
"""


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_persist_descriptors(frontend):
    routine = Subroutine.from_source(FCODE, frontend=frontend)
    SCCPersistentDescriptorsTransformation(persistent_descriptors=True,
                                           persistent_descriptors_device_check=True).apply(routine, role='driver')

    # Local field pointers become SAVE'd and get a _PREV partner; dummy arguments are left alone
    for name in ('t', 'q', 't_PREV', 'q_PREV'):
        var = routine.variable_map[name]
        assert var.type.pointer and var.type.save
        assert str(var.type.initial).lower() == 'null()'
    assert not routine.variable_map['darg'].type.save
    assert 'darg_prev' not in routine.variable_map

    # One remap block per pointer, after the last GET_DEVICE_DATA call and before the driver loop;
    # contiguous pointers also compare the device address of their target
    assert routine.variable_map['t_DEVPTR'].type.save
    assert 'darg_devptr' not in routine.variable_map
    body = routine.body.body
    remaps = [n for n in body if isinstance(n, ir.Conditional) and fgen(n.condition).lower() == 'll_persist_remap']
    assert len(remaps) == 2
    last_call = [i for i, n in enumerate(body) if isinstance(n, ir.CallStatement)][-1]
    first_loop = [i for i, n in enumerate(body) if isinstance(n, ir.Loop)][0]
    assert last_call < body.index(remaps[0]) < body.index(remaps[1]) < first_loop

    PragmaModelTransformation(directive='omp-gpu', emit_thread_limit=False).apply(routine)
    code = fgen(routine).lower().replace(' ', '').replace('&\n&', '')
    assert '!$omptargetexitdatamap(release:t)' in code
    assert '!$omptargetenterdatamap(to:t)' in code
    assert 't_prev=>t' in code
    assert 'll_persist_remap=.not.associated(t_prev,t)' in code
    assert 'if(.not.ll_persist_remap)ll_persist_remap=.not.c_associated(t_devptr,omp_get_mapped_ptr(c_loc(t),omp_get_default_device()))' in code
    assert 'if(associated(t))t_devptr=omp_get_mapped_ptr(c_loc(t),omp_get_default_device())' in code
    assert 'useomp_lib,only:omp_get_mapped_ptr,omp_get_default_device' in code
    assert 'map(to:darg)' not in code

    # Without the device check (default): the association check alone
    routine = Subroutine.from_source(FCODE, frontend=frontend)
    SCCPersistentDescriptorsTransformation(persistent_descriptors=True).apply(routine, role='driver')
    code = fgen(routine).lower().replace(' ', '').replace('&\n&', '')
    assert 'if(.not.associated(t_prev,t))then' in code
    assert 'omp_get_mapped_ptr' not in code and 't_devptr' not in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_persist_descriptors_disabled(frontend):
    routine = Subroutine.from_source(FCODE, frontend=frontend)
    code = fgen(routine)
    SCCPersistentDescriptorsTransformation().apply(routine, role='driver')
    assert fgen(routine) == code
    SCCPersistentDescriptorsTransformation(persistent_descriptors=True).apply(routine, role='kernel')
    assert fgen(routine) == code
    SCCPersistentDescriptorsTransformation(persistent_descriptors=['other']).apply(routine, role='driver')
    assert fgen(routine) == code
    SCCPersistentDescriptorsTransformation(persistent_descriptors=['DRIVER']).apply(routine, role='driver')
    assert 't_prev' in routine.variable_map
