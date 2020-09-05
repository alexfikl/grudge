"""Burgers operator."""

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import numpy.linalg as la

from grudge.second_order import IPDGSecondDerivative
from grudge.models import HyperbolicOperator
from grudge import sym


class BurgersOperator(HyperbolicOperator):
    def __init__(self, ambient_dim,
            flux_type=None,
            quad_tag=None,
            bc=None,
            viscosity=None,
            viscosity_scheme=None):
        super().__init__()
        assert self.ambient_dim == 1

        if viscosity_scheme is None:
            viscosity_scheme = IPDGSecondDerivative()

        if flux_type is None:
            flux_type = "lf"

        self.ambient_dim = ambient_dim

        self.flux_type = flux_type
        self.quad_tag = quad_tag
        self.bc = bc

        self.viscosity = viscosity
        self.viscosity_scheme = viscosity_scheme

    def flux(self, w):
        return w**2 / 2

    def weak_flux(self, w):
        pass

    def sym_variable(self, name="u", dd=None):
        if self.ambient_dim == 1:
            return sym.var(name, dd=dd)

        return sym.make_sym_array(name, self.ambient_dim, dd=dd)

    def sym_operator(self, u):
        def flux(pair):
            return sym.project(pair.dd, face_dd)(self.weak_flux(pair))

        face_dd = sym.DOFDesc(sym.FACE_RESTR_ALL, self.quad_tag)
        boundary_dd = sym.DOFDesc(sym.BTAG_ALL, self.quad_tag)
        quad_dd = sym.DOFDesc(sym.DTAG_VOLUME_ALL, self.quad_tag)

        quad_u = sym.project(sym.DD_VOLUME, quad_dd)(u)
        stiff_t = sym.stiffness_t(self.ambient_dim,
                dd_in=quad_dd, dd_out=sym.DD_VOLUME)

        if self.viscosity is None:
            viscosity = 0
        else:
            raise NotImplementedError

        return -sym.InverseMassOperator()(
                sum(sym.stiff_t[n](self.flux(quad_u))
                    for n in range(self.ambient_dim))
                + sym.FaceMassOperator(face_dd, sym.DD_VOLUME)(
                    flux(sym.int_tpair(u, self.quad_tag))
                    + flux(sym.bc_tpair(boundary_dd, u, self.bc))
                    )
                )

    def sym_operator(self, with_sensor):
        from grudge.symbolic import (
                Field,
                make_stiffness_t,
                make_nabla,
                InverseMassOperator,
                ElementwiseMaxOperator,
                get_flux_operator)

        from grudge.symbolic.operators import (
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        to_quad = QuadratureGridUpsampler("quad")
        to_if_quad = QuadratureInteriorFacesGridUpsampler("quad")

        u = Field("u")
        u0 = Field("u0")

        # boundary conditions -------------------------------------------------
        minv_st = make_stiffness_t(self.dimensions)
        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        def flux(u):
            return u**2/2
            #return u0*u

        emax_u = self.characteristic_velocity_optemplate(u)
        from grudge.flux.tools import make_lax_friedrichs_flux
        from pytools.obj_array import make_obj_array
        num_flux = make_lax_friedrichs_flux(
                #u0,
                to_if_quad(emax_u),
                make_obj_array([to_if_quad(u)]),
                [make_obj_array([flux(to_if_quad(u))])],
                [], strong=False)[0]

        from grudge.second_order import SecondDerivativeTarget

        if self.viscosity is not None or with_sensor:
            viscosity_coeff = 0
            if with_sensor:
                viscosity_coeff += Field("sensor")

            if isinstance(self.viscosity, float):
                viscosity_coeff += self.viscosity
            elif self.viscosity is None:
                pass
            else:
                raise TypeError("unsupported type of viscosity coefficient")

            # strong_form here allows IPDG to reuse the value of grad u.
            grad_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=True,
                    operand=u)

            self.viscosity_scheme.grad(grad_tgt, bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            div_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=False,
                    operand=viscosity_coeff*grad_tgt.minv_all)

            self.viscosity_scheme.div(div_tgt,
                    bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            viscosity_bit = div_tgt.minv_all
        else:
            viscosity_bit = 0

        return m_inv((minv_st[0](flux(to_quad(u)))) - num_flux) \
                + viscosity_bit

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return sym.NodalMax()(sym.fabs(fields))
