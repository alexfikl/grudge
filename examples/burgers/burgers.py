__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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
import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext


class DiscontinuousTravelingWave:
    ambient_dim = 1

    a = -1.0
    b = +1.0
    final_time = 0.4

    @property
    def viscosity(self):
        return

    def __call__(self):
        x = sym.nodes(self.ambient_dim)
        t = sym.var("t", dd=sym.DD_SCALAR)

        # HW Example 5.11, pp. 159
        u_0 = 1.0
        u_1 = 2.0

        return sym.cse(sym.If(
                sym.Comparison(x[0] - 3.0 * t, ">", -0.5), u_0, u_1))


class TanhTravelingWave:
    ambient_dim = 1

    a = -1.0
    b = +1.0
    final_time = 1.0

    def __init__(self, viscosity_sym_name="mu"):
        self.viscosity_sym_name = viscosity_sym_name

    @property
    def viscosity(self):
        return sym.var(self.viscosity_sym_name, dd=sym.DD_SCALAR)

    def __call__(self):
        x = sym.nodes(self.ambient_dim)
        t = sym.var("t", dd=sym.DD_SCALAR)

        # HW Example 7.5, pp. 255
        return sym.cse(1.0 - sym.tanh((x[0] + 0.5 - t) / (2.0 * self.viscosity)))


def main(ctx_factory, flux_type="upwind", visualize=True):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    # {{{ parameters

    order = 3
    npoints = 20

    flux_type = "lf"
    quad_tag = "product"

    viscosity = 0.01
    solution = DiscontinuousTravelingWave()

    context = {}
    if solution.viscosity is not None:
        context["mu"] = viscosity

    # }}}

    # {{{ geometry

    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory, \
            QuadratureSimplexGroupFactory

    quad_tag_to_group_factory = {}
    quad_tag_to_group_factory[sym.QTAG_NONE] = \
            PolynomialWarpAndBlendGroupFactory(order)

    if quad_tag is not None:
        quad_tag_to_group_factory[quad_tag] = \
                QuadratureSimplexGroupFactory(order)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    ambient_dim = solution.ambient_dim
    mesh = generate_regular_rect_mesh(
            a=(solution.a,)*ambient_dim,
            b=(solution.b,)*ambient_dim,
            n=(npoints,)*ambient_dim,
            order=1)

    from grudge import DGDiscretizationWithBoundaries
    discr = DGDiscretizationWithBoundaries(actx, mesh,
            quad_tag_to_group_factory=quad_tag_to_group_factory)

    # }}}

    # {{{ symbolic

    from grudge.models.burgers import BurgersOperator
    op = BurgersOperator(ambient_dim,
            flux_type=flux_type,
            quad_tag=quad_tag,
            bc=solution(),
            viscosity=solution.viscosity)

    sym_u = op.sym_variable("u")
    sym_op = op.sym_operator(sym_u)

    # }}}


if __name__ == "__main__":
    main(cl.create_some_context)
