#!/usr/bin/env python
# encoding: utf-8
r"""
2D shallow water: radial dam break
==================================

Solve the 2D shallow water equations:

.. :math:
    h_t + (hu)_x + (hv)_y & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x + (huv)_y & = 0 \\
    (hv)_t + (huv)_x + (hv^2 + \frac{1}{2}gh^2)_y & = 0.

The initial condition is a circular area with high depth surrounded by lower-depth water.
The top and right boundary conditions reflect, while the bottom and left boundaries
are outflow.
"""

import numpy as np
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import depth, x_momentum, y_momentum, num_eqn
import overridden_fun

class Parameters(object):

    def __init__(self):
        r"""
        Initialization of default parameters.
        """
        self.xlower = -2.5
        self.xupper = 2.5
        self.ylower = -2.5
        self.yupper = 2.5
        self.tfinal = 2.5
        self.num_output_times = 10
        self.max_steps = 100000
        self.tv_check = False
        self.check_lmm_cond = True
        self.use_petsc = False

    def set_initial_cond(self,state,h_in=2.,h_out=1.,dam_radius=1.):
        r"""
        Set initial condition.
        """
        x0=0.
        y0=0.
        X, Y = state.p_centers
        r = np.sqrt((X-x0)**2 + (Y-y0)**2)
        r0 = dam_radius
        h = 1.+ np.exp(-10.*(r-r0)**2)
        state.q[depth     ,:,:] = h#h_in*(r<=dam_radius) + h_out*(r>dam_radius)
        state.q[x_momentum,:,:] = -X*np.exp(-10.*(r-r0)**2)##0.
        state.q[y_momentum,:,:] = -Y*np.exp(-10.*(r-r0)**2)##0.


def setup(nx=[150,150],kernel_language='Fortran',solver_type='sharpclaw',time_integrator='SSP104',lmm_steps=None,\
        cfl=None,lim_type=2,limiter=4,dt_variable=True,dt_initial=None,outdir='./_output',\
        paramtrs=Parameters()):

    if paramtrs.use_petsc:
        import clawpack.petclaw as pyclaw
        claw_package = 'clawpack.petclaw'
    else:
        from clawpack import pyclaw
        claw_package = 'clawpack.pyclaw'

    if kernel_language == 'Fortran':
        riemann_solver = riemann.shallow_roe_with_efix_2D
    else: 
        raise Exception('Use kernel_language=''Fortran''.')

    if solver_type == 'classic':
        solver = overridden_fun.set_solver(pyclaw.ClawSolver2D,riemann_solver,claw_package=claw_package)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.dimensional_split=1
    elif solver_type == 'sharpclaw':
        solver = overridden_fun.set_solver(pyclaw.SharpClawSolver2D,riemann_solver,claw_package=claw_package)
        solver.time_integrator = time_integrator
        solver.lmm_steps = lmm_steps
        solver.check_lmm_cond = paramtrs.check_lmm_cond
        solver.lim_type = lim_type
        if cfl is not None:
            solver.cfl_desired = cfl[0]
            solver.cfl_max = cfl[1]
        if lim_type == 1:
            solver.limiters = limiter
        if dt_variable == False:
            solver.dt_variable = False
            solver.dt_initial = dt_initial

    solver.kernel_language = kernel_language
    solver.tv_check = paramtrs.tv_check
    solver.use_petsc = paramtrs.use_petsc

    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.wall

    # Domain:
    x = pyclaw.Dimension(paramtrs.xlower,paramtrs.xupper,nx[0],name= 'x')
    y = pyclaw.Dimension(paramtrs.ylower,paramtrs.yupper,nx[1],name= 'y')
    domain = pyclaw.Domain([x,y])

    state = pyclaw.State(domain,num_eqn)

    # Gravitational constant
    state.problem_data['grav'] = 1.0

    paramtrs.set_initial_cond(state)

    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.solver.max_steps = paramtrs.max_steps
    claw.tfinal = paramtrs.tfinal
    claw.num_output_times = paramtrs.num_output_times
    claw.setplot = setplot
    claw.keep_copy = True

    return claw

#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    from clawpack.visclaw import colormaps

    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for depth
    plotfigure = plotdata.new_plotfigure(name='Water height', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-2.5, 2.5]
    plotaxes.ylimits = [-2.5, 2.5]
    plotaxes.title = 'Water height'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 0
    plotitem.pcolor_cmap = colormaps.red_yellow_blue
    plotitem.pcolor_cmin = 0.5
    plotitem.pcolor_cmax = 1.5
    plotitem.add_colorbar = True
    
    # Scatter plot of depth
    plotfigure = plotdata.new_plotfigure(name='Scatter plot of h', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [0., 2.5]
    plotaxes.ylimits = [0., 2.1]
    plotaxes.title = 'Scatter plot of h'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_from_2d_data')
    plotitem.plot_var = depth
    def q_vs_radius(current_data):
        from numpy import sqrt
        x = current_data.x
        y = current_data.y
        r = sqrt(x**2 + y**2)
        q = current_data.q[depth,:,:]
        return r,q
    plotitem.map_2d_to_1d = q_vs_radius
    plotitem.plotstyle = 'o'


    # Figure for x-momentum
    plotfigure = plotdata.new_plotfigure(name='Momentum in x direction', figno=2)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-2.5, 2.5]
    plotaxes.ylimits = [-2.5, 2.5]
    plotaxes.title = 'Momentum in x direction'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = x_momentum
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.add_colorbar = True
    plotitem.show = False       # show on plot?
    

    # Figure for y-momentum
    plotfigure = plotdata.new_plotfigure(name='Momentum in y direction', figno=3)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-2.5, 2.5]
    plotaxes.ylimits = [-2.5, 2.5]
    plotaxes.title = 'Momentum in y direction'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = y_momentum
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.add_colorbar = True
    plotitem.show = False       # show on plot?
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
