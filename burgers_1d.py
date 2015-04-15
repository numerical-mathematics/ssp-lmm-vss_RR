#!/usr/bin/env python
# encoding: utf-8

r"""
Solve the inviscid Burgers' equation:

.. math:: 
    q_t + \frac{1}{2} (q^2)_x & = 0.

This is a nonlinear PDE often used as a very simple
model for fluid dynamics.

The initial condition is sinusoidal, but after a short time a shock forms
(due to the nonlinearity).
"""

import numpy as np

class Parameters(object):

    def __init__(self):
        r"""
        Initialization of default parameters.
        """
        self.xleft = 0.0
        self.xright = 1.0
        self.IC = 'sinusoidal'
        self.tfinal = 0.5
        self.num_output_times = 10
        self.max_steps = 100000
        self.tv_check = False
        self.check_lmm_cond = True
        self.use_petsc = False

    def set_initial_cond(self,x,IC=None):
        r"""
        Set initial condition.
        """
        if IC == 'sinusoidal':
            return np.sin(np.pi*2*x) + 0.5
        elif IC == 'Heaviside':
            x0=0.5
            return 1.0*(x<x0)
        elif IC == 'square wave':
            x1 = 0.25; x2 = 0.75
            return 1.0*(x>x1)*(x<x2) 
        elif IC == 'triangle':
            # Need to set: xleft = -1.25, xright = 1.25
            return (1.0 - np.abs(x))*(-1.0<x)*(x<1.0)

def setup(nx=100,kernel_language='Fortran',solver_type='sharpclaw',time_integrator='SSP104',lmm_steps=None,\
        cfl=None,lim_type=2,limiter=4,dt_variable=True,dt_initial=None,outdir='./_output',paramtrs=Parameters()):
    """
    Burgers' equation
    =========================
    Example python script for solving the 1d Burgers' equation.
    """

    from clawpack import riemann
    import overridden_fun

    if paramtrs.use_petsc:
        import clawpack.petclaw as pyclaw
        claw_package = 'clawpack.petclaw'
    else:
        from clawpack import pyclaw
        claw_package = 'clawpack.pyclaw'

    if kernel_language == 'Fortran':
        riemann_solver = riemann.burgers_1D
    elif kernel_language == 'Python':
        riemann_solver = riemann.burgers_1D_py.burgers_1D

    if solver_type=='classic':
        solver = overridden_fun.set_solver(pyclaw.ClawSolver1D,riemann_solver,claw_package=claw_package)
        solver.limiters = pyclaw.limiters.tvd.vanleer
    elif solver_type=='sharpclaw':
        solver = overridden_fun.set_solver(pyclaw.SharpClawSolver1D,riemann_solver,claw_package=claw_package)
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
    else: raise Exception('Unrecognized value of solver_type.')

    solver.kernel_language = kernel_language
    solver.tv_check = paramtrs.tv_check
    solver.use_petsc = paramtrs.use_petsc

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic

    x = pyclaw.Dimension(paramtrs.xleft,paramtrs.xright,nx,name= 'x')
    domain = pyclaw.Domain(x)
    num_eqn = 1
    state = pyclaw.State(domain,num_eqn)

    grid = state.grid
    xc=grid.x.centers
    state.q[0,:] = paramtrs.set_initial_cond(xc,paramtrs.IC)
    state.problem_data['efix']=True

    claw = pyclaw.Controller()
    claw.tfinal = paramtrs.tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True
    claw.num_output_times = paramtrs.num_output_times
    claw.solver.max_steps = paramtrs.max_steps

    return claw

#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for q[0]
    plotfigure = plotdata.new_plotfigure(name='q[0]', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-1., 2.]
    #plotaxes.ylimits = 'auto'
    plotaxes.title = 'q[0]'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-o'
    plotitem.color = 'b'
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
