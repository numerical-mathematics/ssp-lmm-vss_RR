#!/usr/bin/env python
# encoding: utf-8

r"""
One-dimensional advection
=========================

Solve the linear advection equation:

.. math:: 
    q_t + u q_x & = 0.

Here q is the density of some conserved quantity and u is the velocity.

The initial condition is a Gaussian and the boundary conditions are periodic.
The final solution is identical to the initial data because the wave has
crossed the domain exactly once.
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
        self.tfinal = 1.0
        self.num_output_times = 10
        self.max_steps = 100000
        self.tv_check = True
        self.check_lmm_cond = True
        self.use_petsc = False

    def set_initial_cond(self,x,IC=None):
        r"""
        Set initial condition.
        """
        if IC == 'sinusoidal':
            return np.sin(2*np.pi*x)
        elif IC == 'gaussian':
            beta=100; gamma=0; x0=0.5
            return np.exp(-beta * (x-x0)**2) * np.cos(gamma * (x - x0))

def variable_speed(solver,state):
    var_velocity = 2. + 1.5*np.sin(2.*np.pi*state.t)
    if solver.kernel_language == 'Python':
        state.problem_data['u'] = var_velocity
    else:
        solver.rp.cparam.u = var_velocity

def setup(nx=100,kernel_language='Fortran',solver_type='sharpclaw',time_integrator='SSP104',lmm_steps=None,\
        cfl=None,lim_type=2,limiter=4,dt_variable=True,dt_initial=None,outdir='./_output',\
        paramtrs=Parameters()):

    from clawpack import riemann
    import overridden_fun
    
    if paramtrs.use_petsc:
        import clawpack.petclaw as pyclaw
        claw_package = 'clawpack.petclaw'
    else:
        from clawpack import pyclaw
        claw_package = 'clawpack.pyclaw'

    if kernel_language == 'Fortran':
        riemann_solver = riemann.advection_1D
    elif kernel_language == 'Python':
        riemann_solver = riemann.advection_1D_py.advection_1D
            
    if solver_type=='classic':
        solver = overridden_fun.set_solver(pyclaw.ClawSolver1D,riemann_solver,claw_package=claw_package)
    elif solver_type=='sharpclaw':
        solver = overridden_fun.set_solver(pyclaw.SharpClawSolver1D,riemann_solver,claw_package=claw_package)
        solver.time_integrator = time_integrator
        solver.lmm_steps = lmm_steps
        solver.check_lmm_cond = paramtrs.check_lmm_cond
        solver.call_before_step_each_stage = True        
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

    x = pyclaw.Dimension(paramtrs.xleft,paramtrs.xright,nx,name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain,solver.num_eqn)

    state.problem_data['u'] = 2.  # Initial advection velocity
    solver.before_step = variable_speed

    # Initial data    
    xc = state.grid.x.centers
    state.q[0,:] = paramtrs.set_initial_cond(xc,paramtrs.IC)

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.solver.max_steps = paramtrs.max_steps

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None

    claw.tfinal = paramtrs.tfinal
    claw.num_output_times = paramtrs.num_output_times
    claw.setplot = setplot

    return claw

def setplot(plotdata):
    """ 
    Plot solution using VisClaw.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    plotfigure = plotdata.new_plotfigure(name='q', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.ylimits = 'auto'
    plotaxes.xlimits = 'auto'
    plotaxes.title = 'q'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-o'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':2,'markersize':5}
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
