#!/usr/bin/env python
# encoding: utf-8
r"""
Woodward-Colella blast wave problem
===================================

Solve the one-dimensional Euler equations for inviscid, compressible flow:

.. math::
    \rho_t + (\rho u)_x & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x & = 0 \\
    E_t + (u (E + p) )_x & = 0.

The fluid is an ideal gas, with pressure given by :math:`p=\rho (\gamma-1)e` where
e is internal energy.

This script runs the Woodward-Colella blast wave interaction problem,
involving the collision of two shock waves.
"""
from clawpack import riemann
from clawpack.riemann.euler_with_efix_1D_constants import *

try:
    import sharpclaw1
except ImportError:
    from os.path import join as pjoin
    import sys
    import clawpack
    # Import module from clawpack directory
    CLAW_dir = clawpack.__file__.split('/')[:-2]
    euler_dir = pjoin('/'.join(str(p) for p in CLAW_dir), 'pyclaw', 'examples', 'euler_1d')
    sys.path.append(euler_dir)
    import sharpclaw1
    try:
        # Now try to import again
        import sharpclaw1
    except ImportError:
        print >> sys.stderr, "***\nUnable to import problem module or automatically build, try running (in the directory of this file):\n python setup.py build_ext -i\n***"
        raise

class Parameters(object):

    def __init__(self):
        r"""
        Initialization of default parameters.
        """
        self.xleft = 0.0
        self.xright = 1.0
        self.tfinal = 0.04
        self.num_output_times = 10
        self.max_steps = 100000
        self.tv_check = False
        self.check_lmm_cond = True
        self.use_petsc = False

gamma = 1.4 # Ratio of specific heats

def setup(nx=800,kernel_language='Fortran',solver_type='sharpclaw',time_integrator='SSP33',lmm_steps=None,\
        cfl=None,lim_type=1,limiter=4,tfluct_solver=True,dt_variable=True,dt_initial=None,\
        outdir='./_output',paramtrs=Parameters()):
    """
    Woodward-Colella blast wave problem
    ===================================
    This example involves a pair of interacting shock waves in 1D, involving the collision 
    of two shock waves.
    The conserved quantities are density, momentum density, and total energy density.
    """
    import overridden_fun

    if paramtrs.use_petsc:
        import clawpack.petclaw as pyclaw
        claw_package = 'clawpack.petclaw'
    else:
        from clawpack import pyclaw
        claw_package = 'clawpack.pyclaw'

    if kernel_language =='Python':
        riemann_solver = riemann.euler_1D_py.euler_roe_1D
    elif kernel_language =='Fortran':
        riemann_solver = riemann.euler_with_efix_1D

    if solver_type=='sharpclaw':
        solver = overridden_fun.set_solver(pyclaw.SharpClawSolver1D,riemann_solver,claw_package=claw_package)
        solver.time_integrator = time_integrator
        solver.lmm_steps = lmm_steps
        solver.check_lmm_cond = paramtrs.check_lmm_cond
        solver.lim_type = lim_type
        if solver.lim_type == 2: raise Exception('Cannot use WENO, choose lim_type=1')
        solver.limiters = limiter
        if cfl is not None:
            solver.cfl_desired = cfl[0]
            solver.cfl_max = cfl[1]
        elif time_integrator == 'SSP33':
            solver.cfl_max = 0.65
            solver.cfl_desired = 0.6
        solver.tfluct_solver = tfluct_solver
        if dt_variable == False:
            solver.dt_variable = False
            solver.dt_initial = dt_initial
        if solver.tfluct_solver:
            try:
                import euler_tfluct
                solver.tfluct = euler_tfluct
            except ImportError:
                import logging
                logger = logging.getLogger()
                logger.error('Unable to load tfluct solver, did you run make?')
                print 'Unable to load tfluct solver, did you run make?'
                raise
        solver.char_decomp = 2
        solver.limiters = 4
        try:
            import sharpclaw1
            solver.fmod = sharpclaw1
        except ImportError:
            pass
    elif solver_type=='classic':
        solver = overridden_fun.set_solver(pyclaw.ClawSolver1D,riemann_solver,claw_package=claw_package)
        solver.limiters = 4
    else: raise Exception('Unrecognized value of solver_type.')

    solver.kernel_language = kernel_language
    solver.tv_check = paramtrs.tv_check
    solver.use_petsc = paramtrs.use_petsc

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall

    # Initialize domain
    x = pyclaw.Dimension(paramtrs.xleft,paramtrs.xright,nx,name='x')
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn)

    state.problem_data['gamma'] = gamma
    if kernel_language =='Python':
        state.problem_data['efix'] = False

    xc = state.grid.x.centers
    state.q[density ,:] = 1.
    state.q[momentum,:] = 0.
    state.q[energy  ,:] = ( (xc<0.1)*1.e3 + (0.1<=xc)*(xc<0.9)*1.e-2 + (0.9<=xc)*1.e2 ) / (gamma - 1.)

    claw = pyclaw.Controller()
    claw.tfinal = paramtrs.tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = paramtrs.num_output_times
    claw.solver.max_steps = paramtrs.max_steps
    claw.outdir = outdir
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
    plotdata.clearfigures()  # clear any old figures,axes,items data

    plotfigure = plotdata.new_plotfigure(name='', figno=0)

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.title = 'Density'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = density
    plotitem.kwargs = {'linewidth':3}
    
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.title = 'Energy'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = energy
    plotitem.kwargs = {'linewidth':3}
    
    return plotdata

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
