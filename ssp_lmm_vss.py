#!/usr/bin/env python
# encoding: utf-8

r"""
This script tests explicit LMM methods with variable step size on various 1D and 2D problems.
The 2nd and 3rd order LMM methods are implemented in Pyclaw and the problems tested are 
modifications of the relative examples in Clawpack.

The main purpose of this file is to reproduce the numerical results of the paper:
    ``Strong stability preserving explicit linear multistep methods with variable step size``
It also demostrates how to the solve hyperbolic conservation problems by externally accessing 
Pyclaw. This allows to run modified versions for each problem and adjust Pyclaw's solver to our needs.

The problems solved by using this script are:
    1. 1D advection
    2. 1D burgers
    3. 1D Woodward-Colella blast (Euler equations)
    4. 2D radial shallow water problem
    5. 2d shock bubble interaction (Euler equations)

Example 1:
    To generate data for the advection converegence test:
        >>> tab1_getdata(norm=1,grid_level=[7,11],use_petsc=False)
    
    The error is measured in L1-norm using 2^k grid cells with k=7,8,...,11.
    We can plot the results or check the converegence order by running:
        >>> fig_tab1(plot=True)

Example 2:
    To generate the step size plots for the 1D-problems, first create the solution data
        >>> fig2_getdata()

    and then the relative plot:
        >>> plot_fig2()

Note:
    To reproduce the solution data for Shock-bubble interaction problem is recommened to run in parallel,
    using PETSC:
        >>> mpirun -n 4 python run_shockbubble.py

This script is devided in three sections. The first section includes functions that solve the
problems, create the plots and perform converenge test for advection.
The second section contains functions that reproduce the data presented in the paper and the third
section includes functions for visualizing the results.
The default options for each fucntion in the second and third section reproduce the results 
as shown in the paper.
"""

sspcoeff = {
   'SSPLMM32':  0.5,
   'SSPLMM42':  2/3.,
   'SSPLMM43':  1/3.,
   'SSPLMM53':  0.5,
   'Euler':     1.0,
   'SSP33':     1.0,
   'SSP104':    6.0
   }

cfl_nums = {
    'SSPLMM32': [0.24,0.25],
    'SSPLMM42': [0.3,1/3.],
    'SSPLMM43': [0.15,1/6.],
    'SSPLMM53': [0.24,0.25],
    'SSP33':    [0.9,1.0],
    'SSP44':    [2.45,2.5]
    }

lim_type = {
    'SSPLMM32': 1,
    'SSPLMM42': 1,
    'SSPLMM43': 2,
    'SSPLMM53': 2,
    'SSP33':    2,
    'SSP104':   2
    }

spatial = {'1':'TVD','2':'WENO'}
limiters = {'0':'weno','1':'Minmod','2':'Superbee','3':'Van leer','4':'MC'}

class Data:
    r"""
    Class for solution data objects.
    """
    def __init__(self,problem,method):
        self.problem = problem
        self.method = method



#=========================================================
# Functions to perform numerical tests on LMMs
#=========================================================
def run(problem,N=256,method='SSPLMM32',steps=None,solver_type='sharpclaw',cfl_nums=None,lim_type=2,limiter=4,\
        dt_variable=True,dt_initial=None,paramtrs=None,iplot=True):
    r"""
    Solve a given HCL with a LMM or RK method using PyClaw
    ===============================================================================
    problem:     'advection_1d, 'burgers_1d', 'woodward_colella_1d',
                 'radial_shallow_water_2d, 'shock_bubble_interaction_2d'
    method:      'SSPLMMk2', (where k is the number of steps),
                 'SSPLMM43', 'SSPLMM53'
                 + other time-integrator included in Sharpclaw
    steps:       LMM steps (default = 4)
    solver_type: Pyclaw solver (default = sharpclaw)
    cfl_nums:    A dictionary containing the desired and maximum CFL number for each method;
                 If set to None then the default cfl_max/cfl_desired as in Pyclaw is used.
    lim_type:    1: TVD reconstruction, 2: WENO reconstruction
    limiter:     Limiter for TVD reconstruction (default is MC limiter; see Pyclaw for more options)
    dt_variable: Boolean variable, If True variable step size is enabled, otherwise fixed step sized is used.
    dt_initial:  Initial step size, If None then Pyclaw's default is used (dt_initial = 0.1)
    paramtrs:    Setting of parameters for a given problem (see the `Parameters` class
                 in the specific problem script for a full discription of parameters).
                 If not specified then the default parameters for each problem are used.
                 The user can specified parameters by calling the `set_parameters` function first.
    iplot:       Plotting using Visclaw
    """

    from importlib import import_module
    prb = import_module(problem)

    if paramtrs is None:
        # set problem's default parameters
        paramtrs = prb.Parameters()

    if solver_type == 'classic':
        claw = prb.setup(nx=N,solver_type='classic',paramtrs=paramtrs)
    else:
        print '\n', problem, method, N
        if 'SSPLMM' in method:
            steps = int(method[-2])
            if method[-1] == '2':
                method = 'SSPLMMk2'
            elif method[-1] == '3':
                method = 'SSPLMMk3'
        claw = prb.setup(nx=N,time_integrator=method,lmm_steps=steps,cfl=cfl_nums,lim_type=lim_type,\
                limiter=limiter,paramtrs=paramtrs)
        status = claw.run()

    # optional plotting with visclaw
    # This does not work properly in parallel 
    if iplot:
        claw.plot()

    data = Data(problem,method)
    data.status = status
    data.delta = claw.solution.domain.grid.delta
    data.check_lmm_cond = claw.solver.check_lmm_cond
    data.use_petsc = paramtrs.use_petsc
    data.tv_check = claw.solver.tv_check
    data.lim_type = claw.solver.lim_type
    if data.lim_type == 1:
        data.limiters = claw.solver.limiters
    else:
        data.limiters = 0

    return data


def plot_stepsize_cfl(problem,methods,solution_data,saveplot=True):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig2, (ax2, ax3) = plt.subplots(1,2)
    axright = ax.twinx()
    ax3right = ax3.twinx()

    marks = ['b','r','k','m']

    for method,ls in zip(methods,marks):
        data = solution_data[method]

        status = data.status
        t = status['time'][:-1] # neglect the final time since time-step is usually small to fit tfinal
        dt = np.asarray(status['dthistory'][:-1]) # making list a numpy array
        dx = max(data.delta[:])
        maxwavespeed = np.asarray(status['maxwavespeed'][:-1])
        cfl_number = dt/dx*maxwavespeed
        k = int(method[-2])
        num_steps_fixed_dt = np.ceil((t[-1]- t[0]-sum(dt[:k-1]))/min(dt[k-1:]))

        print '\nProblem: {0}, method: {1}, steps with vss / steps with fss: {2:d}/{3:d} = {4:.4f}' \
            .format(problem,method,status['totalnumsteps'],int(num_steps_fixed_dt), \
            (status['totalnumsteps']-k)/num_steps_fixed_dt)
        print 'lim_type =',spatial[str(data.lim_type)],', limiter =',limiters[str(data.limiters)], \
                ', rejected steps =',status['totalnumsteps'] - status['numsteps'], \
                ', check_lmm_cond =',data.check_lmm_cond,', use_petsc =',data.use_petsc

        # plot stepsize and cfl against time
        ax, ymin, ymax = plot(method,ax,t,dt/dx,ls)
        axright, ymin_raxis, ymax_raxis = plot(method,axright,t,cfl_number,ls,raxis=True)

        # close up plot of stepsize against time
        ax2,_,_ = plot(method,ax2,t,dt/dx,ls,closeup=True)
        ax3right,_,_ = plot(method,ax3right,t,cfl_number,ls,raxis=True,closeup=True)

    print '\n'

    # formating plots
    stepsizeoverdx = r'$\frac{h_n}{\Delta x}$'
    cfl = r'$\nu_n$'
    ylim = [ymin,ymax]
    xlim = [0.0,t[35]]
    yrightlim = [ymin_raxis,ymax_raxis]

    plot_formatting(methods[:],ax,stepsizeoverdx,ylim,axright=axright,ylabelright=cfl, \
        yrightlim=yrightlim)

    plot_formatting(methods[:],ax2,stepsizeoverdx,ylim,nbins=4,xlim=xlim,lgd=False)
    yrightlim = [0.0,1.05*ax3right.get_ylim()[1]]
    plot_formatting(methods[:],ax3,'',None,axright=ax3right,ylabelright=cfl,yrightlim=yrightlim, \
        nbins=4,xlim=xlim)

    if saveplot:
        file_name = 'figures/'+problem+'.pdf'
        file_name2 = 'figures/'+problem+'_zoom.pdf'
        fig.savefig(file_name,bbox_inches='tight')
        fig2.savefig(file_name2,bbox_inches='tight')
        plt.close("all")
    else:
        plt.show()

def plot(method,ax,x,y,ls,raxis=False,closeup=False):
    if raxis == True:
        line = '--'+ls
        if "SSPLMM" in method:
            ymin = 0.0
            ymax = 0.35
        else:
            ymin = 0.7*min(y)
            ymax = 1.05*max(y)
    else:
        line = ls
        ymin = 0.0
        ymax = 1.05*ax.get_ylim()[1]

    ax.plot(x,y,line,linewidth=2)

    if closeup == True and "SSPLMM" in method:
        k = int(method[-2])
        ax.plot(x[:k-1],y[:k-1],'Dk',markersize=6,markeredgewidth=2,fillstyle='none', \
            label="_nolegend_")
        ax.plot(x[k-1:],y[k-1:],'o'+ls,markersize=6,markeredgewidth=1.5,fillstyle='none', \
            label="_nolegend_")

    return ax, ymin, ymax


def plot_formatting(methods,ax,ylabel,ylim,axright=None,ylabelright=None,yrightlim=None, \
    nbins=6,xlim=None,lgd=True):
    ax.set_xlabel('$t$',fontsize=26)
    ax.set_ylabel(ylabel,labelpad=20,fontsize=32,rotation=0,position=(0,.4))
    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x',pad=8)
    ax.locator_params(axis='x', nbins=nbins)
    ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.axes.get_yaxis().set_visible(False)
    if axright is not None:
        axright.set_ylabel(ylabelright,labelpad=25,fontsize=26,rotation=0,position=(0,.5))
        axright.tick_params(labelsize=16)
        axright.set_ylim(yrightlim)
    #ax.yaxis.get_major_ticks()[0].set_visible(False)

    if lgd == True:
        for i,method in enumerate(methods):
            if 'SSPLMM' in method:
                methods[i] = 'SSPMSV' + methods[i][-2:]
        if ylim is None:
           axright.legend(methods,loc='best',fontsize=20)
        else:
           ax.legend(methods,loc='best',fontsize=20)
    else:
        ax.legend('',frameon=False)


def convergence_advection(methods=['SSPLMMk2','SSPLMMk3'],cfl_nums=None,lim_type=2,limiter=4,norm=2, \
        grid_level=[7,11],paramtrs=None):
    r"""
    Convergence test for advection in 1D.
    =====================================
    methods:    'SSPLMM32', 'SSPLMM43', 'SSP33', 'SSP104'
    cfl_nums:   A dictionary containing the desired and maximum CFL number for each method;
                If set to None then the default cfl_max/cfl_desired as in Pyclaw is used.
    lim_type:   1: TVD reconstruction, 2: WENO reconstruction
    limiter:    Limiter for TVD reconstruction (default is MC limiter; see Pyclaw for more options)
    norm:       1: L1 norm, 2: L2 norm, 3: max norm
    grid_level: Exponents for smallest and largest number of grid points. Default is [7,11], i.e.
                grid points used are 2^7, 2^8, ..., 2^11.
    paramtrs:   Setting of parameters for a given problem (see the `Parameters` class
                in the specific problem script for a full discription of parameters).
                If not specified then the default parameters for each problem are used.
                The user can specified parameters by calling the `set_parameters` function first.
    """

    import numpy as np
    from clawpack import pyclaw
    from importlib import import_module
    prb = import_module('advection_1d')

    grid_pts = [2**x for x in xrange(grid_level[0],grid_level[1]+1)]
    m = len(grid_pts)
    n = len(methods)

    err = np.zeros([m,n])
    dx = np.zeros(m)

    if paramtrs is None:
        # set default parameters
        paramtrs = prb.Parameters()

    if paramtrs.use_petsc == True:
        file_format = 'petsc'
    else:
        file_format = 'ascii'

    ref_sol = lambda x: paramtrs.set_initial_cond(x,paramtrs.IC)

    for i,N in enumerate(grid_pts):
        for j,method in enumerate(methods):
            data = run('advection_1d',N=N,method=method,cfl_nums=cfl_nums[method],\
                    lim_type=lim_type[method],limiter=limiter,paramtrs=paramtrs,iplot=False)

            if data is not None:
                print 'lim_type =',data.lim_type,', limiter =',data.limiters, \
                    ', steps = ', data.status['numsteps'],', rejected steps =',\
                    data.status['totalnumsteps'] - data.status['numsteps'],\
                    ', check_lmm_cond =',data.check_lmm_cond,', tv_check =',data.tv_check,\
                    ', use_petsc =',data.use_petsc,'\n'

                dx[i] = data.delta[0]
                qfinal = pyclaw.Solution(paramtrs.num_output_times,outdir='_output',file_format=file_format)
                x = qfinal.domain.grid.x.centers

                err[i,j] = compute_err(norm,ref_sol(x),qfinal.q[0,:],dx[i])

    return err, dx


def compute_err(norm,ref_sol,test_sol,dx):
    import numpy as np

    if norm == 'max':
        err  = np.max(np.abs(ref_sol - test_sol))
    else:
        err = dx**(1./norm)*np.linalg.norm(ref_sol - test_sol,norm)

    return err


def plot_err(data,saveplot=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm

    methods = data['methods']
    err = data['error']
    dx = data['dx']
    norm = data['norm']

    n = len(methods)
    
    for i,method in enumerate(methods):
        if 'SSPLMM' in method:
            methods[i] = 'SSPMSV' + methods[i][-2:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if n > 4:
        marks = iter(cm.gist_rainbow(np.linspace(0,1,n)))
        ls = np.zeros([n,4])
    else:
        marks = ['b','r','k','m']
        ls = marks
    mid_index = err.shape[0]/2
    for j in xrange(n):
        ls[j] = next(marks) if n > 4 else marks[j]
        plt.loglog(dx,err[:,j],'s',c=ls[j],linewidth=2)
    for j in xrange(n):
        if 'SSPMSV' in methods[j]:
            p = int(methods[j][-1])
            ax.loglog(dx,dx**p*err[mid_index,j]/dx[mid_index]**p,'--',c=ls[j],linewidth=2)
        elif 'SSP33' in methods:
            ax.loglog(dx,dx**3*err[mid_index,j]/dx[mid_index]**3,'--',c=ls[j],linewidth=2)
        elif 'SSP104' in methods:
            ax.loglog(dx,dx**4*err[mid_index,j]/dx[mid_index]**4,'--',c=ls[j],linewidth=2)
    ax.legend(methods,loc='best',fontsize=16)    
    ax.set_xlabel('$\Delta x$',fontsize=18)
    ax.set_ylabel('$\||error\||_'+str(norm)+'$',fontsize=18,rotation=0,position=(0,.5))
    ax.yaxis.labelpad = 35
    ax.tick_params(labelsize=14)

    if saveplot:
        plt.savefig('figures/conv_advection_1d.pdf',bbox_inches='tight')
    else:
        plt.show()

    plt.close('all')


def set_parameters(problem,xleft,xright,IC,tfinal,num_output_times):
    r"""
    Function to set parameters for a given problem.
    ==================================================
    problem:            'advection_1d, 'burgers_1d', 'woodward_colella_1d',
                        'radial_shallow_water_2d, 'shock_bubble_interaction_2d'
    xleft:              left boundary
    right:              right boundary
    IC:                 type of initial condition
                        advection: 'sinusoidal', 'gaussian'
                        burgers: 'sinusoidal', 'Heaviside', 'square wave', 'triangle'
    tfinal:             final time
    num_output_times:   number of output frames
    """

    from importlib import import_module
    prb = import_module(problem)
    
    paramtrs = prb.Parameters()
    paramtrs.xleft = xleft
    paramtrs.xright = xright
    paramtrs.IC = IC
    paramtrs.tfinal = tfinal
    paramtrs.num_output_times = num_output_times
        
    return paramtrs


def default_parameters(problem):
    from importlib import import_module
    
    prb = import_module(problem)
    paramtrs = prb.Parameters()
    paramtrs.num_output_times = 10
    paramtrs.max_steps = 10000
    paramtrs.tv_check = False
    paramtrs.check_lmm_cond = True
    paramtrs.use_petsc = False

    if problem == 'advection_1d':
        paramtrs.IC = 'sinusoidal'
        paramtrs.xleft = 0.0
        paramtrs.xright = 1.0
        paramtrs.tfinal = 1.0
        paramtrs.N = 256
    elif problem == 'burgers_1d':
        paramtrs.xleft = 0.0
        paramtrs.xright = 1.0
        paramtrs.IC = 'sinusoidal'
        paramtrs.tfinal = 0.8
        paramtrs.N = 256
    elif problem == 'woodward_colella_1d':
        paramtrs.xleft = 0.0
        paramtrs.xright = 1.0
        paramtrs.tfinal = 0.04
        paramtrs.N = 512
    elif problem == 'radial_shallow_water_2d':
        paramtrs.xlower = -2.5
        paramtrs.xupper = 2.5
        paramtrs.ylower = -2.5
        paramtrs.yupper = 2.5
        paramtrs.tfinal = 2.5
        paramtrs.N = [250,250]
    elif problem == 'shock_bubble_interaction_2d':
        paramtrs.xlower = 0.0
        paramtrs.xupper = 2.0
        paramtrs.ylower = 0.0
        paramtrs.yupper = 0.5
        paramtrs.tfinal = 0.6
        paramtrs.N = [640,160]

    return paramtrs



#=========================================================
# Functions to reproduce data used in paper
#=========================================================
def tab1_createdata(norm=1,grid_level=[7,11],use_petsc=False):
    r"""
    Converence test for advection 1D problem with sinusoidal initial data and variable-in-time velocity.
    Error between exact and numerical solution is measured in a given norm.
    """
    import pickle

    paramtrs = default_parameters('advection_1d')
    paramtrs.num_output_times = 1
    paramtrs.max_steps = 200000
    paramtrs.tfinal = 5.0
    paramtrs.tv_check = True
    paramtrs.use_petsc = use_petsc

    methods=['SSPLMM32','SSPLMM42','SSPLMM43','SSPLMM53']
 
    err, dx = convergence_advection(methods=methods,cfl_nums=cfl_nums,lim_type=lim_type,limiter=4,norm=norm,\
                grid_level=grid_level,paramtrs=paramtrs)

    data = {}
    data['methods'] = methods
    data['error'] = err
    data['dx'] = dx
    data['norm'] = norm
    output = open('data/convergence_advection.pkl', "wb")
    pickle.dump(data, output, -1)
    output.close()


def fig2_createdata():
    r"""
    Solve and save solution data for 1D Burgers' and Woodward-Colella blast problem.
    """
    import pickle

    problems=['burgers_1d','woodward_colella_1d']
    methods=['SSPLMM32','SSPLMM43']
    cfl_nums = {'SSPLMM32': [0.24,0.25],'SSPLMM43': [0.14,1/6.]}
    limiter = 4

    for problem in problems:
        paramtrs = default_parameters(problem)
        paramtrs.num_output_times = 1
        paramtrs.max_steps = 200000
        if problem == 'burgers_1d':
            paramtrs.tv_check = True
            lim_type = {'SSPLMM32': 1,'SSPLMM43': 2}
        if problem == 'woodward_colella_1d':
            lim_type = {'SSPLMM32': 1,'SSPLMM43': 1}

        solution_data = {}
        for method in methods:
            solution_data[method] = run(problem,N=paramtrs.N,method=method,cfl_nums=cfl_nums[method],\
                    lim_type=lim_type[method],limiter=limiter,paramtrs=paramtrs,iplot=False)

        output = open('data/'+problem+'.pkl', "wb")
        pickle.dump(solution_data, output, -1)
        output.close()


def fig3_createdata(problem='radial_shallow_water_2d',use_petsc=False):
    r"""
    Solve and save solution data for 2D Radial shallow-water problem.
    """
    from clawpack import pyclaw
    import pickle
    if use_petsc == True:
        try:
            import pkgutil
            pkgutil.find_loader("clawpack.petclaw")
            file_format = 'petsc'
        except ImportError:
            use_petsc = False
            file_format = 'ascii'
            raise 'Unable to import petclaw, is PETSC/petsc4py installed?'
    else:
        file_format = 'ascii'

    methods=['SSPLMM32','SSPLMM43']
    cfl_nums = {'SSPLMM32': [0.24,0.25],'SSPLMM43': [0.15,1/6.]}
    lim_type = {'SSPLMM32': 1,'SSPLMM43': 2}
    limiter = 4

    paramtrs = default_parameters(problem)
    paramtrs.num_output_times = 1
    paramtrs.max_steps = 200000
    paramtrs.use_petsc = use_petsc

    for method in methods:
        solution_data = {}
        solution_data[method] = run(problem,N=paramtrs.N,method=method,cfl_nums=cfl_nums[method],\
                lim_type=lim_type[method],limiter=limiter,paramtrs=paramtrs,iplot=False)
        
        qinitial = pyclaw.Solution(0,outdir='_output',file_format=file_format)
        qfinal = pyclaw.Solution(paramtrs.num_output_times,outdir='_output',file_format=file_format)
        solution_data[method].qinitial = qinitial.q
        solution_data[method].qfinal = qfinal.q

        output = open('data/'+problem+'_'+method+'.pkl', "wb")
        pickle.dump(solution_data, output, -1)
        output.close()


#=========================================================
# Functions to reproduce figures and tables from paper
#=========================================================
def fig_tab1(plot=True):
    r"""
    Create plot for convergence test and output error/convergence order.
    """
    import numpy as np
    import pickle
    import os.path

    filename = 'data/convergence_advection.pkl'
    if not os.path.isfile(filename):
        tab1_createdata()

    data = pickle.load(open(filename, "rb"))

    # error plot
    if plot == True:
        plot_err(data)

    order = np.zeros([data['error'] .shape[0]-1,data['error'] .shape[1]])
    for i in xrange(data['error'] .shape[0]-1):
        order[i,:] = np.log2(data['error'] [i,:]/data['error'] [i+1,:])

    print data['error']
    print order


def plot_fig2():
    r"""
    Plot step size and CFL number against time for 1D problems.
    """
    import pickle
    import os.path

    problems=['burgers_1d','woodward_colella_1d']
    methods=['SSPLMM32','SSPLMM43']

    for problem in problems:
        filename = 'data/'+problem+'.pkl'

        if not os.path.isfile(filename):
            fig2_createdata()

        solution_data = pickle.load(open(filename, "rb"))
        plot_stepsize_cfl(problem,methods,solution_data,saveplot=True)


def plot_fig3():
    r"""
    Plot step size and CFL number against time for 2D problems.
    """
    import pickle
    import os.path

    problems=['radial_shallow_water_2d','shock_bubble_interaction_2d']
    methods=['SSPLMM32','SSPLMM43']

    for problem in problems:
        solution_data = {}
        for method in methods:
            filename = 'data/'+problem+'_'+method+'.pkl'
            if not os.path.isfile(filename):
                if problem == ['radial_shallow_water_2d']:
                    fig3_createdata()
                else:
                    raise Exception('Need to create solution data for shock_bubble_interaction_2d problem.')

            solution_data.update(pickle.load(open(filename, "rb")))
        plot_stepsize_cfl(problem,methods,solution_data,saveplot=True)
