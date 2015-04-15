#!/usr/bin/env python
# encoding: utf-8

r"""
This script overrides the initialization and evolve_to_time() function of the chosen solver type
already existing in pyclaw.
It adds more functionality to the solver, i.e. more data in solver.setup, TVD check, 
maximum wave speed calculation.
"""
import numpy as np
from clawpack import pyclaw

base_solver = pyclaw.solver

def default_compute_gauge_values(q,aux):
    r"""By default, record values of q at gauges.
    """
    return q

def set_solver(solver_type,riemann_solver=None,claw_package=None):

    class Solver(solver_type):

        def __init__(self,riemann_solver=None,claw_package=None):
            r"""
            Override the initialization of a Solver object

            See :class:`Solver` for full documentation
            """
            
            # Default initialization of Pyclaw Solver object
            super(Solver,self).__init__(riemann_solver,claw_package)
            self._isinitialized = False
        
            self.tv_check = False
            self.use_petsc = False

            # Status Dictionary
            self.status = { 'cflmax': -np.inf,
                            'dtmin': np.inf,
                            'dtmax': -np.inf,
                            'numsteps': 0,
                            'totalnumsteps': 0,
                            'dthistory': [],
                            'time': [],
                            'maxwavespeed': []}

            self._isinitialized = True



    # ========== Override evolution routine ======================================
        def evolve_to_time(self,solution,tend=None):
            r"""
            Evolve solution from solution.t to tend.  If tend is not specified,
            take a single step.
            
            This method contains the machinery to evolve the solution object in
            ``solution`` to the requested end time tend if given, or one 
            step if not.          

            :Input:
             - *solution* - (:class:`Solution`) Solution to be evolved
             - *tend* - (float) The end time to evolve to, if not provided then 
               the method will take a single time step.
                
            :Output:
             - (dict) - Returns the status dictionary of the solver
            """
            if not self._is_set_up:
                self.setup(solution)
            
            if tend == None:
                take_one_step = True
            else:
                take_one_step = False
                
            # Parameters for time-stepping
            tstart = solution.t

            num_steps = 0

            # Setup for the run
            if not self.dt_variable:
                if take_one_step:
                    self.max_steps = 1
                else:
                    self.max_steps = int((tend - tstart + 1e-10) / self.dt)
                    if abs(self.max_steps*self.dt - (tend - tstart)) > 1e-5 * (tend-tstart):
                        raise Exception('dt does not divide (tend-tstart) and dt is fixed!')
            if self.dt_variable == 1 and self.cfl_desired > self.cfl_max:
                raise Exception('Variable time-stepping and desired CFL > maximum CFL')
            if tend <= tstart and not take_one_step:
                self.logger.info("Already at or beyond end time: no evolution required.")
                self.max_steps = 0

            if self.status['numsteps'] == 0:
                delta = solution.state.grid.delta
                if self.num_eqn == 2 and delta[0] != delta[1]:
                    compute_maxwavespeed = True
                    # create a PESTC vec to store maximum speed if running in parallel
                    if self.use_petsc == True:
                        from petsc4py import PETSc
                        wave_speed = PETSc.Vec().createWithArray([0])
                else:
                    compute_maxwavespeed = False

            # Main time-stepping loop
            for n in xrange(self.max_steps):
                
                state = solution.state
                
                # Keep a backup in case we need to retake a time step
                if self.dt_variable:
                    q_backup = state.q.copy('F')
                    told = solution.t

                if self.tv_check == True and self.num_eqn == 1 and self.use_petsc == False:
                    if 'SSPLMM' in self.time_integrator:
                        prev_tvd_norm = \
                                max([self._tv_norm(self._registers[i].q[0,:]) for i in range(self.lmm_steps)])
                    else:
                        prev_tvd_norm = self._tv_norm(q_backup[0,:])

                if self.before_step is not None:
                    self.before_step(self,solution.states[0])

                # Note that the solver may alter dt during the step() routine
                self.step(solution,take_one_step,tstart,tend)
                self.status['totalnumsteps'] += 1

                # Check to make sure that the Courant number was not too large
                cfl = self.cfl.get_cached_max()
                self.accept_step = self.accept_reject_step(state)
                if self.accept_step:
                    # Accept this step
                    self.status['cflmax'] = max(cfl, self.status['cflmax'])
                    if self.dt_variable==True:
                        solution.t += self.dt
                    else:
                        #Avoid roundoff error if dt_variable=False:
                        solution.t = tstart+(n+1)*self.dt

                    # Verbose messaging
                    self.logger.debug("Step %i  CFL = %f   dt = %f   t = %f"
                        % (n,cfl,self.dt,solution.t))

                    # Save status data if step is accepted
                    self.status['dthistory'].append(self.dt)
                    self.status['time'].append(solution.t)
                    
                    # compute max wave speed for different riemann solvers
                    if compute_maxwavespeed == True:
                        rs  = riemann_solver.__name__.split('.')[-1]
                        if rs == 'shallow_roe_with_efix_2D':
                            q = state.q
                            g = state.problem_data['grav']
                            c = np.sqrt(g*q[0,:,:])
                            u = q[1,:,:]/q[0,:,:]
                            v = q[2,:,:]/q[0,:,:]
                            s = np.empty((4,u.shape[0],u.shape[1]))
                            s[0,:,:] = u - c
                            s[1,:,:] = u + c
                            s[2,:,:] = v - c
                            s[3,:,:] = v + c
                            maxwavespeed = np.max(abs(s.flatten()))
                        elif rs == 'euler_5wave_2D':
                            q = state.q
                            gamma = state.problem_data['gamma']
                            u = q[1,:,:]/q[0,:,:]
                            v = q[2,:,:]/q[0,:,:]
                            p = (gamma - 1.)* (q[3,:,:] - \
                                    0.5 * (q[1,:,:]**2 / q[0,:,:] + q[2,:,:]**2 / q[0,:,:]))
                            c = np.sqrt(gamma*p/q[0,:,:])
                            s = np.empty((4,u.shape[0],u.shape[1]))
                            s[0,:,:] = u - c
                            s[1,:,:] = u + c
                            s[2,:,:] = v - c
                            s[3,:,:] = v + c
                            maxwavespeed = np.max(abs(s.flatten()))
                        if self.use_petsc == True:
                            wave_speed.array = maxwavespeed
                            self.status['maxwavespeed'].append(wave_speed.max()[1])
                        else:
                            self.status['maxwavespeed'].append(maxwavespeed)
                    else:
                        maxwavespeed = cfl*state.grid.delta[0]/self.dt
                        self.status['maxwavespeed'].append(maxwavespeed)
                    
                    # Check tvd-norm for 1D problems
                    if self.tv_check == True and self.num_eqn == 1 and self.use_petsc == False:
                        q_current = state.q.copy('F')
                        
                        if self.lim_type == 2:
                            rs  = riemann_solver.__name__.split('.')[-1]
                            tol = 1.e-3 if rs == 'advection_1D' else 1.e-4
                        else:
                            tol = 1.e-14
                        if self._tv_norm(q_current[0,:]) - prev_tvd_norm > tol:
                            print 'time = ', solution.t
                            print 'step = ', self.status['numsteps']
                            print 'old tv-norm = ', prev_tvd_norm
                            print 'new tv-norm = ', self._tv_norm(q_current[0,:])
                            print 'difference = ', self._tv_norm(q_current[0,:]) - prev_tvd_norm
                            print ''

                    self.write_gauge_values(solution)
                    # Increment number of time steps completed
                    num_steps += 1
                    self.status['numsteps'] += 1

                else:
                    # Reject this step
                    self.logger.debug("Rejecting time step, CFL number too large")
                    if self.dt_variable:
                        state.q = q_backup
                        solution.t = told
                    else:
                        # Give up, we cannot adapt, abort
                        self.status['cflmax'] = \
                            max(cfl, self.status['cflmax'])
                        raise Exception('CFL too large, giving up!')

                # See if we are finished yet
                if solution.t >= tend or take_one_step:
                    break
          
            # End of main time-stepping loop -------------------------------------

            if self.dt_variable and solution.t < tend \
                    and num_steps == self.max_steps:
                raise Exception("Maximum number of timesteps have been taken")

            return self.status


        def _tv_norm(self,q):
            sol = np.append(q,q[0])
            return sum(abs(np.diff(sol)))


    return Solver(riemann_solver,claw_package)
