#!/usr/bin/env python
# encoding: utf-8

r"""
Runs shock-bubble interaction problem with SSPLMMs to generate solution data.
Checks if PETSC is installed in order to run in parallel.
"""

def create_data(problems,methods,cfl_nums,lim_type):
    import pickle
    import ssp_lmm_vss
    try:
        import pkgutil
        pkgutil.find_loader("clawpack.petclaw")
        use_petsc = True
    except ImportError:
        use_petsc = False
        print 'Unable to import petclaw, is PETSC/petsc4py installed?'
        raise

    for problem in problems:
        paramtrs = ssp_lmm_vss.default_parameters(problem)
        paramtrs.num_output_times = 1 
        paramtrs.max_steps = 200000
        paramtrs.use_petsc = use_petsc
        paramtrs.N = [640,160]

        for method in methods:
            data = ssp_lmm_vss.run(problem,N=paramtrs.N,method=method,cfl_nums=cfl_nums[method],\
                    lim_type=lim_type[method],limiter=4,paramtrs=paramtrs,iplot=False)
            if data is not None:
                solution_data = {}
                solution_data[method] = data
                output = open(problem+'_'+method+'.pkl', "wb")
                pickle.dump(solution_data, output, -1)
                output.close()

if __name__=="__main__":
    methods=['SSPLMM32','SSPLMM43']
    methods=['SSPLMM43']
    cfl_nums = {'SSPLMM32': [0.24,0.25],'SSPLMM43': [0.15,1/6.]}
    lim_type = {'SSPLMM32': 1,'SSPLMM43': 2}
    
    problems=['shock_bubble_interaction_2d']
    create_data(problems,methods,cfl_nums,lim_type)
