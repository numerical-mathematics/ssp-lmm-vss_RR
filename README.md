Reproducibility repository for paper on SSPLMM with variable step size.
=======================================================================

Code to reproduce numerical tests from "Strong stability preserving linear multistep methods with variable step size".

To run these scripts, you should have **Clawpack 5.3.**.

To install Clawpack, follow the setup instructions in [Clawpack 5](http://clawpack.github.io/doc/installing.html#).

All tested problems are modifications of Clawpack's examples.

Files:

- **ssp_lmm_vss.py**: Main file containing scripts that produce the data and figures of the paper.
- **overridden_fun.py**: Pyclaw's overridden classes and functions.
- **advection_1d.py**: solves advection 1D equation.
- **burgers_1d.py**: solves Burgers 1D equation.
- **woodward_colella_1d.py**: solves Woodward-Colella blast wave problem.
- **radial_shallow_water_2d.py**: Solves 2D shallow water equations.
- **shock_bubble_interaction_2d.py**: solves Euler equations of compressible fluid dynamics in 2D.

Subdirectories:

- **data**: Solution data for convergence test (table 1) and 1D/2D problems (Figures 1 & 2).
- **figures**: Figures 1 & 2 (also figure of convergence test)
(Figure names correspond to the arXiv version of the paper.)

The scripts and data in this "data" directory can be used to reproduce all figures from the paper.
Detailed instructions of the main function are included in *ssp_lmm_vss.py*.

For example, to reproduce solution data and Figure 1, in an ipython terminal run:
    import ssp_lmm_vss
    ssp_lmm_vss.fig2_createdata()
    ssp_lmm_vss.plot_fig2()

This saves the solution data in a binary file (.pkl) in the data/ subdirectory and creates Figure 1 in the figures/ directory.

Provided the data exist, you can also just plot the figure:

    import ssp_lmm_vss
    ssp_lmm_vss.plot_fig2()

By default, this uses data stored in the subdirectory of data/.