#!/usr/bin/env python
# encoding: utf-8
r"""
Allows run of shallow-water problem with PETSC to generate solution data.
"""

if __name__=="__main__":
    import ssp_lmm_vss
    ssp_lmm_vss.fig3_createdata(problem='radial_shallow_water_2d',use_petsc=True)
