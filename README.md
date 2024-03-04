## Code for "Natural variability can mask forced permafrost response to stratospheric aerosol injection in the ARISE-SAI-1.5 simulations"
### by A.L. Morrison, E.A. Barnes, and J.W. Hurrell
Contact: [Dr. Ariel Morrison](mailto:ariel.morrison@colostate.edu)

This code uses output data from the Assessing Responses and Impacts of Solar climate intervention on the Earth system with stratospheric aerosol injection (ARISE-SAI) 1.5 simulations (Richter et al., 2022) to perform the analysis and make figures for "Natural variability can mask forced permafrost response to stratospheric aerosol injection in the ARISE-SAI-1.5 simulations." The unprocessed ARISE-SAI-1.5 data are [here](https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.ARISE-SAI-1.5.lnd.proc.monthly_ave.html) and the unprocessed SSP2-4.5 data are [here](https://www.earthsystemgrid.org/dataset/ucar.cgd.cesm2.waccm6.ssp245.lnd.proc.monthly_ave.html). The necessary monthly mean variables for running this code are:
- ALT
- ALTMAX
- TSOI

```gridareaNH.nc``` and ```peatareaGlobal.nc``` are files for calculating the total area covered by permafrost and the peat area covered by permafrost, respectively.  
