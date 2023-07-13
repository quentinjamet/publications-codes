import numpy as np
import matplotlib.pyplot as plt
import qgutils as qg
import pickle

#-- run and directories --
simu = 'abs'
if simu == 'abs':
	dir0 = '/home2/datawork/qjamet/qgcm-data/double_gyre_coupled/outdata_12_dt10/'
elif simu == 'rel':
	dir0 = '/home2/datawork/qjamet/qgcm-data/double_gyre_coupled/outdata_12_dt10_tdiff/'
else:
	print("Don't know that simu: %s" % simu)
#
pfiles = [dir0 + 'ocpo.nc']
ffiles = [dir0 + 'ocsst.nc']

#-- Physical parameters --
gp = np.array([0.025, 0.0125])
dh = np.array([350., 750., 2900])
f0 =  9.37456E-05
beta = 1.75360E-11
#toc = np.array([-13.1693, -23.1693, -24.1693])# dt10 -- read in output files directly ...
Delta = 5000      #dxo=5.00000E+03, but also nsko=2;
d_e = 5
nu = 0
nu4 = 2.e9
# derived parameters
N2 = qg.gprime2N2(dh,gp)
bf = d_e*f0/(2*dh[-1])


#-----------------------------------
#		COMPUTE 
#-----------------------------------
lec = qg.lorenz_cycle(pfiles,dh,N2,f0,Delta,bf, nu=nu, nu4=nu4, \
                      forcing_z=ffiles, forcing_b=ffiles, toc=0, nu_in_b=False, average=False, \
                      bc_fac=0.2*0.5/(0.2*0.5 + 1), maps=True, spec_flx=True)


#-----------------------------------
#		SAVE 
#-----------------------------------
dir_out = '/home2/datawork/qjamet/qgcm-data/double_gyre_coupled/data/'
fileN = str("%s/LEC_2d_maps_spectral_fluxes_%s.pkl" % (dir_out, simu))
with open(fileN, 'wb') as f:
    pickle.dump(lec, f)
