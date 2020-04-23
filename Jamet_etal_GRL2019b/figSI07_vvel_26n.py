import numpy as np
import MITgcmutils as mit
import matplotlib.pyplot as plt


#-- directories and config --
dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/orar/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/'

#-- grid --
xC = mit.rdmds(dir_grd + 'XC')
xC = xC-360
ny, nx = xC.shape
yG = mit.rdmds(dir_grd + 'YG')
#- 26n region -
jj26n = 557
xC_26n = xC[jj26n, :]
rC = mit.rdmds(dir_grd + 'RC')
nr = len(rC)
hS = mit.rdmds(dir_grd + 'hFacS')
#- cell face -
dxG = mit.rdmds(dir_grd + 'DXG')
drF = mit.rdmds(dir_grd + 'DRF')
hfacS = np.squeeze(0.5*(dxG[jj26n-1, :]+dxG[jj26n, :]) * drF) * hS[:, jj26n, :]

#-- runs parameters --
ndump = 73
spy = 86400*365
yyr = np.arange(1963, 2013)
nyr = len(yyr)
ttime = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nmem = 12
#- time mean and std on 1970-1995 -
ttt = np.where( (ttime > 1970) & (ttime < 1996) )[0]

#--------------------
# load data
#--------------------
fileN = 'VVEL_zonal_sec_26n_orar.bin'
f = open(dir_in+fileN,'r')
vv26n = np.fromfile(f,'>f4').reshape([nmem, nt, nr, nx])
f.close()


#-------------------
#	mean/std
# on the ensemble mean for the period  1970-1995
#-------------------
vvm = np.mean(np.mean(vv26n[:, :, :, :], 0), 0)
vvstd = np.std(np.mean(vv26n[:, :, :, :], 0), 0)
vvm[np.where(vvm[:] == 0)] = np.nan
vvstd[np.where(vvstd[:] == 0)] = np.nan

#-- PLOT --

#- mean -
fig1, axs1 = plt.subplots(1, 2)
fig1.set_size_inches(15,6)
#- Florida current -
ii_fc = np.where( (xC_26n > -80.2) & (xC_26n < -78.6) )[0]
cs1 = axs1[0].contourf(xC_26n[ii_fc], np.squeeze(rC), vvm[:, ii_fc]*100, levels=np.arange(0, 180, 10), cmap="hot_r")
axs1[0].set_facecolor((0.5, 0.5, 0.5))
fig1.colorbar(cs1, ax=axs1[0], orientation='vertical', fraction=.05)
axs1[0].set_ylim([-800, 0])
axs1[0].set_ylabel('Depth [m]', fontsize='x-large')
axs1[0].set_xlabel('Longitude', fontsize='x-large')
axs1[0].set_title('Florida Current [cm s$^{-1}$]', fontsize='xx-large')
#- Interior -
ii_int = np.where( (xC_26n > -77.2) & (xC_26n < -73) )[0]
cs2 = axs1[1].contourf(xC_26n[ii_int], np.squeeze(rC)/1000, vvm[:, ii_int]*100, levels=np.arange(-30, 35, 5), cmap="RdBu_r")
axs1[1].contour(xC_26n[ii_int], np.squeeze(rC)/1000, vvm[:, ii_int]*100, [0], colors='k')
axs1[1].set_facecolor((0.5, 0.5, 0.5))
axs1[1].set_ylim([-5, 0])
axs1[1].set_ylabel('Depth [km]', fontsize='x-large')
axs1[1].set_xlabel('Longitude', fontsize='x-large')
axs1[1].set_title('Western Boundary Current [cm s$^{-1}$]', fontsize='xx-large')
fig1.colorbar(cs2, ax=axs1[1], orientation='vertical', fraction=.05)

#plt.show()

figN1 = '1963_2012_mean_florida_current_wbc_orar'
fig1.savefig(dir_fig+figN1+'.pdf', bbox_inches='tight')
fig1.savefig(dir_fig+figN1+'.png', bbox_inches='tight')
plt.close(fig1)


#----------------------
# Transport 
#----------------------

#-- Florida Current --
transp_fc = np.nansum(np.nansum( vvm[:, ii_fc] * hfacS[:, ii_fc], 1), 0) * 1e-6


#-- western boundary current --
kk_up = np.where(rC>-600)[0]
kk_dw = np.where(rC<-600)[0]
vvm_up = vvm[kk_up, :] 
hfacS_up = hfacS[kk_up, :]
vvm_dw = vvm[kk_dw, :] 
hfacS_dw = hfacS[kk_dw, :]
transp_wbt_up = np.nancumsum(np.nansum( vvm_up[:, ii_int] * hfacS_up[:, ii_int], 0 ), 0) * 1e-6
transp_wbt_dw = np.nancumsum(np.nansum( vvm_dw[:, ii_int] * hfacS_dw[:, ii_int], 0 ), 0) * 1e-6


plt.plot(xC_26n[ii_int], transp_wbt_up, 'b')
plt.plot(xC_26n[ii_int], transp_wbt_dw, 'k')
plt.legend(('upper', 'lower'))

plt.grid()
plt.show()


