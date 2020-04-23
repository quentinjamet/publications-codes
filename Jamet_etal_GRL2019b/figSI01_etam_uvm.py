import numpy as np
import MITgcmutils as mit
import matplotlib.pyplot as plt


#-- directories and config --
dir_in = '/tank/chaocean/bill/RUNS/ORAR/memb23/19701995/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/'

#-- grid --
xC = mit.rdmds(dir_grd + 'XC')
xC = xC-360
ny, nx = xC.shape
yC = mit.rdmds(dir_grd + 'YC')
hC = mit.rdmds(dir_grd + 'hFacC')
depth = mit.rdmds(dir_grd + 'Depth')
msk = hC[0, :, :]*1.0
msk[ np.where(msk[:] == 0) ] = np.nan
nr = 46

#-- SSH -- 
fileN = 'Etamean.data'
f = open(dir_in+fileN,'r')
sshm = np.fromfile(f,'>f4').reshape([ny, nx]) * msk
f.close()

#-- velocities
#- zonal -
fileN = 'Umean.data'
f = open(dir_in+fileN,'r')
uum = np.fromfile(f,'>f4').reshape([nr, ny, nx])
f.close()
#- interpolate a T-pts -
um_t = np.zeros([nr, ny, nx])
um_t[:, 0:-1, 0:-1] = 0.5 * (uum[:, 0:-1, 0:-1] + uum[:, 0:-1, 1:]) *\
        hC[:, 0:-1, 0:-1]
um_t[:, 0:-1, -1] = 0.5 * (uum[:, 0:-1, -1] + uum[:, 0:-1, 0]) *\
        hC[:, 0:-1, -1]

#- meridional
fileN = 'Vmean.data'
f = open(dir_in+fileN,'r')
vvm = np.fromfile(f,'>f4').reshape([nr, ny, nx])
f.close()
#- interpolate a T-pts -
vm_t = np.zeros([nr, ny, nx])
vm_t[:, 0:-1, :] = 0.5 * (vvm[:, 0:-1, :] + vvm[:, 1:, :]) *\
        hC[:, 0:-1, :]


uvm = np.sqrt( um_t**2 + vm_t**2 )[0, :, :] * msk


#------------------
# reshape
#------------------
nx_resh = 301

#- coordinate -
xC_resh = np.zeros([ny, nx+nx_resh])
xC_resh[:, 0:nx] = xC
dxC = xC[:,-1]-xC[:,-2]
xC_resh[:, nx:nx+nx_resh] = np.tile(xC[:,-1][:, np.newaxis], (1, nx_resh)) + \
    np.cumsum( np.tile(dxC[:,np.newaxis], (1, nx_resh)),1 )
yC_resh = np.zeros([ny, nx+nx_resh])
yC_resh = np.tile(yC[:, 0][:, np.newaxis], (1, nx+nx_resh))

#-- reshape in regular fashion --
f = open('/tank/chaocean/scripts/mask_cut_gulf_lines.bin','r')
msk_chao12 = np.fromfile(f,'>f4').reshape([ny, nx_resh])
f.close()
#- to move -
ji_tomove = np.where( msk_chao12 == 1 )
#- stay here, stay near -
ji_nomove = np.where( msk_chao12 == 0 )
#- assemble -
def var_resh(var_in):
 var_tomove = np.zeros([ny, nx_resh])
 var_tomove[ji_tomove[0], ji_tomove[1]] = var_in[ji_tomove[0], ji_tomove[1]]
 var_nomove = np.zeros([ny, nx_resh])
 var_nomove[ji_nomove[0], ji_nomove[1]] = var_in[ji_nomove[0], ji_nomove[1]]
 var_out = np.zeros([ny, nx+nx_resh])
 var_out[:, nx_resh:nx] = var_in[:, nx_resh:nx]
 var_out[:, 0:nx_resh] = var_nomove
 var_out[:, nx:nx+nx_resh] = var_tomove
 return var_out


ssh_resh = var_resh(sshm)
ssh_resh[np.where(ssh_resh[:] == 0)] = np.nan
dd_resh = var_resh(depth)
uvm_resh = var_resh(uvm)
uvm_resh[np.where(uvm_resh[:] == 0)] = np.nan



#------------------
#	plot 
#------------------

fig1, axs1 = plt.subplots(1, 2)
fig1.set_size_inches(15,6)
#-- ssh --
llev1 = np.arange(-1.5, 1.6, 0.1)
cs1 = axs1[0].contourf(xC_resh, yC_resh, ssh_resh, levels=llev1, cmap="RdBu_r")
axs1[0].contour(xC_resh, yC_resh, ssh_resh, levels=0, colors='k')
axs1[0].set_facecolor((0.5, 0.5, 0.5))
axs1[0].set_title('Sea Surface Height [m]', fontsize='xx-large')
axs1[0].set_ylabel('Latitude', fontsize='x-large')
axs1[0].set_xlabel('Longitude', fontsize='x-large')
fig1.colorbar(cs1, ax=axs1[0], orientation='vertical', fraction=.05)
#-- surf. currents --
llev2 = np.arange(0, 2, 0.1)
cs2 = axs1[1].contourf(xC_resh, yC_resh, uvm_resh, levels=llev2, cmap="hot_r")
axs1[1].set_facecolor((0.5, 0.5, 0.5))
axs1[1].set_title('Surface currents [m s$^{-1}$]', fontsize='xx-large')
axs1[1].set_xlabel('Longitude', fontsize='x-large')
fig1.colorbar(cs2, ax=axs1[1], orientation='vertical', fraction=.05)

plt.show()

figN1 = '1970_1995_mean_ssh_surf_curr_orar'
fig1.savefig(dir_fig+figN1+'.pdf', bbox_inches='tight')
fig1.savefig(dir_fig+figN1+'.png', bbox_inches='tight')
plt.close(fig1)



