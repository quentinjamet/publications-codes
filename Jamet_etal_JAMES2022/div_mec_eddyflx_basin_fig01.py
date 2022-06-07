import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import xarray as xr
import glob
from tkinter import Tcl
from matplotlib.offsetbox import AnchoredText

plt.ion()

#-- directories --
dir_in = '/gpfsscratch/rech/egi/uup63gs/cdftools/'
dir_in2= '/gpfsstore/rech/egi/uup63gs/medwest60/MEDWEST60-GSL19-S/ens01/1h/'
dir_grd = '/gpfsstore/rech/egi/uup63gs/medwest60/mesh/'
dir_fig = '/linkhome/rech/genige01/uup63gs/Figures/energetics/'
dir_out = '/gpfsstore/rech/egi/uup63gs/medwest60/outputs/'

#-- mesh and mask --
msk   = xr.open_dataset(dir_grd + 'mask.nc')
hgr   = xr.open_dataset(dir_grd + 'mesh_hgr.nc')
zgr   = xr.open_dataset(dir_grd + 'mesh_zgr.nc')
bathy = xr.open_dataset(dir_grd + 'bathy.nc')
mskNaN = msk.tmaskutil[0, :, :].data.astype('float')
mskNaN[np.where(mskNaN>0.0)] = 1.0
mskNaN[np.where(mskNaN==0.0)] = np.nan
[nr, ny, nx] = [ zgr.dims['z'], zgr.dims['y'], zgr.dims['x'] ]
ttt = 1

#-- list of files --
lSSH = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridT-2D/0*gridT-2D_20100406-20100406.nc') )
nmem = len(lSSH)
lMEC  = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'MEC_1d/*MEC_KE_20100406-20100406.nc') )
lEFLX = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'EDDYFLX_1d/*EFLX_KE_20100406-20100406.nc') )

#-- to check contamination due to dissipation --
# mec with dissipation
#lKE40 = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'TEST_ADV_4/*EDDY_KE40_20100406-20100406.nc')  )
# eddyflx with dissipation
#lKE20 = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'TEST_ADV_4/*EDDY_KE20_20100406-20100406.nc')  )

#------------------------------------------------------------------------------
#-- compute global integrated MEC and eddy flux and check that the sum is ~0 --
#------------------------------------------------------------------------------
mec_int = np.zeros([nmem])
eddyflx_int = np.zeros([nmem])
adveke_int  = np.zeros([nmem])
#dissip_mec_int  = np.zeros([nmem])
#dissip_eflx_int  = np.zeros([nmem])
mec_hz = np.zeros([nmem, ny, nx])
eddyflx_hz = np.zeros([nmem, ny, nx])
adveke_hz = np.zeros([nmem, ny, nx])
for imem in range(nmem):
  tmpssh  = xr.open_dataset(lSSH[imem])
  tmpmec  = xr.open_dataset(lMEC[imem])
  tmpeflx = xr.open_dataset(lEFLX[imem])
  #tmpmec_diss    = xr.open_dataset(lKE40[imem])
  #tmpeflx_diss   = xr.open_dataset(lKE20[imem])
  for kkk in range(nr):
    print("memb %02i, level %03i" % (imem, kkk) )
    #
    e123t = ( ( hgr.e1t * hgr.e2t )[0,:,:].data * \
      (zgr.e3t_0[0, kkk, :, :] * (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    mec_int[imem] = mec_int[imem] + \
            np.nansum( (tmpmec.advh_ke_m[ttt, kkk, :, :] \
                       +tmpmec.advz_ke_m[ttt, kkk, :, :] ) * e123t )
    eddyflx_int[imem] = eddyflx_int[imem] + \
            np.nansum( (tmpeflx.advh_ke_pr[ttt, kkk, :, :] \
                       +tmpeflx.advz_ke_pr[ttt, kkk, :, :] ) * e123t )
    adveke_int[imem] = adveke_int[imem] + \
            np.nansum( (tmpmec.advh_ke_pr[ttt, kkk, :, :] \
                       +tmpmec.advz_ke_pr[ttt, kkk, :, :] ) * e123t )
    #dissip1 = tmpmec_diss.advh_ke_m[ttt,kkk,:,:] - tmpmec.advh_ke_m[ttt,kkk,:,:]
    #dissip_mec_int[imem] = dissip_mec_int[imem] + np.nansum( dissip1 * e123t )
    #dissip2 = tmpeflx_diss.advh_ke_pr[ttt,kkk,:,:] - tmpeflx.advh_ke_pr[ttt,kkk,:,:]
    #dissip_eflx_int[imem] = dissip_eflx_int[imem] + np.nansum( dissip2 * e123t )
    #
    mec_hz[imem, :, :] = mec_hz[imem, :, :]   + \
            (tmpmec.advh_ke_m[ttt, kkk, :, :] + tmpmec.advz_ke_m[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    eddyflx_hz[imem, :, :] = eddyflx_hz[imem, :, :] + \
            (tmpeflx.advh_ke_pr[ttt, kkk, :, :] + tmpeflx.advz_ke_pr[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    adveke_hz[imem, :, :] = adveke_hz[imem, :, :]   + \
            (tmpmec.advh_ke_pr[ttt, kkk, :, :] + tmpmec.advz_ke_pr[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)


mec_int2 = np.zeros([nmem])
eddyflx_int2 = np.zeros([nmem])
mec_hz2 = np.zeros([nmem, ny, nx])
eddyflx_hz2 = np.zeros([nmem, ny, nx])
for imem in range(nmem):
  tmpssh  = xr.open_dataset(lSSH[imem])
  tmpmec  = xr.open_dataset(lMEC[imem])
  tmpeflx = xr.open_dataset(lEFLX[imem])
  for kkk in range(nr):
    print("memb %02i, level %03i" % (imem, kkk) )
    #
    e123t = ( ( hgr.e1t * hgr.e2t )[0,:,:].data * \
      (zgr.e3t_0[0, kkk, :, :] * (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    mec_int2[imem] = mec_int2[imem] + \
            np.nansum( (tmpmec.advh_ke_m[ttt, kkk, :, :] ) * e123t )
    eddyflx_int2[imem] = eddyflx_int2[imem] + \
            np.nansum( (tmpeflx.advh_ke_pr[ttt, kkk, :, :] ) * e123t )
    #
    mec_hz2[imem, :, :] = mec_hz2[imem, :, :]   + \
            (tmpmec.advh_ke_m[ttt, kkk, :, :] ).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    eddyflx_hz2[imem, :, :] = eddyflx_hz2[imem, :, :] + \
            (tmpeflx.advh_ke_pr[ttt, kkk, :, :] ).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)







#-- save --
tmp1 = np.zeros([3, nmem])
tmp1[0, :] = mec_int
tmp1[1, :] = eddyflx_int
tmp1[2, :] = adveke_int
tmp2 = np.zeros([3, nmem, ny, nx])
tmp2[0, :, :, :] = mec_hz
tmp2[1, :, :, :] = eddyflx_hz
tmp2[2, :, :, :] = adveke_hz

fileN1 = 'mec_eddyflx_volint_basin.bin'
f = open(dir_out + fileN1, 'wb')
tmp1.reshape([3*nmem]).astype('>f4').tofile(f)
f.close()
fileN2 = 'mec_eddyflx_hzmap_basin.bin'
f = open(dir_out + fileN2, 'wb')
tmp2.reshape([3*nmem*ny*nx]).astype('>f4').tofile(f)
f.close()

#-- load --
fileN1 = 'mec_eddyflx_volint_basin.bin'
f = open(dir_out + fileN1, 'r')
tmp1 = np.fromfile(f,'>f4').reshape([3, nmem])
f.close()
fileN2 = 'mec_eddyflx_hzmap_basin.bin'
f = open(dir_out + fileN2, 'r')
tmp2 = np.fromfile(f,'>f4').reshape([3, nmem, ny, nx])
f.close()
mec_int = tmp1[0, :]
eddyflx_int = tmp1[1, :]
mec_hz = tmp2[0, :, :, :]
eddyflx_hz = tmp2[1, :, :, :]

#-----------------
#   PLOT
#-----------------

#-- global integrals for each members --
fig1 = plt.figure(figsize=(7, 5))
fig1.clf()
ax1 = fig1.add_subplot(1,1,1)
p1 = ax1.plot(np.arange(1, 21), mec_int*1e-9,'k')
p2 = ax1.plot(np.arange(1, 21), eddyflx_int*1e-9,'r')
p3 = ax1.plot(np.arange(1, 21), (eddyflx_int+mec_int)*1e-9,'b')
#p4 = ax1.plot(np.arange(20), -dissip_int*1e-9,'orange')
plt.grid()
ax1.legend((p1[0],p2[0],p3[0]), \
        (r"$-\rho_0 \int \left< \mathbf{u}_h \right> \cdot \nabla \cdot (\mathbf{u}' \otimes \mathbf{u}_h') dV$", \
         r"$-\rho_0 \int (\mathbf{u}' \otimes \mathbf{u}_h') \cdot \nabla \left< \mathbf{u}_h \right> dV$", \
         r"$-\rho_0 \int \nabla \cdot \mathbf{u}' ( \left< \mathbf{u}_h \right> \cdot \mathbf{u}_h' ) dV$") \
        )
ax1.set_xticks(np.arange(0, 25, 5))
ax1.set_xlim([1, 20])
ax1.set_xlabel('Ensemble member', fontsize='xx-large')
ax1.set_ylabel('[GW]', fontsize='xx-large')
#
figN1 = 'medwest60_vint_mec_eddy_flux_div_20memb'
fig1.savefig(dir_fig + figN1 + '.png', dpi=100, bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', dpi=100, bbox_inches='tight')
plt.close(fig1)



#-- for ensemble mean --
fig1 = plt.figure(figsize=(15, 4))
fig1.clf()
llev1 = np.arange(-1, 1.1, 0.1) * 1e-1
cmap = 'RdBu_r'
#
ax1 = fig1.add_subplot(1, 3, 1)
cs1 = ax1.contourf(hgr.nav_lon, hgr.nav_lat, mec_hz.mean(0) * mskNaN, \
            levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$-\rho_0 \left< \mathbf{u}_h \right> \cdot \nabla \cdot \left< \mathbf{u}' \otimes \mathbf{u}_h' \right>$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
at2 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (mec_int.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at2)
#
ax2 = fig1.add_subplot(1, 3, 2)
cs2 = ax2.contourf(hgr.nav_lon, hgr.nav_lat, eddyflx_hz.mean(0) * mskNaN, \
            levels=llev1, extend='both', cmap=cmap)
at2 = AnchoredText(r"$-\rho_0 \left< \mathbf{u}' \otimes \mathbf{u}_h' \right> \cdot \nabla \left< \mathbf{u}_h \right>$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)
at2 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (eddyflx_int.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)
#
ax3 = fig1.add_subplot(1, 3, 3)
cs3 = ax3.contourf(hgr.nav_lon, hgr.nav_lat, (eddyflx_hz.mean(0)+mec_hz.mean(0))* mskNaN, \
            levels=llev1, extend='both', cmap=cmap)
at3 = AnchoredText(r"$-\rho_0 \nabla \cdot \left< \mathbf{u}'(\left< \mathbf{u}_h \right> \cdot \mathbf{u}_h') \right>$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at3)
at3 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % ((eddyflx_int.mean()+mec_int.mean())/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at3)
#
cbax1 = fig1.add_axes([0.92, 0.2, 0.01, 0.6])
cb1 = fig1.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'[W m$^{-2}$]', fontsize='x-large')
#
for ip in range(3):
  ax = fig1.add_subplot(1, 3, ip+1)
  ax.set_facecolor([0.5, 0.5, 0.5])
  ax.set_xlim([-5.5, 9])
  ax.set_ylim([35, 45])
  ax.set_xticks([0, 5])
  ax.set_xticklabels((r"0$^{\circ}$", r"5$^{\circ}$W"), fontsize='x-large')
  ax.set_yticks([38, 42])
  if ip == 0:
    ax.set_yticklabels((r"38$^{\circ}$N", r"42$^{\circ}$N"), fontsize='x-large')
  else:
    ax.set_yticklabels(())
#
#figN1 = 'medwest60_zint_mec_and_eddy_flux_ensemble_mean'
figN1 = 'medwest60_zint_mec_eddy_flux_div_ensemble_mean'
fig1.savefig(dir_fig + figN1 + '.png', dpi=100, bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', dpi=100, bbox_inches='tight')
plt.close(fig1)


#-- for a given member --
imem = 0 
fig1 = plt.figure(figsize=(12, 5))
fig1.clf()
llev1 = np.arange(-5, 5.5, 0.5) * 1e-1
cmap = 'RdBu_r'
#
ax1 = fig1.add_subplot(1, 2, 1)
cs1 = ax1.contourf(hgr.nav_lon, hgr.nav_lat, mec_hz[imem, :, :] * mskNaN, \
            levels=llev1, extend='both', cmap=cmap)
ax1.set_facecolor([0.5, 0.5, 0.5])
ax1.set_yticks([38, 42])
ax1.set_yticklabels((r"38$^{\circ}$N", r"42$^{\circ}$N"), fontsize='x-large')
at1 = AnchoredText(r"$-\rho_0 \int \left< u_h \right> \nabla \cdot ( \mathbf{u}' \otimes \mathbf{u}_h' ) dz$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
at2 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (mec_int[imem]/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at2)
#
ax2 = fig1.add_subplot(1, 2, 2)
cs2 = ax2.contourf(hgr.nav_lon, hgr.nav_lat, eddyflx_hz[imem, :, :] * mskNaN, \
            levels=llev1, extend='both', cmap=cmap)
ax2.set_facecolor([0.5, 0.5, 0.5])
ax2.set_yticks([38, 42])
ax2.set_yticklabels(())
at2 = AnchoredText(r"$-\rho_0 \int ( \mathbf{u}_h' \otimes \mathbf{u}' ) \cdot \nabla \left< u_h \right> dz$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)
at2 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (eddyflx_int[imem]/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)
#
cbax1 = fig1.add_axes([0.92, 0.2, 0.01, 0.6])
cb1 = fig1.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'[W m$^{-2}$]')
#
for ip in range(2):
  ax = fig1.add_subplot(1, 2, ip+1)
  ax.set_xlim([-5.5, 9])
  ax.set_ylim([35, 45])
  ax.set_xticks([0, 6])
  ax.set_xticklabels((r"0$^{\circ}$", r"6$^{\circ}$W"), fontsize='x-large')
#
figN1 = 'medwest60_zint_mec_and_eddy_flux_m' + str("%02i" % imem)
fig1.savefig(dir_fig + 'movie/' + figN1 + '.png', dpi=100)
#fig1.savefig(dir_fig + 'movie/' + figN1 + '.pdf', dpi=100)


