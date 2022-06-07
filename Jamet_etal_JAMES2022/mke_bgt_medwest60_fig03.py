import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import xarray as xr
import glob
from tkinter import Tcl
from matplotlib.offsetbox import AnchoredText
import time

plt.ion()

#-- directories --
dir_in  = '/gpfsscratch/rech/egi/uup63gs/cdftools/'
dir_in2 = '/gpfsstore/rech/egi/uup63gs/medwest60/MEDWEST60-GSL19-S/ens01/1h/'
dir_grd = '/gpfsstore/rech/egi/uup63gs/medwest60/mesh/'
dir_fig = '/linkhome/rech/genige01/uup63gs/Figures/energetics/'
dir_out = '/gpfsstore/rech/egi/uup63gs/medwest60/outputs/'


#-- mesh and mask --
hgr   = xr.open_dataset(dir_grd + 'mesh_hgr.nc')
e12t  = (hgr.e1t*hgr.e2t)[0, :, :]
zgr   = xr.open_dataset(dir_grd + 'mesh_zgr.nc')
msk   = xr.open_dataset(dir_grd + 'mask.nc')
mskNaN = msk.tmaskutil[0, :, :].fillna(0.0).data
mskNaN[ np.where(mskNaN != 0.0) ] = 1.0
mskNaN[ np.where(mskNaN == 0.0) ] = np.nan
bathy = xr.open_dataset(dir_grd + 'bathy.nc')
[nr, ny, nx] = [msk.dims['z'], msk.dims['y'], msk.dims['x']]
dt = 80.0
rau0 = 1026.0
nmem = 20
nt = 24
ttt = 1

#-- list of files --
lSSH = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridT-2D/0*gridT-2D_20100406-20100406.nc') )
lSSHm= Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridT-2D/ESTATS_*gridT-2D_20100406-20100406.nc') )
nmem = len(lSSH)
lDMKEDT   = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'DMKEDT/ESTATS*DMKEDT_20100406-20100406.nc') )
lADVMKE  = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'ADVMKE/ESTATS*ADV_MKE_20100406-20100406.nc') )
lMEC     = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'MEC_1d/*MEC_KE_20100406-20100406.nc') )
#

#-- region of interest --
iiw  = np.where(hgr.nav_lon[200, :] >= 4.8)[0][0]
iie  = np.where(hgr.nav_lon[200, :] >= 7.8)[0][0]
jjs  = np.where(hgr.nav_lat[:, iiw] >= 37.2)[0][0]
jjn  = np.where(hgr.nav_lat[:, iiw] >= 40.45)[0][0]
[ny2, nx2] = [jjn-jjs, iie-iiw]
iiw0  = iiw-50
iie0  = iie+50
jjs0  = jjs-30
jjn0  = jjn+50
[ny0, nx0] = [jjn0-jjs0, iie0-iiw0]


#----------------------------
# Vertical profile and hz map
#----------------------------
mec_int    = np.zeros([nmem])
mec_zprof  = np.zeros([nmem, nr])
mecz_zprof  = np.zeros([nmem, nr])
mec_hz     = np.zeros([nmem, ny, nx])
mecz_hz     = np.zeros([nmem, ny, nx])
for imem in range(nmem):
  tmpssh  = xr.open_dataset(lSSH[imem])
  tmpmec  = xr.open_dataset(lMEC[imem])
  for kkk in range(nr):
    print("memb %02i, level %03i" % (imem, kkk) )
    #
    e123t = ( ( hgr.e1t * hgr.e2t )[0,:,:].data * \
      (zgr.e3t_0[0, kkk, :, :] * (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    mec_int[imem] = mec_int[imem] + \
            np.nansum( (tmpmec.advh_ke_m[ttt, kkk, jjs:jjn, iiw:iie] \
                       +tmpmec.advz_ke_m[ttt, kkk, jjs:jjn, iiw:iie]) \
                       * e123t[jjs:jjn, iiw:iie] )
    mec_zprof[imem, kkk] = \
            np.nansum( np.nansum( \
            (tmpmec.advh_ke_m[ttt, kkk, jjs:jjn, iiw:iie]  \
            +tmpmec.advz_ke_m[ttt, kkk, jjs:jjn, iiw:iie]).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    mecz_zprof[imem, kkk] = \
            np.nansum( np.nansum( \
            (tmpmec.advz_ke_m[ttt, kkk, jjs:jjn, iiw:iie]).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    #
    mec_hz[imem, :, :] = mec_hz[imem, :, :]   + \
            (tmpmec.advh_ke_m[ttt, kkk, :, :] + tmpmec.advz_ke_m[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    mecz_hz[imem, :, :] = mecz_hz[imem, :, :]   + \
            (tmpmec.advz_ke_m[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)

#-- save --
tmp1 = np.zeros([2, nmem, nr])
tmp1[0, :, :] = mec_zprof
tmp1[1, :, :] = mecz_zprof
tmp2 = np.zeros([2, nmem, ny, nx])
tmp2[0, :, :, :] = mec_hz
tmp2[1, :, :, :] = mecz_hz
#
fileN1 = 'mec_mecz_zprof_box.bin'
f = open(dir_out + fileN1, 'wb')
tmp1.reshape([2*nmem*nr]).astype('>f4').tofile(f)
f.close()
fileN2 = 'mec_mecz_hz_map_basin.bin'
f = open(dir_out + fileN2, 'wb')
tmp2.reshape([2*nmem*ny*nx]).astype('>f4').tofile(f)
f.close()

#-- load --
fileN1 = 'mec_mecz_zprof_box.bin'
f = open(dir_out + fileN1, 'r')
tmp1 = np.fromfile(f, '>f4').reshape([2, nmem, nr])
mec_zprof  = tmp1[0, :, :]
mecz_zprof = tmp1[1, :, :]
f.close()
fileN2 = 'mec_mecz_hz_map_basin.bin'
f = open(dir_out + fileN2, 'r')
tmp2 = np.fromfile(f, '>f4').reshape([2, nmem, ny, nx])
mec_hz  = tmp2[0, :, :, :]
mecz_hz = tmp2[1, :, :, :]
#- recompute volume integration -
mec_int = np.nansum( np.nansum( mec_hz[:, jjs:jjn, iiw:iie] * \
        e12t.data[np.newaxis, jjs:jjn, iiw:iie], axis=-1), axis=-1)


#-- dmkedt and adv MKE --
tmpsshm  = xr.open_dataset(lSSH[0])
tmpdt    = xr.open_dataset(lDMKEDT[0])
tmpadv   = xr.open_dataset(lADVMKE[0])
#
dt_mke_int    = 0.0
adv_mke_int   = 0.0
dt_mke_zprof  = np.zeros([nr])
adv_mke_zprof = np.zeros([nr])
dt_mke_hz     = np.zeros([ny, nx])
adv_mke_hz    = np.zeros([ny, nx])
for kkk in range(nr):
    print("level %03i" % (kkk) )
    e123t = ( ( hgr.e1t * hgr.e2t )[0,:,:].data * \
      (zgr.e3t_0[0, kkk, :, :] * (1 + tmpsshm.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    #
    dt_mke_int = dt_mke_int + \
            np.nansum( (tmpdt.dkedt[ttt, kkk, jjs:jjn, iiw:iie]) \
                      * e123t[jjs:jjn, iiw:iie] )
    adv_mke_int = adv_mke_int + \
            np.nansum( (tmpadv.advh_ke_m[ttt, kkk, jjs:jjn, iiw:iie] \
                       +tmpadv.advz_ke_m[ttt, kkk, jjs:jjn, iiw:iie]).data \
                      * e123t[jjs:jjn, iiw:iie] )
    #
    dt_mke_zprof[kkk] = \
            np.nansum( np.nansum( \
            (tmpdt.dkedt[ttt, kkk, jjs:jjn, iiw:iie]).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    adv_mke_zprof[kkk] = \
            np.nansum( np.nansum( \
            (tmpadv.advh_ke_m[ttt, kkk, jjs:jjn, iiw:iie] \
            +tmpadv.advz_ke_m[ttt, kkk, jjs:jjn, iiw:iie]).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    #
    dt_mke_hz = dt_mke_hz   + \
            (tmpdt.dkedt[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpsshm.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    adv_mke_hz = adv_mke_hz   + \
            (tmpadv.advh_ke_m[ttt, kkk, :, :] \
            +tmpadv.advz_ke_m[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpsshm.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)

#-- save --
tmp1 = np.zeros([2, nr])
tmp1[0, :] = dt_mke_zprof
tmp1[1, :] = adv_mke_zprof
tmp2 = np.zeros([2, ny, nx])
tmp2[0, :, :] = dt_mke_hz
tmp2[1, :, :] = adv_mke_hz
#
fileN1 = 'dmkedt_advmke_zprof_box.bin'
f = open(dir_out + fileN1, 'wb')
tmp1.reshape([2*nr]).astype('>f4').tofile(f)
f.close()
fileN2 = 'dmkedt_advmke_hz_map_basin.bin'
f = open(dir_out + fileN2, 'wb')
tmp2.reshape([2*ny*nx]).astype('>f4').tofile(f)
f.close()
#-- load --
fileN1 = 'dmkedt_advmke_zprof_box.bin'
f = open(dir_out + fileN1, 'r')
tmp1 = np.fromfile(f, '>f4').reshape([2, nr])
dt_mke_zprof  = tmp1[0, :]
adv_mke_zprof = tmp1[1, :]
f.close()
fileN2 = 'dmkedt_advmke_hz_map_basin.bin'
f = open(dir_out + fileN2, 'r')
tmp2 = np.fromfile(f, '>f4').reshape([2, ny, nx])
dt_mke_hz  = tmp2[0, :, :]
adv_mke_hz = tmp2[1, :, :]
#- recompute volume integration -
dt_mke_int = np.nansum( np.nansum( dt_mke_hz[jjs:jjn, iiw:iie] * \
        e12t.data[jjs:jjn, iiw:iie], axis=-1), axis=-1)
adv_mke_int = np.nansum( np.nansum( adv_mke_hz[jjs:jjn, iiw:iie] * \
        e12t.data[jjs:jjn, iiw:iie], axis=-1), axis=-1)



#-- time evolution of dmkedt --
dt_mke_hz_time     = np.zeros([nt, ny, nx])
for ttt in range(1, nt-1):
  for kkk in range(nr):
    print("Time %02i, level %03i" % (ttt, kkk) )
    e123t = ( ( hgr.e1t * hgr.e2t )[0,:,:].data * \
      (zgr.e3t_0[0, kkk, :, :] * (1 + tmpsshm.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    #
    dt_mke_hz_time[ttt, :, :] = dt_mke_hz_time[ttt, :, :]   + \
            (tmpdt.dkedt[ttt, kkk, :, :]).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpsshm.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)





#============================================
#               PLOT
#============================================
fig1 = plt.figure(figsize=(5, 6))
fig1.clf()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_position([0.17, 0.11, 0.78, 0.88])
p2 = ax1.plot(adv_mke_zprof / 1e9, -zgr.gdept_1d[0, :], 'b', linewidth=3 )
p0 = ax1.plot(dt_mke_zprof  / 1e9, -zgr.gdept_1d[0, :], 'k', linewidth=3)
p1 = ax1.plot(mec_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'r', linewidth=3)
p11 = ax1.plot(mecz_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'r--')
plt.legend((p0[0], p2[0], p1[0], p11[0]), \
        (r'$\partial_t \tilde{K}$', \
        r'$-\nabla \cdot \left< \mathbf{u} \right> \tilde{K}$', \
	r"$-\rho_0 \left< \mathbf{u}_h \right> \cdot \nabla \cdot \left< \mathbf{u}' \mathbf{u}_h' \right>$", \
        r"$-\rho_0 \left< u_h \right> \partial_z \left< w' u_h' \right>$"), \
        fontsize='x-large')
plt.grid()
ax1.set_ylim([-600, 0])
ax1.set_xlim([-0.008, 0.008])
ax1.set_xticks(np.arange(-0.01, 0.015, 0.005))
ax1.set_ylabel('Depth [m]', fontsize='x-large')
ax1.set_xlabel(r'[GW m^{-1}$$]', fontsize='x-large')
#
figN1 = 'mke_bgt_dkedt_advmke_mec_box'
fig1.savefig(dir_fig + figN1 + '.png', bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', bbox_inches='tight')
plt.close(fig1)


#--
fig2 = plt.figure(figsize=(18, 5))
fig2.clf()
llev = np.arange(-0.2, 0.22, 0.02)
cmap = 'RdBu_r'
#
ax0 = fig2.add_subplot(1, 3, 1)
cs0 = ax0.contourf(tmpadv.nav_lon, tmpadv.nav_lat, \
        dt_mke_hz * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$\partial_t \tilde{K}$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (dt_mke_int/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(at1)
#
ax1 = fig2.add_subplot(1, 3, 2)
cs1 = ax1.contourf(tmpadv.nav_lon, tmpadv.nav_lat, \
	adv_mke_hz * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$-\nabla \cdot \left< \mathbf{u} \right> \tilde{K}$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (adv_mke_int/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
#
ax2 = fig2.add_subplot(1, 3, 3)
cs2 = ax2.contourf(tmpadv.nav_lon, tmpadv.nav_lat, \
        mec_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r"$-\rho_0 \left< u_h \right> \nabla \cdot \left< \mathbf{u}' u_h' \right>$", prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (mec_int.mean(0)/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
#
for ip in range(3):
  ax = fig2.add_subplot(1, 3, ip+1)
  ax.set_ylim([36.5, 41.2])
  ax.set_xlim([3, 9])
  ax.set_facecolor([0.5, 0.5, 0.5])
  ax.set_yticks([37, 40])
  ax.set_yticklabels([r'37 N', r'40 N'])
  ax.set_xticks([4, 8])
  ax.set_xticklabels(['4 E', '8 E'])
  p = mp.patches.Rectangle((4.8, 37.2), 3, 3.25, linewidth=3, fill=False, color='g')
  ax.add_patch(p)
#-- colorbar --
cbax1 = fig2.add_axes([0.92, 0.2, 0.01, 0.6])
cb1 = fig2.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'[W m$^{-2}$]')
#
figN2 = 'KE_medwest60_adv_mke_mec_zint'
fig2.savefig(dir_fig + figN2 + '.png', bbox_inches='tight')
fig2.savefig(dir_fig + figN2 + '.pdf', bbox_inches='tight')
plt.close(fig2)



#-- all together --
fig3 = plt.figure(figsize=(10, 10))
fig3.clf()
llev = np.arange(-0.2, 0.22, 0.02)
cmap = 'RdBu_r'
#
ax1 = fig3.add_subplot(2, 2, 1)
cs1 = ax1.contourf(hgr.nav_lon, hgr.nav_lat, \
        dt_mke_hz * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$\partial_t \tilde{K}$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (dt_mke_int/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
#
ax2 = fig3.add_subplot(2, 2, 2)
cs1 = ax2.contourf(hgr.nav_lon, hgr.nav_lat, \
        adv_mke_hz * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$-\nabla \cdot (\left< \mathbf{u} \right> \tilde{K})$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (adv_mke_int/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
#
ax3 = fig3.add_subplot(2, 2, 3)
cs1 = ax3.contourf(hgr.nav_lon, hgr.nav_lat, \
        mec_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r"$-\rho_0 \left< \mathbf{u}_h \right> \cdot \nabla \cdot \left< \mathbf{u}' \otimes \mathbf{u}_h' \right>$", prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (mec_int.mean(0)/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at1)
#
ax4 = fig3.add_subplot(2, 2, 4)
#ax3.set_position([0.17, 0.11, 0.78, 0.88])
p2  = ax4.plot(adv_mke_zprof / 1e9, -zgr.gdept_1d[0, :], 'b', linewidth=3 )
p0  = ax4.plot(dt_mke_zprof  / 1e9, -zgr.gdept_1d[0, :], 'k', linewidth=3)
p3  = ax4.plot((dt_mke_zprof - (mec_zprof.mean(0)+adv_mke_zprof)) / 1e9, -zgr.gdept_1d[0, :], \
        'g', linewidth=1)
p1  = ax4.plot(mec_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'r', linewidth=3)
#p11 = ax4.plot(mecz_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'r--')
plt.legend((p0[0], p2[0], p1[0], p3[0]), \
        (r'$\partial_t \tilde{K}$', \
        r'$-\nabla \cdot ( \left< \mathbf{u} \right> \tilde{K} )$', \
        r"$-\rho_0 \left< \mathbf{u}_h \right> \cdot \nabla \cdot \left< \mathbf{u}' \otimes \mathbf{u}_h' \right>$", \
        '$Residual$'), \
        fontsize='large')
plt.grid()
ax4.set_ylim([-500, 0])
ax4.set_xlim([-0.008, 0.008])
ax4.set_xticks(np.arange(-0.01, 0.015, 0.005))
ax4.set_ylabel('Depth [m]', fontsize='large')
ax4.set_yticks([-500, -400, -300, -200, -100, -40, 0])
ax4.set_xlabel(r'[GW m$^{-1}$]', fontsize='large')
#
for ip in range(3):
  ax = fig3.add_subplot(2, 2, ip+1)
  ax.set_ylim([36.5, 41.2])
  ax.set_xlim([3, 9])
  ax.set_facecolor([0.5, 0.5, 0.5])
  ax.set_yticks([37, 40])
  ax.set_yticklabels([r'37$^{\circ}$N', r'40$^{\circ}$ N'], fontsize='x-large')
  ax.set_xticks([4, 8])
  ax.set_xticklabels(['4$^{\circ}$E', '8$^{\circ}$E'], fontsize='x-large')
  p = mp.patches.Rectangle((4.8, 37.2), 3, 3.25, linewidth=3, fill=False, color='g')
  ax.add_patch(p)
#-- colorbar --
cbax1 = fig3.add_axes([0.91, 0.2, 0.01, 0.6])
cb1 = fig3.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'[W m$^{-2}$]', fontsize='large')
#
figN3 = 'KE_medwest60_adv_mke_mec_hz_zprof_box'
fig3.savefig(dir_fig + figN3 + '.png', bbox_inches='tight')
fig3.savefig(dir_fig + figN3 + '.pdf', bbox_inches='tight')



#-- movie --
plt.figure();
for ttt in range(1, nt-1):
   plt.clf()
   plt.contourf(dt_mke_hz_time[ttt, jjs0:jjn0, iiw0:iie0], levels=llev, \
           extend='both',cmap=cmap);
   plt.colorbar
   plt.title("Time: %02i" % ttt)
   plt.pause(0.2)
    


