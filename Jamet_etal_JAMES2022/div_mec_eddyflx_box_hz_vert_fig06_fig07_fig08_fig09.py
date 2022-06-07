import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import xarray as xr
import glob
from tkinter import Tcl
from matplotlib.offsetbox import AnchoredText

plt.ion()

#-- directories --
dir_in  = '/gpfsscratch/rech/egi/uup63gs/cdftools/'
dir_in2 = '/gpfsstore/rech/egi/uup63gs/medwest60/MEDWEST60-GSL19-S/ens01/1h/'
dir_grd = '/gpfsstore/rech/egi/uup63gs/medwest60/mesh/'
dir_fig = '/linkhome/rech/genige01/uup63gs/Figures/energetics/'
dir_out = '/gpfsstore/rech/egi/uup63gs/medwest60/outputs/'


#-- mesh and mask --
msk   = xr.open_dataset(dir_grd + 'mask.nc')
hgr   = xr.open_dataset(dir_grd + 'mesh_hgr.nc')
zgr   = xr.open_dataset(dir_grd + 'mesh_zgr.nc')
bathy = xr.open_dataset(dir_grd + 'bathy.nc')
e12t  = (hgr.e1t * hgr.e2t).data
e123t = e12t * zgr.e3t_0[0, :, :, :].data
mskNaN = msk.tmaskutil[0, :, :].data.astype('float')
mskNaN[np.where(mskNaN>0.0)] = 1.0
mskNaN[np.where(mskNaN==0.0)] = np.nan
[nr, ny, nx] = [ zgr.dims['z'], zgr.dims['y'], zgr.dims['x'] ]
dt = 80.0
rau0 = 1026.0
nmem = 20
nt = 24
ttt = 1

#-- list of files --
lUm      = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridU/ESTATS_*_20100406-20100406.nc') )
lVm      = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridV/ESTATS_*_20100406-20100406.nc') )
lMLD     = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridT-2D/0*_20100406-20100406.nc') )
lMEC     = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'MEC_1d/*MEC_KE_20100406-20100406.nc') )
lEDDYFLX = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'EDDYFLX_1d/0*EFLX_KE_20100406-20100406.nc' ) )

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


#----------------------------------------------
#       Ensemble mean velocities
#----------------------------------------------
uum = xr.open_dataset(lUm[0])
vvm = xr.open_dataset(lVm[0])

uvm = np.zeros([nr, ny, nx])
uvm[:, 1:, 1:] = 0.5 * np.sqrt( \
        (uum.vozocrtx[ttt, :, 1:, 1:]+uum.vozocrtx[ttt, :, 1: , :-1]).data**2 \
       +(vvm.vomecrty[ttt, :, 1:, 1:]+vvm.vomecrty[ttt, :, :-1, 1: ]).data**2 )

#-- across the section (South eastawrd) --
#- interpolate at t-point -
umc = np.zeros([nr, ny, nx])
vmc = np.zeros([nr, ny, nx])
umc[:, :, 1:] = 0.5 * (uum.vozocrtx[ttt, :, :, 1:]+uum.vozocrtx[ttt, :, :, :-1])
vmc[:, 1:, :] = 0.5 * (vvm.vomecrty[ttt, :, 1:, :]+vvm.vomecrty[ttt, :, :-1, :])
#- -pi/4 rotation -
theta = np.pi/4
rotmat = [np.cos(theta), -np.sin(theta)] , [np.sin(theta), np.cos(theta)]
um_rot = umc * np.cos(theta) - vmc * np.sin(theta)
vm_rot = umc * np.sin(theta) + vmc * np.cos(theta)

#-- ensemble mean MLD --
mld = np.zeros([nmem, ny, nx])
for imem in range(nmem):
    print("memb%02i" % imem)
    tmpmld = xr.open_dataset(lMLD[imem])
    mld[imem, :, :] = tmpmld.somxl010[ttt,:,:]
mldm = mld.mean(0)

#----------------------------------------------
#       horizontal and vertical contrib
#----------------------------------------------
mec_uv  = np.zeros([nr, ny, nx])
mec_w   = np.zeros([nr, ny, nx])
eflx_uv = np.zeros([nr, ny, nx])
eflx_w  = np.zeros([nr, ny, nx])
for imem in range(nmem):
    print("memb%03i" % imem)
    tmpmec  = xr.open_dataset(lMEC[imem])
    tmpeflx = xr.open_dataset(lEDDYFLX[imem])
    # MEC
    mec_uv = mec_uv + tmpmec.advh_ke_m[ttt, :, :, :].fillna(0)
    mec_w  = mec_w  + tmpmec.advz_ke_m[ttt, :, :, :].fillna(0)
    # EFLX
    eflx_uv = eflx_uv + tmpeflx.advh_ke_pr[ttt, :, :, :].fillna(0)
    eflx_w  = eflx_w  + tmpeflx.advz_ke_pr[ttt, :, :, :].fillna(0)
#
mec_uv  = mec_uv  / float(nmem)
mec_w   = mec_w   / float(nmem)
eflx_uv = eflx_uv / float(nmem)
eflx_w  = eflx_w  / float(nmem)

#-- save --
tmp1 = np.zeros([2, nr, ny, nx])
tmp1[0, :, :, :] = mec_uv
tmp1[1, :, :, :] = mec_w
tmp2 = np.zeros([2, nr, ny, nx])
tmp2[0, :, :, :] = eflx_uv
tmp2[1, :, :, :] = eflx_w
#
fileN1 = 'mec_hz_vert_decomposition_basin.bin'
f = open(dir_out + fileN1, 'wb')
tmp1.reshape([2*nr*ny*nx]).astype('>f4').tofile(f)
f.close()
fileN2 = 'eddyflx_hz_vert_decomposition_basin.bin'
f = open(dir_out + fileN2, 'wb')
tmp2.reshape([2*nr*ny*nx]).astype('>f4').tofile(f)
f.close()
#-- load --
fileN1 = 'mec_hz_vert_decomposition_basin.bin'
f = open(dir_out + fileN1, 'r')
tmp1 = np.fromfile(f, '>f4').reshape([2, nr, ny, nx])
mec_uv = tmp1[0, :, :, :]
mec_w  = tmp1[1, :, :, :]
mec    = mec_uv + mec_w
f.close()
fileN2 = 'eddyflx_hz_vert_decomposition_basin.bin'
f = open(dir_out + fileN2, 'r')
tmp2 = np.fromfile(f, '>f4').reshape([2, nr, ny, nx])
eflx_uv = tmp2[0, :, :, :]
eflx_w  = tmp2[1, :, :, :]
eflx    = eflx_uv + eflx_w
f.close()
#
divef    = mec + eflx
divef_uv = mec_uv + eflx_uv
divef_w  = mec_w + eflx_w

#-- vertical profil within the box --
kkk = np.where(zgr.gdept_1d[0,:] < 500)[0]
# mec 
mec_zprof = ( (mec[:, jjs:jjn, iiw:iie]) * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
mec_uv_zprof = (mec_uv[:, jjs:jjn, iiw:iie] * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
mec_w_zprof = (mec_w[:, jjs:jjn, iiw:iie] * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
# eddyflx 
eflx_zprof = ( (eflx[:, jjs:jjn, iiw:iie]) * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
eflx_uv_zprof = (eflx_uv[:, jjs:jjn, iiw:iie] * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
eflx_w_zprof = (eflx_w[:, jjs:jjn, iiw:iie] * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
# divef
divef_zprof = ( (divef[:, jjs:jjn, iiw:iie]) * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
divef_uv_zprof = ( (divef_uv[:, jjs:jjn, iiw:iie]) * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)
divef_w_zprof = ( (divef_w[:, jjs:jjn, iiw:iie]) * e12t[:, jjs:jjn, iiw:iie]).sum(axis=-1).sum(axis=-1)

#-- full depth hoizontal maps of mec and eflx (full) --
mec_hz    = np.nansum(mec[:, jjs0:jjn0, iiw0:iie0]  * zgr.e3t_0[0, :, jjs0:jjn0, iiw0:iie0], axis=0)
eflx_hz   = np.nansum(eflx[:, jjs0:jjn0, iiw0:iie0] * zgr.e3t_0[0, :, jjs0:jjn0, iiw0:iie0], axis=0)
divef_hz  = np.nansum(divef[:, jjs0:jjn0, iiw0:iie0]  * zgr.e3t_0[0, :, jjs0:jjn0, iiw0:iie0], axis=0)
mec_int   = np.nansum(mec [:,jjs:jjn, iiw:iie]*e123t[:, jjs:jjn, iiw:iie])
eflx_int  = np.nansum(eflx[:,jjs:jjn, iiw:iie]*e123t[:, jjs:jjn, iiw:iie])
divef_int = np.nansum(divef[:,jjs:jjn, iiw:iie]*e123t[:, jjs:jjn, iiw:iie])

#-- upper and lower hz map --
kkk_w = np.where(eflx_w_zprof == eflx_w_zprof.min())[0][0]
kkk_w = np.where(eflx_w_zprof[:kkk_w] > 0)[0][-1]
# mec 
mec_hz_up = np.nansum( (mec_uv+mec_w)[:kkk_w, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, :kkk_w, jjs0:jjn0, iiw0:iie0], axis=0)
mec_uv_hz_up = np.nansum( (mec_uv)[:kkk_w, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, :kkk_w, jjs0:jjn0, iiw0:iie0], axis=0)
mec_w_hz_up = np.nansum( (mec_w)[:kkk_w, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, :kkk_w, jjs0:jjn0, iiw0:iie0], axis=0)
mec_hz_dw = np.nansum( (mec_uv+mec_w)[kkk_w:, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, kkk_w:, jjs0:jjn0, iiw0:iie0], axis=0)
mec_uv_hz_dw = np.nansum( (mec_uv)[kkk_w:, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, kkk_w:, jjs0:jjn0, iiw0:iie0], axis=0)
mec_w_hz_dw = np.nansum( (mec_w)[kkk_w:, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, kkk_w:, jjs0:jjn0, iiw0:iie0], axis=0)
# efx 
eflx_hz_up = np.nansum( (eflx_uv+eflx_w)[:kkk_w, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, :kkk_w, jjs0:jjn0, iiw0:iie0], axis=0)
eflx_uv_hz_up = np.nansum( (eflx_uv)[:kkk_w, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, :kkk_w, jjs0:jjn0, iiw0:iie0], axis=0)
eflx_w_hz_up = np.nansum( (eflx_w)[:kkk_w, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, :kkk_w, jjs0:jjn0, iiw0:iie0], axis=0)
eflx_hz_dw = np.nansum( (eflx_uv+eflx_w)[kkk_w:, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, kkk_w:, jjs0:jjn0, iiw0:iie0], axis=0)
eflx_uv_hz_dw = np.nansum( (eflx_uv)[kkk_w:, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, kkk_w:, jjs0:jjn0, iiw0:iie0], axis=0)
eflx_w_hz_dw = np.nansum( (eflx_w)[kkk_w:, jjs0:jjn0, iiw0:iie0] * \
        zgr.e3t_0[0, kkk_w:, jjs0:jjn0, iiw0:iie0], axis=0)

#-- extract cross-stream section --
ii1 = np.arange(700, 750)
nxny1 = len(ii1)
jj1 = np.arange(250, 250+nxny1)
xx_sec = np.zeros([nxny1])
yy_sec = np.zeros([nxny1])
ddist  = np.zeros([nxny1])
uvm_sec = np.zeros([nr, nxny1])
mec_sec = np.zeros([nr, nxny1])
eflx_sec = np.zeros([nr, nxny1])
divef_sec = np.zeros([nr, nxny1])
mecw_sec = np.zeros([nr, nxny1])
eflxw_sec = np.zeros([nr, nxny1])
divefw_sec = np.zeros([nr, nxny1])
mldm_sec = np.zeros([nxny1])
for iijj in range(nxny1):
    xx_sec[iijj] = hgr.nav_lon[jj1[iijj], ii1[iijj]]
    yy_sec[iijj] = hgr.nav_lat[jj1[iijj], ii1[iijj]]
    if iijj > 0:
        ddist[iijj] = ddist[iijj-1] + \
                np.sqrt( hgr.e1u[0, jj1[iijj-1], ii1[iijj-1]]**2 \
                        +hgr.e2v[0, jj1[iijj-1], ii1[iijj-1]]**2)
    #
    uvm_sec[:, iijj] = um_rot[:, jj1[iijj], ii1[iijj]]
    #
    mec_sec[:, iijj] = mec[:, jj1[iijj], ii1[iijj]]
    eflx_sec[:, iijj] = eflx[:, jj1[iijj], ii1[iijj]]
    divef_sec[:, iijj] = divef[:, jj1[iijj], ii1[iijj]]
    #
    mecw_sec[:, iijj] = mec_w[:, jj1[iijj], ii1[iijj]]
    eflxw_sec[:, iijj] = eflx_w[:, jj1[iijj], ii1[iijj]]
    divefw_sec[:, iijj] = divef_w[:, jj1[iijj], ii1[iijj]]
    #
    mldm_sec[iijj] = mldm[jj1[iijj], ii1[iijj]]


#-----------------------------------------
#               PLOT
#-----------------------------------------

#-- horizontal maps of full depth integrated MEC and EDDYFLX --
llev1 = np.arange(-1, 1.1, 0.1) * 1e-1
llev2 = np.arange(-2, 2.2, 0.2)
scaleF = 1e3
cmap = 'RdBu_r'
xx = hgr.nav_lon[jjs0:jjn0, iiw0:iie0]
yy = hgr.nav_lat[jjs0:jjn0, iiw0:iie0]
msk = mskNaN[jjs0:jjn0, iiw0:iie0]

fig1 = plt.figure(figsize=(15, 5))
fig1.clf()
#
ax1 = fig1.add_subplot(1, 3, 1)
cs1 = ax1.contourf(xx, yy, mec_hz * msk, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$MEC$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
at2 = AnchoredText(r"$\int \cdot dV_{box} = %0.02f$ GW" % (mec_int/1e9), \
       prop=dict(size=10), frameon=True, loc='lower right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at2)
#
ax2 = fig1.add_subplot(1, 3, 2)
cs2 = ax2.contourf(xx, yy, eflx_hz* msk, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$EDDYFLX$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
at2 = AnchoredText(r"$\int \cdot dV_{box} = %0.02f$ GW" % (eflx_int/1e9), \
       prop=dict(size=10), frameon=True, loc='lower right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)
#
ax3 = fig1.add_subplot(1, 3, 3)
cs2 = ax3.contourf(xx, yy, (mec_hz+eflx_hz)* msk, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$DIVEF$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at1)
at2 = AnchoredText(r"$\int \cdot dV_{box} = %0.02f$ GW" % ((mec_int+eflx_int)/1e9), \
       prop=dict(size=10), frameon=True, loc='lower right')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at2)
#
cbax1 = fig1.add_axes([0.92, 0.2, 0.01, 0.6])
cb1 = fig1.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'[W m$^{-2}$]', fontsize='x-large')
#
for ip in range(3):
  ax = fig1.add_subplot(1, 3, ip+1)
  qv = ax.quiver(hgr.nav_lon[::10, ::10], hgr.nav_lat[::10, ::10], \
        uum.vozocrtx[ttt, 0, ::10, ::10], vvm.vomecrty[ttt, 0, ::10, ::10], \
        color='k', alpha=0.5)
  ax.set_facecolor([0.5, 0.5, 0.5])
  ax.set_ylim([36.7, 40.9])
  ax.set_xlim([4.1, 8.5])
  ax.set_xticks([5.0, 7.0])
  ax.set_xticklabels((r"5$^{\circ}$W", r"7$^{\circ}$W"), fontsize='x-large')
  ax.set_yticks([38, 40])
  p = mp.patches.Rectangle((4.8, 37.2), 3, 3.25, linewidth=3, fill=False, color='g')
  ax.add_patch(p)
  ax.plot(xx_sec, yy_sec, 'k.')
  if ip == 0:
    ax.set_yticklabels((r"38$^{\circ}$N", r"40$^{\circ}$N"), fontsize='x-large')
    plt.quiverkey(qv, 1.1, 0.5, 1, '[1 m/s]')
  else:
    ax.set_yticklabels(())
#
figN1 = 'medwest60_zint_mec_eddy_flux_div_ensemble_mean_box'
fig1.savefig(dir_fig + figN1 + '.png', dpi=100, bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', dpi=100, bbox_inches='tight')

#-- cross section --
fig2 = plt.figure(figsize=(15, 5))
fig2.clf()
#
ax1 = fig2.add_subplot(1, 3, 1)
cs1 = ax1.contourf(ddist/1e3, -zgr.gdept_1d[0,:], mec_sec*scaleF, \
        levels=llev2, extend='both', cmap=cmap)
cs11 = ax1.contour(ddist/1e3, -zgr.gdept_1d[0,:], uvm_sec, \
        colors='k', levels=np.arange(0, 0.7, 0.1), alpha=0.5)
at1 = AnchoredText(r"$MEC$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
ax1.clabel(cs11, inline=False, fmt='%0.01f')
#
ax2 = fig2.add_subplot(1, 3, 2)
cs2 = ax2.contourf(ddist/1e3, -zgr.gdept_1d[0,:], eflx_sec*scaleF, \
        levels=llev2, extend='both', cmap=cmap)
cs11 = ax2.contour(ddist/1e3, -zgr.gdept_1d[0,:], uvm_sec, \
        colors='k', levels=np.arange(0, 0.6, 0.1), alpha=0.5)
cs21 = ax2.contour(ddist/1e3, -zgr.gdept_1d[0,:], mec_sec*scaleF, \
        colors='g', levels=np.arange(-3.0, 0.0, 0.5))
at1 = AnchoredText(r"$EDDYFLX$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
ax2.clabel(cs11, inline=False, fmt='%0.01f')
#
ax3 = fig2.add_subplot(1, 3, 3)
cs3 = ax3.contourf(ddist/1e3, -zgr.gdept_1d[0,:], (divef_sec)*scaleF, \
        levels=llev2, extend='both', cmap=cmap)
cs11 = ax3.contour(ddist/1e3, -zgr.gdept_1d[0,:], uvm_sec, \
        colors='k', levels=np.arange(0, 0.6, 0.1), alpha=0.5)
cs21 = ax3.contour(ddist/1e3, -zgr.gdept_1d[0,:], mec_sec*scaleF, \
        colors='g', levels=np.arange(-3.0, 0.0, 0.5))
at1 = AnchoredText(r"$DIVEF$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at1)
ax3.clabel(cs11, inline=False, fmt='%0.01f')
#
cbax1 = fig2.add_axes([0.92, 0.2, 0.01, 0.6])
cb1 = fig2.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'$\times10^{-3}$ [W m$^{-3}$]', fontsize='x-large')
#
for ip in range(3):
    ax = fig2.add_subplot(1, 3, ip+1)
    ax.set_ylim([-400, 0])
    ax.set_xlabel('Distance [km]', fontsize='large')
    if ip == 0:
        ax.set_ylabel('Depth [m]', fontsize='large')
    else:
        ax.set_yticklabels([])
#
figN2 = 'medwest60_mec_eflx_divef_cross_section'
fig2.savefig(dir_fig + figN2 + '.png', dpi=100, bbox_inches='tight')
fig2.savefig(dir_fig + figN2 + '.pdf', dpi=100, bbox_inches='tight')


#-- vertical profiles --
fig3 = plt.figure(figsize=(15, 5))
fig3.clf()
#
ax1 = fig3.add_subplot(1, 3, 1)
p0 = ax1.plot(mec_zprof * 1e-6, -zgr.gdept_1d[0, :], 'k')
p1 = ax1.plot(mec_uv_zprof * 1e-6, -zgr.gdept_1d[0, :], 'b')
p2 = ax1.plot(mec_w_zprof * 1e-6, -zgr.gdept_1d[0, :], 'r')
ax1.legend((p0[0], p1[0], p2[0]), ('$MEC$', '$MEC^{xy}$', '$MEC^{z}$'))
#
ax2 = fig3.add_subplot(1, 3, 2)
p0 = ax2.plot(eflx_zprof * 1e-6, -zgr.gdept_1d[0, :], 'k')
p1 = ax2.plot(eflx_uv_zprof * 1e-6, -zgr.gdept_1d[0, :], 'b')
p2 = ax2.plot(eflx_w_zprof * 1e-6, -zgr.gdept_1d[0, :], 'r')
ax2.legend((p0[0], p1[0], p2[0]), ('$EDDYFLX$', '$EDDYFLX^{xy}$', '$EDDYFLX^{z}$'))
#
ax3 = fig3.add_subplot(1, 3, 3)
p0 = ax3.plot(divef_zprof * 1e-6, -zgr.gdept_1d[0, :], 'k')
p1 = ax3.plot(divef_uv_zprof * 1e-6, -zgr.gdept_1d[0, :], 'b')
p2 = ax3.plot(divef_w_zprof * 1e-6, -zgr.gdept_1d[0, :], 'r')
ax3.legend((p0[0], p1[0], p2[0]), ('$DIVEF$', '$DIVEF^{xy}$', '$DIVEF^{z}$'))
#
for ip in range(3):
    ax = fig3.add_subplot(1, 3, (ip+1))
    #ax.plot([-3, 3], -np.ones(2)*zgr.gdept_1d[0, kkk_w].data, 'k--', alpha=0.5)
    ax.plot([-3, 3], -np.ones(2)*mldm[jjs:jjn, iiw:iie].mean().data, 'k--', alpha=0.5)
    ax.plot(np.zeros(len(kkk)), -zgr.gdept_1d[0, kkk], 'k', alpha=0.5)
    plt.grid()
    ax.set_xlabel('[MW]', fontsize='large')
    ax.set_xticks(np.arange(-3,4))
    ax.set_xlim([-3, 3])
    ax.set_ylim([-500, 0])
    if ip == 0:
        ax.set_ylabel('Depth [m]', fontsize='large')
    else:
        ax.set_yticklabels([])
#
figN3 = 'medwest60_mec_eflx_divef_vert_prof_box_mldm'
fig3.savefig(dir_fig + figN3 + '.png', dpi=100, bbox_inches='tight')
fig3.savefig(dir_fig + figN3 + '.pdf', dpi=100, bbox_inches='tight')


#-- mec, eflx, divef xy and z in surface layer -
llev1 = np.arange(-1, 1.1, 0.1)
scaleF = 1e3
scaleF2 = 1e4
kkk = 10

fig1 = plt.figure(figsize=(12, 10))
fig1.clf()
#
ax1 = fig1.add_subplot(2, 3, 1)
cs1 = ax1.contourf(xx, yy, mec_uv[kkk, jjs0:jjn0, iiw0:iie0] * msk * scaleF, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$MEC^{xy}$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
#
ax2 = fig1.add_subplot(2, 3, 2)
cs2 = ax2.contourf(xx, yy, eflx_uv[kkk, jjs0:jjn0, iiw0:iie0] * msk * scaleF, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$EDDYFLX^{xy}$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
#
ax3 = fig1.add_subplot(2, 3, 3)
cs2 = ax3.contourf(xx, yy, divef_uv[kkk, jjs0:jjn0, iiw0:iie0] * msk * scaleF, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$DIVEF^{xy}$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at1)
#
ax4 = fig1.add_subplot(2, 3, 4)
cs1 = ax4.contourf(xx, yy, mec_w[kkk, jjs0:jjn0, iiw0:iie0] * msk * scaleF, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$MEC^z$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax4.add_artist(at1)
#
ax5 = fig1.add_subplot(2, 3, 5)
cs2 = ax5.contourf(xx, yy, eflx_w[kkk, jjs0:jjn0, iiw0:iie0] * msk * scaleF, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$EDDYFLX^z$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax5.add_artist(at1)
#
ax6 = fig1.add_subplot(2, 3, 6)
cs2 = ax6.contourf(xx, yy, divef_w[kkk, jjs0:jjn0, iiw0:iie0] * msk * scaleF, \
        levels=llev1, extend='both', cmap=cmap)
at1 = AnchoredText(r"$DIVEF^z$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax6.add_artist(at1)
#
cbax1 = fig1.add_axes([0.91, 0.2, 0.01, 0.6])
cb1 = fig1.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'$\times$10$^{-3}$ [W m$^{-2}$]', fontsize='large')
#
for ip in range(6):
  ax = fig1.add_subplot(2, 3, ip+1)
  qv = ax.quiver(hgr.nav_lon[::10, ::10], hgr.nav_lat[::10, ::10], \
        uum.vozocrtx[ttt, 0, ::10, ::10], vvm.vomecrty[ttt, 0, ::10, ::10], \
        color='k', alpha=0.5)
  ax.set_facecolor([0.5, 0.5, 0.5])
  ax.set_ylim([36.7, 40.9])
  ax.set_xlim([4.1, 8.5])
  ax.set_xticks([5.0, 7.0])
  ax.set_yticks([38, 40])
  p = mp.patches.Rectangle((4.8, 37.2), 3, 3.25, linewidth=3, fill=False, color='g')
  ax.add_patch(p)
  ax.plot(xx_sec, yy_sec, 'k.')
  if ip == 0 or ip == 3:
    ax.set_yticklabels((r"38$^{\circ}$N", r"40$^{\circ}$N"), fontsize='x-large')
    plt.quiverkey(qv, 1.1, 0.5, 1, '[1 m/s]')
  else:
    ax.set_yticklabels(())
  if ip > 2:
    ax.set_xticklabels((r"5$^{\circ}$W", r"7$^{\circ}$W"), fontsize='x-large')
  else:
    ax.set_xticklabels(())
#
figN1 = 'medwest60_kk00_mec_efx_div_xy_z_box'
fig1.savefig(dir_fig + figN1 + '.png', dpi=100, bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', dpi=100, bbox_inches='tight')




#-- cross section W ONLY--
fig4 = plt.figure(figsize=(15, 5))
fig4.clf()
scaleF = 5e3
#
ax1 = fig4.add_subplot(1, 3, 1)
cs1 = ax1.contourf(ddist/1e3, -zgr.gdept_1d[0,:], mecw_sec*scaleF, \
        levels=llev2, extend='both', cmap=cmap)
cs11 = ax1.contour(ddist/1e3, -zgr.gdept_1d[0,:], uvm_sec, \
        colors='k', levels=np.arange(0, 0.7, 0.1), alpha=0.5)
ax1.plot(ddist/1e3, -mldm_sec, color='green')
at1 = AnchoredText(r"$MEC^{z}$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
ax1.clabel(cs11, inline=False, fmt='%0.01f')
#
ax2 = fig4.add_subplot(1, 3, 2)
cs2 = ax2.contourf(ddist/1e3, -zgr.gdept_1d[0,:], eflxw_sec*scaleF, \
        levels=llev2, extend='both', cmap=cmap)
cs11 = ax2.contour(ddist/1e3, -zgr.gdept_1d[0,:], uvm_sec, \
        colors='k', levels=np.arange(0, 0.6, 0.1), alpha=0.5)
ax2.plot(ddist/1e3, -mldm_sec, color='green')
at1 = AnchoredText(r"$EDDYFLX^{z}$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
ax2.clabel(cs11, inline=False, fmt='%0.01f')
#
ax3 = fig4.add_subplot(1, 3, 3)
cs3 = ax3.contourf(ddist/1e3, -zgr.gdept_1d[0,:], (divefw_sec)*scaleF, \
        levels=llev2, extend='both', cmap=cmap)
cs11 = ax3.contour(ddist/1e3, -zgr.gdept_1d[0,:], uvm_sec, \
        colors='k', levels=np.arange(0, 0.6, 0.1), alpha=0.5)
ax3.plot(ddist/1e3, -mldm_sec, color='green')
at1 = AnchoredText(r"$DIVEF^{z}$", prop=dict(size=15), frameon=True, loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at1)
ax3.clabel(cs11, inline=False, fmt='%0.01f')
#
cbax1 = fig4.add_axes([0.92, 0.2, 0.01, 0.6])
cb1 = fig4.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'$\times10^{-4}$ [W m$^{-3}$]', fontsize='x-large')
#
for ip in range(3):
    ax = fig4.add_subplot(1, 3, ip+1)
    ax.set_ylim([-400, 0])
    ax.set_xlabel('Distance [km]', fontsize='large')
    if ip == 0:
        ax.set_ylabel('Depth [m]', fontsize='large')
    else:
        ax.set_yticklabels([])
#
figN4 = 'medwest60_mec_eflx_divef_cross_section_wonly'
fig4.savefig(dir_fig + figN4 + '.png', dpi=100, bbox_inches='tight')
fig4.savefig(dir_fig + figN4 + '.pdf', dpi=100, bbox_inches='tight')

