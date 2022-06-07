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
dir_grd3= '/gpfsstore/rech/egi/uup63gs/medwest60_lc1/mesh/'
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
#
hgr3  = xr.open_dataset(dir_grd3 + 'mesh_hgr_LC1.nc')
zgr3  = xr.open_dataset(dir_grd3 + 'mesh_zgr_LC1.nc')
e12t_3= (hgr3.e1t*hgr3.e2t)[0, :, :]
#
#dt = 80.0
dt = 3600.0 #1h in sec
rau0 = 1026.0
nmem = 20
nt = 24
ttt = 1

#-- list of files --
ext = '_20100406-20100406.nc'
lSSH     = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridT-2D/0*gridT-2D'  + ext ) )
lDEKEDT  = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'DEKEDT/0*DEKEDT'      + ext ) )
lADVEKE  = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'ADVEKE_1d/0*ADV_EKE'  + ext ) )
lEDDYFLX = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'EDDYFLX_1d/0*EFLX_KE' + ext ) )
lMEC     = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'MEC_1d/0*MEC_KE'      + ext ) )
lWB      = Tcl().call('lsort', '-dict', glob.glob(dir_in  + 'WB/20100406/0*WB*' ) )
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
# -2 and -1 are for adjustment with iiw and iie
iiw3  = np.where(hgr3.nav_lon[0, :] >= 4.8)[0][0] - 2
iie3  = np.where(hgr3.nav_lon[0, :] >= 7.8)[0][0] - 1
jjs3  = np.where(hgr3.nav_lat[:, iiw3] >= 37.2)[0][0]
jjn3  = np.where(hgr3.nav_lat[:, iiw3] >= 40.45)[0][0]
[ny3, nx3] = [jjn3-jjs3, iie3-iiw3]



#----------------------------
# Vertical profile and hz map
#----------------------------
#- vert. profiles -
dteke_zprof   = np.zeros([nmem, nr])
advmeke_zprof = np.zeros([nmem, nr])
adveeke_zprof = np.zeros([nmem, nr])
eflx_zprof    = np.zeros([nmem, nr])
wb_zprof      = np.zeros([nmem, nr])
#- hz maps -
dteke_hz      = np.zeros([nmem, ny, nx])
advmeke_hz    = np.zeros([nmem, ny, nx])
adveeke_hz    = np.zeros([nmem, ny, nx])
eflx_hz       = np.zeros([nmem, ny, nx])
wb_hz         = np.zeros([nmem, ny3, nx3])
for imem in range(nmem):
  tmpssh  = xr.open_dataset(lSSH[imem])
  tmpdt   = xr.open_dataset(lDEKEDT[imem])
  tmpadvm = xr.open_dataset(lADVEKE[imem])
  tmpadve = xr.open_dataset(lMEC[imem])
  tmpeflx = xr.open_dataset(lEDDYFLX[imem])
  tmpwb   = xr.open_dataset(lWB[imem])
  for kkk in range(nr):
    print("memb %02i, level %03i" % (imem, kkk) )
    #- vert. profiles -
    dteke_zprof[imem, kkk] = \
            np.nansum( np.nansum( \
            ( tmpdt.dkedt[ttt, kkk, jjs:jjn, iiw:iie] ).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    advmeke_zprof[imem, kkk] = \
            np.nansum( np.nansum( \
            ( tmpadvm.advh_ke_pr[ttt, kkk, jjs:jjn, iiw:iie]  \
             +tmpadvm.advz_ke_pr[ttt, kkk, jjs:jjn, iiw:iie] ).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    adveeke_zprof[imem, kkk] = \
            np.nansum( np.nansum( \
            ( tmpadve.advh_ke_pr[ttt, kkk, jjs:jjn, iiw:iie]  \
             +tmpadve.advz_ke_pr[ttt, kkk, jjs:jjn, iiw:iie] ).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    eflx_zprof[imem, kkk] = \
            np.nansum( np.nansum( \
            ( tmpeflx.advh_ke_pr[ttt, kkk, jjs:jjn, iiw:iie]  \
             +tmpeflx.advz_ke_pr[ttt, kkk, jjs:jjn, iiw:iie] ).data \
            * e12t[jjs:jjn, iiw:iie].data, axis=-1 ), axis=-1)
    wb_zprof[imem, kkk] = \
            np.nansum( np.nansum( \
            ( tmpwb.wb[ttt, kkk, jjs3:jjn3, iiw3:iie3] ).data \
            * e12t_3[jjs3:jjn3, iiw3:iie3].data, axis=-1 ), axis=-1)
    #
    dteke_hz[imem, :, :] = dteke_hz[imem, :, :]   + \
            ( tmpdt.dkedt[ttt, kkk, :, :] ).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    advmeke_hz[imem, :, :] = advmeke_hz[imem, :, :]   + \
            ( tmpadvm.advh_ke_pr[ttt, kkk, :, :] + tmpadvm.advz_ke_pr[ttt, kkk, :, :] ).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    adveeke_hz[imem, :, :] = adveeke_hz[imem, :, :]   + \
            ( tmpadve.advh_ke_pr[ttt, kkk, :, :] + tmpadve.advz_ke_pr[ttt, kkk, :, :] ).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    eflx_hz[imem, :, :] = eflx_hz[imem, :, :]   + \
            ( tmpeflx.advh_ke_pr[ttt, kkk, :, :] + tmpeflx.advz_ke_pr[ttt, kkk, :, :] ).fillna(0) * \
            ( (zgr.e3t_0[0, kkk, :, :] * \
            (1 + tmpssh.sossheig[ttt, :, :]/bathy.gdepw_0[0, :, :])) ).fillna(0)
    # variable e3t not condisered here ...
    wb_hz[imem, :, :] = wb_hz[imem, :, :]   + \
            ( tmpwb.wb[ttt, kkk, jjs3:jjn3, iiw3:iie3] ).fillna(0) * \
            ( (zgr3.e3t_0[0, kkk, jjs3:jjn3, iiw3:iie3] ) ).fillna(0)


#-- save --
tmp1 = np.zeros([4, nmem, nr])
tmp1[0, :, :] = dteke_zprof
tmp1[1, :, :] = advmeke_zprof 
tmp1[2, :, :] = adveeke_zprof
tmp1[3, :, :] = eflx_zprof
tmp2 = np.zeros([4, nmem, ny, nx])
tmp2[0, :, :, :] = dteke_hz
tmp2[1, :, :, :] = advmeke_hz
tmp2[2, :, :, :] = adveeke_hz
tmp2[3, :, :, :] = eflx_hz
#
fileN1 = 'dteke_advmeke_adveeke_eddyflx_zprof_box.bin'
f = open(dir_out + fileN1, 'wb')
tmp1.reshape([4*nmem*nr]).astype('>f4').tofile(f)
f.close()
fileN2 = 'dteke_advmeke_adveeke_eddyflx_hz_map_basin.bin'
f = open(dir_out + fileN2, 'wb')
tmp2.reshape([4*nmem*ny*nx]).astype('>f4').tofile(f)
f.close()
#
fileN3 = 'wb_zprof_box.bin'
f = open(dir_out + fileN3, 'wb')
wb_zprof.reshape([nmem*nr]).astype('>f4').tofile(f)
f.close()
fileN4 = 'wb_hz_map_box.bin'
f = open(dir_out + fileN4, 'wb')
wb_hz.reshape([nmem*ny3*nx3]).astype('>f4').tofile(f)
f.close()


#-- load --
fileN1 = 'dteke_advmeke_adveeke_eddyflx_zprof_box.bin'
f = open(dir_out + fileN1, 'r')
tmp1 = np.fromfile(f, '>f4').reshape([4, nmem, nr])
dteke_zprof   = tmp1[0, :, :]
advmeke_zprof = tmp1[1, :, :]
adveeke_zprof = tmp1[2, :, :]
eflx_zprof    = tmp1[3, :, :]
f.close()
fileN2 = 'dteke_advmeke_adveeke_eddyflx_hz_map_basin.bin'
f = open(dir_out + fileN2, 'r')
tmp2 = np.fromfile(f, '>f4').reshape([4, nmem, ny, nx])
dteke_hz   = tmp2[0, :, :, :]
advmeke_hz = tmp2[1, :, :, :]
adveeke_hz = tmp2[2, :, :, :]
eflx_hz    = tmp2[3, :, :, :]
f.close()
#
fileN3 = 'wb_zprof_box.bin'
f = open(dir_out + fileN3, 'r')
wb_zprof = np.fromfile(f, '>f4').reshape([nmem, nr])
f.close()
fileN4 = 'wb_hz_map_box.bin'
f = open(dir_out + fileN4, 'r')
wb_hz = np.fromfile(f, '>f4').reshape([nmem, ny3, nx3])
f.close()


#-- box vol int --
dteke_volint = np.nansum( np.nansum( \
        dteke_hz[:, jjs:jjn, iiw:iie] * e12t.data[np.newaxis, jjs:jjn, iiw:iie], axis=-1), axis=-1)
advmeke_volint = np.nansum( np.nansum( \
        advmeke_hz[:, jjs:jjn, iiw:iie] * e12t.data[np.newaxis, jjs:jjn, iiw:iie], axis=-1), axis=-1)
adveeke_volint = np.nansum( np.nansum( \
        adveeke_hz[:, jjs:jjn, iiw:iie] * e12t.data[np.newaxis, jjs:jjn, iiw:iie], axis=-1), axis=-1)
eflx_volint = np.nansum( np.nansum( \
        eflx_hz[:, jjs:jjn, iiw:iie] * e12t.data[np.newaxis, jjs:jjn, iiw:iie], axis=-1), axis=-1)
wb_volint = np.nansum( np.nansum( \
        wb_hz[:, :, :] * e12t.data[np.newaxis, jjs:jjn, iiw:iie], axis=-1), axis=-1)
res_volint = dteke_volint.mean() - advmeke_volint.mean() - adveeke_volint.mean() \
        - eflx_volint.mean()

#============================================
#               PLOT
#============================================
#-- vertical profiles --
fig1 = plt.figure(figsize=(5, 6))
fig1.clf()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_position([0.17, 0.11, 0.78, 0.88])
p5 = ax1.plot( wb_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'k', alpha=0.3, linewidth=2)
p4 = ax1.plot( (dteke_zprof.mean(0) - advmeke_zprof.mean(0) - adveeke_zprof.mean(0) \
        - eflx_zprof.mean(0)) / 1e9, -zgr.gdept_1d[0, :], 'g', linewidth=1)
p1 = ax1.plot( (advmeke_zprof.mean(0)+adveeke_zprof.mean(0))  / 1e9, \
        -zgr.gdept_1d[0, :], 'b', linewidth=3)
p0 = ax1.plot(dteke_zprof.mean(0)  / 1e9, -zgr.gdept_1d[0, :], 'k', linewidth=3)
p3 = ax1.plot(eflx_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'r', linewidth=3)
plt.legend((p0[0], p1[0], p3[0], p4[0], p5[0]), \
        (r"$\partial_t \left< K^{*} \right>$", \
        r"$-\nabla \cdot \left< \mathbf{u} K^{*} \right>$", \
	r"$-\rho_0 \left< \mathbf{u}' u_h' \right> \cdot \nabla \left< u_h \right>$", \
        r"$Residual$", \
        r"$-\left< w'b' \right>$"), \
        fontsize='x-large')
plt.grid()
ax1.set_ylim([-600, 0])
ax1.set_xticks(np.arange(-0.01, 0.015, 0.0025))
ax1.set_xlim([-0.005, 0.005])
ax1.set_ylabel('Depth [m]', fontsize='x-large')
ax1.set_xlabel(r'[GW m$^{-1}$]', fontsize='x-large')
#
figN1 = 'eke_bgt_dekedt_advmke_adveeke_eddyfflxmec_box_zprof'
fig1.savefig(dir_fig + figN1 + '.png', bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', bbox_inches='tight')
plt.close(fig1)

#-- check amplitude of w'b' --
plt.figure();plt.contourf(wb_hz.mean(axis=0), 20);plt.colorbar()
plt.figure();plt.plot(wb_zprof.mean(axis=0)/1e9, -zgr.gdept_1d[0, :])
plt.ylim([-500, 0])

#--
fig2 = plt.figure(figsize=(12, 9))
fig2.clf()
llev = np.arange(-1.0, 1.1, 0.1)*1e-1
cmap = 'RdBu_r'
#
ax0 = fig2.add_subplot(2, 2, 1)
cs0 = ax0.contourf(hgr.nav_lon, hgr.nav_lat, \
        dteke_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$\partial_t \left< K^{*} \right>$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (dteke_volint.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(at1)
#
ax1 = fig2.add_subplot(2, 2, 2)
cs1 = ax1.contourf(hgr.nav_lon, hgr.nav_lat, \
	advmeke_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$-\nabla \cdot \left< \mathbf{u} \right> \left< K^{*} \right> $', \
        prop=dict(size=15), frameon=True, loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (advmeke_volint.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
#
ax2 = fig2.add_subplot(2, 2, 3)
cs2 = ax2.contourf(hgr.nav_lon, hgr.nav_lat, \
        adveeke_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r"$-\nabla \cdot \left< \mathbf{u}' K^{*} \right> $", \
        prop=dict(size=15), frameon=True, loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (adveeke_volint.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
#
ax3 = fig2.add_subplot(2, 2, 4)
cs3 = ax3.contourf(hgr.nav_lon, hgr.nav_lat, \
        eflx_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r"$-\rho_0 \left< \mathbf{u}' u_h' \right> \nabla \cdot \left< u_h \right>$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (eflx_volint.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at1)
#
for ip in range(4):
  ax = fig2.add_subplot(2, 2, ip+1)
  ax.set_ylim([36.5, 41.2])
  ax.set_xlim([3, 9])
  ax.set_facecolor([0.5, 0.5, 0.5])
  ax.set_yticks([37, 40])
  ax.set_xticks([4, 8])
  if ip == 2 or ip == 3:
    ax.set_xticklabels(['4$^{\circ}$E', r'8$^{\circ}$E'])
  else:
    ax.set_xticklabels(['', ''])
  if ip == 0 or ip == 2:
    ax.set_yticklabels([r'37$^{\circ}$N', r'40$^{\circ}$N'])
  else:
    ax.set_yticklabels(['', ''])
  p = mp.patches.Rectangle((4.8, 37.2), 3, 3.25, linewidth=3, fill=False, color='g')
  ax.add_patch(p)
#-- colorbar --
cbax1 = fig2.add_axes([0.92, 0.2, 0.01, 0.6])
cb1 = fig2.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'[W m$^{-2}$]')
#
figN2 = 'eke_bgt_dekedt_advmke_adveeke_eddyfflxmec_box_hz_map'
fig2.savefig(dir_fig + figN2 + '.png', bbox_inches='tight')
fig2.savefig(dir_fig + figN2 + '.pdf', bbox_inches='tight')
plt.close(fig2)



#-- all together --
fig3 = plt.figure(figsize=(10, 10))
fig3.clf()
llev = np.arange(-1.0, 1.1, 0.1)*1e-1
cmap = 'RdBu_r'
#
ax1 = fig3.add_subplot(2, 2, 1)
cs1 = ax1.contourf(hgr.nav_lon, hgr.nav_lat, \
        dteke_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$\partial_t \left< K^{*} \right>$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (dteke_volint.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
#
ax2 = fig3.add_subplot(2, 2, 2)
cs1 = ax2.contourf(hgr.nav_lon, hgr.nav_lat, \
        (advmeke_hz.mean(0)+adveeke_hz.mean(0)) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r'$-\nabla \cdot \left< \mathbf{u} K^{*} \right> $', \
        prop=dict(size=15), frameon=True, loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % \
        ((advmeke_volint.mean()+adveeke_volint.mean())/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at1)
#
ax3 = fig3.add_subplot(2, 2, 3)
cs3 = ax3.contourf(hgr.nav_lon, hgr.nav_lat, \
        eflx_hz.mean(0) * mskNaN, levels=llev, extend='both', cmap=cmap)
at0 = AnchoredText(r"$-\rho_0 \left< \mathbf{u}' \otimes \mathbf{u}_h' \right> \cdot \nabla \left< \mathbf{u}_h \right>$", \
        prop=dict(size=15), frameon=True, loc='upper left')
at0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at0)
at1 = AnchoredText(r"$\int \cdot dV = %0.02f$ GW" % (eflx_volint.mean()/1e9), \
        prop=dict(size=10), frameon=True, loc='lower right')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at1)
#
ax4 = fig3.add_subplot(2, 2, 4)
#ax3.set_position([0.17, 0.11, 0.78, 0.88])
p5 = ax4.plot( wb_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'k', alpha=0.3, linewidth=2)
p4 = ax4.plot( (dteke_zprof.mean(0) - advmeke_zprof.mean(0) - adveeke_zprof.mean(0) \
        - eflx_zprof.mean(0)) / 1e9, -zgr.gdept_1d[0, :], 'g', linewidth=1)
p1 = ax4.plot( (advmeke_zprof.mean(0)+adveeke_zprof.mean(0))  / 1e9, \
        -zgr.gdept_1d[0, :], 'b', linewidth=3)
p0 = ax4.plot(dteke_zprof.mean(0)  / 1e9, -zgr.gdept_1d[0, :], 'k', linewidth=3)
p3 = ax4.plot(eflx_zprof.mean(0) / 1e9, -zgr.gdept_1d[0, :], 'r', linewidth=3)
plt.legend((p0[0], p1[0], p3[0], p5[0], p4[0]), \
        (r"$\partial_t \left< K^{*} \right>$", \
        r"$-\nabla \cdot \left< \mathbf{u} K^{*} \right>$", \
        r"$-\rho_0 \left< \mathbf{u}' \otimes \mathbf{u}_h' \right> \cdot \nabla \left< \mathbf{u}_h \right>$", \
        r"$-\left< w'b' \right>$", \
        r"$Residual$"), \
        fontsize='x-large')
plt.grid()
ax4.set_ylim([-500, 0])
ax4.set_xticks(np.arange(-0.01, 0.015, 0.0025))
ax4.set_xlim([-0.005, 0.005])
ax4.set_ylabel('Depth [m]', fontsize='large')
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
figN3 = 'eke_bgt_dekedt_advmke_adveeke_eddyfflxmec_wb_box_hz_map_zprof'
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
    


