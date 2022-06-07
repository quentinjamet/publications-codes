import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import xarray as xr
import glob
from tkinter import Tcl
from matplotlib.offsetbox import AnchoredText

plt.ion()

#-- directories --
dir_in   = '/gpfsstore/rech/egi/uup63gs/medwest60/outputs/'
dir_grd  = '/gpfsstore/rech/egi/uup63gs/medwest60_lc1/mesh/'
dir_fig  = '/linkhome/rech/genige01/uup63gs/Figures/energetics/'

#-- mesh and mask --
msk   = xr.open_dataset(dir_grd + 'mask_LC1.nc')
hgr   = xr.open_dataset(dir_grd + 'mesh_hgr_LC1.nc')
zgr   = xr.open_dataset(dir_grd + 'mesh_zgr_LC1.nc')
bathy = xr.open_dataset(dir_grd + 'bathy_gdepw_0_LC1.nc')
e12t = (hgr.e1t * hgr.e2t).data
mskNaN = msk.tmaskutil[0, :, :].data.astype('float')
mskNaN[np.where(mskNaN>0.0)] = 1.0
mskNaN[np.where(mskNaN==0.0)] = np.nan
[nr, ny, nx] = [ zgr.dims['z'], zgr.dims['y'], zgr.dims['x'] ]
period = np.arange(20100206, 20100229)
period = np.concatenate( (period, np.arange(20100301, 20100332)), axis=0)
period = np.concatenate( (period, np.arange(20100401, 20100431)), axis=0)   # missing day 30
period = np.concatenate( (period, np.arange(20100501, 20100532)), axis=0)
period = np.concatenate( (period, np.arange(20100601, 20100606)), axis=0)
nper = len(period)
nhr = 24
nmem= 20
ttime   = np.arange(0, nper, 1/nhr)

#-- focus on eddying region --
iiw0  = np.where(hgr.nav_lon[200, :] >= 4.8)[0][0]
iie0  = np.where(hgr.nav_lon[200, :] >= 7.8)[0][0]
jjs0  = np.where(hgr.nav_lat[:, iiw0] >= 37.2)[0][0]
jjn0  = np.where(hgr.nav_lat[:, iiw0] >= 40.45)[0][0]
[ny0, nx0] = [jjn0-jjs0, iie0-iiw0]



#----------------------------------------
#   time series, pos/neg contrib
#----------------------------------------
#- int over 60 days : MEC = ~-1.0e15 J ; EDDYFLX = ~ 1.16e15 J
#   MEC neg contrib : -4.1e15 J
#   MEC pos contrib : +3.0e15 J
#   EFLX neg contrib: -1.1e15 J
#   EFLX pos contrib: +2.2e15 J
#- int over 120 days : MEC = ~-2.12e15 J ; EDDYFLX = ~ 2.41e15 J ; WB = ~+1.38e15 J

ts   = np.zeros([4, nper*nhr])
tint_plus = np.zeros([2, nper*nhr])
tint_minus = np.zeros([2, nper*nhr])
for iper in range(nper):
  print("--- period: %s" % (period[iper]) )
  #-- load pre-extracted mec and eddyflx hz maps --
  f = open( str("%s/mec_eflx/mec_eddyflx_box_%s.bin" % (dir_in, period[iper])), 'r')
  tmp2 = np.fromfile(f,'>f4').reshape([2, nmem, nhr, ny, nx])
  f.close()
  mec_hz = tmp2[0, :, :, :, :].mean(0)
  eddyflx_hz = tmp2[1, :, :, :, :].mean(0)
  #-- load pre-extracted wb hz maps --
  f = open( str("%s/wb/wb_eddy_eddy_box_%s.bin" % (dir_in, period[iper])), 'r')
  tmp2 = np.fromfile(f,'>f4').reshape([nmem, nhr, ny, nx])
  f.close()
  wb_hz = tmp2.mean(0)
  #-- integration --
  ts[0, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( mec_hz[:, jjs0:jjn0, iiw0:iie0] * \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)
  ts[1, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( eddyflx_hz[:, jjs0:jjn0, iiw0:iie0] * \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)
  ts[2, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( (mec_hz[:, jjs0:jjn0, iiw0:iie0] + eddyflx_hz[:, jjs0:jjn0, iiw0:iie0] )* \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)
  ts[3, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( wb_hz[:, jjs0:jjn0, iiw0:iie0] * \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)
  #-- negative contributions --
  tmpmec = mec_hz * 1.0
  tmpmec[tmpmec > 0.0 ] = 0.0
  tmpeflx = eddyflx_hz * 1.0
  tmpeflx[tmpeflx > 0.0 ] = 0.0
  tint_minus[0, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( tmpmec[:, jjs0:jjn0, iiw0:iie0] * \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)
  tint_minus[1, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( tmpeflx[:, jjs0:jjn0, iiw0:iie0] * \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)
  #-- positive contributions --
  tmpmec = mec_hz * 1.0
  tmpmec[tmpmec < 0.0 ] = 0.0
  tmpeflx = eddyflx_hz * 1.0
  tmpeflx[tmpeflx < 0.0 ] = 0.0
  tint_plus[0, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( tmpmec[:, jjs0:jjn0, iiw0:iie0] * \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)
  tint_plus[1, nhr*iper:nhr*(iper+1)] = \
          np.nansum( np.nansum( tmpeflx[:, jjs0:jjn0, iiw0:iie0] * \
          e12t[:, jjs0:jjn0, iiw0:iie0], axis=-1), axis=-1)

#-- plot --
fig1 = plt.figure(figsize=(12, 4))
fig1.clf()
#
ax1 = fig1.add_subplot(1, 2, 1)
ax1.plot(ttime, np.zeros(nper*nhr), 'k', alpha=0.5)
p10 = ax1.plot(ttime, ts[3, :]/1e9, 'k', linewidth=1, alpha=0.3)
p00 = ax1.plot(ttime, ts[0, :]/1e9, 'g', linewidth=1)
#p10 = ax1.plot(ttime, (tint_minus[0, :]*3600).cumsum()/1e15, 'g--', linewidth=1)
p01 = ax1.plot(ttime, ts[1, :]/1e9, 'r', linewidth=1)
#p11 = ax1.plot(ttime, (tint_plus[0, :]*3600).cumsum()/1e15, 'r--', linewidth=1)
p02 = ax1.plot(ttime, ts[2, :]/1e9, 'b', linewidth=1, alpha=0.5)
ax1.legend((p10[0], p00[0], p01[0], p02[0]), \
        (r"$-\left<w'b'\right>$", r'$MEC$', r'$EDDYFLX$', r'$DIVEF$'))
#
ax2 = fig1.add_subplot(1, 2, 2)
ax2.plot(ttime, np.zeros(nper*nhr), 'k', alpha=0.5)
p10 = ax2.plot(ttime, (ts[3, :]*3600).cumsum()/1e15, 'k', alpha=0.3)
p00 = ax2.plot(ttime, (ts[0, :]*3600).cumsum()/1e15, 'g')
#p10 = ax1.plot(ttime, (tint_minus[0, :]*3600).cumsum()/1e15, 'g--', linewidth=1)
p01 = ax2.plot(ttime, (ts[1, :]*3600).cumsum()/1e15, 'r')
#p11 = ax1.plot(ttime, (tint_plus[0, :]*3600).cumsum()/1e15, 'r--', linewidth=1)
p02 = ax2.plot(ttime, (ts[2, :]*3600).cumsum()/1e15, 'b', alpha=0.5)
ax2.legend((p10[0], p00[0], p01[0], p02[0]), \
        (r"$-\left<w'b'\right>$", r'$MEC$', r'$EDDYFLX$', r'$DIVEF$'))
for ip in range(2):
    ax = fig1.add_subplot(1, 2, (ip+1))
    ax.set_xticks(np.arange(0, 150, 30))
    ax.set_xlabel('Time [days]', fontsize='large')
    ax.set_xlim([0, 120])
    if ip == 0:
        ax.set_ylabel(r'[GJ s$^{-1}$]', fontsize='large')
        #ax.set_yticks(np.arange(-0.4, 0.6, 0.2))
        #ax.set_ylim([-0.6, 0.6])
        ax.set_yticks(np.arange(-0.9, 1.2, 0.3))
        ax.set_ylim([-1.0, 1.0])
    else:
        ax.set_ylabel('[PJ]', fontsize='large')
        ax.set_yticks(np.arange(-2, 3, 1))
        ax.set_ylim([-2.5, 2.5])
    plt.grid()
#
figN = 'MEC_EDDYFLX_WB_box_ts_tint'
fig1.savefig(dir_fig + figN + '.png', dpi=100, bbox_inches='tight')
fig1.savefig(dir_fig + figN + '.pdf', dpi=100, bbox_inches='tight')
plt.close(fig1)


#----------------------------------------------------
# Contribution to the EKE and MKE time rate of change 
# 1/ in the box
#   EKE production : +0.72e15 J (60 days), +0.98e15 J (120 days)
#   MKE destruction: -0.44e15 J (60 days), -0.91e15 J (120 days)
# 2/ for the full domain
#   EKE production : +1.89e15 J (60 days), +2.27e15 J (120 days)
#   MKE destruction: (2.46e15 - 4.12e15) = -1.67e15 J (60 days)
#                    (1.51e15 - 4.12e15) = -2.61e15 J (120 days)
#----------------------------------------------------
dir_in2  = '/gpfsstore/rech/egi/uup63gs/medwest60/MEDWEST60-GSL19-S/ens01/1h/'
dir_grd2 = '/gpfsstore/rech/egi/commun/MEDWEST60/MEDWEST60-I/'
hgr2 = xr.open_dataset(dir_grd2 + 'MEDWEST60_mesh_hgr.nc4')
zgr2 = xr.open_dataset(dir_grd2 + 'MEDWEST60_mesh_zgr.nc4')
e123t = (hgr2.e1t * hgr2.e2t).data * zgr2.e3t_0[0, :, :, :].data
[nr, ny, nx] = [ zgr2.dims['z'], zgr2.dims['y'], zgr2.dims['x'] ]
#-- list of files --
listU = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridU/0*.nc') )
listU = np.reshape(listU, (nmem, nper))
listV = Tcl().call('lsort', '-dict', glob.glob(dir_in2 + 'gridV/0*.nc') )
listV = np.reshape(listV, (nmem, nper))

#-- define a region --
iiw  = np.where(hgr2.nav_lon[200, :] >= 4.8)[0][0]
iie  = np.where(hgr2.nav_lon[200, :] >= 7.8)[0][0]
jjs  = np.where(hgr2.nav_lat[:, iiw] >= 37.2)[0][0]
jjn  = np.where(hgr2.nav_lat[:, iiw] >= 40.45)[0][0]
[ny2, nx2] = [jjn-jjs, iie-iiw]

#-- compute MKE/EKE --
# CAUTION to compute time integration consistent with model time stepping:
# with AB (MITgcm): int_t (u \partial_t u) deltaT 
#                               = \sum( (u^n+1 + u^n)/2 * (u^n+1-u^n)/deltaT ) * deltaT
#                               = 1/2 * (u^N+1*u^N+1 - u^1*u^1)
#   same for v, and interpolation at tracer point (bringing an additional 1/2)
# with leap-frog (NEMO): int_t (u\partial_t u) deltaT = 
#                               = sum( u^n * (u^n+1 - u^n-1)/2*deltaT ) * deltaT
#                               = 1/2 * (u^N*u^N+1 - u^1*u^0)
#   same for v, and interpolation at tracer point (bringing an additional 1/2)

rho0 = 1026.0   #volumic mass of reference     [kg/m3]
#- MKE at t=0 (approx as KE of one member) -
tmpu  = xr.open_dataset(listU[0][0]).vozocrtx[0, :, :, :].data
tmpv  = xr.open_dataset(listV[0][0]).vomecrty[0, :, :, :].data
mke0 = np.zeros([nr, ny, nx])
mke0[:, 1:, 1:] = 0.25 * rho0 * ( \
        tmpu[:, 1:, 1:]**2 + tmpu[:, 1:, :-1]**2 + tmpv[:, 1:, 1:]**2 + tmpv[:, :-1, 1:] **2 )
#- MKE at t=60d -
tmpum = np.zeros([nr, ny, nx])
tmpvm = np.zeros([nr, ny, nx])
for imem in range(nmem):
    print("---  MKE, memb%03i  ---" % imem)
    tmpu = xr.open_dataset(listU[imem][-1])
    tmpv = xr.open_dataset(listV[imem][-1])
    tmpum = tmpum + tmpu.vozocrtx[-1, :, :, :].data
    tmpvm = tmpvm + tmpv.vomecrty[-1, :, :, :].data
#
tmpum = tmpum / float(nmem)
tmpvm = tmpvm / float(nmem)
mke60 = np.zeros([nr, ny, nx])
mke60[:, 1:, 1:] = 0.25 * rho0 * ( \
        tmpum[:, 1:, 1:]**2 + tmpum[:, 1:, :-1]**2 + tmpvm[:, 1:, 1:]**2 + tmpvm[:, :-1, 1:] **2 )
#- EKE at t=60d -
eke60 = np.zeros([nr, ny, nx])
for imem in range(nmem):
    print("---  EKE, memb%03i  ---" % imem)
    tmpu = xr.open_dataset(listU[imem][-1])
    tmpv = xr.open_dataset(listV[imem][-1])
    #
    eke60[:, 1:, 1:] = eke60[:, 1:, 1:] + 0.25 * rho0 * \
            ( (tmpu.vozocrtx[-1, :, 1: , 1: ].data - tmpum[:, 1:, 1:] )**2 \
             +(tmpu.vozocrtx[-1, :, 1: , :-1].data - tmpum[:, 1:, :-1])**2 \
            + (tmpv.vomecrty[-1, :, 1: , 1: ].data - tmpvm[:, 1:, 1:] )**2 \
             +(tmpv.vomecrty[-1, :, :-1, 1: ].data - tmpvm[:, :-1, 1:])**2 \
            )
eke60 = eke60 / float(nmem)

#-- volume integration --
#- basin -
mke0int  = np.nansum( mke0  * e123t )
mke60int = np.nansum( mke60 * e123t )
eke60int = np.nansum( eke60 * e123t )
#- box -
mke0int_box  = np.nansum( mke0 [:, jjs:jjn, iiw:iie] * e123t[:, jjs:jjn, iiw:iie] )
mke60int_box = np.nansum( mke60[:, jjs:jjn, iiw:iie] * e123t[:, jjs:jjn, iiw:iie] )
eke60int_box = np.nansum( eke60[:, jjs:jjn, iiw:iie] * e123t[:, jjs:jjn, iiw:iie] )
