import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import xarray as xr
import glob
from tkinter import Tcl
from matplotlib.offsetbox import AnchoredText

plt.ion()

#-- directories --
#dir_in = '/gpfsscratch/rech/egi/uup63gs/cdftools/'
#dir_in2= '/gpfsstore/rech/egi/uup63gs/medwest60/MEDWEST60-GSL19-S/ens01/1h/'
dir_in3 = '/gpfsstore/rech/egi/uup63gs/medwest60/outputs/'
dir_grd = '/gpfsstore/rech/egi/uup63gs/medwest60/mesh/'
dir_grd2= '/gpfsstore/rech/egi/uup63gs/medwest60_lc1/mesh/'
dir_fig = '/linkhome/rech/genige01/uup63gs/Figures/energetics/'
dir_out = '/gpfsstore/rech/egi/uup63gs/medwest60/data/'


#-- mesh and mask --
msk   = xr.open_dataset(dir_grd + 'mask.nc')
msk2  = xr.open_dataset(dir_grd2+ 'mask_LC1.nc')
hgr   = xr.open_dataset(dir_grd + 'mesh_hgr.nc')
hgr2  = xr.open_dataset(dir_grd2+ 'mesh_hgr_LC1.nc')
zgr   = xr.open_dataset(dir_grd + 'mesh_zgr.nc')
bathy = xr.open_dataset(dir_grd + 'bathy.nc')
mskNaN = msk.tmaskutil[0, :, :].data.astype('float')
mskNaN[np.where(mskNaN>0.0)] = 1.0
mskNaN[np.where(mskNaN==0.0)] = np.nan
mskNaN2= msk2.tmaskutil[0, :, :].data.astype('float')
mskNaN2[np.where(mskNaN2>0.0)] = 1.0
mskNaN2[np.where(mskNaN2==0.0)] = np.nan
e12t   = (hgr.e1t * hgr.e2t)[0, :, :].data
e12t_2 = (hgr2.e1t * hgr2.e2t)[0, :, :].data
[nr, ny, nx] = [ zgr.dims['z'], zgr.dims['y'], zgr.dims['x'] ]
[ny2, nx2]   = [ hgr2.dims['y'], hgr2.dims['x'] ]
ttt = 1
nmem = 20

#-- load pre-extracted mec and eddyflx hz maps --
fileN2 = 'mec_eddyflx_hzmap_basin.bin'
f = open(dir_in3 + fileN2, 'r')
tmp2 = np.fromfile(f,'>f4').reshape([3, nmem, ny, nx])
f.close()


#-----------------------------------------------------
#               Coarse graining 
#-----------------------------------------------------
#cs_fac_list = [1, 2, 3, 5, 10, 20, 30, 50, 60, 120, 200, 400]
#[ny0, nx0] = [2**9, 2**9]
#cs_fac_list = 2**np.arange(9)
cs_fac_list = 3**np.arange(7)
[ny0, nx0] = [cs_fac_list[-1], cs_fac_list[-1]]
ncs = len(cs_fac_list)
balance  = np.zeros([4, ncs])
balance2 = np.zeros([4, ncs])
for ireg in range(4):
 print("Region %02i" % ireg )
 if ireg == 0:
   [nyoff, nxoff] = [0, 0]              #lower left corner
 elif ireg == 1:
   [nyoff, nxoff] = [ny-ny0, 0]         #upper left corner
 elif ireg == 2:
   [nyoff, nxoff] = [0, nx-nx0]         #lower righy corner
 elif ireg == 3:
   [nyoff, nxoff] = [ny-ny0, nx-nx0]    #upper righy corner
 mec_hz_basin  = tmp2[0, :, nyoff:nyoff+ny0, nxoff:nxoff+nx0].mean(0)
 eflx_hz_basin = tmp2[1, :, nyoff:nyoff+ny0, nxoff:nxoff+nx0].mean(0)
 divef_hz_basin = tmp2[2, :, nyoff:nyoff+ny0, nxoff:nxoff+nx0].mean(0)
 for ics in range(ncs-1):
    cs_fac = cs_fac_list[ics]
    print("cs_fac: %03i" % cs_fac)
    [ny_cs, nx_cs] = [int(np.floor(ny0/cs_fac)), int(np.floor(nx0/cs_fac))]
    tmpmec_cs  = np.zeros([ny_cs, nx_cs])
    tmpeflx_cs = np.zeros([ny_cs, nx_cs])
    tmpdivef_cs = np.zeros([ny_cs, nx_cs])
    for ii in range(nx_cs):
        for jj in range(ny_cs):
            iii = ii*cs_fac
            jjj = jj*cs_fac
            tmpmec_cs[jj, ii] = np.nansum(mec_hz_basin[jjj:jjj+cs_fac, iii:iii+cs_fac]   * \
                    e12t[jjj:jjj+cs_fac, iii:iii+cs_fac] )
            tmpeflx_cs[jj, ii] = np.nansum(eflx_hz_basin[jjj:jjj+cs_fac, iii:iii+cs_fac] * \
                    e12t[jjj:jjj+cs_fac, iii:iii+cs_fac] ) 
            tmpdivef_cs[jj, ii] = np.nansum(divef_hz_basin[jjj:jjj+cs_fac, iii:iii+cs_fac]   * \
                    e12t[jjj:jjj+cs_fac, iii:iii+cs_fac] ) / \
                    np.nansum(e12t[jjj:jjj+cs_fac, iii:iii+cs_fac])
    #
    tmpmsk = np.ones([ny_cs, nx_cs])
    tmpmsk[np.where(np.isnan(tmpmec_cs))] = np.nan
    #- construct vector for covariance, removing land points -
    wetpts = int(np.nansum(tmpmsk))
    ppp = np.zeros([2, wetpts])
    ppp2 = np.zeros([wetpts])
    ij = 0 
    for jj in range(ny_cs):
        for ii in range(nx_cs):
            if not np.isnan(tmpmsk[jj, ii]):
                ppp[0, ij] = tmpmec_cs[jj, ii]
                ppp[1, ij] = tmpeflx_cs[jj, ii]
                ppp2[ij]   = tmpdivef_cs[jj, ii]
                ij = ij + 1 
    # compute covariance matrix
    #ccov = np.cov(ppp)
    #balance[ireg, ics] = ccov[0, 1] / ccov[0, 0]
    ccor = np.corrcoef(ppp)
    balance[ireg, ics] = ccor[0, 1]
    vvar = np.nanstd(ppp2, ddof=1)
    balance2[ireg, ics] = vvar


fileN = 'cov_mec_eddyflx_coarse_as_box_size_pw3.bin'
f = open(dir_out + fileN, 'wb')
balance.astype('>f4').tofile(f)
f.close()

#-- eflx mec correl function of time --
period = np.arange(20100206, 20100229)
period = np.concatenate( (period, np.arange(20100301, 20100332)), axis=0)
period = np.concatenate( (period, np.arange(20100401, 20100431)), axis=0)   # missing day 30
period = np.concatenate( (period, np.arange(20100501, 20100532)), axis=0)
period = np.concatenate( (period, np.arange(20100601, 20100606)), axis=0)
nper = len(period)
nhr = 24
nmem= 20
ttime   = np.arange(0, nper, 1/nhr)

cs_fac_list = 3**np.arange(6)
[ny0, nx0] = [cs_fac_list[-1], cs_fac_list[-1]]
ncs = len(cs_fac_list)
balance3 = np.zeros([nper, nhr, 4, ncs])
for iper in range(nper):
  print("--- period: %s" % (period[iper]) )
  #-- load pre-extracted mec and eddyflx hz maps --
  f = open( str("%s/mec_eflx/mec_eddyflx_box_%s.bin" % (dir_in3, period[iper])), 'r')
  tmp2 = np.fromfile(f,'>f4').reshape([2, nmem, nhr, ny2, nx2])
  f.close()
  #mec_hz = tmp2[0, :, :, :, :].mean(0)
  #eddyflx_hz = tmp2[1, :, :, :, :].mean(0)
  for ireg in range(4):
   print("Region %02i" % ireg )
   if ireg == 0:
     [nyoff, nxoff] = [0, 0]              #lower left corner
   elif ireg == 1:
     [nyoff, nxoff] = [ny2-ny0, 0]         #upper left corner
   elif ireg == 2:
     [nyoff, nxoff] = [0, nx2-nx0]         #lower righy corner
   elif ireg == 3:
     [nyoff, nxoff] = [ny2-ny0, nx2-nx0]    #upper righy corner
   #for ihr in range(nhr):
   for ihr in range(1):
     mec_hz  = tmp2[0, :, ihr, nyoff:nyoff+ny0, nxoff:nxoff+nx0].mean(0)
     eflx_hz = tmp2[1, :, ihr, nyoff:nyoff+ny0, nxoff:nxoff+nx0].mean(0)
     divef_hz = mec_hz + eflx_hz
     for ics in range(ncs-1):
        cs_fac = cs_fac_list[ics]
        print("cs_fac: %03i" % cs_fac)
        [ny_cs, nx_cs] = [int(np.floor(ny0/cs_fac)), int(np.floor(nx0/cs_fac))]
        #tmpmec_cs  = np.zeros([ny_cs, nx_cs])
        #tmpeflx_cs = np.zeros([ny_cs, nx_cs])
        tmpdivef_cs = np.zeros([ny_cs, nx_cs])
        for ii in range(nx_cs):
            for jj in range(ny_cs):
                iii = ii*cs_fac
                jjj = jj*cs_fac
                #tmpmec_cs[jj, ii] = np.nansum(mec_hz[jjj:jjj+cs_fac, iii:iii+cs_fac]   * \
                #        e12t_2[jjj:jjj+cs_fac, iii:iii+cs_fac] )
                #tmpeflx_cs[jj, ii] = np.nansum(eflx_hz[jjj:jjj+cs_fac, iii:iii+cs_fac] * \
                #        e12t_2[jjj:jjj+cs_fac, iii:iii+cs_fac] )
                tmpdivef_cs[jj, ii] = np.nansum(divef_hz[jjj:jjj+cs_fac, iii:iii+cs_fac] * \
                    e12t_2[jjj:jjj+cs_fac, iii:iii+cs_fac] ) / \
                    np.nansum(e12t_2[jjj:jjj+cs_fac, iii:iii+cs_fac])

        #
        tmpmsk = np.ones([ny_cs, nx_cs])
        tmpmsk[np.where(np.isnan(tmpdivef_cs))] = np.nan
        #- construct vector for covariance, removing land points -
        wetpts = int(np.nansum(tmpmsk))
        #ppp = np.zeros([2, wetpts])
        ppp2 = np.zeros([wetpts])
        ij = 0
        for jj in range(ny_cs):
            for ii in range(nx_cs):
                if not np.isnan(tmpmsk[jj, ii]):
                    #ppp[0, ij] = tmpmec_cs[jj, ii]
                    #ppp[1, ij] = tmpeflx_cs[jj, ii]
                    ppp2[ij]   = tmpdivef_cs[jj, ii]

                    ij = ij + 1
        # compute covariance matrix
        #ccor = np.corrcoef(ppp)
        #balance3[iper, ihr, ireg, ics] = ccor[0, 1]
        vvar = np.nanstd(ppp2, ddof=1)
        balance3[iper, ihr, ireg, ics] = vvar

#- save -
#fileN = 'cov_mec_eddyflx_coarse_as_box_size_pw3_tseries.bin'
fileN = 'divef_coarse_as_box_size_pw3_tseries.bin'
f = open(dir_out + fileN, 'wb')
balance3.reshape([nper*nhr*4*ncs]).astype('>f4').tofile(f)
f.close()
# load 
#fileN = 'cov_mec_eddyflx_coarse_as_box_size_pw3_tseries.bin'
fileN = 'divef_coarse_as_box_size_pw3_tseries.bin'
f = open(dir_out + fileN, 'r')
balance3 = np.fromfile(f, '>f4').reshape([nper, nhr, 4, ncs])
f.close()



#-- compute corse-grained divef --
divef = tmp2[2, :, :, :].mean(0)
divef_cs = np.zeros([3, ny, nx])
cs_fac_list2 = [3**0, 3**2, 3**4]
ncs2 = len(cs_fac_list2)
for ics in range(ncs2):
    cs_fac = cs_fac_list2[ics]
    print("cs_fac: %03i" % cs_fac)
    [ny_cs, nx_cs] = [int(np.floor(ny/cs_fac)), int(np.floor(nx/cs_fac))]
    for ii in range(nx_cs):
        for jj in range(ny_cs):
            iii = ii*cs_fac
            jjj = jj*cs_fac
            divef_cs[ics, jj*cs_fac:(jj+1)*cs_fac, ii*cs_fac:(ii+1)*cs_fac] = \
                    np.nansum(divef[jjj:jjj+cs_fac, iii:iii+cs_fac]  \
                    * e12t[jjj:jjj+cs_fac, iii:iii+cs_fac] ) \
                    / np.nansum(e12t[jjj:jjj+cs_fac, iii:iii+cs_fac])







#-----------------------------------------
#               PLOT
#-----------------------------------------

fig1 = plt.figure(figsize=(15, 4))
llev = np.arange(-1.0, 1.1, 0.1)*1e-1
cs_fac_deg = (r"$1/60^{\circ}$", r"$\sim 1/6^{\circ}$", r"$\sim 1^{\circ}$")
fig1.clf()
#
for ip in range(3):
    ax = fig1.add_subplot(1, 3, ip+1)
    cs = ax.contourf(hgr.nav_lon, hgr.nav_lat, divef_cs[ip, :, :] * mskNaN, \
            levels=llev, cmap='RdBu_r', extend='both')
    ax.set_ylim([35, 45])
    ax.set_xlim([-5.5, 9])
    ax.set_facecolor([0.5, 0.5, 0.5])
    ax.set_yticks([38, 42])
    ax.set_xticks([0, 5])
    ax.set_xticklabels(['0$^{\circ}$', '5$^{\circ}$E'], fontsize='x-large')
    at = AnchoredText(cs_fac_deg[ip], prop=dict(size=15), frameon=True, \
        loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    if ip == 0:
        ax.set_yticklabels([r'38$^{\circ}$N', r'42$^{\circ}$N'], fontsize='x-large')
    else:
        ax.set_yticklabels([])
#
cbax = fig1.add_axes([0.92, 0.2, 0.01, 0.6])
cb = fig1.colorbar(cs, ax=ax, orientation='vertical', cax=cbax)
cb.set_label(r'[W m$^{-2}$]', fontsize='x-large')
#
figN1 = 'medwest60_divef_fcn_of_coarsening'
fig1.savefig(dir_fig + figN1 + '.png', dpi=100, bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', dpi=100, bbox_inches='tight')



fig2 = plt.figure(figsize=(7, 5))
#scaleF = 1e2
scaleF = 1
fig2.clf()
#
ax1 = fig2.add_subplot(1, 2, 1)
ax1.set_position([0.12, 0.13, 0.83, 0.85])
p01= ax1.plot(np.arange(ncs), balance[0, :]*scaleF, 'b', linewidth=1)
p02= ax1.plot(np.arange(ncs), balance[1, :]*scaleF, 'r', linewidth=1)
p03= ax1.plot(np.arange(ncs), balance[2, :]*scaleF, 'g', linewidth=1)
p04= ax1.plot(np.arange(ncs), balance[3, :]*scaleF, 'y', linewidth=1)
p1 = ax1.plot(np.arange(ncs), balance.mean(0)*scaleF, 'k', linewidth=4)
plt.legend((p01[0], p02[0], p03[0], p04[0], p1[0]), \
        ('lower left', 'upper left', 'lower right', 'upper right', 'mean'))
plt.grid()
ax1.tick_params(axis='y', labelsize='large')
ax1.set_xticklabels([\
        r"$3^0$", \
        r"$3^1$", \
        r"$3^2$", \
        r"$3^3$", \
        r"$3^4$", \
        r"$3^5$", \
        ], fontsize='large')
ax1.set_xlabel('Coarse graining factor', fontsize='x-large')
ax1.set_ylabel('r(MEC, EDDYFLX)', fontsize='x-large')
#ax1.set_ylabel(r"$\sigma_{xy}(DIVEF)$ [$\times10^{-2}$ W m$^{-2}$]", fontsize='x-large')
ax1.set_xlim([0, ncs-2])
ax1.set_ylim([-1, 0])
#ax1.set_ylim([0, 1.6])
ax12 = ax1.twiny()
ax12.set_position([0.12, 0.13, 0.83, 0.80])
ax12.xaxis.tick_top()
ax12.set_xticklabels([\
        r"$1/60^{\circ}$", \
        r"$\sim 1/20^{\circ}$", \
        r"$\sim 1/6^{\circ}$", \
        r"$\sim 1/2^{\circ}$", \
        r"$\sim 1^{\circ}$", \
        r"$\sim 4^{\circ}$", \
        ], fontsize='large')
#
ax2 = fig2.add_subplot(1, 2, 2)
ax2.set_position([0.13, 0.14, 0.24, 0.3])
ax2.contourf(mskNaN, cmap='binary', levels=np.arange(2, 10), extend='both')
ax2.set_facecolor([0.5, 0.5, 0.5])
p1 = mp.patches.Rectangle((0, 0)          , nx0, ny0, linewidth=2, fill=False, color='b', alpha=0.5)
p2 = mp.patches.Rectangle((0, ny-ny0)     , nx0, ny0, linewidth=2, fill=False, color='r', alpha=0.5)
p3 = mp.patches.Rectangle((nx-nx0, 0)     , nx0, ny0, linewidth=2, fill=False, color='g', alpha=0.5)
p4 = mp.patches.Rectangle((nx-nx0, ny-ny0), nx0, ny0, linewidth=2, fill=False, color='y', alpha=0.5)
ax2.add_patch(p1)
ax2.add_patch(p2)
ax2.add_patch(p3)
ax2.add_patch(p4)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim([0, nx])
ax2.set_ylim([0, ny])
#
figN2 = 'medwest60_local_vs_nonlocal_energy_exchanges_std_divef_pw3'
fig2.savefig(dir_fig + figN2 + '.png', dpi=100, bbox_inches='tight')
fig2.savefig(dir_fig + figN2 + '.pdf', dpi=100, bbox_inches='tight')
plt.close(fig2)


#-- time series of correlation --
fig3 = plt.figure(figsize=(7, 5))
fig3.clf()
#llev = np.arange(-1.0, 0.02, 0.02)
llev = np.arange(0.0, 0.04, 0.001)
#
ax1 = fig3.add_subplot(1, 1, 1)
cs1 = ax1.contourf(ttime[0:None:nhr], np.arange(ncs-1), \
        np.transpose(balance3[:, 0, :, :-1].mean(1), [1, 0]), levels=llev, cmap='RdBu_r', \
        extend='both')
#ax1.contour(ttime[0:None:nhr], np.arange(ncs-1), \
#        np.transpose(balance3[:, 0, :, :-1].mean(1), [1, 0]), levels=[-0.5], colors='k')
ax1.set_yticks(np.arange(ncs-1))
ax1.set_yticklabels([\
        r"$1/60^{\circ}$", \
        r"$\sim 1/20^{\circ}$", \
        r"$\sim 1/6^{\circ}$", \
        r"$\sim 1/2^{\circ}$", \
        r"$\sim 1^{\circ}$", \
        ], fontsize='large')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Coarse graining factor')
#
cbax = fig3.add_axes([0.91, 0.2, 0.01, 0.6])
cb = fig3.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax)
cb.set_label(r"$\sigma^2(DIVEF)$")
#
figN3 = 'medwest60_local_vs_nonlocal_energy_exchanges_std_divef_pw3_tseries'
fig3.savefig(dir_fig + figN3 + '.png', dpi=100, bbox_inches='tight')
fig3.savefig(dir_fig + figN3 + '.pdf', dpi=100, bbox_inches='tight')

