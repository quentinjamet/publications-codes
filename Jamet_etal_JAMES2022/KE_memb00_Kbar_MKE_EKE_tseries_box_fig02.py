import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import xarray as xr
import glob
from tkinter import Tcl
from matplotlib.offsetbox import AnchoredText
import powerspec as ps
import scipy.fftpack as sfft


plt.ion()

#-- directories --
dir_in  = '/gpfsstore/rech/egi/uup63gs/medwest60/MEDWEST60-GSL19-S/ens01/1h/'
#dir_in  = '/gpfsstore/rech/egi/commun/MEDWEST60/MEDWEST60-GSL19-S/ens01/1h/'
dir_in2 = '/gpfsstore/rech/egi/uup63gs/medwest60/outputs/'
dir_grd = '/gpfsstore/rech/egi/commun/MEDWEST60/MEDWEST60-I/'
dir_fig = '/linkhome/rech/genige01/uup63gs/Figures/energetics/'

#-- mesh and mask --
nmem    = 20
nday    = 120
ttime   = np.arange(0, nday, 1/24)
nt      = len(ttime)
msk = xr.open_dataset(dir_grd + 'MEDWEST60_mask.nc4')
hgr = xr.open_dataset(dir_grd + 'MEDWEST60_mesh_hgr.nc4')
mskNaN = msk.tmaskutil[0, :, :].data.astype('float')
mskNaN[np.where(mskNaN>0.0)] = 1.0
mskNaN[np.where(mskNaN==0.0)] = np.nan

#-- list of files --
ext = '20100406.nc'
#ext = '20100306.nc'
#ext = '20100506.nc'
listU = Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridU-2D/0*' + ext) )
listV = Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridV-2D/0*' + ext) )
listUm= Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridU-2D/ESTATS_*' + ext) )
listVm= Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridV-2D/ESTATS_*' + ext) )
#
tmp = xr.open_dataset(listU[0])
[nhr, ny, nx] = [ tmp.dims['time_counter'], tmp.dims['y'], tmp.dims['x'] ]

#-- define a region --
iiw  = np.where(hgr.nav_lon[200, :] >= 4.8)[0][0]
iie  = np.where(hgr.nav_lon[200, :] >= 7.8)[0][0]
jjs  = np.where(hgr.nav_lat[:, iiw] >= 37.2)[0][0]
jjn  = np.where(hgr.nav_lat[:, iiw] >= 40.45)[0][0]
[ny2, nx2] = [jjn-jjs, iie-iiw]

#-- as a square for spectra estimates --
iiw0  = np.where(hgr.nav_lon[200, :] >= 4.8)[0][0]
iie0  = np.where(hgr.nav_lon[200, :] >= 7.8)[0][0]
nx0   = iie0-iiw0
ny0   = nx0
jjs0  = np.where(hgr.nav_lat[:, iiw0] >= 37.2)[0][0]
jjn0  = jjs0+ny0

#--------------------------------
# Compute ensemble mean K and MKE
#--------------------------------
rau0 = 1026.0   #[kg/m3]
ttt = -1

#- MKE -
tmpum = xr.open_dataset(listUm[0])
tmpvm = xr.open_dataset(listVm[0])
mke = np.zeros([ny, nx])
mke[1:, 1:] = rau0/4 * \
        ( tmpum.sozocrtx[ttt, 1:, :-1]**2+tmpum.sozocrtx[ttt, 1:, 1:]**2 \
         +tmpvm.somecrty[ttt, :-1, 1:]**2+tmpvm.somecrty[ttt, 1:, 1:]**2)

#- KEM and EKE -
kem = np.zeros([ny, nx])
eke = np.zeros([ny, nx])
for imem in range(nmem):
  print('memb: %03i' % (imem+1))
  tmpu = xr.open_dataset(listU[imem])
  tmpv = xr.open_dataset(listV[imem])
  kem[1:, 1:] = kem[1:, 1:] + rau0/4 * \
        ( tmpu.sozocrtx[ttt, 1:, :-1]**2+tmpu.sozocrtx[ttt, 1:, 1:]**2 \
         +tmpv.somecrty[ttt, :-1, 1:]**2+tmpv.somecrty[ttt, 1:, 1:]**2)
  eke[1:, 1:] = eke[1:, 1:] + rau0/4 * \
          ( (tmpu.sozocrtx[ttt, 1:, :-1]-tmpum.sozocrtx[ttt, 1:, :-1])**2 \
           +(tmpu.sozocrtx[ttt, 1:, 1: ]-tmpum.sozocrtx[ttt, 1:, 1:])**2 \
          + (tmpv.somecrty[ttt, :-1, 1:]-tmpvm.somecrty[ttt, :-1, 1:])**2 \
           +(tmpv.somecrty[ttt, 1:, 1: ]-tmpvm.somecrty[ttt, 1:, 1:])**2 \
          )
kem = kem / float(nmem)
eke = eke / float(nmem)


#-----------------------------------
# Compute spectra at 30 and 60 days
# based on classical of Fourier analysis
#-----------------------------------
wwind = 'Tukey'
ddetr = 'Both'
ext_list = ['20100306.nc', '20100406.nc']
ndays = len(ext_list)
nkl = int(np.min([ny2, nx2])/2 + 2)
psd_um   = np.zeros([ndays, nkl])
psd_vm   = np.zeros([ndays, nkl])
psd_kemu = np.zeros([ndays, nmem, nkl]) 
psd_kemv = np.zeros([ndays, nmem, nkl]) 
psd_ekeu = np.zeros([ndays, nmem, nkl])
psd_ekev = np.zeros([ndays, nmem, nkl])

for iday in range(ndays):
    listU = Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridU-2D/0*' + ext_list[iday]) )
    listV = Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridV-2D/0*' + ext_list[iday]) )
    listUm= Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridU-2D/ESTATS_*' + ext_list[iday]) )
    listVm= Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridV-2D/ESTATS_*' + ext_list[iday]) )
    #
    tmpum = xr.open_dataset(listUm[0])
    tmpvm = xr.open_dataset(listVm[0])
    #-- ensemble mean flow --
    [wavenumber, psd_um[iday, :]] = ps.wavenumber_spectra( \
        tmpum.sozocrtx[ttt, jjs:jjn, iiw:iie].to_masked_array(), \
        tmpum.nav_lon[jjs:jjn, iiw:iie], tmpum.nav_lat[jjs:jjn, iiw:iie], wwind, ddetr)
    [wavenumber, psd_vm[iday, :]] = ps.wavenumber_spectra( \
        tmpvm.somecrty[ttt, jjs:jjn, iiw:iie].to_masked_array(), \
        tmpvm.nav_lon[jjs:jjn, iiw:iie], tmpvm.nav_lat[jjs:jjn, iiw:iie], wwind, ddetr)

    #-- kem and eke --
    for imem in range(nmem):
        print('memb: %03i' % (imem+1))
        tmpu = xr.open_dataset(listU[imem])
        tmpv = xr.open_dataset(listV[imem])
        #
        [tmp, psd_kemu[iday, imem, :]] = ps.wavenumber_spectra( \
         ( tmpu.sozocrtx[ttt, jjs:jjn, iiw:iie] ).to_masked_array(), \
         tmpum.nav_lon[jjs:jjn, iiw:iie], tmpum.nav_lat[jjs:jjn, iiw:iie], wwind, ddetr)
        [tmp, psd_kemv[iday, imem, :]] = ps.wavenumber_spectra( \
         ( tmpv.somecrty[ttt, jjs:jjn, iiw:iie] ).to_masked_array(), \
         tmpvm.nav_lon[jjs:jjn, iiw:iie], tmpvm.nav_lat[jjs:jjn, iiw:iie], wwind, ddetr)
        #
        [tmp, psd_ekeu[iday, imem, :]] = ps.wavenumber_spectra( \
         ( tmpu.sozocrtx[ttt, jjs:jjn, iiw:iie] \
          -tmpum.sozocrtx[ttt, jjs:jjn, iiw:iie] ).to_masked_array(), \
         tmpum.nav_lon[jjs:jjn, iiw:iie], tmpum.nav_lat[jjs:jjn, iiw:iie], wwind, ddetr)
        [tmp, psd_ekev[iday, imem, :]] = ps.wavenumber_spectra( \
         ( tmpv.somecrty[ttt, jjs:jjn, iiw:iie] \
          -tmpvm.somecrty[ttt, jjs:jjn, iiw:iie] ).to_masked_array(), \
         tmpvm.nav_lon[jjs:jjn, iiw:iie], tmpvm.nav_lat[jjs:jjn, iiw:iie], wwind, ddetr)


plt.figure(figsize=(5,4))
plt.loglog(wavenumber*1e3,0.5*(psd_um+psd_vm))
plt.loglog(wavenumber*1e3,0.5*(psd_kemu+psd_kemv).mean(0))
plt.loglog(wavenumber*1e3,0.5*(psd_ekeu+psd_ekev).mean(0))
plt.xlabel('wavenumber (cpkm)',fontsize=13)
plt.ylabel('PSD '+r'[$m^{2}s^{-2}/cpm$]',fontsize=13)
plt.grid(ls='dotted')
plt.tight_layout()


#-----------------------------------
# Compute spectra at 30 and 60 days
# based on ensemble generatlization of Fourier analysis
# (Uchida et al (2022))
#-----------------------------------
ext_list = ['20100306.nc', '20100406.nc']
nper = len(ext_list)

#-- compute for the full ensemble --
xxx0 = hgr.nav_lon[jjs0  :jjn0  , iiw0:iie0]
yyy0 = hgr.nav_lat[jjs0  :jjn0  , iiw0:iie0]
dxx0 = hgr.e1t[0, jjs0, iiw0].data
dyy0 = hgr.e2t[0, jjs0, iiw0].data
kxx0 = np.fft.fftfreq(nx0, dxx0)
lyy0 = np.fft.fftfreq(ny0, dyy0)
kxx  = sfft.fftshift(kxx0)
lyy  = sfft.fftshift(lyy0)
k0, l0 = np.meshgrid(kxx, lyy)
kl0  = np.sqrt(k0**2 + l0**2)
#
fft_uv0     = np.zeros([nper, ny0, nx0])
fft_uv0_dtr = np.zeros([nper, ny0, nx0])
for iper in range(nper):
 tmplU = Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridU-2D/0*' + ext_list[iper]) )
 tmplV = Tcl().call('lsort', '-dict', glob.glob(dir_in + 'gridV-2D/0*' + ext_list[iper]) )
 for imem in range(nmem):
    print("period, memb: %s, %02i" % (ext_list[iper], imem) )
    tmpu = xr.open_dataset(tmplU[imem])
    tmpv = xr.open_dataset(tmplV[imem])
    uuu0 = 0.5 * \
        ( tmpu.sozocrtx[ttt, jjs0:jjn0, iiw0  :iie0  ] \
         +tmpu.sozocrtx[ttt, jjs0:jjn0, iiw0-1:iie0-1] )
    vvv0 = 0.5 * \
        ( tmpv.somecrty[ttt, jjs0  :jjn0  , iiw0:iie0] \
         +tmpv.somecrty[ttt, jjs0-1:jjn0-1, iiw0:iie0] )
    #-- 2D PSD of original velocities --
    spec_fft = np.fft.fft2(uuu0)
    spec_2D = (spec_fft*spec_fft.conj()).real*(dyy0*dxx0)/(ny0*nx0)
    fft_uu0 = sfft.fftshift(spec_2D)
    spec_fft = np.fft.fft2(vvv0)
    spec_2D = (spec_fft*spec_fft.conj()).real*(dyy0*dxx0)/(ny0*nx0)
    fft_vv0 = sfft.fftshift(spec_2D)
    fft_uv0[iper, :, :] = fft_uv0[iper, :, :] + 0.5*(fft_uu0 + fft_vv0)
    #-- 2D PSD of detrended velocities -- 
    udtr0 = detrend(uuu0)
    vdtr0 = detrend(vvv0)
    spec_fft = np.fft.fft2(udtr0)
    spec_2D = (spec_fft*spec_fft.conj()).real*(dyy0*dxx0)/(ny0*nx0)
    fft_uu0 = sfft.fftshift(spec_2D)
    spec_fft = np.fft.fft2(vdtr0)
    spec_2D = (spec_fft*spec_fft.conj()).real*(dyy0*dxx0)/(ny0*nx0)
    fft_vv0 = sfft.fftshift(spec_2D)
    fft_uv0_dtr[iper, :, :] = fft_uv0_dtr[iper, :, :] + 0.5*(fft_uu0 + fft_vv0)

fft_uv0     = fft_uv0 / float(nmem)
fft_uv0_dtr = fft_uv0_dtr / float(nmem)

#-- 1D isotropic spectra --
fft_uv0_iso     = np.zeros([nper, int(nx0/2)])
fft_uv0_dtr_iso = np.zeros([nper, int(nx0/2)])
for iper in range(nper):
  fft_uv0_iso[iper, :]     = _get_1D_psd(kl0, kxx[int(nx0/2):], fft_uv0[iper, :, :])
  fft_uv0_dtr_iso[iper, :] = _get_1D_psd(kl0, kxx[int(nx0/2):], fft_uv0_dtr[iper, :, :])



#-------------------------------------------------------------------
# Load pre-computed K, \oveline{K}, \widetilde{K} and \overline{K^*}
# cf ../Extr/extr_KE_medwest60_box.py
#   0: KE of memb#00
#   1: ensemble mean KE
#   2: KE of the ensemble mean flow (MKE)
#   3: KE of the perturbation       (EKE)
#-------------------------------------------------------------------
fileN1 = 'KE_surf_ens00_mean_MKE_EKE_box_medwest60_tseries.bin'     # first  60 days
fileN2 = 'KE_surf_ens00_mean_MKE_EKE_box_medwest60_tseries_2.bin'     # second 60 days
nd2 = int(nday/2)
ke_diags = np.zeros([4, nday*nhr])
#
f = open(dir_in2 + fileN1,'r')
tmp = np.fromfile(f,'>f4').reshape([nd2, 4, nhr])      #big-indian ('>'), real*4 ('f4')
f.close()
ke_diags[:, :(nd2)*nhr] = np.transpose(tmp, (1, 0, 2)).reshape([4, nd2*nhr])
#
f = open(dir_in2 + fileN2,'r')
tmp = np.fromfile(f,'>f4').reshape([nd2, 4, nhr])      #big-indian ('>'), real*4 ('f4')
f.close()
ke_diags[:, (nd2)*nhr:] = np.transpose(tmp, (1, 0, 2)).reshape([4, nd2*nhr])




#----------
#-- PLOT --
#----------

fig1 = plt.figure(figsize=(15, 8))
llev = np.arange(0, 2.2, 0.2)*1e2
fig1.clf()
# ensemble mean KE
ax1 = fig1.add_subplot(2, 3, 1)
cs1 = ax1.contourf(hgr.nav_lon, hgr.nav_lat, kem * mskNaN, \
        levels=llev, extend='max', cmap='Blues_r')
at1 = AnchoredText(r'$\left< K \right>$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at1)
# MKE
ax2 = fig1.add_subplot(2, 3, 2)
cs2 = ax2.contourf(hgr.nav_lon, hgr.nav_lat, mke * mskNaN, \
        levels=llev, extend='max', cmap='Blues_r')
at2 = AnchoredText(r'$\widetilde{K}$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)
# EKE
ax3 = fig1.add_subplot(2, 3, 3)
cs3 = ax3.contourf(hgr.nav_lon, hgr.nav_lat, eke * mskNaN, \
        levels=llev, extend='max', cmap='Blues_r')
at3 = AnchoredText(r'$\left< K^{*} \right>$', prop=dict(size=15), frameon=True, \
        loc='upper left')
at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at3)
# box int
ax4 = fig1.add_subplot(2, 3, (4, 6))
p0 = ax4.plot(ttime, ke_diags[0, :] * 1e-12, linewidth=1, color='k', alpha=0.5)
p1 = ax4.plot(ttime, ke_diags[1, :] * 1e-12, linewidth=2, color='k')
p2 = ax4.plot(ttime, ke_diags[2, :] * 1e-12, linewidth=2, color='g')
p3 = ax4.plot(ttime, ke_diags[3, :] * 1e-12, linewidth=2, color='r')
ax4.legend((p1[0], p2[0], p3[0], p0[0]), \
        (r'$\left< K \right>$', r'FKE -- $\widetilde{K}$', \
        r'IKE -- $\left< K^* \right>$', r'$K^{memb00}$'))
ax4.set_xticks([0, 30, 60, 90, 120])
ax4.set_yticks(np.arange(0, 12, 2))
ax4.set_xlim([0, 120])
ax4.set_ylim([0, 10])
ax4.set_xlabel('Time [days]', fontsize='x-large')
ax4.set_ylabel(r'Kinetic Energy [TJ]', fontsize='x-large')
plt.grid()
# spectra 
#ax6 = fig1.add_subplot(2, 3, 6)
#p11 = ax6.loglog(kxx[int(nx0/2):]*1e3, fft_uv0_iso[0, :], color='m')
#p12 = ax6.loglog(kxx[int(nx0/2):]*1e3, fft_uv0_iso[1, :], color='m', linestyle='--')
#p21 = ax6.loglog(kxx[int(nx0/2):]*1e3, fft_uv0_dtr[0, int(nx0/2) , int(nx0/2):], color='b')
#p22 = ax6.loglog(kxx[int(nx0/2):]*1e3, fft_uv0_dtr[1, int(nx0/2) , int(nx0/2):], color='b', linestyle='--')
#p31 = ax6.loglog(kxx[int(nx0/2):]*1e3, fft_uv0_dtr[0, int(nx0/2):, int(nx0/2)], color='cyan')
#p32 = ax6.loglog(kxx[int(nx0/2):]*1e3, fft_uv0_dtr[1, int(nx0/2):, int(nx0/2)], color='cyan', linestyle='--')
#ax6.set_ylabel(r"PSD [$m^{2}s^{-2}/cpm$]",fontsize='x-large')
#ax6.yaxis.tick_right()
#ax6.yaxis.set_label_position("right")
#plt.grid(ls='dotted')
#-- colorbar --
cbax1 = fig1.add_axes([0.92, 0.5, 0.01, 0.4])
cb1 = fig1.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
cb1.set_label(r'Kinetic Energy [J m$^{-3}$]', fontsize='x-large')
#
for ip in (1, 2, 3):
    ax = fig1.add_subplot(2, 3, ip)
    ax.set_ylim([35, 45])
    ax.set_xlim([-5.5, 9])
    ax.set_facecolor([0.5, 0.5, 0.5])
    ax.set_yticks([38, 42])
    ax.set_xticks([0, 5])
    ax.set_xticklabels(['0$^{\circ}$', '5$^{\circ}$E'], fontsize='x-large')
    if ip == 1:
        ax.set_yticklabels([r'38$^{\circ}$N', r'42$^{\circ}$N'], fontsize='x-large')
        # add rectangle
        p1 = mp.patches.Rectangle((4.4, 37.45), 2.56, 1.91, linewidth=2, fill=False, color='k', alpha=0.5)
        ax.add_patch(p1)
        p2 = mp.patches.Rectangle((4.8, 37.2), 3, 3.25, linewidth=3, fill=False, color='g')
        ax.add_patch(p2)
    else:
        ax.set_yticklabels([])
#
figN = 'KE_surf_box_mke_eke_medwest60_tseries'
fig1.savefig(dir_fig + figN + '.png', dpi=100, bbox_inches='tight')
fig1.savefig(dir_fig + figN + '.pdf', dpi=100, bbox_inches='tight')
plt.close(fig1)



#---------------------------------------
#       Some tools 
#---------------------------------------

#- detrending -
def detrend(tmpin):
    [nyy, nxx] = tmpin.shape
    # detrend zonally
    tmpx1 = (tmpin[:, -1] - tmpin[:, 0]) / nxx
    tmpin1 = tmpin - tmpx1.data[:, np.newaxis] * np.arange(nxx)[np.newaxis, :]
    # detrend meridionally
    tmpy1 = (tmpin1[-1, :] - tmpin1[0, :]) / nyy
    tmpin2 = tmpin1 - tmpy1.data[np.newaxis, :] * np.arange(nxx)[:, np.newaxis]
    #
    tmpout = tmpin2
    return tmpout

#- get isotropic specta (from Ajayi) -
def _get_1D_psd(kradial,wavnum,spec_2D):
    ''' Compute the azimuthaly avearge of the 2D spectrum '''
    spec_1D = np.zeros(len(wavnum))
    for i in range(wavnum.size):
        kfilt =  (kradial>=wavnum[i] - wavnum[0]) & (kradial<=wavnum[i])
        N = kfilt.sum()
        spec_1D[i] = (spec_2D[kfilt].sum())*wavnum[i]/N
    return spec_1D


