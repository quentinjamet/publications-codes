import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob
import sys

plt.ion()

#-- directories --
dir_in1  = '/mnt/meom/workdir/jametq/MEDWEST60/MEDWEST60-GSL21-S/ens02/1h/'
dir_in2  = '/mnt/meom/workdir/jametq/CDFTOOLS/outputs/'
dir_grd  = '/mnt/meom/workdir/jametq/MEDWEST60/MEDWEST60-GSL18-S/mesh/'
dir_fig  = '/home/jametq/Figures/energetics/'

#-- mesh and mask --
hgr   = xr.open_dataset(dir_grd + 'mesh_hgr.nc')
zgr   = xr.open_dataset(dir_grd + 'mesh_zgr.nc')
msk   = xr.open_dataset(dir_grd + 'mask.nc')
bathy = xr.open_dataset(dir_grd + 'bathy.nc')
[nr, ny, nx] = [msk.dims['z'], msk.dims['y'], msk.dims['x']]
dt = 3600.0		#[sec] - hourly model outputs
rau0 = 1026.0

#-- NEMO output KE trends --
#- file names -
#!! Be sure they are sorted in appropriate ordering ... !!
listSSH    = sorted( glob.glob(dir_in1 + "*_gridT-2D_*") )
listKE1    = sorted( glob.glob(dir_in1 + "*_KEtrd1_*") )
listKE2    = sorted( glob.glob(dir_in1 + "*_KEtrd2_*") )
#- check file number match -
nfiles = len(listSSH)
if ( len(listKE1) != nfiles ): sys.exit("KE1 has not the correct number of files")
if ( len(listKE2) != nfiles ): sys.exit("KE2 has not the correct number of files")

#-- CDFTOOLS version of KE budget --
#- file names -
list_dt  = sorted( glob.glob(dir_in2 + "medwest60_test_1h_dkedt_*")  )
list_adv = sorted( glob.glob(dir_in2 + "medwest60_test_1h_adv_ke_*") )
list_hpg = sorted( glob.glob(dir_in2 + "medwest60_test_1h_hpg_ke_*") )
list_zdf = sorted( glob.glob(dir_in2 + "medwest60_test_1h_zdf_ke_*") )
#- check file number match -
if ( len(list_dt)  != nfiles ): sys.exit("dkedt has not the correct number of files")
if ( len(list_adv) != nfiles ): sys.exit("adv   has not the correct number of files")
if ( len(list_hpg) != nfiles ): sys.exit("hpg   has not the correct number of files")
if ( len(list_zdf) != nfiles ): sys.exit("zdf   has not the correct number of files")



#------------------------------
# Hz maps of errors
#------------------------------

#-- Load data in the list of files --
ifile = 0
#- NEMO -
ztrdke1= xr.open_dataset(listKE1[ifile])  # hpg, spg, rvo, pvo, KE, atf
ztrdke2= xr.open_dataset(listKE2[ifile])  # keg, zad, udx, zdf
#- CDFTOOLS -
dkedt  = xr.open_dataset(list_dt[ifile])
adv    = xr.open_dataset(list_adv[ifile])
hpg    = xr.open_dataset(list_hpg[ifile])
zdf    = xr.open_dataset(list_zdf[ifile])

#-- Define a general figure style --
def fig_hz_map(xx, yy, zz_nemo, zz_cdftools, llev, figN, tit, scaleF, scaleErr, saveF):
  error = zz_cdftools - zz_nemo
  fig1 = plt.figure(figsize=(15, 5))
  fig1.suptitle(tit)
  #-- nemo --
  ax1 = fig1.add_subplot(1, 3, 1)
  cs1 = ax1.contourf(xx, yy, zz_nemo * scaleF, levels=llev, extend='both')
  ax1.set_facecolor([0.5, 0.5, 0.5])
  ax1.set_title('A/ Model output')
  ax1.set_ylabel('Latitude')
  #-- cdftools --
  ax2 = fig1.add_subplot(1, 3, 2)
  cs2 = ax2.contourf(xx, yy, zz_cdftools * scaleF, levels=llev, extend='both')
  #ax2.contour(xx, yy,  error * scaleF * scaleErr, \
  #	levels=llev[0:None:4], colors='white', linewidth=1, alpha=0.5)
  ax2.set_facecolor([0.5, 0.5, 0.5])
  ax2.set_title('B/ CDFTOOLS output')
  ax2.set_xlabel('Longitude')
  ax2.set_yticklabels([''])
  cbax1 = fig1.add_axes([0.02, 0.1, 0.01, 0.8])
  cb1 = fig1.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
  cb1.set_label(r'[W m$^{-3}$]')
  cb1.set_label(r'$\times$ %0.0e ; [W m$^{-3}$]' % (1/scaleF) )
  #-- error --
  ax3 = fig1.add_subplot(1, 3, 3)
  cs3 = ax3.contourf(xx, yy,  error * scaleF * scaleErr,  \
	levels=llev, extend='both', cmap='RdBu_r')
  ax3.set_facecolor([0.5, 0.5, 0.5])
  ax3.set_title(r'C/ Error (B/-A/)')
  ax3.set_yticklabels([''])
  cbax3 = fig1.add_axes([0.92, 0.1, 0.01, 0.8])
  cb3 = fig1.colorbar(cs3, ax=ax3, orientation='vertical', cax=cbax3)
  cb3.set_label(r'$\times$ %0.0e ; [W m$^{-3}$]' % (1/(scaleF*scaleErr)) )
  #
  if saveF:
    fig1.savefig(dir_fig + figN + '.png', bbox_inches='tight')
    fig1.savefig(dir_fig + figN + '.pdf', bbox_inches='tight')
    plt.close(fig1)


 
#--  Display different terms --
kk = 0
tt = -2
llevs = np.arange(-1, 1.1, 0.1)
saveF = False
  
#- time tendency -
figN1 = str('comp_dKEdt_nemo_cdftools_hz_map_k%i_1h' % kk)
tit = 'Time tendency'
scaleF = 1e2
scaleErr = 1e2
# horizontal map
fig_hz_map(dkedt.nav_lon, dkedt.nav_lat, \
        (ztrdke1.ketrd_tot_nb[tt, kk, :, :] + ztrdke1.ketrd_atf_nb[tt+1, kk, :, :] ), \
        (dkedt.dkedt[tt, kk, :, :]), \
        llevs, figN1, tit, scaleF, scaleErr, saveF)

#- advection  -
figN1 = str('comp_KE_adv_nemo_cdftools_hz_map_k%i_1h' % kk)
tit = 'Advection'
scaleF = 1e2
scaleErr = 1e3
fig_hz_map(adv.nav_lon, adv.nav_lat, \
        (ztrdke2.ketrd_keg_nb[tt, kk, :, :] + ztrdke2.ketrd_zad_nb[tt, kk, :, :]), \
        (adv.advh_ke[tt, kk, :, :] + adv.advz_ke[tt, kk, :, :]), \
        llevs, figN1, tit, scaleF, scaleErr, saveF)

#- hydrostatic pressure gradient -
figN1 = str('comp_KE_hpg_nemo_cdftools_hz_map_k%i_1h' % kk)
tit = 'Hydrostatic pressure gradient'
scaleF = 1e2
scaleErr = 1e3
fig_hz_map(hpg.nav_lon, hpg.nav_lat, \
	(ztrdke1.ketrd_hpg_nb[tt, kk, :, :]), \
	(hpg.hpg_ke[tt, kk, :, :]), \
	llevs, figN1, tit, scaleF, scaleErr, saveF)

#- vertical physics -
figN1 = str('comp_KE_zdf_nemo_cdftools_hz_map_k%i_1h' % kk)
tit = 'Vertical dissipation'
scaleF = 1e2
scaleErr = 1e1
fig_hz_map(zdf.nav_lon, zdf.nav_lat, \
        (ztrdke2.ketrd_zdf_nb[tt, kk, :, :]), \
        (zdf.zdf_ke[tt, kk, :, :]), \
        llevs, figN1, tit, scaleF, scaleErr, saveF)

#- spg as residual -
figN1 = str('comp_KE_spg_residual_nemo_cdftools_hz_map_k%i_1h' % kk)
tit = 'Surface pressure correction (as residual)'
scaleF = 1e3
scaleErr = 1e0
fig_hz_map(ztrdke1.nav_lon, ztrdke1.nav_lat, \
        (ztrdke1.ketrd_spg_nb[tt, kk, :, :]), \
        (spg_as_res[tt, kk, :, :]), \
        llevs, figN1, tit, scaleF, scaleErr, saveF)


#-- Error quantification --
def q_err(zz1, zz2):
  err = zz2.data-zz1.data
  std_fld  = np.reshape(zz1.data, ny*nx).std(ddof=1)
  std_err  = np.reshape(err[3:-3, 3:-3], (ny-6)*(nx-6)).std(ddof=1) / \
	     np.reshape(zz1[3:-3, 3:-3].data, (ny-6)*(nx-6)).std(ddof=1)
  return std_err

#-
err_dt = q_err(ztrdke1.ketrd_tot_nb[tt, kk, :, :] + ztrdke1.ketrd_atf_nb[tt+1, kk, :, :], \
	dkedt.dkedt[tt, kk, :, :])
err_adv = q_err(ztrdke2.ketrd_keg_nb[tt, kk, :, :] + ztrdke2.ketrd_zad_nb[tt, kk, :, :], \
	 adv.advh_ke[tt, kk, :, :] + adv.advz_ke[tt, kk, :, :])
err_hpg = q_err(ztrdke1.ketrd_hpg_nb[tt, kk, :, :], hpg.hpg_ke[tt, kk, :, :])
err_zdf = q_err(ztrdke2.ketrd_zdf_nb[tt, kk, :, :], zdf.zdf_ke[tt, kk, :, :])

print("dt: %0.01e" % err_dt)
print("adv: %0.01e" % err_adv)
print("hpg: %0.01e" % err_hpg)
print("zdf: %0.01e" % err_zdf)


#------------------------------
# Horizontally integrated error
#------------------------------
#-- definition --
nt = dkedt.dims['time_counter']
def hz_int(zz_nemo, zz_cdf):
  tmp_out_nemo = np.zeros([nt, nr])
  tmp_out_cdf = np.zeros([nt, nr])
  tmp_out_err = np.zeros([nt, nr])
  for iii in range(1, nt-1):
    print("time: %i" % iii)
    #- re-compute e3t -
    e3t = zgr.e3t_0[0, :, :, :] \
        * (1 + sshn.sossheig[iii, :, :]/bathy.gdepw_0[0, :, :])
    e123t = np.tile(  hgr.e1t * hgr.e2t , (nr, 1, 1)) * e3t
    #- hz avg NEMO -
    tmp_out_nemo[iii, :] = np.reshape( \
	zz_nemo[iii, :, 3:-3, 3:-3].data * e123t[:, 3:-3, 3:-3].data, \
	(nr, (ny-6)*(nx-6)) ).sum(-1)
    #- hz avg CDFTOOLS -
    tmp_out_cdf[iii, :] = np.reshape( \
	zz_cdf[iii, :, 3:-3, 3:-3].data * e123t[:, 3:-3, 3:-3].data, \
	(nr, (ny-6)*(nx-6)) ).sum(-1)
    #- hz avg ERROR -
    tmp_out_err[iii, :] = np.reshape( \
	( zz_cdf[iii, :, 3:-3, 3:-3].data - zz_nemo[iii, :, 3:-3, 3:-3].data ) \
	* e123t[:, 3:-3, 3:-3].data, \
	(nr, (ny-6)*(nx-6)) ).sum(-1)
  #
  return tmp_out_nemo, tmp_out_cdf, tmp_out_err

#-- compute --
#- advection -
[adv_nemo_hz_int, adv_cdf_hz_int, adv_err_hz_int] = \
	hz_int(ztrdke2.ketrd_keg_nb + ztrdke2.ketrd_zad_nb, \
	adv.advh_ke + adv.advz_ke)

#-- Plot --
tt = -2
scaleF = 1e3
plt.figure()
p0 = plt.plot(adv_nemo_hz_int[-2, :], -adv.deptht, 'k')
p1 = plt.plot(adv_cdf_hz_int[-2, :], -adv.deptht, 'r.')
p2 = plt.plot(adv_err_hz_int[-2, :] * scaleF, -adv.deptht, 'b')
plt.grid()
plt.legend((p0[0], p1[0], p2[0]),\
	('A/ NEMO', 'B/ CDFTOOLS', r"A/-B/ ($\times$ %0.0e)" %(1/scaleF)))



#------------------------
# Volume integrated error
#------------------------
#-- temporal dimension --
sshn   = xr.open_dataset(listSSH[0])
nt = sshn.dims['time_counter'] * nfiles

#-- definition --
def vol_int(zz_nemo, zz_cdf, tmpssh):
  #- load file -
  tmpnt   = tmpssh.dims['time_counter']
  tmp_out_nemo = np.zeros([tmpnt, 1])
  tmp_out_cdf = np.zeros([tmpnt, 1])
  tmp_out_err = np.zeros([tmpnt, 1])
  #- compute -
  for iii in range(tmpnt):
    #- re-compute e3t -
    e3t = zgr.e3t_0[0, :, :, :] \
        * (1 + tmpssh.sossheig[iii, :, :]/bathy.gdepw_0[0, :, :])
    e123t = np.tile(  hgr.e1t * hgr.e2t , (nr, 1, 1)) * e3t
    #- hz avg NEMO -
    tmp_out_nemo[iii] = np.reshape( \
        zz_nemo[iii, :, 3:-3, 3:-3].data * e123t[:, 3:-3, 3:-3].data, \
        (nr*(ny-6)*(nx-6)) ).sum()
    #- hz avg CDFTOOLS -
    tmp_out_cdf[iii] = np.reshape( \
        zz_cdf[iii, :, 3:-3, 3:-3].data * e123t[:, 3:-3, 3:-3].data, \
        (nr*(ny-6)*(nx-6)) ).sum()
    #- hz avg ERROR -
    tmp_out_err[iii] = np.reshape( \
        ( zz_cdf[iii, :, 3:-3, 3:-3].data - zz_nemo[iii, :, 3:-3, 3:-3].data ) \
        * e123t[:, 3:-3, 3:-3].data, \
        (nr*(ny-6)*(nx-6)) ).sum()
  #
  return tmp_out_nemo, tmp_out_cdf, tmp_out_err

#-- compute --
dt_vol_int  = np.zeros([nt, 3])		#3: [nemo, cdf, error]
adv_vol_int = np.zeros([nt, 3])
hpg_vol_int = np.zeros([nt, 3])
spg_vol_int = np.zeros([nt, 3])
zdf_vol_int = np.zeros([nt, 3])
for ifile in range(nfiles):
    print("%i/%i" % (ifile, nfiles-1))
    #- load file -
    tmpssh  = xr.open_dataset(listSSH[ifile] )
    tmpke1  = xr.open_dataset(listKE1[ifile] )
    tmpke2  = xr.open_dataset(listKE2[ifile] )
    tmp_dt  = xr.open_dataset(list_dt[ifile] )
    tmp_adv = xr.open_dataset(list_adv[ifile])
    tmp_hpg = xr.open_dataset(list_hpg[ifile])
    tmp_zdf = xr.open_dataset(list_zdf[ifile])
    tmpnt   = tmpssh.dims['time_counter']
    #- time rate of change (without Asselin filter) -
    print('-- Time rate of change --')
    [tmp_nemo, tmp_cdf, tmp_err] = \
        vol_int(tmpke1.ketrd_tot_nb, tmp_dt.dkedt, tmpssh)
    dt_vol_int[ifile*tmpnt:(ifile+1)*tmpnt, :] = \
	np.concatenate((tmp_nemo, tmp_cdf, tmp_err), axis=-1)
    #- advection -
    print('-- Advection --')
    [tmp_nemo, tmp_cdf, tmp_err] = \
        vol_int(tmpke2.ketrd_keg_nb + tmpke2.ketrd_zad_nb, \
	tmp_adv.advh_ke + tmp_adv.advz_ke, tmpssh)
    adv_vol_int[ifile*tmpnt:(ifile+1)*tmpnt, :] = \
	np.concatenate((tmp_nemo, tmp_cdf, tmp_err), axis=-1)
    #- hpg -
    print('-- Pressure gradients --')
    [tmp_nemo, tmp_cdf, tmp_err] = \
        vol_int(tmpke1.ketrd_hpg_nb, tmp_hpg.hpg_ke, tmpssh)
    hpg_vol_int[ifile*tmpnt:(ifile+1)*tmpnt, :] = \
        np.concatenate((tmp_nemo, tmp_cdf, tmp_err), axis=-1)
    #- spg (only valid for nemo outputs) -
    print('-- SPG (NEMO ONLY) --')
    [tmp_nemo, tmp_cdf, tmp_err] = \
        vol_int(tmpke1.ketrd_spg_nb, tmp_hpg.hpg_ke, tmpssh)
    spg_vol_int[ifile*tmpnt:(ifile+1)*tmpnt, :] = \
        np.concatenate((tmp_nemo, tmp_cdf, tmp_err), axis=-1)
    #- zdf -
    print('-- Vertical physics --')
    [tmp_nemo, tmp_cdf, tmp_err] = \
        vol_int(tmpke2.ketrd_zdf_nb, tmp_zdf.zdf_ke, tmpssh)
    zdf_vol_int[ifile*tmpnt:(ifile+1)*tmpnt, :] = \
        np.concatenate((tmp_nemo, tmp_cdf, tmp_err), axis=-1)


#-- Plot --
time = np.arange(nt)+1

#- check (again and again) that NEMO budget balances -
rhs = adv_vol_int[:, 0] + hpg_vol_int[:, 0] + spg_vol_int[:, 0] + zdf_vol_int[:, 0]
plt.figure()
p0 = plt.plot(time, dt_vol_int[:, 0], 'k', linewidth=2)
p1 = plt.plot(time, adv_vol_int[:, 0], alpha=0.5, linewidth=1)
p2 = plt.plot(time, hpg_vol_int[:, 0], alpha=0.5, linewidth=1)
p3 = plt.plot(time, spg_vol_int[:, 0], alpha=0.5, linewidth=1)
p4 = plt.plot(time, zdf_vol_int[:, 0], alpha=0.5, linewidth=1)
p5 = plt.plot(time, rhs              , 'r.') 
p6 = plt.plot(time, (dt_vol_int[:, 0] - rhs), 'b') 
plt.grid()



def ffig(zzz, scaleF, tit, ip):
  ax = fig1.add_subplot(2, 2, ip)
  p0 = ax.plot(time[1:-1], zzz[1:-1, 0].cumsum() /1e9, 'k')
  p1 = ax.plot(time[1:-1], zzz[1:-1, 1].cumsum() /1e9, 'r.')
  #p2 = ax.plot(time[1:-1], zzz[1:-1, 2] * scaleF / 1e9, 'b')
  p2 = ax.plot(time[1:-1], zzz[1:-1, 2].cumsum() * scaleF / 1e9, 'b')
  plt.grid()
  plt.xlim([0, time[-1]])
  plt.legend((p0[0], p1[0], p2[0]),\
        ('A/ NEMO', 'B/ CDFTOOLS', r"A/-B/ ($\times$ %0.0e)" %(1/scaleF)))
  ax.set_title(tit)
  ax.set_xlabel('Time [hours]')
  ax.set_ylabel('[GW hr]')


fig1 = plt.figure(figsize=(10, 6))
#- dkedt -
scaleF = 1e1
ffig(dt_vol_int, scaleF, r'$\int_t \partial_t KE ~dt$', 1)
#- adv -
scaleF = 1e3
ffig(adv_vol_int, scaleF, r'$\int_t adv~dt$', 2)
#- hpg -
scaleF = 1e3
ffig(hpg_vol_int, scaleF, r'$\int_t HPG~dt$', 3)
#- zdf -
scaleF = 1e0
ffig(zdf_vol_int, scaleF, r'$\int_t ZDF~dt$', 4)
#- save -
figN1 = 'vol_int_errors_ke_10d_tint'
fig1.savefig(dir_fig + figN1 + '.png', bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', bbox_inches='tight')
plt.close(fig1)


def ffig(zzz, scaleF, tit, ip):
  ax = fig1.add_subplot(1, 3, ip)
  p0 = ax.plot(time[1:-1], zzz[1:-1, 0].cumsum() /1e9, 'k')
  p1 = ax.plot(time[1:-1], zzz[1:-1, 1].cumsum() /1e9, 'r.')
  #p2 = ax.plot(time[1:-1], zzz[1:-1, 2] * scaleF / 1e9, 'b')
  p2 = ax.plot(time[1:-1], zzz[1:-1, 2].cumsum() * scaleF / 1e9, 'b')
  plt.grid()
  plt.xlim([0, time[-1]])
  plt.legend((p0[0], p1[0], p2[0]),\
        ('A/ NEMO', 'B/ CDFTOOLS', r"A/-B/ ($\times$ %0.0e)" %(1/scaleF)))
  ax.set_title(tit)
  ax.set_xlabel('Time [hours]')
  ax.set_ylabel('[GW hr]')


fig1 = plt.figure(figsize=(15, 5))
#- dkedt -
scaleF = 1e1
ffig(dt_vol_int, scaleF, r'$\int_t \partial_t KE ~dt$', 1)
#- adv -
scaleF = 1e3
ffig(adv_vol_int, scaleF, r'$\int_t adv~dt$', 2)
#- hpg -
scaleF = 1e3
ffig(hpg_vol_int, scaleF, r'$\int_t HPG~dt$', 3)
#- save -
figN1 = 'vol_int_errors_ke_10d_tint_no_zdf'
fig1.savefig(dir_fig + figN1 + '.png', bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.pdf', bbox_inches='tight')
plt.close(fig1)

