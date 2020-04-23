import numpy as np
from scipy import signal
import MITgcmutils as mit
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy import interpolate
import statsmodels.api as sm    # for LOESS (or LOWESS) smoothing


#-- directories --
dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/orar/'
dir_rapid = '/tank/chaocean/rapid26N/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/'


#-- load grid --
yG = mit.rdmds(dir_grd + 'YG')
rF = mit.rdmds(dir_grd + 'RF')
ny, nx = yG.shape
nr = len(rF)-1
jj26n = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0]
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0]


#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt_ens = ndump*nyr
nmem = 24

#---------------------------
# load RAPID data 
#---------------------------
# yearly mean from Smeed et al., OS 2014

f2r = Dataset(dir_rapid+'moc_vertical.nc','r')
time_rapid = np.array(f2r.variables['time'])/365.0 + 2004.0 + 91.0/365.0
rF_rapid = -np.array(f2r.variables['depth'])
nr_rapid = len(rF_rapid)
amoc_rapid = np.array(f2r.variables['stream_function_mar'])
f2r2 = Dataset(dir_rapid+'moc_transports.nc','r')

#-- time --
tt_rapid = np.where(time_rapid <= 2013.0)[0]
tt_rapid = tt_rapid[7:]
nt_rapid = len(tt_rapid)
time_rapid = time_rapid[tt_rapid]

#-- keep rapid amoc data for the considered period --
amoc_rapid = amoc_rapid[:,tt_rapid]

#-- compute 5-d avg (rapid sample are every 12 hr) --
amoc_rapid_5d = np.mean( amoc_rapid.reshape([nr_rapid, nt_rapid/10, 10])*1.0 ,2)
time_rapid = time_rapid[4:-1:10]
nt_rapid = len(time_rapid)


#-- vertical interpolation on chao12 grid --
f = interpolate.interp1d(rF_rapid, amoc_rapid_5d, axis=0)
amoc_rapid_5d_46z = f(np.squeeze(rF[0:nr]))

#-- select given depth --
amoc_26n_rapid = np.squeeze(amoc_rapid_5d_46z[kdepth,:])

#-- anomaly --
amoc_ano_rapid = amoc_26n_rapid - np.tile( np.mean(amoc_26n_rapid), (nt_rapid,) ) 


#------------------------------
# load pre-extracted amoc files
#------------------------------
mocyzt = np.zeros([nmem, nt_ens, nr, ny])
#-- first 12 members --
fileN1 = 'MOCyzt_orar_ensemble.bin'
f = open(dir_in+fileN1,'r')
tmp_moc = np.fromfile(f,'>f4')      #big-indian ('>'), real*4 ('f4')
f.close()
tmp_moc = tmp_moc.reshape([nyr, nmem/2, ndump, nr, ny])
tmp_moc = np.transpose(tmp_moc, (1, 0, 2, 3, 4)).reshape([nmem/2, nt_ens, nr, ny])
mocyzt[0:12,:,:,:] = tmp_moc
del tmp_moc
#-- next 12 members --
fileN1 = 'MOCyzt_orar_ensemble_2.bin'
f = open(dir_in+fileN1,'r')
tmp_moc = np.fromfile(f,'>f4')      #big-indian ('>'), real*4 ('f4')
f.close()
tmp_moc = tmp_moc.reshape([nyr, nmem/2, ndump, nr, ny])
tmp_moc = np.transpose(tmp_moc, (1, 0, 2, 3, 4)).reshape([nmem/2, nt_ens, nr, ny])
mocyzt[12:,:,:,:] = tmp_moc
del tmp_moc

#-- time series at 26n and 1200m --
moc26n = np.squeeze(mocyzt[:,:,kdepth,jj26n])
moc26n = moc26n - np.tile( np.mean(moc26n,1)[:,np.newaxis], (1, nt_ens) )

del mocyzt

#-----------------------------------
# Low-pass filter
#-----------------------------------
moc_lpf = np.zeros([nmem, nt_ens])
fs = ndump*1.0          # sampling freq [yr-1]
cof = 1.0               # cut-off freq [yr-1]
b, a = signal.butter(10, cof/(fs/2), btype='low')
#- for rapid data -
moc_rapid_lpf = signal.filtfilt(b, a, amoc_ano_rapid)
#- for ensemble -
for imem in range(0,nmem):
  moc_lpf[imem,:] = signal.filtfilt(b, a, moc26n[imem,:])


#-- PLOT --
fig1 = plt.figure(figsize=(8,4.2))
p00 = plt.plot(time_rapid, np.zeros(nt_rapid,), 'k--')
p0  = plt.plot(time_ens, np.transpose(moc_lpf, (1, 0)), color='grey', linewidth=0.5)
p1  = plt.plot(time_ens, np.mean(moc_lpf, 0), 'k')
p2  = plt.plot(time_rapid,moc_rapid_lpf,'r')
plt.grid()
plt.xlim((2005,2012))
plt.xlabel('Time [yr]', fontsize='x-large')
plt.ylabel('AMOC anomalies [Sv]', fontsize='x-large')
#plt.title('A look at RAPID-MOCHA-WBTS location(26.5N)', fontsize='xx-large')
plt.legend((p0[0], p1[0], p2[0]), ('Members','Ens. mean','RAPID '))
plt.ylim((-7, 6))
#plt.show()
fileN1 = '1yr_lpf_moc26n_ano_rapid_orar24mem_fig03'
fig1.savefig(dir_fig + fileN1 + '.pdf',dpi=300)
fig1.savefig(dir_fig + fileN1 + '.png',dpi=300)

#-- only ensemble with forced time mean --
fig3 = plt.figure(figsize=(8,4.2))
p00 = plt.plot(time_ens, np.zeros(nt_ens,), 'k--')
p0  = plt.plot(time_ens, np.transpose(moc_lpf, (1, 0)), color='grey', linewidth=0.5)
p1  = plt.plot(time_ens, np.mean(moc_lpf, 0), 'k')
plt.grid()
plt.xlim((2005,2012))
plt.xlabel('Time [yr]', fontsize='x-large')
plt.ylabel('Anomalies de AMOC at 26.5N', fontsize='x-large')
plt.legend((p0[0], p1[0]), ('Membres','Moyenne d''ensemble'))
plt.ylim((-7, 6))
#plt.show()
fileN = '1yr_lpf_moc26n_ano_orar24mem'
fig1.savefig(dir_fig + fileN1 + '.pdf',dpi=300)
fig1.savefig(dir_fig + fileN1 + '.png',dpi=300)


#-----------------------------------
# 	Spectrum
# of the fully processed AMOC
#-----------------------------------
#-- load processed amoc --
mocyzt = np.zeros([nmem, nt_ens, nr, ny])
#-- first 12 members --
fileN1 = 'MOCyzt_orar_ensemble_detrend_1ylpf.bin'
f = open(dir_in+fileN1,'r')
mocyzt[0:12, :, :, :] = np.fromfile(f,'>f4').reshape([nmem/2, nt_ens, nr, ny])
f.close()
#-- next 12 members --
fileN1 = 'MOCyzt_orar_ensemble_2_detrend_1ylpf.bin'
f = open(dir_in+fileN1,'r')
mocyzt[12:, :, :, :] = np.fromfile(f,'>f4').reshape([nmem/2, nt_ens, nr, ny])
f.close()

#- select 26.5N, 1200m and remove first and last years -
moc_proc_26n = np.squeeze(mocyzt[:, ndump:-ndump, kdepth, jj26n])
nt_ens2 = nt_ens-2*ndump 

#- construct forced/intrinsi signal -
moc_f = np.mean(moc_proc_26n, 0)
moc_i = moc_proc_26n - np.tile(moc_f[np.newaxis, :], (nmem, 1))

#- compute spectra -
nfft = nt_ens2/2+1
fs = ndump*1.0          # sampling freq [yr-1]
# forced
f1, pxx_f = signal.periodogram(moc_f, fs, scaling='density')
# intrinsic
pxx_i = np.zeros([nmem, nfft])
for imem in range(0,nmem):
  f1, pxx_i[imem,:] = signal.periodogram(moc_i[imem,:], fs, scaling='density')

pxx_i = np.mean(pxx_i, 0)

#- smooth PSD -
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

span = 2
pxx_f_smooth = smooth(pxx_f, span)
pxx_i_smooth = smooth(pxx_i, span)



fig2 = plt.figure(figsize=(6,5))
p0 = plt.semilogx(1/f1, pxx_f_smooth,color='k')
p1 = plt.semilogx(1/f1, pxx_i_smooth,color='g')
plt.legend((p0[0], p1[0]),('FORCED','INTRINSIC'))
plt.xlim(0.5,50)
plt.xlabel('Period [yr]', fontsize='x-large')
plt.ylabel('PSD [Sv$^{2}$ yr]', fontsize='x-large')
#plt.title('Power Spectral Density', fontsize='xx-large')
plt.grid()
plt.gca().invert_xaxis()
#plt.show()

fileN2 = '1yr_lpf_moc26n_ano_rapid_orar24mem_psd'
fig2.savefig(dir_fig + fileN2 + '.pdf',dpi=300)
fig2.savefig(dir_fig + fileN2 + '.png',dpi=300)


