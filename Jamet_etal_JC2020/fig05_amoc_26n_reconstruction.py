import numpy as np
from multiprocessing import Pool
import statsmodels.api as sm    # for LOESS (or LOWESS) smoothing
import time
from scipy import signal
import MITgcmutils as mit
import matplotlib.pyplot as plt
# for RAPID data
from netCDF4 import Dataset
from scipy import interpolate


#run it with python, not ipython ....

plt.ion()

config=['orar','ocar','orac','ocac']
nconf = len(config)
dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/'
dir_rapid = '/tank/chaocean/rapid26N/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/forced_amoc/'
dir_rapid = '/tank/chaocean/rapid26N/'

#-- load grid --
yG = mit.rdmds(dir_grd + 'YG')
rF = mit.rdmds(dir_grd + 'RF')
ny, nx = yG.shape
nr = len(rF)-1
jj26n = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0][0]
#jj26n = np.where(np.abs(yG[:,1]-34.3) == np.min(np.abs(yG[:,1]-34.3)))[0][0]
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0][0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
nyr2 = 48
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nt2 = ndump*nyr2
nmem = 12


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

#-- low-pass filter --
fs = ndump*1.0          # sampling freq [yr-1]
cof = 1.0               # cut-off freq [yr-1]
b, a = signal.butter(10, cof/(fs/2), btype='low')
moc_rapid_lpf = signal.filtfilt(b, a, amoc_ano_rapid)



#---------------------------
# Load (time processed) AMOC
#---------------------------

#-- load time processes AMOC ; organized as [nmem, nt, nr, ny] --
moc_26n_4conf = np.zeros([nconf+1, nmem, nt2])
imem = 0
moc_1m_4conf  = np.zeros([nconf+1, nt2])
for iconf in range(0,nconf):
  print("conf: " + config[iconf])
  fileN1 = 'MOCyzt_' + config[iconf] + '_ensemble_detrend_1ylpf.bin'
  f = open(dir_in+config[iconf]+'/'+fileN1,'r')
  tmp_moc = np.fromfile(f,'>f4').reshape([nmem, nt, nr, ny])      #big-indian ('>'), real*4 ('f4')
  f.close()
  moc_26n_4conf[iconf, :, :] = tmp_moc[:, ndump:-ndump, kdepth, jj26n]
  moc_1m_4conf[iconf, :]  = tmp_moc[imem, ndump:-ndump, kdepth, jj26n]

#-- construct ensemble mean --
moc_f = moc_26n_4conf.mean(1)

#-- construct ensemble spread --
moc_i = moc_26n_4conf - np.tile(moc_f[:, np.newaxis, :], (1, nmem, 1))
moc_std = np.std(moc_i, 1)

#-- make reconstruction from ORAC+OCAR --
moc_f[-1, :] = moc_f[1, :] + moc_f[2, :]
moc_1m_4conf[-1, :] = moc_1m_4conf[1, :] + moc_1m_4conf[2, :]

#-- correlation --
r_corr = np.zeros(nconf+1)
r_corr2= np.zeros(nconf+1)
for iconf in range(nconf+1):
  r_corr[iconf] = np.corrcoef(moc_f[0, :], moc_f[iconf, :])[0, 1]
  r_corr2[iconf]= np.corrcoef(moc_1m_4conf[0, :], moc_1m_4conf[iconf, :])[0, 1]


#-- periodogram --
fs = ndump*1.0          # sampling freq [yr-1]
#nfft = 1753
#pxx_proc = np.zeros([nconf,nfft])
#for iconf in range(0,nconf+1):
f1, pxx_proc = signal.periodogram(moc_f, fs, scaling='density',axis=-1)
f1, pxx_1m   = signal.periodogram(moc_1m_4conf, fs, scaling='density',axis=-1)
nfft = pxx_proc.shape[1]

# smooth PSD 
#from: https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

pxx_smooth  = np.zeros([nconf+1,nfft])
pxx2_smooth = np.zeros([nconf+1,nfft])
span = 5
for iconf in range(nconf+1):
  pxx_smooth[iconf, :]  = smooth(pxx_proc[iconf, :], span)
  pxx2_smooth[iconf, :] = smooth(pxx_1m[iconf, :], span)



#---------------------------
#	PLOT
#---------------------------

fig1 = plt.figure(figsize=(15,8))
#-- time series --
ax1 = fig1.add_subplot(2, 3, (1,2))
#ax1 = fig1.add_subplot(1, 3, (1,2))
ax1.plot(time_ens[ndump:-ndump], np.zeros(nt-2*ndump), 'k', alpha=0.5)
ttt = np.where(time_rapid>=2005.0)[0]
p00 = ax1.plot(time_rapid[ttt], moc_rapid_lpf[ttt], 'm', alpha=0.8, linewidth=1)
p0 = ax1.plot(time_ens[ndump:-ndump], moc_f[0,:], color='k')
p1 = ax1.plot(time_ens[ndump:-ndump], moc_f[1,:], color='r')
p2 = ax1.plot(time_ens[ndump:-ndump], moc_f[2,:], color='b')
p3 = ax1.plot(time_ens[ndump:-ndump], moc_f[3,:], color='g')
ax1.legend((p0[0], p1[0], p2[0], p3[0]), \
	(config[0].upper(),config[1].upper(),config[2].upper(),config[3].upper()))
ax1.grid()
ax1.set_xlim(1964, 2012)
ax1.set_ylim(-5, 5)
#plt.title('Yearly AMOC anomalies time series at 26.5$^{\circ}$N, 1200 m', fontsize='xx-large')
ax1.set_xlabel('Time [yr]', fontsize='x-large')
ax1.set_ylabel('AMOC anomalies at 26.5$^{\circ}$N [Sv]', fontsize='x-large')
#-- spectra --
ax2 = fig1.add_subplot(2, 3, 3)
#ax2 = fig1.add_subplot(1, 3, 3)
p0 = ax2.semilogx(1/f1, pxx_smooth[0,:], color='k')
p1 = ax2.semilogx(1/f1, pxx_smooth[1,:], color='r')
p2 = ax2.semilogx(1/f1, pxx_smooth[2,:], color='b')
p3 = ax2.semilogx(1/f1, pxx_smooth[3,:], color='g')
#ax2.legend((p0[0], p1[0], p2[0], p3[0]), (config[0].upper(),config[1].upper(),config[2].upper(),config[3].upper()))
ax2.grid()
ax2.set_xlim(0.5, 50)
ax2.set_ylim(0, 4)
ax2.set_xlabel('Period [yr]', fontsize='x-large')
ax2.set_ylabel('PSD [Sv$^2$ yr]', fontsize='x-large')
ax2.invert_xaxis()
#-- reconstruction --
ax3 = fig1.add_subplot(2, 3, (4,5))
p00 = ax3.fill_between(time_ens[ndump:-ndump], moc_f[0,:]-moc_std[0, :], moc_f[0,:]+moc_std[0,:], \
    alpha=0.3, facecolor='k',linewidth=0)
ax3.plot(time_ens[ndump:-ndump], np.zeros(nt-2*ndump), 'k', alpha=0.5)
#p2 = ax3.plot(time_ens[ndump:-ndump], moc_1m_4conf[-1,:], color='m', alpha=0.5)
p0 = ax3.plot(time_ens[ndump:-ndump], moc_f[0,:], color='k')
p1 = ax3.plot(time_ens[ndump:-ndump], moc_f[-1,:], color='c')
#ax3.legend((p0[0], p1[0], p2[0]), ('<ORAR> ($\pm$ 1 std)', \
#	'<ORAC>+<OCAR>', r'ORAC$_{memb\#00}$+OCAR$_{memb\#00}$'))
ax3.legend((p0[0], p1[0]), ('ORAR ($\pm$ 1 std)', 'ORAC+OCAR'))
ax3.grid()
ax3.set_xlim(1964,2012)
ax3.set_ylim(-5, 5)
ax3.set_xlabel('Time [yr]', fontsize='x-large')
ax3.set_ylabel('AMOC anomalies [Sv]', fontsize='x-large')
#-- associated spectra --
ax4 = fig1.add_subplot(2, 3, 6)
#p0 = ax4.semilogx(1/f1, pxx2_smooth[-1,:], color='m', alpha=0.5)
p0 = ax4.semilogx(1/f1, pxx_smooth[0,:], color='k')
p1 = ax4.semilogx(1/f1, pxx_smooth[-1,:], color='c')
ax4.grid()
ax4.set_xlim(0.5, 50)
ax4.set_ylim(0, 4)
ax4.set_xlabel('Period [yr]', fontsize='x-large')
ax4.set_ylabel('PSD [Sv$^2$ yr]', fontsize='x-large')
ax4.invert_xaxis()
#-- save --
plt.tight_layout()
fileN1 = 'moc26n_tseries_psd_smmoth_4conf_and_reconstruction_2'
#fileN1 = 'moc34n_tseries_psd_smmoth_4conf_and_reconstruction'
#fileN1 = 'moc26n_tseries_psd_smmoth_4conf'
#fileN1 = 'moc26n_tseries_psd_smmoth_4conf_and_reconstruction_ensMean_oneMemb'
fig1.savefig(dir_fig+fileN1 + '.pdf',dpi=300)
fig1.savefig(dir_fig+fileN1 + '.png',dpi=300)






fig2 = plt.figure(figsize=(12,4))
#-- time series --
ax1 = fig2.add_subplot(1, 1, 1)
ax1.plot(time_ens[ndump:-ndump], np.zeros(nt-2*ndump), 'k', alpha=0.5)
ttt = np.where(time_rapid>=2005.0)[0]
p00 = ax1.plot(time_rapid[ttt], moc_rapid_lpf[ttt], 'm', alpha=0.8, linewidth=1)
p0 = ax1.plot(time_ens[ndump:-ndump], moc_f[0,:], color='k')
p1 = ax1.plot(time_ens[ndump:-ndump], moc_f[1,:], color='r')
p2 = ax1.plot(time_ens[ndump:-ndump], moc_f[2,:], color='b')
p3 = ax1.plot(time_ens[ndump:-ndump], moc_f[3,:], color='g')
ax1.legend((p0[0], p1[0], p2[0], p3[0], p00[0]), \
        (config[0].upper(),config[1].upper(),config[2].upper(),config[3].upper(), 'RAPID'))
ax1.grid()
ax1.set_xlim(1964, 2012)
ax1.set_ylim(-6, 6)
#plt.title('Yearly AMOC anomalies time series at 26.5$^{\circ}$N, 1200 m', fontsize='xx-large')
ax1.set_xlabel('Time [yr]', fontsize='x-large')
ax1.set_ylabel('AMOC anomalies at 26.5$^{\circ}$N [Sv]', fontsize='x-large')
#-- save --
fileN2 = 'moc26n_tseries_psd_smmoth_4conf_and_reconstruction_RAPID'
fig2.savefig(dir_fig+fileN2 + '.pdf',dpi=300)
fig2.savefig(dir_fig+fileN2 + '.png',dpi=300)

