import numpy as np
from multiprocessing import Pool
import statsmodels.api as sm    # for LOESS (or LOWESS) smoothing
import time
import MITgcmutils as mit
import matplotlib.pyplot as plt
from scipy import signal



config=['orac','ocac']
nconf = 4
dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/forced_amoc/'

#-- load grid --
yG = mit.rdmds(dir_grd + 'YG')
rF = mit.rdmds(dir_grd + 'RF')
ny, nx = yG.shape
nr = len(rF)-1
jj26n = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0][0]
jj30n = np.where(np.abs(yG[:,1]-30) == np.min(np.abs(yG[:,1]-30)))[0][0]
jj00n = np.where(np.abs(yG[:,1]) == np.min(np.abs(yG[:,1])))[0][0]
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0]
#kdepth = np.where( np.abs(rF+3000) == np.min(np.abs(rF+3000)) )[0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nt2 = ndump*(nyr-2)
time_ens2 = time_ens[ndump:-ndump]
nmem = 12


#-----------------------------------------------------------------------------
# First identify the member of OCAC that has the largest intrinsic variability
#-----------------------------------------------------------------------------

fileN0 = 'MOCyzt_ocac_ensemble_detrend_1ylpf.bin'
f = open(dir_in +'ocac/' + fileN0,'r')
moc_ocac = np.squeeze(np.fromfile(f,'>f4').reshape([nmem, nt, nr, ny])[:, ndump:-ndump, kdepth, jj00n:jj30n])
f.close()

moc_i = moc_ocac - np.tile( moc_ocac.mean(0)[np.newaxis, :, :], (nmem, 1, 1) )
var_moc = moc_i.var(1)
imem = np.where( var_moc.mean(1) == var_moc.mean(1).max() )[0][0]

#plt.plot(np.transpose(var_moc, (1, 0)), yG[jj00n:jj30n, 0])
#plt.plot(var_moc[imem, :], yG[jj00n:jj30n, 0], 'k')
#plt.grid()
#plt.show()


#---------------------------
# Load (time processed) AMOC
#---------------------------

#-- load time processes AMOC ; organized as [nmem, nt, nr, ny] --
moc_kdepth   = np.zeros([nconf, nt2, ny])	
for iconf in range(2):
  print("conf: " + config[iconf])
  fileN1 = 'MOCyzt_' + config[iconf] + '_ensemble_detrend_1ylpf.bin'
  f = open(dir_in+config[iconf]+'/'+fileN1,'r')
  tmp_moc = np.fromfile(f,'>f4').reshape([nmem, nt, nr, ny])      #big-indian ('>'), real*4 ('f4')
  f.close()
  if iconf == 0:
    moc_kdepth[iconf, :, :] = np.squeeze(tmp_moc[:, ndump:-ndump, kdepth, :].mean(0))
  else:
   moc_kdepth[iconf, :, :] = np.squeeze(tmp_moc[imem, ndump:-ndump, kdepth, :])

#-- load runN and runS --
# runN
fileIN = 'MOCyzt_ocac_runN_detrend_1ylpf.bin'
f = open(dir_in + config[-1] + '/' + fileIN)
tmp_moc = np.fromfile(f,'>f4').reshape([nt, nr, ny])
f.close()
moc_kdepth[2, :, :] = np.squeeze(tmp_moc[ndump:-ndump, kdepth, :])

# runS
fileIN = 'MOCyzt_ocac_runS_detrend_1ylpf.bin'
f = open(dir_in + config[-1] + '/' + fileIN)
tmp_moc = np.fromfile(f,'>f4').reshape([nt, nr, ny])
f.close()
moc_kdepth[3, :, :] = np.squeeze(tmp_moc[ndump:-ndump, kdepth, :])


#-----------------------------------------------------------------
# Load unprocessed AMOC for filtered very low-fq in time procesing
#-----------------------------------------------------------------

moc_detrend = np.zeros([nconf, nt])
for iconf in range(2):
  print("conf: " + config[iconf])
  fileN1 = 'MOCyzt_' + config[iconf] + '_ensemble.bin'
  f = open(dir_in+config[iconf]+'/'+fileN1,'r')
  tmp_moc = np.fromfile(f,'>f4').reshape([nyr, nmem, ndump, nr, ny])
  tmp_moc = np.transpose(tmp_moc, (1, 0, 2, 3, 4)).reshape([nmem, nt, nr, ny])
  f.close()
  if iconf == 0:
    moc_detrend[iconf, :] = np.squeeze(tmp_moc[:, :, kdepth, jj26n].mean(0))
  else:
    moc_detrend[iconf, :] = np.squeeze(tmp_moc[imem, :, kdepth, jj26n])

#-- load runN and runS --
# runN
fileIN = 'MOCyzt_ocac_runN.bin'
f = open(dir_in + config[-1] + '/' + fileIN)
tmp_moc = np.fromfile(f,'>f4').reshape([nt, nr, ny])
f.close()
moc_detrend[2, :] = np.squeeze(tmp_moc[:, kdepth, jj26n])

# runS
fileIN = 'MOCyzt_ocac_runS.bin'
f = open(dir_in + config[-1] + '/' + fileIN)
tmp_moc = np.fromfile(f,'>f4').reshape([nt, nr, ny])
f.close()
moc_detrend[3, :] = np.squeeze(tmp_moc[:, kdepth, jj26n])


#-- apply detrending --
lowess = sm.nonparametric.lowess
moc_detrend2 = np.zeros([nconf, nt])
for iconf in range(4):
  moc_detrend2[iconf, :] = lowess(moc_detrend[iconf, :], time_ens, return_sorted=False)


#-----------------------
#	Spectra
#-----------------------
fs = ndump*1.0          # sampling freq [yr-1]

#- unfiltered single simu -
f1, pxx_moc = signal.periodogram(moc_kdepth, fs, scaling='density',axis=1)
nfft = pxx_moc.shape[1]

# smooth PSD
#from: https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

pxx_smooth = np.zeros([nconf, nfft, ny])
span = 5
for iconf in range(nconf):
  for jj in range(ny):
    pxx_smooth[iconf, :, jj] = smooth(pxx_moc[iconf, :, jj], span)


#---------------------------
#	PLOT
#---------------------------

zz = moc_kdepth
tmp_pxx = pxx_smooth
ttitle = ['<ORAC>', 'OCAC (memb#02)', 'runN', 'runS']
figN1 = 'amoc_hovmuller_kdepth_ORACm_OCACm02_runNS'


#-- Hovmuller diag at 1200 m --
cmap = "RdBu_r"
fig1, axs1 = plt.subplots(2, 2)
fig1.set_size_inches(10,9)
#fig1.suptitle('Hovmuller diag. of AMOC at 1200 m', fontsize='xx-large')
levels = np.arange(-2, 2.5, 0.5)
images = []
ip=0
for i in range(2):
 for j in range(2):
   images.append(axs1[i, j].contourf(time_ens2, yG[:, 0],  np.transpose(zz[ip, :, :], (1, 0)), \
        levels=levels, cmap=cmap, extend="both"))
   images.append(axs1[i, j].plot(time_ens2, np.ones(nt2)*26.5, 'k--'))
   axs1[i, j].label_outer()
   axs1[i, j].set_title(ttitle[ip])
   axs1[i, j].set_xlim(1964, 2012)
   ip = ip+1

#- set label --
axs1[1, 0].set_xlabel('Time [yr]', fontsize='x-large')
axs1[1, 1].set_xlabel('Time [yr]', fontsize='x-large')
axs1[0, 0].set_ylabel('Latitude', fontsize='x-large')
axs1[1, 0].set_ylabel('Latitude', fontsize='x-large')

cbar = fig1.colorbar(images[0], ax=axs1, orientation='vertical', fraction=.02)
cbar.set_label('[Sv]', fontsize='x-large')

#plt.show()

fig1.savefig(dir_fig+figN1+'.pdf', dpi=300)
fig1.savefig(dir_fig+figN1+'.png', dpi=300)
plt.close(fig1)


#-- time series at 26N --
fig1 = plt.figure(figsize=(12,6))
#-- time series --
ax1 = fig1.add_subplot(1, 3, (1,2))
ax1.plot(time_ens2, np.zeros(nt2), 'k', alpha=0.5)
p00 = ax1.plot(time_ens2, zz[2, :, jj26n]+zz[3, :, jj26n], color='k', alpha=0.5)
p0 = ax1.plot(time_ens2, zz[0, :, jj26n], color='k')
p1 = ax1.plot(time_ens2, zz[1, :, jj26n], color='r')
p2 = ax1.plot(time_ens2, zz[2, :, jj26n], color='b')
p3 = ax1.plot(time_ens2, zz[3, :, jj26n], color='g')
ax1.legend((p0[0], p1[0], p2[0], p3[0]), (ttitle[0], ttitle[1], ttitle[2], ttitle[3]))
ax1.grid()
ax1.set_xlim(1964, 2012)
ax1.set_ylim(-4.5, 4.5)
ax1.set_xlabel('Time [yr]', fontsize='x-large')
ax1.set_ylabel('AMOC anomalies at 26.5$^{\circ}$N [Sv]', fontsize='x-large')
#-- spectra --
ax2 = fig1.add_subplot(1, 3, 3)
p0 = ax2.semilogx(1/f1, tmp_pxx[0,:, jj26n], color='k')
p1 = ax2.semilogx(1/f1, tmp_pxx[1,:, jj26n], color='r')
p2 = ax2.semilogx(1/f1, tmp_pxx[2,:, jj26n], color='b')
p3 = ax2.semilogx(1/f1, tmp_pxx[3,:, jj26n], color='g')
ax2.legend((p0[0], p1[0], p2[0], p3[0]), ('ORAC (m00)', 'OCAC (m00)', 'runN', 'runS') )
ax2.grid()
ax2.set_xlim(0.5, 50)
ax2.set_ylim(0, 4)
ax2.set_xlabel('Period [yr]', fontsize='x-large')
ax2.set_ylabel('PSD [Sv$^2$ yr]', fontsize='x-large')
ax2.invert_xaxis()

plt.show()


