import numpy as np
from multiprocessing import Pool
import statsmodels.api as sm    # for LOESS (or LOWESS) smoothing
import time
from scipy import signal
import MITgcmutils as mit
import matplotlib.pyplot as plt

plt.ion()

config=['orar','ocar','orac','ocac']
nconf = len(config)
dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/forced_amoc/'

#-- load grid --
yG = mit.rdmds(dir_grd + 'YG')
rF = mit.rdmds(dir_grd + 'RF')
ny, nx = yG.shape
nr = len(rF)-1
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0]
#kdepth = np.where( np.abs(rF+3000) == np.min(np.abs(rF+3000)) )[0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nmem = 12


#---------------------------
# Load (time processed) AMOC
#---------------------------

#-- load time processes AMOC ; organized as [nmem, nt, nr, ny] --
moc_kdepth_4conf = np.zeros([nconf, nmem, nt, ny])
for iconf in range(0,nconf):
  print("conf: " + config[iconf])
  fileN1 = 'MOCyzt_' + config[iconf] + '_ensemble_detrend_1ylpf.bin'
  f = open(dir_in+config[iconf]+'/'+fileN1,'r')
  tmp_moc = np.fromfile(f,'>f4').reshape([nmem, nt, nr, ny])      #big-indian ('>'), real*4 ('f4')
  f.close()
  moc_kdepth_4conf[iconf, :, :, :] = np.squeeze(tmp_moc[:, :, kdepth, :])


#-- construct the forced signal --
#(first an last years discarded due to side effects of the time processing)
moc_f = np.mean(moc_kdepth_4conf[:, :, ndump:-ndump, :], 1)

#-- make reconstruction from ORAC+OCAR --
tmp_rec = moc_f[1, :, :] + moc_f[2, :, :]
moc_f = np.concatenate((moc_f, tmp_rec[np.newaxis, :, :]), axis=0)

#-- periodogram --
fs = ndump*1.0          # sampling freq [yr-1]
f1, pxx_proc = signal.periodogram(moc_f, fs, scaling='density', axis=1)

#- smooth PSD -
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

nfft = pxx_proc.shape[1]
pxx_smooth = np.zeros([nconf+1, nfft, ny])
span = 5
for iconf in range(nconf+1):
 for jj in range(ny):
  pxx_smooth[iconf, :, jj] = smooth(pxx_proc[iconf, :, jj],span)

#-- residual of spectra ORAR-(ORAC+OCAR) --
pxx_resid = pxx_smooth[0, :, :] - pxx_smooth[-1, :, :]

#>>>>>>>>>>>>>>>>> check at a given lat. --
ylat = 37
#jj = np.where(np.abs(yG[:,1]-ylat) == np.min(np.abs(yG[:,1]-ylat)))[0]
#jj = 693 # to be kept
jj = 1 # to be kept
fig1 = plt.figure(figsize=(14,5))
#-- time series --
ax1 = fig1.add_subplot(1, 3, (1,2))
p0 = ax1.plot(time_ens[ndump:-ndump], np.squeeze(moc_f[0, :, jj]), color='k')
p1 = ax1.plot(time_ens[ndump:-ndump], np.squeeze(moc_f[1, :, jj]), color='r')
p2 = ax1.plot(time_ens[ndump:-ndump], np.squeeze(moc_f[2, :, jj]), color='b')
p3 = ax1.plot(time_ens[ndump:-ndump], np.squeeze(moc_f[3, :, jj]), color='g')
ax1.legend((p0[0], p1[0], p2[0], p3[0]),(config[0].upper(),config[1].upper(),config[2].upper(),config[3].upper()))
ax1.grid()
ax1.set_xlim(1964,2012)
ax1.set_xlabel('Time [yr]', fontsize='x-large')
#-- spectra --
ax2 = fig1.add_subplot(1, 3, 3)
p0 = ax2.semilogx(1/f1, np.squeeze(pxx_smooth[0, :, jj]), color='k')
p1 = ax2.semilogx(1/f1, np.squeeze(pxx_smooth[1, :, jj]), color='r')
p2 = ax2.semilogx(1/f1, np.squeeze(pxx_smooth[2, :, jj]), color='b')
p3 = ax2.semilogx(1/f1, np.squeeze(pxx_smooth[3, :, jj]), color='g')
ax2.legend((p0[0], p1[0], p2[0], p3[0]),(config[0].upper(),config[1].upper(),config[2].upper(),config[3].upper()))
ax2.grid()
ax2.set_xlim(0.5,50)
ax2.set_xlabel('Period [yr]', fontsize='x-large')
ax2.set_ylabel('PSD [Sv$^2$ yr]', fontsize='x-large')
ax2.set_title('PSD at %0.0fN' %ylat  )
ax2.invert_xaxis()
plt.show()
figN1 = 'amoc_kdepth_psd_forced_4conf_37.8N_jj693'
fig1.savefig(dir_fig+figN1+'.pdf', dpi=300)
fig1.savefig(dir_fig+figN1+'.png', dpi=300)
plt.close(fig1)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



#---------------------------
#	PLOT
#---------------------------

#-- spectral maps --
# with contourf: display in frequence, turn to log scale and re-arrange x-ticks
# since invert_xaxis does crap with contourf
xx = f1[np.where(1/f1 >= 1)[0]]
nx = len(xx)
yy = yG[:, 0]
#- define x-ticks/lables in log -
xmin = np.ceil(np.log10(np.abs(xx[1,]))) - 1
xmax = np.ceil(np.log10(np.abs(xx[-1,])))
new_major_ticks = 10 ** np.arange(xmin + 1, xmax+1, 1)
new_major_ticklabels = (1. / new_major_ticks)
A = np.arange(2, 10, 2)[np.newaxis]
B = 10 ** (np.arange(-xmax, -xmin, 1)[np.newaxis])
C = np.dot(B.transpose(), A)
new_minor_ticklabels = C.flatten()[:6]
new_minor_ticks = 1. / new_minor_ticklabels

fig1 = plt.figure(figsize=(10,9))
cmap = "hot_r"
cmap2 = "RdBu_r"
llev = np.arange(0, 4.5, 0.5)
llev2 = np.arange(-2, 2.5, 0.5)
ip=0
for i in range(2):
 for j in range(2):
  if ip == 3:
   zz = np.transpose(pxx_resid[0:nx, :], (1, 0))
   ax = fig1.add_subplot(2, 2, ip+1)
   cs2 = ax.contourf(xx, yy, zz, levels=llev2, cmap=cmap2, extend="both")
   ax.set_title('Residual PSD')
  else:
   zz = np.transpose(pxx_smooth[ip, 0:nx, :], (1, 0))
   ax = fig1.add_subplot(2, 2, ip+1)
   cs = ax.contourf(xx, yy, zz, levels=llev, cmap=cmap, extend="max")
   if ip == 0:
    zz2 = np.transpose(pxx_smooth[-1, 0:nx, :], (1, 0))
    cs3 = ax.contour(xx, yy, zz2, levels=llev, colors=[(0.8, 0.8, 0.8)], linewidths=1)
   ax.set_title(config[ip].upper())
  ax.plot(xx, np.ones(nx)*26.5, 'k--')
  if j == 0:
    ax.set_ylabel('Latitude')
  if i == 1:
    ax.set_xlabel('Period [yr]')
  #- from Guillaume Sérazin -
  ax.set_xscale('log', nonposx='clip')
  ax.set_xticks(new_major_ticks)
  ax.set_xticklabels(new_major_ticklabels, rotation=60, fontsize=10)
  ax.set_xticks(new_minor_ticks, minor=True)
  ax.set_xticklabels(new_minor_ticklabels, minor=True, rotation=60, fontsize=8)
  ax.grid(True, which='both',linestyle=":",color="k")
  ip = ip+1

#cbaxes = fig1.add_axes([0.92, 0.25, 0.01, 0.5]) 
cbaxes = fig1.add_axes([0.12, 0.95, 0.8, 0.01]) 
cbaxes2 = fig1.add_axes([0.92, 0.14, 0.01, 0.3])
cbar = plt.colorbar(cs, orientation='horizontal', cax=cbaxes)
cbar.set_label('PSD [Sv$^2$ yr]')
cbar2 = plt.colorbar(cs2, orientation='vertical', cax=cbaxes2)
cbar2.set_label('[Sv$^2$ yr]')

#plt.show()
figN1 = 'amoc_forced_4conf_psd_maps_kdepth_2'
fig1.savefig(dir_fig+figN1+'.pdf', dpi=300)
fig1.savefig(dir_fig+figN1+'.png', dpi=300)
plt.close(fig1)

