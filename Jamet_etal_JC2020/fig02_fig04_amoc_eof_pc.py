import numpy as np
import MITgcmutils as mit
import matplotlib.pyplot as plt
#import scipy.io as sio		# to load .mat file
from scipy import signal
import pickle

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
jj26n = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0]
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 48		#first and last year removed for PCA (lpf contamination)
time_ens = 1964 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nmem = 12

#------------------------------
# load pre-processed MOC EOF/PC
# files contain eof, pc, expvar, totvar
#------------------------------
neof = 10
eof_4conf = np.zeros([nconf, neof, nr, ny])
pc_4conf = np.zeros([nconf, neof, nt]) 
expvar_4conf = np.zeros([nconf, neof])
#- from .mat files -
#for iconf in range(0,nconf):
#  mat_cont = sio.loadmat(dir_in+config[iconf]+'/AMOC_processed_eofpc_' + config[iconf] + '_1ylpf.mat')
#  eof_4conf[iconf, :, :, :] = np.transpose(mat_cont['eof_ens'][:, :, :, -1], (2, 1, 0))
#  pc_4conf[iconf, :, :] = np.transpose(mat_cont['pc_ens'][:, :, -1], (1, 0))
#  expvar_4conf[iconf, :] = mat_cont['expvar_ens'][:,-1]

#- from eof computed with python -
for iconf in range(0,nconf):
  tmp_fileN = 'AMOC_processed_eofpc_' + config[iconf] + '_1ylpf_py.bin'
  tmp_dir = dir_in + config[iconf] + '/'
  with open(tmp_dir + tmp_fileN, 'rb') as f:
    eofpc = pickle.load(f)
  #
  eof_4conf[iconf, :, :, :] = eofpc[-1][0]
  pc_4conf[iconf, :, :] = eofpc[-1][1]
  expvar_4conf[iconf, :] = eofpc[-1][2]*100.0


  
#-- make sign consistent -
sign_eof = np.sign(eof_4conf[:, 0, kdepth, jj26n])
eof_4conf = eof_4conf * np.tile( sign_eof[:, :, np.newaxis, np.newaxis], (1, neof, nr, ny) )
pc_4conf = pc_4conf * np.tile( sign_eof[:, :, np.newaxis], (1, neof, nt) )

eof_4conf[np.where(eof_4conf == 0)] = np.nan


#---------------------------
#       Spectrum
#---------------------------
#-- PCs have been normalized, reput some signal for spectrum comparison 
# by multiplying the PC with EOF magntitude at rapid location --
# useless for spectrum but good for looking at PCs

ieof = 0
#tmp_pc = pc_4conf[:, ieof, :] * np.tile(eof_4conf[:, ieof, kdepth, jj26n], (1, nt)) 
#tmp_pc = pc_4conf[:, ieof, :] 
tmp_pc = pc_4conf[:, ieof, :] * np.tile( np.nanmax(np.nanmax(eof_4conf[:, ieof, :, :], 2), 1)[:, np.newaxis], (1, nt)) 
nfft = nt/2+1
fs = ndump*1.0          # sampling freq [yr-1]
pxx_pc = np.zeros([nconf,nfft])
for iconf in range(0,nconf):
  f1, pxx_pc[iconf,:] = signal.periodogram(tmp_pc[iconf, :], fs, scaling='density')


#- smooth PSD -
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

pxx_smooth = np.zeros([nconf,nfft])
span = 5
for iconf in range(0,nconf):
  pxx_smooth[iconf,:] = smooth(pxx_pc[iconf,:],span)


#---------------------------
#	PLOT
#---------------------------

#-- EOF1 4 conf --
cmap = "RdBu_r"
fig1, axs1 = plt.subplots(2, 2)
fig1.set_size_inches(12,7)
#fig1.suptitle('Leading mode (EOF1) of the forced AMOC interannual-to-decadal variability', fontsize='xx-large')
levels = np.arange(-1.2, 1.3, 0.1)
images = []
ip=0
for i in range(2):
 for j in range(2):
  images.append(axs1[i, j].contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, eof_4conf[ip, ieof, :, :], levels, cmap=cmap))
  images.append(axs1[i, j].contour(yG[:, 0], np.squeeze(rF[0:-1])/1000, eof_4conf[ip, ieof, :, :], 0 , colors='k'))
  images.append(axs1[i, j].plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000, 'k--'))
  images.append(axs1[i, j].plot(26.5, np.squeeze(rF[kdepth])/1000, 'ko'))
  axs1[i, j].label_outer()
  axs1[i, j].set_facecolor((0.5, 0.5, 0.5))
  axs1[i, j].set_title(config[ip].upper() + '(' + str("%.01f" % expvar_4conf[ip, ieof]) + '%)')
  ip = ip+1
  
#- set label --
axs1[1, 0].set_xlabel('Latitude')
axs1[1, 1].set_xlabel('Latitude')
axs1[0, 0].set_ylabel('Depth [km]')
axs1[1, 0].set_ylabel('Depth [km]')

cbar = fig1.colorbar(images[0], ax=axs1, orientation='vertical', fraction=.02)
cbar.set_label('[Sv]')

#plt.show()

figN1 = 'AMOC_forced_eof' + "%.00f" %(ieof+1) + '_4conf'
fig1.savefig(dir_fig+figN1+'.pdf', dpi=300)
fig1.savefig(dir_fig+figN1+'.png', dpi=300)
plt.close(fig1)




fig2 = plt.figure(figsize=(15,5))
#-- time series --
ax1 = fig2.add_subplot(1, 3, (1,2))
p0 = ax1.plot(time_ens,tmp_pc[0,],'k')
p1 = ax1.plot(time_ens,tmp_pc[1,],'r')
p2 = ax1.plot(time_ens,tmp_pc[2,],'b')
p3 = ax1.plot(time_ens,tmp_pc[3,],'g')
ax1.grid()
#plt.title('First Principal Component (PC1)', fontsize='xx-large')
ax1.set_xlabel('Time [yr]', fontsize='x-large')
ax1.set_ylabel('PC' + "%.00f" %(ieof+1) + '*max(EOF' + "%.00f" %(ieof+1) + ') [Sv]', fontsize='x-large')
ax1.set_xlim(1964, 2012)
ax1.set_ylim(-4.5, 4.5)
ax1.legend((p0[0], p1[0], p2[0], p3[0]), \
        (config[0].upper() + '(' + str("%.01f" % expvar_4conf[0, ieof]) + '%)',\
         config[1].upper() + '(' + str("%.01f" % expvar_4conf[1, ieof]) + '%)',\
         config[2].upper() + '(' + str("%.01f" % expvar_4conf[2, ieof]) + '%)',\
         config[3].upper() + '(' + str("%.01f" % expvar_4conf[3, ieof]) + '%)'))
#-- associated spectra --
ax2 = fig2.add_subplot(1, 3, 3)
p0 = ax2.semilogx(1/f1, pxx_smooth[0,:],color='k')
p1 = ax2.semilogx(1/f1, pxx_smooth[1,:],color='r')
p2 = ax2.semilogx(1/f1, pxx_smooth[2,:],color='b')
p3 = ax2.semilogx(1/f1, pxx_smooth[3,:],color='g')
ax2.grid()
#plt.title('Power Spectral Density of PC1', fontsize='xx-large')
ax2.set_xlabel('Period [yr]', fontsize='x-large')
ax2.set_ylabel('PSD [Sv$^2$ yr]', fontsize='x-large')
ax2.set_xlim(0.5, 50)
ax2.set_ylim(0, 4)
ax2.invert_xaxis()
#-- save --
plt.tight_layout()
plt.show()
figN2 = 'AMOC_forced_pc' + "%.00f" %(ieof+1) + '_psd_4conf'
fig2.savefig(dir_fig + figN2 + '.pdf', dpi=300)
fig2.savefig(dir_fig + figN2 + '.png', dpi=300)
plt.close(fig2)


