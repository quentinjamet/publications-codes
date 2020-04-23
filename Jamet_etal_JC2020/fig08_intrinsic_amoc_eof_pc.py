import numpy as np
import MITgcmutils as mit
import matplotlib.pyplot as plt
#import scipy.io as sio		# to load .mat file
from scipy import signal
import pickle



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
ieof = 0
eof_4conf = np.zeros([nconf, nmem, nr, ny])
pc_4conf = np.zeros([nconf, nmem, nt]) 
expvar_4conf = np.zeros([nconf, nmem])
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
  for imem in range(nmem):
    eof_4conf[iconf, imem, :, :] = eofpc[imem][0][ieof, :, :]
    pc_4conf[iconf, imem, :] = eofpc[imem][1][ieof, :]
    expvar_4conf[iconf, imem] = eofpc[imem][2][ieof]*100.0


  
#-- make sign consistent -
sign_eof = np.sign(eof_4conf[:, :, kdepth, jj26n])
eof_4conf = eof_4conf * np.tile( sign_eof[:, :, :, np.newaxis], (1, 1, nr, ny) )
pc_4conf = pc_4conf * np.tile( sign_eof, (1, 1, nt) )

eof_4conf[np.where(eof_4conf == 0)] = np.nan

#---------------------------
#       Spectrum
#---------------------------
#-- PCs have been normalized, reput some signal for spectrum comparison 
# by multiplying the PC with EOF magntitude at rapid location --
# useless for spectrum but good for looking at PCs

#tmp_pc = pc_4conf
tmp_pc = pc_4conf * np.tile( np.nanmax(np.nanmax(eof_4conf, 3), 2)[:, :, np.newaxis], (1, 1, nt))
nfft = nt/2+1
fs = ndump*1.0          # sampling freq [yr-1]
f1, pxx_pc = signal.periodogram(tmp_pc, fs, scaling='density', axis=-1)


#- smooth PSD -
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

pxx_smooth = np.zeros([nconf, nmem, nfft])
span = 2
for iconf in range(0,nconf):
 for imem in range(nmem):
  pxx_smooth[iconf, imem, :] = smooth(pxx_pc[iconf, imem, :],span)




#---------------------------
#	PLOT
#---------------------------

#-- EOF1 4 conf --
cmap = "RdBu_r"
fig1, axs1 = plt.subplots(2, 2)
fig1.set_size_inches(12,7)
fig1.suptitle('EOF1 intrinsic AMOC variability')
levels = np.arange(-1.2, 1.3, 0.1)
images = []
ip=0
for i in range(2):
 for j in range(2):
  images.append(axs1[i, j].contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, np.mean(eof_4conf[ip, :, :, :], 0), levels, cmap=cmap))
  images.append(axs1[i, j].contour(yG[:, 0], np.squeeze(rF[0:-1])/1000, np.mean(eof_4conf[ip, :, :, :], 0), 0 , colors='k'))
  images.append(axs1[i, j].plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000, 'k--'))
  axs1[i, j].label_outer()
  axs1[i, j].set_facecolor((0.5, 0.5, 0.5))
  axs1[i, j].set_title(config[ip].upper() + '(' + str("%.01f" % np.mean(expvar_4conf[ip, :]) ) + '%)')
  ip = ip+1
  
#- set label --
axs1[1, 0].set_xlabel('Latitude')
axs1[1, 1].set_xlabel('Latitude')
axs1[0, 0].set_ylabel('Depth [km]')
axs1[1, 0].set_ylabel('Depth [km]')

cbar = fig1.colorbar(images[0], ax=axs1, orientation='vertical', fraction=.02)
cbar.set_label('[Sv]')

#plt.show()

figN1 = 'AMOC_intrinsic_eof' + "%.0f" %(ieof+1) + '_4conf'
fig1.savefig(dir_fig+figN1+'.pdf', dpi=300)
fig1.savefig(dir_fig+figN1+'.png', dpi=300)
plt.close(fig1)


#-- plot associated spectra --
fig3 = plt.figure(figsize=(6,5))
p0 = plt.semilogx(1/f1, np.mean(pxx_smooth[0, :, :], 0),color='k')
p1 = plt.semilogx(1/f1, np.mean(pxx_smooth[1, :, :], 0),color='r')
p2 = plt.semilogx(1/f1, np.mean(pxx_smooth[2, :, :], 0),color='b')
p3 = plt.semilogx(1/f1, np.mean(pxx_smooth[3, :, :], 0),color='g')
plt.grid()
#plt.title('Power Spectral Density of PC' + "%.0f" %(ieof+1))
plt.xlabel('Period [yr]')
plt.ylabel('<PSD of PC1*max(EOF1)> [Sv$^2$ yr]')
plt.xlim(0.5,50)
plt.gca().invert_xaxis()
plt.legend((p0[0], p1[0], p2[0], p3[0]), \
        (config[0].upper() + '(' + str("%.01f" %np.mean(expvar_4conf[0, :], 0)) + '%)',\
         config[1].upper() + '(' + str("%.01f" %np.mean(expvar_4conf[1, :], 0)) + '%)',\
         config[2].upper() + '(' + str("%.01f" %np.mean(expvar_4conf[2, :], 0)) + '%)',\
         config[3].upper() + '(' + str("%.01f" %np.mean(expvar_4conf[3, :], 0)) + '%)'))
#plt.show()

figN3 = 'AMOC_intrinsic_pc' + "%.0f" %(ieof+1) + '_maxEOF_psd_4conf'
fig3.savefig(dir_fig+figN3+'.pdf', dpi=300)
fig3.savefig(dir_fig+figN3+'.png', dpi=300)
plt.close(fig3)

