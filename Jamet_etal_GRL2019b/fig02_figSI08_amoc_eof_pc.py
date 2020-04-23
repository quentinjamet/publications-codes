import numpy as np
import matplotlib.pyplot as plt
import MITgcmutils as mit
import pickle
from scipy import signal


# directories
dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/orar/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/'


#-- load grid --
yG = mit.rdmds(dir_grd + 'YG')
rF = mit.rdmds(dir_grd + 'RF')
ny = yG.shape[0]
nr = len(rF)-1
jj26n = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0]
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*(nyr-2)	# first and last years removed due to filter contamination 
nmem = 24


#-------------------------------------------------
# Load pre-processed AMOC EOF/PC 
#-------------------------------------------------
fileN1 = 'AMOC_processed_eofpc_orar_24memb.bin'
with open(dir_in + fileN1, 'rb') as f:
    eofpc = pickle.load(f)

neof = 10
eof_ens = np.zeros([nmem+1, neof, nr, ny])
pc_ens = np.zeros([nmem+1, neof, nt])
expvar_ens = np.zeros([nmem+1, neof])
for imem in range(0,nmem+1):
  eof_ens[imem, :, :, :] = eofpc[imem][0]
  pc_ens[imem, :, :] = np.transpose(eofpc[imem][1], (1, 0))
  expvar_ens[imem, :] = eofpc[imem][2]*100.0

del eofpc

#-- make signs consistent --
sign_eof = np.sign(eof_ens[:, 0, kdepth, jj26n])
eof_ens = eof_ens * np.tile(sign_eof[:, :, np.newaxis, np.newaxis], (1, neof, nr, ny)) 
#- change PCs accordingly -
pc_ens = pc_ens * np.tile(sign_eof[:, :, np.newaxis], (1, neof, nt))

#-- make sign consistent for the second intrinsic mode --
# this implies that intrinsic EOF1s will now be 'out of phase'
# which indicates that EOF1s and EOF2s does not share the same 'phasing' 
# in each members.
sign_eof2 = np.sign(eof_ens[:, 1, kdepth, jj26n])
eof_ens2 = eof_ens * np.tile(sign_eof2[:, :, np.newaxis, np.newaxis], (1, neof, nr, ny))
pc_ens2 = pc_ens * np.tile(sign_eof2[:, :, np.newaxis], (1, neof, nt))

#---------------------------
#       Spectrum
#---------------------------
#-- PCs have been normalized, such that amplitude of spectra can be directly compared

nfft = nt/2+1
fs = ndump*1.0          # sampling freq [yr-1]
f1, pxx_pc = signal.periodogram(pc_ens, fs, scaling='density', axis=-1)


#- smooth PSD -
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

pxx_smooth = np.zeros([nmem+1, neof, nfft])
span = 5
for imem in range(0, nmem+1):
 for ieof in range(0, neof):
  pxx_smooth[imem, ieof, :] = smooth(pxx_pc[imem, ieof, :], span)


#------------------------------------------------
# Construct forced and intrinsic EOFs and spectra
#------------------------------------------------

#-- forced --
eof_f = eof_ens[-1, :, :, :]
eof_f[np.where(eof_f[:] == 0)] = np.nan
pc_f = pc_ens[-1, :, :]
expvar_f = expvar_ens[-1, :]
pxx_f = pxx_pc[-1, : , :]
pxx_s_f = pxx_smooth[-1, :, :]
#-- intrinsic --
eof_i = np.mean(eof_ens[0:-1, :, :, :], 0)
eof_i[np.where(eof_i[:] == 0)] = np.nan
pc_i = np.mean(pc_ens[0:-1, :, :], 0)
expvar_i = np.mean(expvar_ens[0:-1, :], 0)
pxx_i = np.mean(pxx_pc[0:-1, : , :], 0)
pxx_s_i = np.mean(pxx_smooth[0:-1, :, :], 0)


#-- with consistent sign on the EOF2 --
eof_i_2 = np.mean(eof_ens2[0:-1, :, :, :], 0)
eof_i_2[np.where(eof_i_2[:] == 0)] = np.nan
pc_i_2 = np.mean(pc_ens2[0:-1, :, :], 0)



#---------------------------
#       PLOT
#---------------------------
zz = np.squeeze(rF[0:-1])/1000
cmap = "RdBu_r"


#-- EOF --
fig1, axs1 = plt.subplots(2, 2)
fig1.set_size_inches(12,8)
levels = np.arange(-1.2, 1.25, 0.05)
images = []
#-- forced EOF1 --
images.append(axs1[0, 0].contourf(yG[:, 0], zz, eof_f[0, :, :], levels, cmap=cmap))
images.append(axs1[0, 0].contour(yG[:, 0], zz, eof_f[0, :, :], 0 , colors='k'))
#-- forced EOF2 --
images.append(axs1[0, 1].contourf(yG[:, 0], zz, eof_f[1, :, :], levels, cmap=cmap))
images.append(axs1[0, 1].contour(yG[:, 0], zz, eof_f[1, :, :], 0 , colors='k'))
#-- intrinsic EOF1 --
images.append(axs1[1, 0].contourf(yG[:, 0], zz, eof_i[0, :, :], levels, cmap=cmap))
images.append(axs1[1, 0].contour(yG[:, 0], zz, eof_i[0, :, :], 0 , colors='k'))
#-- intrinsic EOF2 --
images.append(axs1[1, 1].contourf(yG[:, 0], zz, eof_i_2[1, :, :], levels, cmap=cmap))
images.append(axs1[1, 1].contour(yG[:, 0], zz, eof_i_2[1, :, :], 0 , colors='k'))
#-- common --
for jj in range(0, 2):
 for ii in range(0, 2):
  images.append(axs1[jj, ii].plot(26.5*np.ones(nr,), zz, 'k--'))
  axs1[jj, ii].set_facecolor((0.3, 0.3, 0.3))

#-- title and axes --
axs1[0, 0].set_title('EOF1 - Forced (' + str("%.01f" % expvar_f[0]) + '%)', fontsize='xx-large')
axs1[0, 1].set_title('EOF2 - Forced (' + str("%.01f" % expvar_f[1]) + '%)', fontsize='xx-large')
axs1[1, 0].set_title('EOF1 - Intrinsic (' + str("%.01f" % expvar_i[0]) + '%)', fontsize='xx-large')
axs1[1, 1].set_title('EOF2 - Intrinsic (' + str("%.01f" % expvar_i[1]) + '%)', fontsize='xx-large')
axs1[1, 0].set_xlabel('Latitude', fontsize='x-large')
axs1[1, 1].set_xlabel('Latitude', fontsize='x-large')
axs1[0, 0].set_ylabel('Depth [km]', fontsize='x-large')
axs1[1, 0].set_ylabel('Depth [km]', fontsize='x-large')
cbar = fig1.colorbar(images[0], ax=axs1, orientation='vertical', fraction=.02)
cbar.set_label('[Sv]', fontsize='x-large')

plt.show()

figN1 = 'AMOC_forced_intrinsic_eof12_24mem_phase_EOF2'
fig1.savefig(dir_fig+figN1+'.pdf', dpi=300)
fig1.savefig(dir_fig+figN1+'.png', dpi=300)
plt.close(fig1)


#-- spectra --
fig2, axs2 = plt.subplots(1, 2)
fig2.set_size_inches(12,5)
images = []
#- pc1 -
images.append(axs2[0].semilogx(1/f1, pxx_s_f[0, :], color='k'))
images.append(axs2[0].semilogx(1/f1, pxx_s_i[0, :], color='g'))
axs2[0].grid()
axs2[0].set_xlim(0.5, 50)
axs2[0].invert_xaxis()
axs2[0].legend(('FORCED', 'INTRINSIC'))
axs2[0].set_xlabel('Period [yr]', fontsize='x-large')
axs2[0].set_ylabel('Normalized PSD ', fontsize='x-large')
axs2[0].set_title('PC1', fontsize='xx-large')
#- pc2 -
images.append(axs2[1].semilogx(1/f1, pxx_s_f[1, :], color='k'))
images.append(axs2[1].semilogx(1/f1, pxx_s_i[1, :], color='g'))
axs2[1].set_xlim(0.5, 50)
axs2[1].grid()
axs2[1].invert_xaxis()
axs2[1].legend(('FORCED', 'INTRINSIC'))
axs2[1].set_xlabel('Period [yr]', fontsize='x-large')
axs2[1].set_title('PC2', fontsize='xx-large')

plt.show()

figN2 = 'AMOC_forced_intrinsic_eof12_spectra_24mem'
fig2.savefig(dir_fig+figN2+'.pdf', dpi=300)
fig2.savefig(dir_fig+figN2+'.png', dpi=300)
plt.close(fig2)


#-- EOF for each members --
ieof = 1
tmp_eof = eof_ens2[:, ieof, :, :]
tmp_eof[ np.where(tmp_eof == 0) ] = np.nan
fig3, axs3 = plt.subplots(3, 4)
fig3.set_size_inches(18,10)
fig3.suptitle('EOF' + str(ieof+1), fontsize='xx-large')
images = []
ip=0
for i in range(3):
  for j in range(4):
    images.append(axs3[i, j].contourf(yG[:, 0], zz, tmp_eof[ip, :, :], levels, cmap=cmap))
    images.append(axs3[i, j].contour(yG[:, 0], zz, tmp_eof[ip, :, :], 0, colors='k'))
    images.append(axs3[i, j].plot(26.5*np.ones(nr,), zz, 'k--'))
    axs3[i, j].label_outer()
    axs3[i, j].set_title('memb#' + str(ip) + ' (' + str("%.01f" % expvar_ens[ip, ieof]) + '%)')
    axs3[i, j].set_facecolor((0.3, 0.3, 0.3))
    ip = ip + 1


axs3[1, 0].set_ylabel('Depth [km]', fontsize='x-large')
axs3[2, 1].set_xlabel('Latitude', fontsize='x-large')
axs3[2, 2].set_xlabel('Latitude', fontsize='x-large')
cbar = fig3.colorbar(images[0], ax=axs3, orientation='vertical', fraction=.02)
cbar.set_label('[Sv]', fontsize='x-large')

#plt.show()

figN3 = 'AMOC_intrinsic_eof2_12memb'
fig3.savefig(dir_fig+figN3+'.pdf', dpi=300)
fig3.savefig(dir_fig+figN3+'.png', dpi=300)
plt.close(fig3)




#--------------------------------------------------------
# Look at the depth of max amoc only (Leroux et al, 2018)
#--------------------------------------------------------
nt2 = 3650
fileN0 = 'MOCyzt_orar_ensemble.bin'
f = open(dir_in + fileN0,'r')
moc_m = np.mean( np.mean( np.fromfile(f,'>f4').reshape([nmem, nt2, nr, ny]), 0), 0)
f.close()

kkmax = np.zeros(ny)
for jj in range(ny):
  kkmax[jj] = np.where(moc_m[:, jj] == np.max(moc_m[:, jj]))[0]

#- remove surface maxima -
kkmax[np.where(kkmax==8)]=24
kkmax[np.where(kkmax==7)]=25

eof_i_kkmax = np.zeros((10, 900))
for ieof in range(10):
  for jj in range(ny):
    eof_i_kkmax[ieof, jj] = eof_i[ieof, int(kkmax[jj]), jj]


#-- PLOT --
fig4, axs4 = plt.subplots(1, 1)
plt.plot(np.zeros(ny), yG[:, 0],'k--')
p0 = plt.plot(eof_i_kkmax[0, :], yG[:, 0],'k')
p1 = plt.plot(eof_i_kkmax[1, :], yG[:, 0],'b')
plt.legend((p0[0], p1[0]), ('EOF1', 'EOF2'))
plt.grid()
plt.ylabel('Latitude')
plt.xlabel('EOF [Sv]')
plt.ylim((-20, 55))
plt.show()
figN4 = 'AMOC_intrinsic_eof1_2_max_mean_amoc_24mem'
fig4.savefig(dir_fig+figN4+'.pdf', dpi=300)
fig4.savefig(dir_fig+figN4+'.png', dpi=300)
plt.close(fig4)


