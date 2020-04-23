import numpy as np
from multiprocessing import Pool
import statsmodels.api as sm    # for LOESS (or LOWESS) smoothing
import time
from scipy import signal
import MITgcmutils as mit
import matplotlib.pyplot as plt



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
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0][0]
#kdepth = np.where( np.abs(rF+3000) == np.min(np.abs(rF+3000)) )[0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
nyr2 = 48
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nt2= ndump*nyr2
nmem = 12


#---------------------------
# Load (time processed) AMOC
#---------------------------

#-- load time processes AMOC ; organized as [nmem, nt, nr, ny] --
moc_f_4conf = np.zeros([nconf+1, nt2, nr, ny])
imem = 0
moc_1memb_4conf = np.zeros([nconf+1, nt2, nr, ny])
for iconf in range(0,nconf):
  print("conf: " + config[iconf])
  fileN1 = 'MOCyzt_' + config[iconf] + '_ensemble_detrend_1ylpf.bin'
  f = open(dir_in+config[iconf]+'/'+fileN1,'r')
  tmp_moc = np.fromfile(f,'>f4').reshape([nmem, nt, nr, ny])      #big-indian ('>'), real*4 ('f4')
  f.close()
  moc_f_4conf[iconf, :, :, :] = np.mean(tmp_moc[:, ndump:-ndump, :, :], 0)
  moc_1memb_4conf[iconf, :, :, :] = tmp_moc[imem, ndump:-ndump, :, :]



#-- make reconstruction from ORAC+OCAR --
moc_f_4conf[-1, :, :, :] = moc_f_4conf[1, :, :, :] + moc_f_4conf[2, :, :, :]
moc_1memb_4conf[-1, :, :, :] = moc_1memb_4conf[1, :, :, :] + moc_1memb_4conf[2, :, :, :]

#-- correlation --
r_correl = np.zeros([nconf, nr, ny])
r_correl2 = np.zeros([nconf, nr, ny])
for iconf in range(nconf):
 for kkk in range(nr):
  for jjj in range(ny):
    r_correl[iconf, kkk, jjj] = np.corrcoef(moc_f_4conf[0, :, kkk, jjj], \
	moc_f_4conf[iconf+1, :, kkk, jjj], ddof=1)[1,0]
    r_correl2[iconf, kkk, jjj] = np.corrcoef(moc_1memb_4conf[0, :, kkk, jjj], \
	moc_1memb_4conf[iconf+1, :, kkk, jjj], ddof=1)[1,0]



#---------------------------
#	PLOT
#---------------------------


#-- y-z maps --

fig1 = plt.figure(figsize=(6,8))
cmap = "hot_r"
llev = np.arange(0, 1.1, 0.1)
#-- ensemble mean --
ax1 = fig1.add_subplot(2, 1, 1)
cs1 = ax1.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, \
        r_correl[-1, :, :], levels=llev, cmap=cmap)
ax1.plot(yG[:, 0], rF[kdepth, 0, 0]*np.ones(ny)/1000, 'w--')
ax1.plot(yG[jj26n, 0], rF[kdepth, 0, 0]/1000, 'wo')
ax1.set_title('r($<$OCAR$>$+$<$ORAC$>$, $<$ORAR$>$)')
ax1.set_ylabel('Depth [km]')
#ax1.set_xlabel('Latitude')
ax1.set_facecolor((0.5, 0.5, 0.5))
#-- one member --
ax2 = fig1.add_subplot(2, 1, 2)
cs2 = ax2.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, \
        r_correl2[-1, :, :], levels=llev, cmap=cmap, extend='min')
ax2.plot(yG[:, 0], rF[kdepth, 0, 0]*np.ones(ny)/1000, 'w--')
ax2.plot(yG[jj26n, 0], rF[kdepth, 0, 0]/1000, 'wo')
ax2.set_title('r(OCAR$_{memb\#00}$+ORAC$_{memb\#00}$, ORAR$_{memb\#00}$)')
ax2.set_xlabel('Latitude')
ax2.set_ylabel('Depth [km]')
ax2.set_facecolor((0.5, 0.5, 0.5))
##-- at a given depth --
#ax3 = fig1.add_subplot(1, 3, 3)
#p0 = ax3.plot(r_correl[-1, kdepth, 1:], yG[1:, 0],'r')
#p1 = ax3.plot(r_correl2[-1, kdepth, 1:], yG[1:, 0],'grey')
#ax3.grid()
#ax3.set_xlabel('Correlation coefficients')
#ax3.set_ylabel('Latitude')
#ax3.set_title('At 1200 m')
#ax3.set_ylim(-20, 55)
#ax3.set_xlim(0.2, 1.0)
#ax3.set_xticks(np.arange(0.2, 1.1, 0.1))
#ax3.legend((p0[0], p1[0]), (r'Ens. mean $<.>$', '$memb\#00$'))

cbaxes = fig1.add_axes([0.92, 0.30, 0.01, 0.4])
cbar = plt.colorbar(cs1, orientation='vertical', cax=cbaxes)

figN1 = 'correl_reconst_orar_ensMean_oneMemb_2'
fig1.savefig(dir_fig+figN1+'.pdf', dpi=300)
fig1.savefig(dir_fig+figN1+'.png', dpi=300)
plt.close(fig1)



