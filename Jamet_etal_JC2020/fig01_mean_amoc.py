import numpy as np
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
jj26n = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0]
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nmem = 12


#-- load raw MOC files for all simulations ; organized as [ny, nmem, ndump, nr, ny] --
mean_amoc_4conf = np.zeros([nconf, nr, ny])
for iconf in range(0,nconf):
  print(config[iconf])
  fileN1 = 'MOCyzt_' + config[iconf] + '_ensemble.bin'
  f = open(dir_in + config[iconf] + '/' + fileN1, 'r')
  tmp_moc = np.fromfile(f,'>f4').reshape([nyr, nmem, ndump, nr, ny])
  f.close()
  tmp_moc = np.transpose(tmp_moc, (1, 0, 2, 3, 4)).reshape([nmem*nt, nr, ny])
  mean_amoc_4conf[iconf, :, :] = np.mean(tmp_moc, 0)

mean_amoc_4conf[np.where(mean_amoc_4conf == 0)] = np.nan
mean_amoc_4conf[1:, :, :] = mean_amoc_4conf[1:, :, :] - np.tile(mean_amoc_4conf[0, :, :][np.newaxis, :, :], (3, 1, 1))
#---------------------------
#	PLOT
#---------------------------

fig1 = plt.figure(figsize=(14,7))
#fig1.suptitle('Time mean AMOC')
levels = np.arange(-18, 20, 2)
lev2 = np.arange(-2, 2.2, 0.2)
#-- ORAR --
ax = fig1.add_subplot(2, 2, 1)
plt.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[0, :, :], levels=levels, cmap='RdBu_r')
cbar = plt.colorbar()
cbar.set_label('[Sv]')
plt.contour(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[0, :, :], 0, colors='k')
plt.plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000, 'k--')
plt.plot(26.5, np.squeeze(rF[kdepth])/1000, 'ko')
plt.title('ORAR', fontsize='xx-large')
plt.ylabel('Depth [km]', fontsize='x-large')
ax.set_facecolor((0.5, 0.5, 0.5))
#-- OCAR-ORAR --
ax = fig1.add_subplot(2, 2, 2)
plt.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[1, :, :], levels=lev2, cmap='BrBG_r')
cbar = plt.colorbar()
cbar.set_label('[Sv]')
plt.contour(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[1, :, :], 0, colors='k')
plt.plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000, 'k--')
plt.plot(26.5, np.squeeze(rF[kdepth])/1000, 'ko')
plt.title('OCAR-ORAR', fontsize='xx-large')
ax.set_facecolor((0.5, 0.5, 0.5))
#-- ORAC-ORAR --
ax = fig1.add_subplot(2, 2, 3)
plt.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[2, :, :], levels=lev2, cmap='BrBG_r')
cbar = plt.colorbar()
cbar.set_label('[Sv]')
plt.contour(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[2, :, :], 0, colors='k')
plt.plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000, 'k--')
plt.plot(26.5, np.squeeze(rF[kdepth])/1000, 'ko')
plt.title('ORAC-ORAR', fontsize='xx-large')
ax.set_facecolor((0.5, 0.5, 0.5))
plt.ylabel('Depth [km]', fontsize='x-large')
plt.xlabel('Latitude', fontsize='x-large')
#-- ORAC-ORAR --
ax = fig1.add_subplot(2, 2, 4)
plt.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[3, :, :], levels=lev2, cmap='BrBG_r')
cbar = plt.colorbar()
cbar.set_label('[Sv]')
plt.contour(yG[:, 0], np.squeeze(rF[0:-1])/1000, mean_amoc_4conf[3, :, :], 0, colors='k')
plt.plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000, 'k--')
plt.plot(26.5, np.squeeze(rF[kdepth])/1000, 'ko')
plt.title('OCAC-ORAR', fontsize='xx-large')
ax.set_facecolor((0.5, 0.5, 0.5))
plt.xlabel('Latitude', fontsize='x-large')

#plt.show()

#- SAVE -
figN1 = 'mean_AMOC_4conf_tmp'
fig1.savefig(dir_fig + figN1 + '.pdf', bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.png', bbox_inches='tight')
plt.close(fig1)


