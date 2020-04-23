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


#-- load time processed MOC files for all simulations ; organized as [nmem, nt, nr, ny] --
imem = 0
std_amoc_4conf = np.zeros([nconf, nr, ny])
for iconf in range(0,nconf):
  print(config[iconf])
  fileN1 = 'MOCyzt_' + config[iconf] + '_ensemble_detrend_1ylpf.bin'
  f = open(dir_in + config[iconf] + '/' + fileN1, 'r')
  tmp_moc = np.fromfile(f,'>f4').reshape([nmem, nt, nr, ny])
  f.close()
  #- first and last years discarded due to filtering side effects -
  # effectc are very large near the equator !!!
  tmp_moc = tmp_moc[:, ndump:-ndump, :, :]
  std_amoc_4conf[iconf, :, :] = np.std(np.mean(tmp_moc, 0), 0)
  #std_amoc_4conf[iconf, :, :] = np.std(tmp_moc[imem, :, :, :], 0)

std_amoc_4conf[np.where(std_amoc_4conf == 0)] = np.nan
#---------------------------
#	PLOT
#---------------------------

fig1 = plt.figure(figsize=(14,7))
#fig1.suptitle('Time mean AMOC')
levels = np.arange(0, 2.1, .1)
ip=0
for i in range(2):
 for j in range(2):
  ax = fig1.add_subplot(2, 2, ip+1)
  cs = plt.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, std_amoc_4conf[ip, :, :], levels=levels, cmap='hot_r', extend='max')
  plt.plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000, 'k--')
  plt.title(config[ip].upper())
  if j == 0:
    ax.set_ylabel('Depth [km]')
  if i == 1:
    ax.set_xlabel('Latitude')
  ax.set_facecolor((0.5, 0.5, 0.5))
  ip = ip+1

cbaxes = fig1.add_axes([0.92, 0.25, 0.01, 0.5])
cbar = plt.colorbar(cs, orientation='vertical', cax = cbaxes)
#cbar.set_label(r"$\sigma_F(AMOC)$ [Sv]")
cbar.set_label(r"$\sigma_F(AMOC_{memb00})$ [Sv]")
#plt.show()

#- SAVE -
#figN1 = 'std_AMOC_4conf'
figN1 = 'std_AMOC_memb00_4conf'
fig1.savefig(dir_fig + figN1 + '.pdf', bbox_inches='tight')
fig1.savefig(dir_fig + figN1 + '.png', bbox_inches='tight')
plt.close(fig1)


