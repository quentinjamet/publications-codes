import numpy as np
import MITgcmutils as mit
import matplotlib.pyplot as plt
import statsmodels.api as sm    # for LOESS (or LOWESS) smoothing



dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/orar/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/'

#-- load grid --
yG = mit.rdmds(dir_grd + 'YG')
rF = mit.rdmds(dir_grd + 'RF')
ny, nx = yG.shape
nr = len(rF)-1
jj26n = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0][0]
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0][0]

#-- runs parameters --
ndump = 73
spy = 86400*365
nyr = 50
time_ens = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr
nmem = 24

#-------------------------------------------------------------------------------
# load processed MOC files for all simulations ; organized as [nmem, nt, nr, ny]
#-------------------------------------------------------------------------------
#-- load unprocessed moc for time mean --
moc_m = np.zeros([nr, ny])
fileN0 = 'MOCyzt_orar_ensemble.bin'
f = open(dir_in + fileN0,'r')
moc_m = np.mean( np.mean( np.fromfile(f,'>f4').reshape([nmem/2, nt, nr, ny]), 0), 0)
f.close()


#-- load processed AMOC to compute ratio --
mocyzt = np.zeros([nmem, nt, nr, ny])
#- first 12 members -
fileN1 = 'MOCyzt_orar_ensemble_detrend_1ylpf.bin'
f = open(dir_in + fileN1,'r')
mocyzt[0:12, :, :, :] = np.fromfile(f,'>f4').reshape([nmem/2, nt, nr, ny])	
f.close()
#- next 12 members -
fileN1 = 'MOCyzt_orar_ensemble_2_detrend_1ylpf.bin'
f = open(dir_in + fileN1,'r')
mocyzt[12:, :, :, :] = np.fromfile(f,'>f4').reshape([nmem/2, nt, nr, ny])	
f.close()


#------------------------------------------------------
# Computed intrinsic / total variance ratio for subsets
#------------------------------------------------------

#- for the 24 members -
moc_f = np.mean(mocyzt, 0)
moc_i = mocyzt - np.tile(moc_f[np.newaxis, :, :, :],(nmem, 1, 1, 1))
var_f = np.var(moc_f, 0, ddof=0)
var_i = np.mean(np.var(moc_i,0, ddof=0), 0) 
var_ratio = var_i / (var_i+var_f)

#-- estimate vol. transport at Rapid location --
moc_26n = mocyzt[:, :, kdepth, jj26n]
frac = 10.0*ndump/nt	# 10-yr high-pass filter
moc_26n_dec = np.zeros([nmem, nt])
lowess = sm.nonparametric.lowess
for imem in range(nmem):
 moc_26n_dec[imem, :] = lowess(moc_26n[imem, :], time_ens, frac=frac, return_sorted=False)

moc_26n_inter = moc_26n - moc_26n_dec
sigma_i_inter = np.mean( np.std(moc_26n_inter, 0, ddof=0), 0)
sigma_i_dec = np.mean( np.std(moc_26n_dec, 0, ddof=0), 0)
sigma_f_inter = np.std( np.mean(moc_26n_inter, 0), 0, ddof=0)
sigma_f_dec = np.std( np.mean(moc_26n_dec, 0), 0, ddof=0)



#---------------------------
#	PLOT
#---------------------------

#-- var ratio --
fig1 = plt.figure(figsize=(10,5))
ax = fig1.add_subplot(1, 1, 1)
plt.contourf(yG[:, 0], np.squeeze(rF[0:-1])/1000, var_ratio, np.arange(0, 1.1, 0.1), cmap='hot_r')
cbar = plt.colorbar()
cbar.set_label('$\sigma_I^2$ / $\sigma_T^2$')
plt.contour(yG[:, 0], np.squeeze(rF[0:-1])/1000, moc_m, np.arange(-15, 25, 5), colors='grey', alpha=0.5, linewidths= 1)
plt.contour(yG[:, 0], np.squeeze(rF[0:39])/1000, moc_m[0:39, :], 0, colors='grey', alpha=0.5, linewidths= 2)
plt.plot(26.5*np.ones(nr,), np.squeeze(rF[0:-1])/1000,'k--')
plt.plot(26.5, -1.2, 'ko')
plt.xlabel('Latitude', fontsize='x-large')
plt.ylabel('Dpeth [km]', fontsize='x-large')
plt.title('AMOC intrinsic-to-total variance ratio', fontsize='xx-large')
ax.set_facecolor((0.3, 0.3, 0.3))
#plt.show()
fileN1 = 'AMOC_var_ratio_orar_24mem_fig01'
fig1.savefig(dir_fig + fileN1 + '.png', dpi=300)
fig1.savefig(dir_fig + fileN1 + '.pdf', dpi=300)
plt.close(fig1)


#-- vertical profil of variance --
kdepth = np.where( np.abs(rF+1200) == np.min(np.abs(rF+1200)) )[0]
jj = np.where(var_ratio[kdepth, :] == np.nanmax(var_ratio[kdepth, :]))[1][0]
jj2 = np.where(np.abs(yG[:,1]-26.5) == np.min(np.abs(yG[:,1]-26.5)))[0][0]

fig2 = plt.figure(figsize=(8, 8))
ax1 = fig2.add_subplot(1, 2, 2)
p1 = plt.plot(var_f[:, jj], np.squeeze(rF[0:-1])/1000, 'k')
p2 = plt.plot(var_i[:, jj], np.squeeze(rF[0:-1])/1000, 'g')
plt.grid()
plt.legend((p1[0], p2[0]), ('Forced', 'Intrinsic'))
plt.title('Variance at ' + str("%.01f" % yG[jj, 0]) + ' N')
plt.xlabel('[Sv$^2$]')
plt.ylabel('Depth [km]')
ax2 = fig2.add_subplot(1, 2, 1)
p1 = plt.plot(var_f[:, jj2], np.squeeze(rF[0:-1])/1000, 'k')
p2 = plt.plot(var_i[:, jj2], np.squeeze(rF[0:-1])/1000, 'g')
plt.grid()
plt.legend((p1[0], p2[0]), ('Forced', 'Intrinsic'))
plt.title('Variance at ' + str("%.01f" % yG[jj2, 0]) + ' N')
plt.xlabel('[Sv$^2$]')
plt.ylabel('Depth [km]')



#plt.show()

fileN2 = 'AMOC_vert_struct_var_forc_intrinsic'
fig2.savefig(dir_fig + fileN2 + '.png', dpi=300)
fig2.savefig(dir_fig + fileN2 + '.pdf', dpi=300)


