import numpy as np
import MITgcmutils as mit
import matplotlib.pyplot as plt


#-- directories and config --
dir_in1 = '/tank/chaocean/qjamet/RUNS/ORAC/memb00/'
dir_in2 = '/tank/chaocean/qjamet/RUNS/forc_ny/memb00/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_fig = '/tank/users/qjamet/Figures/publi/note_bams/'

#-- runs parameters --
ndump = 73
spy = 86400*365
# remove 10 first years because issue in cheapaml with xperiodic
yyr = np.arange(1973, 2013)
nyr = len(yyr)
ttime = 1963 + np.arange(1/float(ndump),nyr+1/float(ndump),1/float(ndump))
nt = ndump*nyr

#-- grid --
xC = mit.rdmds(dir_grd + 'XC')
xC = xC-360
yC = mit.rdmds(dir_grd + 'YC')
dxG = mit.rdmds(dir_grd + 'DXG')
dyG = mit.rdmds(dir_grd + 'DYG')
hC = mit.rdmds(dir_grd + 'hFacC')
msk = hC[0, :, :]*1.0
msk[ np.where(msk[:] == 0) ] = np.nan
rA = dxG*dyG
ny, nx = rA.shape

#-- load sst data --
t2 = np.zeros([nyr, ny, nx])
sst1 = np.zeros([nyr, ny, nx])
sst2 = np.zeros([nyr, ny, nx])
for iyr in range(0,nyr):
  print("year: %i \n" %yyr[iyr] )
  #- orar -
  # t2
  tmpdir = dir_in1 + 'run' + str(yyr[iyr]) + '/cheapaml/'
  tmp_var = mit.rdmds(tmpdir + 'diag_cheapAML', np.nan, rec=0)
  t2[iyr, :, :] = np.mean(tmp_var, 0)
  # sst
  tmpdir = dir_in1 + 'run' + str(yyr[iyr]) + '/ocn/'
  tmp_var = mit.rdmds(tmpdir + 'diag_ocnTave', np.nan, rec=0, lev=0, usememmap=True)
  sst1[iyr, :, :] = np.mean(tmp_var, 0)
  #- fprc_ny -
  tmpdir = dir_in2 + 'run' + str(yyr[iyr]) + '/ocn/'
  tmp_var = mit.rdmds(tmpdir + 'diag_ocnTave', np.nan, rec=0, lev=0, usememmap=True)
  sst2[iyr, :, :] = np.mean(tmp_var, 0)

t2_std = np.std(t2, 0) * msk
sst1_std = np.std(sst1, 0) * msk
sst2_std = np.std(sst2, 0) * msk


#------------------
#	reshape 
#------------------

#-- reshape in regular fashion --
nx_resh = 301
#- coordinate -
xC_resh = np.zeros([ny, nx+nx_resh])
xC_resh[:, 0:nx] = xC
dxC = xC[:,-1]-xC[:,-2]
xC_resh[:, nx:nx+nx_resh] = np.tile(xC[:,-1][:, np.newaxis], (1, nx_resh)) + \
    np.cumsum( np.tile(dxC[:,np.newaxis], (1, nx_resh)),1 )
yC_resh = np.zeros([ny, nx+nx_resh])
yC_resh = np.tile(yC[:, 0][:, np.newaxis], (1, nx+nx_resh))

f = open('/tank/chaocean/scripts/mask_cut_gulf_lines.bin','r')
msk_chao12 = np.fromfile(f,'>f4').reshape([ny, nx_resh])
f.close()
#- to move -
ji_tomove = np.where( msk_chao12 == 1 )
#- stay here, stay near -
ji_nomove = np.where( msk_chao12 == 0 )
#- assemble -
def var_resh(var_in):
 var_tomove = np.zeros([ny, nx_resh])
 var_tomove[ji_tomove[0], ji_tomove[1]] = var_in[ji_tomove[0], ji_tomove[1]]
 var_nomove = np.zeros([ny, nx_resh])
 var_nomove[ji_nomove[0], ji_nomove[1]] = var_in[ji_nomove[0], ji_nomove[1]]
 var_out = np.zeros([ny, nx+nx_resh])
 var_out[:, nx_resh:nx] = var_in[:, nx_resh:nx]
 var_out[:, 0:nx_resh] = var_nomove
 var_out[:, nx:nx+nx_resh] = var_tomove
 return var_out


t2_std_resh = var_resh(t2_std)
t2_std_resh[ np.where(t2_std_resh[:] == 0) ] = np.nan
sst1_std_resh = var_resh(sst1_std)
sst1_std_resh[ np.where(sst1_std_resh[:] == 0) ] = np.nan
sst2_std_resh = var_resh(sst2_std)
sst2_std_resh[ np.where(sst2_std_resh[:] == 0) ] = np.nan
diff = sst1_std_resh-sst2_std_resh

#------------------
#	PLOT 
#------------------


fig1 = plt.figure(figsize=(15,6))
#-- t2 ORAC --
llev1 = np.arange(0, 0.72, 0.05)
ax = fig1.add_subplot(1, 2, 1)
plt.contourf(xC_resh, yC_resh, t2_std_resh, llev1)
plt.xlabel('Longitude', fontsize='x-large')
plt.ylabel('Latitude', fontsize='x-large')
plt.title('Yearly Atmo. Temp. std', fontsize='xx-large')
ax.set_facecolor((0.8, 0.8, 0.8))
cb = plt.colorbar(ticks=np.arange(0, 0.72, 0.1))
cb.set_label('$\sigma(t2)$ [$^{\circ}$C]')
#-- sst diff --
llev2 = np.arange(-0.5, 0.52, 0.05)
ax = fig1.add_subplot(1, 2, 2)
cs = plt.contourf(xC_resh, yC_resh, sst1_std_resh-sst2_std_resh, llev2, extend='both', cmap='RdBu_r')
plt.contour(xC_resh, yC_resh, sst1_std_resh-sst2_std_resh, 
	[0], colors=['black'], alpha=0.5, linewidths=1)
plt.xlabel('Longitude', fontsize='x-large')
plt.ylabel('Latitude', fontsize='x-large')
plt.title('Yearly SST std diff.', fontsize='xx-large')
ax.set_facecolor((0.8, 0.8, 0.8))
cb = plt.colorbar(cs, ticks=np.arange(-0.5, 0.6, 0.1))
cb.set_label('$\sigma(sst)_{AML} - \sigma(sst)_{FORC}$ [$^{\circ}$C]')

#plt.show()

figN1 = 't2_yr_std_amlny_sst_yr_std_diff_aml_forc_ny'
fig1.savefig(dir_fig+figN1+'.pdf', bbox_inches='tight')
fig1.savefig(dir_fig+figN1+'.png', bbox_inches='tight')
plt.close(fig1)


fig2 = plt.figure(figsize=(10,8))
#-- sst relative diff --
llev2 = np.arange(-1., 1.1, 0.1)
ax = fig1.add_subplot(1, 1, 1)
cs = plt.contourf(xC_resh, yC_resh, diff / sst1_std_resh, llev2, extend='both', cmap='RdBu_r')
plt.contour(xC_resh, yC_resh, diff / sst1_std_resh,      
        [0], colors=['black'], alpha=0.5, linewidths=1)
plt.xlabel('Longitude', fontsize='x-large')
plt.ylabel('Latitude', fontsize='x-large')
plt.title('Yearly SST std rel. diff.', fontsize='xx-large')
ax.set_facecolor((0.8, 0.8, 0.8))
cb = plt.colorbar(cs, ticks=np.arange(-1, 1.1, 0.1))
cb.set_label('$\sigma(sst)_{AML} - \sigma(sst)_{FORC} / \sigma(sst)_{AML}$ ')

#plt.show()

figN2 = 't2_yr_std_amlny_sst_yr_std_rel_diff_aml_forc_ny'
fig2.savefig(dir_fig+figN2 + '.pdf', bbox_inches='tight')
fig2.savefig(dir_fig+figN2 + '.png', bbox_inches='tight')
plt.close(fig2)





