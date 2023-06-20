import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf

plt.ion()

dir_in = '/tank/chaocean/bderemble/kh/outdir_0007/'
dir_fig = '/tank/users/qjamet/Figures/publi/efficiency/'

#-- model parameters --
#- run 0006, run_0007 -
Re = 5000.
Ri = 0.2
Pr = 1.0
N2 = 0.016

#-- load variables --
# all variables are at tracer points 
f = netcdf.netcdf_file(dir_in + 'vars.nc', 'r')
nx = f.dimensions['x']
nr = f.dimensions['y']
xx = f.variables['x'][:]-f.variables['x'][0]
zz = f.variables['y'][:]
kkt = np.where(zz<5)[0][-1]
kkb = np.where(zz>-5)[0][0]
dx = xx[1]-xx[0]
dz = zz[1]-zz[0]
msk = np.ones([nr, nx])*(dz*dx)
#-- time dimension --
nt = len(f.variables['b'][:, 0, 0, 0])
time = np.arange(nt)
#-- some snapshots --
ttime = [0, int(nt/2), nt-1]
ntime = len(ttime)
bbb = np.zeros([ntime, nr, nx])
for iii in range(ntime):
  bbb[iii, :, :] = Ri*f.variables['b'][ttime[iii], 0, :, :]


bbound = Ri*np.tanh(zz[::2**4]) + N2*zz[::2**4]
bbound = np.delete(bbound, np.where(bbound<Ri*0.0))
bbound = np.delete(bbound, np.where(bbound>Ri*1.1))
nbound = len(bbound)



#----------------------
#	PLOT
#----------------------

fig1 = plt.figure(figsize=(12, 10))
fig1.clf()
ii0 = 0
ii1 = 5
ii2 = -1	#int_vol[:, -2] == bbound[-1] because full domain computation included in int_vol
#
for iii in range(ntime):
  ax = fig1.add_subplot(2,2,(iii+1), adjustable='box', aspect=1.0)
  cs = ax.contourf(xx, zz, bbb[iii, :, :], cmap='hot', levels=100)
  cs0 = ax.contour(xx, zz, bbb[iii, :, :], [-bbound[ii0]], colors='gray', linestyles='solid', alpha=0.5)
  cs1 = ax.contour(xx, zz, bbb[iii, :, :], [bbound[ii0]], colors='gray', linestyles='solid', alpha=0.5)
  cs2 = ax.contour(xx, zz, bbb[iii, :, :], [-bbound[ii1]], colors='g', linestyles='solid')
  cs3 = ax.contour(xx, zz, bbb[iii, :, :], [bbound[ii1]], colors='g', linestyles='solid')
  cs4 = ax.contour(xx, zz, bbb[iii, :, :], [-bbound[ii2]], colors='b', linestyles='solid')
  cs5 = ax.contour(xx, zz, bbb[iii, :, :], [bbound[ii2]], colors='b', linestyles='solid')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  ax.set_title( str("b @ t=%i" % time[ttime[iii]]), fontsize='x-large' )
#
cbax = fig1.add_axes([0.68, 0.15, 0.01, 0.3])
cb = fig1.colorbar(cs, ax=ax, orientation='vertical', cax=cbax)
cb.set_label(r'Buoyancy [nd]' )
#
figN = 'buoyancy_snapshot'
fig1.savefig(dir_fig + figN + '.png', dpi=300)
fig1.savefig(dir_fig + figN + '.pdf', dpi=300)
fig1.close()
