import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from multiprocessing import Pool

plt.ion()

dir_in = '/tank/chaocean/bderemble/kh/outdir_0007/'
dir_fig = '/tank/users/qjamet/Figures/publi/efficiency/'

#-- model parameters --
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
hC  = np.ones([nr, nx])*(dz*dx)
msk = np.ones([nr, nx])
#-- time dimension --
nt = len(f.variables['b'][:, 0, 0, 0])
time = np.arange(nt)

#-- bounding buoyancy for volume integration --
bbound = Ri*np.tanh(zz[::2**4]) + N2*zz[::2**4]
bbound = np.delete(bbound, np.where(bbound<Ri*0.0))
bbound = np.delete(bbound, np.where(bbound>Ri*1.1))
nbound = len(bbound)

#--------------------------------
#	Define some computation
#--------------------------------

#-- dissipation --
def eps_comp(tmpu, tmpw, dx, dz): 
  #- zonal derivatives (at t-points) -
  # uu
  tmp_dudx = np.zeros([nr, nx+1]) 
  tmp_dudx[:, 1:-1] = (tmpu[:, 1:]-tmpu[:, :-1])/dx
  tmp_dudx[:,  0] = (tmpu[:,  0]-tmpu[:,  -1])/dx
  tmp_dudx[:, -1] = tmp_dudx[:,  0]
  dudx_2 = ( (tmp_dudx[:, 1:]+tmp_dudx[:, :-1])/2 )**2
  # ww
  tmp_dwdx = np.zeros([nr, nx+1]) 
  tmp_dwdx[:, 1:-1] = (tmpw[:, 1:]-tmpw[:, :-1])/dx
  tmp_dwdx[:,  0] = (tmpw[:,  0]-tmpw[:,  -1])/dx
  tmp_dwdx[:, -1] = tmp_dwdx[:,  0]
  dwdx_2 = ( (tmp_dwdx[:, 1:]+tmp_dwdx[:, :-1])/2 )**2
  #- vertical derivatives (at t-points) -
  # uu
  tmp_dudz = np.zeros([nr+1, nx])
  tmp_dudz[1:-1, :] = (tmpu[1:, :]-tmpu[:-1])/dz
  dudz_2 = ( (tmp_dudz[:-1, :]+tmp_dudz[1:, :])/2 )**2
  # ww
  tmp_dwdz = np.zeros([nr+1, nx])
  tmp_dwdz[1:-1, :] = (tmpw[1:, :]-tmpw[:-1])/dz
  dwdz_2 = ( (tmp_dwdz[:-1, :]+tmp_dwdz[1:, :])/2 )**2
  # dissipation
  tmpepsilon = 1/Re * (dudx_2 + dudz_2 + dwdx_2 + dwdz_2)
  return tmpepsilon


#-------------------------------
#	Compute
#-------------------------------


#-- define parallel computing --
def vol_int(ibound):
  msk = hC*1.0
  if ibound < nbound:
    msk[ np.where( bbb<-bbound[ibound] ) ] = 0.0
    msk[ np.where( bbb> bbound[ibound] ) ] = 0.0
  #
  tmp1 = msk.sum()
  tmp2 = (bbb*msk).sum()
  tmp3 = (epsilon*msk).sum()
  tmp4 = (wb*msk).sum()
  return tmp1, tmp2, tmp3, tmp4;

#-- compute for different volum of integration --
int_val = np.zeros([nt, nbound+1, 4])
for iii in range(nt):
  print("%04.f" % iii)
  #- load -
  uuu = f.variables['u.x'][iii, 0, :, :]
  www = f.variables['u.y'][iii, 0, :, :]
  bbb = Ri*f.variables['b'][iii, 0, :, :]
  #- dissip -
  epsilon = eps_comp(uuu, www, dx, dz)
  #- wb -
  wbar = 1/(msk.sum()) * (www*msk).sum()
  bbar = 1/(msk.sum()) * (bbb*msk).sum()
  wpr  = www - wbar
  bpr  = bbb - bbar
  wb   = wpr * bpr
  #- integration over different volume -
  if __name__ == '__main__':
    p = Pool(25)
    tmp = p.map(vol_int, np.arange(0,nbound+1))
  for ibound in range(nbound+1):
    int_val[iii, ibound, :] = tmp[ibound]


#---------------------------------
#	Save and Load
#---------------------------------
#- save -
fileN = 'eff_as_vol_int.data'
dir_out = '/tank/users/qjamet/python/publis/efficiency/'
f2 = open(dir_out+fileN, 'wb')
int_val.reshape([nt*(nbound+1)*4]).astype('>f4').tofile(f2)
f2.close()

#- load -
fileN = 'eff_as_vol_int.data'
dir_out = '/tank/users/qjamet/python/publis/efficiency/'
f2 = open(dir_out+fileN, 'r')
int_val = np.fromfile(f2, '>f4').reshape([nt, nbound+1, 4])
f2.close()

#
int_vol = int_val[:, :, 0]
bbb_int = int_val[:, :, 1]
eps_int = int_val[:, :, 2]
wb_int  = int_val[:, :, 3]

#-- compute efficiency --
# the first time steps are disregarded 
# because esp_int is contaminated by dissipation
# associated with initialisation adjustement
#
eff  = np.zeros([nt, nbound+1])
eff[20:, :]  = -wb_int[20:, :].cumsum(0) / eps_int[20:, :].cumsum(0) 


#----------------------
#	PLOT
#----------------------
ii0 = 0
ii1 = 5
ii2 = -2


#-- volume as a fcn of b clound --
# max volume is 42% of full domain
fig1 = plt.figure(figsize=(6,5))
fig1.clf()
#
ax1 = fig1.add_subplot(1, 1, 1)
p0 = ax1.plot(time, int_vol[:, :nbound-1] , \
        'gray', alpha=0.5, linestyle='--')
p1 = ax1.plot(time, int_vol[:, ii2] , 'b', linewidth=3)
p2 = ax1.plot(time, int_vol[:, ii1] , 'g', linewidth=3)
p3 = ax1.plot(time, int_vol[:, ii0] , 'gray', linewidth=3)
ax1.set_ylabel(r'Volume [nd]')
ax1.set_xlabel('Time [nd]')
ax1.set_title('Cloud volume')
ax1.set_xlim([0, 1000])
ax1.grid()
#
figN = 'vol_of_int_fcn_of_b_cloud_7'
fig1.savefig(dir_fig + figN + '.png', dpi=300)
fig1.savefig(dir_fig + figN + '.pdf', dpi=300)
plt.close(fig1)


#-- efficiency as a fcn of b cloud --
fig1 = plt.figure(figsize=(12,5))
fig1.clf()
#
ax3 = fig1.add_subplot(111)
p0 = ax3.plot(time, eff[:, :nbound-1], 'gray', alpha=0.5, linestyle='--')
p0 = ax3.plot(time, eff[:, ii0], 'gray', linewidth=3)
p1 = ax3.plot(time, eff[:, ii2], 'b', linewidth=3)
p2 = ax3.plot(time, eff[:, ii1], 'g', linewidth=3)
p3 = ax3.plot(time, eff[:,  nbound], 'r', linewidth=3)
ax3.legend((p1[0],p2[0],p3[0]), \
        ('Max buoyancy cloud', 'Min buoyancy cloud', 'full domain'))
ax3.set_xlabel('Time [nd]')
ax3.set_title(r"Mixing efficiency ($\Gamma = \frac{- \int_t \int_{V_a}w'b'~d\mathbf{x}dt}{\int_t \int_{V_a} \epsilon ~d\mathbf{x}dt}$)")
ax3.set_xlim([0, 1000])
ax3.set_ylim([-0.2, 2.7])
ax3.grid()
#
figN = 'efficieny_fcn_of_b_cloud_7'
fig1.savefig(dir_fig + figN + '.png', dpi=300)
fig1.savefig(dir_fig + figN + '.pdf', dpi=300)
plt.close(fig1)

