import numpy as np
import matplotlib.pyplot as plt
import MITgcmutils as mit

plt.ion()

#-- directories 
dir_in = '/mnt/meom/workdir/jametq/runs/run_kh/'
#dir_run = dir_in + 'run_KEdiag/'
dir_run = dir_in + 'run_KEdiag_ptr/'
dir_grd = dir_run + 'grd/'
dir_fig = '/home/jametq/Figures/kh_run/'
fileN = 'diagKEs*'

#-- dimension and config params --
xC    = mit.rdmds(dir_grd+'XC*')
yC    = mit.rdmds(dir_grd+'YC*')
xG    = mit.rdmds(dir_grd+'XG*')
yG    = mit.rdmds(dir_grd+'YG*')
rC    = mit.rdmds(dir_grd+'RC*')
rF    = mit.rdmds(dir_grd+'RF*')
hC    = mit.rdmds(dir_grd+'hFacC*')
ny, nx = xC.shape
nr = len(rC)
dx = 0.01
dz = 0.01
cellvol = hC*dx*dz

iters = list(range(250, 270000+250, 250))
nt = len(iters)
deltaT = (iters[1]-iters[0])*0.02

#------------------------
# On couple of time steps
#------------------------
nit = 10
diagKEs = mit.rdmds(dir_run + 'ocn/' + fileN, iters[1:nit])
rhs = np.sum(diagKEs[:, :, :, :, :], 1) 
del diagKEs
#- compute dKE/dt from u,v,w directly -
uu = mit.rdmds(dir_run + 'ocn/' + 'U*', iters[0:nit])
ww = np.zeros([nit, nr+1, 1, nx])
ww[:, 0:nr, :, :] = mit.rdmds(dir_run + 'ocn/' + 'W*', iters[0:nit])
ke = np.zeros([nit, nr, 1, nx])
ke[:, :, :, 0:-1] = 0.25 * ( (uu[:, :, :, 0:-1]**2 + uu[:, :, :, 1:]**2) + \
	      (ww[:, 0:-1, :, 0:-1]**2 + ww[:, 1:, :, 0:-1]**2) )
ke[:, :, :, -1] = 0.25 * ( (uu[:, :, :, 0]**2 + uu[:, :, :, -1]**2) + \
              (ww[:, 0:-1, :, -1]**2 + ww[:, 1:, :, -1]**2) )
del uu
del ww

#- time rate of change of KE -
ket = (ke[1:, :, :, :] - ke[0:-1, :, :, :])/deltaT

#- residual -
res = ket-rhs



#-----------
#   PLOT 
#-----------


def fig_kh(xx, yy, rhs, ket, res, llev, scaleF, scaleErr, figN, saveF):
  fig1 = plt.figure(figsize=(12, 4))
  #
  ax1 = fig1.add_subplot(1, 3, 1)
  cs1 = ax1.contourf(xx, yy, rhs * scaleF, \
  	levels=llev, extend='both')
  ax1.set_title('RHS')
  cbax1 = fig1.add_axes([0.02, 0.1, 0.01, 0.8])
  cb1 = fig1.colorbar(cs1, ax=ax1, orientation='vertical', cax=cbax1)
  cb1.set_label(r'$\times$ %0.0e ; [m/s]' % (1/scaleF))
  #
  ax2 = fig1.add_subplot(1, 3, 2)
  cs2 = ax2.contourf(xx, yy, ket * scaleF, \
	levels=llev, extend='both')
  ax2.set_title('$\partial_t KE$')
  #
  ax3 = fig1.add_subplot(1, 3, 3)
  cs3 = ax3.contourf(xx, yy, res * scaleF * scaleErr, \
	levels=llev, cmap='RdBu_r', extend='both')
  ax3.set_title('$\partial_t KE$ - RHS')
  cbax3 = fig1.add_axes([0.92, 0.1, 0.01, 0.8])
  cb3 = fig1.colorbar(cs3, ax=ax3, orientation='vertical', cax=cbax3)
  cb3.set_label(r'$\times$ %0.0e ; [m/s]' % (1/(scaleF*scaleErr)) )
  #
  if saveF:
    fig1.savefig(dir_fig + figN+'.pdf',dpi=100)
    fig1.savefig(dir_fig + figN+'.png',dpi=100)
    plt.close(fig1)


#-- model outputs --
iit = 5
llev = np.arange(-4, 4.1, 0.1)
scaleF = 1e5
scaleErr = 1e13
figN = 'KE_bgt_NH_setup'
saveF = False
fig_kh(xC[0, :], rC[:, 0, 0], \
	rhs[iit, :, 0, :], ket[iit, :, 0, :], res[iit, :, 0, :], \
	llev, scaleF, scaleErr, figN, saveF)



#------------------------
# Time integration
#------------------------
[kk, ii] = [nr/2, nx/2]
itfi = 200

ke_int  = np.zeros([nt])
adv_int = np.zeros([nt])
hpg_int = np.zeros([nt])
nhpg_int = np.zeros([nt])
diss_int = np.zeros([nt])
ab_int = np.zeros([nt])
for iii in range(itfi):
  print("%04.f" % iii)
  #-- KE --
  uu = mit.rdmds(dir_run + 'ocn/' + 'U*', iters[iii])
  ww = np.zeros([nr+1, 1, nx])
  ww[0:nr, :, :] = mit.rdmds(dir_run + 'ocn/' + 'W*', iters[iii])
  ke = np.zeros([nr, 1, nx])
  ke[:, :, 0:-1] = 0.25 * ( (uu[:, :, 0:-1]**2 + uu[:, :, 1:]**2) + \
                (ww[0:-1, :, 0:-1]**2 + ww[1:, :, 0:-1]**2) )
  ke[:, :, -1] = 0.25 * ( (uu[:, :, 0]**2 + uu[:, :, -1]**2) + \
                (ww[0:-1, :, -1]**2 + ww[1:, :, -1]**2) )
  ke_int[iii] = ke[kk, 0, ii]
  #-- advection --
  adv_int[iii] = mit.rdmds(dir_run + 'ocn/' + fileN, iters[iii], rec=0)[kk, 0, ii]
  #-- hydrostatic pressure gradient --
  hpg_int[iii] = mit.rdmds(dir_run + 'ocn/' + fileN, iters[iii], rec=1)[kk, 0, ii]
  #-- (surface and) non-hydrostatic pressure gradients --
  nhpg_int[iii] = mit.rdmds(dir_run + 'ocn/' + fileN, iters[iii], rec=2)[kk, 0, ii]
  #-- dissipation --
  diss_int[iii] = mit.rdmds(dir_run + 'ocn/' + fileN, iters[iii], rec=3)[kk, 0, ii]
  #-- Adams-Bashforth --
  ab_int[iii] = mit.rdmds(dir_run + 'ocn/' + fileN, iters[iii], rec=6)[kk, 0, ii]
  #-- RHS --
 
rhs_int = adv_int + hpg_int + nhpg_int + diss_int + ab_int

%-- plot --
fig1 = plt.figure(figsize=(10,6))
ax1 = fig1.add_subplot(1,1,1)
p0 = ax1.plot(ke_int[0:itfi]-ke_int[0], 'k', lw=3) 
p1 = ax1.plot((rhs_int[0:itfi]*deltaT).cumsum(0), 'r--')
p2 = ax1.plot((adv_int[0:itfi]*deltaT).cumsum(0), lw=1)
p3 = ax1.plot((hpg_int[0:itfi]*deltaT).cumsum(0), lw=1)
p4 = ax1.plot((nhpg_int[0:itfi]*deltaT).cumsum(0), lw=1)
p5 = ax1.plot((diss_int[0:itfi]*deltaT).cumsum(0), lw=1)
p6 = ax1.plot((ab_int[0:itfi]*deltaT).cumsum(0), lw=1)
plt.legend((p0[0], p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), \
	(r"KE(t)-KE(t=0)", 'RHS', 'ADV', 'HPG', 'NHPG', 'Diss', 'AB'))
plt.grid()
ax1.set_title('Integrated trends of KE at a single grid point')
ax1.set_ylabel('[m$^2$/s$^2$]')
ax1.set_xlabel('Counts (x 5sec)')
#
figN1 = 'Time_int_KE_trends'
fig1.savefig(dir_fig+figN1+'.png',dpi=100)
fig1.savefig(dir_fig+figN1+'.pdf',dpi=100)

