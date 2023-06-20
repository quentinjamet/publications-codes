#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys, glob,os,re
import xarray as xr

plt.ion()

flag_nc = 0

#dir0 = "../outdir_"
#dir0 = "/run/media/bderembl/workd/basilisk/myrun/kh/outdir_"
dir0 = "/tank/chaocean/bderemble/kh/outdir_0007/"

if len(sys.argv) > 1:
  dir0 = dir0 + str(format(sys.argv[1])).zfill(4) + '/'

exec(open(dir0 + "params.in").read())

fileb = 'b*'
fileu = 'u*'
filepsi = 'psi*'
filepart = 'part*'
filenc = 'vars.nc'

allfilesb = sorted(glob.glob(dir0 + fileb));
allfilesu = sorted(glob.glob(dir0 + fileu));
allfilespart = sorted(glob.glob(dir0 + filepart));
si_t  = len(allfilesb);
si_fp  = len(allfilespart);

if len(allfilesb) == 0:
  flag_nc = 1
  ds = xr.open_dataset(dir0+filenc) 
  si_t = len(ds['time'])

# get number of procs
if si_fp > 0:
  itb = 0
  for it in range(0,si_fp):
    itn = int(allfilespart[it][-18:-9])
    if itn != itb:
      break
    
  nprocs = it

# dimensions
if flag_nc == 0:
  b = np.fromfile(allfilesb[0],'f4')
  N = int(b[0])
else:
  N = len(ds['x'])
N1 = N + 1

Delta = L0/N
X0 = -L0/2
Y0 = -L0/2

Delta_zf = L0/(N*N)

x = np.linspace(X0+0.5*Delta, X0+L0 - 0.5*Delta,N)
z = np.linspace(Y0+0.5*Delta, Y0+L0 - 0.5*Delta,N)
zo = np.linspace(Y0+0.5*Delta_zf, Y0+L0 - 0.5*Delta_zf,N*N)
xc,zc = np.meshgrid(x,z)

b_bg = np.tanh(zc)
u_bg = np.tanh(zc)

pe0 = np.sum(-Ri*b_bg*zc)*Delta**2
ke0 = 0.5*np.sum(u_bg**2)*Delta**2

pe = np.zeros(si_t)
ke = np.zeros(si_t)
epsilon = np.zeros(si_t)
wb = np.zeros(si_t)
pe_bg = np.zeros(si_t)

it = -1
for it in range(0,si_t):
  print(it)
  if flag_nc == 0:
    b = np.fromfile(allfilesb[it],'f4').reshape(N1,N1).transpose(1,0)
    b = b[1:,1:]
  
    uv = np.fromfile(allfilesu[it],'f4').reshape(2,N1,N1).transpose(0,2,1)
    u = uv[0,1:,1:]
    v = uv[1,1:,1:]
  else:
    b = ds['b'][it,:,:].squeeze()
    u = ds['u.x'][it,:,:].squeeze()
    v = ds['u.y'][it,:,:].squeeze()

  pe[it] = -Ri*np.sum(b*zc)*Delta**2
  ke[it] = 0.5*np.sum(u**2 + v**2)*Delta**2
  pe_bg[it] = -Ri*np.sum(np.sort(b,None)*zo)*Delta**2
  
  dudx, dudz = np.gradient(u,Delta)
  dvdx, dvdz = np.gradient(v,Delta)
  
  epsilon[it] = np.sum(dudx**2 + dudz**2 + dvdx**2 + dvdz**2)/Re*Delta**2
  wb[it] = np.sum(v*b)*Ri*Delta**2

dkedt = np.diff(ke)
dpedt = np.diff(pe)
M = np.diff(pe_bg)

Gamma = -np.cumsum(wb[5:])/np.cumsum(epsilon[5:])
eta_1 = -np.cumsum(wb[5:])/np.cumsum(-wb[5:]+epsilon[5:])
eta_2 = np.cumsum(M[4:])/np.cumsum(M[4:]+epsilon[5:])

eta_2 = M[4:]/(M[4:]+epsilon[5:])

plt.figure(); 
plt.plot(np.cumsum(-wb[5:]),'k',label=r"$-w'b'$")
plt.plot(pe_bg[5:]-pe_bg[0],'r--',label=r'$\Delta h^*$')
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$\Delta E$')
plt.savefig(dir0 + "w95.pdf")

plt.figure(); 
plt.plot(np.cumsum(wb[5:]),'k',label="$w'b'$")
plt.plot(np.cumsum(-epsilon[5:]),'k--',label=r"$\epsilon$")
plt.plot(np.cumsum(wb[5:]-epsilon[5:]),'r',label="$w'b' + \epsilon$")
plt.plot(ke[5:]-ke[5],'b--',label="$K$")
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$\Delta E$')
plt.savefig(dir0 + "budget.pdf")


##################################
#	SUPPLEMENTARY
##################################

plt.figure()
plt.imshow(b,cmap=plt.cm.afmhot, origin='lower', extent=(0,25,0,25))
plt.xlabel('x')
plt.ylabel('z')
plt.colorbar()
plt.savefig(dir0 + "snapshot1000.pdf")

#print("Delta KE = {0}".format(ke[-1]-ke0))
#print("Delta PE = {0}".format(pe[-1]-pe0))

plt.figure()
plt.plot(dkedt[5:],label='dkedt')
plt.plot(dpedt[5:],label='dpedt')
plt.plot(dpedt[5:] + dkedt[5:],label='dEdt')
plt.plot(-epsilon[5:],label='epsilon')
plt.plot(wb[5:],label='wb')
plt.legend()

plt.figure()
plt.plot(Gamma,label="Gamma")
plt.plot(eta_1,label="eta_1")
plt.plot(eta_2,label="eta_2")
plt.xlabel('time')
plt.ylabel('efficiency')
plt.legend()
