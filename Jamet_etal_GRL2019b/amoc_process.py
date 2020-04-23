import numpy as np
from multiprocessing import Pool
import statsmodels.api as sm 	# for LOESS (or LOWESS) smoothing
import time
from scipy import signal

#-- directories --
config = 'orar'
dir_in = '/tank/chaocean/qjamet/RUNS/data_chao12/'+config+'/'
#dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_out = dir_in

flg_detrend = 1

#-- runs parameters --
nDump = 73
spy = 86400*365
nYr = 50
time_ens = 1963 + np.arange(1/float(nDump),nYr,1/float(nDump))
ny, nr, nt, nb_memb = 900, 46, nDump*nYr, 12
dim1 = nb_memb*nr*ny


#---------------------------------
#	Load AMOC data 
#	dim: [nYear nb_memb nDump nr ny]
#---------------------------------
fileN = 'MOCyzt_' + config + '_ensemble.bin'
f = open(dir_in+fileN,'r')
moc = np.fromfile(f,'>f4')	#big-indian ('>'), real*4 ('f4')
f.close()
moc = moc.reshape([nYr,nb_memb,nDump,nr,ny])
moc = np.transpose(moc, (0,2,1,3,4)).reshape([nt,nb_memb*nr*ny])


#---------------------------------------
#		detrending
#---------------------------------------
if flg_detrend:
  lowess = sm.nonparametric.lowess
  # to get a reference time ...
  t0 = time.time()
  toto = lowess(moc[:,0],time_ens)[:,1]
  t1 = time.time()
#  print( "Without parallelization, should have take %f seconds.\n" %(t1-t0)*dim1 )
  #-- define the detrending procedure to parallelize --
  def moc_proc(ijk):
    print("Detrending: %f pc\n" %ijk )
    trend = lowess(moc[:,ijk],time_ens,return_sorted=False)
    return moc[:,ijk]-trend
  t0 = time.time()
  if __name__ == '__main__':
    p = Pool(16)
    tmp_mocd=p.map(moc_proc, np.arange(0,dim1))
  t1 = time.time()
  print( "Now takes only %f seconds.\n" %(t1-t0) )
  #-- transform the ijk list of nt-long time series 
  # into a nb_memb x nt x nr x ny matrix and save --
  moc_d = np.zeros([nt,dim1])
  for ijk in range(0,dim1):
    moc_d[:,ijk] = tmp_mocd[ijk]
  moc_d = moc_d.reshape([nt,nb_memb,nr,ny])
  moc_d = np.transpose(moc_d, (1,0,2,3))	#[nb_memb,nt,nr,ny]
  #-- save detrended data --
  fileN2 = 'MOCyzt_'+config+'_ensemble_detrend_py.bin'
  f = open(dir_out+fileN2,'wb')
  moc_d.reshape([nb_memb*nt*nr*ny]).astype('>f4').tofile(f)
  f.close()
else:
  fileN2 = 'MOCyzt_'+config+'_ensemble_detrend_py.bin'
  f = open(dir_in+fileN2,'r')
  moc_d = np.fromfile(f,'>f4')		# comes out at [nb_memb*nYr*nDump*nr*ny,1]
  f.close()


#---------------------------------------
# Remove ensemble mean seasonal cycle
#---------------------------------------
moc_seas = np.mean(np.mean(moc_d.reshape([nb_memb,nYr,nDump,nr,ny]),0),0)
moc_d = moc_d.reshape([nb_memb,nt,nr,ny]) - \
        np.tile(moc_seas[np.newaxis,np.newaxis,:,:,:],(nb_memb,nYr,1,1,1)).reshape([nb_memb,nt,nr,ny])


#--------------------------------------
#	Low-pass filter
#--------------------------------------
moc_d = np.transpose(moc_d, (1,0,2,3)).reshape([nt,nb_memb*nr*ny])
fs = nDump*1.0		# sampling freq [yr-1]
cof = 1.0		# cut-off freq [yr-1]
b, a = signal.butter(10, cof/(fs/2), btype='low')
#-- define the low-pass filter procedure to parallelize --
def moc_lpf(ijk):
  return signal.filtfilt(b, a, moc_d[:,ijk])

if __name__ == '__main__':
  p = Pool(16)
  tmp_moc = p.map(moc_lpf, np.arange(0,dim1))

moc_lpf = np.zeros([nt,dim1])
for ijk in range(0,dim1):
  moc_lpf[:,ijk] = tmp_moc[ijk]

moc_lpf = np.transpose(np.asarray(tmp_moc), (1, 0)).reshape([nt, nb_memb, nr, ny])
moc_lpf = moc_lpf.reshape([nt,nb_memb,nr,ny])
moc_lpf = np.transpose(moc_lpf, (1,0,2,3))        #[nb_memb,nt,nr,ny]


#--------------------------------------
#	 	SAVE	
#--------------------------------------
fileN3 = 'MOCyzt_'+config+'_ensemble_detrend_1ylpd_py.bin'
f = open(dir_out+fileN3,'wb')
moc_lpf.reshape([nb_memb*nt*nr*ny]).astype('>f4').tofile(f)
f.close()

import sys
sys.exit()



#==========================================================
fileN3 = 'MOCyzt_'+config+'_ensemble_detrend_1ylpd_py.bin'
f = open(dir_out+fileN3,'r')
moc_lpf = np.fromfile(f,'>f4').reshape([nb_memb, nt, nr, ny])
f.close()



#moc_lpf = signal.filtfilt(b, a, moc_d, axis=1)
#-- to plot some results --
#f1, Pxx_ts = signal.periodogram(moc_ts[0,:], fs)
#f1, Pxx_lpf = signal.periodogram(moc_lpf[0,:], fs)
#plt.semilogx(1/f1, Pxx_ts)
#plt.semilogx(1/f1, Pxx_lpf)
#plt.show()

#-- remove first and last years --
moc_lpf = moc_lpf[:,nDump-1:-nDump,:,:]
nt2 = moc_lpf.shape[1]
moc_lpf_f = np.mean(moc_lpf,0)
moc_lpf_i = moc_lpf - np.tile(moc_lpf_f[np.newaxis,:,:,:], (nb_memb, 1, 1, 1))
#- forced variance -
A2_f = np.var(moc_lpf_f,0);
#- intrinsic variance -
A2_i = np.mean( np.var(moc_lpf_i,0), 0);
#- total variability -
A2_tot = A2_i + A2_f;

tmpy, tmpr = np.meshgrid(yG[:,0],rF[0:nr])
tmp_rat = A2_i/A2_tot
tmp_rat[np.where(tmp_rat == 0)] = np.nan
fig1 = plt.figure(figsize=(10,5))
plt.contourf(tmpy, tmpr, tmp_rat, np.arange(0, 1, 0.1))
plt.colorbar()
plt.show()


