import numpy as np
import matplotlib.pyplot as plt
import MITgcmutils as mit
import time

#-- config and directories --
config = 'ocac'
dir_in  = '/tank/chaocean/qjamet/RUNS/' + config.upper() + '/'
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'
dir_out = '/tank/chaocean/qjamet/RUNS/data_chao12/' + config + '/'
fileN = 'MOCyzt_' + config + '_ensemble_macroICs_py.bin'

#-- membres and runs --
mmem = np.arange(12, 24)
nmem = len(mmem);
yyr = np.arange(1963, 2013)
nyr = len(yyr)
yrIni = 1963

#-- time parameters --
dt = 200
spy = 86400*365
dump = 5*86400       #5-d dumps
d_iter = int(dump/dt)
ndump = int(86400*365/dump)
offset = int((yyr[0]-1958)*spy/dt)
iter = np.arange(d_iter, (ndump*d_iter)*nyr+d_iter, d_iter) + offset
niter = len(iter)

#-- grid --
dxG = mit.rdmds(dir_grd + 'DXG')
drF = mit.rdmds(dir_grd + 'DRF')
ny, nx = dxG.shape
nr = len(drF)
#- mask -
hS = mit.rdmds(dir_grd + 'hFacS')
hFac = np.tile(drF, (1, ny, nx)) * np.tile(dxG[np.newaxis, :, :], (nr, 1, 1)) * hS

#-------------------------------------------------
#	load data, compute and save AMOC
# Dimension of output file is [nyr*nmem*ndump*nr*ny]
#-------------------------------------------------
moc=np.zeros([nr, ny])
#
for iyr in range(nyr):
  tmp_iter = iter[iyr*ndump:(iyr+1)*ndump]
  for imem in range(nmem):
    for iiter in range(ndump):
      print("(year: %04.f, memb: %02.f, iter: %010.f, dump: %02.f)" % \
        (yyr[iyr], mmem[imem], tmp_iter[iiter], iiter) )
      t0 = time.time()
      tmp = mit.rdmds(dir_in + "memb%02.f/run%04.f/ocn/diag_ocnTave" \
        % (mmem[imem], yyr[iyr]), itrs=tmp_iter[iiter], rec=3, usememmap=True)
      #- compute transport [Sv] -
      tmp = (tmp*hFac*1e-6)
      #- compute amoc -
      #tmp = -tmp.sum(2)[nr:None:-1, :].cumsum(0)
      moc[nr:None:-1, :] = -tmp.sum(2)[nr:None:-1, :].cumsum(0)
      t1 = time.time()
      print("It takes %02.f sec" % (t1-t0))
      #- save -
      f = open(dir_out + fileN, 'ab')
      moc.reshape([nr*ny]).astype('>f4').tofile(f)
      f.close()

