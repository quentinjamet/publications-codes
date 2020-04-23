% Look at gloab mean trends for the diff simulations
% Dimensions of files are
%       [nDim nConf nDump nYear], with
%       - nDim: the spacial dimension (ny*nr for the moc_yzt) or the number of indices
%       (5 for the global mean T, S and KE, 1 for eta)
%       - nDump: the number of dump /yr (i.e. 73),
%       - nConf: the number of configuraions (i.e. 6)
%       - nYear: the number of year
%
% KPPhbl is evaluated at tracer points
%
%-- estimation of the diff in upward longwave radiation 
%   induced by a 8°C SST difference (24 in FULL, 32 in CLIM) --
%
%C net upward long wave
%      Rnl = 0.96 _d 0*(stefan*(tsw+celsius2K)**4) !Net longwave (up = +).
%
%	stefan = 5.67 _d -8
%	tsw: surface temperature withou skin correction
%
% TODO:
% 	find where goes the excess of heat in CHEAP-CLIM
%	- Look at THFLUX, and if no OK, then the missing part should be in WTHMASS 
%	(see ./MITgcm/doc/diags_changes.txt)




clear all; close all


%-- directories --
dir_in = '/tank/chaocean/qjamet/RUNS/test_cheap025/data/';
dir_in2 = '/tank/chaocean/qjamet/RUNS/test_cheap025/';
dir_grd = '/tank/chaocean/qjamet/RUNS/test_cheap025/gridMIT/';
dir_fig = '/tank/users/qjamet/Figures/publi/note_bams/';

config = {'cheap_clim_atm','cheap_fv_wind','rest_clim_atm','rest_fv_wind'};
nConf = length(config);
ieee = 'b';
accu = 'real*4';
if strcmp(accu,'real*4')
  accu2 = 4;
else
  accu2 = 8;
end

%-- load grid param --
global xC yC xG yG rC Depth dxC dyC rA rAs rAw drC xG yG drF rF dxG dyG
loadGRD(dir_grd)
xC = xC-360;
xG = xG-360;
[nx,ny] = size(xC);
[nr] = length(rC);
%-- box definition --
[ii] = find(xC(:,1)>=-40 & xC(:,1)<=-35);
[jj] = find(yC(1,:)>=30 & yC(1,:)<=35);
[nx1] = length(ii);
[ny1] = length(jj);
dxC1 = dxC(ii,jj);
dyC1 = dyC(ii,jj);
dxG1 = dxG(ii,jj);
dyG1 = dyG(ii,jj);
xC1 = xC(ii,jj);
yC1 = yC(ii,jj);
xG1 = xG(ii,jj);
yG1 = yG(ii,jj);


%-- time parameters --
runs = 1958:1967;
nYr = length(runs);
yrIni = 1958;

%- time parameters - 
[dt] = 450;
spy = 86400*365;
[dump] = 5*86400;       %5-d dumps
d_iter = dump/dt;
nDump = 86400*365/dump;
offset = (runs(1)-1958)*spy/dt;;
iter = [0:d_iter:(nDump*d_iter)*nYr] + offset;
iter = iter(2:end);
[niter] = length(iter);
date_y = floor((iter*dt/86400-1)/365 ) + 1958;
date_d = (iter*dt/86400);
ddate = date_d/365 + 1958;
niter2 = 2*nDump;

nDiag1 = 5;             % ocean
nDiag2 = 9;             % cheapaml


%------------------------
% Load pre-extracted data
% dim: 	[nx1,ny1,nr,niter,nDiag1,nConf]
%	[nx1,ny1,niter,nDiag2,nConf]
% ocn diag: t,s,u,v,w
% cheap diag: t2, q2, sh, lh, qnet, emp, u10, v10, precip
%------------------------

%-- oceanic diag --
fid = fopen([dir_in 'ocn_diag_box1_1958_67.bin'],'r','b');
diag_ocn = fread(fid,'real*4');
fclose(fid);
diag_ocn = reshape(diag_ocn,[nx1 ny1 nr niter nDiag1 nConf]);
%-- cheap diag --
fid = fopen([dir_in 'cheap_diag_box1_1958_67.bin'],'r','b');
diag_cheap = fread(fid,'real*4');
fclose(fid);
diag_cheap = reshape(diag_cheap,[nx1 ny1 niter nDiag2 nConf]);



%===================================================
%		COMPUTE HEAT BUDGET
%===================================================
tt = squeeze(diag_ocn(:,:,:,1:niter2,1,:));
uu = squeeze(diag_ocn(:,:,:,1:niter2,3,:));
vv = squeeze(diag_ocn(:,:,:,1:niter2,4,:));
ww = squeeze(diag_ocn(:,:,:,1:niter2,5,:));

rA2_box = rA(ii(2:end-1),jj(2:end-1));

%------------------------------------
% get the mixed layer depth
%------------------------------------
kpphbl = zeros(nx1-2,ny1-2,niter2,nConf);
for iconf = 1:2
 for irun = 1:2
  tmp_diag = rdmds([dir_in2 config{iconf} '/run' num2str(runs(irun)) '_kpp/ocn/kpp2d'],NaN);
  kpphbl(:,:,(irun-1)*nDump+1:irun*nDump,iconf) = tmp_diag(ii(2:nx1-1),jj(2:ny1-1),1,:);
 end % for irun
end % for iConf
kpphbl(:,:,:,3:4) = kpphbl(:,:,:,1:2);
tmp_kpphbl = reshape(kpphbl,[(nx1-2)*(ny1-2)*niter2 nConf]);

%-- averaged within the box --
hbl_avg = squeeze(sum(sum( kpphbl.*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ sum(rA2_box(:)));



%---------------------------------------------------
% Compute tendency in the mixed layer \partial_t <T>
%---------------------------------------------------
% computed earlier <\partial_t T>	-->> dTdt_hbl
% while it is \partial_t <T>		-->> dThbl_dt
% where the entrainement term is evaluated separately ...


%-- first integrate over the mixed layer depth --
tmp1 = reshape(permute(tt(2:nx1-1,2:ny1-1,:,:,:),[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2 nConf]);
tmp_hbl = zeros((nx1-2)*(ny1-2)*niter2,nConf);
tmp_hbl_full = zeros(1,(nx1-2)*(ny1-2)*(niter2));
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*niter2;
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  tmp_hbl(ijt,iconf) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  if iconf == 1
   kk = find(abs(rC) < tmp_kpphbl(ijt,2));
   tmp_hbl_full(ijt) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  end
 end % for ijtc
end % for iconf
tmp_hbl = reshape(tmp_hbl,[(nx1-2) (ny1-2) niter2 nConf]);
tmp_hbl_full = reshape(tmp_hbl_full,[(nx1-2) (ny1-2) niter2]);

%-- compute time derivative --
tmp_dt = (tmp_hbl(:,:,2:niter2,:)-tmp_hbl(:,:,1:niter2-1,:)) ./ ...
	repmat(dump,[nx1-2 ny1-2 niter2-1 nConf]);
tmp_dt_full = (tmp_hbl_full(:,:,2:niter2)-tmp_hbl_full(:,:,1:niter2-1)) ./ ...
	repmat(dump,[nx1-2 ny1-2 niter2-1]);

%-- averaging within the box --
dThbl_dt = squeeze(sum(sum( tmp_dt.*repmat(rA2_box,[1 1 niter2-1 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));
dThbl_dt_full = squeeze(sum(sum( tmp_dt_full.*repmat(rA2_box,[1 1 niter2-1]) ,1),2 ) ./ ...
        sum(rA2_box(:)));



%------------------
% Compute flux term
% [+=up], so look at -Qnet
%------------------
Cp=4.1655e3;
rho0 = 1e3;
tmp = -squeeze(diag_cheap(2:nx1-1,2:ny1-1,1:niter2,5,:)) ./ ...
	(Cp.*rho0.*kpphbl);
qnet_box = squeeze(sum(sum( tmp.*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
	sum(rA2_box(:)));
%-- compute qnet for clim with kpphbl of full --
tmp = -squeeze(diag_cheap(2:nx1-1,2:ny1-1,1:niter2,5,1)) ./ ...
        (Cp.*rho0.*kpphbl(:,:,:,2));
qnet_box_full = squeeze(sum(sum( tmp.*repmat(rA2_box,[1 1 niter2]) ,1),2 ) ./ ...
        sum(rA2_box(:)));



%----------------------------
% Compute advective terms 
%----------------------------
%-- compute --
udxt_u = uu(2:nx1,:,:,:,:) .* ( tt(2:nx1,:,:,:,:)-tt(1:nx1-1,:,:,:,:) ) ./ ...
   repmat(dxC1(2:nx1,:),[1 1 nr niter2 nConf]);
vdyt_v = vv(:,2:ny1,:,:,:) .* ( tt(:,2:ny1,:,:,:)-tt(:,1:ny1-1,:,:,:) ) ./ ...
   repmat(dyC1(:,2:ny1),[1 1 nr niter2 nConf]);
wdzt_w =  ww(2:nx1-1,2:ny1-1,2:nr,:,:) .* ... 
   ( tt(2:nx1-1,2:ny1-1,2:nr,:,:)-tt(2:nx1-1,2:ny1-1,1:nr-1,:,:) ) ./ ...
   repmat(drC(2:nr),[nx1-2 ny1-2 1 niter2 nConf]);

%-- interpolate at t-pts --
%- u,v -
toto_u = zeros(nx1-1,ny1);
[t_aux,tri_u,wei_u] = my_griddata1(xG1(2:nx1,:),yC1(2:nx1,:),toto_u,...
	xC1(2:nx1,:),yC1(2:nx1,:),{'QJ'});
toto_v = zeros(nx1,ny1-1);
[t_aux,tri_v,wei_v] = my_griddata1(xC1(:,2:ny1),yG1(:,2:ny1),toto_v,...
	xC1(:,2:ny1),yC1(:,2:ny1),{'QJ'});

udxt_u = reshape(udxt_u,[nx1-1 ny1 nr*niter2*nConf]);
vdyt_v = reshape(vdyt_v,[nx1 ny1-1 nr*niter2*nConf]);
udxt_c = zeros(nx1-1,ny1,nr*niter2*nConf);
vdyt_c = zeros(nx1,ny1-1,nr*niter2*nConf);
for ktc = 1:nr*niter2*nConf
  udxt_c(:,:,ktc) = my_griddata2(xG1(2:nx1,:),yC1(2:nx1,:),udxt_u(:,:,ktc),...
	xC1(2:nx1,:),yC1(2:nx1,:),tri_u,wei_u);
  vdyt_c(:,:,ktc) = my_griddata2(xC1(:,2:ny1),yG1(:,2:ny1),vdyt_v(:,:,ktc),...
	xC1(:,2:ny1),yC1(:,2:ny1),tri_v,wei_v);
end % for ktc
udxt_c = udxt_c(1:end-1,2:ny1-1,:);
vdyt_c = vdyt_c(2:nx1-1,1:end-1,:);
udxt_c = reshape(udxt_c,[nx1-2 ny1-2 nr niter2 nConf]);
vdyt_c = reshape(vdyt_c,[nx1-2 ny1-2 nr niter2 nConf]);
%- w -
wdzt_w = reshape(permute(wdzt_w,[3 1 2 4 5]),[nr-1 (nx1-2)*(ny1-2)*niter2*nConf]);
wdzt_c = zeros(nr,(nx1-2)*(ny1-2)*niter2*nConf);
for iii = 1:size(wdzt_c,2)
  tmp = zeros(1,nr);
  tmp(2:nr) = wdzt_w(:,iii);
  tmp(1) = tmp(2) - ((tmp(3)-tmp(2))*drF(2))/drF(1);
  wdzt_c(:,iii) = interp1(rF(1:nr),tmp,rC);
end
wdzt_c = permute(reshape(wdzt_c,[nr (nx1-2) (ny1-2) niter2 nConf]),[2 3 1 4 5]);
advec_xy = udxt_c + vdyt_c;
advec_z = wdzt_c;


%-- vertical integral within the mixed layer --
tmp1 = reshape(permute(advec_xy,[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2 nConf]);
tmp1_hbl = zeros((nx1-2)*(ny1-2)*niter2,nConf);
tmp1_hbl_full = zeros(1,(nx1-2)*(ny1-2)*niter2);
tmp2 = reshape(permute(advec_z,[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2 nConf]);
tmp2_hbl = zeros((nx1-2)*(ny1-2)*niter2,nConf);
tmp2_hbl_full = zeros(1,(nx1-2)*(ny1-2)*niter2);
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*(niter2-1);
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  tmp1_hbl(ijt,iconf) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  tmp2_hbl(ijt,iconf) = sum(tmp2(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  if iconf == 1
   kk = find(abs(rC) < tmp_kpphbl(ijt,2));
   tmp1_hbl_full(ijt) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
   tmp2_hbl_full(ijt) = sum(tmp2(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  end
 end % for ijtc
end % for iconf
tmp1_hbl = reshape(tmp1_hbl,[(nx1-2) (ny1-2) niter2 nConf]);
tmp1_hbl_full = reshape(tmp1_hbl_full,[(nx1-2) (ny1-2) niter2]);
tmp2_hbl = reshape(tmp2_hbl,[(nx1-2) (ny1-2) niter2 nConf]);
tmp2_hbl_full = reshape(tmp2_hbl_full,[(nx1-2) (ny1-2) niter2]);

%-- averaging within the box --
advxy_hbl = squeeze(sum(sum( tmp1_hbl.*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));
advxy_hbl_full = squeeze(sum(sum( tmp1_hbl_full.*repmat(rA2_box,[1 1 niter2]) ,1),2 ) ./ ...
        sum(rA2_box(:)));
advz_hbl = squeeze(sum(sum( tmp2_hbl.*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));
advz_hbl_full = squeeze(sum(sum( tmp2_hbl_full.*repmat(rA2_box,[1 1 niter2]) ,1),2 ) ./ ...
        sum(rA2_box(:)));




%-------------------------------------------
% Compute dissipative terms and entrainement
%-------------------------------------------

%-- as residual --
% interpolate in time the tendency term
tmp_dTdt_hbl = zeros(niter2,nConf);
for iConf = 1:nConf
  tmp_dTdt_hbl(:,iConf) = interp1( (ddate(1:niter2-1)+ddate(2:niter2))./2, ...
	dThbl_dt(:,iConf),ddate(1:niter2));	% change from <\partial_t> to 
%	dTdt_hbl(:,iConf),ddate(1:niter2));	% \partial <T>
end % for iConf
tmp_dTdt_hbl_full = zeros(niter2);
tmp_dTdt_hbl_full = interp1( (ddate(1:niter2-1)+ddate(2:niter2))./2, ...
	dTdt_hbl_full,ddate(1:niter2));

%-- extrapolate at both ends --
tmp_dTdt_hbl(1,:) = tmp_dTdt_hbl(2,:) - ...
	(tmp_dTdt_hbl(3,:)-tmp_dTdt_hbl(2,:));
tmp_dTdt_hbl(end,:) = tmp_dTdt_hbl(end-1,:) + ...
	(tmp_dTdt_hbl(end-1,:)-tmp_dTdt_hbl(end-2,:));
tmp_dTdt_hbl_full(1) = tmp_dTdt_hbl_full(2) - ...
	(tmp_dTdt_hbl_full(3)-tmp_dTdt_hbl_full(2));
tmp_dTdt_hbl_full(end) = tmp_dTdt_hbl_full(end-1) + ...
	(tmp_dTdt_hbl_full(end-1)-tmp_dTdt_hbl_full(end-2));

%res_hbl = tmp_dTdt_hbl + advxy_hbl + advz_hbl - qnet_box;
%res_hbl_full = tmp_dTdt_hbl_full' + advxy_hbl_full + advz_hbl_full - qnet_box_full;
res_hbl = tmp_dTdt_hbl - qnet_box;
res_hbl_full = tmp_dTdt_hbl_full' - qnet_box_full;
%---------------


%-- horizontal diffusion -- 
%	O(1e-4 dec/day) for CHEAP-FULL
kh = 200;		%[m^2.s]
khdTdx = kh .* ( tt(2:nx1,:,:,:,:)-tt(1:nx1-1,:,:,:,:) ) ./ ...
	repmat(dxC1(2:nx1,:),[1 1 nr niter2 nConf]);
khdTdx_dx = ( khdTdx(2:nx1-1,2:ny1-1,:,:,:)-khdTdx(1:nx1-2,2:ny1-1,:,:,:) ) ./ ...
	repmat(dxG1(3:nx1,2:ny1-1),[1 1 nr niter2 nConf]);
khdTdy = kh .* ( tt(:,2:ny1,:,:,:)-tt(:,1:ny1-1,:,:,:) ) ./ ...
	repmat(dyC1(:,2:ny1),[1 1 nr niter2 nConf]);
khdTdy_dy = ( khdTdy(2:nx1-1,2:ny1-1,:,:,:)-khdTdy(2:nx1-1,1:ny1-2,:,:,:) ) ./ ...
	repmat(dyG1(2:nx1-1,3:ny1),[1 1 nr niter2 nConf]);
diff_h = khdTdx_dx + khdTdy_dy;

%-- vertical integral within the mixed layer --
tmp1 = reshape(permute(diff_h,[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2 nConf]);
tmp1_hbl = zeros((nx1-2)*(ny1-2)*niter2,nConf);
tmp1_hbl_full = zeros(1,(nx1-2)*(ny1-2)*niter2);
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*(niter2-1);
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  tmp1_hbl(ijt,iconf) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  if iconf == 1
   kk = find(abs(rC) < tmp_kpphbl(ijt,2));
   tmp1_hbl_full(ijt) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  end
 end % for ijtc
end % for iconf
tmp1_hbl = reshape(tmp1_hbl,[(nx1-2) (ny1-2) niter2 nConf]);
tmp1_hbl_full = reshape(tmp1_hbl_full,[(nx1-2) (ny1-2) niter2]);

%-- averaging within the box --
diffh_hbl = squeeze(sum(sum( tmp1_hbl.*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));
diffh_hbl_full = squeeze(sum(sum( tmp1_hbl_full.*repmat(rA2_box,[1 1 niter2]) ,1),2 ) ./ ...
        sum(rA2_box(:)));




%-- Vertical diffusion < \nabla.(K.\nabla.(T)) > --
% Kv is expressed at w-pts (SM_P__LR)
% Kv(:,:,1) = 0 -->> by definition
kv_backG = 1e-5;					%[m^2.s^-1]
kv_tot = zeros(nx1-2,ny1-2,nr,niter2,nConf);		%[m^2.s^-2]
for iconf = 1:2
 for irun = 1:2
  tmp_diag = rdmds([dir_in2 config{iconf} '/run' num2str(runs(irun)) '_kpp/ocn/kpp3d_1'],NaN);
  kv_tot(:,:,:,(irun-1)*nDump+1:irun*nDump,iconf) = tmp_diag(ii(2:nx1-1),jj(2:ny1-1),:,2,:);
 end % for irun
end % for iConf
clear tmp_diag
% add background vertical diffusivity
kv_tot = kv_tot + kv_backG;

%- dtdz_w -
rr_dtdz = [0; rC(1:nr-1)-[squeeze(drC(2:nr))/2] ];
tmp = zeros(nx1-2,ny1-2,nr,niter2,nConf);
tmp(:,:,2:nr,:,:) = ...
	( tt(2:nx1-1,2:ny1-1,2:nr,:,:)-tt(2:nx1-1,2:ny1-1,1:nr-1,:,:) ) ./ ...
        repmat( drC(2:nr),[nx1-2 ny1-2 1 niter2 nConf] );
% interpolated linearly at w-pts (drF)
dtdz_w = zeros(nr,(nx1-2)*(ny1-2)*niter2*nConf);
tmp = reshape(permute(tmp,[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2*nConf]);
for iiii = 1:(nx1-2)*(ny1-2)*niter2*nConf
  dtdz_w(:,iiii) = interp1(rr_dtdz,tmp(:,iiii),rF(1:nr));
end
dtdz_w = permute(reshape(dtdz_w,[nr (nx1-2) (ny1-2) niter2 nConf]),[2 3 1 4 5]);

%- Kv \partial_z T (at rF(1:nr), w-pts ; =0 at surface, i.e. k=1) -
kv_dtdz = kv_tot .* dtdz_w; 


%- extract kv and dTdz at the base of the mixed layer -
tmp1 = reshape(permute(kv_tot,[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2 nConf]);
tmp_kv_bot = zeros((nx1-2)*(ny1-2)*niter2,nConf);
tmp2 = reshape(permute(dtdz_w,[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2 nConf]);
tmp_dTdz_bot = zeros((nx1-2)*(ny1-2)*niter2,nConf);
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*niter2
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  kk = kk(end);
  %- kv_tot and dtdz_w are at w-pts, and kpphbl is a t-pts -
  tmp_kv_bot(ijt,iconf) = ( tmp1(kk,ijt,iconf) + tmp1(kk+1,ijt,iconf))./2;
  tmp_dTdz_bot(ijt,iconf) = ( tmp2(kk,ijt,iconf) + tmp2(kk+1,ijt,iconf))./2;
 end % for ijt
end % for iconf
tmp_kv_bot = reshape(tmp_kv_bot,[(nx1-2) (ny1-2) niter2 nConf]);
tmp_dTdz_bot = reshape(tmp_dTdz_bot,[(nx1-2) (ny1-2) niter2 nConf]);
%- averaging within the box -
tmp_kv = squeeze(sum(sum( tmp_kv_bot .*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));
tmp_dtdz = squeeze(sum(sum( tmp_dTdz_bot .* repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));




%- compute 1/h Kv \partial_z T at z=-h - 
tmp1 = reshape(permute(kv_dtdz,[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*niter2 nConf]);
tmp_hbot = zeros((nx1-2)*(ny1-2)*niter2,nConf);
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*niter2
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  kk = kk(end);
  %- kv_dtdz is at w-pts, and kpphbl is a t-pts -
  tmp_hbot(ijt,iconf) = (1/tmp_kpphbl(ijt,iconf)) .* ...
	 ( tmp1(kk,ijt,iconf)+ tmp1(kk+1,ijt,iconf))./2;
 end % for ijt
end % for iconf
tmp_hbot = reshape(tmp_hbot,[(nx1-2) (ny1-2) niter2 nConf]);
%-- averaging within the box --
diffz_hbot = squeeze(sum(sum( tmp_hbot.*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));



%- compute \partial_z kv_dtdz at t-pts (by construction)
%	last grid point on the vert. is missing, but first one is OK -
dkvtdz = (kv_dtdz(:,:,2:nr,:,:) - kv_dtdz(:,:,1:nr-1,:,:)) .* ...
	repmat(drF(1:nr-1),[nx1-2 ny1-2 1 niter2 nConf]);


%-- vertical integral within the mixed layer --
tmp1 = reshape(permute(dkvtdz,[3 1 2 4 5]),[nr-1 (nx1-2)*(ny1-2)*niter2 nConf]);
tmp_hbl = zeros((nx1-2)*(ny1-2)*niter2,nConf);
tmp_hbl_full = zeros(1,(nx1-2)*(ny1-2)*niter2);
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*niter2;
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  tmp_hbl(ijt,iconf) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  if iconf == 1
   kk = find(abs(rC) < tmp_kpphbl(ijt,2));
   tmp_hbl_full(ijt) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  end
 end % for ijtc
end % for iconf
tmp_hbl = reshape(tmp_hbl,[(nx1-2) (ny1-2) niter2 nConf]);
tmp_hbl_full = reshape(tmp_hbl_full,[(nx1-2) (ny1-2) niter2]);

%-- averaging within the box --
diffz_hbl = squeeze(sum(sum( tmp_hbl.*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
        sum(rA2_box(:)));
diffz_hbl_full = squeeze(sum(sum( tmp_hbl_full.*repmat(rA2_box,[1 1 niter2]) ,1),2 ) ./ ...
        sum(rA2_box(:)));




%-- Entrainement [(1/h)*\partial_t(h) + w|z=-h] * [<T> - T|z=-h] 
%	see Peter et al., JGR 2006 --

%- 1/h \partial_t(h) -
dhdt = zeros(nx1-2,ny1-2,niter2+2,nConf);
dhdt(:,:,2:niter2,:) = (kpphbl(:,:,2:niter2,:)-kpphbl(:,:,1:niter2-1,:)) ./ ...
	repmat(dump,[nx1-2 ny1-2 niter2-1 nConf]);
%- extrapolate at both ends -
dhdt(:,:,1,:) = dhdt(:,:,2,:) - (dhdt(:,:,2,:)-dhdt(:,:,2,:));
dhdt(:,:,niter2+2,:) = dhdt(:,:,niter2+1,:) + (dhdt(:,:,niter2+1,:)-dhdt(:,:,niter2-2,:));
%- interpolate in time -
tmp = reshape(permute(dhdt,[3 1 2 4]),[niter2+2 (nx1-2)*(ny1-2)*nConf]);
dhdt_t = zeros(niter2,(nx1-2)*(ny1-2)*nConf);
for iiii = 1:(nx1-2)*(ny1-2)*nConf
  dhdt_t(:,iiii) = interp1([0:niter2+1],tmp(:,iiii),[1:niter2]);
end %for iiii
dhdt_t = permute(reshape(dhdt_t,[niter2 (nx1-2) (ny1-2) nConf]),[2 3 1 4]);

%- extract w|z=-h 
%	from the development of the advection term, I think -
tmp1 = reshape(permute(ww(2:nx1-1,2:ny1-1,:,:,:),[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*(niter2) nConf]);
ww_hh = zeros((nx1-2)*(ny1-2)*(niter2),nConf);
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*(niter2);
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  ww_hh(ijt,iconf) = tmp1(kk(end),ijt,iconf);
 end
end
ww_hh = reshape(ww_hh,[(nx1-2) (ny1-2) (niter2) nConf]);

%- <T> - T(z=-h) -
tmp1 = reshape(permute(tt(2:nx1-1,2:ny1-1,:,:,:),[3 1 2 4 5]),[nr (nx1-2)*(ny1-2)*(niter2) nConf]);
tt_hbl = zeros((nx1-2)*(ny1-2)*(niter2),nConf);
tt_hh = zeros((nx1-2)*(ny1-2)*(niter2),nConf);
tt_hbl_full = zeros(1,(nx1-2)*(ny1-2)*(niter2));
tt_hh_full = zeros(1,(nx1-2)*(ny1-2)*(niter2));
for iconf = 1:nConf
 for ijt = 1:(nx1-2)*(ny1-2)*(niter2);
  kk = find(abs(rC) < tmp_kpphbl(ijt,iconf));
  tt_hbl(ijt,iconf) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
  tt_hh(ijt,iconf) = tmp1(kk(end),ijt,iconf);
  if iconf == 1
    kk = find(abs(rC) < tmp_kpphbl(ijt,2));
    tt_hbl_full(ijt) = sum(tmp1(kk,ijt,iconf) .* squeeze(drF(kk))) ./ sum(drF(kk));
    tt_hh_full(ijt) = tmp1(kk(end),ijt,iconf);
  end % if
 end % for ijtc
end % for iconf
tt_hbl = reshape(tt_hbl,[(nx1-2) (ny1-2) (niter2) nConf]);
tt_hh = reshape(tt_hh,[(nx1-2) (ny1-2) (niter2) nConf]);
tt_hbl_full = reshape(tt_hbl_full,[(nx1-2) (ny1-2) (niter2)]);
tt_hh_full = reshape(tt_hh_full,[(nx1-2) (ny1-2) (niter2)]);

%- Entrainement -
% the vertical component of the advective part can be placed here instead ...
%entrain = - 1/kpphbl .* (dhdt_t+ww_hh) .* (tt_hbl-tt_hh); 
entrain = - 1/kpphbl .* dhdt_t .* (tt_hbl-tt_hh);
entrain_full = - 1/kpphbl(:,:,:,2) .* dhdt_t(:,:,:,2) .* (tt_hbl_full-tt_hh_full) ;

% use SST instead of <T> [Renault et al., 2012]
%	Weakly impact results (~O(.005 degC/day))
%entrain = - dhdt_t .* (squeeze(tt(2:nx1-1,2:ny1-1,1,:,:))-tt_hh);


%-- averaging within the box --
entrain_box = squeeze(sum(sum(entrain .*repmat(rA2_box,[1 1 niter2 nConf]) ,1),2 ) ./ ...
         sum(rA2_box(:)));
entrain_box_full = squeeze(sum(sum(entrain_full .*repmat(rA2_box,[1 1 niter2]) ,1),2 ) ./ ...
         sum(rA2_box(:)));




%-- regroup 2 last terms in vmix --
%vmix_hbl = dkvtdz_hbl - entrain;  
%vmix_hbl_full = dkvtdz_hbl_full - entrain_full;  


return



%--------------------------------------
% 		PLOT
%--------------------------------------
spd = 86400;

%-- remake Fig04 in 4 subplots comparing the 4 terms between experiments --
figure(02)
clf
set(gcf,'position',[50 50 1400 800])
%-- tempe tendency --
subplot(221)
hold on
plot(ddate(1:niter2),zeros(1,niter2),'k')
[pf] = plot((ddate(1:niter2-1)+ddate(2:niter2))./2,...
        dThbl_dt(:,2)*spd,'b','lineW',1.2);
[pc] = plot((ddate(1:niter2-1)+ddate(2:niter2))./2,...
        dThbl_dt(:,1)*spd,'r','lineW',1.2);
grid on
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}]);
xlabel('Time [days of the year]')
ylabel('[^{o}C/day]')
title('Temperature tendency ($\partial_t <T>$)','Interpreter','latex')
h = legend([pf;pc],'AML\_FULL','AML\_CLIM');
%-- qnet --
subplot(222)
hold on
plot(ddate(1:niter2),zeros(1,niter2),'k')
yyaxis right
[p0] = plot(ddate(1:niter2),hbl_avg(1:niter2,2),'color',[.4 .4 .4]);
[p1] = plot(ddate(1:niter2),hbl_avg(1:niter2,1),'color',[.7 .7 .7]);
ylabel('Mixed layer depth [m]')
set(gca,'yLim',[-40 120],'YColor',[.7 .7 .7])
yyaxis left
[pf] = plot(ddate(1:niter2),qnet_box(:,2).*spd,'b','lineW',1.2);
[pc] = plot(ddate(1:niter2),qnet_box(:,1).*spd,'r','lineW',1.2);
grid on
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}]);
xlabel('Time [days of the year]')
ylabel('[^{o}C/day]')
title('Heat fluxes ($\frac{-Qnet}{\rho_0 C_p h}$)','Interpreter','latex')
h = legend([pf;pc;p0;p1],'AML\_FULL','AML\_CLIM','MLD_{FULL}','MLD_{CLIM}');
%-- diff_z --
subplot(223)
hold on
plot(ddate(1:niter2),zeros(1,niter2),'k')
[pf] = plot(ddate(1:niter2),diffz_hbot(:,2).*spd,'b','lineW',1.2);
[pc] = plot(ddate(1:niter2),diffz_hbot(:,1).*spd,'r','lineW',1.2);
grid on
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}]);
xlabel('Time [days of the year]')
ylabel('[^{o}C/day]')
title('Vertical diffusion ($\frac{1}{h} K_z \partial_z T|_{z=h}$)','Interpreter','latex')
h = legend([pf;pc],'AML\_FULL','AML\_CLIM','location','southEast');
%-- residual --
subplot(224)
hold on
plot(ddate(1:niter2),zeros(1,niter2),'k')
[pf] = plot(ddate(1:niter2),res_hbl(:,2).*spd,'b','lineW',1.2);
[pc] = plot(ddate(1:niter2),res_hbl(:,1).*spd,'r','lineW',1.2);
grid on
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}]);
xlabel('Time [days of the year]')
ylabel('[^{o}C/day]')
title('Residual ($\partial_t <T> + \frac{-Qnet}{\rho_0 C_p h}$)','Interpreter','latex')
h = legend([pf;pc],'AML\_FULL','AML\_CLIM','location','southEast');
%- save -
fileN02 = ['heat_bgt_CHEAP-FULL_box1_Rev01.pdf'];
exportfig(figure(02),[dir_fig fileN02],...
    'width',10,'color','rgb','resolution',300);





%%%%%%%%%%%%%%%%%
plot(ddate(1:niter2),qnet_box(:,2).*spd,'b','lineW',1.2);
[pc] = plot(ddate(1:niter2),qnet_box(:,1).*spd,'r','lineW',1.2);
grid on
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}]);
xlabel('Time [days of the year]')
ylabel('[^{o}C/day]')
title('Qnet')
h = legend([pf;pc],'AML\_FULL','AML\_CLIM');





[p1] = plot(ddate(1:niter2),res_hbl(:,iconf).*spd,'color',[.3 .3 .3],'lineW',1.2);
[p2] = plot(ddate(1:niter2),diffz_hbot(:,iconf).*spd,'color',[0 .8 .2],'lineW',1.2);
[p3] = plot(ddate(1:niter2),qnet_box(:,iconf).*spd,'r','lineW',1.2);
[p4] = plot((ddate(1:niter2-1)+ddate(2:niter2))./2,...
        dThbl_dt(:,iconf)*spd,'b','lineW',1.2);
%plot([ddate(32) ddate(32)],[-1 2],'k--')






%-- All terms for each exp --
iconf = 2;
figure(01)
clf
set(gcf,'position',[50 500 800 400])
hold on
%plot([ddate(32) ddate(32)],[-0.2 1.2],'k--')
%plot(ddate(1:niter2),zeros(1,niter2),'k')
[p0] = plot(ddate(1:niter2),diffh_hbl(:,iconf).*spd,'color',[.7 .7 .7]);
[p1] = plot(ddate(1:niter2),advxy_hbl(:,iconf).*spd,'color',[.3 .3 .3]);
[p2] = plot(ddate(1:niter2),advz_hbl(:,iconf).*spd,'k');
[p3] = plot(ddate(1:niter2),res_hbl(:,iconf).*spd,'m');
[p4] = plot(ddate(1:niter2),diffz_hbot(:,iconf).*spd,'g');
[p5] = plot(ddate(1:niter2),entrain_box(:,iconf).*spd,'c');
[p6] = plot(ddate(1:niter2),qnet_box(:,iconf).*spd,'r');
[p7] = plot((ddate(1:niter2-1)+ddate(2:niter2))./2,...
	dTdt_hbl(:,iconf)*spd,'b');
h = legend([p1;p2;p0;p3;p4;p5;p6;p7],...
	'adv_h',...
        'adv_z',...
	'diff_h',...
	'residual',...
        'diff_z',...
	'entrain',...
	'Qnet',...
	'Tendency');


%-- CHEAP-FULL --
iconf = 2;
figure(10)
clf
set(gcf,'position',[50 500 800 400])
hold on
plot(ddate(1:niter2),zeros(1,niter2),'k')
[p1] = plot(ddate(1:niter2),res_hbl(:,iconf).*spd,'color',[.3 .3 .3],'lineW',1.2);
[p2] = plot(ddate(1:niter2),diffz_hbot(:,iconf).*spd,'color',[0 .8 .2],'lineW',1.2);
[p3] = plot(ddate(1:niter2),qnet_box(:,iconf).*spd,'r','lineW',1.2);
[p4] = plot((ddate(1:niter2-1)+ddate(2:niter2))./2,...
        dThbl_dt(:,iconf)*spd,'b','lineW',1.2);
%plot([ddate(32) ddate(32)],[-1 2],'k--')
yyaxis right
[p0]=plot(ddate(1:niter2),hbl_avg(1:niter2,iconf),'color',[.7 .7 .7]);
ylabel('Mixed layer depth [m]')
set(gca,'yLim',[-100 100],'YColor',[.7 .7 .7])
yyaxis left
h = legend([p4;p3;p1;p2;p0],...
	'\partial_t <T>',...
        'Qnet',...
        'residual',...
        'Vert. diff.',...
	'MLD');
set(h,'position',[0.73    0.77    0.1250    0.1787])
grid on
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}],...
      'yLim',[-1 1]);
xlabel('Time [days of the year]')
ylabel('Temperature tendency [^{o}C/day]')
title('AML\_FULL')
%- save -
fileN10 = ['heat_bgt_CHEAP-FULL_box1_ter.pdf'];
exportfig(figure(10),[dir_fig fileN10],...
    'width',6,'color','rgb','resolution',300);



%-- CHEAP-CLIM --
iconf = 1;
figure(20)
clf
set(gcf,'position',[50 500 800 400])
hold on
%plot([ddate(32) ddate(32)],[-1 2],'k--')
plot(ddate(1:niter2),zeros(1,niter2),'k')
%[p03] = plot(ddate(1:niter2),qnet_box_full.*spd,'r','lineS',':','lineW',1.1);
[p1] = plot(ddate(1:niter2),res_hbl(:,iconf).*spd,'color',[.3 .3 .3],'lineW',1.2);
[p2] = plot(ddate(1:niter2),diffz_hbot(:,iconf).*spd,'color',[0 .8 .2],'lineW',1.2);
[p3] = plot(ddate(1:niter2),qnet_box(:,iconf).*spd,'r','lineW',1.2);
[p4] = plot((ddate(1:niter2-1)+ddate(2:niter2))./2,...
        dThbl_dt(:,iconf)*spd,'b','lineW',1.2);
yyaxis right
[p0]=plot(ddate(1:niter2),hbl_avg(1:niter2,iconf),'color',[.7 .7 .7]);
ylabel('Mixed layer depth [m]')
set(gca,'yLim',[-100 100],'YColor',[.7 .7 .7])
yyaxis left
h = legend([p4;p3;p1;p2;p0],...
        '\partial_t <T>',...
        'Qnet',...
        'residual',...
        'Vert. diff.',...
	'MLD');
set(h,'position',[0.73    0.77    0.1250    0.1787])
grid on
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}],...
      'yLim',[-1 1]);
xlabel('Time [days of the year]')
ylabel('Temperature tendency [^{o}C/day]')
title('AML\_CLIM')
%- save -
fileN20 = ['heat_bgt_CHEAP-CLIM_box1_ter.pdf'];
exportfig(figure(20),[dir_fig fileN20],...
    'width',6,'color','rgb','resolution',300);



