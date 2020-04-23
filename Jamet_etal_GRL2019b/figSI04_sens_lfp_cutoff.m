% DESCRIPTION  
% Map of intrinsic / forced variability of the yearly and detrended AMOC
%


clear all; close all;


%-- directories --
config = 'orar';
dir_in = ['/tank/chaocean/qjamet/RUNS/data_chao12/' config '/'];
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'; 
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/';


load('MyCmap_serazin2.mat')

%-- options --
ieee = 'b';
accu = 'real*4';
if strcmp(accu,'real*4')
  accu2 = 4;
else
  accu2 = 8;
end

%-- grid --
% AMOC is computed at v-pts, i.e. (xC,yG)
global yG rF drF dyC
loadGRD(dir_grd)
[nx,ny] = size(yG);
[nr] = length(rF)-1;
% get rapid location 
jj = find( abs(yG(1,:)-26.5) == min(abs(yG(1,:)-26.5)) );

%-- runs parameters --
nb_memb = 12;
nDump = 73;
spy = 86400*365;
%- get time dimension for ensemble -
data = dir([dir_in 'globalEta_' config '_ensemble.bin']);
nYr_ens = data.bytes/accu2/nDump/nb_memb;
time_ens = 1963 + [1/nDump:1/nDump:nYr_ens];
[nt] = length(time_ens);

ttime = time_ens(1:nDump:end)+0.5;

% ----------------------------------
% Load original AMOC(y,z,t,memb)
% ----------------------------------
% Dimensions of files are [ny*nr nDump nb_memb nYear]

imem = 1;

fprintf('-- Loading AMOC files for config %s --\n',config)
fid = fopen([dir_in 'MOCyzt_' config '_ensemble.bin'],'r',ieee);
tmpmoc = fread(fid,accu);
fclose(fid);
tmpmoc = reshape(tmpmoc,[ny*nr nDump nb_memb nYr_ens]);
tmpmoc = reshape(permute(tmpmoc,[1 2 4 3]),[ny nr nDump*nYr_ens nb_memb]);
%- compute time mean for imem -
tmpmoc_m = mean(tmpmoc(:,:,:,imem),3);
clear tmpmoc


fid = fopen([dir_in 'MOCyzt_' config '_ensemble_detrend.bin'],'r',ieee);
moc_yzt = fread(fid,accu);
fclose(fid);
moc_yzt = reshape(moc_yzt,[ny nr nDump*nYr_ens nb_memb]);

%- remove seasonal cycle -
moc_seasonal = mean(reshape(mean(moc_yzt,4),[ny nr nDump nYr_ens]),4);
moc_yzt = moc_yzt - ...
	reshape(repmat(moc_seasonal,[1 1 1 nYr_ens nb_memb]),...
	[ny nr nDump*nYr_ens nb_memb]);


%------------------------------
% sensitivity to time filtering
%------------------------------

fs = 1/5;             %[day-1]
norder = 73*2;

%-- 5-d avg raw data --
%- define forced and intrinsic signals -
moc_yzt_f = mean(moc_yzt(:,:,nDump+1:nt-nDump,:),4);
moc_yzt_i = moc_yzt(:,:,nDump+1:nt-nDump,:) - ...
	repmat(moc_yzt_f,[1 1 1 nb_memb]);
%- compute variance -
A2_f = var(moc_yzt_f,0,3);
A2_i = mean( var(moc_yzt_i,0,4),3 );
A2_tot = A2_i + A2_f;
%- compute ratio -
ratio = A2_i ./ A2_tot;


figure(10)
clf
set(gcf,'position',[40 200 770 930])
subplot(411)
[c1,h1] = contourf(yG(1,:),rF(1:nr)./1000,ratio',[0:0.1:1.1]);
caxis([0 1])
set(h1,'lineS','none')
colormap(mycmap)
hc = colorbar;
hc.Position = [.91 .25 .01 .5];
hc.FontSize = 10;
hc.Label.String = '\sigma^2_I / \sigma^2_T';
hold on
plot([yG(1,jj) yG(1,jj)],[rF(end-1)/1000 0],'k--','lineW',1.2)
plot([0 0],[rF(end-1)/1000 0],'lineS','--','color',[.4 .4 .4])
title('5-d averaged data')

%-- loop over different filters --
fc = [1/30 1/120 1/182.5];
nfilt = length(fc);
tit_str = {'30-days ';'120-days ';'1/2-year '};

moc_yzt = reshape(permute(moc_yzt,[1 2 4 3]),[ny*nr*nb_memb nt]);
for ifilt = 1:nfilt
 ifilt
 moc_yzt_lpf = nan(ny*nr*nb_memb,nt-2*nDump);
 %- filter -
 Wn = (2/fs)*fc(ifilt);
 b = fir1(norder,Wn,'low',chebwin(norder+1,30));
 for jk = 1:ny*nr*nb_memb
   tmp = filtfilt(b,1,moc_yzt(jk,:));
   moc_yzt_lpf(jk,:) = tmp(nDump+1:nt-nDump);
 end % for jk
 moc_yzt_lpf = permute(reshape(moc_yzt_lpf,[ny nr nb_memb nt-2*nDump]),[1 2 4 3]);

 %- define forced and intrinsic signals -
 moc_yzt_f = mean(moc_yzt_lpf,4);
 moc_yzt_i = moc_yzt_lpf - repmat(moc_yzt_f,[1 1 1 nb_memb]);
 %- compute variance -
 A2_f = var(moc_yzt_f,0,3);
 A2_i = mean( var(moc_yzt_i,0,4),3 );
 A2_tot = A2_i + A2_f;
 %- compute ratio -
 ratio = A2_i ./ A2_tot;
 

 %- plot -
 if ifilt == 4 
  figure(20)
  [c1,h1] = contourf(yG(1,:),rF(1:nr)./1000,ratio',[0:0.1:1.1]);
  caxis([0 1])
  set(h1,'lineS','none')
  hold on
  plot([yG(1,jj) yG(1,jj)],[rF(end-1)/1000 0],'k--','lineW',1.2)
  plot([0 0],[rF(end-1)/1000 0],'lineS','--','color',[.4 .4 .4])
  title([tit_str{ifilt} 'low-pass filtered data'])

 else
  figure(10)
  subplot(4,1,ifilt+1)
  [c1,h1] = contourf(yG(1,:),rF(1:nr)./1000,ratio',[0:0.1:1.1]);
  caxis([0 1])
  set(h1,'lineS','none')
  hold on
  plot([yG(1,jj) yG(1,jj)],[rF(end-1)/1000 0],'k--','lineW',1.2)
  plot([0 0],[rF(end-1)/1000 0],'lineS','--','color',[.4 .4 .4])
  title([tit_str{ifilt} 'low-pass filtered data'])

  %- plot options -
  if ifilt == 3
   xlabel('Latitude')
  end
 end

 clear moc_yzt_lpf

end % for ifilt
moc_yzt = permute(reshape(moc_yzt,[ny nr nb_memb nt]),[1 2 4 3]);

return
fileN10 = 'amoc_detrend_noSeasCycle_intrinsic_to_total_ratio_time_filt_map_48yr';
exportfig(figure(10),[dir_fig fileN10 '.pdf'],...
      'width',6,'color','rgb','resolution',300);
exportfig(figure(10),[dir_fig fileN10 '.png'],...
      'width',6,'color','rgb','resolution',300);



%------------------------------------------------------
% time filtering sensitivity at different location
%-----------------------------------------------------

%-- rapid location at 1200m (max of mean amoc in simulation) --
jj1 = find( abs(yG(1,:)-26.5) == min(abs(yG(1,:)-26.5)) );
kk1 = find( abs(rF+1200) == min(abs(rF+1200)) );
moc_rapid = squeeze(moc_yzt(jj1,kk1,:,:));

%-- gulf stream separation (38N) and 1100m (max of mean amoc in simulation) --
jj2 = find( abs(yG(1,:)-38) == min(abs(yG(1,:)-38)) );
kk2 = find( abs(rF+1100) == min(abs(rF+1100)) );
moc_gs = squeeze(moc_yzt(jj2,kk2,:,:));

%-- maxi/min of the time mean AMOC --
[jj3,kk3] = find(tmpmoc_m == max(tmpmoc_m(:)));
mocmax = squeeze(moc_yzt(jj3,kk3,:,:));

tmp = tmpmoc_m(:,35:end);		%bottom cell, below 3000 m
[jj4,kk4] = find(tmpmoc_m == min(tmp(:)));
mocmin = squeeze(moc_yzt(jj4,kk4,:,:));


%-- apply different time filtering --
fc = 1./[30:30:10*365];
nfc = length(fc);

ratio = zeros(2,nfc);
A2_f = zeros(2,nfc);
A2_i = zeros(2,nfc);
A2_tot = zeros(2,nfc);
for iloc = 1:2
 if iloc == 1
   tmp_moc = moc_rapid;
 elseif iloc == 2
   tmp_moc = moc_gs;
 elseif iloc == 3
   tmp_moc = mocmax;
 elseif iloc == 4
   tmp_moc = mocmin;
 end
% %- raw 5-d avg output -
% tmp_intrin = tmp_moc(:,imem) - mean(tmp_moc,2);
% tmp_forced = mean(tmp_moc,2);
% ratio(iloc,1) = std(tmp_intrin,0,1) ./ std(tmp_forced,0,1);

 for ifc = 1:nfc
  Wn = (2/fs)*fc(ifc);
  b = fir1(norder,Wn,'low',chebwin(norder+1,30));
  tmp_moc_lpf = zeros(nt,nb_memb);
  for imem2 = 1:nb_memb
   tmp_moc_lpf(:,imem2) = filtfilt(b,1,tmp_moc(:,imem2));
  end % for imem
  moc_f = mean(tmp_moc_lpf(nDump+1:nt-nDump,:),2);
  moc_i = tmp_moc_lpf(nDump+1:nt-nDump,:) - repmat(moc_f,[1 nb_memb]);

  %- compute variance -
  A2_f(iloc,ifc) = var(moc_f,0,1);
  A2_i(iloc,ifc) = mean( var(moc_i,0,2),1 );
  A2_tot(iloc,ifc) = A2_i(iloc,ifc) + A2_f(iloc,ifc);
  %- compute ratio -
  ratio(iloc,ifc) = A2_i(iloc,ifc) ./ A2_tot(iloc,ifc);

 end % for ifc
end % for iloc



figure(20)
clf
set(gcf,'position',[40 200 800 300])
hold on
[p1] = plot(1./fc./365,ratio(1,:),'r.-');
[p2] = plot(1./fc./365,ratio(2,:),'b.-');
%[p3] = plot(1./[1/5 fc]./365,ratio(3,:),'lineS','-.','color',[0 .6 .6]);
%[p4] = plot(1./[1/5 fc]./365,ratio(4,:),'lineS','-.','color',[0 .6 0]);
hl = legend([p1;p2],'RAPID (26N, -1200m)','GS (38N, -1100m)');
%	'max(<AMOC>)','min(<AMOC>)','location','NorthEast');
grid on
xlabel('Low-pass filter cut-off period [yr]')
ylabel('\sigma^{2}_I / \sigma^{2}_T')
%- save -
fileN20 = 'amoc_detrend_noSeasCycle_intrinsic_to_total_ratio_time_filt_RAPID_GS';
exportfig(figure(20),[dir_fig fileN20 '.pdf'],...
      'width',6,'color','rgb','resolution',300);
exportfig(figure(20),[dir_fig fileN20 '.png'],...
      'width',6,'color','rgb','resolution',300);




%==========================================================
% intrinsic std + forced std > total std
% compute the intrinsic/forced ratio as in Leroux et al, JC2018
%	- forced = time-variance of the ensemble mean
%	- intrinsic = time-mean of the ensemble-variance
%=========================================================i
clear all; close all;

%-- directories --
config = 'orar';
dir_in = ['/tank/chaocean/qjamet/RUNS/data_chao12/' config '/'];
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/';
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/';


load('MyCmap_serazin2.mat')

%-- options --
ieee = 'b';
accu = 'real*4';
if strcmp(accu,'real*4')
  accu2 = 4;
else
  accu2 = 8;
end

%-- grid --
% AMOC is computed at v-pts, i.e. (xC,yG)
global yG rF drF dyC
loadGRD(dir_grd)
[nx,ny] = size(yG);
[nr] = length(rF)-1;
% get rapid location
jj = find( abs(yG(1,:)-26.5) == min(abs(yG(1,:)-26.5)) );

%-- runs parameters --
nb_memb = 12;
nDump = 73;
spy = 86400*365;
%- get time dimension for ensemble -
data = dir([dir_in 'globalEta_' config '_ensemble.bin']);
nYr_ens = data.bytes/accu2/nDump/nb_memb;
time_ens = 1963 + [1/nDump:1/nDump:nYr_ens];
[nt] = length(time_ens);

ttime = time_ens(1:nDump:end)+0.5;


%-------------------------------
% load processed amoc
%-------------------------------
imem = 1;

%-- load detrended, seasonally removed low-pass filter data --
fid = fopen([dir_in 'MOCyzt_' config '_ensemble_detrend_1ylpf.bin'],'r',ieee);
moc_yzt_lpf = fread(fid,accu);
fclose(fid);
moc_yzt_lpf = reshape(moc_yzt_lpf,[ny nr nDump*nYr_ens nb_memb]);
moc_yzt_lpf_f = mean(moc_yzt_lpf,4);
moc_yzt_lpf_i = moc_yzt_lpf - repmat(moc_yzt_lpf_f,[1 1 1 nb_memb]);

%-- following Leroux et al., JC 2018 for Intrinsic/Forced ratio --
%- total variability -
sigma2_tot = var(moc_yzt_lpf,0,3);
%- forced variance -
sigma2_forced = var(moc_yzt_lpf_f,0,3);
%- intrinsic variance -
sigma2_intrin = var(moc_yzt_lpf_i,0,4);

%- ratio Intrinsic/Forced -
% made with std
ratio_IvsF = sqrt( mean(sigma2_intrin,3) ) ./ sqrt(sigma2_forced);

%- ratio Intrinsic/Tot -
% made with variance
ratio_IvsT = mean(sigma2_intrin,3) ./ mean(sigma2_tot,4);


