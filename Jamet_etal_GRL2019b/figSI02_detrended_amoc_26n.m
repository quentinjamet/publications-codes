% DESCRIPTION :
% Compare model amoc (for the orar config) to obs at RAPID location
% RAPID data are from
%
% correlation RAPID and ensemble mean: r=0.8
% mean correlation between  RAPID and ensembles: r=0.7



clear all; close all;


%-- directories --
config = 'orar';
dir_in = ['/tank/chaocean/qjamet/RUNS/data_chao12/' config '/'];
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'; 
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/';
dir_rapid = '/tank/chaocean/rapid26N/';

load('MyCmap_redBlue.mat')


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
global yG rF drF dyC hS
loadGRD(dir_grd)
[nx,ny] = size(yG);
[nr] = length(rF)-1;
msk_lnd = squeeze(sum(hS,1));
msk_lnd(msk_lnd == 0) = NaN;
msk_lnd(~isnan(msk_lnd)) = 1;
%- select 26.5N, -1200 m depth -
[jj_26N] = find( abs(yG(1,:)-26.5) == min(abs(yG(1,:)-26.5)) );
zdepth = 1200;
[kdepth] = find(abs(rF+zdepth) == min(abs(rF+zdepth)));



%-- runs parameters --
nb_memb = 12;
nDump = 73;
spy = 86400*365;
%- get time dimension for ensemble -
data = dir([dir_in 'globalEta_' config '_ensemble.bin']);
nYr_ens = data.bytes/accu2/nDump/nb_memb;
time_ens = 1963 + [1/nDump:1/nDump:nYr_ens];
[nt_ens] = length(time_ens);

%-- get time corresponding to rapid period (starts at 04/01/2004) --
[tt_ens] = find(time_ens > 2004+92/365);
%remove the first time record
tt_ens = tt_ens(2:end);


%---------------------------
% Load RAPID array AMOC data
%---------------------------

% April 1st 2004 == 92th days of the year 2004
time_rapid =  ncread([dir_rapid 'moc_vertical.nc'],'time')/365 + 2004 + 91/365;
rF_rapid = -ncread([dir_rapid 'moc_vertical.nc'],'depth');
[nr_rapid] = length(rF_rapid);
amoc_rapid = ncread([dir_rapid 'moc_vertical.nc'],'stream_function_mar');
amoc_overt_transp = ncread([dir_rapid 'moc_transports.nc'],'moc_mar_hc10');

%-- get time from rapid < 12/31/2012 --
[tt_rapid] = find(time_rapid <= 2013);
% remove 7 first time record for computing 5-d mean
tt_rapid = tt_rapid(8:end);
[nt_rapid] = length(tt_rapid);

%-- keep rapid amoc data for the considered period --
amoc_rapid = amoc_rapid(tt_rapid,:);
amoc_overt_transp = amoc_overt_transp(tt_rapid);

%-- compute 5-d avg (rapid sample are every 12 hr) --
amoc_rapid_5d = reshape(amoc_rapid,[10 nt_rapid/10 nr_rapid]);
amoc_rapid_5d = squeeze(mean(amoc_rapid_5d,1));
time_rapid_5d = time_rapid(tt_rapid(10:10:end));
[nt] = length(time_rapid_5d);

%-- vertical interpolation on chao12 grid --
amoc_rapid_5d_46z = zeros(nt,nr);
for it = 1:nt
  amoc_rapid_5d_46z(it,:) = interp1(rF_rapid,amoc_rapid_5d(it,:),rF(1:nr));
end % for it

%-- time mean vertical profil --
amocz_rapid_tm = mean(amoc_rapid_5d_46z,1);

% ----------------------------------
% Load pre-computed AMOC(y,z,t,memb)
% ----------------------------------
fprintf('-- Extract AMOC(y,z,t,memb) for config: %s --\n',config)

%-- load non-processed data for time mean --
fid = fopen([dir_in 'MOCyzt_' config '_ensemble.bin'],'r',ieee);
moc_ens = fread(fid,accu);
fclose(fid);
moc_ens = reshape(moc_ens,[ny*nr nDump nb_memb nYr_ens]);
moc_ens = reshape(permute(moc_ens,[1 2 4 3]),[ny nr nt_ens nb_memb]);
moc_ens_yr = squeeze(mean(reshape(moc_ens,[ny nr nDump nYr_ens nb_memb]),3));


%- select time period consistent with RAPID data, and compute time mean -
moc_ens_26n_tm = squeeze( mean(moc_ens(jj_26N,:,tt_ens,:),3) );

%- select amoc ano at 26N, 1200m for the yealy mean -
moc_ens_yr_26n = squeeze(moc_ens_yr(jj_26N,kdepth,:,:));
moc_ano_yr_26n = moc_ens_yr_26n - mean(mean(moc_ens_yr_26n,2),1);

%- select raw amoc at 26n, 1200m and high pass filter -
span=50*nDump;
tmpmoc = squeeze(moc_ens(jj_26N,kdepth,:,:));
moc_hpf = zeros(nt_ens,nb_memb);
for imem = 1:nb_memb
 moc_hpf(:,imem) = smooth(time_ens,tmpmoc(:,imem),span,'loess');
end


%--------------------
%	PLOT
%--------------------


%-- mean vertical profil --
figure(10)
clf
set(gcf,'position',[40 400 600 800])
%- 
hold on
plot([0 0],[0 rF(nr+1)]./1000,'k--');
[p00] = plot(moc_ens_26n_tm,rF(1:nr)./1000,'color',[.6 .6 .6]);
[p01] = plot([mean(moc_ens_26n_tm,2);0],rF./1000,'k','lineW',2);
[p02] = plot([amocz_rapid_tm 0],rF./1000,'r','lineW',2);
grid on
set(gca,'yLim',[rF(nr+1) 0]./1000,'xLim',[-10 20]);
h = legend([p00(1);p01;p02],'members','ensemble mean','RAPID','location','southEast');
xlabel('AMOC [Sv]')
ylabel('Depth [km]')
%- save -
%fileN10 = ['amoc_26n_tmean_zprof_' config];
%exportfig(figure(10),[dir_fig fileN10 '.png'],...
%      'width',4,'color','rgb','resolution',300);
%exportfig(figure(10),[dir_fig fileN10 '.pdf'],...
%      'width',4,'color','rgb','resolution',300);



%-- yearly mean time series at 26N, 1200m --
figure(20)
clf
set(gcf,'position',[40 400 1500 500])
%-
hold on
plot([1963 2013],[0 0],'k--')
%[p_hpf] = plot(time_ens,moc_hpf,'b');
[p_hpf] = plot(time_ens,moc_hpf-repmat(mean(moc_hpf,1),[nt_ens 1]),'color',[0 0 1 .5]);
[p00] = plot([1963.5:2012.5],moc_ano_yr_26n,'color',[.6 .6 .6]);
[p01] = plot([1963.5:2012.5],mean(moc_ano_yr_26n,2),'k','lineW',2);
grid on
set(gca,'xLim',[1963 2013],'yLim',[-4 4]);
h = legend([p00(1);p01;p_hpf(1)],'members','ensemble mean','LOESS',...
	'location','south','Orientation','Horizontal');
xlabel('Time [yr]')
ylabel('AMOC anomalies [Sv]')
%- save -
fileN20 = ['amoc_ano_26n_1200m_yrm_' config];
exportfig(figure(20),[dir_fig fileN20 '.png'],...
      'width',9,'color','rgb','resolution',300);
exportfig(figure(20),[dir_fig fileN20 '.pdf'],...
      'width',9,'color','rgb','resolution',300);

