
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
global xC yC xG yG rC Depth dxC dyC rA drC xG yG drF
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
xC1 = xC(ii,jj);
yC1 = yC(ii,jj);
xG1 = xG(ii,jj);
yG1 = yG(ii,jj);


%-- time parameters --
years = 1958:1967;
nYr = length(years);
yrIni = 1958;

%- time parameters - 
[dt] = 450;
spy = 86400*365;
[dump] = 5*86400;       %5-d dumps
d_iter = dump/dt;
nDump = 86400*365/dump;
offset = (years(1)-1958)*spy/dt;;
iter = [0:d_iter:(nDump*d_iter)*nYr] + offset;
iter = iter(2:end);
[niter] = length(iter);
date_y = floor((iter*dt/86400-1)/365 ) + 1958;
date_d = (iter*dt/86400);
ddate = date_d/365 + 1958;
niter2 = 2*73;

nDiag1 = 5;             % ocean
nDiag2 = 9;             % cheapaml


%------------------------
% Load pre-extracted data
% dim: 	[nx1,ny1,nr,niter,nDiag1,nConf]
%	[nx1,ny1,niter,nDiag2,nConf]
% ocn diag: t,s,u,v,w
% cheap diag: t2, q2, sh, lh, qnet, emp, u10, v10, precip
%------------------------

%-- cheap diag --
fid = fopen([dir_in 'cheap_diag_box1_1958_67.bin'],'r','b');
diag_cheap = fread(fid,'real*4');
fclose(fid);
diag_cheap = reshape(diag_cheap,[nx1 ny1 niter nDiag2 nConf]);

%-- original atm fields (5-d avg processed) --
fid = fopen([dir_in 'atm_field_box1_1958_67.bin'],'r','b');
atm_fld = fread(fid,'real*4');
fclose(fid);
atm_fld = reshape(atm_fld,[nx1 ny1 niter 2]);	%radsw, radlw

%-- ocn diag --
fid = fopen([dir_in 'ocn_diag_box1_1958_67.bin'],'r','b');
diag_ocn = fread(fid,'real*4');
fclose(fid);
diag_ocn = reshape(diag_ocn,[nx1 ny1 nr niter nDiag1 nConf]);



%-----------------------
% spatial averaging
%-----------------------

rA_box = rA(ii,jj);

%-- Qnet --
tmp = squeeze(diag_cheap(:,:,:,5,:));
qnet = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter nConf]) ,1),2 ) ./ sum(rA_box(:)));

%-- sensH --
tmp = squeeze(diag_cheap(:,:,:,3,:));
sensH = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter nConf]) ,1),2 ) ./ sum(rA_box(:)));
tmp_sh = reshape(tmp(1:nx1-1,1:ny1-1,:,:),[(nx1-1)*(ny1-1) niter nConf]);

%-- latH --
tmp = squeeze(diag_cheap(:,:,:,4,:));
latH = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter nConf]) ,1),2 ) ./ sum(rA_box(:)));

%-- radsw --
tmp = squeeze(atm_fld(:,:,:,1));
radsw = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter]) ,1),2 ) ./ sum(rA_box(:)));

%-- radlw --
tmp = squeeze(atm_fld(:,:,:,2));
radlw = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter]) ,1),2 ) ./ sum(rA_box(:)));

%-- LW emissivity of the surface layer --
xolw0 = qnet + repmat(radsw,[1 nConf]) + repmat(radlw,[1 nConf]) -...
	sensH - latH;

%-- t2 --
tmp = squeeze(diag_cheap(:,:,:,1,:));
t2 = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter nConf]) ,1),2 ) ./ sum(rA_box(:)));
tmp_t2 = reshape(tmp(1:nx1-1,1:ny1-1,:,:),[(nx1-1)*(ny1-1) niter nConf]);

%-- u10 --
tmp = squeeze(diag_cheap(:,:,:,7,:));
tmp_u = tmp;
u10 = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter nConf]) ,1),2 ) ./ sum(rA_box(:)));

%-- v10 --
tmp = squeeze(diag_cheap(:,:,:,8,:));
tmp_v = tmp;
v10 = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter nConf]) ,1),2 ) ./ sum(rA_box(:)));

%-- |u| --
tmp_uv = sqrt( ( (tmp_u(2:nx1,1:ny1-1,:,:)+tmp_u(1:nx1-1,1:ny1-1,:,:))./2) .^2 + ...
        ( (tmp_v(1:nx1-1,2:ny1,:,:)+tmp_v(1:nx1-1,1:ny1-1,:,:))./2) .^2 );
uv = squeeze(sum(sum( ...
        tmp_uv.*repmat(rA_box(1:nx1-1,1:ny1-1),[1 1 niter nConf]) ,1),2 ) ./ ...
        sum(sum(rA_box(1:nx1-1,1:ny1-1),1),2));
tmp_uv = reshape(tmp_uv,[(nx1-1)*(ny1-1) niter nConf]);

%-- sst --
tmp = squeeze(diag_ocn(:,:,1,:,1,:));
sst = squeeze(sum(sum( tmp.*repmat(rA_box,[1 1 niter nConf]) ,1),2 ) ./ sum(rA_box(:)));
tmp_sst = reshape(tmp(1:nx1-1,1:ny1-1,:,:),[(nx1-1)*(ny1-1) niter nConf]);

%-- trend of SST --
trend_sst = zeros(2,nConf);
for iconf = 1:nConf
 trend_sst(:,iconf) = polyfit(ddate(nDump+1:end)-1958,sst(nDump+1:end,iconf)',1);
end

%--------------------------------
%		PLOT
%--------------------------------


figure(10)
clf
set(gcf,'position',[50 500 1600 800])
%-- fluxes time series AML --
subplot(2,3,1:2)
hold on
[p0] = plot(ddate(1:niter2),zeros(1,niter2),'k');
[p11] = plot(ddate(1:niter2),qnet(1:niter2,1),'color',[0 0 0 .3],'lineW',1.2);
[p12] = plot(ddate(1:niter2),qnet(1:niter2,2),'color',[0 0 0 .8],'lineW',1.2);
[p31] = plot(ddate(1:niter2),latH(1:niter2,1)+sensH(1:niter2,1),...
	'color',[0 .8 .2],'lineW',1.2);
[p32] = plot(ddate(1:niter2),latH(1:niter2,2)+sensH(1:niter2,2),'b','lineW',1.2);
grid on
h=legend([p32;p12;p31;p11],...
	'Lat. + Sens. HF','Qnet (AML\_FULL)',...
	'Lat. + Sens. HF','Qnet (AML\_CLIM)',...
	'Orientation','Horizontal');
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}],...
      'yLim',[-200 400]);
ylabel('[W m^{-2}]')
%xlabel('Time [days of the year]')
title('AML\_FULL and AML\_CLIM')
%-- fluxes time series FORC --
subplot(2,3,4:5)
hold on
[p0] = plot(ddate(1:niter2),zeros(1,niter2),'k');
[p11] = plot(ddate(1:niter2),qnet(1:niter2,3),'color',[0 0 0 .3],'lineW',1.2);
[p12] = plot(ddate(1:niter2),qnet(1:niter2,4),'color',[0 0 0 .8],'lineW',1.2);
[p31] = plot(ddate(1:niter2),latH(1:niter2,3)+sensH(1:niter2,3),...
	'color',[0 .8 .2],'lineW',1.2);
[p32] = plot(ddate(1:niter2),latH(1:niter2,4)+sensH(1:niter2,4),'b','lineW',1.2);
grid on
h=legend([p32;p12;p31;p11],...
        'Lat. + Sens. HF','Qnet (FORC\_FULL)',...
        'Lat. + Sens. HF','Qnet (FORC\_CLIM)',...
	'Orientation','Horizontal');
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}],...
      'yLim',[-200 400]);
ylabel('[W m^{-2}]')
xlabel('Time [days of the year]')
title('FORC\_FULL and FORC\_CLIM')
%-- scatterplots --
ij = 200;
%- sensH & |u| -
subplot(233)
hold on
plot([0 14],[0 0],'color',[.5 .5 .5])
%[p1]=plot(uv(:,2),sensH(:,2),'k.');
%[p2]=plot(uv(:,1),sensH(:,1),'r.');
%[p3]=plot(uv(:,3),sensH(:,3),'b.');
[p1]=plot(tmp_uv(ij,:,2),tmp_sh(ij,:,2),'k.');
[p2]=plot(tmp_uv(ij,:,1),tmp_sh(ij,:,1),'r.');
[p3]=plot(tmp_uv(ij,:,3),tmp_sh(ij,:,3),'b.');
grid on
legend([p1;p2;p3],'AML\_FULL','AML\_CLIM','FORC\_CLIM','location','northWest')
xlabel('|u| [m.s^{-1}]')
ylabel('Sens. HF [W.m^{-2}]')
set(gca,'xLim',[0 14],'yLim',[-30 60],'position',[0.67    0.5838    0.2134    0.3412])
%- sensH & sst-ta -
subplot(236)
hold on
plot([-2 5],[0 0],'color',[.5 .5 .5])
plot([0 0],[-30 60],'color',[.5 .5 .5])
%[p1]=plot(sst(:,2)-t2(:,2),sensH(:,2),'k.');
%[p2]=plot(sst(:,1)-t2(:,1),sensH(:,1),'r.');
%[p3]=plot(sst(:,3)-t2(:,3),sensH(:,3),'b.');
[p1]=plot(tmp_sst(ij,:,2)-tmp_t2(ij,:,2),tmp_sh(ij,:,2),'k.');
[p2]=plot(tmp_sst(ij,:,1)-tmp_t2(ij,:,1),tmp_sh(ij,:,1),'r.');
[p3]=plot(tmp_sst(ij,:,3)-tmp_t2(ij,:,3),tmp_sh(ij,:,3),'b.');
grid on
legend([p1;p2;p3],'AML\_FULL','AML\_CLIM','FORC\_CLIM','location','northWest')
xlabel('SST - T_a [^oC]')
ylabel('Sens. HF [W.m^{-2}]')
set(gca,'xLim',[-2 5],'yLim',[-30 60],'position',[0.67    0.1100    0.2134    0.3412])
%-- save --
fileN10 = ['heat_fluxes_fig03.pdf'];
exportfig(figure(10),[dir_fig fileN10],...
    'width',12,'color','rgb','resolution',300);



%------------------------------
% Rev#2, comments 7.
%------------------------------
tmp = squeeze(diag_cheap(1:nx1-1,1:ny1-1,:,3,:));	% -1 is to be consistent with |u|
tmp = reshape(tmp,[(nx1-1)*(ny1-1) niter nConf]);
tmp_uv = reshape(tmp_uv,[(nx1-1)*(ny1-1) niter nConf]);
%-- 'ensemble' variance --
tmp_xyvar = squeeze(var(tmp,0,1));
tmp_tvar = squeeze(var(mean(tmp,1),0,2));

figure(100)
clf
subplot()


%-- test that scatter plots look the same everywhere --
r = randi([1 20*20],1,20*20);
figure(200)
for iii = 1:20*20
  clf
  hold on
  [p1]=plot(tmp_uv(r(iii),:,2),tmp_sh(r(iii),:,2),'k.');
  [p2]=plot(tmp_uv(r(iii),:,1),tmp_sh(r(iii),:,1),'r.');
  [p3]=plot(tmp_uv(r(iii),:,3),tmp_sh(r(iii),:,3),'b.');
  grid on
  legend([p1;p2;p3],'AML\_FULL','AML\_CLIM','FORC\_CLIM','location','northWest')
  xlabel('|u| [m.s^{-1}]')
  ylabel('Sens. HF [W.m^{-2}]')
  set(gca,'xLim',[0 14],'yLim',[-33 93])
  pause(0.1)
end


%-----------------------------
% Additional plots 
%-----------------------------


tit_conf = {'CHEAP-CLIM','CHEAP-FULL'}
for iConf = 1:2
  figure(iConf)
  set(gcf,'position',[40 400 1000 400])
  clf
  hold on
  [p0] = plot(ddate(1:73),zeros(1,73),'k');
  [p1] = plot(ddate(1:73),qnet(1:73,iConf),'k','lineW',1.2);
  [p2] = plot(ddate(1:73),-radsw(1:73),'b');
  [p3] = plot(ddate(1:73),xolw0(1:73,iConf),'color',[.7 .7 .7]);
  [p4] = plot(ddate(1:73),-radlw(1:73),'c');
  [p5] = plot(ddate(1:73),sensH(1:73,iConf),'g');
  [p6] = plot(ddate(1:73),latH(1:73,iConf),'m');
  title(['Total heat flux and components (+=up)  --  ' tit_conf{iConf}])
  grid on
  legend([p1;p2;p3;p4;p5;p6],'Qnet','-radsw','xolw0','-radlw','sensH','LatH')
  set(gca,'xTick',[1958 1958+[30:30:330]/365 1959],...
        'xTickLabel',[{'01/01/1958'},{'30'},{'60'},{'90'},{'120'},{'150'},{'180'},...
        {'210'},{'240'},{'270'},{'300'},{'330'},{'01/10/1959'}]);
  ylabel('[W m^{-2}]')
  xlabel('Time [days of the year 1958]')
  fileN = ['Qnet_components_box1_' tit_conf{iConf} '.pdf'];
  exportfig(figure(iConf),[dir_fig fileN],...
      'width',6,'color','rgb','resolution',300);
end %for iConf


figure(10)
clf
set(gcf,'position',[50 500 800 400])
hold on
plot([ddate(32) ddate(32)],[-400 500],'k--')
[p0] = plot(ddate(1:niter2),zeros(1,niter2),'k');
[p1] = plot(ddate(1:niter2),-radsw(1:niter2),'color',[.7 .7 .7]);
[p2] = plot(ddate(1:niter2),-radlw(1:niter2),'color',[.3 .3 .3]);
[p31] = plot(ddate(1:niter2),xolw0(1:niter2,1),'r');
[p32] = plot(ddate(1:niter2),xolw0(1:niter2,2),'b');
[p41] = plot(ddate(1:niter2),qnet(1:niter2,1),'r','lineW',1.2);
[p42] = plot(ddate(1:niter2),qnet(1:niter2,2),'b','lineW',1.2);
grid on
h=legend([p42;p41;p32;p1;p2],'Net HF  --  CHEAP-FULL','Net HF -- CHEAP-CLIM',...
	'Upward longwave HF (\propto T^4)',...
	'Solar shortwave HF','Downward longwave HF');
set(h,'position',[.43 .75 .237 .2])
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
	{'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
	{'\color{gray}240'},{'\color{gray}300'},{'1960'}],...
      'yLim',[-400 500]);
ylabel('[W m^{-2}]')
xlabel('Time [days of the year 1958]')
fileN10 = ['Qnet_components1_box1.pdf'];
exportfig(figure(10),[dir_fig fileN10],...
    'width',6,'color','rgb','resolution',300);




%-- only Qnet --
figure(12)
clf
set(gcf,'position',[50 500 1200 400])
hold on
plot([ddate(32) ddate(32)],[-200 300],'k--')
[p0] = plot(ddate(1:niter2),zeros(1,niter2),'k');
[p13] = plot(ddate(1:niter2),qnet(1:niter2,3),'color',[.7 .7 .7],'lineW',1.2);
[p14] = plot(ddate(1:niter2),qnet(1:niter2,4),'color',[.3 .3 .3],'lineW',1.2);
[p11] = plot(ddate(1:niter2),qnet(1:niter2,1),'r','lineW',1.2);
[p12] = plot(ddate(1:niter2),qnet(1:niter2,2),'b','lineW',1.2);
grid on
h=legend([p12;p11],'Q_{net} --  CHEAP-FULL','Q_{net} -- CHEAP-CLIM');
set(h,'position',[.62 .65 .13 .11],'box','off')
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}],...
      'yLim',[-200 300]);
ylabel('[W m^{-2}]')
xlabel('Time [days of the year]')
%xlabel('Time [year]')
fileN12 = ['Qnet_box1.pdf'];
exportfig(figure(12),[dir_fig fileN12],...
    'width',6,'color','rgb','resolution',300);


%-- longwave upward question --
figure(13)
clf
set(gcf,'position',[50 500 1200 400])
hold on
plot([ddate(32) ddate(32)],[390 500],'k--')
[p0] = plot(ddate(1:niter2),zeros(1,niter2),'k');
%[p13] = plot(ddate(1:niter2),xolw0(1:niter2,3),'color',[.7 .7 .7],'lineW',1.2);
%[p14] = plot(ddate(1:niter2),xolw0(1:niter2,4),'color',[.3 .3 .3],'lineW',1.2);
[p11] = plot(ddate(1:niter2),xolw0(1:niter2,1),'r','lineW',1.2);
[p12] = plot(ddate(1:niter2),xolw0(1:niter2,2),'b','lineW',1.2);
grid on
h=legend([p12;p11],'CHEAP-FULL','CHEAP-CLIM');
set(h,'position',[.5 .7 .13 .11],'box','off')
set(gca,'xTick',[1958 1958+[60:60:330]/365 1959 1959+[60:60:330]/365 1960],...
      'xTickLabel',[{'1958'},{'\color{gray}60'},{'\color{gray}120'},...
        {'\color{gray}180'},{'\color{gray}240'},{'\color{gray}300'},{'1959'},...
      {'\color{gray}60'},{'\color{gray}120'},{'\color{gray}180'},...
        {'\color{gray}240'},{'\color{gray}300'},{'1960'}],...
      'yLim',[390 500]);
ylabel('Upward LW radiation (\propto SST^4) [W m^{-2}]')
xlabel('Time [days of the year]')
%xlabel('Time [year]')
fileN13 = ['upward_LWrad_box1.pdf'];
exportfig(figure(13),[dir_fig fileN13],...
    'width',6,'color','rgb','resolution',300);


