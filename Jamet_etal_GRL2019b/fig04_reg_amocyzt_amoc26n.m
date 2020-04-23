% DESCRIPTION :
% Look a the AMOC(y,z,t) regressed on the AMOC time series at 26.5N and -1200 m depth 
% Notes: 
%	- made with processes data (high- and lo-pass filtered)
%	- made for forced and intrinsic signals separatly
%	- First and last years are discard due to detrending effects


clear all; close all;


%-- directories --
config = 'orar';
dir_in = ['/tank/chaocean/qjamet/RUNS/data_chao12/' config '/'];
dir_grd = '/tank/chaocean/grid_chaO/gridMIT_update1/'; 
dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/';

load('MyCmap_redBlue.mat')
load('MyColorMapGray')


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

%-- runs parameters --
nb_memb = 24;
nDump = 73;
spy = 86400*365;
%- get time dimension for ensemble -
nYr_ens = 50;
time_ens = 1963 + [1/nDump:1/nDump:nYr_ens];
nt = length(time_ens)-2*nDump;


% --------------------------------------------------------------
% Load pre-computed (extr_reg_amoc_amoc26n.m) regression of the 
% 3-D AMOC(y,z,t) onto the time series of the maximum mean AMOC at 26.5N
% --------------------------------------------------------------

%load([dir_in 'AMOC_processed_' config '_reg_moc26n.mat']);
load([dir_in 'AMOC_processed_orar_reg_moc26n_24memb.mat']);

%-- construct forced and intrinsic regression and associated significance --
thres = 0.05;

%- forced -
reg_f = reg_moc(:,:,nb_memb+1);
jk_sign = find(sig_moc(:,:,nb_memb+1) <= thres);
jk_nosign = find(sig_moc(:,:,nb_memb+1) > thres);
reg_f_sign = nan(ny,nr);
reg_f_sign(jk_sign) = reg_f(jk_sign);
reg_f_nosign = nan(ny,nr);
reg_f_nosign(jk_nosign) = reg_f(jk_nosign);

%- intrinsic -
reg_i = mean(reg_moc(:,:,1:nb_memb),3);
tmp_sign = sig_moc(:,:,1:nb_memb);
tmp_sign(tmp_sign > thres) = 99999;
tmp_sign(tmp_sign <= thres) = 1;
tmp_sign(tmp_sign == 99999) = 0;
tmp_sign = mean(tmp_sign,3);
jk_sign = find(tmp_sign == 1);
jk_nosign = find(tmp_sign < 1);
reg_i_sign = nan(ny,nr);
reg_i_sign(jk_sign) = reg_i(jk_sign);
reg_i_nosign = nan(ny,nr);
reg_i_nosign(jk_nosign) = reg_i(jk_nosign);




%--------------------------
%	PLOT
%--------------------------
cAx = 1;
jj = find( abs(yG(1,:)-26.5) == min(abs(yG(1,:)-26.5)) );

figure(10)
clf
set(gcf,'position',[40 400 1200 400])
%- FORCED -
subplot(121)
pcolor(yG(1,:),rF(1:nr)./1000,reg_f_nosign'.*msk_lnd');
shading flat
caxis([-cAx cAx])
colormap(mycmapGray)
freezeColors
hold on
pcolor(yG(1,:),rF(1:nr)./1000,reg_f_sign'.*msk_lnd');
shading flat
colormap(mycmap)
caxis([-cAx cAx])
plot([0 0],[rF(nr)/1000 0],'color',[.3 .3 .3],'lineS','--')
plot([yG(1,jj) yG(1,jj)],[rF(nr)/1000 0],'k--','lineW',1.2)
title(['Forced'])
xlabel('Latitude')
ylabel('Depth [km]')
set(gca,'Color',[.2 .2 .2])
set(gcf,'Color',[1 1 1])

%- INTRINSIC -
subplot(122)
pcolor(yG(1,:),rF(1:nr)./1000,reg_i_nosign'.*msk_lnd');
shading flat
caxis([-cAx cAx])
colormap(mycmapGray)
freezeColors
hold on
pcolor(yG(1,:),rF(1:nr)./1000,reg_i_sign'.*msk_lnd');
shading flat
colormap(mycmap)
caxis([-cAx cAx])
plot([0 0],[rF(nr)/1000 0],'color',[.3 .3 .3],'lineS','--')
plot([yG(1,jj) yG(1,jj)],[rF(nr)/1000 0],'k--','lineW',1.2)
title(['Intrinsic'])
xlabel('Latitude')
ylabel('Depth [km]')
%- colorbar -
hc = colorbar;
hc.Position = [.91 .25 .01 .5];
hc.FontSize = 8;
hc.Label.String = '[Sv]';
%- background color -
set(gca,'Color',[.2 .2 .2])
set(gcf,'Color',[1 1 1])

%- save -
fig=gcf;
fig.InvertHardcopy = 'off';
fileN10 = ['AMOCyzt_' config '_reg_26n_forced_intrinsic_24memb'];
exportfig(figure(10),[dir_fig fileN10 '.png'],...
       'width',6,'color','rgb','resolution',300);
exportfig(figure(10),[dir_fig fileN10 '.pdf'],...
       'width',6,'color','rgb','resolution',300);


return

%-----------------------------------------------
% Look at regression for each individual members
%-----------------------------------------------

figure(20)
clf
set(gcf,'position',[40 400 1500 1000])

for imem = 1:nb_memb
%- intrinsic -
 reg_i = reg_moc(:,:,imem);
 tmp_sign = sig_moc(:,:,imem);
 tmp_sign(tmp_sign > thres) = 99999;
 tmp_sign(tmp_sign <= thres) = 1;
 tmp_sign(tmp_sign == 99999) = 0;
 jk_sign = find(tmp_sign == 1);
 jk_nosign = find(tmp_sign < 1);
 reg_i_sign = nan(ny,nr);
 reg_i_sign(jk_sign) = reg_i(jk_sign);
 reg_i_nosign = nan(ny,nr);
 reg_i_nosign(jk_nosign) = reg_i(jk_nosign);

 subplot(4,3,imem)
 pcolor(yG(1,:),rF(1:nr)./1000,reg_i_nosign'.*msk_lnd');
 shading flat
 caxis([-cAx cAx])
 colormap(mycmapGray)
 freezeColors
 hold on 
 pcolor(yG(1,:),rF(1:nr)./1000,reg_i_sign'.*msk_lnd');
 shading flat
 colormap(mycmap)
 caxis([-cAx cAx])
 plot([0 0],[rF(nr)/1000 0],'color',[.3 .3 .3],'lineS','--')
 plot([yG(1,jj) yG(1,jj)],[rF(nr)/1000 0],'k--','lineW',1.2)
 title(['memb#' num2str(imem-1,'%02.f')])
end % for imem 
%- colorbar -
hc = colorbar;
hc.Position = [.91 .25 .01 .5];
hc.FontSize = 8;
hc.Label.String = '[Sv]';

%- save -
fig=gcf;
fig.InvertHardcopy = 'off';
fileN20 = ['AMOCyzt_' config '_reg_26n_intrinsic_12memb'];
exportfig(figure(20),[dir_fig fileN20 '.png'],...
       'width',6,'color','rgb','resolution',300);
exportfig(figure(20),[dir_fig fileN20 '.pdf'],...
       'width',6,'color','rgb','resolution',300);



