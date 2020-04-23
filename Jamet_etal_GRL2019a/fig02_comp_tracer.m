% DESCRIPTION
%
% Look at differences between runs for tracers, 
% at different levels 
% 	*_clim_atm	-> clim atm fields
% 	*_clim_wind	-> only clim winds
% 	*_fv_wind	-> ALL atm fields are 'real forcing'

clear all; close all


flg_save = 0;

%-- directories --
dir_in = '/tank/chaocean/qjamet/RUNS/test_cheap025/';
dir_grd = '/tank/chaocean/qjamet/RUNS/test_cheap025/gridMIT/';
dir_fig = '/tank/users/qjamet/Figures/publi/note_bams/';

config = {'cheap_clim_atm','cheap_fv_wind',...
          'rest_clim_atm','rest_fv_wind'};
%config = {'rest_clim_atm','rest_clim_wind'};
nConf = length(config);

%-- load grid --
global xC yC dxG dyG drF rC nx ny nr hC rA
loadGRD(dir_grd)
xC = xC-360;
volCell = dxG.*dyG;
mskNaN = hC;
mskNaN(mskNaN ~= 0) = 1;
mskNaN(mskNaN == 0) = NaN;
[ii] = find(xC(:,1)>=-40 & xC(:,1)<=-35);
[jj] = find(yC(1,:)>=30 & yC(1,:)<=35);
[nx1] = length(ii);
[ny1] = length(jj);
rA1 = rA(ii,jj);
xC1 = xC(ii,jj);
yC1 = yC(ii,jj);


%- time parameters -
runs = 1967;
nRun = length(runs);
runIni = 1958;
[dt] = 450;
spy = 86400*365;
[dump] = 5*86400;       %5-d dumps
d_iter = dump/dt;
nDump = 86400*365/dump;
offset = (runs(1)-1958)*spy/dt;;
iter = [0:d_iter:(nDump*d_iter)*nRun] + offset;
iter = iter(2:end);
[niter] = length(iter);

date_y = floor((iter*dt/86400-1)/365 ) + 1958;
date_d = (iter*dt/86400);
ddate = date_d/365 + 1958;


%- initialisation -
flg_tr = 'T';
tr_m = zeros(nx,ny,nr,nConf);		% clim wind, cheapaml (REF)

for iConf = 1:nConf
 sprintf('-- loading config: %s',config{iConf});
 tmp_dir = [dir_in config{iConf} '/run' num2str(runs) '/'];
 for iIter = 1:niter
  tmp = rdmds([tmp_dir 'ocn/diag_ocnTave'],iter(iIter));
  if strcmp(flg_tr,'T')
   tr_m(:,:,:,iConf) = tr_m(:,:,:,iConf) + tmp(:,:,:,1);
  elseif strcmp(flg_tr,'S')
   tr_m(:,:,:,iConf) = tr_m(:,:,:,iConf) + tmp(:,:,:,2);
  end
 end %for iIter
end % for iConf

tr_m = tr_m ./ niter;



%-----------------------
% compute sst difference
%-----------------------


nk = 1;
hh = sum(drF(nk));
tr_m_zm = sum( tr_m(:,:,nk,:) .* repmat(drF(:,:,nk),[nx ny 1 nConf])  ,3) ./ hh .* ...
          repmat(mskNaN(:,:,1),[1 1 1 nConf]);
tr_m_zm = squeeze(tr_m_zm);

diff_cheap_zm = tr_m_zm(:,:,1) - tr_m_zm(:,:,2);
diff_flux_zm = tr_m_zm(:,:,3) - tr_m_zm(:,:,4);


%---------------------------
% sst evolution in box 
%---------------------------
niter2 = 730;
ddate = [5/365:5/365:5*niter2/365] +1958;
fid = fopen([dir_in 'data/ocn_diag_box1_1958_67.bin'],'r','b');
diag_ocn = fread(fid,'real*4');
fclose(fid);
diag_ocn = reshape(diag_ocn,[nx1 ny1 nr niter2 5 4]);

%-- sst --
tmp = squeeze(diag_ocn(:,:,1,:,1,:));
sst = squeeze(sum(sum( tmp.*repmat(rA1,[1 1 niter2 nConf]) ,1),2 ) ./ sum(rA1(:)));

%-- compute a trend for the last 9 years --
% from Qnet/(rho0*Cp*h) -->> 7 K.yr^{-1} -->> diffusion accures I guess

pInd = zeros(2,nConf);
for iconf = 1:nConf
%  pInd(:,iconf) = polyfit(ddate(nDump+1:end)-1958,sst(nDump+1:end,iconf)',1);
  pInd(:,iconf) = polyfit(ddate(37:end)-1958,sst(37:end,iconf)',1);
end


%-----------------------------------
%		PLOT
%-----------------------------------

load('/tank/users/qjamet/MatLab/MyCmap_redBlue.mat')
mycmap1 = mycmap;
myscale = [-10 10];
ticks_wanted = [-10:5:10];
units = '[^0C]';
str = 'Temp.';



figure(10)
clf
set(gcf,'Position',[0 400 1200 500]);
m_proj('lambert','long',[min(xC(:))  max(xC(:))],...
                 'lat',[min(yC(:)) max(yC(:))]);
%-- spatial SST diff --
subplot(121)
[C,h1]=m_contourf(xC,yC,diff_cheap_zm,[-10:0.5:10]);
set(h1,'lineStyle','none')
hold on
[C0,h0]=m_contour(xC,yC,diff_cheap_zm,[0 0],'k','lineW',1.2);
clabel(C0,h0)
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title([{'SST'},{'(AML\_CLIM) - (AML\_FULL)'}])
h = colorbar('location','southOutside');
h.Position = [.05 .13 .3 .02];
h.FontSize = 10;
h.Label.String = '^oC';
h.Label.FontSize = 12;
set(gca,'OuterPosition',[-.1 .175 0.55 .65])
% box in subploar gyre
bndry_lon=[-40 -40 -35 -35 -40];
bndry_lat=[ 30  35  35  30  30];
m_line(bndry_lon,bndry_lat,'lineW',1.2,'color','g');
%-- SST diff in box --
subplot(122)
hold on
[p00]= plot(ddate,25*ones(1,niter2),'color',[.7 .7 .7],'lineS','--');
plot([ddate(32) ddate(32)],[18 40],'k--')
[p0] = plot(ddate,(sst(:,1)-sst(:,2))+25,'color',[.7 .7 .7]);
[p1] = plot(ddate,sst(:,1),'r','lineW',1.2);
[p2] = plot(ddate,sst(:,2),'b','lineW',1.2);
h=legend([p2;p1;p0],'AML\_FULL','AML\_CLIM','location','northWest');
set(h,'box','off')
grid on
set(gca,'Xlim',[1958 ddate(niter2)],...
     'xTick',[1958:ddate(niter2)],...
     'xTickLabel',[{'1958'},{''},{'1960'},{''},{''},{''},{''},{'1965'},{''},{''},{''}],...
     'yLim',[18 40],'OuterPosition',[0.4 .175 0.55 .65],...
     'yTick',[20:5:40],'yTickLabel',[{'20 - \color{gray}-5'},...
     {'25 - \color{gray}0'},...
     {'30 - \color{gray}5'},{'35 - \color{gray}10'},{'40 - \color{gray}15'}])
xlabel('Time [yr]')
ylabel('[^oC]')
title('Subtropical Gyre SST')
%- save -
fileN10 = ['SST_diff_climVSfull_cheap_and_box_avg_bis.pdf'];
exportfig(figure(10),[dir_fig fileN10],...
    'width',10,'color','rgb','resolution',300);


figure(11)
clf
set(gcf,'Position',[0 400 1200 500]);
m_proj('lambert','long',[min(xC(:))  max(xC(:))],...
                 'lat',[min(yC(:)) max(yC(:))]);
%-- Spatial sst diff --
subplot(121)
[C,h1]=m_contourf(xC,yC,diff_flux_zm,[-10:0.5:10]);
set(h1,'lineStyle','none')
hold on
[C0,h0]=m_contour(xC,yC,diff_flux_zm,[0 0],'k','lineW',1.2);
clabel(C0,h0)
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title([{'SST'},{'(FORC\_CLIM) - (FORC\_FULL)'}])
h = colorbar('location','southOutside');
h.Position = [.05 .13 .3 .02];
h.FontSize = 10;
h.Label.String = '^oC';
h.Label.FontSize = 12;
set(gca,'OuterPosition',[-.1 .175 0.55 .65])
% box in subploar gyre
bndry_lon=[-40 -40 -35 -35 -40];
bndry_lat=[ 30  35  35  30  30];
m_line(bndry_lon,bndry_lat,'lineW',1.2,'color','g');
%-- sst diff in box --
subplot(122)
hold on
[p00]= plot(ddate,25*ones(1,niter2),'color',[.7 .7 .7],'lineS','--');
plot([ddate(32) ddate(32)],[18 40],'k--')
[p0] = plot(ddate,(sst(:,3)-sst(:,4))+25,'color',[.7 .7 .7]);
[p1] = plot(ddate,sst(:,3),'r','lineW',1.2);
[p2] = plot(ddate,sst(:,4),'b','lineW',1.2);
h=legend([p2;p1;p0],'FORC\_FULL','FORC\_CLIM','location','northWest');
set(h,'box','off')
grid on
set(gca,'Xlim',[1958 ddate(niter2)],...
     'xTick',[1958:ddate(niter2)],...
     'xTickLabel',[{'1958'},{''},{'1960'},{''},{''},{''},{''},{'1965'},{''},{''},{''}],...
     'yLim',[18 40],'OuterPosition',[0.4 .175 0.55 .65],...
     'yTick',[20:5:40],'yTickLabel',[{'20 - \color{gray}-5'},...
     {'25 - \color{gray}0'},...
     {'30 - \color{gray}5'},{'35 - \color{gray}10'},{'40 - \color{gray}15'}])
xlabel('Time [yr]')
ylabel('[^oC]')
title('Subtropical Gyre SST')
%- save -
fileN11 = ['SST_diff_climVSfull_FLUX_and_box_avg_bis.pdf'];
exportfig(figure(11),[dir_fig fileN11],...
    'width',10,'color','rgb','resolution',300);

return


%-- FLUX config --
myscale = [-5 5];
ticks_wanted = [-5:1:5];
units = '[^0C]';
str = 'Temp.';

figure(20)
clf
set(gcf,'Position',[0 400 800 500]);
m_proj('lambert','long',[min(xC(:))  max(xC(:))],...
                 'lat',[min(yC(:)) max(yC(:))]);
[C,h1]=m_contourf(xC,yC,diff_flux_zm,[-5:0.1:5]);
set(h1,'lineStyle','none')
hold on
[C0,h0]=m_contour(xC,yC,diff_flux_zm,[0 0],'k','lineW',1.2);
clabel(C0,h0)
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title([{'FLUX'},{['SST^{CLIM} - SST^{FULL}']}])
h = colorbar('location','southOutside');
h.Position = [.05 .13 .3 .02];
h.FontSize = 10;
h.Label.String = '^oC';
h.Label.FontSize = 12;
%set(gca,'OuterPosition',[-.1 .175 0.55 .65])
% box in subploar gyre
bndry_lon=[-40 -40 -35 -35 -40];
bndry_lat=[ 30  35  35  30  30];
m_line(bndry_lon,bndry_lat,'lineW',1.2,'color','g');
%- save -
fileN10 = ['SST_diff_climVSfull_cheap_and_box_avg_bis.pdf'];
exportfig(figure(10),[dir_fig fileN10],...
    'width',10,'color','rgb','resolution',300);


