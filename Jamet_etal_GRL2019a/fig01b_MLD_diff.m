% DESCRIPTION:
%
% Compare the Mixed layer depth for:
%	- with cheapAML and no restoring over the ocean (Ref exp)
%	- with cheapAML and a restoring (T and Q) at a rate tauRelaxOcn = 1 [d]
%	- with cheapAML and a restoring (T and Q) at a rate tauRelaxOcn = 0.1 [d]

clear all; close all


flg_save = 0;

%-- directories --
dir_in = '/tank/chaocean/qjamet/RUNS/test_cheap025/';
dir_grd = '/tank/chaocean/qjamet/RUNS/test_cheap025/gridMIT/';
dir_fig = '/tank/users/qjamet/Figures/publi/note_bams/';

%config = {'cheap_clim_atm','cheap_clim_wind','cheap_fv_wind',...
%          'rest_clim_atm','rest_clim_wind','rest_fv_wind'};
config = {'cheap_clim_atm','cheap_fv_wind','cheap_norm_yr'};
nConf = length(config);

%-- load grid --
global xC yC dxG dyG drF rC nx ny nr hC
loadGRD(dir_grd)
xC = xC-360;
volCell = dxG.*dyG;
mskNaN = hC(:,:,1);
mskNaN(mskNaN ~= 0) = 1;
mskNaN(mskNaN == 0) = NaN;

%- time parameters -
runs = 1958;
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

kpphbl = zeros(nx,ny,niter,nConf);
%-- load data --
for iConf = 1:nConf
 if iConf == nConf
  tmp_dir = [dir_in config{iConf} '/run' num2str(runs) '/'];
 else
  tmp_dir = [dir_in config{iConf} '/run' num2str(runs) '_kpp/'];
 end
 tmp = rdmds([tmp_dir 'ocn/kpp2d'],iter);
 kpphbl(:,:,:,iConf) = squeeze(tmp(:,:,1,:));
end % for iConf

kpphbl_m = squeeze(mean(kpphbl,3));
kpphbl_m = kpphbl_m .* repmat(mskNaN,[1 1 nConf]);
kpphbl_std = squeeze(std(kpphbl,0,3));
kpphbl_std = kpphbl_std .* repmat(mskNaN,[1 1 nConf]);
kpphbl_max = zeros(nx*ny,nConf);
tmp1 = reshape(kpphbl,[nx*ny niter nConf]);
for iconf = 1:nConf
 for ij = 1:nx*ny
  kpphbl_max(ij,iconf) = max(tmp1(ij,:,iconf));
 end
end
kpphbl_max = reshape(kpphbl_max,[nx ny nConf]) .* repmat(mskNaN,[1 1 nConf]);


%-------------------------
% 	PLOT
%-------------------------

load('/tank/users/qjamet/MatLab/MyCmap_serazin.mat')
mycmap1 = mycmap;

toto = kpphbl_max;
myscale = [0 350];
ticks_wanted = [0:10:350];

%-- CHEAPAML clim vs real atm --
figure(10)
clf
set(gcf,'Position',[0 400 1400 500]);
m_proj('lambert','long',[min(xC(:))  max(xC(:))],...
                 'lat',[min(yC(:)) max(yC(:))]);
%-- real atm --
subplot(122)
[C,h1]=m_contourf(xC,yC,toto(:,:,1),...
    [myscale(1) ticks_wanted myscale(2)]);
%[C,h1]=m_contourf(xC,yC,toto(:,:,3),...
%    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
hold on
%[C2,h2]=m_contour(xC,yC,toto2(:,:,1),[0:10:100],'k','lineW',1.2);
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['AML\_CLIM'])
%title(['AML\_NY'])
colorbar off
% box in subploar gyre
bndry_lon=[-40 -40 -35 -35 -40];
bndry_lat=[ 30  35  35  30  30];
m_line(bndry_lon,bndry_lat,'lineW',1.2,'color','g');
%-- WIND --
subplot(121)
[C,h1]=m_contourf(xC,yC,toto(:,:,2),...
    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
hold on
%[C2,h2]=m_contour(xC,yC,toto2(:,:,2),[0:10:100],'k','lineW',1.2);
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['AML\_FULL'])
% box in subploar gyre
bndry_lon=[-40 -40 -35 -35 -40];
bndry_lat=[ 30  35  35  30  30];
m_line(bndry_lon,bndry_lat,'lineW',1.2,'color','g');
h = colorbar('location','southOutside');
h.Position = [.3 .13 .43 .02];
h.FontSize = 10;
h.Label.String = '[m]';
h.Label.FontSize = 12;
set(h,'YTick',ticks_wanted(1:5:end));
set(h,'YTickLabel',ticks_wanted(1:5:end));
%- save -
if flg_save
  fileN10 = ['KPPhbl_max_fvVSclim_cheapaml_1958.pdf'];
%  fileN10 = ['KPPhbl_max_fvVSny_cheapaml_1958.pdf'];
  exportfig(figure(10),[dir_fig fileN10],...
      'width',10,'color','rgb','resolution',300);
end


%-- RESTORING experiments --
myscale = [0 120];
ticks_wanted = [0:2:120];
figure(11)
clf
set(gcf,'Position',[0 400 1200 500]);
m_proj('lambert','long',[min(xC(:))  max(xC(:))],...
                 'lat',[min(yC(:)) max(yC(:))]);
%-- clim atm --
subplot(122)
[C,h1]=m_contourf(xC,yC,mld_m(:,:,3),...
    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['CLIM -- Flux exp'])
colorbar off
%-- real atm --
subplot(121)
[C,h1]=m_contourf(xC,yC,mld_m(:,:,4),...
    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['FULL -- Flux exp'])
h = colorbar('location','southOutside');
h.Position = [.3 .13 .43 .02];
h.FontSize = 10;
h.Label.String = '[m]';
h.Label.FontSize = 12;
set(h,'YTick',ticks_wanted(1:10:end));
set(h,'YTickLabel',ticks_wanted(1:10:end));
%- save -
if flg_save
  fileN11 = ['MLD_fvVSclim_RESTORING_yeralymean.pdf'];
  exportfig(figure(11),[dir_fig fileN11],...
      'width',10,'color','rgb','resolution',300);
end


