clear all; close all

ieee = 'b';
accu = 'real*4';
if strcmp(accu,'real*4')
  accu2 = 4;
else
  accu2 = 8;
end


%-- directories --
dir_in = '/tank/chaocean/qjamet/RUNS/test_cheap025/';
dir_grd = '/tank/chaocean/qjamet/RUNS/test_cheap025/gridMIT/';
dir_fig = '/tank/users/qjamet/Figures/publi/note_bams/';

config = {'cheap_clim_atm','cheap_fv_wind','cheap_norm_yr'...
          'rest_clim_atm','rest_fv_wind'};
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
runs = 1958:1962;
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

sst = nan(nx,ny,niter,nConf+1);
t2 = nan(nx,ny,niter,nConf+1);
for irun = 1:nRun
  fprintf('Year: %.00f\n',runs(irun));
  tmpyr = runs(irun);
  tmpiter = iter((irun-1)*nDump+1:irun*nDump);
  %- corrected LW radiations first -
  if runs(irun) <= 1959
    disp('    -->> add corrected LW run')
    tmpdir = [dir_in config{1} '/'];
    for iiter = 1:nDump
      %- sst -
      fid = fopen([tmpdir 'run' num2str(runs(irun)) '_corrLW/ocn/diag_ocnTave.' ...
	num2str(tmpiter(iiter),'%010.f') '.data'],'r',ieee);
      sst(:,:,(irun-1)*nDump+iiter,1) = fread(fid,[nx ny],accu);
      fclose(fid);
      %- t2 -
      fid = fopen([tmpdir 'run' num2str(runs(irun)) '_corrLW/cheapaml/diag_cheapAML.' ...
        num2str(tmpiter(iiter),'%010.f') '.data'],'r',ieee);
      t2(:,:,(irun-1)*nDump+iiter,1) = fread(fid,[nx ny],accu);
      fclose(fid);
    end % for iiter
  end % if yr <= 1959
  %- other config -
  for iconf = 1:nConf
    fprintf('Conifg: %s\n',config{iconf});
    tmpdir = [dir_in config{iconf} '/'];
    for iiter = 1:nDump
      %- sst -
      fid = fopen([tmpdir 'run' num2str(runs(irun)) '/ocn/diag_ocnTave.' ...
	num2str(tmpiter(iiter),'%010.f') '.data'],'r',ieee);
      sst(:,:,(irun-1)*nDump+iiter,iconf+1) = fread(fid,[nx ny],accu);
      fclose(fid);
      %- t2 -
      fid = fopen([tmpdir 'run' num2str(runs(irun)) '/cheapaml/diag_cheapAML.' ...
        num2str(tmpiter(iiter),'%010.f') '.data'],'r',ieee);
      t2(:,:,(irun-1)*nDump+iiter,iconf+1) = fread(fid,[nx ny],accu);
      fclose(fid);
    end %iiter
  end % iconf 
end % irun

%-- load climatological t2 forcing --
fid = fopen('/tank/chaocean/qjamet/Config/Test_cheapAML0.25/data_in/atm_cd/t2_clim.box','r',ieee);
t2_clim = fread(fid,accu);
t2_clim_5d = squeeze(mean(reshape(t2_clim,[nx ny 20 nDump]),3));
t2_clim_5d_tserie = repmat(t2_clim_5d,[1 1 5]);


%-- load t2 from ORAC ensemble --
dir_orac = '/tank/chaocean/qjamet/RUNS/ORAC/memb00/';
run12 = 1963:2012;
nrun12 = length(run12);
[dt] = 200;
spy = 86400*365;
[dump] = 5*86400;       %5-d dumps
d_iter = dump/dt;
nDump = 86400*365/dump;
offset = (run12(1)-1958)*spy/dt;
iter12 = [d_iter:d_iter:(nDump*d_iter)*nrun12] + offset;
[niter12] = length(iter12);
t2_orac = zeros(1000,900,nDump,nrun1);
for irun = 1:nrun12
  tmpIter = iter12((irun-1)*nDump+1:irun*nDump);
  fid = fopen([dir_orac 'run' num2str(run12(irun)) '/cheapaml/diag_cheapAML')
end


%-- averaging sst within the subtropical box --
sst_box = squeeze(sum(sum( sst(ii,jj,:,:) .* repmat(rA1,[1 1 niter nConf+1]) ,2),1) ./ sum(rA1(:)));
t2_box = squeeze(sum(sum( t2(ii,jj,:,:) .* repmat(rA1,[1 1 niter nConf+1]) ,2),1) ./ sum(rA1(:)));
t2_clim_box = squeeze(sum(sum( t2_clim_5d_tserie(ii,jj,:) .* ...
	repmat(rA1,[1 1 niter]) ,2),1) ./ sum(rA1(:)));

figure(10);
clf;
set(gcf,'position',[0 400 1000 500])
hold on
[p3] = plot(ddate,sst_box(:,3),'b','lineW',1.2);
%[p6] = plot(ddate,sst_box(:,6),'k--');
[p2] = plot(ddate,sst_box(:,2),'r','lineW',1.2);
%[p5] = plot(ddate,sst_box(:,5),'b--');
[p4] = plot(ddate,sst_box(:,4),'k','lineW',1.2);
[p1] = plot(ddate,sst_box(:,1),'color',[.1 .6 .3],'lineW',1.2);
legend([p3;p2;p4;p1],...
	'AML\_FULL','AML\_CLIM','AML\_NY','AML\_CORR\_LW');
grid on
set(gca,'xTickLabel',[{'1958'},{''},{'1959'},{''},{'1960'},{''},{'1961'},{''},{'1962'},{''},{'1963'}])
title('Subtropical gyre SST')
xlabel('Time [yr]')
ylabel('SST [^{o}C]')
%- save -
fileN10 = ['SST_gyre_diff_conf_rev01.pdf'];
exportfig(figure(10),[dir_fig fileN10],...
    'width',6,'color','rgb','resolution',300);


%-- map of time mean SST difference --
sst_m = squeeze(mean(sst(:,:,end-nDump:end,:),3));

figure(20);
clf
set(gcf,'Position',[0 400 600 500]);
m_proj('lambert','long',[min(xC(:))  max(xC(:))],...
                 'lat',[min(yC(:)) max(yC(:))]);
% AML_NY - AML_FULL 
%subplot(121)
[c,h1] = m_contourf(xC, yC, sst_m(:,:,4)-sst_m(:,:,3), [-2:0.1:2]);
set(h1, 'lineS', 'none')
caxis([-2 2])
hold on
[c0,h0] = m_contour(xC, yC, sst_m(:,:,4)-sst_m(:,:,3), [0 0], 'k', 'lineW', 1.2);
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
xlabel('Longitude')
xlabel('Latitude')
title([{'SST'},{'(AML\_NY) - (AML\_FULL)'}])
% colorbar
load('MyCmap_redBlue.mat')
colormap(mycmap)
h = colorbar('location','southOutside');
h.Position = [.3 .13 .43 .02];
h.FontSize = 10;
h.Label.String = '[^oC]';
h.Label.FontSize = 12;
set(h,'YTick',[-2:1:2]);
set(h,'YTickLabel',[-2:1:2]);
fileN20 = ['diff_sst_aml_ny_full3.pdf'];
exportfig(figure(20),[dir_fig fileN20],...
    'width',6,'color','rgb','resolution',300);




% AML_CLIM - AML_FULL
subplot(122)
[c,h2] = m_contourf(xC, yC, sst_m(:,:,2)-sst_m(:,:,3), [-8:0.1:8]);
set(h2, 'lineS', 'none')
caxis([-10 10])
hold on
[c0,h0] = m_contour(xC, yC, sst_m(:,:,2)-sst_m(:,:,3), [0 0], 'k', 'lineW', 1.2);
xlabel('Longitude')
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
xlabel('Latitude')
title([{'SST'},{'(AML\_CLIM) - (AML\_FULL)'}])
% colorbar
load('MyCmap_redBlue.mat')
colormap(mycmap)
h = colorbar('location','southOutside');
h.Position = [.3 .13 .43 .02];
h.FontSize = 10;
h.Label.String = '[^oC]';
h.Label.FontSize = 12;
set(h,'YTick',[-10:5:10]);
set(h,'YTickLabel',[-10:5:10]);
%- save -
fileN20 = ['diff_sst_aml_ny_full2.pdf'];
exportfig(figure(20),[dir_fig fileN20],...
    'width',10,'color','rgb','resolution',300);


%-- spectrum of t2 at the center of the gyre --
fs = 73;                        % sampling frequency [yr-1]
freq =  fs*(0:(niter/2))/niter;
nfft = length(freq);
fft_t2_box = zeros(nfft,nConf);
for iconf = 1:nConf
  tmp = fft(t2_box(:,iconf+1));		% corr_LW is the first, not considered
  tmp2 = abs(tmp/niter);
  tmp1 = tmp2(1:floor(niter/2)+1);
  tmp1(2:end-1) = 2*tmp1(2:end-1);
  fft_t2_box(:,iconf) = tmp1;
end
tmp = fft(t2_clim_box);
tmp2 = abs(tmp/niter);
tmp1 = tmp2(1:floor(niter/2)+1);
tmp1(2:end-1) = 2*tmp1(2:end-1);
fft_t2_clim = tmp1;
%- look at what remains in t2 for AML_NY once seasonal cycle is removed -
t2_ny = t2_box(:,4);
t2_ny_resid = t2_ny - reshape(repmat( mean(reshape(t2_ny,[nDump nRun]),2), [1 nRun] ),[niter 1]);
tmp = fft(t2_ny_resid);
tmp2 = abs(tmp/niter);
tmp1 = tmp2(1:floor(niter/2)+1);
tmp1(2:end-1) = 2*tmp1(2:end-1);
fft_t2_ny = tmp1;


figure(30)
clf
%[p1] = semilogx(1./freq,fft_t2_ny,'k','lineW',1.2);
[p1] = semilogx(1./freq,fft_t2_box(:,2),'k','lineW',1.2);
hold on
[p2] = semilogx(1./freq,fft_t2_box(:,3),'r','lineW',1.2);
[p3] = semilogx(1./freq,fft_t2_box(:,4),'b');
set(gca,'Xdir','reverse','xLim',[2/73 5])
grid on
xlabel('Period [yr]')
ylabel('Peridogram')
%- save -
fileN30 = ['diff_sst_aml_ny_full.pdf'];
exportfig(figure(20),[dir_fig fileN20],...
    'width',10,'color','rgb','resolution',300);

