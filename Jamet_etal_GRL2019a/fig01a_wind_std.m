% Compute climatological winds in the traditional way

clear all; close all;

load('/tank/users/qjamet/MatLab/MyCmap_serazin.mat')
mycmap1 = mycmap;
load('/tank/users/qjamet/MatLab/MyCmap_redBlue.mat')
mycmap2 = mycmap;

%-- directories and grid --
dir_atm = '/tank/chaocean/atmospheric_data/DFS4.4_NorthAtl/';
dir_ini = '/tank/chaocean/initial_data/grid/'; 
dir_data = '/tank/users/qjamet/MatLab/publis/note_bams/data/';
dir_fig = '/tank/users/qjamet/Figures/publi/note_bams/';


%------------------------
% Get spatial dimension
% u and v are on the same t-pts grid
%------------------------
lat = double(ncread([dir_atm 'u10_DFS4.4_y1963_chaO.nc'],'lat'));
lon = double(ncread([dir_atm 'u10_DFS4.4_y1963_chaO.nc'],'lon'));
[nx] = size(lon,1);
[ny] = size(lat,1);
for ii =1:nx
  if (lon(ii)> 180); lon(ii) = lon(ii) - 360; end
end
[yC,xC]=meshgrid(lat,lon);

%-- grid spacing and cell area --
rEarth = 6.371e6;%[m]
xG = zeros(nx+1,ny);
xG(2:nx,:) = (xC(1:nx-1,:)+xC(2:nx,:)) ./2;
xG(1,:) = xG(2,:) - (xC(2,:)-xC(1,:));
xG(nx+1,:) = xG(nx,:) + ( xC(nx,:)-xC(nx-1,:) );
yG = zeros(nx,ny+1);
yG(:,2:ny) = ( yC(:,1:ny-1)+yC(:,2:ny) ) ./2;
yG(:,1) = yG(:,2) - ( yC(:,2)-yC(:,1) );
yG(:,ny+1) = yG(:,ny) + ( yC(:,ny)-yC(:,ny-1) );
dxG = deg2rad( xG(2:nx+1,:)-xG(1:nx,:) ) .* rEarth .* cosd( yC );
dyG = deg2rad( yG(:,2:ny+1)-yG(:,1:ny) ) .* rEarth;
rA = dxG.*dyG;

%-- time dim --
ttime = double(ncread([dir_atm 'u10_DFS4.4_y1958_chaO.nc'],'time'));
[nt] = length(ttime);
yyears = 1958:1977;
nYr = length(yyears);

%-- initialisation --
u10_clim = zeros(nx,ny,nt);
v10_clim = zeros(nx,ny,nt);
umod_clim = zeros(nx,ny,nt);
umod_fv = zeros(nx,ny,nt*nYr);
u10_fv = zeros(nx,ny,nt*nYr);
v10_fv = zeros(nx,ny,nt*nYr);

%-- compute climatology --
for iYr = 1:nYr
  fprintf('-- Year: %i --\n',yyears(iYr))
  tmp_u10 = ncread([dir_atm 'u10_DFS4.4_y' ...
	num2str(yyears(iYr)) '_chaO.nc'],'u10');
  tmp_v10 = ncread([dir_atm 'v10_DFS4.4_y' ...
	num2str(yyears(iYr)) '_chaO.nc'],'v10');
  tmp_mod = sqrt(tmp_u10.^2 + tmp_v10.^2);

  umod_fv(:,:,(iYr-1)*nt+1:iYr*nt) = tmp_mod;
  u10_fv(:,:,(iYr-1)*nt+1:iYr*nt) = tmp_u10;
  v10_fv(:,:,(iYr-1)*nt+1:iYr*nt) = tmp_v10;

  u10_clim = u10_clim + tmp_u10;
  v10_clim = v10_clim + tmp_v10;
  umod_clim = umod_clim + sqrt(tmp_u10.^2 + tmp_v10.^2);

end % for iYr

u10_clim = u10_clim ./ nYr;
v10_clim = v10_clim ./ nYr;
umod_clim = umod_clim ./ nYr;
umod_clim_extd = repmat(umod_clim,[1 1 nYr]);

%-- remove seasonal cycle (and trend for fv) --
span = 120;
%- fully varying -
tmp = reshape(umod_fv,[nx*ny nt*nYr]);
umod_fv_detrend = zeros(nx*ny,nt*nYr);
for ij = 1:nx*ny
 fprintf('%i/%i\n',ij,nx*ny)
 trend = smooth(tmp(ij,:),span,'loess');
 umod_fv_detrend(ij,:) = tmp(ij,:) - trend';
end
umod_fv_detrend = reshape(umod_fv_detrend,[nx ny nt*nYr]);
%- climatology -
tmp = reshape(umod_clim,[nx*ny nt]);
umod_clim_detrend = zeros(nx*ny,nt);
for ij = 1:nx*ny
 fprintf('%i/%i\n',ij,nx*ny)
 trend = smooth(tmp(ij,:),span,'loess');
 umod_clim_detrend(ij,:) = tmp(ij,:) - trend';
end
umod_clim_detrend = reshape(umod_clim_detrend,[nx ny nt]);


%----------------------------------
% Compute eof/pc and power spectrum
%----------------------------------
neof = 10;
Fs = 1/6;             %[hr-1]
NFFT = 2^nextpow2(nt);
freq =  Fs/2*linspace(0,1,NFFT/2+1);


%- climatological winds --
%- apply area weight -
tmp = umod_clim .* repmat(rA,[1 1 nt]);
%- reshape -
tmp = reshape(permute(tmp,[3 1 2]),[nt nx*ny]);
[E,pc,expvar] = caleof(tmp,neof,2);
eof = permute(reshape(E,[neof nx ny]),[2 3 1]);
%- remove area weight -
eof = eof ./ repmat(rA,[1 1 neof]);
%- normalise the pc -
eof_clim = eof .* repmat(reshape(std(pc,0,2),[1 1 neof]),[nx ny 1]);
pc_clim = pc ./ repmat(std(pc,0,2),[1 nt]);

%- remove seasonal cycle
span = 120;	% monthly
pc2_clim = zeros(neof,nt);
for ieof = 1:neof
 trend = smooth(pc_clim(ieof,:),span,'loess');
 pc2_clim(ieof,:) = pc_clim(ieof,:) - trend';
end % for ieof

%- power sepctrum -
fft_pc_clim = zeros(neof,NFFT/2+1);
for ieof = 1:neof
  tmp = fft(pc2_clim(ieof,:));
  fft_pc_clim(ieof,:) = 2*abs(tmp(1:NFFT/2+1));
end
[p1] = semilogx(1./freq,fft_pc_clim(1,:),'b','lineW',1.2);



%-- fully varying winds --
%- apply area weight -
tmp = umod_fv .* repmat(rA,[1 1 nt*nYr]);
tmp = reshape(tmp,[nx ny nt nYr]);
%- loop over years -
eof_fv = zeros(nx,ny,neof,nYr);
pc_fv = zeros(neof,nt,nYr);
fft_pc_fv = zeros(neof,NFFT/2+1,nYr);
expvar_fv = zeros(neof,nYr);
for iYr = 1:nYr
 %- reshape -
 tmp2 = reshape(permute(tmp(:,:,:,iYr),[3 1 2]),[nt nx*ny]);
 [E,pc,expvar] = caleof(tmp2,neof,2);
 eof = mat2map(ones(nx,ny),E);
 eof = permute(eof,[2 3 1]);
 %- remove area weight -
 eof = eof ./ repmat(rA,[1 1 neof]);
 %- normalise the eof/pc -
 eof = eof .* repmat(reshape(std(pc,0,2),[1 1 neof]),[nx ny 1]);
 pc = pc ./ repmat(std(pc,0,2),[1 nt]);

 %- remove seasonal cycle
 span = 120;	% monthly
 pc2 = zeros(neof,nt);
 for ieof = 1:neof
   trend = smooth(pc(ieof,:),span,'loess');
   pc2(ieof,:) = pc(ieof,:) - trend';
 end % for ieof

 %- power sepctrum -
 tmp_fft = zeros(neof,NFFT/2+1);
 for ieof = 1:neof
   tmp = fft(pc2(ieof,:));
   tmp_fft(ieof,:) = 2*abs(tmp(1:NFFT/2+1));
 end

 %- store -
 eof_fv(:,:,:,iYr) = eof;
 pc_fv(:,:,iYr) = pc;
 fft_pc_fv(:,:,iYr) = tmp_fft;
 expvar_fv(:,iYr) = expvar;
 
end % for iYr




Fs = 1/6;             %[hr-1]
NFFT = 2^nextpow2(nYr);
freq =  Fs/2*linspace(0,1,NFFT/2+1);
fft_pc = zeros(n_eof,length(freq));
for ieof = 1:n_eof
  tmp = fft(pc(ieof,:));
  fft_pc(ieof,:) = 2*abs(tmp(1:NFFT/2+1));
end





%-- variance of long time series vs repeated clim time series --
%umod_fv_std = std(umod_fv,0,3);
%umod_clim2_std = std(umod_clim_extd,0,3);
tmp = reshape(umod_fv,[nx ny nt nYr]);
umod_fv_std = mean(std(tmp,0,3),4);
umod_clim2_std = std(umod_clim,0,3);

%- choose |u|, u10 or v10 -
tmp_fv_std = umod_fv_std;
tmp_clim_std = umod_clim2_std;

%- plot -
myscale = [.5 5];
ticks_wanted = [.5:.1:5];
figure(60)
clf
set(gcf,'position',[50 400 1400 500])
m_proj('lambert','long',[min(lon(:))  max(lon(:))],...
                 'lat',[min(lat(:)) max(lat(:))]);
%- fully varying -
subplot(121)
[C,h1]=m_contourf(xC,yC,tmp_fv_std,...
    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['Fully varying winds '])
colorbar off
%- climatology -
subplot(122)
[C,h1]=m_contourf(xC,yC,tmp_clim_std,...
    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['Climatological winds '])
h = colorbar('location','southOutside');
h.Position = [.3 .13 .43 .02];
h.FontSize = 10;
h.Label.String = '\sigma [m s^{-1}]';
h.Label.FontSize = 12;
set(h,'YTick',ticks_wanted(1:5:end));
set(h,'YTickLabel',ticks_wanted(1:5:end));
%- ratio -
%subplot(133)
%[C,h1]=m_contourf(xLon,yLat,tmp_fv_std - tmp_clim_std,...
%    [myscale(1) ticks_wanted  myscale(2)]);
%set(h1,'lineStyle','none')
%colormap(mycmap1)
%h = colorbar('location','southOutside');
%caxis(myscale)
%%set(h,'YTick',ticks_wanted(1:5:end));
%%set(h,'YTickLabel',ticks_wanted(1:5:end));
%m_coast('patch',[.3 .3 .3]);
%m_grid('box','fancy','tickdir','in');
%title([{'C/ Difference A-B'},{tmpstr}])
%- save -
fileN60 = ['Map_std_fullyVaryingvsClim_umod.pdf'];
exportfig(figure(60),[dir_fig fileN60],...
    'width',10,'color','rgb','resolution',300);


%-- MEAN --
%- plot -
myscale = [1 11];
ticks_wanted = [1:.1:11];
figure(61)
clf
set(gcf,'position',[50 400 1400 500])
m_proj('lambert','long',[min(xLon(:))  max(xLon(:))],...
                 'lat',[min(yLat(:)) max(yLat(:))]);
%- fully varying -
subplot(121)
[C,h1]=m_contourf(xLon,yLat,tmp_fv_mean,...
    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['Fully varying winds -- ' tmpstr])
colorbar off
%- climatology -
subplot(122)
[C,h1]=m_contourf(xLon,yLat,tmp_clim_mean,...
    [myscale(1) ticks_wanted myscale(2)]);
set(h1,'lineStyle','none')
colormap(mycmap1)
caxis(myscale)
m_coast('patch',[.3 .3 .3]);
m_grid('box','fancy','tickdir','in');
title(['Climatological winds -- ' tmpstr])
h = colorbar('location','southOutside');
h.Position = [.3 .13 .43 .02];
h.FontSize = 10;
h.Label.String = '<.>_{t} [m s^{-1}]';
h.Label.FontSize = 12;
set(h,'YTick',ticks_wanted(1:6:end));
set(h,'YTickLabel',ticks_wanted(1:5:end));
%- save -
fileN61 = ['Map_mean_fullyVaryingvsClim_umod.pdf'];
exportfig(figure(61),[dir_fig fileN61],...
    'width',10,'color','rgb','resolution',300);


