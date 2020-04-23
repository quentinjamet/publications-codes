% DESCRIPTION
%
% look at the best period of the year to make the connection for the 
% 'neutral year' definition
%
% For the NA, the best period to loop for the neutral year is between 
% July and August (lowest variance over years found for the day 210, 
% i.e. 07/28-29), not the April-May as suggested by Handy Hogg for global runs
% found day 200.5 (07/18) with the global averaged 2003/2004 std
%
% This as been obtained by spacially averaging the std of winds (modulus) 
% over for both years 2003 and 2004 for each time-records (1460 in the 6-hourly data case), 
% and the variance is highly dominated by the >30N region, 
% where the variance of winds is found to be high

clear all; close all;

%-- directories and grid --
dir_atm = '/tank/chaocean/atmospheric_data/DFS4.4_NorthAtl/';
dir_fig = '/tank/users/qjamet/Figures/publi/note_bams/';

%-- grid parameters --
% u and v are on the same t-pts grid
tmpx = double(ncread([dir_atm 'u10_DFS4.4_y1958_chaO.nc'],'lon'));
tmpy = double(ncread([dir_atm 'u10_DFS4.4_y1958_chaO.nc'],'lat'));
[nx] = size(tmpx,1);
[ny] = size(tmpy,1);
for ii =1:nx
  if (tmpx(ii)> 180); tmpx(ii) = tmpx(ii) - 360; end
end
[yLat,xLon]=meshgrid(tmpy,tmpx);
%- select >30N region for seasonal std -
jLat_30N = find(tmpy > 30);
[ny_30N] = size(jLat_30N,1);

%- cell face -
rEarth = 6370000; %[m]
xC = zeros(nx+2,ny);
xC(2:nx+1,:) = xLon;
xC(1,:) = xLon(1,:) - (xLon(2,1) - xLon(1,1));
xC(nx+2,:) = xLon(nx,:) + (xLon(nx,1) - xLon(nx-1,1));
xG = (xC(2:nx+2,:) + xC(1:nx+1,:))./2;
yC = zeros(nx,ny+2);
yC(:,2:ny+1) = yLat;
yC(:,1) = yLat(:,1) - (yLat(:,2) - yLat(:,1));
yC(:,ny+2) = yLat(:,ny) - (yLat(:,ny) - yLat(:,ny-1));
yG = (yC(:,2:ny+2) + yC(:,1:ny+1)) ./ 2;
dxG = deg2rad(xG(2:nx+1,:) - xG(1:nx,:)) .* rEarth .* ...
      cosd(yC(:,2:ny+1));
dyG =  deg2rad(yG(:,2:ny+1) - yG(:,1:ny)) .* rEarth;
rA = dxG .* dyG;


%-- time parameters --
yYear = 2003:2004;
nYr = length(yYear);
ttime = double(ncread([dir_atm 'u10_DFS4.4_y1958_chaO.nc'],'time'));
[nt] = length(ttime);

uv_mod_2003_04 = zeros(nx,ny,nt,2);

for iYr = 1:nYr
 tmp_u = ncread([dir_atm 'u10_DFS4.4_y' num2str(yYear(iYr)) '_chaO.nc'],'u10');
 tmp_v = ncread([dir_atm 'v10_DFS4.4_y' num2str(yYear(iYr)) '_chaO.nc'],'v10');
 uv_mod_2003_04(:,:,:,iYr) = sqrt( tmp_u.^2 + tmp_v.^2 );
end % for iYr

%diff_yr = ( uv_mod_2003_04(:,:,:,1)-uv_mod_2003_04(:,:,:,2) ).^2;
% should define that as as a variance
diff_yr = std(uv_mod_2003_04,0,4);
%nrecord = zeros(nx,ny);
%for jj = 1:ny
% for ii = 1:nx
%  tmp = find( diff_yr(ii,jj,:) == min(diff_yr(ii,jj,:)) );
%  nrecord(ii,jj) = tmp(1);
% end
%end

seas_std = squeeze(sum(sum(diff_yr.*repmat(rA,[1 1 nt]),1),2) ./ ...
       sum(rA(:)));
filt = 30*4;
seas_std_lpf = smooth(seas_std,filt);
%seas_std_lpf = seas_std;
[ii] = find(seas_std_lpf == min(seas_std_lpf));



%---------------------
%	PLOT
%---------------------

figure(10)
clf
set(gcf,'position',[40 500 1000 400])
hold on
%plot(ttime,seas_std,'color',[.8 .8 .8])
plot(ttime(filt/2:nt-filt/2),seas_std_lpf(filt/2:nt-filt/2),'color',[0 0 0])
plot(ttime(ii),seas_std_lpf(ii),'r*')
grid on
set(gca,'xLim',[0 ttime(nt)])
%xlabel('Days of the year')
ylabel('\sigma [m.s^{-1}]')
title([ {'Variance of |u| over years, averaged in the region 30-55N'} ; ...
        {[ num2str(yYear(1)) '-' num2str(yYear(nYr))]} ])
set(gca,'xTick',cumsum([0 31 28 31 30 31 30 31 31 30 31 30]));
set(gca,'xTickLabel',[{'Jan'} {'Feb'} {'Mar'} {'Apl'} {'May'} {'Jun'} {'Jul'} ...
                      {'Aug'} {'Sep'} {'Oct'} {'Nov'} {'Dec'}]);
%text(ttime(ii)-30,2.6,['min at day ' num2str(ttime(ii)) ])
%save
fileN10 = ['winds_std_2003_2004.pdf'];
exportfig(figure(10),[dir_fig fileN10],...
      'width',6,'color','rgb','resolution',300);

