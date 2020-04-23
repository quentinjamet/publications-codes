% DESCRIPTION
%	test the effect of both lowpass filter and detrending processes 
%	on a white noise 600 year long timeserie, 
%	that is then divided in 12 50-yr long timeseries

clear all; close all

dir_fig = '/tank/users/qjamet/Figures/publi/nature_amoc_rapid/';


%--------------------------------
% Construct the with noise signal
%--------------------------------


%-- time parameters --
fs = 1/5;		% sampling frequency	[d-1]
T = 1/fs;		% Sampling period	[d]
L = 600*73;		% Length of signal	[d]
tt = (1:L)*T;		% Time vector


%-- construct a white noise signal --
%ss = 0.05*sin(2*pi*.01*tt) + 0.08*sin(2*pi*0.002*tt);
ss = randn(size(tt));
ss = ss ./ std(ss);

%-- split into 12 50-yr long time series --
ss_12 = reshape(ss,[L/12 12]);
tt_12 = tt(1:L/12);
fs_yr = 1/365;		% sampling frequency    [d-1]

%-------------------------
% yearly mean
%-------------------------
ss_yr = squeeze(mean(reshape(ss_12, [73, 50, 12]), 1));
tt_yr = 0.5:1:49.5;

%-------------------------
% 50-yr cut-off high-pass filter 
%-------------------------

span=50*73;

ss_12_detrend = zeros(L/12,12);
trend_12 = zeros(L/12,12);
for imem = 1:12
  imem
  trend = smooth(tt_12,ss_12(:,imem),span,'loess');
  trend_12(:,imem) = trend;
  ss_12_detrend(:,imem) = ss_12(:,imem) - trend; 
end % for imem

span_yr = 50;
ss_yr_detrend = zeros(50, 12);
for imem = 1:12
  imem
  trend = smooth(tt_yr, ss_yr(:,imem), span_yr, 'loess');
  ss_yr_detrend(:, imem) = ss_yr(:,imem) - trend;
end

%-------------------------
% Low-pass filter the data 
%-------------------------
nt = 73*50;

%-- low-pass filtering at 365-d --
fc = 1/365;		% this is the low pass frequency [day-1]
Wn = (2/fs)*fc;
norder = 500;
b = fir1(norder,Wn,'low',chebwin(norder+1,30));
ss_12_lpf = zeros(L/12,12);
for imem = 1:12
 ss_12_lpf(:,imem) = filtfilt(b,1,ss_12_detrend(:,imem));
end




%-------------------------
%	Compute FFT
%-------------------------


%-- for the 12 50-yr data set (== ensemble) --
freq2 = fs*(0:(L/12/2))/(L/12);        % associated frequenct  [d-1]
nt_fft = length(freq2);

fft_ss_nondetrend = zeros(nt_fft,12);
fft_ss_detrend = zeros(nt_fft,12);
fft_ss_lpf = zeros(nt_fft,12);
for imem = 1:12

 %-- unfilter --
 tmp = fft(ss_12(:,imem));
 tmp2 = abs(tmp/(L/12));
 fft_ss_nondetrend(:,imem) = tmp2(1:(L/12)/2+1);
 fft_ss_nondetrend(2:end-1,imem) = 2*fft_ss_nondetrend(2:end-1,imem);

 %-- high-pass filtered --
 tmp = fft(ss_12_detrend(:,imem));
 tmp2 = abs(tmp/(L/12));
 fft_ss_detrend(:,imem) = tmp2(1:(L/12)/2+1);
 fft_ss_detrend(2:end-1,imem) = 2*fft_ss_detrend(2:end-1,imem);
 
 %-- + low-pass filtered --
 tmp = fft(ss_12_lpf(:,imem));
 tmp2 = abs(tmp/(L/12));
 fft_ss_lpf(:,imem) = tmp2(1:(L/12)/2+1);
 fft_ss_lpf(2:end-1,imem) = 2*fft_ss_lpf(2:end-1,imem);

end % for imem


%-- for yearly averaged data --
freq_yr = fs_yr*(0:(50/2))/(50);
nfft_yr = length(freq_yr);

fft_ss_yr_nondetrend = zeros(nfft_yr,12);
fft_ss_yr_detrend = zeros(nfft_yr,12);
for imem = 1:12
 %- undetrended -
 tmp = fft(ss_yr(:,imem));
 tmp2 = abs(tmp/50);
 fft_ss_yr_nondetrend(:,imem) = tmp2(1:50/2+1);
 fft_ss_yr_nondetrend(2:end-1,imem) = 2*fft_ss_yr_nondetrend(2:end-1,imem);

 %- detrended -
 tmp = fft(ss_yr_detrend(:,imem));
 tmp2 = abs(tmp/50);
 fft_ss_yr_detrend(:,imem) = tmp2(1:50/2+1);
 fft_ss_yr_detrend(2:end-1,imem) = 2*fft_ss_yr_detrend(2:end-1,imem);
end

%-------------------------
%	PLOT
%-------------------------


%-- periodogram for the detrended data --
figure(10)
clf
set(gcf,'position',[40 400 800 400])
%-- unfiltered data --
semilogx(1./freq2./365,mean(fft_ss_nondetrend,2),'b','lineW',1.2)
hold on
%-- high-pass filtered --
semilogx(1./freq2./365,mean(fft_ss_detrend,2),'r')
%-- + low-pass filtered --
semilogx(1./freq2./365,mean(fft_ss_lpf,2),'g','lineW',0.8)
%-- yearly mean --
semilogx(1./freq_yr./365,mean(fft_ss_yr_nondetrend,2),'k','lineW',0.8)
%-- detrended yearly mean --
semilogx(1./freq_yr./365,mean(fft_ss_yr_detrend,2),'m','lineW',0.8)
%-- labels/legend --
set(gca,'Xdir','reverse','xLim',[1/freq2(end) 1/freq2(2)]./365)
hl=legend('unfiltered data','50-yr high-pass filter',...
	'1-yr low-pass filter','location','South');
set(hl,'position',[0.2281    0.2208    0.2087    0.1250])
grid on
%set(gca,'yLim',[6e-3 5e-2],'yTick',[0.01:0.01:0.05])
xlabel('Period [yr]')
ylabel('Power Spectral Density')
%title('Effects of detrending at 50-yr with LOESS')
%
%fileN30 = '50yr_loess_detrending_white_noise_loglog';
fileN10 = '50yr_hpf_1yr_lpf_white_noise';
exportfig(figure(10),[dir_fig fileN10 '.pdf'],...
      'width',6,'color','rgb','resolution',300);
exportfig(figure(10),[dir_fig fileN10 '.png'],...
      'width',6,'color','rgb','resolution',300);





