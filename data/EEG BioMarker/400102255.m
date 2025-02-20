clear;
load 'C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\Preprocess\subject1'
Subject01 = transpose(table2array(subject1));
Subject01 = Subject01(1:19,:);
save Subject01.mat Subject01
figure
     EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\HAMAHANG\\Desktop\\sut\\S4\\SS\\PROJECT\\workplace\\Dataset\\Dataset\\Preprocess\\Subject01.mat','setname','imported01','srate',200,'pnts',0,'xmin',0);
     pop_eegplot( EEG, 1, 1, 1)
     EEG=pop_chanedit(EEG, []);
     EEG = pop_editset(EEG, 'chanlocs', 'C:\\Users\\HAMAHANG\\Desktop\\sut\\S4\\SS\\PROJECT\\workplace\\eeglab2023.0\\sample_locs\\Standard-10-20-Cap19.locs');
     EEG = pop_reref( EEG, []);
     EEG.setname='imported01_reref';
     EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',40.5,'plotfreqz',5);
     EEG.setname='imported01_reref_fltr';

step 2:
 EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
     figure; topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
     EEG = pop_iclabel(EEG, 'default');
     EEG = pop_subcomp( EEG, [1   2   3   4   5   7  18], 0);
     EEG.setname='imported01_reref_fltr_ICA_Purebrain';
     EEG = eeg_checkset( EEG );
%      pop_eegplot( EEG, 1, 1, 1);
a = EEG.data(1,1:240000);
b = reshape(a,[1 2000 120]);
for i=2:1:19
  a = EEG.data(i,1:240000);
  b = cat(1,b,reshape(a,[1 2000 120]));
end
b = b(:,1401:2000,:);
save Subject01_trials.mat b
EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\HAMAHANG\\Desktop\\sut\\S4\\SS\\PROJECT\\workplace\\eeglab2023.0\\Subject01_trials.mat','srate',200,'pnts',0,'xmin',0);
     EEG.setname='imported0101';
%      denoising excecution
     EEG.setname='imported0101_selected';
     EEG = eeg_checkset( EEG );
     EEG=pop_chanedit(EEG, []);
     EEG = pop_select( EEG, 'channel',{'Fp1','Fz','Cz','Pz'});
     EEG.setname='imported0101_selected_subbed';     
     EEG = eeg_checkset( EEG );
     pop_eegplot( EEG, 1, 1, 1);
     clear;
load 'C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\Preprocess\subject2'
Subject02 = transpose(table2array(subject2));
Subject02 = Subject02(1:19,:);
save Subject02.mat Subject02 
EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\HAMAHANG\\Desktop\\sut\\S4\\SS\\PROJECT\\workplace\\Subject02.mat','setname','imported02','srate',200,'pnts',0,'xmin',0);
     EEG=pop_chanedit(EEG, []);
     EEG = pop_select( EEG, 'rmtime',[0 16.4] );
     EEG.setname='imported02_strt';
     EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',40.5,'plotfreqz',1);
     EEG.setname='imported02_strt_fltr';
     EEG = pop_reref( EEG, []);
     EEG.setname='imported02_strt_fltr_reref';
     EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
     EEG = pop_iclabel(EEG, 'default');
     pop_eegplot( EEG, 1, 1, 1);
     EEG = pop_iclabel(EEG, 'default');
     EEG = pop_editset(EEG);
     EEG = pop_subcomp( EEG, [1   4   8  18  19], 0);
     EEG.setname='imported02_strt_fltr_reref_ICA_Purebrain';
     EEG = eeg_checkset( EEG );
     pop_eegplot( EEG, 1, 1, 1);
     EEG = pop_saveset( EEG, 'filename','imported02_reref_fltr_ICA_Purebrain.set','filepath','C:\\Users\\HAMAHANG\\Desktop\\sut\\S4\\SS\\PROJECT\\workplace\\');
     %      epoching
     a = EEG.data(1,1:240000);
b = reshape(a,[1 2000 120]);
for i=2:1:19
  a = EEG.data(i,1:240000);
  b = cat(1,b,reshape(a,[1 2000 120]));
end
b = b(:,1401:2000,:);
save Subject02_trials.mat b
%       deniosing
EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\HAMAHANG\\Desktop\\sut\\S4\\SS\\PROJECT\\workplace\\eeglab2023.0\\Subject02_trials.mat','srate',200,'pnts',0,'xmin',0);
     EEG.setname='imported0201';
%      denoising excecution
     EEG.setname='imported0201_selected';
     EEG = eeg_checkset( EEG );
     EEG=pop_chanedit(EEG, []);
     EEG = pop_select( EEG, 'channel',{'Fp1','Fz','Cz','Pz'});
     EEG.setname='imported0201_selected_subbed';     
     EEG = eeg_checkset( EEG );
     pop_eegplot( EEG, 1, 1, 1);
     % plv calculation for AD patients
load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\AD.mat
load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\Normal.mat
AD_PLV = zeros(13,2,1);
for i=1:13
    pat_tri=AD(i).epoch;
    pat_odr=AD(i).odor;
    pat_lem=find(~pat_odr);
    pat_ros=find(pat_odr); 
    pat_tri_lem=pat_tri(:,:,pat_lem);
    pat_tri_ros=pat_tri(:,:,pat_ros);
    for j=1:size(pat_lem,1)
fz = pat_tri_lem(2,:,j);
cz = pat_tri_lem(3,:,j);
AD_PLV(i,1,1)=AD_PLV(i,1,1)+calculatePLV(fz,cz,200,[35 40]);
    end
    AD_PLV(i,1,1) = AD_PLV(i,1,1)/size(pat_lem,1);    
    for j=1:size(pat_ros,1)
fz = pat_tri_ros(2,:,j);
cz = pat_tri_ros(3,:,j);
AD_PLV(i,2,1)=AD_PLV(i,2,1)+calculatePLV(fz,cz,200,[35 40]);
    end
    AD_PLV(i,2,1) = AD_PLV(i,2,1)/size(pat_ros,1);
end
% plv calculation for normal patients
Nrm_PLV = zeros(15,2,1);
for i=1:15
    pat_tri=normal(i).epoch;
    pat_odr=normal(i).odor;
    pat_lem=find(~pat_odr);
    pat_ros=find(pat_odr); 
    pat_tri_lem=pat_tri(:,:,pat_lem);
    pat_tri_ros=pat_tri(:,:,pat_ros);
    for j=1:size(pat_lem,1)
fz = pat_tri_lem(2,:,j);
cz = pat_tri_lem(3,:,j);
Nrm_PLV(i,1,1)=Nrm_PLV(i,1,1)+calculatePLV(fz,cz,200,[35 40]);
    end
    Nrm_PLV(i,1,1) = Nrm_PLV(i,1,1)/size(pat_lem,1);    
    for j=1:size(pat_ros,1)
fz = pat_tri_ros(2,:,j);
cz = pat_tri_ros(3,:,j);
Nrm_PLV(i,2,1)=Nrm_PLV(i,2,1)+calculatePLV(fz,cz,200,[35 40]);
    end
    Nrm_PLV(i,2,1) = Nrm_PLV(i,2,1)/size(pat_ros,1);
end
save AD_PLV.mat AD_PLV
save Nrm_PLV.mat Nrm_PLV
figure
load AD_PLV.mat
load Nrm_PLV.mat
subplot(1,2,1)
boxplot(AD_PLV)
title('AD (rare=right)')
subplot(1,2,2)
boxplot(Nrm_PLV)
title('Normal (rare=right)')
signal = AD_PLV(:,1); 
figure
% Fit a normal distribution to the signal
    pd = fitdist(signal, 'Normal');
fprintf('mu = %f\nsigma = %f',pd.mu,pd.sigma);
    % Generate x-values for the plot based on the range of the signal
    x = linspace(min(signal), max(signal), 100);

    % Compute the probability density function (PDF) values for the fitted distribution
    y = pdf(pd, x);
    subplot(2,1,1)
    % Plot the signal histogram and the fitted normal distribution PDF
    histogram(signal, 'Normalization', 'pdf', 'BinMethod', 'auto');
    subplot(2,1,2)
    plot(x, y, 'r', 'LineWidth', 2);

    % Add labels  
    xlabel('AD-LEM-GAUSS');
    ylabel('Probability Density');

    % Adjust plot aesthetics if desired
    grid on;
    title('AD-LEM-GAUSS');
    signal = AD_PLV(:,2);
    % Fit a normal distribution to the signal
    pd = fitdist(signal, 'Normal');
    fprintf('mu = %f\nsigma = %f',pd.mu,pd.sigma);
figure
    % Generate x-values for the plot based on the range of the signal
    x = linspace(min(signal), max(signal), 100);

    % Compute the probability density function (PDF) values for the fitted distribution
    y = pdf(pd, x);
    subplot(2,1,1)
    % Plot the signal histogram and the fitted normal distribution PDF
    histogram(signal, 'Normalization', 'pdf', 'BinMethod', 'auto');
    subplot(2,1,2)
    plot(x, y, 'r', 'LineWidth', 2);
    title('AD-ROS-GAUSS');
    % Add labels  
    xlabel('AD-ROS-GAUSS');
    ylabel('Probability Density');

    % Adjust plot aesthetics if desired
    grid on;
    signal = Nrm_PLV(:,1);
    % Fit a normal distribution to the signal
    pd = fitdist(signal, 'Normal');
    fprintf('mu = %f\nsigma = %f',pd.mu,pd.sigma);
figure
    % Generate x-values for the plot based on the range of the signal
    x = linspace(min(signal), max(signal), 100);

    % Compute the probability density function (PDF) values for the fitted distribution
    y = pdf(pd, x);
    subplot(2,1,1)
    % Plot the signal histogram and the fitted normal distribution PDF
    histogram(signal, 'Normalization', 'pdf', 'BinMethod', 'auto');
    subplot(2,1,2)
    plot(x, y, 'r', 'LineWidth', 2);

    % Add labels  
    xlabel('Nrm-LEM-GAUSS');
    ylabel('Probability Density');

    % Adjust plot aesthetics if desired
    grid on;
    title('Nrm-LEM-GAUSS');
    signal = Nrm_PLV(:,2);
    % Fit a normal distribution to the signal
    pd = fitdist(signal, 'Normal');
    fprintf('mu = %f\nsigma = %f',pd.mu,pd.sigma);
figure
    % Generate x-values for the plot based on the range of the signal
    x = linspace(min(signal), max(signal), 100);

    % Compute the probability density function (PDF) values for the fitted distribution
    y = pdf(pd, x);
    subplot(2,1,1)
    % Plot the signal histogram and the fitted normal distribution PDF
    histogram(signal, 'Normalization', 'pdf', 'BinMethod', 'auto');
    subplot(2,1,2)
    plot(x, y, 'r', 'LineWidth', 2);

    % Add labels  
    xlabel('Nrm-ROS-GAUSS');
    ylabel('Probability Density');

    % Adjust plot aesthetics if desired
    grid on;
    title('Nrm-ROS-GAUSS');
      % Perform two-sample t-test for lemmon odor
[~, p] = ttest2(AD_PLV(:,1), Nrm_PLV(:,1));
% Compare p-value to significance level
significance_level = 0.05;
fprintf('lemmon p-value = %f',p)
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
    % Perform two-sample t-test for rose odor
[~, p] = ttest2(AD_PLV(:,2), Nrm_PLV(:,2));
% Compare p-value to significance level
significance_level = 0.05;
fprintf('rose p-value = %f',p)
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
% phase difference calculation for a random AD patient
load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\AD.mat
load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\Normal.mat
p = randi(13);
AD_Ph_diff_lem= 0;
AD_Ph_diff_ros= 0;
    pat_tri=AD(p).epoch;
    pat_odr=AD(p).odor;
    pat_lem=find(~pat_odr);
    pat_ros=find(pat_odr); 
    pat_tri_lem=pat_tri(:,:,pat_lem);
    pat_tri_ros=pat_tri(:,:,pat_ros);
    for j=1:size(pat_lem,1)
fz = pat_tri_lem(2,:,j);
cz = pat_tri_lem(3,:,j);
AD_Ph_diff_lem=AD_Ph_diff_lem+phaseCAL(fz,cz,200,[35 40]);
    end
    AD_Ph_diff_lem = AD_Ph_diff_lem/size(pat_lem,1);    
    for j=1:size(pat_ros,1)
fz = pat_tri_ros(2,:,j);
cz = pat_tri_ros(3,:,j);
AD_Ph_diff_ros=AD_Ph_diff_ros+phaseCAL(fz,cz,200,[35 40]);
    end
    AD_Ph_diff_ros = AD_Ph_diff_ros/size(pat_ros,1);
    figure 
    subplot(1,2,1)
polarhistogram(AD_Ph_diff_lem);
    subplot(1,2,2)
polarhistogram(AD_Ph_diff_ros);
    % phase difference calculation for a random Normal patient
    p = randi(15);
Nrm_Ph_diff_lem= 0;
Nrm_Ph_diff_ros= 0;
    pat_tri=normal(p).epoch;
    pat_odr=normal(p).odor;
    pat_lem=find(~pat_odr);
    pat_ros=find(pat_odr);
    pat_tri_lem=pat_tri(:,:,pat_lem);
    pat_tri_ros=pat_tri(:,:,pat_ros);
    for j=1:size(pat_lem,1)
fz = pat_tri_lem(2,:,j);
cz = pat_tri_lem(3,:,j);
Nrm_Ph_diff_lem=Nrm_Ph_diff_lem+phaseCAL(fz,cz,200,[35 40]);
    end
    Nrm_Ph_diff_lem = Nrm_Ph_diff_lem/size(pat_lem,1);    
    for j=1:size(pat_ros,1)
fz = pat_tri_ros(2,:,j);
cz = pat_tri_ros(3,:,j);
Nrm_Ph_diff_ros=Nrm_Ph_diff_ros+phaseCAL(fz,cz,200,[35 40]);
    end
    Nrm_Ph_diff_ros = Nrm_Ph_diff_ros/size(pat_ros,1);
    figure 
    subplot(1,2,1)
polarhistogram(Nrm_Ph_diff_lem);
    subplot(1,2,2)
polarhistogram(Nrm_Ph_diff_ros);
% phase difference calculation for all AD patients
AD_Ph_diff_lem_all= 0;
AD_Ph_diff_ros_all= 0;
for i=1:13
AD_Ph_diff_lem= 0;
AD_Ph_diff_ros= 0;
    pat_tri=AD(i).epoch;
    pat_odr=AD(i).odor;
    pat_lem=find(~pat_odr);
    pat_ros=find(pat_odr); 
    pat_tri_lem=pat_tri(:,:,pat_lem);
    pat_tri_ros=pat_tri(:,:,pat_ros);
    for j=1:size(pat_lem,1)
fz = pat_tri_lem(2,:,j);
cz = pat_tri_lem(3,:,j);
AD_Ph_diff_lem=AD_Ph_diff_lem+phaseCAL(fz,cz,200,[35 40]);
    end
    AD_Ph_diff_lem = AD_Ph_diff_lem/size(pat_lem,1);    
    for j=1:size(pat_ros,1)
fz = pat_tri_ros(2,:,j);
cz = pat_tri_ros(3,:,j);
AD_Ph_diff_ros=AD_Ph_diff_ros+phaseCAL(fz,cz,200,[35 40]);
    end
    AD_Ph_diff_ros = AD_Ph_diff_ros/size(pat_ros,1);
    AD_Ph_diff_lem_all = AD_Ph_diff_lem_all + AD_Ph_diff_lem;
    AD_Ph_diff_ros_all = AD_Ph_diff_ros_all +AD_Ph_diff_ros;
end
AD_Ph_diff_lem_all = AD_Ph_diff_lem_all/13;
AD_Ph_diff_ros_all = AD_Ph_diff_ros_all/13;
save AD_Ph_diff_lem_all.mat AD_Ph_diff_lem_all
save AD_Ph_diff_ros_all.mat AD_Ph_diff_ros_all
load AD_Ph_diff_lem_all.mat
load AD_Ph_diff_ros_all.mat
    figure 
    subplot(1,2,1)
polarhistogram(AD_Ph_diff_lem_all);
    subplot(1,2,2)
polarhistogram(AD_Ph_diff_ros_all);
% phase difference calculation for all normal patients
Nrm_Ph_diff_lem_all= 0;
Nrm_Ph_diff_ros_all= 0;
for i=1:15
Nrm_Ph_diff_lem= 0;
Nrm_Ph_diff_ros= 0;
    pat_tri=normal(i).epoch;
    pat_odr=normal(i).odor;
    pat_lem=find(~pat_odr);
    pat_ros=find(pat_odr); 
    pat_tri_lem=pat_tri(:,:,pat_lem);
    pat_tri_ros=pat_tri(:,:,pat_ros);
    for j=1:size(pat_lem,1)
fz = pat_tri_lem(2,:,j);
cz = pat_tri_lem(3,:,j);
Nrm_Ph_diff_lem=Nrm_Ph_diff_lem+phaseCAL(fz,cz,200,[35 40]);
    end
    Nrm_Ph_diff_lem = Nrm_Ph_diff_lem/size(pat_lem,1);    
    for j=1:size(pat_ros,1)
fz = pat_tri_ros(2,:,j);
cz = pat_tri_ros(3,:,j);
Nrm_Ph_diff_ros=Nrm_Ph_diff_ros+phaseCAL(fz,cz,200,[35 40]);
    end
    Nrm_Ph_diff_ros = Nrm_Ph_diff_ros/size(pat_ros,1);
    Nrm_Ph_diff_lem_all = Nrm_Ph_diff_lem_all + Nrm_Ph_diff_lem;
    Nrm_Ph_diff_ros_all = Nrm_Ph_diff_ros_all +Nrm_Ph_diff_ros;
end
Nrm_Ph_diff_lem_all = Nrm_Ph_diff_lem_all/15;
Nrm_Ph_diff_ros_all = Nrm_Ph_diff_ros_all/15;
save Nrm_Ph_diff_lem_all.mat Nrm_Ph_diff_lem_all
save Nrm_Ph_diff_ros_all.mat Nrm_Ph_diff_ros_all
load Nrm_Ph_diff_lem_all.mat
load Nrm_Ph_diff_ros_all.mat
    figure 
    subplot(1,2,1)
polarhistogram(Nrm_Ph_diff_lem_all);
    subplot(1,2,2)
polarhistogram(Nrm_Ph_diff_ros_all);
% load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\AD.mat
% load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\Normal.mat
% % case 1:Fz,Cz
% AD_PLV_CH1_CH2 = zeros(6,2);
% AD_PLV_CH1_CH2(1,[1 2]) = extendedPLV(AD,13,2,3); 
% Nrm_PLV_CH1_CH2 = zeros(6,2);
% Nrm_PLV_CH1_CH2(1,[1 2]) = extendedPLV(normal,15,2,3);
% % case 2:Fz,Fp1
% AD_PLV_CH1_CH2(2,[1 2]) = extendedPLV(AD,13,1,2);
% Nrm_PLV_CH1_CH2(2,[1 2]) = extendedPLV(normal,15,1,2);
% % case 3:Fz,Pz
% AD_PLV_CH1_CH2(3,[1 2]) = extendedPLV(AD,13,2,4); 
% Nrm_PLV_CH1_CH2(3,[1 2]) = extendedPLV(normal,15,2,4);
% % case 4:Fp1,Cz
% AD_PLV_CH1_CH2(4,[1 2]) = extendedPLV(AD,13,1,3); 
% Nrm_PLV_CH1_CH2(4,[1 2]) = extendedPLV(normal,15,1,3);
% % case 5:Fp1,Pz
% AD_PLV_CH1_CH2(5,[1 2]) = extendedPLV(AD,13,1,4); 
% Nrm_PLV_CH1_CH2(5,[1 2]) = extendedPLV(normal,15,1,4);
% % case 6:Pz,Cz
% AD_PLV_CH1_CH2(6,[1 2]) = extendedPLV(AD,13,3,4); 
% Nrm_PLV_CH1_CH2(6,[1 2]) = extendedPLV(normal,15,3,4);
figure
load AD_PLV_CH1_CH2.mat
load Nrm_PLV_CH1_CH2.mat
cdata = cat(2,AD_PLV_CH1_CH2,Nrm_PLV_CH1_CH2);
xvalues = {'AD-LEM','AD-ROS','Nrm-LEM','Nrm-ROS'};
yvalues = {'Fz,Cz','Fz,Fp1','Fz,Pz','Fp1,Cz','Fp1,Pz','Cz,Pz'};
h = heatmap(xvalues,yvalues,cdata);
h.Title = 'PLV Heatmap';
h.XLabel = 'Odors-Patient';
h.YLabel = 'Channels';
% % p value lemmon ad
p = ttest(AD_PLV_CH1_CH2(:,1),AD_PLV_CH1_CH2(1,1));
if p == 0
    disp('Reject the null hypothesis.');
else
    disp('Fail to reject the null hypothesis.');
end
% % p value rose ad
p = ttest(AD_PLV_CH1_CH2(:,2),AD_PLV_CH1_CH2(1,2));
if p == 0
    disp('Reject the null hypothesis.');
else
    disp('Fail to reject the null hypothesis.');
end
% % p value lemmon nrm
p = ttest(Nrm_PLV_CH1_CH2(:,1),Nrm_PLV_CH1_CH2(1,1));
if p == 0
    disp('Reject the null hypothesis.');
else
    disp('Fail to reject the null hypothesis.');
end
% % p value rose nrm
p = ttest(Nrm_PLV_CH1_CH2(:,2),Nrm_PLV_CH1_CH2(1,2));
if p == 0
    disp('Reject the null hypothesis.');
else
    disp('Fail to reject the null hypothesis.');
end
% load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\MCI.mat
load AD_PLV.mat
load Nrm_PLV.mat
% MCI_PLV = zeros(7,2,1);
% for i=1:7
%     pat_tri=AD(i).epoch;
%     pat_odr=AD(i).odor;
%     pat_lem=find(~pat_odr);
%     pat_ros=find(pat_odr); 
%     pat_tri_lem=pat_tri(:,:,pat_lem);
%     pat_tri_ros=pat_tri(:,:,pat_ros);
%     for j=1:size(pat_lem,1)
% fz = pat_tri_lem(2,:,j);
% cz = pat_tri_lem(3,:,j);
% MCI_PLV(i,1,1)=MCI_PLV(i,1,1)+calculatePLV(fz,cz,200,[35 40]);
%     end
%     MCI_PLV(i,1,1) = MCI_PLV(i,1,1)/size(pat_lem,1);    
%     for j=1:size(pat_ros,1)
% fz = pat_tri_ros(2,:,j);
% cz = pat_tri_ros(3,:,j);
% MCI_PLV(i,2,1)=MCI_PLV(i,2,1)+calculatePLV(fz,cz,200,[35 40]);
%     end
%     MCI_PLV(i,2,1) = MCI_PLV(i,2,1)/size(pat_ros,1);
% end
% save MCI_PLV.mat MCI_PLV
load MCI_PLV.mat 
figure
subplot(1,3,1)
boxplot(AD_PLV)
title('AD (rare=right)')
subplot(1,3,2)
boxplot(Nrm_PLV)
title('Normal (rare=right)')
subplot(1,3,3)
boxplot(MCI_PLV)
title('MCI (rare=right)')
    % Perform two-sample t-test for lemmon odor between Normal and MCI
[~, p] = ttest2(Nrm_PLV(:,1), MCI_PLV(:,1));
% Compare p-value to significance level
significance_level = 0.05;
fprintf('rose p-value = %f',p)
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
    % Perform two-sample t-test for rose odor between Normal and MCI
[~, p] = ttest2(Nrm_PLV(:,2), MCI_PLV(:,2));
% Compare p-value to significance level
significance_level = 0.05;
fprintf('rose p-value = %f',p)
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
    % Perform two-sample t-test for lemmon odor between AD and MCI
[~, p] = ttest2(MCI_PLV(:,1),AD_PLV(:,2));
% Compare p-value to significance level
significance_level = 0.05;
fprintf('rose p-value = %f',p)
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
    % Perform two-sample t-test for rose odor between AD and MCI
[~, p] = ttest2( MCI_PLV(:,2),AD_PLV(:,2));
% Compare p-value to significance level
significance_level = 0.05;
fprintf('rose p-value = %f',p)
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
significance_level = 0.05;
lem_all = cat(2,transpose(AD_PLV(:,1)),transpose(Nrm_PLV(:,1)),transpose(MCI_PLV(:,1)));
% Perform Kruskal-Wallis test for lemmon
[p, tbl, stats] = kruskalwallis(lem_all, [], 'on');  % 'on' disables display
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
ros_all = cat(2,transpose(AD_PLV(:,2)),transpose(Nrm_PLV(:,2)),transpose(MCI_PLV(:,2)));
% Perform Kruskal-Wallis test for rose 
[p, tbl, stats] = kruskalwallis(ros_all, [], 'on');  % 'on' disables display
if p < significance_level
    disp('Reject the null hypothesis. There is a significant difference between the means.');
else
    disp('Fail to reject the null hypothesis. There is no significant difference between the means.');
end
load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\AD.mat
load C:\Users\HAMAHANG\Desktop\sut\S4\SS\PROJECT\workplace\Dataset\Dataset\Normal.mat
% MI calculation for AD Patients
% AD_MI = zeros(13,2,1);
% for i=1:13
%     pat_tri=AD(i).epoch;
%     pat_odr=AD(i).odor;
%     pat_lem=find(~pat_odr);
%     pat_ros=find(pat_odr); 
%     pat_tri_lem=pat_tri(:,:,pat_lem);
%     pat_tri_ros=pat_tri(:,:,pat_ros);
%     for j=1:size(pat_lem,1)
% fz = pat_tri_lem(2,:,j);
% cz = pat_tri_lem(3,:,j);
% AD_MI(i,1,1)=AD_MI(i,1,1)+MIcal(cz,fz);
%     end
%     AD_MI(i,1,1) = AD_MI(i,1,1)/size(pat_lem,1);    
%     for j=1:size(pat_ros,1)
% fz = pat_tri_ros(2,:,j);
% cz = pat_tri_ros(3,:,j);
% AD_MI(i,2,1)=AD_MI(i,2,1)+MIcal(cz,fz);
%     end
%  AD_MI(i,2,1) = AD_MI(i,2,1)/size(pat_ros,1);
% end
% % MI calculation for normal patients
% Nrm_MI = zeros(15,2,1);
% for i=1:15
%     pat_tri=normal(i).epoch;
%     pat_odr=normal(i).odor;
%     pat_lem=find(~pat_odr);
%     pat_ros=find(pat_odr); 
%     pat_tri_lem=pat_tri(:,:,pat_lem);
%     pat_tri_ros=pat_tri(:,:,pat_ros);
%     for j=1:size(pat_lem,1)
% fz = pat_tri_lem(2,:,j);
% cz = pat_tri_lem(3,:,j);
% Nrm_MI(i,1,1)=Nrm_MI(i,1,1)+MIcal(fz,cz);
%     end
%     Nrm_MI(i,1,1) = Nrm_MI(i,1,1)/size(pat_lem,1);    
%     for j=1:size(pat_ros,1)
% fz = pat_tri_ros(2,:,j);
% cz = pat_tri_ros(3,:,j);
% Nrm_MI(i,2,1)=Nrm_MI(i,2,1)+MIcal(fz,cz);
%     end
%     Nrm_MI(i,2,1) = Nrm_MI(i,2,1)/size(pat_ros,1);
% end
% save AD_MI.mat AD_MI
% save Nrm_MI.mat Nrm_MI
figure
load AD_PLV.mat
load Nrm_PLV.mat
subplot(1,2,1)
boxplot(AD_MI)
title('AD (rare=right)')
subplot(1,2,2)
boxplot(Nrm_MI)
title('Normal (rare=right)')
% MI Calculator
function MI = MIcal(data1,data2)
% Set the number of phase bins for the modulator signal
numBins = 12;  % Adjust the number of bins as desired
data1 = transpose(abs(data1));
data2 = transpose((data2));
% Compute the phase angles of the modulator signal
phase = angle(data1);  % Assuming data1 is the modulator signal

% Compute the instantaneous amplitude of the carrier signal
amplitude = abs(data2);  % Assuming data2 is the carrier signal

% Compute the Modulation Index (MI)
edges = linspace(-pi, pi, numBins+1);  % Define edges of the phase bins
binIndices = discretize(phase, edges);  % Assign each phase angle to a bin
binAmp = accumarray(binIndices, amplitude, [], @mean);  % Compute mean amplitude in each bin
MI = mean(binAmp) - 1;  % Compute the Modulation Index
end
% Extended PLV
function [PLV_ch1_ch2_lem_all,PLV_ch1_ch2_ros_all] = extendedPLV(kind,t,ch1_num,ch2_num)

PLV_ch1_ch2_lem_all= 0;
PLV_ch1_ch2_ros_all= 0;
for i=1:t
    PLV_ch1_ch2_lem = 0;
    PLV_ch1_ch2_ros = 0;
    pat_tri=kind(i).epoch;
    pat_odr=kind(i).odor;
    pat_lem=find(~pat_odr);
    pat_ros=find(pat_odr); 
    pat_tri_lem=pat_tri(:,:,pat_lem);
    pat_tri_ros=pat_tri(:,:,pat_ros);
    for j=1:size(pat_lem,1)
ch1 = pat_tri_lem(ch1_num,:,j);
ch2 = pat_tri_lem(ch2_num,:,j);
PLV_ch1_ch2_lem=PLV_ch1_ch2_lem+calculatePLV(ch1,ch2,200,[35 40]);
    end
    PLV_ch1_ch2_lem = PLV_ch1_ch2_lem/size(pat_lem,1);    
    for j=1:size(pat_ros,1)
ch1 = pat_tri_ros(ch1_num,:,j);
ch2 = pat_tri_ros(ch2_num,:,j);
PLV_ch1_ch2_ros=PLV_ch1_ch2_ros+calculatePLV(ch1,ch2,200,[35 40]);
    end
    PLV_ch1_ch2_ros = PLV_ch1_ch2_ros/size(pat_ros,1);
    PLV_ch1_ch2_lem_all= PLV_ch1_ch2_lem_all + PLV_ch1_ch2_lem;
    PLV_ch1_ch2_ros_all = PLV_ch1_ch2_ros_all + PLV_ch1_ch2_ros;
end
PLV_ch1_ch2_lem_all = PLV_ch1_ch2_lem_all/t;
PLV_ch1_ch2_ros_all = PLV_ch1_ch2_ros_all/t;
end
% Applied function for Phase difference calculation
function phi = phaseCAL(signal1, signal2, samplingRate, frequencyRange)

%     Apply bandpass filter to extract desired frequency range
    bpFilt = designfilt('bandpassiir','FilterOrder',6,'HalfPowerFrequency1',frequencyRange(1), 'HalfPowerFrequency2',frequencyRange(2),'SampleRate',samplingRate);
    filteredSignal1 = filtfilt(bpFilt, double(signal1));
    filteredSignal2 = filtfilt(bpFilt, double(signal2));

%     Calculate instantaneous phase of each signal
    phase1 = angle(hilbert(filteredSignal1));
    phase2 = angle(hilbert(filteredSignal2));

%     Calculate Phase Difference
    phi = phase1 - phase2;
end
% Applied function for PLV
function plv = calculatePLV(signal1, signal2, samplingRate, frequencyRange)

%     Apply bandpass filter to extract desired frequency range
    bpFilt = designfilt('bandpassiir','FilterOrder',6,'HalfPowerFrequency1',frequencyRange(1), 'HalfPowerFrequency2',frequencyRange(2),'SampleRate',samplingRate);
    filteredSignal1 = filtfilt(bpFilt, double(signal1));
    filteredSignal2 = filtfilt(bpFilt, double(signal2));

%     Calculate instantaneous phase of each signal
    phase1 = angle(hilbert(filteredSignal1));
    phase2 = angle(hilbert(filteredSignal2));

%     Calculate Phase Difference
    phaseDiff = phase1 - phase2;

%     Calculate PLV
    plv = abs(mean(exp(1i * phaseDiff)));

end
