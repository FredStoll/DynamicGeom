%% main_000_create_dataset.m
%
% Build pooled session files (*_pool.mat) in data_final from raw *_spk.mat files.
% - Input: raw spiking dataset folder specified by pathspk.
% - Output: data_final/*_pool.mat used by downstream main scripts.
% - Note: pathspk is machine-specific and should be edited locally.

clear

overwrite = true; % if you want to redo the matrices

f = mfilename('fullpath');
if isempty(f) 
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end
addpath(genpath([currentPath '\scripts\'])); % add scripts folder to path
pathspk = [currentPath '\data_final\'];
list = dir([pathspk '*_spk.mat']);

% reorder files by dates 
for i = 1 : length(list)
    dates(i) = datenum(list(i).name(2:7),'mmddyy');
end
[~,idx] = sort(dates);  
list = list(idx);

% set parameters
param.min_nbTr = 50;
%param.task = {'1FC' '2AFC'}; % '1FC';
param.cond = {'task' 'proba_1FC' 'flavor_1FC' 'side_1FC' 'chosenproba_2AFC' 'unchosenproba_2AFC' 'chosenflavor_2AFC' 'unchosenflavor_2AFC' 'chosenside_2AFC'};
param.evt = {'stim_on' 'resp_fix' 'fb_on' 'rew_on'};
param.pre = [1250 750 750 750]; % pre-stimulus time (ms)
param.post = [1250 1250 750 1000]; % post-stimulus time (ms)
param.binsize = 100; % bins size (ms)
param.step = 10; % step size (ms)

param.bins =[];
for i = 1 : length(param.pre)
    param.bins = [param.bins , [-param.pre(i):param.step:param.post(i)-param.binsize ; i*ones(1,length(-param.pre(i):param.step:param.post(i)-param.binsize))]];
end

% load the files and extract the info
for sess = 1 : length(list)

    clearvars -except sess list pathspk area2test param anova_res nb lda_res nb_units session path2save overwrite

    if ~overwrite & exist([path2save list(sess).name(1:end-7) 'pool.mat'],'file')
        disp(['Session ' num2str(sess) ' already processed!'])
        continue
    end

    %- load spiking and behav data for that session
    disp(['Processing session ' num2str(sess) ' of ' num2str(length(list)) '...'])
    spk = load([pathspk list(sess).name]);

    %- take the trials 2 consider and variable of interest
    for cd = 1 : length(param.cond)
        idx_cond{cd} = ismember(spk.behav.trialtype_header,param.cond{cd});
    end
    completed_tr = spk.behav.trialtype(:,ismember(spk.behav.trialtype_header,'brk'))==0 & ismember(spk.behav.trialtype(:,ismember(spk.behav.trialtype_header,'task')),[1 2]);

    %- condition and timestamp of event
    cond = [];
    for cd = 1 : length(param.cond)

        if ismember(param.cond(cd),'unchosenflavor_2AFC') %- this doesn't exist so need to be inferred from other variables!
            side_temp = spk.behav.trialtype(:,ismember(spk.behav.trialtype_header,'chosenside_2AFC'));
            rmv = isnan(side_temp);
            side_temp_inv = (side_temp==1)+1;
            side_temp_inv(rmv)=NaN;
            unchosenflav = cell(size(side_temp_inv));
            unchosenflav(side_temp_inv==1)={'flavorL_2AFC'};
            unchosenflav(side_temp_inv==2)={'flavorR_2AFC'};
            unchosenflavor = NaN(size(side_temp_inv));
            for tt = 1 : size(spk.behav.trialtype)
                if ~isempty(unchosenflav{tt})
                    unchosenflavor(tt) = spk.behav.trialtype(tt,ismember(spk.behav.trialtype_header,unchosenflav{tt}));  
                end
            end
            cond(:,cd) = unchosenflavor;

        else
            cond(:,cd) = spk.behav.trialtype(:,idx_cond{cd});
        end
    end
    evt_time = [];
    for e = 1 : length(param.evt)
        evt_time(:,e) = spk.behav.t_evt.(param.evt{e})(:);
    end

    %- remove unwanted probas
    if sum(ismember(param.cond,'proba_1FC'))~=0 
        idx_tr1 = ~ismember(cond(:,ismember(param.cond,'proba_1FC')),[10 30 50 70 90]) & ismember(cond(:,ismember(param.cond,'task')),1)   ;
    else
        idx_tr1 = zeros(size(cond,1),1);
    end
    if sum(ismember(param.cond,'chosenproba_2AFC'))~=0 
        idx_tr2 = ~ismember(cond(:,ismember(param.cond,'chosenproba_2AFC')),[30 50 70 90]) & ismember(cond(:,ismember(param.cond,'task')),2)   ;
    else
        idx_tr2 = zeros(size(cond,1),1);
    end
    if sum(ismember(param.cond,'unchosenproba_2AFC'))~=0 
        idx_tr3 = ~ismember(cond(:,ismember(param.cond,'unchosenproba_2AFC')),[10 30 50 70]) & ismember(cond(:,ismember(param.cond,'task')),2)   ;
    else
        idx_tr3 = zeros(size(cond,1),1);
    end
    idx_tr = idx_tr1 | idx_tr2 | idx_tr3 | ~completed_tr;
    cond(idx_tr,:)=[];
    evt_time(idx_tr,:)=[];

    trial_idx = find(idx_tr==0);

    %-  only process session if enough trials 
    if sum(cond(:,1)==1)<param.min_nbTr | sum(cond(:,1)==2)<param.min_nbTr
        disp('      ... not enough trials!')
    continue   
    end

    %- look at neurons now
    all_psth = NaN(size(spk.unit,2),size(evt_time,1),size(param.bins,2));
    info = table();
    for u = 1 : size(spk.unit,2)
        disp(['    ---> Processing unit ' num2str(u) ' of ' num2str(size(spk.unit,2)) '...'])

        for e = 1 : length(param.evt)

            %- make PSTH around event of interest
            psth_trials = zeros(param.pre(e)+param.post(e)+1,length(evt_time(:,e))); % will be a matrix of size nTR x nTIME 
            trialspx = cell(numel(evt_time(:,e)),1);
            for tr = 1:numel(evt_time(:,e)) %- for each trial
                clear spikes
                spikes = (spk.unit{u}.timestamps - evt_time(tr,e))*1000;         % all spikes relative to current trigtime
                trialspx{tr} = round(spikes(spikes>=-param.pre(e) & spikes<=param.post(e)));   % spikes close to current trigtime
                psth_trials(trialspx{tr}+param.pre(e)+1,tr) = psth_trials(trialspx{tr}+param.pre(e)+1,tr)+1; % add a 1 when spike
            end
            time = -param.pre(e):param.post(e); %- time vector

            % bin psth
            idxs = find(param.bins(2,:)==e);
            bins_sub = param.bins(1,idxs);
            psth_bin = NaN(size(psth_trials,2),length(bins_sub)-1);
            for b = 1 : length(bins_sub)-1
                idx = time>=bins_sub(b) & time<bins_sub(b)+param.binsize ;
                psth_bin(:,b) = sum(psth_trials(idx,:),1) / (param.binsize / 1000) ; % convert to Hz
            end
    
            all_psth(u,:,idxs(1:end-1)) = psth_bin;
        end

        info = [info ; table({spk.unit{u}.session},{spk.unit{u}.ch},spk.unit{u}.clust_id,{spk.unit{u}.area},'VariableNames',{'session' 'ch' 'clust_id' 'area'})];

    end
    spkpool = single(all_psth); clear all_psth
    
    % make cond into a table
    % cond = array2table(cond,'VariableNames',param.cond);
    cond = array2table([trial_idx cond],'VariableNames',[{'trial'} param.cond]);

    % save the different variables (1 per session)
    save([pathspk list(sess).name(1:end-7) 'pool.mat'],'spkpool','info','cond','param','-v7.3') 

end

%% convert EOG data to session-level mat files
% This block was used to extract raw eye-movement data from .pl2 recordings.
% Paths below are machine-specific and will need updating before use.
% Left commented out because EOG files are already extracted for this dataset.

% clear

% path4data = 'Q:\MIMIC_POTT2_RAW_part2\'; % path to the folder containing the .pl2 files
% % 'P:\MORBIER_POTT_RAW\' 'P:\MIMIC_POTT_RAW\'  'P:\MIMIC_POTT2_RAW_part1\'  'Q:\MIMIC_POTT2_RAW_part2\'
% path2save = 'I:\Morbier_EOG\' ; % path to the folder where the EOG data will be saved

% list = dir([path4data '*a.pl2']);

% for s = 1 : length(list)

%     filename = [path4data list(s).name];
%     disp([num2str(length(list)-s) ' files left...'])

%     if exist([path2save list(s).name(1:8) '_EOG.mat'],'file') == 2
%         disp('File already processed')
%         continue
%     end

%     eog = [];
%     for ch = 1 : 4
%         [adfreq, n, ts, fn, ad] = plx_ad_v(filename, ['AI0' num2str(ch)]);
%         eog(:,ch) = single(ad);
%         clear ad
%     end
%     FS_eog = adfreq;
%     save([path2save list(s).name(1:8) '_EOG.mat'],'eog','FS_eog')

%     clear eog

% end