
%% main_001_anova_lda.m
%
% Single-neuron ANOVA for probability, flavor, and side encoding;
% population LDA decoding timecourses and cross-subspace analysis.
% make Figures S1 and S2 in Stoll, Valluru & Rudebeck (2026)

clear


f = mfilename('fullpath');
if isempty(f) 
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end
addpath(genpath([currentPath '\scripts\'])); % add scripts folder to path
pathspk = [currentPath '\data_final\'];
pathout  = [currentPath '\processed\'];
if ~exist(pathout, 'dir'), mkdir(pathout); end

report_dir = [currentPath '\report_001\'];
if ~exist(report_dir, 'dir'), mkdir(report_dir); end
log_file = [report_dir 'output_001.txt'];
fid_log = fopen(log_file, 'w');
log_cleanup = onCleanup(@() fclose(fid_log));  % flush/close even on error
utils_diary(fid_log, '\n============================================================\n');
utils_diary(fid_log, 'Report generated: %s\n', datestr(now));
warning off

overwrite = false;
if exist([pathout 'anova_lda.mat'],'file') && ~overwrite
    disp('ANOVA results already computed, loading...');
    skip = true;
else 
    skip = false;
end

if ~skip


    list = dir([pathspk '*_pool.mat']);
    if isempty(list)
        error('No *_pool.mat files found in %s', pathspk);
    end
    % reorder files by dates 
    dates = zeros(1, length(list));
    for i = 1 : length(list)
        dates(i) = datenum(list(i).name(2:7),'mmddyy');
    end
    [~,idx] = sort(dates);  
    list = list(idx);

    conds = {{'proba_1FC' 'flavor_1FC' 'side_1FC'} ; {'chosenproba_2AFC' 'unchosenproba_2AFC' 'chosenflavor_2AFC' 'chosenside_2AFC'}}; 

    area2test =      {'24c' {'6DR' '6DC' '6Va/Vb'}  {'8' '8B' '8A' '46d' '46df' '46v'}    {'IFG' '44' '45'} {'12r' '12m' '12m/r' '12o' '12l'}   'AI'  {'13l' '13m' '11m/l'} {'cd'  'pu'}  'AMG'};
    area2test_name = {'MFC' 'PMC' 'dlPFC'  'IFG'       'vlPFC'  'AI'  'OFC'    'STR' 'AMG'};
    area_group =     {'MFC' 'PMC' 'dlPFC'  'IFG'       'vlPFC'  'AI'  'OFC'    'STR' 'AMG'};

    % for LDA
    %param.baseline_norm = [-900 -100]; % to normalize raw firing rate (independently for 1FC and 2AFC)
    param.min_firing_threshold = .5;  % Define minimum firing rate threshold
    param.kfold = 10; % to assess cross-validation pav-pav and ins-ins perf. 
    param.cond = {[1 1];[2 1];[2 2]}; % [task param] from param4lda{task}(:,cond)
    param.cond_name = {'proba_1FC' 'chosenproba_2AFC' 'unchosenproba_2AFC'};
    param.bin4subspace_decoding = [200 700]; % in ms, used for decoding

    % define conditions ( note that I ignore unchosen probability here cos can't match it between tasks! )
    % note that ANOVA includes 5 levels for proba 1FC (10,30,50,70,90) and 4 for chosen proba 2AFC (30,50,70,90)/unchosen proba 2AFC (10,30,50,70)
    %- cross decoding is on 5 levels (10,30,50,70,90)

    all_pbs = [30 50 70 90];
    all_unch_pbs = [10 30 50 70];
    all_flavors = [1 2];
    all_sides = [1 2];
    all_conds = []; all_conds_unch = [];
    for p = 1 : length(all_pbs)
        for f = 1 : length(all_flavors)
            for s = 1 : length(all_sides)
                all_conds = [all_conds ; all_pbs(p) all_flavors(f) all_sides(s)];
                all_conds_unch = [all_conds_unch ; all_unch_pbs(p) all_flavors(f) all_sides(s)];
            end
        end
    end
    all_conds_both = [all_conds ; all_conds_unch];
    all_conds_both = unique(all_conds_both,'rows'); % only keep unique rows

    nb = zeros(length(area2test),1); %- for counting number of session per area
    info = table();
    lda_tab = table();
    all_fr = [];
    for sess = 1 : length(list)

        clearvars -except param sess list pathspk pathout report_dir anova_res conds info lda_perf lda_tab area2test area2test_name area_group all_fr all_conds all_conds_unch all_conds_both nb lda_res nb_units session fid_log log_cleanup

        %- load spiking and behav data for that session
        disp(['Processing session ' num2str(sess) ' of ' num2str(length(list)) '...'])
        spk = load([pathspk list(sess).name]);

        % anova on single neurons
        clear all_fr_norm fr4lda
        tr2take{1} = [];
        tr2take{2} = [];

        for u = 1 : height(spk.info)

            disp(['    ---> Processing unit ' num2str(u) ' of ' num2str(height(spk.info)) '...'])

            for t = 1 : size(conds,1) % for each task

                param2use =[];
                for c = 1 : length(conds{t})
                    param2use(c) = find(ismember(spk.cond.Properties.VariableNames,conds{t}(c)));
                end
                continuous_var = ismember(conds{t},{'proba_1FC' 'chosenproba_2AFC' 'unchosenproba_2AFC'}); %- tell if continuous variable

                keeptr = table2array(spk.cond(:,ismember(spk.cond.Properties.VariableNames,'task'))) == t;

                param_raw = table2array(spk.cond(keeptr,conds{t}));
                param_norm = param_raw;
                % express each param between 0 and 1
                for p = 1 : size(param_norm,2)
                    param_norm(:,p) = (param_norm(:,p) - min(param_norm(:,p))) / (max(param_norm(:,p)) - min(param_norm(:,p)));
                end

                if t==2 && u==1 %- randomize trial selection only once (so can use the same trials for all units in decoding)
                    nbTr_1fc = sum(spk.cond.task==1);
                    nbTr_2afc = sum(spk.cond.task==2);

                    if nbTr_1fc < nbTr_2afc
                    % here, we are matching the trial numbers across both tasks (all 1FC trials vs subset of 2AFC trials)
                    % do that by taking a similar number of trial PER CONDITION. If less 2AFC than 1FC, take the rest randomly from available 2afc trials
                        for cd = 1 : size(all_conds_both,1)
                            nCd_1FC = sum(spk.cond.task==1 & spk.cond.proba_1FC==all_conds_both(cd,1) & spk.cond.flavor_1FC==all_conds_both(cd,2) & spk.cond.side_1FC==all_conds_both(cd,3));
                            % find trials with similar profiles (for proba, can be either chosen or unchosen)
                            temp = find(spk.cond.task==2 & spk.cond.chosenflavor_2AFC==all_conds_both(cd,2) & spk.cond.chosenside_2AFC==all_conds_both(cd,3) & (spk.cond.chosenproba_2AFC==all_conds_both(cd,1) | spk.cond.unchosenproba_2AFC==all_conds_both(cd,1))) - nbTr_1fc;
                            nCd_2AFC = length(temp);
                            % disp([num2str(cd) ' : ' num2str(nCd_1FC) ' / ' num2str(nCd_2AFC)])
                            if nCd_2AFC>0 & nCd_2AFC<=nCd_1FC % not enough or same number (take them all)
                                tr2take{t} = [tr2take{t} ; temp];
                            elseif nCd_2AFC>nCd_1FC
                                idx_temp = randperm(nCd_2AFC,nCd_1FC);
                                tr2take{t} = [tr2take{t} ; temp(idx_temp)];
                            elseif nCd_2AFC==0
                                % disp(['missing ' num2str(cd)])
                            end

                        end
                        if length(tr2take{t})<nbTr_1fc % if less trial than 1FC, take the rest randomly from available 2afc trials
                            tr_id = 1:nbTr_2afc;
                            tr_id = tr_id(~ismember(tr_id,tr2take{t}));
                            tr2take{t} = [tr2take{t} ; tr_id(randperm(length(tr_id),nbTr_1fc-length(tr2take{t})))'];
                        end
                    else
                            tr2take{t} = (1:nbTr_2afc)';
                    end

                    tr2take{t} = sortrows(tr2take{t});
                    
                    spk_temp = squeeze(spk.spkpool(u,keeptr,:));
                    fr = spk_temp(tr2take{t},:); %- get the FR for that unit and trials to take
                    param_norm = param_norm(tr2take{t},:);

                elseif t==2 && u~=1

                    spk_temp = squeeze(spk.spkpool(u,keeptr,:));
                    fr = spk_temp(tr2take{t},:); %- get the FR for that unit and trials to take
                    param_norm = param_norm(tr2take{t},:);

                else
                    tr2take{t} = (1:sum(keeptr))';
                    fr = squeeze(spk.spkpool(u,keeptr,:));
                end

                %- min max norm
                fr_norm = (fr - min(fr(:))) / (max(fr(:)) - min(fr(:)));

                % fake anova to check the param order and get the coeffs
                if u == 1 
                    [p,tbl,stats,terms] = anovan(fr_norm(:,1),param_norm,'varnames',conds{t},'continuous',continuous_var,'display','off');
                    % get the coeff for each param (if categorical param, keep the first level as reference)
                    for p = 1 : size(param_norm,2)
                        if continuous_var(p)
                            coeffs{t}(p) = find(ismember(stats.coeffnames,conds{t}(p)));
                        else
                            coeffs{t}(p) = find(ismember(stats.coeffnames,[conds{t}{p} '=0']));
                        end
                    end
                end

                % ANOVA
                clear pval pevs omega2 r2 coeff
                parfor b = 1 : length(spk.param.bins)
                    if sum(isnan(fr_norm(:,b)))==0
                        [p,tbl,stats,terms] = anovan(fr_norm(:,b),param_norm,'varnames',conds{t},'continuous',continuous_var,'display','off');
                        pval(b,:) = single(p);
                        pevs(b,:) = single((([tbl{(1:size(param2use,2))+1,2}]/tbl{end,2}))*100);
                        omega2(b,:) = single(([tbl{(1:size(param2use,2))+1,2}]-[tbl{(1:size(param2use,2))+1,3}]*tbl{end-1,5})/(tbl{end-1,5}+tbl{end,2}));
                        r2(b,1) = single(1 - (tbl{end-1,2}/tbl{end,2}));
                        coeff(b,:) = single(stats.coeffs([1 coeffs{t}])); % keep the constant too

                    else
                        pval(b,:) = NaN(1,size(param_norm,2));
                        pevs(b,:) = NaN(1,size(param_norm,2));
                        omega2(b,:) = NaN(1,size(param_norm,2));
                        r2(b,1) = NaN;
                        coeff(b,:) = NaN(1,size(param_norm,2)+1);
                    end
                end

                anova_res(sess,t).pval(u,:,:) = pval;
                anova_res(sess,t).pevs(u,:,:) = pevs;
                anova_res(sess,t).omega2(u,:,:) = omega2;
                anova_res(sess,t).coeff(u,:,:) = coeff;
                anova_res(sess,t).r2(u,:,1) = r2; 
                anova_res(sess,t).trials(u,:) = single(tr2take{t});

                % average fr_norm for each condition (for projection of FR)
                for c = 1 : size(all_conds,1)
                    if t==1
                        idx = ismember(param_raw,all_conds(c,:),'rows');
                    else
                        idx = ismember(param_raw(tr2take{t},[1 3 4]),all_conds(c,:),'rows');
                        idx_unch = ismember(param_raw(tr2take{t},[2 3 4]),all_conds_unch(c,:),'rows');
                    end

                    if sum(idx)>0
                        all_fr_norm{t}(u,c,:) = single(nanmean(fr_norm(idx,:),1));
                    else
                        all_fr_norm{t}(u,c,:) = NaN(1,size(fr_norm,2));
                    end
                    if t==2 
                        if sum(idx_unch)>0
                            all_fr_norm{t+1}(u,c,:) = single(nanmean(fr_norm(idx_unch,:),1));
                        else
                            all_fr_norm{t+1}(u,c,:) = NaN(1,size(fr_norm,2));
                        end
                    end
                end

                fr4lda{t}(u,:,:) = fr;
                param4lda{t} = param_raw(tr2take{t},:);

            end
        end

        all_fr = cat(1,all_fr,cat(2,all_fr_norm{1},all_fr_norm{2},all_fr_norm{3}));
        info = [info ; spk.info];


        %- population decoding (LDA)
        fr4lda_both = cat(2,fr4lda{1},fr4lda{2}); % concatenate trials across tasks
        areas = spk.info.area;
        keep_units = nanmean(fr4lda_both,[2 3])>param.min_firing_threshold;

        for ar = 1 : length(area2test)
            unit4decoding = keep_units & ismember(areas,area2test{ar});
            
            if sum(unit4decoding)<5
            continue
            end

            disp(['          ---> Population decoding area ' area2test_name{ar} '...']) 
            nb(ar) = nb(ar) + 1;
            session{ar}(nb(ar),1) = sess;
            nb_units{ar}(nb(ar),1) = sum(unit4decoding);

            % classify using LDA  
            perf_proj = NaN(length(param.cond),length(param.cond));
            for cd = 1 : length(param.cond)
                curr_cond = param4lda{param.cond{cd}(1)}(:,param.cond{cd}(2));
                curr_fr = fr4lda{param.cond{cd}(1)}(unit4decoding,:,:);
                perf_lda=[];
                for b = 1 : size(curr_fr,3)

                    cv = cvpartition(curr_cond,'KFold',10); %- cross validation partitions: 10 fold
                    numFolds = cv.NumTestSets;

                    for i = 1:numFolds
                        %- take the trial according to the training/testing partitions
                        testInds = cv.test(i);
                        trainInds = cv.training(i);

                        x_train = squeeze(curr_fr(:,trainInds,b))';
                        y_train = curr_cond(trainInds);
                        x_test = squeeze(curr_fr(:,testInds,b))';
                        y_test = curr_cond(testInds);

                        class = [];
                        try [class,err,posterior,logp,coeff] = classify(x_test,x_train,y_train,'diaglinear');
                        end
                        if ~isempty(class) && ~isnan(mean(posterior(:)))
                            perf_lda(testInds,b) = y_test==class;
                        else
                            perf_lda(testInds,b) = NaN;
                        end
                    end
                end
                lda_res{ar}.(param.cond_name{cd}).avg_perf(nb(ar),:) = NaN(1,size(perf_lda,2));
                lda_res{ar}.(param.cond_name{cd}).avg_perf(nb(ar),:) = single(nanmean(perf_lda,1));

                % cross subspace decoding (extract subspace for 1 condition and project the other condition on it before decoding)
                curr_fr_bin = {};
                temp =  fr4lda{1}(unit4decoding,:,:);
                curr_fr_bin{1}= squeeze(nanmean(temp(:,:,spk.param.bins(2,:)==1 & spk.param.bins(1,:)>=param.bin4subspace_decoding(1) & spk.param.bins(1,:)<=param.bin4subspace_decoding(2)),3))'; % get the FR in the bin of interest
                temp =  fr4lda{2}(unit4decoding,:,:);
                curr_fr_bin{2}= squeeze(nanmean(temp(:,:,spk.param.bins(2,:)==1 & spk.param.bins(1,:)>=param.bin4subspace_decoding(1) & spk.param.bins(1,:)<=param.bin4subspace_decoding(2)),3))'; % get the FR in the bin of interest
                for cd2 = 1: length(param.cond)
          
                    % compute avg firing rate for cd2

                    fr_avg = [];
                    curr_cond2 = param4lda{param.cond{cd2}(1)}(:,param.cond{cd2}(2));
                    nb_cd = unique(curr_cond2);
                    for c = 1 : length(nb_cd)
                        idx = curr_cond2==nb_cd(c);
                        if sum(idx)>1
                            fr_avg(c,:) = single(nanmean(curr_fr_bin{param.cond{cd2}(1)}(idx,:),1));
                        else
                            fr_avg(c,:) = curr_fr_bin{param.cond{cd2}(1)}(idx,:);
                        end
                    end
                    % get mean and std for fr_avg so can normalize test set
                    fr_avg_mean = nanmean(fr_avg,1);
                    fr_avg_std = nanstd(fr_avg,0,1);

                    % drop units with no variance across conditions in the window (e.g. silent between 200-700ms):
                    % z-scoring gives 0/0 = NaN so PCA then fails
                    unit_ok = fr_avg_std>0 & ~any(isnan(fr_avg),1);
                    fr_avg = fr_avg(:,unit_ok);
                    fr_avg_mean = fr_avg_mean(unit_ok);
                    fr_avg_std = fr_avg_std(unit_ok);

                    % norm training
                    fr_avg = (fr_avg - repmat(fr_avg_mean,size(fr_avg,1),1)) ./ repmat(fr_avg_std,size(fr_avg,1),1); % norm FR
                    % norm testing
                    temp_fr = curr_fr_bin{param.cond{cd}(1)}(:,unit_ok);
                    temp_fr = (temp_fr - fr_avg_mean) ./ fr_avg_std;

                    % extract subspace
                    PC.nComp=1;
                    [PC.eigenvectors,PC.score,PC.eigenvalues,~,PC.explained,PC.mu] = pca(fr_avg,'NumComponents',PC.nComp);

                    % project all trials of the other condition on the subspace
                    curr_fr_proj = (temp_fr * PC.eigenvectors(:,1:PC.nComp)) ;
                    
                    % do LDA on the projected FR
                    perf_lda_proj = [];

                        for i = 1:numFolds
                            %- take the trial according to the training/testing partitions
                            testInds = cv.test(i);
                            trainInds = cv.training(i);

                            x_train = curr_fr_proj(trainInds,:);
                            y_train = curr_cond(trainInds);
                            x_test = curr_fr_proj(testInds,:);
                            y_test = curr_cond(testInds);

                            class = [];
                            try [class,err,posterior,logp,coeff] = classify(x_test,x_train,y_train,'diaglinear');
                            end
                            if ~isempty(class) && ~isnan(mean(posterior(:)))
                                perf_lda_proj(testInds,1) = y_test==class;
                            else
                                perf_lda_proj(testInds,1) = NaN;
                            end
                        end
                    perf_proj(cd,cd2) = nanmean(perf_lda_proj,1);

                end

            end
          lda_res{ar}.subspace_perf(nb(ar),:,:) =  perf_proj;

        end

    end

    lda_param = param;

    param = spk.param;
    param.conds = conds;
    param.lda = lda_param;
    param.matchTr = true;

    all_cond = [all_conds, NaN(size(all_conds,1),4) ; 
                NaN(size(all_conds,1),3) , all_conds(:,1) , NaN(size(all_conds,1),1) , all_conds(:,2:3) ;
                NaN(size(all_conds,1),3) , NaN(size(all_conds,1),1) , all_conds_unch(:,1:3) ];

    param.fr_conds = all_cond;

    save([pathout 'anova_lda.mat'],'anova_res','lda_res','all_fr','area2test','area2test_name','area_group','param','info','nb_units','session','-v7.3')

else
    load([pathout 'anova_lda.mat']);
    disp('ANOVA results loaded');
end


%% ANOVA POST hoc

areagrp = 1:length(area2test);
colorareas = [230 171 2 ; 152 78 163 ; 237 87 90 ; 252 141 98 ; 141 160 203 ; ...
     166 216 84 ; 102 194 165 ; 180 180 180 ; 231 138 195];

col = [100 200 160 ; 240 90 90 ; 0 0 0 ]/255; % state colors
col_light = col + (1-col)*0.6; % lighter version for dots

mk_col = [51 128 230; 140 89 38]/255; % Monkey colors

task = {'1FC' '2AFC'};

p_thr = 0.01;
n_thr = 5;
evt2avg = {'stim_on' 'stim_on'};
time2avg = {[-700 -200] [200 700]};

%- some param fro plotting
step = 20; % step in time for plotting purposes only (write down value every X time bin)
timesel = [1 2 3 4];
timesel_sub = [-.4 .98 ; -.01 .98 ; -.21 .38 ; -.21 .8];

gaps = [1 find(diff(param.bins(1,:))~=param.step)+1 length(param.bins(1,:))];
for t = 1 : length(timesel)
    time_considered = find(param.bins(2,:) == timesel(t) & param.bins(1,:)>=1000*timesel_sub(t,1) & param.bins(1,:)<=1000*timesel_sub(t,2));
    plot_me{t} = time_considered;
end

figure('Position',[28 451 2517 764]);
keep_units_all=zeros(size(info.area));
for t = 1 : length(task)
    pevs = cat(1,anova_res(:,t).pevs);
    nb_evts = length(param.evt);
    nb_cd = length(param.conds{t});    
    keep_units = sum(isnan(pevs),[2 3])==(nb_cd*nb_evts) ; %- remove neurons where anova failed (too low firing rate) Note that there is a NaN btw each evt alignments so n_evt*nb_cond NaNs
    keep_units_all = keep_units_all + keep_units;
end
keep_units = keep_units_all==2; %- keep only neurons that were kept in both tasks

for t = 1 : length(task)
    pvals = cat(1,anova_res(:,t).pval);
    pevs = cat(1,anova_res(:,t).pevs);
    coeffs = cat(1,anova_res(:,t).coeff);

    nb_evts = length(param.evt);
    nb_cd = length(param.conds{t});

    % find neurons with significant effect
    pvals_sig = false(size(pvals));
    for u = 1 : size(pvals,1)
        for cd = 1 : size(pvals,3)
            [idx,idxs] = utils_findenough(squeeze(pvals(u,:,cd)),p_thr,n_thr,'<');
            if ~isempty(idxs)
                pvals_sig(u,idxs,cd) = true;  
            end
        end
    end

    %- FIGURE S1a-c - ANOVA of probability and flavor   
    perc_sig{t} = NaN(length(area2test),size(pvals,3),size(pvals,2));
    for ar = 1 : length(area2test)
        for cd = 1 : size(pvals,3)
            subplot(2,4,cd+(t-1)*4) 
            idx = ismember(info.area,area2test{ar}) & keep_units;
            perc = squeeze(mean(pvals_sig(idx,:,cd),1));
            perc_sem = squeeze(std(pvals_sig(idx,:,cd),1)/sqrt(sum(idx)));
            timestart = 1;
            xlab = [];
            for ti = 1 : length(timesel)
                timeend = timestart+length(plot_me{ti})-1;
                % plot(timestart:timeend,perc(plot_me{ti}),'Color',colorareas(areagrp(ar),:)/255,'LineWidth',2); hold on
                plot(timestart:timeend,perc(plot_me{ti}),'Color',colorareas(ar,:)/255,'LineWidth',2); hold on
                % plot(timestart:timeend,perc(plot_me{ti})+perc_sem(plot_me{ti}),'Color',colorareas(areagrp(ar),:)/255,'LineWidth',.5); hold on
                % plot(timestart:timeend,perc(plot_me{ti})-perc_sem(plot_me{ti}),'Color',colorareas(areagrp(ar),:)/255,'LineWidth',.5); hold on
                line([timeend timeend],[0 80],'Color',[.6 .6 .6])
                timestart = timeend;
                xlab = [xlab param.bins(1,plot_me{ti})];
            end

            set(gca,'Xtick',1:step:length(xlab),'XtickLabel',xlab(1:step:end)/1000,'XtickLabelRotation',30,'FontSize',16)
            if cd == 2
                % text(10,.4-(ar/100),[area2test_name{ar} ' - ' num2str(sum(idx)) ],'Color',colorareas(areagrp(ar),:)/255,'FontSize',9)
                text(10,.4-(ar/100),[area2test_name{ar} ' - ' num2str(sum(idx)) ],'Color',colorareas(ar,:)/255,'FontSize',9)
            end

            % plot(squeeze(mean(pvals_sig(idx,:,cd),1)),'Color',colorareas(areagrp(ar),:)/255); hold on
            title(param.conds{t}(cd))
            perc_sig{t}(ar,cd,:) = squeeze(mean(pvals_sig(idx,:,cd),1));
            xlim([0 timeend+1])
            ylim([0 .425])
        end
    end

    for ee = 1 : length(evt2avg)
        %- sig or not across units during stim
        temp = single(squeeze(sum(pvals_sig(:,param.bins(1,:)>time2avg{ee}(1) & param.bins(1,:)<time2avg{ee}(2) & param.bins(2,:)==find(ismember(param.evt,evt2avg{ee})),:),2)>0));
        temp(~keep_units,:) = NaN;
        sig_units_evt{t,ee} = array2table(temp,'VariableNames',param.conds{t});

        temp_coeff = squeeze(mean(coeffs(:,param.bins(1,:)>time2avg{ee}(1) & param.bins(1,:)<time2avg{ee}(2) & param.bins(2,:)==find(ismember(param.evt,evt2avg{ee})),:),2));
        temp_coeff(~keep_units,:) = NaN;
        coeffs_evt{t,ee} = array2table(temp_coeff,'VariableNames',[{['intercept_' task{t} ]} param.conds{t}]);
    end
end


mks_all = cellfun(@(x) x(1),info.session,'UniformOutput',false);
sig_units_evt_bl = [array2table([mks_all,info.area,num2cell(keep_units_all==2)],'VariableNames',{'mk','area','keep'}) , sig_units_evt{1,1} , sig_units_evt{2,1}];
sig_units_evt_all = [array2table([mks_all,info.area,num2cell(keep_units_all==2)],'VariableNames',{'mk','area','keep'}) , sig_units_evt{1,2} , sig_units_evt{2,2}];

% add a column with neurons sig for both chosen proba and unchosen proba during 2AFC
sig_units_evt_all.bothproba_2AFC = sig_units_evt_all.chosenproba_2AFC==1 | sig_units_evt_all.unchosenproba_2AFC==1;
sig_units_evt_bl.bothproba_2AFC = sig_units_evt_bl.chosenproba_2AFC==1 | sig_units_evt_bl.unchosenproba_2AFC==1;

sig_units_evt_bl = sig_units_evt_bl([sig_units_evt_bl.keep{:}]',:);
sig_units_evt_all = sig_units_evt_all([sig_units_evt_all.keep{:}]',:);

coeffs_evt_all = [array2table([mks_all,info.area,num2cell(keep_units_all==2)],'VariableNames',{'mk','area','keep'}) , coeffs_evt{1,2} , coeffs_evt{2,2}];
coeffs_evt_all = coeffs_evt_all([coeffs_evt_all.keep{:}]',:);

%- for each area, count the number of neurons encoding none, one of both conditions
pair_cd = {'proba_1FC' 'bothproba_2AFC' ; 'flavor_1FC' 'chosenflavor_2AFC' ; 'side_1FC' 'chosenside_2AFC'};
clear prop_sig prop_sig_bl 
monks = unique(sig_units_evt_all.mk);
for c = 1 : length(pair_cd)
    for ar = 1 : length(area2test)
        for m = 1 : length(monks)+1
            if m == 1
                takeme = ismember(sig_units_evt_all.area,area2test{ar});
            else
                takeme = ismember(sig_units_evt_all.area,area2test{ar}) & ismember(sig_units_evt_all.mk,monks(m-1));
            end
            temp = sig_units_evt_all(takeme,pair_cd(c,:));
            %- [sig 1 / sig 2 / sig both / sig none]
            prop_sig{c,m}(ar,:) = [sum(temp.(pair_cd{c,1}) & ~temp.(pair_cd{c,2})) ...
                                sum(~temp.(pair_cd{c,1}) & temp.(pair_cd{c,2})) ... 
                                sum(temp.(pair_cd{c,1}) & temp.(pair_cd{c,2})) ...
                                sum(~temp.(pair_cd{c,1}) & ~temp.(pair_cd{c,2})) ] ;

            temp_bl = sig_units_evt_bl(takeme,pair_cd(c,:));
            %- [sig 1 / sig 2 / sig both / sig none]
            prop_sig_bl{c,m}(ar,:) = [sum(temp_bl.(pair_cd{c,1}) & ~temp_bl.(pair_cd{c,2})) ...
                                sum(~temp_bl.(pair_cd{c,1}) & temp_bl.(pair_cd{c,2})) ... 
                                sum(temp_bl.(pair_cd{c,1}) & temp_bl.(pair_cd{c,2})) ...
                                sum(~temp_bl.(pair_cd{c,1}) & ~temp_bl.(pair_cd{c,2})) ] ;
        end
    end
end

saveas(gcf, [report_dir 'Fig_S1abc_timecourse.png']); 

barcol = [100 60 150 ; 150 150 150 ; 250 130 190]/255;
pair_title = {'Probability' 'Flavor' 'Side'};
row_h = 4.2;  row_gap = 1.6;  y0 = 1.2;   % cm
W = 17;
H = y0 + 3*row_h + 2*row_gap + 1.4;
fg = figure('Units', 'centimeters', 'Position', [2 2 W H], 'Color', 'w', ...
    'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial');
cm = @(x, y, w, h) [x y w h] ./ [W H W H];

for c = 1 : length(pair_cd)
    y = y0 + (length(pair_cd)-c)*(row_h + row_gap);

    %- left: proportion of 1FC-only / both / 2AFC-only among significant neurons
    ax = axes(fg, 'Position', cm(0.6, y, 8.6, row_h));
    prop_plot = (prop_sig{c,1}(:,1:3)./repmat(sum(prop_sig{c,1}(:,1:3),2),1,3))*100; %- ignore the non significant for both
    prop_plot = prop_plot(:,[1 3 2]);
    [~,order] = sortrows(prop_plot(:,2));
    b = barh(ax, prop_plot(order,:),'stacked','FaceColor','flat','BarWidth',0.8,'LineWidth',0.5);
    for i = 1 : 3
        b(i).CData = barcol(i,:);
    end
    % write proportion on the center of each stacked bar
    for ar = 1 : length(area2test)
        for i = 1 : 3
            if prop_plot(order(ar),i)>0
                text(ax, sum(prop_plot(order(ar),1:i-1))+prop_plot(order(ar),i)/2,ar,num2str(round(prop_plot(order(ar),i))), ...
                    'Color','w','HorizontalAlignment','center','FontSize',9,'FontAngle','italic');
            end
        end
    end
    xlim(ax, [0 100]); ylim(ax, [0.4 length(area2test)+0.6]);
    set(ax,'YTick',[],'FontSize',10,'YDir','reverse','LineWidth',0.5);
    title(ax, pair_title{c}, 'FontSize', 11, 'FontWeight', 'bold');
    if c == length(pair_cd), xlabel(ax, 'Percent of significant neurons', 'FontSize', 11); end
    if c == 1
        lg = legend(ax, b, {'1FC' 'both' '2AFC'}, 'Orientation', 'horizontal', 'FontSize', 10, ...
            'FontAngle', 'italic', 'Units', 'centimeters', 'AutoUpdate', 'off');
        lg.Position(1:2) = [0.6 + (8.6 - lg.Position(3))/2, y + row_h + 0.75];
        lg.ItemTokenSize = [18 8];
    end

    %- right: percent of neurons significant in either task (bar: both monkeys, markers: each monkey, white dot: baseline)
    ax = axes(fg, 'Position', cm(11, y, 5.6, row_h));
    prop_plot_either = (sum(prop_sig{c,1}(:,1:3),2)./sum(prop_sig{c,1},2))*100; %- out of all neurons
    b = barh(ax, prop_plot_either(order,:),'FaceColor','k','BarWidth',0.8);
    hold(ax, 'on')
    prop_plot_either_bl = (sum(prop_sig_bl{c,1}(:,1:3),2)./sum(prop_sig_bl{c,1},2))*100; %- out of all neurons
    plot(ax, prop_plot_either_bl(order,:),1:length(area2test),'.','MarkerSize',6,'MarkerFaceColor','w','MarkerEdgeColor','w')

    prop_plot_either_mk = (sum(prop_sig{c,2}(:,1:3),2)./sum(prop_sig{c,2},2))*100; %- monkey M
    pM = plot(ax, prop_plot_either_mk(order,:),1:length(area2test),'o','MarkerSize',5,'MarkerFaceColor',[.6 .6 .6],'MarkerEdgeColor','none');
    prop_plot_either_mk = (sum(prop_sig{c,3}(:,1:3),2)./sum(prop_sig{c,3},2))*100; %- monkey X
    pX = plot(ax, prop_plot_either_mk(order,:),1:length(area2test),'v','MarkerSize',5,'MarkerFaceColor',[.6 .6 .6],'MarkerEdgeColor','none');

    set(ax,'YTick',1:length(area2test),'YTickLabel',area2test_name(order),'FontSize',10,'YDir','reverse', ...
        'XTick',0:20:100,'LineWidth',0.5);
    xlim(ax, [0 100]); ylim(ax, [0.4 length(area2test)+0.6]);
    if c == length(pair_cd), xlabel(ax, 'Percent of significant neurons', 'FontSize', 11); end
    if c == 1
        lg = legend(ax, [pM pX b], {'mk M' 'mk X' 'both'}, 'Orientation', 'horizontal', 'FontSize', 10, ...
            'FontAngle', 'italic', 'Units', 'centimeters', 'AutoUpdate', 'off');
        lg.Position(1:2) = [11 + (5.6 - lg.Position(3))/2, y + row_h + 0.75];
        lg.ItemTokenSize = [12 8];
    end
end
exportgraphics(fg, [report_dir 'Fig_S1de_proportions.pdf'], 'ContentType', 'vector');

figure('Position',[1440 247 642 1039]);
for c = 1 : length(pair_cd)
    subplot(length(pair_cd),1,c)
    line([0 1],[0 1],'Color','k')  ; hold on; box on
    coords = zeros(length(area2test),2);
    for ar = 1 : length(area2test)
        fc = sum(prop_sig{c,1}(ar,[1 3])) / sum(prop_sig{c,1}(ar,:));
        afc = sum(prop_sig{c,1}(ar,[2 3])) / sum(prop_sig{c,1}(ar,:));
        plot(fc, afc, '.', 'Markersize', 25, 'Color', colorareas(ar,:)/255);
        coords(ar,:) = [fc, afc];
    end

    utils_movelabels(coords, area2test_name, colorareas, gca); % place the labels with limited overlap! 

    set(gca, 'FontSize', 14); axis square
    xlabel(pair_cd{c,1})
    ylabel(pair_cd{c,2})
end

saveas(gcf, [report_dir 'Fig_S1f_scatter.png']);

%- plot consistency across monkeys
figure('Position',[1440 247 642 1039]);x = 0;
for c = 1 : length(pair_cd)
    x = x+1;
    subplot(length(pair_cd),2,x)
    line([0 1],[0 1],'Color','k')  ; hold on;box on
    coords = zeros(length(area2test),2);
    for ar = 1 : length(area2test)
        mk1_fc = sum(prop_sig{c,2}(ar,[1 3]))/sum(prop_sig{c,2}(ar,:));
        mk2_fc = sum(prop_sig{c,3}(ar,[1 3]))/sum(prop_sig{c,3}(ar,:));
        plot(mk1_fc,mk2_fc,'.','Markersize',25,'Color',colorareas(ar,:)/255);
        hold on
        coords(ar,:) = [mk1_fc, mk2_fc];
   end
    utils_movelabels(coords, area2test_name, colorareas, gca); % place the labels with limited overlap! 
    set(gca,'FontSize',14);xlim([0 1]);ylim([0 1]); axis square
    xlabel('mk M');ylabel('mk X')
    title(pair_cd{c,1})

    x = x+1;
    subplot(length(pair_cd),2,x)
    line([0 1],[0 1],'Color','k')  ; hold on;box on
    coords = zeros(length(area2test),2);
    for ar = 1 : length(area2test)
        mk1_afc = sum(prop_sig{c,2}(ar,[2 3]))/sum(prop_sig{c,2}(ar,:));
        mk2_afc = sum(prop_sig{c,3}(ar,[2 3]))/sum(prop_sig{c,3}(ar,:));
        plot(mk1_afc,mk2_afc,'.','Markersize',25,'Color',colorareas(ar,:)/255);
        hold on
        coords(ar,:) = [mk1_afc, mk2_afc];
    end
    utils_movelabels(coords, area2test_name, colorareas, gca); % place the labels with limited overlap! 
    set(gca,'FontSize',14);xlim([0 1]);ylim([0 1]); axis square
    xlabel('mk M');ylabel('mk X')
    title(pair_cd{c,2})
end

saveas(gcf, [report_dir 'Fig S1g_monkey.png']); 

% table with counts of analyzed neurons per monkey and combined across areas (using table info)
nb_rec = [];
for ar = 1 : length(area2test)
    nb_rec(ar,1) = sum(ismember(info.area,area2test{ar}) & ismember(mks_all,monks{1}));
    nb_rec(ar,2) = sum(ismember(info.area,area2test{ar}) & ismember(mks_all,monks{2}));
    nb_rec(ar,3) = sum(ismember(info.area,area2test{ar}) & ismember(mks_all,monks));
end
nb_rec = array2table(nb_rec,'VariableNames',{'mk_M' 'mk_X' 'both_mk'},'RowNames',area2test_name);

% table with counts of neurons kept across both tasks
nb_rec_keep = [];
for ar = 1 : length(area2test)
    nb_rec_keep(ar,1) = sum(ismember(sig_units_evt_all.area,area2test{ar}) & ismember(sig_units_evt_all.mk,monks{1}));
    nb_rec_keep(ar,2) = sum(ismember(sig_units_evt_all.area,area2test{ar}) & ismember(sig_units_evt_all.mk,monks{2}));
    nb_rec_keep(ar,3) = sum(ismember(sig_units_evt_all.area,area2test{ar}) & ismember(sig_units_evt_all.mk,monks));
end
nb_rec_keep = array2table(nb_rec_keep,'VariableNames',{'mk_M' 'mk_X' 'both_mk'},'RowNames',area2test_name);

%% Extract neurons with flavor encoding in 1FC and for each of them keep the sign of encoding (J1>J2 or J2>J1) - used by main_004

fl_neurons = sig_units_evt_all.flavor_1FC==1;

flavor_1FC_beta = coeffs_evt_all.flavor_1FC(fl_neurons);  % raw coefficient
flavor_1FC_sign = flavor_1FC_beta;
flavor_1FC_sign(flavor_1FC_sign>0) = 1;
flavor_1FC_sign(flavor_1FC_sign<0) = -1;

unit_nb = find(keep_units_all==2);
unit_nb = unit_nb(fl_neurons);

table_flavor_1FC = [array2table(unit_nb) , info(unit_nb,{'session','ch','clust_id','area'}) , ...
    array2table(flavor_1FC_beta,'VariableNames',{'flavor_1FC_beta'}) , ...
    array2table(flavor_1FC_sign,'VariableNames',{'flavor_1FC_sign'})];

if ~exist([pathout 'flavor_1fc.mat'], 'file')
    save([pathout 'flavor_1fc.mat'],'table_flavor_1FC');
end

%% Extract neurons with side encoding in 1FC and for each of them keep the sign of encoding (L>R or R>L) - used by main_004

sd_neurons = sig_units_evt_all.side_1FC==1;

side_1FC_beta = coeffs_evt_all.side_1FC(sd_neurons);  % raw coefficient
side_1FC_sign = side_1FC_beta;
side_1FC_sign(side_1FC_sign>0) = 1;
side_1FC_sign(side_1FC_sign<0) = -1;

unit_nb_side = find(keep_units_all==2);
unit_nb_side = unit_nb_side(sd_neurons);

table_side_1FC = [array2table(unit_nb_side,'VariableNames',{'unit_nb'}) , info(unit_nb_side,{'session','ch','clust_id','area'}) , ...
    array2table(side_1FC_beta,'VariableNames',{'side_1FC_beta'}) , ...
    array2table(side_1FC_sign,'VariableNames',{'side_1FC_sign'})];

if ~exist([pathout 'side_1fc.mat'], 'file')
    save([pathout 'side_1fc.mat'],'table_side_1FC');
end

saveas(gcf, [report_dir 'Fig S1g_monkey.png']); 

%% Posthoc LDA 

%- some param fro plotting
step = 20; % step in time for plotting purposes only (write down value every X time bin)
timesel = [1];
timesel_sub = [-.4 .98];
clear plot_me
gaps = [1 find(diff(param.bins(1,:))~=param.step)+1 length(param.bins(1,:))];
for t = 1 : length(timesel)
    time_considered = find(param.bins(2,:) == timesel(t) & param.bins(1,:)>=1000*timesel_sub(t,1) & param.bins(1,:)<=1000*timesel_sub(t,2));
    plot_me{t} = time_considered;
end

show = {'proba_1FC' 'chosenproba_2AFC' 'unchosenproba_2AFC'};

%- FIGURE S2 - LDA decoding of probability and flavor
fig_S2 = figure('Position',[415 50 1342 1306]); % Fig S2: panels A-D here, E-F in the cross-subspace section
for cd = 1 : length(show)
    subplot(3,5,1+(cd-1)*5)
    for ar = 1 : length(area2test)
        % plot(nanmean(lda_res{ar}.(show{cd}).avg_perf,1),'Color',colorareas(ar,:)/255);

        perf = nanmean(lda_res{ar}.(show{cd}).avg_perf,1);

        timestart = 1;
        xlab = [];
        for ti = 1 : length(timesel)
            timeend = timestart+length(plot_me{ti})-1;
            plot(timestart:timeend,100*perf(plot_me{ti}),'Color',colorareas(ar,:)/255,'LineWidth',2); hold on
            line([timeend timeend],[0 100],'Color',[.6 .6 .6])
            timestart = timeend;
            xlab = [xlab param.bins(1,plot_me{ti})];
        end

    end
    title(show{cd})
    set(gca,'Xtick',1:step:length(xlab),'XtickLabel',xlab(1:step:end)/1000,'XtickLabelRotation',30,'FontSize',12)
    if strcmp(show{cd},'proba_1FC') 
        ylim([15 50])
    elseif strcmp(show{cd},'chosenproba_2AFC') || strcmp(show{cd},'unchosenproba_2AFC')
        ylim([20 55])
    end
end

%- box and raincloud plot (swapped x/y axes)
wdth = .65;
time2test = param.bins(1,:) >= 200 & param.bins(1,:) <= 700 & param.bins(2,:)==1;
for cd = 1 : length(show)
    subplot(3,5,[2 3]+(cd-1)*5)
    for ar = 1 : length(area2test)
        perf2plot = nanmean(lda_res{ar}.(show{cd}).avg_perf(:,time2test),2);
        xl = ar;

        quartiles   = quantile(perf2plot, [0.25 0.75 0.5]);
        iqr         = quartiles(2) - quartiles(1);
        Xs          = sort(perf2plot);
        whiskers(1) = min(Xs(Xs > (quartiles(1) - (1.5 * iqr))));
        whiskers(2) = max(Xs(Xs < (quartiles(2) + (1.5 * iqr))));
        Y           = [quartiles whiskers];
        jit = (rand(size(perf2plot)) - 0.5) * (0.65*wdth);
        drops_pos = jit + xl ;
        box_pos = [xl(1)-(wdth * 0.5) Y(1) wdth Y(2)-Y(1)];

        curr_col = colorareas(ar,:)/255;

        h{2} = scatter(drops_pos, perf2plot,'SizeData',10,'MarkerEdgeColor','none','MarkerFaceColor',curr_col);
        h{3} = rectangle('Position', box_pos,'EdgeColor', curr_col(1,:),'LineWidth', 1.5);
        h{4} = line([xl(1)-(wdth * 0.5) xl(1) + (wdth * 0.5)], [Y(3) Y(3)], 'col', curr_col(1,:), 'LineWidth', 2);
        h{5} = line([xl(1) xl(1)], [Y(2) Y(5)], 'col', curr_col(1,:), 'LineWidth', 1);
        h{6} = line([xl(1) xl(1)], [Y(1) Y(4)], 'col', curr_col(1,:), 'LineWidth', 1);

        hold on
        title(show{cd})
    end
    set(gca,'XTick',1:length(area2test),'XTickLabel',area2test_name,'FontSize',12)
    % axis ij
    xlim([0 length(area2test)+1]); 
    if strcmp(show{cd},'proba_1FC') 
        ylim([.1 .6])
    elseif strcmp(show{cd},'chosenproba_2AFC') || strcmp(show{cd},'unchosenproba_2AFC')
        ylim([.15 .65])
    end 
    box on
end

%- GLME stats + significance stars on LDA boxplots
% Rebuild file list for monkey lookup if not in workspace (e.g. loaded from mat)
if ~exist('list','var')
    list = dir([pathspk '*_pool.mat']);
    dates_tmp = zeros(1,length(list));
    for i = 1:length(list)
        dates_tmp(i) = datenum(list(i).name(2:7),'mmddyy');
    end
    [~,idx_sort] = sort(dates_tmp);
    list = list(idx_sort);
end

for cd = 1 : length(show)
    % Build GLME table: one row per session per area
    modeldata_lda = table();
    for ar = 1 : length(area2test)
        if isempty(session{ar}), continue; end
        perf_sess  = nanmean(lda_res{ar}.(show{cd}).avg_perf(:,time2test),2);
        n_sess     = length(perf_sess);
        mk_sess    = arrayfun(@(s) list(s).name(1), session{ar}, 'UniformOutput', false);
        area_rep   = repmat(area2test_name(ar), n_sess, 1);
        sess_id    = session{ar};
        nunits_sess = double(nb_units{ar});  % number of units in decoder per session
        modeldata_lda = [modeldata_lda ; table(perf_sess, area_rep, mk_sess, sess_id, nunits_sess, ...
            'VariableNames', {'perf','area','mk','sess','n_units'})];
    end
    modeldata_lda.area = categorical(modeldata_lda.area);
    modeldata_lda.mk   = categorical(modeldata_lda.mk);
    modeldata_lda.sess = categorical(modeldata_lda.sess);

    % Fit GLME: area fixed effect, n_units covariate to control for neuron-count bias,
    % monkey and session as random intercepts
    lme_lda = fitglme(modeldata_lda, 'perf ~ 1 + area + n_units + (1|mk) + (1|sess)');
    utils_diary(fid_log, '\n--- LDA decoding stats: %s ---\n', show{cd});
    utils_diary(fid_log, '%s', evalc('disp(anova(lme_lda))'));

    % Pairwise posthoc (BH FDR-corrected)
    [~, ~, ~, pval_adj_lda] = utils_areaposthoc(lme_lda, area2test_name, 'n', true);
    % pad to square (utils_areaposthoc only writes columns 1..n-1, so output is n x n-1)
    n_ar = length(area2test_name);
    pval_adj_lda(n_ar, n_ar) = 0; % force square
    pval2_lda = pval_adj_lda + pval_adj_lda'; % symmetrize lower-triangular matrix

    % Add significance stars to the boxplot subplot (p < 0.01 FDR-corrected)
    subplot(3,5,[2 3]+(cd-1)*5);
    hold on
    for ar1 = 1 : length(area2test)
        perf1  = nanmean(lda_res{ar1}.(show{cd}).avg_perf(:,time2test),2);
        Ymean1 = nanmean(perf1);
        Ymax1  = max(perf1);
        Ymin1  = min(perf1);
        offset_up   = 0.01;
        offset_down = 0.01;
        for ar2 = 1 : length(area2test)
            if ar1 == ar2, continue; end
            if pval2_lda(ar1,ar2) < 0.01
                perf2  = nanmean(lda_res{ar2}.(show{cd}).avg_perf(:,time2test),2);
                Ymean2 = nanmean(perf2);
                if Ymean1 < Ymean2  % ar1 worse than ar2: star above ar1, colored by ar2
                    text(ar1, Ymax1 + offset_up, '*', 'Color', colorareas(ar2,:)/255, ...
                        'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                    offset_up = offset_up + 0.02;
                else                % ar1 better than ar2: star below ar1, colored by ar2
                    text(ar1, Ymin1 - offset_down, '*', 'Color', colorareas(ar2,:)/255, ...
                        'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                    offset_down = offset_down + 0.02;
                end
            end
        end
    end
end

% count number of sessions per area and monkeys and write to log
modeldata_lda = sortrows(modeldata_lda, {'area','mk'});
utils_diary(fid_log, '\n========== LDA DECODING: SESSION COUNTS ==========\n');
for ar = 1 : length(area2test)
    for m = 1 : length(monks)
        n_sess_mk = sum(modeldata_lda.area == area2test_name{ar} & modeldata_lda.mk == monks{m});
        utils_diary(fid_log, '%-10s  %s  %d sessions\n', area2test_name{ar}, monks{m}, n_sess_mk);
    end
end
utils_diary(fid_log, '\n=====================================================\n');

%- Per-area LDA summary: mean +/- SEM, n sessions, t-test vs chance
utils_diary(fid_log, '\n========== LDA DECODING: PER-AREA SUMMARY ==========\n');
chance_level = containers.Map( ...
    {'proba_1FC','chosenproba_2AFC','unchosenproba_2AFC'}, ...
    {0.20, 0.25, 0.25});
for cd = 1 : length(show)
    utils_diary(fid_log, '\n%s  (chance = %.0f%%)\n', show{cd}, chance_level(show{cd})*100);
    utils_diary(fid_log, '%-10s  %7s  %7s  %4s  %7s  %7s\n', 'Area','Mean%','SEM%','n','t','p');
    for ar = 1 : length(area2test)
        perf_ar = nanmean(lda_res{ar}.(show{cd}).avg_perf(:,time2test), 2);
        n_ar = sum(~isnan(perf_ar));
        if n_ar < 2, continue; end
        mu  = nanmean(perf_ar) * 100;
        sem = nanstd(perf_ar) / sqrt(n_ar) * 100;
        [~, p_t, ~, st] = ttest(perf_ar, chance_level(show{cd}));
        utils_diary(fid_log, '%-10s  %7.1f  %7.1f  %4d  %7.2f  %7.4f\n', area2test_name{ar}, mu, sem, n_ar, st.tstat, p_t);
    end
end
utils_diary(fid_log, '\n=====================================================\n');


%% Revision - Latency of LDA decoding

% --- parameters ---
lat_stim_win   = [0 1000];    % ms: window to search for peak / threshold crossing
lat_bl_win     = [-900 -100]; % ms: pre-stim baseline for the exceed criterion
lat_exceed_nsd = 2;          % threshold = BL_mean + N * BL_SD per session
lat_smooth_k   = 5;          % moving-average kernel (bins); set to 1 to skip

stim_bins  = param.bins(1,:) >= lat_stim_win(1) & param.bins(1,:) <= lat_stim_win(2) & param.bins(2,:)==1;
bl_bins    = param.bins(1,:) >= lat_bl_win(1)   & param.bins(1,:) <= lat_bl_win(2)   & param.bins(2,:)==1;
stim_times = param.bins(1, stim_bins);   % ms, 1 x n_stimbins

if ~exist('list','var')
    list = dir([pathspk '*_pool.mat']);
    dates_tmp = zeros(1,length(list));
    for i = 1:length(list)
        dates_tmp(i) = datenum(list(i).name(2:7),'mmddyy');
    end
    [~,idx_sort] = sort(dates_tmp);
    list = list(idx_sort);
end
if ~exist('wdth','var'), wdth = .65; end

lat_types  = {'lat_exceed','lat_peak'};
lat_labels = {'Exceed-BL latency (ms)', 'Peak latency (ms)'};

figure('Position',[415 50 1342 900]);
for cd = 1 : length(show)

    % --- collect latency per session × area ---
    lat_peak_all   = [];
    lat_exceed_all = [];
    area_lat       = {};
    mk_lat         = {};
    sess_lat       = [];
    nunits_lat     = [];

    for ar = 1 : length(area2test)
        if isempty(session{ar}), continue; end

        perf_all  = double(lda_res{ar}.(show{cd}).avg_perf);   % n_sess × n_timebins
        perf_stim = perf_all(:, stim_bins);
        perf_bl   = perf_all(:, bl_bins);
        n_sess    = size(perf_stim, 1);

        lp = NaN(n_sess, 1);
        le = NaN(n_sess, 1);
        for s = 1 : n_sess
            row = perf_stim(s,:);
            if all(isnan(row)), continue; end
            if lat_smooth_k > 1
                row = movmean(row, lat_smooth_k, 'omitnan');
            end
            % peak latency
            [~, pk_idx]  = max(row);
            lp(s)        = stim_times(pk_idx);
            % exceed-baseline latency
            bl_mu  = nanmean(perf_bl(s,:));
            bl_sd  = nanstd(perf_bl(s,:));
            ex_idx = find(row > bl_mu + lat_exceed_nsd * bl_sd, 1, 'first');
            if ~isempty(ex_idx)
                le(s) = stim_times(ex_idx);
            end
        end

        lat_peak_all   = [lat_peak_all   ; lp];
        lat_exceed_all = [lat_exceed_all ; le];
        area_lat       = [area_lat ; repmat(area2test_name(ar), n_sess, 1)];
        mk_lat         = [mk_lat   ; arrayfun(@(s) list(s).name(1), session{ar}, 'UniformOutput', false)];
        sess_lat       = [sess_lat   ; session{ar}];
        nunits_lat     = [nunits_lat ; double(nb_units{ar})];
    end

    modeldata_lat = table(lat_peak_all, lat_exceed_all, area_lat, mk_lat, sess_lat, nunits_lat, ...
        'VariableNames', {'lat_peak','lat_exceed','area','mk','sess','n_units'});
    modeldata_lat.area = categorical(modeldata_lat.area);
    modeldata_lat.mk   = categorical(modeldata_lat.mk);
    modeldata_lat.sess = categorical(modeldata_lat.sess);

    % --- GLME + posthoc for each latency type ---
    pval2_lat = {zeros(length(area2test_name)), zeros(length(area2test_name))};  % default: no sig

    for lt = 1 : 2
        valid = ~isnan(modeldata_lat.(lat_types{lt}));
        % only include areas with >= 2 valid sessions to avoid degenerate contrasts
        valid_areas_lt = {};
        for ar = 1 : length(area2test_name)
            if sum(valid & modeldata_lat.area == area2test_name{ar}) >= 2
                valid_areas_lt{end+1} = area2test_name{ar};
            end
        end
        if length(valid_areas_lt) < 2, continue; end

        formula  = [lat_types{lt} ' ~ 1 + area + n_units + (1|mk) + (1|sess)'];
        lme_lat  = fitglme(modeldata_lat(valid,:), formula);
        utils_diary(fid_log, '\n--- LDA %s: %s ---\n', lat_labels{lt}, show{cd});
        utils_diary(fid_log, '%s', evalc('disp(anova(lme_lat))'));

        % per-area summary
        utils_diary(fid_log, '%-10s  %7s  %7s  %4s\n', 'Area','Mean(ms)','SEM(ms)','n');
        for ar = 1 : length(area2test_name)
            d_ar = modeldata_lat.(lat_types{lt})(modeldata_lat.area == area2test_name{ar});
            d_ar = d_ar(~isnan(d_ar));
            if isempty(d_ar), continue; end
            utils_diary(fid_log, '%-10s  %7.1f  %7.1f  %4d\n', area2test_name{ar}, ...
                nanmean(d_ar), nanstd(d_ar)/sqrt(numel(d_ar)), numel(d_ar));
        end

        % posthoc only on areas with enough data
        [~,~,~, pv_adj] = utils_areaposthoc(lme_lat, valid_areas_lt, 'n', true);
        n_va = length(valid_areas_lt);
        pv_adj(n_va, n_va) = 0;
        pv_sym = pv_adj + pv_adj';   % symmetrize lower-triangular

        % remap into full-area pval matrix
        pval2_full = zeros(length(area2test_name));
        for a1 = 1 : n_va
            for a2 = 1 : n_va
                i1 = find(strcmp(area2test_name, valid_areas_lt{a1}));
                i2 = find(strcmp(area2test_name, valid_areas_lt{a2}));
                if ~isempty(i1) && ~isempty(i2)
                    pval2_full(i1,i2) = pv_sym(a1,a2);
                end
            end
        end
        pval2_lat{lt} = pval2_full;
    end

    % boxplot
    for lt = 1 : 2
        subplot(length(show), 2, (cd-1)*2 + lt)
        lat_data = modeldata_lat.(lat_types{lt});

        for ar = 1 : length(area2test)
            d = lat_data(modeldata_lat.area == area2test_name{ar});
            d = d(~isnan(d));
            if numel(d) < 2, continue; end

            quartiles  = quantile(d, [0.25 0.75 0.5]);
            iqr_v      = quartiles(2) - quartiles(1);
            Xs         = sort(d);
            wh(1) = min(Xs(Xs > quartiles(1) - 1.5*iqr_v));
            wh(2) = max(Xs(Xs < quartiles(2) + 1.5*iqr_v));
            Y     = [quartiles wh];
            jit   = (rand(size(d)) - 0.5) * (0.65*wdth);
            curr_col = colorareas(ar,:)/255;

            scatter(d, ar + jit, 'SizeData',10,'MarkerEdgeColor','none','MarkerFaceColor',curr_col); hold on
            rectangle('Position',[Y(1), ar-wdth*0.5, Y(2)-Y(1), wdth], 'EdgeColor',curr_col,'LineWidth',1.5);
            line([Y(3) Y(3)], [ar-wdth*0.5, ar+wdth*0.5], 'Color',curr_col,'LineWidth',2);
            line([Y(2) Y(5)], [ar ar], 'Color',curr_col,'LineWidth',1);
            line([Y(1) Y(4)], [ar ar], 'Color',curr_col,'LineWidth',1);
        end

        % significance stars: ar1 slower (higher latency) than ar2 = star above ar1 colored by ar2
        for ar1 = 1 : length(area2test)
            d1 = lat_data(modeldata_lat.area == area2test_name{ar1});
            d1 = d1(~isnan(d1));
            if isempty(d1), continue; end
            Ymax1  = max(d1);
            Ymean1 = nanmean(d1);
            offset_up = 20;
            for ar2 = 1 : length(area2test)
                if ar1==ar2, continue; end
                if pval2_lat{lt}(ar1,ar2) < 0.01
                    d2     = lat_data(modeldata_lat.area == area2test_name{ar2});
                    Ymean2 = nanmean(d2(~isnan(d2)));
                    if Ymean1 > Ymean2   % ar1 is slower: annotate above ar1
                        text(Ymax1 + offset_up, ar1, '*', 'Color',colorareas(ar2,:)/255, ...
                            'FontSize',14,'FontWeight','bold','HorizontalAlignment','center');
                        offset_up = offset_up + 30;
                    end
                end
            end
        end

        set(gca,'YTick',1:length(area2test),'YTickLabel',area2test_name,'FontSize',10,'YDir','reverse')
        ylim([0 length(area2test)+1])
        xlabel('Latency (ms)')
        title([lat_labels{lt} ' — ' show{cd}])
        box on
    end
end

saveas(gcf, [report_dir 'Fig_latency_lda.png']);

modeldata_lat = table(lat_peak_all, lat_exceed_all, area_lat, mk_lat, sess_lat, nunits_lat,'VariableNames', {'lat_peak','lat_exceed','area','mk','sess','n_units'});
modeldata_lat.area = categorical(modeldata_lat.area);
modeldata_lat.mk   = categorical(modeldata_lat.mk);
modeldata_lat.sess = categorical(modeldata_lat.sess);

% counting neumber of lat_peak and exceed at 0 or NaN and write to log
utils_diary(fid_log, '\n========== LDA DECODING: LATENCY COUNTS ==========\n');
for ar = 1 : length(area2test)
    for m = 1 : length(monks)
        n_sess_mk = sum(modeldata_lat.area == area2test_name{ar} & modeldata_lat.mk == monks{m});
        n_peak0   = sum(modeldata_lat.area == area2test_name{ar} & modeldata_lat.mk == monks{m} & modeldata_lat.lat_peak == 0);
        n_ex0     = sum(modeldata_lat.area == area2test_name{ar} & modeldata_lat.mk == monks{m} & modeldata_lat.lat_exceed == 0);
        n_peakNaN = sum(modeldata_lat.area == area2test_name{ar} & modeldata_lat.mk == monks{m} & isnan(modeldata_lat.lat_peak));
        n_exNaN   = sum(modeldata_lat.area == area2test_name{ar} & modeldata_lat.mk == monks{m} & isnan(modeldata_lat.lat_exceed));
        utils_diary(fid_log, '%-10s  %s  %d sessions (peak=0: %d, exceed=0: %d, peak=NaN: %d, exceed=NaN: %d)\n', ...
            area2test_name{ar}, monks{m}, n_sess_mk, n_peak0, n_ex0, n_peakNaN, n_exNaN);
    end
end
% and sum across areas and monkeys 
utils_diary(fid_log, '%-10s  %s  %d sessions (peak=0: %d, exceed=0: %d, peak=NaN: %d, exceed=NaN: %d)\n', ...
    'ALL', 'ALL', height(modeldata_lat), sum(modeldata_lat.lat_peak==0), sum(modeldata_lat.lat_exceed==0), ...
    sum(isnan(modeldata_lat.lat_peak)), sum(isnan(modeldata_lat.lat_exceed)));

%% Cross-subspace decoding

order = {'proba_1FC' 'chosenproba_2AFC' 'unchosenproba_2AFC'};

cds = {'proba_1FC' 'chosenproba_2AFC' 'unchosenproba_2AFC'};
subspaces = {'proba_1FC' 'chosenproba_2AFC'};

avg_perf = NaN(length(area2test),length(cds),2);
sem_perf = NaN(length(area2test),length(cds),2);
diff_perf = cell(length(area2test),1);

for ar = 1 : length(area2test)
    for cd = 1 : length(cds)
        perf = squeeze(lda_res{ar}.subspace_perf(:,ismember(order,cds(cd)),ismember(order,subspaces)));

        % get mean and sem across sessions in both dimensions
        avg_perf(ar,cd,:) = nanmean(perf,1);
        sem_perf(ar,cd,:) = nanstd(perf,0,1)./sqrt(size(perf,1));

        % perf difference
        diff_perf{ar}(:,cd) = perf(:,1) - perf(:,2); 
    end
end

%- Cross-subspace stats: decoding vs chance, only when the subspace comes from the other task
cross_pairs = {'proba_1FC' 'chosenproba_2AFC' ;   % {decoded condition, subspace}
               'chosenproba_2AFC' 'proba_1FC'};
utils_diary(fid_log, '\n========== CROSS-SUBSPACE DECODING STATS (vs chance) ==========\n');
pval_cs = []; % [p model area]
coef_cs = cell(size(cross_pairs,1),1);
for m = 1 : size(cross_pairs,1)
    modeldata_cs = table();
    for ar = 1 : length(area2test)
        if isempty(session{ar}), continue; end
        perf_sess = double(lda_res{ar}.subspace_perf(:, ismember(order,cross_pairs(m,1)), ismember(order,cross_pairs(m,2))));
        n_sess    = length(perf_sess);
        mk_sess   = arrayfun(@(s) list(s).name(1), session{ar}, 'UniformOutput', false);
        modeldata_cs = [modeldata_cs ; table(perf_sess - chance_level(cross_pairs{m,1}), repmat(area2test_name(ar), n_sess, 1), ...
            mk_sess, session{ar}, double(nb_units{ar}), 'VariableNames', {'perf_c','area','mk','sess','n_units'})];
    end
    modeldata_cs = modeldata_cs(~isnan(modeldata_cs.perf_c),:);
    modeldata_cs.n_units = modeldata_cs.n_units - mean(modeldata_cs.n_units); % centred: area estimates at the average decoder size
    modeldata_cs.area = categorical(modeldata_cs.area);
    modeldata_cs.mk   = categorical(modeldata_cs.mk);
    modeldata_cs.sess = categorical(modeldata_cs.sess);

    lme_cs = fitglme(modeldata_cs, 'perf_c ~ -1 + area + n_units + (1|mk) + (1|sess)', 'DummyVarCoding', 'full');
    if sum(startsWith(lme_cs.Coefficients.Name,'area_')) ~= numel(categories(modeldata_cs.area))
        error('Cross-subspace GLME: missing area coefficients');
    end
    coef_cs{m} = lme_cs.Coefficients;
    for ar = 1 : length(area2test)
        idx = strcmp(coef_cs{m}.Name, ['area_' area2test_name{ar}]);
        if any(idx), pval_cs = [pval_cs ; coef_cs{m}.pValue(idx) m ar]; end
    end
end
[~, ~, pval_cs(:,4)] = utils_fdr_bh(pval_cs(:,1)); % adjusted p in column 4

for m = 1 : size(cross_pairs,1)
    utils_diary(fid_log, '\nDecoding %s on %s subspace (chance = %.0f%%)\n', cross_pairs{m,1}, cross_pairs{m,2}, chance_level(cross_pairs{m,1})*100);
    utils_diary(fid_log, 'Model: perf - chance ~ -1 + area + n_units + (1|mk) + (1|sess)\n');
    utils_diary(fid_log, '%-10s  %8s  %7s  %7s  %5s  %9s  %9s\n', 'Area', 'Est(%)', 'SE(%)', 't', 'DF', 'p', 'p_FDR');
    for ar = 1 : length(area2test)
        idx = strcmp(coef_cs{m}.Name, ['area_' area2test_name{ar}]);
        row = pval_cs(:,2)==m & pval_cs(:,3)==ar;
        if ~any(idx), continue; end
        utils_diary(fid_log, '%-10s  %8.1f  %7.1f  %7.2f  %5d  %9.2e  %9.2e\n', area2test_name{ar}, coef_cs{m}.Estimate(idx)*100, ...
            coef_cs{m}.SE(idx)*100, coef_cs{m}.tStat(idx), coef_cs{m}.DF(idx), pval_cs(row,1), pval_cs(row,4));
    end
end
utils_diary(fid_log, '\nFDR across all areas of both models (n = %d tests)\n', size(pval_cs,1));
utils_diary(fid_log, '\n===============================================================\n');

% plot the avg with sem (back on the Fig S2 figure: the latency figure was opened in between)
figure(fig_S2);
for cd = 1 : length(cds)
    subplot(3,5,[4 5]+(cd-1)*5)

    coords = zeros(length(area2test),2);
    for ar = 1 : length(area2test)
        errorbar(avg_perf(ar,cd,1),avg_perf(ar,cd,2),sem_perf(ar,cd,2),sem_perf(ar,cd,2),sem_perf(ar,cd,1),sem_perf(ar,cd,1),'o','Markersize',5,'MarkerFaceColor',colorareas(ar,:)/255,'MarkerEdgeColor','k','Color','k','LineWidth',1); hold on
        coords(ar,:) = [avg_perf(ar,cd,1), avg_perf(ar,cd,2)];
    end
    utils_movelabels(coords, area2test_name, colorareas, gca,0.025,0.0175); % place the labels with limited overlap! 

    line([0 1],[0 1],'Color','k')  ; hold on; box on
    xlim([.2 .4]);ylim([.2 .4]);
    axis square
    set(gca,'FontSize',14);
    xlabel([subspaces{1} ' subspace'])
    ylabel([subspaces{2} ' subspace'])
    title(['Decoding ' cds{cd}])
    grid on
end

exportgraphics(fig_S2, [report_dir 'Fig_S2_subspace.pdf'], 'ContentType', 'vector');
% saveas(fig_S2, [report_dir 'Fig_S2_subspace.png']);

fig(1)
%- plot the difference in performance between the two subspaces (line histogram)
for cd = 1 : length(cds)
    subplot(1,3,cd)
    hold on
    for ar = 1 : length(area2test)
        % Compute histogram data
        [counts, edges] = histcounts(diff_perf{ar}(:,cd), 'Normalization', 'probability', 'BinEdges', linspace(-0.4, 0.4, 15));
        centers = edges(1:end-1) + diff(edges)/2;
        plot(centers, counts, 'LineWidth', 1, 'Color', colorareas(ar,:)/255);
    end
    line([0 0],[0 1],'Color','k','LineWidth',1);
    xlim([-0.5 0.5]);
    title(['Diff ' cds{cd}])
end

saveas(gcf, [report_dir 'Fig_extra_subspacediff.png']); 

fclose(fid_log); 

