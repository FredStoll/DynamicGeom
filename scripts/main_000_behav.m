%% main_000_behav.m
%
% Behavioral preference and hesitation analysis.
% - Loads or computes saccade summaries and behavioral preference outputs.
% - Writes report_000/output_000.txt and saves main behavior figures in report_000.
% - Uses project-relative paths so the script can run from any working directory.

clear

overwrite = false;

f = mfilename('fullpath');
if isempty(f) 
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end
addpath(genpath([currentPath '\scripts\'])); % add scripts folder to path
pathspk = [currentPath '\data_final\'];
pathout  = [currentPath '\processed\'];
report_dir = [currentPath '\report_000\'];
if ~exist(pathout, 'dir'), mkdir(pathout); end
if ~exist(report_dir, 'dir'), mkdir(report_dir); end
fid_log = fopen([report_dir 'output_000.txt'], 'w');
log_cleanup = onCleanup(@() fclose(fid_log));
utils_diary(fid_log, '\n============================================================\n');
utils_diary(fid_log, 'Report generated: %s\n', datestr(now));

list = dir([pathspk '*_pool.mat']);

% reorder files by dates 
for i = 1 : length(list)
    dates(i) = datenum(list(i).name(2:7),'mmddyy');
end
[~,idx] = sort(dates);  
list = list(idx);

norm = @(data) -1+((data-min(data))*2)/(max(data)-min(data)) ;

if overwrite
    files2delete = {'saccadeCounts_all.mat', 'saccadeCounts_reduced.mat', 'behav_pref.mat'};
    for iDel = 1:length(files2delete)
        fDel = [pathout files2delete{iDel}];
        if exist(fDel, 'file')
            delete(fDel);
        end
    end
end

% load saccades and plot saccade number vs abs(log ratio)
if exist([ pathout 'saccadeCounts_reduced.mat'],'file')
    disp('Loading existing saccade counts...')
    load([pathout 'saccadeCounts_reduced.mat'])
else
    disp('No existing saccade counts found, extracting now... This will take a few hours if not already done.')
    utils_extract_saccades(pathspk, pathout);
    load([pathout 'saccadeCounts_reduced.mat'])
end

if exist([ pathout 'behav_pref.mat'],'file')

    load([ pathout 'behav_pref.mat'],'preference','xrange','saccade_pb')

else

    preference = table();

    for sess = 1 : length(list)
    
        clearvars -except sess list pathspk choice_bias norm preference saccadeTable saccade_pb pathout currentPath report_dir overwrite fid_log log_cleanup
        
        %- load spiking and behav data for that session
        disp(['Processing session ' num2str(sess) ' of ' num2str(length(list)) '...'])
        spk = load([pathspk list(sess).name],'cond','info');

        if istable(spk.cond)
            task_all       = spk.cond.task;
            chosenside_all = spk.cond.chosenside_2AFC;
            spk.cond = table2array(spk.cond);
        end

        %- side bias: proportion of 2AFC trials where side 1 was chosen (0.5 = unbiased)
        is2afc    = task_all==2 & ~isnan(chosenside_all);
        n_2afc    = sum(is2afc);
        side_bias = mean(chosenside_all(is2afc)==1); % NaN if no valid 2AFC trials

        probaJ1 = NaN(length(spk.cond),1); % Initialize probaJ1 to NaN
        probaJ2 = NaN(length(spk.cond),1); % Initialize probaJ2 to NaN


        % Map probabilities to J1 and J2 based on which side each flavor appeared on.
        % cond col 8 = chosen flavor, col 9 = unchosen flavor, col 6 = chosen proba, col 7 = unchosen proba.
        %proba_side1 = NaN(length(spk.cond),1);
        %proba_side2 = NaN(length(spk.cond),1);

        diff_fl = spk.cond(:,8)~=spk.cond(:,9);
        probaJ1(spk.cond(:,8)==1) = spk.cond(spk.cond(:,8)==1,6);
        probaJ1(spk.cond(:,9)==1) = spk.cond(spk.cond(:,9)==1,7);
        probaJ2(spk.cond(:,8)==2) = spk.cond(spk.cond(:,8)==2,6);
        probaJ2(spk.cond(:,9)==2) = spk.cond(spk.cond(:,9)==2,7);

        tr_ID = spk.cond(:,1);
        choice = spk.cond(:,8); % choice 1 or 2
        keepme = ~isnan(probaJ1) & ~isnan(probaJ2); % remove NaN values (NaN for 1FC and when both flavor same!)

        probaJ2 = probaJ2(keepme); % keep only non-NaN values
        probaJ1 = probaJ1(keepme); % keep only non-NaN values
        choice = choice(keepme); % keep only non-NaN values
        tr_ID = tr_ID(keepme); % keep only non-NaN values
        logpb = log(probaJ1./probaJ2);
        
        saccades = saccadeTable.NumSaccades(ismember(string(saccadeTable.SessionID), string(list(sess).name(1:7))) & ismember(saccadeTable.Trial,tr_ID));

        saccade_pb{sess} = [double(saccades), logpb]; %only for different flavor trials

        T = table(choice==1,logpb,norm(probaJ1),norm(probaJ2),'VariableNames',{'choice','prob','probaJ1','probaJ2'}); % create a table with choice and log odds of the probabilities
    % T.choice=categorical(T.choice);

        % REVISION: Fit without penalty first to detect if perfect separation occurs
        lastwarn('', '');
        warning('off', 'stats:glmfit:PerfectSeparation');
        warning('off', 'stats:glmfit:IterationLimit');
        mdl_check = fitglm(T,'choice ~ 1 + prob','Distribution','binomial','Link','logit');
        [~, warnId] = lastwarn;
        separation_flag = contains(warnId, 'PerfectSeparation') || contains(warnId, 'IterationLimit');
        warning('on', 'stats:glmfit:PerfectSeparation');
        warning('on', 'stats:glmfit:IterationLimit');
        % -------------------------------------

        % full model with interaction term, using Jeffreys prior to handle potential separation pbs
        lastwarn('', '');
        mdl = fitglm(T,'choice ~ 1 + probaJ1*probaJ2','Distribution','binomial','Link','logit','LikelihoodPenalty','jeffreys-prior');
        mdl2 = fitglm(T,'choice ~ 1 + prob','Distribution','binomial','Link','logit','LikelihoodPenalty','jeffreys-prior');

        clear newf newc
        probas = norm([10:1:90]);
        thr = NaN(length(probas),1);
        
        for i = 1 : length(probas)
            
            tab = table((probas(i)*ones(length(probas),1)),probas','VariableNames',{'probaJ1','probaJ2',});
            [newf(i,:) , newc] = predict(mdl,tab); %- last column is the sessiooon
            
            if ~isempty(find(newf(i,:)>=0.5,1,'first'))
                thr(i,1) = find(newf(i,:)>=0.5,1,'first');
            end
        end
        
        %- extract the preference for J1 vs J2
        %- sensitivity doesn't take into account the interaction factor... so can use the nb of
        %- estimated probaChoiceJ1>0.5.. if pref>0.5, prefer the J1
        sensitivity = mdl.Coefficients{'probaJ1','Estimate'}/mdl.Coefficients{'probaJ2','Estimate'};
        choice_bias = sum(sum(newf<0.5))/(length(newf(:))-sum(sum(newf==0.5)));
        
        % get proba influence (t-statistic for the probability coefficient in the simpler model)
        proba_tstat = mdl2.Coefficients.tStat(2);
        proba_r2 = mdl2.Rsquared.Adjusted;
        proba_est = mdl2.Coefficients.Estimate(2);

        %- preference from log ratio model
        bias_point = -mdl2.Coefficients{'(Intercept)','Estimate'} / ...
                mdl2.Coefficients{'prob','Estimate'};

        indifference_ratio = exp(bias_point);   % probaJ1 : probaJ2 ratio
        % If indifference_ratio = 1 -> unbiased (needs equal J1 and J2 to be 50/50).
        % If indifference_ratio > 1 -> animal needs J1 stronger than J2 to be indifferent -> bias toward J2.
        % If indifference_ratio < 1 -> animal needs J2 stronger than J1 to be indifferent -> bias toward J1.

        %- predicted values
        xrange = linspace(-log(90/10),log(90/10),200)';  % covers full proba range (log(90/10) ~= 2.197)
        tab = table(xrange,'VariableNames',{'prob'});
        [predictedP,~] = predict(mdl2,tab);

        % make a table with session name, choice bias, probability model metrics, and predictedP
        preference = [preference ; table(spk.info.session(1), choice_bias, proba_r2 , proba_tstat,  proba_est, bias_point, indifference_ratio, {predictedP}, separation_flag, side_bias, n_2afc, ...
            'VariableNames', {'session', 'choice_bias', 'proba_r2','proba_tstat',  'proba_est', 'bias_point', 'indifference_ratio', 'predictedP', 'separation_flag', 'side_bias', 'n_2afc'})];
        % make a table with session name, choice bias, probability model metrics, and predictedP
        %preference = [preference ; table(spk.info.session(1), choice_bias, proba_r2 , proba_tstat,  proba_est, bias_point, indifference_ratio, {predictedP}, ...
        %    'VariableNames', {'session', 'choice_bias', 'proba_r2','proba_tstat',  'proba_est', 'bias_point', 'indifference_ratio', 'predictedP'})];

    end

    save([ pathout 'behav_pref.mat'],'preference','xrange','saccade_pb')
    
end

%% some plots
monkey = cellfun(@(x) x(1), preference.session, 'UniformOutput', false);
monkey_names = unique(monkey);

main_colors = containers.Map({'M', 'X'}, {[0.2, 0.5, 0.9], [0.55, 0.35, 0.15]});
light_colors = containers.Map({'M', 'X'}, {[0.7, 0.85, 1], [0.85, 0.75, 0.6]});

fig(1);

subplot_indices = {[3 4], [8 9]}; % subplot positions for each monkey

for i = 1:numel(monkey_names)
    subplot(2,5,subplot_indices{i});
    hold on;
    idx = strcmp(monkey, monkey_names{i});
    if any(idx)
        preds = cat(2,preference.predictedP{idx});
        plot(xrange, preds, '-', 'Color', light_colors(monkey_names{i}), 'LineWidth', 1.5);
        boxplot(preference.bias_point(idx), 'orientation', 'horizontal', ...
            'positions', 0.5, 'Colors', main_colors(monkey_names{i}), 'Symbol', '');
        ylim([0 1]);
        set(gca, 'YTick', [0 0.5 1], 'YTickLabel', {'0','0.5','1'}, 'FontSize', 16);
    end
    hold off;
    xlabel('log(probaJ1/probaJ2)', 'FontSize', 16);
    ylabel('P(choice J1)', 'FontSize', 16);
end

% R2
subplot(2,5,[1 6])
hold on;
for i = 1:numel(monkey_names)
    idx = strcmp(monkey, monkey_names{i});
    scatter(i*ones(sum(idx),1), preference.proba_r2(idx), 40, light_colors(monkey_names{i}), 'filled', 'jitter','on','jitterAmount',0.15);
end
colors = cell2mat(values(main_colors, monkey_names));
h = boxplot(preference.proba_r2, monkey, 'Colors', colors, 'Symbol', '');
ylim([.2 1])
ylabel('Adjusted R^2');
set(gca, 'FontSize', 16);
set(gca, 'Box', 'off');
set(gca, 'XColor', 'none'); % Remove x-axis line
ax = gca;
ax.XRuler.Axle.Visible = 'off'; % Remove x-axis left line
ax.YAxis.TickDirection = 'out';
% Remove right/top lines
ax.LineWidth = 1;
ax.TickLength = [0.02 0.02];

% t-stat
subplot(2,5,[2 7])
hold on;
for i = 1:numel(monkey_names)
    idx = strcmp(monkey, monkey_names{i});
    scatter(i*ones(sum(idx),1), preference.proba_tstat(idx), 40, light_colors(monkey_names{i}), 'filled', 'jitter','on','jitterAmount',0.15);
end
colors = cell2mat(values(main_colors, monkey_names));
h = boxplot(preference.proba_tstat, monkey, 'Colors', colors, 'Symbol', '');
ylim([0 10])
ylabel('log(probaJ1/probaJ2) t-statistic');
set(gca, 'FontSize', 16);
set(gca, 'Box', 'off');
set(gca, 'XColor', 'none'); % Remove x-axis line
ax = gca;
ax.XRuler.Axle.Visible = 'off'; % Remove x-axis left line
ax.YAxis.TickDirection = 'out';
ax.LineWidth = 1;
ax.TickLength = [0.02 0.02];

%- extract saccade likelihood against proba diff + plot
log_pb_values = [];
all_log_pb_values = table();
monkey_values = [];
monkey_list = cellfun(@(x) x(1), {list.name}, 'UniformOutput', false);
for sess = 1:length(saccade_pb)
    data = saccade_pb{sess};

    bias_value = preference.bias_point(sess);
    data(:,2) = abs(data(:,2) - bias_value); % center the log(proba) values

    valid_saccades = data(data(:,1) > 1, :); % filter for saccades > 1
    log_pb_values = [log_pb_values; valid_saccades(:,2)];
    % Add monkey info to all_log_pb_values and make it a table
    monkey_col = repmat(monkey_list(sess), size(data, 1), 1);
    all_log_pb_values = [all_log_pb_values; table(data(:,1)>1, data(:,2), monkey_col,'VariableNames', {'hesit', 'log_pb', 'monkey'})];  
    monkey_values = [monkey_values; repmat(monkey_list(sess), size(valid_saccades, 1), 1)];
end

num_bins = 5;
subplot(2,5,10)
for m = 1:numel(monkey_names)
    idx_monkey = strcmp(all_log_pb_values.monkey, monkey_names{m});
    hesit = all_log_pb_values.hesit(idx_monkey);
    log_pb = all_log_pb_values.log_pb(idx_monkey);

    % Quantile binning
    quantiles = quantile(log_pb, linspace(0,1,num_bins+1));
    bin_edges = unique(quantiles); % Remove duplicate edges if any
    [~,~,bin_idx] = histcounts(log_pb, bin_edges);

    prop_hesit  = arrayfun(@(b) mean(hesit(bin_idx==b)), 1:length(bin_edges)-1);
    prop_ci95   = arrayfun(@(b) 1.96 * std(hesit(bin_idx==b)) / sqrt(sum(bin_idx==b)), 1:length(bin_edges)-1);
    centers     = arrayfun(@(b) mean(log_pb(bin_idx==b)), 1:length(bin_edges)-1);

    errorbar(centers, prop_hesit, prop_ci95, '-', 'Color', main_colors(monkey_names{m}), 'LineWidth', 2, 'Marker', 'o', 'MarkerFaceColor', main_colors(monkey_names{m}));
    hold on;
    xlabel('Abs/Centered log(PbJ1/PbJ2)');
    ylabel('Hesitation probability');
    set(gca, 'FontSize', 16);
end
ylim([0 0.1])
xlim([0 2.5])

% stats for log_pb_values with monkeys as random effect
lme = fitglme(all_log_pb_values, 'hesit ~ 1 + log_pb + (1|monkey)', 'Distribution', 'binomial', 'Link', 'logit');
anova_hesit = anova(lme);
disp(anova_hesit)
utils_diary(fid_log, '\n--- Hesitation GLME: hesit ~ log_pb + (1|monkey) ---\n');
utils_diary(fid_log, '%s', evalc('disp(anova_hesit)'));

%compute overall proportion of saccade>1
nHesit = NaN(size(saccade_pb));
pHesit = NaN(size(saccade_pb));
nQuick = NaN(size(saccade_pb));

for sess = 1:length(saccade_pb)
    pHesit(sess) = 100*(sum(saccade_pb{sess}(:,1)>1) / length(saccade_pb{sess}(:,1)));
    nHesit(sess) = sum(saccade_pb{sess}(:,1)>1);
    nQuick(sess) = sum(saccade_pb{sess}(:,1)==1);
end

% plot that
subplot(2,5,5)
% t-stat
hold on;
for i = 1:numel(monkey_names)
    idx = strcmp(monkey, monkey_names{i});
    scatter(i*ones(sum(idx),1), pHesit(idx), 40, light_colors(monkey_names{i}), 'filled', 'jitter','on','jitterAmount',0.15);
end
colors = cell2mat(values(main_colors, monkey_names));
h = boxplot(pHesit, monkey, 'Colors', colors, 'Symbol', '');
%ylim([0 10])
ylabel('Percent of hesitation trials');
set(gca, 'FontSize', 16);
set(gca, 'Box', 'off');
set(gca, 'XColor', 'none'); % Remove x-axis line
ax = gca;
ax.XRuler.Axle.Visible = 'off'; % Remove x-axis left line
ax.YAxis.TickDirection = 'out';
ax.LineWidth = 1;
ax.TickLength = [0.02 0.02];

% display mean + std for each monkey + total number of trials
for i = 1:numel(monkey_names)
    idx = strcmp(monkey, monkey_names{i});
    mean_val = mean(pHesit(idx), 'omitnan');
    std_val = std(pHesit(idx), 'omitnan');
    nb_hes = sum(nHesit(idx));
    nb_quick = sum(nQuick(idx));
    utils_diary(fid_log, 'Monkey: %s, Mean+std: %.2f+%.2f, n Hesitation Trials: %d, n Quick Trials: %d\n', monkey_names{i}, mean_val, std_val, nb_hes, nb_quick );
end

% Save the main behavior summary figure in the report_000 folder.
fig_main = figure(1);
saveas(fig_main, [report_dir 'Fig_1bcdf_behav.png']);

% fig(1);plot(preference.choice_bias,preference.bias_point,'o','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','none');

num_sep = sum(preference.separation_flag);
num_sep_monkey = arrayfun(@(m) sum(preference.separation_flag(strcmp(monkey, m))), monkey_names);
total_sess = height(preference);
utils_diary(fid_log, '\n--- Jeffreys Prior / Separation Check ---\n');
utils_diary(fid_log, 'Perfect separation (so need Jeffreys prior) occurred in %d of %d sessions (%.1f%%).\n\n', num_sep, total_sess, (num_sep/total_sess)*100);
fprintf('\nSeparation occurred in %d of %d sessions (%.1f%%).\n\n', num_sep, total_sess, (num_sep/total_sess)*100);
fprintf('Per monkey:\n');
for m = 1:numel(monkey_names)
    fprintf('Monkey %s: %d of %d sessions (%.1f%%)\n', monkey_names{m}, num_sep_monkey(m), sum(strcmp(monkey, monkey_names{m})), (num_sep_monkey(m)/sum(strcmp(monkey, monkey_names{m})))*100);
end


%% spike statistics (fano, fr, burstiness)

if overwrite
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
    param.cond = {'task' 'proba_1FC' 'flavor_1FC' 'side_1FC' 'chosenproba_2AFC' 'chosenflavor_2AFC' 'chosenside_2AFC'};
    param.evt = {'stim_on' };
    param.pre = [750 ]; % pre-stimulus time (ms)
    param.post = [1250 ]; % post-stimulus time (ms)
    param.binsize = 200; % bins size (ms)
    param.step = 10; % step size (ms)
    param.sdf = 20; % smooth param for sdf (ms)

    param.bins =[];
    for i = 1 : length(param.pre)
        param.bins = [param.bins , [-param.pre(i):param.step:param.post(i)-param.binsize ; i*ones(1,length(-param.pre(i):param.step:param.post(i)-param.binsize))]];
    end

    info = table();

    % load the files and extract the info
    for sess = 1 : length(list)

        clearvars -except sess list pathspk area2test param anova_res nb lda_res nb_units session path2save overwrite info currentPath pathout report_dir fid_log log_cleanup

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
            cond(:,cd) = spk.behav.trialtype(:,idx_cond{cd});
        end
        evt_time = [];
        for e = 1 : length(param.evt)
            evt_time(:,e) = spk.behav.t_evt.(param.evt{e})(:);
        end

        %- remove unwanted probas
        if sum(ismember(param.cond,'proba_1FC'))~=0 
            % idx_tr1 = ~ismember(cond(:,ismember(param.cond,'proba_1FC')),[10 30 50 70 90]) & ismember(cond(:,ismember(param.cond,'task')),1)   ;
            idx_tr1 = ~ismember(cond(:,ismember(param.cond,'proba_1FC')),[30 50 70 90]) & ismember(cond(:,ismember(param.cond,'task')),1)   ;
        else
            idx_tr1 = zeros(size(cond,1),1);
        end
        if sum(ismember(param.cond,'chosenproba_2AFC'))~=0 
            idx_tr2 = ~ismember(cond(:,ismember(param.cond,'chosenproba_2AFC')),[30 50 70 90]) & ismember(cond(:,ismember(param.cond,'task')),2)   ;
        else
            idx_tr2 = zeros(size(cond,1),1);
        end
        idx_tr = idx_tr1 | idx_tr2 | ~completed_tr;
        cond(idx_tr,:)=[];
        evt_time(idx_tr,:)=[];

        %-  only process session if enough trials 
        if sum(cond(:,1)==1)<param.min_nbTr | sum(cond(:,1)==2)<param.min_nbTr
            disp('      ... not enough trials!')
        continue   
        end

        %- look at neurons now
        for u = 1 : size(spk.unit,2)
            disp(['    ---> Processing unit ' num2str(u) ' of ' num2str(size(spk.unit,2)) '...'])


                %- make PSTH around event of interest
                psth_trials = zeros(param.pre(e)+param.post(e)+1,length(evt_time(:,e))); % will be a matrix of size nTR x nTIME 
                trialspx = cell(numel(evt_time(:,e)),1);
                trialspx_raw = [];
                for tr = 1:numel(evt_time(:,e)) %- for each trial
                    clear spikes
                    spikes = (spk.unit{u}.timestamps - evt_time(tr,e))*1000;         % all spikes relative to current trigtime
                    trialspx{tr} = round(spikes(spikes>=-param.pre(e) & spikes<=param.post(e)));   % spikes close to current trigtime
                    temp_spk = spk.unit{u}.timestamps(spk.unit{u}.timestamps>=evt_time(tr,e)-(param.pre(e)/1000) & spk.unit{u}.timestamps<=evt_time(tr,e)+(param.post(e)/1000)) ;
                    trialspx_raw = [trialspx_raw ; temp_spk' repmat(cond(tr,1),size(temp_spk'))];
                    psth_trials(trialspx{tr}+param.pre(e)+1,tr) = psth_trials(trialspx{tr}+param.pre(e)+1,tr)+1; % add a 1 when spike
                end
                time = -param.pre(e):param.post(e); %- time vector


                for t = 1 : 2
                    temp = psth_trials(:,cond(:,1)==t);
                    avg_FR(u,t) = mean(temp(:))*1000;
                end
            
                for t = 1 : 2
                    bursti = [];
                    subtr = trialspx(cond(:,1)==t);
                    for tr = 1 : length(subtr)
                        if length(subtr{tr})>5
                            isis = diff(subtr{tr})/1000;
                            bursti(tr)= (std(isis)-mean(isis))/(std(isis)+mean(isis));
                        else
                            bursti(tr)=NaN;
                        end
                    end
                    burstiness(u,t) = nanmean(bursti);
                end

                % get fano factor for 1FC and 2AFC
                for t = 1 : 2
                    temp = psth_trials(:,cond(:,1)==t);
                    fano(u,t) = var(sum(temp))/mean(sum(temp));
                end

                % Store results in the info table
                info = [info ; table({spk.unit{u}.session},{spk.unit{u}.ch},spk.unit{u}.clust_id,{spk.unit{u}.area},avg_FR(u,1), avg_FR(u,2), burstiness(u,1), burstiness(u,2),fano(u,1),fano(u,2),'VariableNames',{'session' 'ch' 'clust_id' 'area' 'fr_1FC' 'fr_2AFC' 'burstiness_1FC' 'burstiness_2AFC' 'fano_1FC' 'fano_2AFC'})];

            % info = [info; {sess, u, avg_FR(u,1), avg_FR(u,2), burstiness(u,1), burstiness(u,2)}];
        end



    % figure;plot(burstiness(:,1),burstiness(:,2),'o')


    end

    save([pathout 'spiking_info.mat'],'info')
else

    load([pathout 'spiking_info.mat'])
    disp('Info table loaded!')
end

%% process the info mat file
%- check for differences in firing statistics between 1FC and 2AFC trials for each area/monkey
utils_diary(fid_log, '\n============================================================\n');

area2test =      {'24c' {'6DR' '6DC' '6Va/Vb'}  {'8' '8B' '8A' '46d' '46df' '46v'}    {'IFG' '44' '45'} {'12r' '12m' '12m/r' '12o' '12l'}   'AI'  {'13l' '13m' '11m/l'} {'cd'  'pu'}  'AMG'};
area2test_name = {'MFC' 'PMC'  'dlPFC'            'IFG'       'vlPFC'   'AI'  'OFC' 'STR'  'AMG'};

mk = cellfun(@(x) x(1),info.session,'UniformOutput',false);

% Assign grouped area names to each entry in info.area
grouped_area = cell(size(info.area));
for i = 1:length(area2test)
    if iscell(area2test{i})
        idx = ismember(info.area, area2test{i});
    else
        idx = ismember(info.area, area2test(i));
    end
    grouped_area(idx) = area2test_name(i);
end

% mixed effect model to explain differences in firing rate between 1FC and 2AFC trials with fixed param area and random param monkey
tbl = table(info.fr_2AFC-info.fr_1FC, grouped_area, mk, 'VariableNames', {'diff_fr' 'area' 'monkey'});
lme = fitlme(tbl,'diff_fr ~ area + (1|monkey)');
% MAKE A TABLE COUNTING NUMBER OF NEURONS PER AREA AND MONKEY USING TBL
monkey_list = unique(tbl.monkey);
% Ensure area_list is sorted according to area2test_name
area_list = area2test_name(:);
nb_units = zeros(length(area_list), length(monkey_list));
for a = 1:length(area_list)
    for m = 1:length(monkey_list)
        nb_units(a,m) = sum(strcmp(tbl.area, area_list{a}) & strcmp(tbl.monkey, monkey_list{m}));
    end
end
% Add sum across monkeys (per area)
nb_units_sum_monkey = sum(nb_units,2);
% Add sum across areas (per monkey)
nb_units_sum_area = sum(nb_units,1);

% Create table with sums
nb_units_table = array2table(nb_units, 'VariableNames', monkey_list, 'RowNames', area_list);
nb_units_table.SumAcrossMonkeys = nb_units_sum_monkey;
utils_diary(fid_log, '\n--- Neuron counts per area and monkey ---\n');
disp(nb_units_table)
% Write formatted neuron counts table
utils_diary(fid_log, '\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  NEURON COUNTS PER AREA\n');
utils_diary(fid_log, '============================================================\n');
hdr_str = sprintf('  %-8s', 'Area');
sep_str = sprintf('  %-8s', '--------');
for mi = 1:length(monkey_list)
    hdr_str = [hdr_str sprintf('  %6s', monkey_list{mi})];
    sep_str = [sep_str sprintf('  %6s', '------')];
end
hdr_str = [hdr_str sprintf('  %8s\n', 'Total')];
sep_str = [sep_str sprintf('  %8s\n', '--------')];
utils_diary(fid_log, '%s', hdr_str);
utils_diary(fid_log, '%s', sep_str);
for ai = 1:length(area_list)
    row_str = sprintf('  %-8s', area_list{ai});
    for mi = 1:length(monkey_list)
        row_str = [row_str sprintf('  %6d', nb_units(ai,mi))];
    end
    row_str = [row_str sprintf('  %8d\n', nb_units_sum_monkey(ai))];
    utils_diary(fid_log, '%s', row_str);
end
utils_diary(fid_log, '%s', sep_str);
tot_str = sprintf('  %-8s', 'Total');
for mi = 1:length(monkey_list)
    tot_str = [tot_str sprintf('  %6d', nb_units_sum_area(mi))];
end
tot_str = [tot_str sprintf('  %8d\n', sum(nb_units(:)))];
utils_diary(fid_log, '%s', tot_str);
utils_diary(fid_log, '\n');

% Display sum across areas (per monkey)
sum_across_areas = array2table(nb_units_sum_area, 'VariableNames', monkey_list);
utils_diary(fid_log, 'Sum across areas (per monkey):\n');
disp(sum_across_areas)

% Overall LME for firing rate
utils_diary(fid_log, '\n--- LME: diff FR ~ area + (1|monkey) ---\n');
anova_overall = anova(fitlme(table(info.fr_2AFC-info.fr_1FC, grouped_area, mk, 'VariableNames', {'diff_fr' 'area' 'monkey'}),'diff_fr ~ area + (1|monkey)'));
disp(anova_overall);
% Write formatted overall LME table
utils_diary(fid_log, '\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  OVERALL LME: diff_FR ~ area + (1|monkey)\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  %-14s  %8s  %4s  %6s  %12s  %s\n', 'Term', 'F-stat', 'DF1', 'DF2', 'p-value', 'Sig');
utils_diary(fid_log, '  %-14s  %8s  %4s  %6s  %12s\n', '--------------', '--------', '----', '------', '------------');
for ri = 1:height(anova_overall)
    pval = anova_overall.pValue(ri);
    if pval < 0.001, sig_str = '***'; elseif pval < 0.01, sig_str = '**'; elseif pval < 0.05, sig_str = '*'; else, sig_str = ''; end
    utils_diary(fid_log, '  %-14s  %8.3f  %4d  %6d  %12.2e  %s\n', anova_overall.Term{ri}, anova_overall.FStat(ri), anova_overall.DF1(ri), anova_overall.DF2(ri), pval, sig_str);
end
utils_diary(fid_log, '\n');

% Initialize result arrays for summary tables
fr_res    = zeros(length(area2test), 4);   % [N  FStat  DF2  pValue]
fano_res  = zeros(length(area2test), 4);
burst_res = zeros(length(area2test), 4);

% stat model (mixed effect) to see whether firing rate is significantly differente between 1FC and 2AFC task for each area and random effect monkey 
% plot the diff_fr for each area, the average of diff_fr, and whether significantly different from 0, for both monkey combined
figure;
% Subplot 1: Difference in Firing Rate
subplot(1,3,1)
hold on;
for ar = 1 : length(area2test)
    idx = ismember(info.area,area2test{ar});
    tbl = table(info.fr_2AFC(idx)-info.fr_1FC(idx),info.area(idx),mk(idx),'VariableNames',{'diff_fr' 'area' 'monkey'});
    lme = fitlme(tbl,'diff_fr ~ 1 + (1|monkey)');
    res = anova(lme);
    disp(area2test_name{ar})
    disp(res)
    fr_res(ar,:) = [height(tbl) res.FStat(1) res.DF2(1) res.pValue(1)];
    boxplot(info.fr_2AFC(idx)-info.fr_1FC(idx), 'Positions', ar, 'Widths', 0.5, 'Symbol', '')
    
    % Indicate significance
    if res.pValue(1) < 0.01
        text(ar, nanmean(tbl.diff_fr), '*', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 14, 'Color', 'r')
    end
end
set(gca, 'XTick', 1:length(area2test), 'XTickLabel', area2test_name)
xtickangle(45)
ylim([-5 5])
ylabel('Difference in Firing Rate (2AFC - 1FC)')
title('Firing Rate')
hold off;

% Write FR summary table
utils_diary(fid_log, '\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  FIRING RATE: diff (2AFC - 1FC) per area\n');
utils_diary(fid_log, '  Model: diff_fr ~ 1 + (1|monkey)\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  %-8s  %6s  %8s  %6s  %12s  %s\n', 'Area', 'N', 'F-stat', 'DF2', 'p-value', 'Sig');
utils_diary(fid_log, '  %-8s  %6s  %8s  %6s  %12s\n', '--------', '------', '--------', '------', '------------');
for ar = 1:length(area2test_name)
    pval = fr_res(ar,4);
    if pval < 0.001, sig_str = '***'; elseif pval < 0.01, sig_str = '**'; elseif pval < 0.05, sig_str = '*'; else, sig_str = ''; end
    utils_diary(fid_log, '  %-8s  %6d  %8.3f  %6d  %12.2e  %s\n', area2test_name{ar}, fr_res(ar,1), fr_res(ar,2), fr_res(ar,3), pval, sig_str);
end
utils_diary(fid_log, '\n');

% Subplot 2: Difference in Fano Factor
subplot(1,3,2)
hold on;
for ar = 1 : length(area2test)
    idx = ismember(info.area,area2test{ar});
    tbl = table(info.fano_2AFC(idx)-info.fano_1FC(idx),info.area(idx),mk(idx),'VariableNames',{'diff_fano' 'area' 'monkey'});
    lme = fitlme(tbl,'diff_fano ~ 1 + (1|monkey)');
    res = anova(lme);
    disp(area2test_name{ar})
    disp(res)
    fano_res(ar,:) = [height(tbl) res.FStat(1) res.DF2(1) res.pValue(1)];
    boxplot(info.fano_2AFC(idx)-info.fano_1FC(idx), 'Positions', ar, 'Widths', 0.5, 'Symbol', '')
    
    % Indicate significance
    if res.pValue(1) < 0.01
        text(ar, nanmean(tbl.diff_fano), '*', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 14, 'Color', 'r')
    end
end
set(gca, 'XTick', 1:length(area2test), 'XTickLabel', area2test_name)
xtickangle(45)
ylim([-5 5])
ylabel('Difference in Fano Factor (2AFC - 1FC)')
title('Fano Factor')
hold off;

% Write Fano summary table
utils_diary(fid_log, '\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  FANO FACTOR: diff (2AFC - 1FC) per area\n');
utils_diary(fid_log, '  Model: diff_fano ~ 1 + (1|monkey)\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  %-8s  %6s  %8s  %6s  %12s  %s\n', 'Area', 'N', 'F-stat', 'DF2', 'p-value', 'Sig');
utils_diary(fid_log, '  %-8s  %6s  %8s  %6s  %12s\n', '--------', '------', '--------', '------', '------------');
for ar = 1:length(area2test_name)
    pval = fano_res(ar,4);
    if pval < 0.001, sig_str = '***'; elseif pval < 0.01, sig_str = '**'; elseif pval < 0.05, sig_str = '*'; else, sig_str = ''; end
    utils_diary(fid_log, '  %-8s  %6d  %8.3f  %6d  %12.2e  %s\n', area2test_name{ar}, fano_res(ar,1), fano_res(ar,2), fano_res(ar,3), pval, sig_str);
end
utils_diary(fid_log, '\n');

% Subplot 3: Difference in Burstiness
subplot(1,3,3)
hold on;
for ar = 1 : length(area2test)
    idx = ismember(info.area,area2test{ar});
    tbl = table(double(info.burstiness_2AFC(idx)-info.burstiness_1FC(idx)),info.area(idx),mk(idx),'VariableNames',{'diff_burstiness' 'area' 'monkey'});
    % remove NaN for burstiness
    tbl = tbl(~isnan(tbl.diff_burstiness),:);

    lme = fitlme(tbl,'diff_burstiness ~ 1 + (1|monkey)');
    res = anova(lme);
    disp(area2test_name{ar})
    disp(res)
    burst_res(ar,:) = [height(tbl) res.FStat(1) res.DF2(1) res.pValue(1)];
    boxplot(info.burstiness_2AFC(idx)-info.burstiness_1FC(idx), 'Positions', ar, 'Widths', 0.5, 'Symbol', '')
    
    % Indicate significance
    if res.pValue(1) < 0.01
        text(ar, mean(tbl.diff_burstiness), '*', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 14, 'Color', 'r')
    end
end
set(gca, 'XTick', 1:length(area2test), 'XTickLabel', area2test_name)
ylim([-.25 .25])
xtickangle(45)
ylabel('Difference in Burstiness (2AFC - 1FC)')
title('Burstiness')
hold off;

% Write Burstiness summary table
utils_diary(fid_log, '\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  BURSTINESS: diff (2AFC - 1FC) per area\n');
utils_diary(fid_log, '  Model: diff_burstiness ~ 1 + (1|monkey)\n');
utils_diary(fid_log, '============================================================\n');
utils_diary(fid_log, '  %-8s  %6s  %8s  %6s  %12s  %s\n', 'Area', 'N', 'F-stat', 'DF2', 'p-value', 'Sig');
utils_diary(fid_log, '  %-8s  %6s  %8s  %6s  %12s\n', '--------', '------', '--------', '------', '------------');
for ar = 1:length(area2test_name)
    pval = burst_res(ar,4);
    if pval < 0.001, sig_str = '***'; elseif pval < 0.01, sig_str = '**'; elseif pval < 0.05, sig_str = '*'; else, sig_str = ''; end
    utils_diary(fid_log, '  %-8s  %6d  %8.3f  %6d  %12.2e  %s\n', area2test_name{ar}, burst_res(ar,1), burst_res(ar,2), burst_res(ar,3), pval, sig_str);
end
utils_diary(fid_log, '\n');

saveas(gcf, [report_dir 'Fig_S3b_spiking.png']);

%% plot distribution of Fano factor, burstiness, and firing rate for each task (averaged across areas)
col_task = [100 60 150 ; 250 130 190]/255;

figure;
% Subplot 1: Firing Rate
subplot(1,3,1)
[f1, x1] = hist(log(info.fr_1FC), 30);
[f2, x2] = hist(log(info.fr_2AFC), 30);
f1 = f1 / sum(f1); % Convert to probability
f2 = f2 / sum(f2); % Convert to probability
plot(x1, f1, 'Color',col_task(1,:) , 'LineWidth', 2);
hold on;
plot(x2, f2, 'Color',col_task(2,:) , 'LineWidth', 2);
set(gca, 'XTick', log([0.1 1 10 100]), 'XTickLabel', {'0.1', '1', '10', '100'})
xlabel('Log(Firing Rate) (Hz)')
ylabel('Probability')
legend('1FC', '2AFC')
title('Firing Rate')
% Subplot 2: Fano Factor
subplot(1,3,2)
[f1, x1] = hist(log(info.fano_1FC), 30);
[f2, x2] = hist(log(info.fano_2AFC), 30);
f1 = f1 / sum(f1); % Convert to probability
f2 = f2 / sum(f2); % Convert to probability
plot(x1, f1, 'Color',col_task(1,:) , 'LineWidth', 2);
hold on;
plot(x2, f2, 'Color',col_task(2,:) , 'LineWidth', 2);
set(gca, 'XTick', log([0.1 1 10 100]), 'XTickLabel', {'0.1', '1', '10', '100'})
xlabel('Log(Fano Factor)')
ylabel('Probability')
legend('1FC', '2AFC')
title('Fano Factor')
hold off;

subplot(1,3,3)
[f1, x1] = hist(info.burstiness_1FC, 30);
[f2, x2] = hist(info.burstiness_2AFC, 30);
f1 = f1 / sum(f1); % Convert to probability
f2 = f2 / sum(f2); % Convert to probability
plot(x1, f1, 'Color',col_task(1,:) , 'LineWidth', 2);
hold on;
plot(x2, f2, 'Color',col_task(2,:) , 'LineWidth', 2);
xlabel('Burstiness')
ylabel('Probability')
legend('1FC', '2AFC')
title('Burstiness')
hold off;

saveas(gcf, [report_dir 'Fig_S3a_spiking.png']);
