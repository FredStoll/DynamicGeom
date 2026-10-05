%% main_002_states.m
%
% ============================================================
% State decoding analysis for 2AFC sessions.
% - Builds states_2afc_final.mat in processed, used by all later scripts
% - make report_002 figures and statistics:
%   Fig 2B-F, 2H and Fig S5A-D (plus response-to-reviewer Figs R5-R6).
%
% Dependencies: utils_decoding_crosstask_rmvarea, utils_createpseudopop,
%               utils_fitDecod, utils_designvec_ab, utils_areaposthoc,
%               utils_groupvector, utils_fdr_bh
% ============================================================

clear

overwrite = false; % if you want to redo the matrices

f = mfilename('fullpath');
if isempty(f) 
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end
addpath(genpath([currentPath '\scripts\'])); % add scripts folder to path

param.path2go  = [currentPath '\data_final\'];
param.path2proc = [currentPath '\processed\'];

if ~exist(param.path2proc, 'dir'), mkdir(param.path2proc); end
report_dir = [currentPath '\report_002\'];
if ~exist(report_dir, 'dir'), mkdir(report_dir); end
fid_log = fopen([report_dir 'output_002.txt'], 'w');
log_cleanup = onCleanup(@() fclose(fid_log));
utils_diary(fid_log, '\n============================================================\n');
utils_diary(fid_log, 'Report generated: %s\n', datestr(now));

param.labels = [10 30 50 70 90];
param.juice2keep = [1 2]; % make it so that it can take both values (can be 1, 2 or [1 2])
param.Same_juice_only = false;
param.trainjuice = [1 2]; % juices to use for training
param.alignment = 'stim_on'; % align to stim_on
param.baseline_norm = [-900 -100]; % to normalize raw firing rate (independently for 1FC and 2AFC)
param.min_firing_threshold = .5; % 0.05; % Define minimum firing rate threshold
param.bin2train = [200 700]; % training 1FC classifier on the average activity during that time
param.kfold = 10; % to assess cross-validation pav-pav perf. cross decoding is done with all Pav trials! 
param.time_state = [0 800]; % time period to count states
param.thr_state = 0.3; % state threshold (state computed on raw prediction score (chance would be 0.2, so anything >0.3/0.4 should be decent). 
% It also requires that the prediction is > threshold AND the highest of all 3 predictions
param.thr_state_dur = 5; % how many consecutive bins it should be > threshold + MAX

param.area2test =      {'24c' {'6DR' '6DC' '6Va/Vb'}  {'8' '8B' '8A' '46d' '46df' '46v'}    {'IFG' '44' '45'} {'12r' '12m' '12m/r' '12o' '12l'}   'AI'  {'13l' '13m' '11m/l'} {'cd'  'pu'}  'AMG'};
param.area2test_name = {'MFC' 'PMC'  'dlPFC'            'IFG'       'vlPFC'   'AI'  'OFC' 'STR'  'AMG'};
   
if length(param.trainjuice) ~= 2
    endname  = ['juice' num2str(param.trainjuice)];
else
    endname = 'final';
    % endname = 'final_timept';
end

if overwrite || ~exist([param.path2proc 'states_2afc_' endname '.mat'], 'file')

    list = dir([param.path2go '*_pool.mat']);

    % reorder files by dates 
    for i = 1 : length(list)
        dates(i) = datenum(list(i).name(2:7),'mmddyy');
    end
    [~,idx] = sort(dates);  
    list = list(idx);

    out_all = [];
    for i = 1 : length(list)

        disp(['Processing session ' num2str(i) ' of ' num2str(length(list)) '...'])
        session = list(i).name(1:7);
        try 
            out = utils_decoding_crosstask_rmvarea(session,param);
            % out = utils_decoding_crosstask_rmvarea_timept(session,param);
            out_all = [out_all out];
        catch
            disp('#################################### ')
            disp('#################################### ')
            disp(['     Error in session ' session]);
            disp('#################################### ')
            disp('#################################### ')

        end
    end
    save([param.path2proc 'states_2afc_' endname '.mat'],"out_all","list","param",'-v7.3')

else
    disp('Loading existing data...')
    load([param.path2proc 'states_2afc_' endname '.mat'])
end

%% some initial param for posthoc

area2test = {'MFC' 'PMC'  'dlPFC'            'IFG'       'vlPFC'   'AI'  'OFC' 'STR'  'AMG'};
area2test_name = {'MFC' 'PMC'  'dlPFC'            'IFG'       'vlPFC'   'AI'  'OFC' 'STR'  'AMG'};
colorareas = [230 171 2 ; 152 78 163 ; 237 87 90 ; 252 141 98 ; 141 160 203 ; ...
     166 216 84 ; 102 194 165 ; 180 180 180 ; 231 138 195];

col = [100 200 160 ; 240 90 90 ; 0 0 0 ]/255; % state colors
col_light = col + (1-col)*0.6; % lighter version for dots

mk_col = [51 128 230; 140 89 38]/255; % Monkey colors

% keep only the sessions without dropping any area
out = out_all(cellfun(@isempty, {out_all(:).area4unit_removed}));

%% count number of areas and neurons per session (response to reviewers R2.2 / R3.m1)
out_rmv = out_all(~cellfun(@isempty, {out_all(:).area4unit_removed}));
% from out_rmv, get session and length of 1st dim in fr_heldout into a table
tbl = table();
for i = 1:length(out_rmv)
    tbl.session{i} = out_rmv(i).session;
    tbl.n_units(i) = size(out_rmv(i).fr_heldout, 1);
end
% for each unique session, count how many time it's name is in the table
unique_sessions = unique(tbl.session);
% keep the first letter of unique_sessions to get the monkey name
for i = 1:length(unique_sessions)
    monkey{i} = unique_sessions{i}(1);
end
for i = 1:length(unique_sessions)
    n_duplicates(i) = sum(strcmp(tbl.session, unique_sessions{i}));
end
tbl2 = table(unique_sessions, monkey', n_duplicates', 'VariableNames', {'session', 'monkey', 'n_duplicates'});

% per monkey get median/min/max of the number of areas per session
median_areas = NaN(length(unique(tbl2.monkey)),3);
for i = 1:length(unique(tbl2.monkey))
    monkey_name = unique(tbl2.monkey);
    monkey_name = monkey_name{i};
    idx = strcmp(tbl2.monkey, monkey_name);
    median_areas(i,:) = [median(tbl2.n_duplicates(idx)) min(tbl2.n_duplicates(idx)) max(tbl2.n_duplicates(idx))];
end

clear monkey
for i = 1:height(tbl)
    monkey(i) = tbl.session{i}(1);
end
tbl.monkey = monkey';

median_units = NaN(length(unique(tbl.monkey)),3);
for i = 1:length(unique(tbl.monkey))
    monkey_name = unique(tbl.monkey);
    monkey_name = monkey_name(i);
    idx = ismember(tbl.monkey, monkey_name);
    median_units(i,:) = [median(tbl.n_units(idx)) min(tbl.n_units(idx)) max(tbl.n_units(idx))];
end

% from out (full population), get session and length of 1st dim in fr_heldout into a table
tbl3 = table();
for i = 1:length(out)
    tbl3.session{i} = out(i).session;
    tbl3.n_units(i) = size(out(i).fr_heldout, 1);
end
% keep the first letter of each session to get the monkey name
tbl3.monkey = cellfun(@(s) s(1), tbl3.session, 'UniformOutput', false);

% per monkey get median/min/max of the number of neurons per session
median_units_sess = NaN(length(unique(tbl3.monkey)),3);
for i = 1:length(unique(tbl3.monkey))
    monkey_name = unique(tbl3.monkey);
    monkey_name = monkey_name{i};
    idx = strcmp(tbl3.monkey, monkey_name);
    median_units_sess(i,:) = [median(tbl3.n_units(idx)) min(tbl3.n_units(idx)) max(tbl3.n_units(idx))];
end

mk_list = unique(tbl3.monkey);
for i = 1:length(mk_list)
    utils_diary(fid_log, 'Monkey %s - areas/session: median %g [%g-%g] | neurons/area: median %g [%g-%g] | neurons/session: median %g [%g-%g]\n', ...
        mk_list{i}, median_areas(i,:), median_units(i,:), median_units_sess(i,:));
end

%% REVISION - topology of probability centroids 

num_sessions = length(out);
true_prob_dists = out(1,1).topology.true_prob_dists; % Constant across sessions
num_pairs = length(true_prob_dists);

% Preallocate arrays to gather data across all 360 sessions
all_rhos = NaN(num_sessions, 1);
all_pvals = NaN(num_sessions, 1);
all_pc1_explained = NaN(num_sessions, 1);
all_neural_dists = NaN(num_sessions, num_pairs);

for u = 1:num_sessions
    if isfield(out(1,u), 'topology') && ~isempty(out(1,u).topology)
        all_rhos(u) = out(1,u).topology.rho;
        all_pvals(u) = out(1,u).topology.pval;
        all_neural_dists(u, :) = out(1,u).topology.neural_dists;
        
        % Guard against sessions where PCA couldn't calculate variance
        if ~isempty(out(1,u).topology.pca_explained)
            all_pc1_explained(u) = out(1,u).topology.pca_explained(1);
        end
    end
end

%- Distance Correlation (Parametric Scaling)
figure('Position', [100, 100, 900, 400]);

subplot(1,2,1);
histogram(all_rhos, 20, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'w');
hold on;
xline(nanmean(all_rhos), 'r-', 'LineWidth', 2);
xline(0, 'k--', 'LineWidth', 1.5);
title('Distribution of Distance Correlations (\rho)');
xlabel('Spearman \rho (Neural vs. True Distance)');
ylabel('Number of Sessions');
legend('Session \rho', sprintf('Mean \\rho = %.2f', nanmean(all_rhos)), 'Location', 'NorthWest');
grid on;

subplot(1,2,2);
% Calculate mean and Standard Error of the Mean (SEM) across sessions
mean_neural_dists = nanmean(all_neural_dists, 1);
sem_neural_dists = nanstd(all_neural_dists, 0, 1) ./ sqrt(sum(~isnan(all_neural_dists(:,1))));
errorbar(true_prob_dists, mean_neural_dists, sem_neural_dists, 'o', ...
    'MarkerFaceColor', [0.2 0.6 0.8], 'MarkerEdgeColor', 'w', ...
    'LineWidth', 1.5, 'CapSize', 0, 'MarkerSize', 8);
hold on;

p_fit = polyfit(true_prob_dists, mean_neural_dists, 1);
x_fit = linspace(min(true_prob_dists), max(true_prob_dists), 100);
y_fit = polyval(p_fit, x_fit);
plot(x_fit, y_fit, 'r-', 'LineWidth', 1.5);

title('Average Population Scaling');
xlabel('\Delta Probability (Mathematical Difference)');
ylabel('Average Neural Distance');
legend('Population Mean \pm SEM', 'Linear Fit', 'Location', 'NorthWest');
grid on;
saveas(gcf, [report_dir 'Fig_R5ab_topology_distances.png']);

%- Linearity + PCA Check
figure('Position', [150, 150, 450, 400]);
histogram(all_pc1_explained, 20, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'w');
hold on;
xline(nanmean(all_pc1_explained), 'r-', 'LineWidth', 2);
title('Linearity of Probability Topology');
xlabel('Variance Explained by PC1 (%)');
ylabel('Number of Sessions');
legend('Sessions', sprintf('Mean PC1 Variance = %.1f%%', nanmean(all_pc1_explained)), 'Location', 'NorthWest');
grid on;
saveas(gcf, [report_dir 'Fig_R5c_topology_pc1.png']);

%% FIG 2B - Plot example trials decoding performance for some sessions

fig(1);
example_sess = randperm(length(list),24);

x = 0;
for i = example_sess
    if ~isempty(out(i).adjusted_prediction_scores_rand)
        example_trials = randperm(size(out(i).adjusted_prediction_scores_rand,1),1);
        time_sub = out(i).time>=-500 & out(i).time<=1000;
        pred_norm = out(i).adjusted_prediction_scores_rand; %./ repmat(sum(adjusted_prediction_scores_rand,3),1,1,3);
        for tr = example_trials
            x = x + 1;
            subplot(6,4,x)

            for c = 1 : 3
                plot(out(i).time(time_sub),squeeze(pred_norm(tr,time_sub,c)),"Color",col(c,:));hold on
                % shading with sem using patch
                %y1 = mean(squeeze(pred_norm(:,time_sub,c))) - std(squeeze(pred_norm(:,time_sub,c))) / sqrt(size(pred_norm,1));
                %y2 = mean(squeeze(pred_norm(:,time_sub,c))) + std(squeeze(pred_norm(:,time_sub,c))) / sqrt(size(pred_norm,1));
                %x2 = [out(i).time(time_sub) fliplr(out(i).time(time_sub))];
                %inBetween = [y1 fliplr(y2)];
                %fill(x2, inBetween, col_light(c,:),'FaceAlpha',0.2,'EdgeColor','none');
            end
            chosen_pb = out(i).cond.chosenproba_2AFC(tr);
            unchosen_pb = out(i).cond.unchosenproba_2AFC(tr);
            title([list(i).name(1:7) ' - Ch= ' num2str(chosen_pb) ' - Unch= ' num2str(unchosen_pb)])
        end
        xlim([-500 1000])
    end
end
saveas(gcf, [report_dir 'Fig_2b_example_trials.png']);

%% FIG 2C-D and S5A - extract number, duration and start time of states + PLOT/STATISTICS

monks = cellfun(@(x) x(1), {list.name}, 'UniformOutput', false)';
mk_names = {'M','X'};

sess_num = [1:sum(strcmp(monks, 'M')) , 1:sum(strcmp(monks, 'X'))]';

st_dur_list = [0 100:10:1000]/1000; % use that to convert number of bins to ms (bins 100ms / steps of 10ms)

nb_states = NaN(length(list),3);
dur_states = NaN(length(list),3);
t_states = NaN(length(list),3);
for i = 1 : length(list)
    if ~isempty(out(i).nb_states)
        nb_states(i,:) = (sum(out(i).nb_states))/length(out(i).nb_states);

        dur_temp = st_dur_list(out(i).dur_states+1); % convert to ms
        dur_temp(out(i).nb_states == 0) = NaN; % undefined duration when no state occurred
        dur_states(i,:) = mean(dur_temp, 1, 'omitnan');

        t_states(i,:) = nanmean(out(i).t_states)/1000;
    end
end

st_data = {nb_states, dur_states, t_states};
st_ylab = {'Number of states/trial', 'Duration of states (s)', 'State latencies (s)'};
st_ylim = {[0 2.05], [0.15 0.55], [0 0.4]};
[fg, cm] = fig_cm(22, 6.6);
for k = 1 : 3
    for m = 1 : length(mk_names)
        X   = st_data{k}(ismember(monks, mk_names{m}), :);
        rng(1); % for reproducibility of jitter
        jit = (rand(size(X,1),3)-0.5)*0.35;
        ax  = axes(fg, 'Position', cm(1.6 + ((k-1)*2 + m - 1)*3.6, 0.7, 2.1, 5.2));
        box_dots(ax, X, 1:3, col, col_light, {[1 2 3]}, jit);
        style_box_axes(ax, true, st_ylab{k}, st_ylim{k}, [0.4 3.6]);
        set(ax, 'XTick', []);
        text(ax, 2, st_ylim{k}(2), ['mk ' mk_names{m}], 'FontSize', 11, 'FontAngle', 'italic', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end
state_legend(fg, cm(1.6 + 4*3.6 + 0.15, 0.8, 1.8, 1.3), {'Chosen', 'Unchosen', 'Other'}, col);
save_fig(fg, [report_dir 'Fig_2cd_S5a_states']);

% stat on nb_states with states as predictors (chosen, unchosen, other) across sessions and monkey as random effect (fitglme)
data_table = table();
data_table.nb_states = nb_states(:);
data_table.dur_states = dur_states(:);
data_table.t_states = t_states(:);
temp = repmat(monks', 1, 3);
data_table.monkey = temp(:); 
temp = repmat({'Chosen'; 'Unchosen'; 'Other'}', length(list), 1);
data_table.state = temp(:);
% temp = repmat((1:length(list))',1, 3);
temp = repmat(sess_num,1, 3);
data_table.session = categorical(temp(:)); % session as categorical

mdl = fitlme(data_table, 'nb_states ~ 1 + state + (1|monkey) + (1|monkey:session)');
utils_diary(fid_log, 'LME results for number of states:\n');
utils_diary(fid_log, '%s', evalc('disp(mdl.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(mdl)'));

mdl2 = fitlme(data_table, 'dur_states ~ 1 + state + (1|monkey) + (1|monkey:session)');
utils_diary(fid_log, 'LME results for duration of states:\n');
utils_diary(fid_log, '%s', evalc('disp(mdl2.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(mdl2)'));

mdl3 = fitlme(data_table, 't_states ~ 1 + state + (1|monkey) + (1|monkey:session)');
utils_diary(fid_log, 'LME results for start time of states:\n');
utils_diary(fid_log, '%s', evalc('disp(mdl3.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(mdl3)'));

%% real vs shuffle LME (label shuffle control)
% Tests whether detected states exceed what a decoder with NO probability
% information would find. The condition x state interaction is key:
% does real exceed shuffle more for chosen/unchosen than for other?

nb_states_shuf  = NaN(length(list), 3);
dur_states_shuf = NaN(length(list), 3);
for i = 1:length(list)
    if ~isempty(out(i).nb_states_shuffle)
        nb_states_shuf(i,:)  = mean(out(i).nb_states_shuffle, 1, 'omitnan');
        dur_temp_shuf        = st_dur_list(out(i).dur_states_shuffle + 1); % convert bins to s
        dur_temp_shuf(out(i).nb_states_shuffle == 0) = NaN; % undefined when no state
        dur_states_shuf(i,:) = mean(dur_temp_shuf, 1, 'omitnan');
    end
end

% Stack real + shuffle into long-format table (6 rows per session: 3 states x 2 conditions)
nb_combined  = [nb_states(:);  nb_states_shuf(:)];
dur_combined = [dur_states(:); dur_states_shuf(:)];

temp_cond  = [repmat({'real'}, length(list)*3, 1); repmat({'shuffle'}, length(list)*3, 1)];
temp_state = repmat({'Chosen'; 'Unchosen'; 'Other'}', length(list), 1);
temp_state = [temp_state(:); temp_state(:)];
temp_monk_shuf  = repmat(monks', 1, 3); temp_monk_shuf  = [temp_monk_shuf(:); temp_monk_shuf(:)];
temp_sess_shuf  = repmat(sess_num, 1, 3); temp_sess_shuf = categorical([temp_sess_shuf(:); temp_sess_shuf(:)]);

shuf_table = table(nb_combined, dur_combined, temp_cond, temp_state, temp_monk_shuf, temp_sess_shuf, ...
    'VariableNames', {'nb_states', 'dur_states', 'condition', 'state', 'monkey', 'session'});

% nb_states: condition x state interaction
mdl_shuf_nb = fitlme(shuf_table, 'nb_states ~ 1 + condition * state + (1|monkey) + (1|monkey:session)');
utils_diary(fid_log, 'LME results: nb_states real vs shuffle (label permutation control):\n');
utils_diary(fid_log, '%s', evalc('disp(mdl_shuf_nb.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(mdl_shuf_nb)'));

% dur_states: condition x state interaction
mdl_shuf_dur = fitlme(shuf_table, 'dur_states ~ 1 + condition * state + (1|monkey) + (1|monkey:session)');
utils_diary(fid_log, 'LME results: dur_states real vs shuffle (label permutation control):\n');
utils_diary(fid_log, '%s', evalc('disp(mdl_shuf_dur.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(mdl_shuf_dur)'));

%% FIG 2E and S5B - ratio of chosen/unchosen against proba difference

diff_cd = [20 40 60];
nb_states_ch_unch = NaN(length(list),3);
nb_states_ch_oth = NaN(length(list),3);
dur_states_ch_unch = NaN(length(list),3);
dur_states_ch_oth = NaN(length(list),3);
t_states_ch_unch = NaN(length(list),3);
t_states_ch_oth = NaN(length(list),3);

for i = 1 : length(list)
    if ~isempty(out(i).nb_states)
        cds = abs(out(i).cond.chosenproba_2AFC - out(i).cond.unchosenproba_2AFC);
        for cd = 1 : length(diff_cd)
            take_idx = cds == diff_cd(cd);
            if sum(take_idx) > 0
                nb_states_cd(cd,:) = (sum(out(i).nb_states(take_idx,:)))/length(out(i).nb_states(take_idx,:));
                dur_temp = st_dur_list(out(i).dur_states(take_idx,:)+1); % convert to ms
                dur_temp(out(i).nb_states(take_idx,:) == 0) = NaN; % undefined duration when no state occurred
                dur_states_cd(cd,:) = mean(dur_temp, 1, 'omitnan');
                t_states_cd(cd,:) = nanmean(out(i).t_states(take_idx,:))/1000;
            else
                nb_states_cd(cd,:) = NaN(1,3);
                dur_states_cd(cd,:) = NaN(1,3);
                t_states_cd(cd,:) = NaN(1,3);
            end
        end
        nb_states_ch_unch(i,:) = (nb_states_cd(:,1) ./ nb_states_cd(:,2));
        nb_states_ch_oth(i,:) = (nb_states_cd(:,1) ./ nb_states_cd(:,3));
        dur_states_ch_unch(i,:) = (dur_states_cd(:,1) ./ dur_states_cd(:,2));
        dur_states_ch_oth(i,:) = (dur_states_cd(:,1) ./ dur_states_cd(:,3));
        t_states_ch_unch(i,:) = (t_states_cd(:,1) ./ t_states_cd(:,2));
        t_states_ch_oth(i,:) = (t_states_cd(:,1) ./ t_states_cd(:,3));
    end
end

rt_data = {nb_states_ch_unch, dur_states_ch_unch, t_states_ch_unch};
rt_ylab = {{'Ratio of Chosen/Unchosen', 'state number'}, {'Ratio of Chosen/Unchosen', 'state duration'}, ...
           {'Ratio of Chosen/Unchosen', 'state latency'}};
jitter = 0.35;
rng(1); % reproducible jitter
rt_X = cell(3, length(mk_names));  rt_jit = rt_X;
for m = 1:length(mk_names)
    for k = 1:3
        X = rt_data{k}(strcmp(monks, mk_names{m}), :);
        X = X(~all(isnan(X), 2), :);
        rt_X{k,m}   = X;
        rt_jit{k,m} = (rand(size(X,1),3)-0.5)*jitter;
    end
end

[fg, cm] = fig_cm(22.4, 6.8);
for k = 1:3
    yl = [0, max(cellfun(@(X) max([X(:); 0]), rt_X(k,:))) * 1.03];
    x0 = 1.9 + (k-1)*2*3.6;
    for m = 1:length(mk_names)
        ax = axes(fg, 'Position', cm(x0 + (m-1)*3.6, 1.5, 2.1, 4.6));
        box_dots(ax, rt_X{k,m}, 1:3, repmat(mk_col(m,:), 3, 1), repmat(mk_col(m,:)*0.4 + 0.6, 3, 1), ...
            {[1 2 3]}, rt_jit{k,m});
        style_box_axes(ax, true, rt_ylab{k}, yl, [0.4 3.6]);
        set(ax, 'XTick', 1:3, 'XTickLabel', arrayfun(@num2str, diff_cd, 'UniformOutput', false));
        text(ax, 2, yl(2), ['mk ' mk_names{m}], 'Color', mk_col(m,:), 'FontSize', 11, 'FontAngle', 'italic', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    fig_text(fg, cm(x0, 0.1, 3.6 + 2.1, 0.5), 'Proba difference (%)', 11);
end
save_fig(fg, [report_dir 'Fig_2e_S5b_ratio_vs_probadiff']);

%- Stat on ratios with diff (proba difference) as predictor using mixed effects (like before)
% Build table with stacked data (rows = sessions x diff levels)
data_table_ratio = table();
data_table_ratio.nb_ratio = nb_states_ch_unch(:);
data_table_ratio.dur_ratio = dur_states_ch_unch(:);
data_table_ratio.t_ratio = t_states_ch_unch(:);
temp = repmat(monks', 1, length(diff_cd));
data_table_ratio.monkey = temp(:);
temp = repmat(diff_cd, length(list), 1);
data_table_ratio.diff = categorical(temp(:));
temp = repmat(sess_num, 1, length(diff_cd));
data_table_ratio.session = categorical(temp(:));

% LME for nb_ratio
tbl_nb = data_table_ratio(~isnan(data_table_ratio.nb_ratio), :);
if ~isempty(tbl_nb)
    mdl_nb = fitlme(tbl_nb, 'nb_ratio ~ 1 + diff + (1|monkey) + (1|monkey:session)');
    utils_diary(fid_log, 'LME results for number-of-states ratio (Chosen/Unchosen):\n');
    utils_diary(fid_log, '%s', evalc('disp(mdl_nb.Coefficients)'));
    utils_diary(fid_log, '%s', evalc('anova(mdl_nb)'));
end

% LME for dur_ratio
tbl_dur = data_table_ratio(~isnan(data_table_ratio.dur_ratio), :);
if ~isempty(tbl_dur)
    mdl_dur = fitlme(tbl_dur, 'dur_ratio ~ 1 + diff + (1|monkey) + (1|monkey:session)');
    utils_diary(fid_log, 'LME results for duration-of-states ratio (Chosen/Unchosen):\n');
    utils_diary(fid_log, '%s', evalc('disp(mdl_dur.Coefficients)'));
    utils_diary(fid_log, '%s', evalc('anova(mdl_dur)'));
end

% LME for t_ratio
tbl_t = data_table_ratio(~isnan(data_table_ratio.t_ratio), :);
if ~isempty(tbl_t)
    mdl_t = fitlme(tbl_t, 't_ratio ~ 1 + diff + (1|monkey) + (1|monkey:session)');
    utils_diary(fid_log, 'LME results for start-time-of-states ratio (Chosen/Unchosen):\n');
    utils_diary(fid_log, '%s', evalc('disp(mdl_t.Coefficients)'));
    utils_diary(fid_log, '%s', evalc('anova(mdl_t)'));
end

%% ratio of chosen/unchosen states number against preference bias
load([param.path2proc 'behav_pref.mat'],'preference');
ratio_ch_unch = nb_states(:,1) ./ nb_states(:,2);
ratio_dur_ch_unch = dur_states(:,1) ./ dur_states(:,2);

figure;
for m = 1:length(mk_names)
    mk_idx = strcmp(monks, mk_names{m});
    valid_idx = mk_idx & ~isnan(ratio_ch_unch) & ~isnan(preference.bias_point);
    valid_idx_dur = mk_idx & ~isnan(ratio_dur_ch_unch) & ~isnan(preference.bias_point);

    subplot(2,2,2*m-1)
    scatter(abs(preference.bias_point(valid_idx)), ratio_ch_unch(valid_idx), 50, 'o', 'MarkerEdgeColor', mk_col(m,:)); hold on;
    xlabel('|Preference Bias Point|');
    ylabel('Ratio of Ch/Unch states');
    [r_ratio, p_ratio] = corrcoef(abs(preference.bias_point(valid_idx)), ratio_ch_unch(valid_idx));
    title([mk_names{m} ': Ratio Ch/Unch states vs |Pref Bias| (r=' num2str(r_ratio(1,2),'%.2f') ', p=' num2str(p_ratio(1,2),'%.4f') ')']);

    subplot(2,2,2*m)
    scatter(abs(preference.bias_point(valid_idx_dur)), ratio_dur_ch_unch(valid_idx_dur), 50, 'o', 'MarkerEdgeColor', mk_col(m,:)); hold on;
    xlabel('|Preference Bias Point|');
    ylabel('Ratio of Ch/Unch state duration');
    [r_dur_ratio, p_dur_ratio] = corrcoef(abs(preference.bias_point(valid_idx_dur)), ratio_dur_ch_unch(valid_idx_dur));
    title([mk_names{m} ': Ratio Ch/Unch dur vs |Pref Bias| (r=' num2str(r_dur_ratio(1,2),'%.2f') ', p=' num2str(p_dur_ratio(1,2),'%.4f') ')']);
end

saveas(gcf, [report_dir 'Fig_extra_pref_states.png']); 

%% look at posteriors and saccades

% average across sessions
time_bl = out(1).time>=-500 & out(1).time<=0;

pred_all = [];
x = 0;
for sess = 1 : length(out)
    if ~isempty(out(sess).adjusted_prediction_scores_rand)
        x=x+1;
        pred_norm = out(sess).adjusted_prediction_scores_rand; 
        pred_all(x,:,:) = squeeze(mean(pred_norm(:,time_sub,:),1));
    end
end

% load saccades and compare decoding posteriors for quick vs hesitating saccades
load([param.path2proc 'saccadeCounts_reduced.mat']); % table with saccade counts per trial

for sess = 1 : length(out)
    if ~isempty(out(sess).adjusted_prediction_scores_rand)
        out(sess).saccades = saccadeTable.NumSaccades(ismember(string(saccadeTable.SessionID), string(list(sess).name(1:7))) & ismember(saccadeTable.Trial,out(sess).keepTr));
    end
end


time_sub = out(sess).time>=-500 & out(sess).time<=1000;

% random trials or match behavior
match_tr = true; % set to true to match behavior btw quick/hesit, false to take random trials (but matched in numbers)

% stats with repeated measures (monkey and nested session as random effect)
x = 0;
pred_quick = [];
pred_hesit = [];
nb_states_quick = [];
nb_states_hesit = [];
dur_states_quick = [];
dur_states_hesit = [];
for sess = 1 : length(out)
    % if ~isempty(out(sess).saccades) & strcmp(list(sess).name(1),'M')
    if ~isempty(out(sess).saccades) 
        x=x+1;
        pred_norm = out(sess).adjusted_prediction_scores_rand; 
        for c = 1 : 3
            pred_norm(:,:,c) = pred_norm(:,:,c) - mean(pred_norm(:,time_bl,c),2);
        end
        nb2take = min([sum(out(sess).saccades==1) sum(out(sess).saccades>1)]);
        quick_idx = find(out(sess).saccades==1);
        hesit_idx = find(out(sess).saccades>1);    
        if match_tr % match number but also match behavior as best as possible
            % get choice proba for quick and hesit
            quick_pb = [out(sess).cond.chosenproba_2AFC(quick_idx) out(sess).cond.unchosenproba_2AFC(quick_idx)];
            hesit_pb = [out(sess).cond.chosenproba_2AFC(hesit_idx) out(sess).cond.unchosenproba_2AFC(hesit_idx)];
            % for each hesit trial, find all quick trials with minimal distance and pick the one closest in trial/time
            used_quick = false(length(quick_idx),1);
            sel_quick = NaN(nb2take,1);
            for h = 1 : nb2take
                hesit_vector = hesit_pb(h,:);
                dists = sqrt(sum((quick_pb - hesit_vector).^2,2));
                dists(used_quick) = Inf; % exclude already used quick trials
                min_dist = min(dists);
                min_idx_all = find(abs(dists - min_dist) < 1e-10); % all indices with minimal distance (tolerance for floating point)
                if ~isempty(min_idx_all)
                    % Find quick_idx among min_idx_all that is closest in trial/time to hesit_idx(h)
                    [~, closest_idx] = min(abs(quick_idx(min_idx_all) - hesit_idx(h)));
                    pick_idx = min_idx_all(closest_idx);
                    sel_quick(h) = quick_idx(pick_idx);
                    used_quick(pick_idx) = true;
                else % if no match, find any unused quick trial
                    avail_idx = find(~used_quick);
                    sel_quick(h) = quick_idx(avail_idx(1));
                    used_quick(avail_idx(1)) = true;
                    warning('No matching quick trial found based on proba, using any unused quick trial instead.');
                end
            end

            % Only keep selected quick and hesit trials for pred_norm
            pred_norm = out(sess).adjusted_prediction_scores_rand([sel_quick; hesit_idx(1:nb2take)], :, :);
            for c = 1 : 3
            pred_norm(:,:,c) = pred_norm(:,:,c) - mean(pred_norm(:,time_bl,c),2);
            end

            % Split back to quick and hesit
            pred_quick(x,:,:) = squeeze(mean(pred_norm(1:nb2take,time_sub,:),1));
            pred_hesit(x,:,:) = squeeze(mean(pred_norm(nb2take+1:end,time_sub,:),1));

            % Extract nb_states and duration for matched quick/hesit trials
            if ~isempty(out(sess).nb_states) && ~isempty(out(sess).dur_states)
                st_dur_list_local = [0 100:10:1000]/1000; % bin-count to seconds
                nb_states_quick(x,:) = mean(out(sess).nb_states(sel_quick,:), 1);
                nb_states_hesit(x,:) = mean(out(sess).nb_states(hesit_idx(1:nb2take),:), 1);
                dur_q_mat = st_dur_list_local(out(sess).dur_states(sel_quick,:) + 1);
                dur_h_mat = st_dur_list_local(out(sess).dur_states(hesit_idx(1:nb2take),:) + 1);
                % NaN out trials where no state occurred (duration undefined when nb_states=0)
                dur_q_mat(out(sess).nb_states(sel_quick,:) == 0) = NaN;
                dur_h_mat(out(sess).nb_states(hesit_idx(1:nb2take),:) == 0) = NaN;
                dur_states_quick(x,:) = mean(dur_q_mat, 1, 'omitnan');
                dur_states_hesit(x,:) = mean(dur_h_mat, 1, 'omitnan');
            else
                nb_states_quick(x,:) = NaN(1,3);
                nb_states_hesit(x,:) = NaN(1,3);
                dur_states_quick(x,:) = NaN(1,3);
                dur_states_hesit(x,:) = NaN(1,3);
            end
        else
            sel_quick_rand = quick_idx(randperm(length(quick_idx), nb2take));
            sel_hesit_rand = hesit_idx(randperm(length(hesit_idx), nb2take));
            pred_quick(x,:,:) = squeeze(mean(pred_norm(sel_quick_rand, time_sub,:),1));
            pred_hesit(x,:,:) = squeeze(mean(pred_norm(sel_hesit_rand, time_sub,:),1));
            if ~isempty(out(sess).nb_states) && ~isempty(out(sess).dur_states)
                st_dur_list_local = [0 100:10:1000]/1000;
                nb_states_quick(x,:) = mean(out(sess).nb_states(sel_quick_rand,:), 1);
                nb_states_hesit(x,:) = mean(out(sess).nb_states(sel_hesit_rand,:), 1);
                dur_q_mat = st_dur_list_local(out(sess).dur_states(sel_quick_rand,:) + 1);
                dur_h_mat = st_dur_list_local(out(sess).dur_states(sel_hesit_rand,:) + 1);
                % NaN out trials where no state occurred (duration undefined when nb_states=0)
                dur_q_mat(out(sess).nb_states(sel_quick_rand,:) == 0) = NaN;
                dur_h_mat(out(sess).nb_states(sel_hesit_rand,:) == 0) = NaN;
                dur_states_quick(x,:) = mean(dur_q_mat, 1, 'omitnan');
                dur_states_hesit(x,:) = mean(dur_h_mat, 1, 'omitnan');
            else
                nb_states_quick(x,:) = NaN(1,3);
                nb_states_hesit(x,:) = NaN(1,3);
                dur_states_quick(x,:) = NaN(1,3);
                dur_states_hesit(x,:) = NaN(1,3);
            end
        end

        animal_names{x} = list(sess).name(1);  % Store animal names for each session

    end
end

% Preallocate
pval = NaN(size(pred_quick,2), 4);  % main effects + interaction
tukey_results = cell(size(pred_quick,2), 1);
tukey_tstat = cell(size(pred_quick,2), 1);
% Labels for groups
levels = {'quick-ch', 'hesit-ch', 'quick-unch', 'hesit-unch', 'quick-na', 'hesit-na'};
groups = {'Quick-Chosen','Hesit-Chosen','Quick-Unchosen','Hesit-Unchosen','Quick-Other','Hesit-Other'};
within_quick = [1 3; 1 5; 3 5];   % Comparisons within quick: ch-unch, ch-na, unch-na
within_hesit = [2 4; 2 6; 4 6];   % Comparisons within hesit: ch-unch, ch-na, unch-na
posthoc_comp = [1 2; 3 4];        % Quick-chosen vs Hesit-chosen, Quick-unchosen vs Hesit-unchosen
comparisons = [within_quick; within_hesit; posthoc_comp];
nComp = size(comparisons,1);

estimates = NaN(size(pred_quick,2), nComp);
signif = false(size(pred_quick,2), nComp);

% Loop over timepoints
for t = 1:size(pred_quick,2)

    % Collect data across sessions and trial types
    all_data = [];
    all_saccade = {};
    all_tr = {};
    all_session = [];
    all_animal = {};
    sess_mk = 0;
    new_mk = false;  % Flag to track new mk
    for sess = 1:size(pred_quick,1)
        sess_mk = sess_mk + 1;  % Increment session index for each session
        if ismember(animal_names(sess), 'X') & ~new_mk
            sess_mk = 1; new_mk = true;  % Reset for new mk
        end
        q = squeeze(pred_quick(sess,t,:));
        h = squeeze(pred_hesit(sess,t,:));
        all_data = [all_data; q; h];
        all_saccade = [all_saccade; repmat({'quick'}, numel(q), 1); repmat({'hesit'}, numel(h), 1)];

        tr = repmat({'ch','unch','na'}', 1, size(q,1)/3)';
        tr = tr(:);
        all_tr = [all_tr; tr; tr];  % repeat for both quick and hesit

        all_session = [all_session; repmat(sess_mk, numel(q)+numel(h), 1)];
        all_animal = [all_animal; repmat(animal_names(sess), numel(q)+numel(h), 1)];
    end

    % Create table
    tbl = table(all_data, all_saccade, all_tr, all_session, all_animal, ...
        'VariableNames', {'Data','Saccade','TrialType','Session','Animal'});

    % Fit linear mixed model with random intercepts for Session and Animal
    lme = fitlme(tbl, 'Data ~ Saccade*TrialType + (1|Animal) + (1|Animal:Session)');

    % Extract ANOVA table and store p-values
    anova_results = anova(lme);
    pval(t,:) = anova_results.pValue;  % [Saccade, TrialType, Interaction, Error]

    % Coefficients and names
    cnames = lme.CoefficientNames;
    beta = fixedEffects(lme);
    covB = lme.CoefficientCovariance;
    dof = lme.DFE;

    % Initialize storage for this timepoint
    tukey_results{t} = NaN(nComp, 6); % comp1, comp2, CI_low, diff, CI_high, pval
    tukey_tstat{t} = NaN(nComp, 1);

    % Calculate contrasts for defined comparisons
    for c = 1:nComp
        g1 = levels{comparisons(c,1)};
        g2 = levels{comparisons(c,2)};

        % Parse saccade and trial type from label strings
        s1 = split(g1, '-'); 
        s2 = split(g2, '-'); 

        % Use external helper function here:
        mu1 = utils_groupvector(cnames, s1{1}, s1{2});
        mu2 = utils_groupvector(cnames, s2{1}, s2{2});

        % Contrast vector = difference of predicted means
        C = mu1 - mu2;

        % Estimate difference and SE
        est_diff = C * beta;
        se = sqrt(C * covB * C');

        % t-statistic and p-value
        tval = est_diff / se;
        pval_t = 2 * (1 - tcdf(abs(tval), dof));

        % 95% Confidence interval
        ci_half = 1.96 * se;
        ci_low = est_diff - ci_half;
        ci_high = est_diff + ci_half;

        % Store results
        tukey_results{t}(c,:) = [comparisons(c,:), ci_low, est_diff, ci_high, pval_t];
        tukey_tstat{t}(c) = tval;
        estimates(t,c) = tval;
        signif(t,c) = pval_t < 0.05;  % uncorrected
    end
end

% === FDR correction across timepoints for each comparison ===
signif_fdr = false(size(pred_quick,2), nComp);  % Initialize FDR significance matrix
for c = 1:nComp
    p_all = cell2mat(cellfun(@(x) x(c,6), tukey_results, 'UniformOutput', false));
    [~, ~, adj_p] = utils_fdr_bh(p_all);
    signif_fdr(:,c) = adj_p < 0.05;
end
[~, ~, adj_pval_int] = utils_fdr_bh(pval(:,4));

[~, idxs] = utils_findenough(adj_pval_int',0.05,3,'<=');
sig_pval_int = false(size(adj_pval_int));
sig_pval_int(idxs) = true;

newtime = out(sess).time(time_sub);

%% FIG 2F and S5D -- State count & duration: Quick vs Hesit (probability-matched trials) --
% Compares nb of states and duration of states between quick and hesit trials,
% using the same probability-matched trial pairs as the posterior analysis above.
% Mirrors the across-all-trials analysis (line ~199) but here quick vs hesit.

nS_sq = size(nb_states_quick, 1);

% Build long-format table: type (quick/hesit) x state (chosen/unchosen/other)
nb_long_sq   = [nb_states_quick(:,1);  nb_states_quick(:,2);  nb_states_quick(:,3); ...
                nb_states_hesit(:,1);  nb_states_hesit(:,2);  nb_states_hesit(:,3)];
dur_long_sq  = [dur_states_quick(:,1); dur_states_quick(:,2); dur_states_quick(:,3); ...
                dur_states_hesit(:,1); dur_states_hesit(:,2); dur_states_hesit(:,3)];
type_long_sq = [repmat({'quick'},3*nS_sq,1); repmat({'hesit'},3*nS_sq,1)];
state_long_sq = [repmat({'Chosen'},nS_sq,1);   repmat({'Unchosen'},nS_sq,1); repmat({'Other'},nS_sq,1); ...
                 repmat({'Chosen'},nS_sq,1);   repmat({'Unchosen'},nS_sq,1); repmat({'Other'},nS_sq,1)];
monk_long_sq  = repmat(animal_names(:), 6, 1);
sess_long_sq  = categorical(repmat((1:nS_sq)', 6, 1));

tbl_sq = table(nb_long_sq, dur_long_sq, type_long_sq, state_long_sq, monk_long_sq, sess_long_sq, ...
    'VariableNames', {'NbStates','DurStates','Type','StateType','Monkey','Session'});
tbl_sq = tbl_sq(~isnan(tbl_sq.NbStates), :);

lme_nb_sq = fitlme(tbl_sq, 'NbStates ~ Type*StateType + (1|Monkey) + (1|Monkey:Session)');
utils_diary(fid_log, '=== LME: Number of states, Quick vs Hesit (prob-matched) ===\n');
utils_diary(fid_log, '%s', evalc('disp(lme_nb_sq.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(lme_nb_sq)'));

lme_dur_sq = fitlme(tbl_sq, 'DurStates ~ Type*StateType + (1|Monkey) + (1|Monkey:Session)');
utils_diary(fid_log, '=== LME: Duration of states, Quick vs Hesit (prob-matched) ===\n');
utils_diary(fid_log, '%s', evalc('disp(lme_dur_sq.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(lme_dur_sq)'));

% === Posthoc pairwise contrasts from the full model (same approach as posterior analysis) ===
% Build contrast vectors C = mu1 - mu2, then t = C*beta / sqrt(C*covB*C')
% Reference coding (alphabetical): Type ref='hesit', StateType ref='Chosen'

cnames_nb  = lme_nb_sq.CoefficientNames;
beta_nb    = fixedEffects(lme_nb_sq);
covB_nb    = lme_nb_sq.CoefficientCovariance;
dof_nb     = lme_nb_sq.DFE;

cnames_dur = lme_dur_sq.CoefficientNames;
beta_dur   = fixedEffects(lme_dur_sq);
covB_dur   = lme_dur_sq.CoefficientCovariance;
dof_dur    = lme_dur_sq.DFE;

% Columns: Type1, State1, Type2, State2, Label, xpos1, xpos2
ph_comps = { ...
    'quick','Chosen',   'hesit','Chosen',   'Ch: Q vs H',   1, 2; ...
    'quick','Unchosen', 'hesit','Unchosen', 'Unch: Q vs H', 3, 4; ...
    'quick','Other',    'hesit','Other',    'Other: Q vs H',5, 6; ...
    };
nPH = size(ph_comps, 1);

ph_pval_nb  = NaN(nPH,1);  ph_tstat_nb  = NaN(nPH,1);
ph_pval_dur = NaN(nPH,1);  ph_tstat_dur = NaN(nPH,1);

for ph = 1:nPH
    mu1_nb  = utils_sq_groupvec(cnames_nb,  ph_comps{ph,1}, ph_comps{ph,2});
    mu2_nb  = utils_sq_groupvec(cnames_nb,  ph_comps{ph,3}, ph_comps{ph,4});
    C_nb    = mu1_nb - mu2_nb;
    est_nb  = C_nb * beta_nb;
    se_nb   = sqrt(C_nb * covB_nb * C_nb');
    ph_tstat_nb(ph) = est_nb / se_nb;
    ph_pval_nb(ph)  = 2 * (1 - tcdf(abs(ph_tstat_nb(ph)), dof_nb));

    mu1_dur = utils_sq_groupvec(cnames_dur, ph_comps{ph,1}, ph_comps{ph,2});
    mu2_dur = utils_sq_groupvec(cnames_dur, ph_comps{ph,3}, ph_comps{ph,4});
    C_dur   = mu1_dur - mu2_dur;
    est_dur = C_dur * beta_dur;
    se_dur  = sqrt(C_dur * covB_dur * C_dur');
    ph_tstat_dur(ph) = est_dur / se_dur;
    ph_pval_dur(ph)  = 2 * (1 - tcdf(abs(ph_tstat_dur(ph)), dof_dur));
end

[~, ~, ph_adj_nb]  = utils_fdr_bh(ph_pval_nb);
[~, ~, ph_adj_dur] = utils_fdr_bh(ph_pval_dur);

% Format p-value labels for brackets (raw p-values, no FDR)
ph_plabel_nb  = arrayfun(@fmt_p, ph_pval_nb,  'UniformOutput', false);
ph_plabel_dur = arrayfun(@fmt_p, ph_pval_dur, 'UniformOutput', false);

utils_diary(fid_log, '=== Posthoc: Number of states (uncorrected) ===\n');
for ph = 1:nPH
    utils_diary(fid_log, '  %-22s t=%6.2f  p=%.4f  (adj=%.4f)\n', ...
        ph_comps{ph,5}, ph_tstat_nb(ph), ph_pval_nb(ph), ph_adj_nb(ph));
end
utils_diary(fid_log, '=== Posthoc: Duration of states (uncorrected) ===\n');
for ph = 1:nPH
    utils_diary(fid_log, '  %-22s t=%6.2f  p=%.4f  (adj=%.4f)\n', ...
        ph_comps{ph,5}, ph_tstat_dur(ph), ph_pval_dur(ph), ph_adj_dur(ph));
end

% === Plot with significance brackets ===
col_type_sq = [0.2 0.5 0.8; 0.8 0.3 0.3]; % blue=quick, red=hesit
xlabels_sq = {'Ch-Quick','Ch-Hesit','Unch-Quick','Unch-Hesit','Other-Quick','Other-Hesit'};
jitter_sq = 0.45;

% Assemble data matrix [nS x 6]: columns = Ch-Q, Ch-H, Unch-Q, Unch-H, Other-Q, Other-H
data_nb_sq  = [nb_states_quick(:,1)  nb_states_hesit(:,1)  nb_states_quick(:,2) ...
               nb_states_hesit(:,2)  nb_states_quick(:,3)  nb_states_hesit(:,3)];
data_dur_sq = [dur_states_quick(:,1) dur_states_hesit(:,1) dur_states_quick(:,2) ...
               dur_states_hesit(:,2) dur_states_quick(:,3) dur_states_hesit(:,3)];

% State colors: Ch=col(1), Unch=col(2), Other=col(3) - same for Quick and Hesit
scatter_cols_sq       = col([1 1 2 2 3 3], :);
scatter_cols_sq_light = col_light([1 1 2 2 3 3], :);

valid_rows_nb  = ~any(isnan(data_nb_sq),  2);
valid_rows_dur = ~any(isnan(data_dur_sq), 2);
dot_jit_sq  = (rand(nS_sq, 6)-0.5)*jitter_sq;
dot_jit_sq2 = (rand(nS_sq, 6)-0.5)*jitter_sq;

sq_data  = {data_nb_sq, data_dur_sq};
sq_valid = {valid_rows_nb, valid_rows_dur};
sq_jit   = {dot_jit_sq, dot_jit_sq2};
sq_ylab  = {'Average number of states', 'Average state duration (s)'};
sq_plab  = {ph_plabel_nb, ph_plabel_dur};
[fg, cm] = fig_cm(12.1, 8.6);
for k = 1:2
    v  = sq_valid{k};
    ax = axes(fg, 'Position', cm(1.8 + (k-1)*6.0, 2.4, 4.0, 5.9));
    box_dots(ax, sq_data{k}(v,:), 1:6, scatter_cols_sq, scatter_cols_sq_light, {[1 2], [3 4], [5 6]}, sq_jit{k}(v,:));
    style_box_axes(ax, true, sq_ylab{k}, [], [0.4 6.6]);
    set(ax, 'XTick', 1:6, 'XTickLabel', xlabels_sq, 'XTickLabelRotation', 45);
    add_brackets(ax, cell2mat(ph_comps(:, 6:7)), sq_plab{k}, sq_data{k}(v,:), 1:6);
end
save_fig(fg, [report_dir 'Fig_2f_S5d_hesit_states']);


%% -- Chosen/Unchosen ratio: Quick vs Hesit --
% For each session, compute the Ch/Unch ratio of nb_states and dur_states
% separately for quick and hesit trials, then compare the two directly.
% Uses nb_states_quick / nb_states_hesit already computed above.

% Ratio = Chosen / Unchosen (col 1 / col 2)
nb_ratio_quick_qh  = nb_states_quick(:,1)  ./ nb_states_quick(:,2);
nb_ratio_hesit_qh  = nb_states_hesit(:,1)  ./ nb_states_hesit(:,2);
dur_ratio_quick_qh = dur_states_quick(:,1) ./ dur_states_quick(:,2);
dur_ratio_hesit_qh = dur_states_hesit(:,1) ./ dur_states_hesit(:,2);

nS_qh = length(nb_ratio_quick_qh);

% Plot: simple Quick vs Hesit boxplot (same style as earlier)
rng(1);
data_nb_qh  = [nb_ratio_quick_qh  nb_ratio_hesit_qh];
data_dur_qh = [dur_ratio_quick_qh dur_ratio_hesit_qh];
valid_nb_qh  = ~any(isnan(data_nb_qh),  2);
valid_dur_qh = ~any(isnan(data_dur_qh), 2);
col_qh = [col_type_sq(1,:); col_type_sq(2,:)];
dot_jit_qh  = (rand(nS_qh, 2)-0.5)*0.25;
dot_jit_qh2 = (rand(nS_qh, 2)-0.5)*0.25;

qh_data  = {data_nb_qh, data_dur_qh};
qh_valid = {valid_nb_qh, valid_dur_qh};
qh_jit   = {dot_jit_qh, dot_jit_qh2};
qh_ylab  = {{'Ratio of Chosen/Unchosen', 'state number'}, {'Ratio of Chosen/Unchosen', 'state duration'}};
[fg, cm] = fig_cm(9, 6.4);
for k = 1:2
    v  = qh_valid{k};
    ax = axes(fg, 'Position', cm(1.8 + (k-1)*4.3, 1.2, 2.6, 4.8));
    plot(ax, [0.5 2.5], [1 1], 'k--', 'LineWidth', 0.5);
    box_dots(ax, qh_data{k}(v,:), [1 2], col_qh, col_qh*0.4 + 0.6, {[1 2]}, qh_jit{k}(v,:));
    style_box_axes(ax, true, qh_ylab{k}, [], [0.5 2.5]);
    set(ax, 'XTick', [1 2], 'XTickLabel', {'Quick', 'Hesit'});
end
save_fig(fg, [report_dir 'Fig_extra_hesit_ratio']);

% --- LME: ratio ~ type + (1|monkey) + (1|monkey:session) ---
nb_ratio_all_qh  = [nb_ratio_quick_qh;  nb_ratio_hesit_qh];
dur_ratio_all_qh = [dur_ratio_quick_qh; dur_ratio_hesit_qh];
type_all_qh      = [repmat({'quick'}, nS_qh, 1); repmat({'hesit'}, nS_qh, 1)];
monk_all_qh      = [animal_names(:); animal_names(:)];
sess_all_qh      = categorical([(1:nS_qh)'; (1:nS_qh)']);

tbl_ratio_qh = table(nb_ratio_all_qh, dur_ratio_all_qh, type_all_qh, monk_all_qh, sess_all_qh, ...
    'VariableNames', {'nb_ratio','dur_ratio','type','monkey','session'});

tbl_nb_qh = tbl_ratio_qh(~isnan(tbl_ratio_qh.nb_ratio) & ~isinf(tbl_ratio_qh.nb_ratio), :);
if ~isempty(tbl_nb_qh)
    mdl_nb_qh = fitlme(tbl_nb_qh, 'nb_ratio ~ 1 + type + (1|monkey) + (1|monkey:session)');
    utils_diary(fid_log, '=== LME: Ch/Unch nb states ratio ~ type (Quick vs Hesit) ===\n');
    utils_diary(fid_log, '%s', evalc('disp(mdl_nb_qh.Coefficients)'));
    utils_diary(fid_log, '%s', evalc('anova(mdl_nb_qh)'));
end

tbl_dur_qh = tbl_ratio_qh(~isnan(tbl_ratio_qh.dur_ratio) & ~isinf(tbl_ratio_qh.dur_ratio), :);
if ~isempty(tbl_dur_qh)
    mdl_dur_qh = fitlme(tbl_dur_qh, 'dur_ratio ~ 1 + type + (1|monkey) + (1|monkey:session)');
    utils_diary(fid_log, '=== LME: Ch/Unch duration ratio ~ type (Quick vs Hesit) ===\n');
    utils_diary(fid_log, '%s', evalc('disp(mdl_dur_qh.Coefficients)'));
    utils_diary(fid_log, '%s', evalc('anova(mdl_dur_qh)'));
end

%% FIG S5C - Fixation-linked posteriors: look-chosen vs look-unchosen
% (the dwell-time sweep at the end is response to reviewers Fig R6)

dwell_times = 100:50:500;  % dwell times to sweep (ms)
match_look  = true;       % probability-match look-chosen vs look-unchosen trials (like quick/hesit)

load([param.path2proc 'saccadeCounts_all.mat']); % table with saccade counts per trial

for sess = 1 : length(out)
    if ~isempty(out(sess).adjusted_prediction_scores_rand)
        out(sess).saccades_all = saccadeTimes(ismember(string(saccadeTimes.SessionID), string(list(sess).name(1:7))) & ismember(saccadeTimes.Trial,out(sess).keepTr),:);
    end
end

% Monkey and session index (constant across all dwell times)
monks_lk    = cellfun(@(x) x(1), {out(:).session}, 'UniformOutput', false)';
mk_u_lk     = unique(monks_lk);
sess_num_lk = zeros(length(out), 1);
mk_cnt_lk   = zeros(length(mk_u_lk), 1);
for i = 1:length(out)
    m_idx = strcmp(mk_u_lk, monks_lk{i});
    mk_cnt_lk(m_idx) = mk_cnt_lk(m_idx) + 1;
    sess_num_lk(i)   = mk_cnt_lk(m_idx);
end

nDW = length(dwell_times);
ph_tstat_dwell = NaN(nDW, 3);
ph_pval_dwell  = NaN(nDW, 3);

for dw_idx = 1:nDW
    disp(['=== Fixation-linked posteriors: dwell time = ' num2str(dwell_times(dw_idx)) ' ms ===']);
    dwell_ms = dwell_times(dw_idx);

    post_look_ch   = NaN(length(out), 3);
    post_look_unch = NaN(length(out), 3);

    for sess = 1:length(out)
        if isempty(out(sess).adjusted_prediction_scores_rand) || isempty(out(sess).saccades_all)
            continue;
        end
        sacc_tbl = out(sess).saccades_all;
        pred_raw = out(sess).adjusted_prediction_scores_rand;
        t        = out(sess).time;

        bl_mask = t >= -500 & t <= 0;
        pred_bl = pred_raw;
        for c = 1:3
            pred_bl(:,:,c) = pred_bl(:,:,c) - mean(pred_bl(:, bl_mask, c), 2);
        end

        nTr_s        = height(sacc_tbl);
        tr_post_ch   = NaN(nTr_s, 3);
        tr_post_unch = NaN(nTr_s, 3);
        tr_idx_map   = NaN(nTr_s, 1);

        for r = 1:nTr_s
            tr_num = sacc_tbl.Trial(r);
            tr_idx = find(out(sess).keepTr == tr_num, 1);
            if isempty(tr_idx), continue; end
            tr_idx_map(r) = tr_idx;

            cs = out(sess).cond.chosenside_2AFC(tr_idx);
            if cs == 1, ch_dir = "R"; else, ch_dir = "L"; end
            if ch_dir == "R", unch_dir = "L"; else, unch_dir = "R"; end

            et = sacc_tbl.EnterTimes_rel{r};
            ed = sacc_tbl.EnterDir{r};
            if isempty(et) || isempty(ed), continue; end

            ch_wins = []; unch_wins = [];
            for e = 1:numel(et)
                if isnan(et(e)), continue; end
                win_mask = t >= et(e) & t < et(e) + dwell_ms;
                if ~any(win_mask), continue; end
                post_win = squeeze(mean(pred_bl(tr_idx, win_mask, :), 2))';

                if ed(e) == ch_dir
                    ch_wins = [ch_wins; post_win];
                elseif ed(e) == unch_dir
                    unch_wins = [unch_wins; post_win];
                end
            end

            if ~isempty(ch_wins),   tr_post_ch(r,:)   = mean(ch_wins,   1, 'omitnan'); end
            if ~isempty(unch_wins), tr_post_unch(r,:) = mean(unch_wins, 1, 'omitnan'); end
        end

        ch_valid   = find(~any(isnan(tr_post_ch),   2) & ~isnan(tr_idx_map));
        unch_valid = find(~any(isnan(tr_post_unch), 2) & ~isnan(tr_idx_map));
        nb2take_lk = min(length(ch_valid), length(unch_valid));

        if nb2take_lk > 0
            if match_look
                ch_pb   = [out(sess).cond.chosenproba_2AFC(tr_idx_map(ch_valid))   out(sess).cond.unchosenproba_2AFC(tr_idx_map(ch_valid))];
                unch_pb = [out(sess).cond.chosenproba_2AFC(tr_idx_map(unch_valid)) out(sess).cond.unchosenproba_2AFC(tr_idx_map(unch_valid))];

                used_ch = false(length(ch_valid), 1);
                sel_ch  = NaN(nb2take_lk, 1);
                for h = 1:nb2take_lk
                    dists = sqrt(sum((ch_pb - unch_pb(h,:)).^2, 2));
                    dists(used_ch) = Inf;
                    min_dist    = min(dists);
                    min_idx_all = find(abs(dists - min_dist) < 1e-10);
                    [~, ci]     = min(abs(ch_valid(min_idx_all) - unch_valid(h)));
                    pick        = min_idx_all(ci);
                    sel_ch(h)   = ch_valid(pick);
                    used_ch(pick) = true;
                end
                sel_unch = unch_valid(1:nb2take_lk);
            else
                sel_ch   = ch_valid(randperm(length(ch_valid),   nb2take_lk));
                sel_unch = unch_valid(randperm(length(unch_valid), nb2take_lk));
            end

            post_look_ch(sess,:)   = mean(tr_post_ch(sel_ch,   :), 1, 'omitnan');
            post_look_unch(sess,:) = mean(tr_post_unch(sel_unch, :), 1, 'omitnan');
        end
    end

    % LME: Posterior ~ LookType * StateType + (1|Monkey) + (1|Monkey:Session)
    nS_lk     = length(out);
    post_vec  = [post_look_ch(:,1);   post_look_ch(:,2);   post_look_ch(:,3); ...
                 post_look_unch(:,1); post_look_unch(:,2); post_look_unch(:,3)];
    look_vec  = [repmat({'chosen'},  3*nS_lk, 1); repmat({'unchosen'}, 3*nS_lk, 1)];
    state_vec = [repmat({'Chosen'},  nS_lk, 1); repmat({'Unchosen'}, nS_lk, 1); repmat({'Other'}, nS_lk, 1); ...
                 repmat({'Chosen'},  nS_lk, 1); repmat({'Unchosen'}, nS_lk, 1); repmat({'Other'}, nS_lk, 1)];
    monk_vec  = repmat(monks_lk, 6, 1);
    sess_vec  = categorical(repmat(sess_num_lk, 6, 1));

    tbl_lk = table(post_vec, look_vec, state_vec, monk_vec, sess_vec, ...
        'VariableNames', {'Posterior','LookType','StateType','Monkey','Session'});
    tbl_lk = tbl_lk(~isnan(tbl_lk.Posterior), :);

    lme_lk = fitlme(tbl_lk, 'Posterior ~ LookType * StateType + (1|Monkey) + (1|Monkey:Session)');

    % Posthoc: contrast look-chosen vs look-unchosen for each state type
    % MATLAB ref coding: LookType ref='chosen' (c<u), StateType ref='Chosen' (C<O<U)
    cnames_lk = lme_lk.CoefficientNames;
    beta_lk   = fixedEffects(lme_lk);
    covB_lk   = lme_lk.CoefficientCovariance;
    dof_lk    = lme_lk.DFE;

    I_lk  = strcmp(cnames_lk, '(Intercept)');
    Lu    = ~cellfun(@isempty, regexp(cnames_lk, '^LookType_unchosen$'));
    SO    = ~cellfun(@isempty, regexp(cnames_lk, '^StateType_Other$'));
    SU    = ~cellfun(@isempty, regexp(cnames_lk, '^StateType_Unchosen$'));
    LuSO  = ~cellfun(@isempty, regexp(cnames_lk, 'LookType_unchosen:StateType_Other|StateType_Other:LookType_unchosen'));
    LuSU  = ~cellfun(@isempty, regexp(cnames_lk, 'LookType_unchosen:StateType_Unchosen|StateType_Unchosen:LookType_unchosen'));

    mu_ch_Ch     = double(I_lk);
    mu_ch_Unch   = double(I_lk | SU);
    mu_ch_Oth    = double(I_lk | SO);
    mu_unch_Ch   = double(I_lk | Lu);
    mu_unch_Unch = double(I_lk | Lu | SU | LuSU);
    mu_unch_Oth  = double(I_lk | Lu | SO | LuSO);

    ph_comps_lk = {
        mu_ch_Ch   - mu_unch_Ch,   'Ch post: look-Ch vs look-Unch',    1, 2;
        mu_ch_Unch - mu_unch_Unch, 'Unch post: look-Ch vs look-Unch',  3, 4;
        mu_ch_Oth  - mu_unch_Oth,  'Other post: look-Ch vs look-Unch', 5, 6;
    };
    nPH_lk = size(ph_comps_lk, 1);

    ph_tstat_lk = NaN(nPH_lk, 1);
    ph_pval_lk  = NaN(nPH_lk, 1);
    for ph = 1:nPH_lk
        C = ph_comps_lk{ph,1};
        est = C * beta_lk;
        se  = sqrt(C * covB_lk * C');
        ph_tstat_lk(ph) = est / se;
        ph_pval_lk(ph)  = 2 * (1 - tcdf(abs(ph_tstat_lk(ph)), dof_lk));
    end
    [~, ~, ph_adj_lk] = utils_fdr_bh(ph_pval_lk);
    ph_plabel_lk = arrayfun(@fmt_p, ph_pval_lk, 'UniformOutput', false);

    ph_tstat_dwell(dw_idx, :) = ph_tstat_lk';
    ph_pval_dwell(dw_idx,  :) = ph_pval_lk';

    if dwell_ms == 250
        utils_diary(fid_log, '=== LME: Fixation-gated posteriors ~ LookType * StateType (dwell=%d ms) ===\n', dwell_ms);
        utils_diary(fid_log, '%s', evalc('disp(lme_lk.Coefficients)'));
        utils_diary(fid_log, '%s', evalc('anova(lme_lk)'));
        utils_diary(fid_log, '=== Posthoc: Fixation-gated posteriors (uncorrected, dwell=%d ms) ===\n', dwell_ms);
        for ph = 1:nPH_lk
            utils_diary(fid_log, '  %-42s t=%6.2f  p=%.4f  (adj=%.4f)\n', ...
                ph_comps_lk{ph,2}, ph_tstat_lk(ph), ph_pval_lk(ph), ph_adj_lk(ph));
        end

        % Main 6-group boxplot
        data_lk = [post_look_ch(:,1)  post_look_unch(:,1) ...
                   post_look_ch(:,2)  post_look_unch(:,2) ...
                   post_look_ch(:,3)  post_look_unch(:,3)];
        xlabels_lk      = {'Ch-FixCh','Ch-FixUnch','Unch-FixCh','Unch-FixUnch','Other-FixCh','Other-FixUnch'};
        sc_col_lk       = [col(1,:); col(1,:); col(2,:); col(2,:); col(3,:); col(3,:)];
        sc_col_lk_light = [col_light(1,:); col_light(1,:); col_light(2,:); col_light(2,:); col_light(3,:); col_light(3,:)];
        valid_lk        = ~any(isnan(data_lk), 2);

        rng(1);
        dot_jit_lk = (rand(nS_lk, 6)-0.5)*0.45;
        [fg, cm] = fig_cm(6.2, 8.6);
        ax = axes(fg, 'Position', cm(1.8, 2.4, 4.0, 5.9));
        plot(ax, [0.4 6.6], [0 0], 'k--', 'LineWidth', 0.5);
        box_dots(ax, data_lk(valid_lk,:), 1:6, sc_col_lk, sc_col_lk_light, {[1 2], [3 4], [5 6]}, ...
            dot_jit_lk(valid_lk,:));
        style_box_axes(ax, true, 'Normalized posterior probability', [], [0.4 6.6]);
        set(ax, 'XTick', 1:6, 'XTickLabel', xlabels_lk, 'XTickLabelRotation', 45);
        add_brackets(ax, cell2mat(ph_comps_lk(:, 3:4)), ph_plabel_lk, data_lk(valid_lk,:), 1:6);
        save_fig(fg, [report_dir 'Fig_S5c_fixation_posteriors']);
    end

end  % end dwell loop

% FDR-correct per contrast across dwell times
ph_adj_dwell = NaN(nDW, 3);
for c = 1:3
    [~, ~, ph_adj_dwell(:,c)] = utils_fdr_bh(ph_pval_dwell(:,c));
end

contrast_labels_dw = {'Ch post: look-Ch vs look-Unch', 'Unch post: look-Ch vs look-Unch', 'Other post: look-Ch vs look-Unch'};
utils_diary(fid_log, '=== Dwell-time sweep: t-stats and FDR-adjusted p-values ===\n');
for c = 1:3
    utils_diary(fid_log, '  Contrast: %s\n', contrast_labels_dw{c});
    for dw = 1:nDW
        utils_diary(fid_log, '    dwell=%3d ms  t=%6.2f  p=%.4f  (adj=%.4f)\n', ...
            dwell_times(dw), ph_tstat_dwell(dw,c), ph_pval_dwell(dw,c), ph_adj_dwell(dw,c));
    end
end

% Figure: t-stat vs dwell time, 3 lines (one per posthoc contrast)
contrast_labels_dw_short = {'Ch posterior','Unch posterior','Other posterior'};
figure("Position",[200 200 700 450]); set(gcf,'Renderer','painters'); hold on;
for c = 1:3
    plot(dwell_times, ph_tstat_dwell(:,c), '-o', 'Color', col(c,:), ...
        'LineWidth', 1.5, 'MarkerFaceColor', col(c,:), 'MarkerSize', 6);
    sig_dw = ph_adj_dwell(:,c) < 0.05;
    if any(sig_dw)
        scatter(dwell_times(sig_dw), ph_tstat_dwell(sig_dw,c), 60, col(c,:), ...
            'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    end
end
yline(0,'k--','LineWidth',1);
xlabel('Dwell time (ms)','FontSize',14);
ylabel('t-statistic (look-chosen vs look-unchosen)','FontSize',14);
title('Posthoc t-stats across dwell times','FontSize',14);
legend(contrast_labels_dw_short, 'Location','best','FontSize',12);
set(gca,'XTick',dwell_times,'FontSize',12);
saveas(gcf,[report_dir 'Fig_R6_fixation_dwell_sweep.png']);


%% FIG 2H - check r2 when removing neurons

all_sess = {out_all(:).session}';
monkey_name = cellfun(@(x) x(1), all_sess, 'UniformOutput', false);
areas = {out_all(:).area4unit_removed}';
sess_list = unique(all_sess);
xx = 0;
dim4corr = [1 2];


out_table = table();
for s = 1 : length(sess_list)
    disp(s)
    takeme = find(ismember(all_sess, sess_list{s}) & ~cellfun(@isempty, areas));
    takeme_full = find(ismember(all_sess, sess_list{s}) & cellfun(@isempty, areas));

    full = out_all(takeme_full).adjusted_prediction_scores_rand(:,out_all(1).time>=param.time_state(1) & out_all(1).time<=param.time_state(2),dim4corr);
    full_chpb_perf = out_all(takeme_full).accuracy(end);

    for u = 1 : length(takeme)
        minus1 = out_all(takeme(u)).adjusted_prediction_scores_rand(:,out_all(1).time>=param.time_state(1) & out_all(1).time<=param.time_state(2),dim4corr);
        nbrmv = size(out_all(takeme(u)).fr_heldout,1);
        xx = xx + 1;
        [r, p] = corrcoef(full(:), minus1(:));
        minus1_chpb_perf = out_all(takeme(u)).accuracy(end);

        out_table = [out_table; table(r(1,2), full_chpb_perf-minus1_chpb_perf,out_all(takeme(u)).area4unit_removed(1),monkey_name(takeme(u)), all_sess(takeme(u)),nbrmv,...
            'VariableNames', {'R','Ch_perf','area', 'Monkey', 'Session','nb_removed'})];
    end
end

out_table.R = double(out_table.R);
out_table.Rlog = log(out_table.R ./ (1 - (out_table.R-eps)));

% fit with nb_removed as main factor (LOG of nb removed... might need to recheck)
% NOTE: sq_groupvec is defined in sq_groupvec.m (same folder)

% mixed effect model with monkey and session as random
% Fit the model including nb_removed as a covariate (log-transformed)
out_table.nb_removed_log = log(out_table.nb_removed);
lme = fitglme(out_table, 'Rlog ~ 1 + area + nb_removed_log + (1|Monkey) + (1|Monkey:Session)');

% Display the results
utils_diary(fid_log, '%s', evalc('disp(lme)'));
% Conditional R^2 (fixed + random effects, Nakagawa & Schielzeth 2013)
r2_res = sum(residuals(lme).^2);
r2_tot = sum((lme.Variables.Rlog - mean(lme.Variables.Rlog)).^2);
r2_conditional = 1 - (r2_res / r2_tot);
utils_diary(fid_log, 'Conditional R^2 (fixed + random effects): %.4f\n', r2_conditional);
utils_diary(fid_log, '%s', evalc('anova(lme)'));
[pval,wald,thr,pval_adj] = utils_areaposthoc(lme,area2test_name,'y',false); axis square

% Calculate the median number of neurons removed (and its log)
median_nb_removed = median(out_table.nb_removed);
median_nb_removed_log = log(median_nb_removed);

% Plot: Estimated R per area at median nb_removed, and performance drop vs nb_removed
figure('Position',[ 443   296   522   858]);
subplot(3,1,[2 3]); hold on;

% Find which area is the reference (intercept) in the model
ref_area = '';
for ar = 1:length(area2test_name)
    coef_name = ['area_' area2test_name{ar}];
    if ~any(strcmp(lme.Coefficients.Name, coef_name))
        ref_area = area2test_name{ar};
        break
    end
end

for ar = 1:length(area2test_name)
    this_color = colorareas(ar,:)/255;
    if strcmp(area2test_name{ar}, ref_area)
        % Reference area
        estRlog = lme.Coefficients.Estimate(strcmp(lme.Coefficients.Name, '(Intercept)')) + ...
               lme.Coefficients.Estimate(strcmp(lme.Coefficients.Name, 'nb_removed_log')) * median_nb_removed_log;
        % Standard error: combine SEs and covariance
        intercept_idx = find(strcmp(lme.Coefficients.Name, '(Intercept)'));
        nbrem_idx = find(strcmp(lme.Coefficients.Name, 'nb_removed_log'));
        covar = lme.CoefficientCovariance([intercept_idx nbrem_idx],[intercept_idx nbrem_idx]);
        seRlog = sqrt(covar(1,1) + (median_nb_removed_log^2)*covar(2,2) + 2*median_nb_removed_log*covar(1,2));
    else
        coef_name = ['area_' area2test_name{ar}];
        idx = strcmp(lme.Coefficients.Name, coef_name);
        if any(idx)
            estRlog = lme.Coefficients.Estimate(strcmp(lme.Coefficients.Name, '(Intercept)')) + ...
                   lme.Coefficients.Estimate(idx) + ...
                   lme.Coefficients.Estimate(strcmp(lme.Coefficients.Name, 'nb_removed_log')) * median_nb_removed_log;
            coef_idx = find(idx);
            intercept_idx = find(strcmp(lme.Coefficients.Name, '(Intercept)'));
            nbrem_idx = find(strcmp(lme.Coefficients.Name, 'nb_removed_log'));
            covar = lme.CoefficientCovariance([intercept_idx coef_idx nbrem_idx],[intercept_idx coef_idx nbrem_idx]);
            % SE for sum: intercept + area + nb_removed_log*median
            seRlog = sqrt(covar(1,1) + covar(2,2) + (median_nb_removed_log^2)*covar(3,3) + ...
                2*covar(1,2) + 2*median_nb_removed_log*covar(1,3) + 2*median_nb_removed_log*covar(2,3));
        else
            estRlog = NaN;
            seRlog = NaN;
        end
    end
    % Convert logit back to R and propagate error
    estR = exp(estRlog) ./ (1 + exp(estRlog));
    % Delta method for SE: d/dx [exp(x)/(1+exp(x))] = exp(x)/(1+exp(x))^2
    dR_dlog = exp(estRlog) ./ (1 + exp(estRlog)).^2;
    seR = abs(dR_dlog) * seRlog;
    errorbar(ar, estR, 1.96*seR, 'o', 'Color', this_color, ...
        'MarkerFaceColor', this_color, 'MarkerEdgeColor', this_color, 'LineWidth', 2,'MarkerSize', 14);
end
set(gca,'XTick',1:length(area2test_name),'XTickLabel',area2test_name,'FontSize',16);
ylabel(['Estimated R (at ' num2str(median_nb_removed) ' neurons removed)']);
title('Estimated R by Area (from LME)');
ylim([.85 1]); hold off;
xlim([0 length(area2test_name)+1]);

% Subplot: Performance drop (R) vs nb_removed (all data, all areas)
subplot(3,1,1); hold on;
nb_unique = unique(out_table.nb_removed);
mean_R = zeros(size(nb_unique));
ci95_R = zeros(size(nb_unique));
for i = 1:length(nb_unique)
    this_nb = nb_unique(i);
    vals = out_table.R(out_table.nb_removed == this_nb);
    mean_R(i) = mean(vals, 'omitnan');
    ci95_R(i) = 1.96 * std(vals, 'omitnan') / sqrt(sum(~isnan(vals)));
end
errorbar(nb_unique, mean_R, ci95_R, 'o', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'MarkerFaceColor', [0.5 0.5 0.5]);

% --- LME-based prediction of performance drop vs nb_removed (average across areas) ---
% Use the LME to predict R as a function of nb_removed, averaging across areas (fixed effects only)
nb_range = linspace(min(out_table.nb_removed), max(out_table.nb_removed), 100);
nb_range_log = log(nb_range);

% For each area, compute the fixed effect prediction
R_pred = zeros(length(area2test_name), length(nb_range));
for ar = 1:length(area2test_name)
    % Area effect
    if strcmp(area2test_name{ar}, ref_area)
        area_effect = 0;
    else
        coef_name = ['area_' area2test_name{ar}];
        idx = strcmp(lme.Coefficients.Name, coef_name);
        if any(idx)
            area_effect = lme.Coefficients.Estimate(idx);
        else
            area_effect = 0;
        end
    end
    intercept = lme.Coefficients.Estimate(strcmp(lme.Coefficients.Name, '(Intercept)'));
    nbrem_coef = lme.Coefficients.Estimate(strcmp(lme.Coefficients.Name, 'nb_removed_log'));
    % Predict in logit space, then convert to R
    Rlog_pred = intercept + area_effect + nbrem_coef * nb_range_log;
    R_pred(ar,:) = exp(Rlog_pred) ./ (1 + exp(Rlog_pred));
end

% Average across areas
R_pred_mean = mean(R_pred,1);

plot(nb_range, R_pred_mean, 'k-', 'LineWidth', 2);
set(gca,'FontSize',16)
xlabel('Number of Neurons Removed');
ylabel('Correlation Coefficient (R)');
title('Performance Drop (from LME)');
box off; hold off;

saveas(gcf, [report_dir 'Fig_2h_ablation.png']);

%% ═══ Local functions: boxplot figure style ═══════════════════════════════════

function [fg, cm] = fig_cm(W, H)
% Figure of W x H cm (Arial); cm(x, y, w, h) converts cm to normalized units.
fg = figure('Units', 'centimeters', 'Position', [2 2 W H], 'Color', 'w', ...
    'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial');
cm = @(x, y, w, h) [x y w h] ./ [W H W H];
end

function box_dots(ax, X, pos, box_col, dot_col, links, jit)
% Sessions x groups data X at x positions pos: grey lines joining each
% session's values within each group of links, jittered dots, boxplots.
hold(ax, 'on');
for i = 1 : size(X, 1)
    for L = 1 : numel(links)
        g = links{L};
        plot(ax, pos(g) + jit(i,g), X(i,g), '-', 'Color', [0.78 0.78 0.78], 'LineWidth', 0.3);
    end
end
for g = 1 : size(X, 2)
    plot(ax, pos(g) + jit(:,g), X(:,g), 'o', 'MarkerSize', 1.8, 'MarkerFaceColor', dot_col(g,:), ...
        'MarkerEdgeColor', 'none');
end
p0 = get(ax, 'Position');
h  = boxplot(ax, X, 'Colors', box_col, 'Symbol', '', 'Widths', 0.6, 'Positions', pos);
set(h, 'LineWidth', 1);
set(ax, 'PositionConstraint', 'innerposition', 'Position', p0);   % same plot-box size in every panel
end

function style_box_axes(ax, show_y, ylab, yl, xl)
% Shared axes style; show_y = false hides the y tick labels (second monkey).
set(ax, 'FontSize', 10, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 0.5, ...
    'XColor', 'k', 'YColor', 'k', 'TickLength', [0.015 0.015]);
if ~isempty(yl), ylim(ax, yl); end
xlim(ax, xl);
if show_y
    ylabel(ax, ylab, 'FontSize', 11);
else
    set(ax, 'YTickLabel', []);
end
end

function add_brackets(ax, pairs, labels, X, pos)
% Significance brackets just above the highest point of the compared groups
% (pairs: nComparisons x 2 group positions); overlapping brackets are raised
% above each other. The y-limits are then fitted to the data and brackets.
lo  = min(X(:));  hi = max(X(:));  rg = hi - lo;
bky = NaN(size(pairs, 1), 1);
for k = 1 : size(pairs, 1)
    g      = pos >= min(pairs(k,:)) & pos <= max(pairs(k,:));
    bky(k) = max(X(:, g), [], 'all') + 0.05*rg;
    for j = 1 : k - 1        % raise above earlier brackets it overlaps
        if min(pairs(k,:)) <= max(pairs(j,:)) && min(pairs(j,:)) <= max(pairs(k,:))
            bky(k) = max(bky(k), bky(j) + 0.10*rg);
        end
    end
    plot(ax, pairs(k, [1 1 2 2]), bky(k) + [-0.025 0 0 -0.025]*rg, 'k-', 'LineWidth', 0.5);
    text(ax, mean(pairs(k,:)), bky(k) + 0.01*rg, labels{k}, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 9);
end
ylim(ax, [lo - 0.04*rg, max(bky) + 0.09*rg]);
end

function str = fmt_p(p)
% p-value label: 'p=0.012' when p >= 0.001, otherwise scientific notation ('p=3.2e-5').
if p >= 0.001
    str = sprintf('p=%.3f', p);
else
    str = regexprep(sprintf('p=%.1e', p), 'e([-+])0*(\d)', 'e$1$2');
end
end

function fig_text(fg, pos, str, fs)
% Text centred in a box given in normalized figure units.
ax = axes(fg, 'Position', pos, 'XLim', [0 1], 'YLim', [0 1]);  axis(ax, 'off');
text(ax, 0.5, 0.5, str, 'FontSize', fs, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

function state_legend(fg, pos, labels, cols)
% Boxed legend of colored italic labels.
ax = axes(fg, 'Position', pos, 'XLim', [0 1], 'YLim', [0 1]);  axis(ax, 'off');  hold(ax, 'on');
rectangle(ax, 'Position', [0 0 1 1], 'EdgeColor', 'k', 'LineWidth', 0.5);
for i = 1 : numel(labels)
    text(ax, 0.08, 1 - (i - 0.5)/numel(labels), labels{i}, 'Color', cols(i,:), 'FontSize', 11, ...
        'FontAngle', 'italic', 'VerticalAlignment', 'middle');
end
end

function save_fig(fg, fname)
% PNG (300 dpi) and vector PDF (for CorelDRAW / Illustrator).
% exportgraphics(fg, [fname '.png'], 'Resolution', 300);
exportgraphics(fg, [fname '.pdf'], 'ContentType', 'vector');
end
