%% main_005_thr_sensitivity.m
%
% Threshold-sensitivity robustness analysis.
%
% Varies thr_state  in {0.25, 0.30, 0.35, 0.40}  (amplitude threshold)
%        thr_state_dur in {4, 5, 6}              (minimum consecutive bins)
%
% For each of the 12 combos:
%   1. Re-applies state detection to the saved adjusted_prediction_scores_rand
%      (classifiers are NOT re-trained; only the thresholding step is re-run)
%   2. Computes Fig-2c/d equivalent: nb/dur states per session + LME
%   3. Computes Fig-2e equivalent:   Ch/Unch ratio vs proba diff + LME
%   4. Computes Fig-2f equivalent:   Ch/Unch ratio quick vs hesit + LME
%   5. Builds FR cache for each combo
%   6. Runs CCGP 2x2 decoding (Repetition = 50) for chosenflavor + chosenside
%   7. Runs CCGP LME + produces Fig-3c/3d-equivalent heatmaps + contrast bars
%   8. Saves per-combo results to processed/ and report_005/<thr_tag>/
%
% After the loop, summary grid figures compare all 12 combos side by side.
%
% No existing files are modified.
% Source files required:
%   processed/states_2afc_final.mat    (main_002_states.m)
%   processed/saccadeCounts_reduced.mat
%   processed/behav_pref.mat
%
% Dependencies: utils_findenough, utils_createpseudopop, utils_fdr_bh,
%               utils_diary
%
% The per-combo summaries used by the summary figures are saved to
% processed/states_2afc_thr_summary.mat. When this file exists, steps 1-8 are
% skipped and only the summary figures are rebuilt (set overwrite_summary =
% true to rerun all combos; the CCGP decoding itself is cached per combo in
% processed/states_2afc_ccgp_<thr_tag>.mat).

clear
overwrite_summary = false;   % rerun all combos instead of loading the saved summary

%% ============================================================
%% Paths and shared setup
%% ============================================================

f = mfilename('fullpath');
if isempty(f)
    currentPath = pwd;
else
    currentPath = fileparts(fileparts(f));  % go up from scripts/
end
addpath(genpath([currentPath '\scripts\']));

path2proc = [currentPath '\processed\'];
summary_file = [path2proc 'states_2afc_thr_summary.mat'];

%% ============================================================
%% Threshold grid
%% ============================================================

thr_vals = [0.25, 0.30, 0.35, 0.40];   % amplitude threshold
dur_vals = [4, 5, 6];                   % minimum consecutive bins
nThr = length(thr_vals);
nDur = length(dur_vals);

%% ============================================================
%% CCGP parameters  (mirrors main_003_crossdecoding.m)
%% ============================================================

param_ccgp.pseudopop    = round(logspace(log10(25), log10(500), 25));
param_ccgp.param2decode = {'chosenflavor_2AFC', 'chosenside_2AFC'};
param_ccgp.minTr        = [20 20];
param_ccgp.Repetition   = 100;         % reduced from 100 for speed

N_eval_global  = 200;
fdr_alpha      = 0.05;
area2test_name = {'MFC','PMC','dlPFC','IFG','vlPFC','AI','OFC','STR','AMG'};
nAreas         = length(area2test_name);
colorareas     = [230 171 2 ; 152 78 163 ; 237 87 90 ; 252 141 98 ; 141 160 203 ; ...
                  166 216 84 ; 102 194 165 ; 180 180 180 ; 231 138 195];
state_labels   = {'Chosen','Unchosen'};
mks            = {'M','X'};

col       = [100 200 160 ; 240 90 90 ; 0 0 0] / 255;  % Chosen / Unchosen / Other
col_light = col + (1-col)*0.6;
mk_col    = [51 128 230 ; 140 89 38] / 255;            % Monkey M / X

st_dur_list = [0 100:10:1000] / 1000;   % bin index -> seconds
diff_cd     = [20 40 60];               % proba differences for Fig 2e

if ~exist(summary_file, 'file') || overwrite_summary

    %% ============================================================
    %% Load data once
    %% ============================================================

    fprintf('Loading states_2afc_final.mat ...\n');
    load([path2proc 'states_2afc_final.mat'], 'out_all', 'list', 'param');

    fprintf('Loading saccadeCounts_reduced.mat ...\n');
    load([path2proc 'saccadeCounts_reduced.mat']);   % -> saccadeTable

    fprintf('Loading behav_pref.mat ...\n');
    load([path2proc 'behav_pref.mat']);              % -> preference

    % Keep only full-population entries (no area held out)
    out   = out_all(cellfun(@isempty, {out_all(:).area4unit_removed}));
    nSess = length(out);

    % Session metadata (preserved across combos)
    sess_names = {out(:).session}';
    monks      = cellfun(@(x) x(1), sess_names, 'UniformOutput', false);
    mk_names   = {'M','X'};
    sess_num   = [1:sum(strcmp(monks,'M')), 1:sum(strcmp(monks,'X'))]';

    % Time window for state detection (same as param.time_state used in main_002)
    time_state_start = param.time_state(1);   % 0 ms
    time_state_end   = param.time_state(2);   % 800 ms

    % Pre-assign saccade counts to each session (independent of threshold)
    fprintf('Assigning saccade data to sessions ...\n');
    for i = 1:nSess
        out(i).saccades = saccadeTable.NumSaccades( ...
            ismember(string(saccadeTable.SessionID), string(out(i).session)) & ...
            ismember(saccadeTable.Trial, out(i).keepTr));
    end

    %% ============================================================
    %% Storage for cross-combo summary
    %% ============================================================

    summary_nb         = cell(nThr, nDur);   % per-combo [nSess x 3] nb_states
    summary_dur        = cell(nThr, nDur);   % per-combo [nSess x 3] dur_states
    summary_ccgp       = cell(nThr, nDur);   % per-combo all_contrasts cell
    summary_valid_frac = NaN(nThr, nDur);    % fraction of trials with >=1 Ch AND >=1 Unch state
    summary_pval_nb_state  = NaN(nThr, nDur); % p-value for LME on nb_states 
    summary_pval_dur_state = NaN(nThr, nDur); % p-value for LME on dur_states

    summary_dir = [currentPath '\report_005\'];
    if ~exist(summary_dir,'dir'), mkdir(summary_dir); end

    %% ============================================================
    %% Posterior score distribution (justification for thr=0.30)
    %% ============================================================

    fprintf('\nCharacterising posterior score distribution ...\n');

    all_scores_min = [];
    all_scores_mid = [];
    all_scores_max = [];

    for i = 1:nSess
        if isempty(out(i).adjusted_prediction_scores_rand), continue; end
        time_mask_i = out(i).time >= time_state_start & out(i).time <= time_state_end;
        sc = single(out(i).adjusted_prediction_scores_rand(:, time_mask_i, :));
        sc_flat   = reshape(sc, [], 3);          % [nTr*nBins x 3]
        sc_sorted = sort(sc_flat, 2);            % sort ascending per bin
        all_scores_min = [all_scores_min; sc_sorted(:,1)];
        all_scores_mid = [all_scores_mid; sc_sorted(:,2)];
        all_scores_max = [all_scores_max; sc_sorted(:,3)];
    end
    all_scores_min = all_scores_min(all_scores_min > 0);
    all_scores_mid = all_scores_mid(all_scores_mid > 0);
    all_scores_max = all_scores_max(all_scores_max > 0);

    thr_lines  = [0.25 0.30 0.35 0.40];
    thr_clrs   = [0.55 0.55 0.55; 0.20 0.20 0.80; 0.20 0.65 0.20; 0.80 0.20 0.20];
    chance_lvl = 1/5;

    %% ============================================================
    %% Posterior distribution with label-shuffle overlay
    %% ============================================================

    edges = 0:0.01:1;  centers = edges(1:end-1) + 0.005;
    [n_min,~] = histcounts(all_scores_min, edges, 'Normalization','probability');
    [n_mid,~] = histcounts(all_scores_mid, edges, 'Normalization','probability');
    [n_max,~] = histcounts(all_scores_max, edges, 'Normalization','probability');

    fig_ranked = figure('Position',[30 30 1240 480],'Color','w');
    hold on; box off;
    plot(centers, n_min, 'Color',[0.65 0.65 0.65],'LineWidth',1.8,'DisplayName','Min category');
    plot(centers, n_mid, 'Color',[0.20 0.50 0.80],'LineWidth',1.8,'DisplayName','Mid category');
    plot(centers, n_max, 'Color',[0.80 0.20 0.20],'LineWidth',1.8,'DisplayName','Max category');
    xline(chance_lvl,'--','Color',[0.4 0.4 0.4],'LineWidth',1.8,...
        'Label',sprintf('Chance (%.2f)',chance_lvl),'LabelOrientation','horizontal','FontSize',8);
    for k = 1:length(thr_lines)
        xline(thr_lines(k),'Color',thr_clrs(k,:),'LineWidth',2.2,...
            'Label',sprintf('%.2f',thr_lines(k)),'LabelOrientation','horizontal','FontSize',8);
    end
    xlabel('Posterior score','FontSize',11); ylabel('Proportion of bins','FontSize',11);
    title({'Posterior by rank (min / mid / max category)',...
        sprintf('N = %.1fM bins', numel(all_scores_max)/1e6)},'FontSize',11);
    xlim([0 1]); set(gca,'TickLength',[0 0]);
    legend('Location','northeast','FontSize',9,'Box','off');
    exportgraphics(fig_ranked, [summary_dir 'Fig_R1a_posterior_distribution.pdf'], 'ContentType', 'vector');
    fprintf('  Saved: Fig_R1a_posterior_distribution.pdf\n');

    clear all_scores_min all_scores_mid all_scores_max

    %% ============================================================
    %%  MAIN LOOP
    %% ============================================================

    for ti = 1:nThr
    for di = 1:nDur

        thr_now = thr_vals(ti);
        dur_now = dur_vals(di);

        thr_tag = sprintf('thr%s_dur%d', strrep(sprintf('%.2f',thr_now),'.','p'), dur_now);
        % e.g.  thr0p30_dur5

        report_dir = [currentPath '\report_005\' thr_tag '\'];
        if ~exist(report_dir,'dir'), mkdir(report_dir); end

        fr_cache = [path2proc 'states_2afc_fr_'   thr_tag '.mat'];
        savefile = [path2proc 'states_2afc_ccgp_' thr_tag '.mat'];

        fprintf('\n================================================================\n');
        fprintf('  thr = %.2f   dur = %d   [%d / %d]\n', ...
                thr_now, dur_now, (ti-1)*nDur+di, nThr*nDur);
        fprintf('================================================================\n');

        fid_log = fopen([report_dir 'output_005.txt'], 'w');
        log_cleanup = onCleanup(@() fclose(fid_log));
        utils_diary(fid_log, '=== Threshold sensitivity: thr=%.2f  dur=%d ===\n', thr_now, dur_now);
        utils_diary(fid_log, 'Generated: %s\n', datestr(now));

        %% ------------------------------------------------------------
        %% STEP 1 — Re-apply state detection
        %%          (same state detection as utils_decoding_crosstask_rmvarea.m)
        %% ------------------------------------------------------------
        fprintf('  [1/7] Re-applying state detection (thr=%.2f, dur=%d) ...\n', thr_now, dur_now);

        % Use separate arrays to avoid copying large struct fields
        states_cell    = cell(nSess, 1);
        nb_states_cell = cell(nSess, 1);
        dur_states_cell= cell(nSess, 1);
        t_states_cell  = cell(nSess, 1);

        for i = 1:nSess
            if isempty(out(i).adjusted_prediction_scores_rand)
                nb_states_cell{i}  = NaN(0, 3);
                dur_states_cell{i} = NaN(0, 3);
                t_states_cell{i}   = NaN(0, 3);
                states_cell{i}     = single([]);
                continue
            end

            time_i    = out(i).time;
            time_mask = time_i >= time_state_start & time_i <= time_state_end;
            subtime   = time_i(time_mask);
            nTr_i     = size(out(i).adjusted_prediction_scores_rand, 1);
            nBins_i   = sum(time_mask);

            new_nb     = NaN(nTr_i, 3);
            new_dur    = NaN(nTr_i, 3);
            new_t      = NaN(nTr_i, 3);
            new_states = single(NaN(nTr_i, nBins_i));

            for tr = 1:nTr_i
                pred = squeeze(out(i).adjusted_prediction_scores_rand(tr, time_mask, :));
                % [nBins x 3]

                pred(pred < thr_now) = 0;
                [~, loc_max] = max(pred');  % loc_max: [1 x nBins]

                % Keep only the maximum-scoring category at each bin
                pred_max = zeros(size(pred));
                for c = 1:3
                    pred_max(loc_max==c, c) = pred(loc_max==c, c);
                end

                for c = 1:3
                    [idx, idxs] = utils_findenough(pred_max(:,c)', 0, dur_now, '>');
                    new_nb(tr, c)       = length(idx);
                    new_dur(tr, c)      = length(idxs);
                    new_states(tr, idxs)= c;
                    if ~isempty(idx)
                        new_t(tr, c) = subtime(idx(1));
                    end
                end
            end

            nb_states_cell{i}  = new_nb;
            dur_states_cell{i} = new_dur;
            t_states_cell{i}   = new_t;
            states_cell{i}     = new_states;
        end
        fprintf('     State detection done.\n');

        %% ------------------------------------------------------------
        %% STEP 2 — Fig 2c-d: nb/dur states per session + LME
        %% ------------------------------------------------------------
        fprintf('  [2/7] Fig 2c-d (nb/dur states) ...\n');

        nb_states  = NaN(nSess, 3);
        dur_states = NaN(nSess, 3);
        t_states   = NaN(nSess, 3);

        for i = 1:nSess
            if ~isempty(nb_states_cell{i}) && size(nb_states_cell{i},1) > 0
                nb_states(i,:)  = sum(nb_states_cell{i}, 1) / size(nb_states_cell{i}, 1);
                dur_temp = st_dur_list(dur_states_cell{i} + 1);
                dur_temp(nb_states_cell{i} == 0) = NaN;
                dur_states(i,:) = mean(dur_temp, 1, 'omitnan');
                t_states(i,:)   = nanmean(t_states_cell{i}) / 1000;
            end
        end

        summary_nb{ti,di}  = nb_states;
        summary_dur{ti,di} = dur_states;

        % Count trials satisfying the double-state criterion (>=1 Ch AND >=1 Unch)
        n_valid_ds = 0;  n_tot_ds = 0;
        for i_vs = 1:nSess
            if isempty(nb_states_cell{i_vs}), continue; end
            nb_i       = nb_states_cell{i_vs};
            n_valid_ds = n_valid_ds + sum(nb_i(:,1) >= 1 & nb_i(:,2) >= 1);
            n_tot_ds   = n_tot_ds   + size(nb_i, 1);
        end
        summary_valid_frac(ti, di) = n_valid_ds / max(n_tot_ds, 1);
        fprintf('     Valid (Ch+Unch): %d/%d = %.1f%%\n', ...
            n_valid_ds, n_tot_ds, 100*summary_valid_frac(ti,di));

        % LME
        temp_mk    = repmat(monks', 1, 3);
        temp_state = repmat({'Chosen';'Unchosen';'Other'}', nSess, 1);
        temp_sess  = repmat(sess_num, 1, 3);
        data_tbl = table(nb_states(:), dur_states(:), t_states(:), ...
            temp_mk(:), temp_state(:), categorical(temp_sess(:)), ...
            'VariableNames',{'nb_states','dur_states','t_states','monkey','state','session'});

        mdl_nb  = fitlme(data_tbl, 'nb_states  ~ 1 + state + (1|monkey) + (1|monkey:session)');
        mdl_dur = fitlme(data_tbl, 'dur_states ~ 1 + state + (1|monkey) + (1|monkey:session)');
        mdl_t   = fitlme(data_tbl, 't_states   ~ 1 + state + (1|monkey) + (1|monkey:session)');

        %- keep values for summary table
        anv_nb = anova(mdl_nb);
        anv_dur = anova(mdl_dur);
        summary_pval_nb_state(ti,di)  = anv_nb.pValue(2);
        summary_pval_dur_state(ti,di) = anv_dur.pValue(2);

        utils_diary(fid_log, '\n=== Fig 2c-d: nb_states LME ===\n');
        utils_diary(fid_log, '%s', evalc('disp(mdl_nb.Coefficients)'));
        utils_diary(fid_log, '%s', evalc('anova(mdl_nb)'));
        utils_diary(fid_log, '\n=== Fig 2c-d: dur_states LME ===\n');
        utils_diary(fid_log, '%s', evalc('disp(mdl_dur.Coefficients)'));
        utils_diary(fid_log, '%s', evalc('anova(mdl_dur)'));
        utils_diary(fid_log, '\n=== Fig 2c-d: t_states LME ===\n');
        utils_diary(fid_log, '%s', evalc('disp(mdl_t.Coefficients)'));
        utils_diary(fid_log, '%s', evalc('anova(mdl_t)'));

        % Figure 2c-d
        jitter = 0.35; rng(1);
        fig2cd = figure('Position',[50 50 950 720],'Color','w');
        for m = 1:length(mk_names)
            mk_idx = strcmp(monks, mk_names{m});
            nb_m   = nb_states(mk_idx,:);
            dur_m  = dur_states(mk_idx,:);
            nR = sum(mk_idx);

            subplot(2,3,(m-1)*3+1); hold on;
            dj = (rand(nR,3)-0.5)*jitter;
            for ii=1:nR
                plot((1:3)+dj(ii,:), nb_m(ii,:), '-','Color',[0.7 0.7 0.7],'LineWidth',0.8);
            end
            for c=1:3
                scatter(c+dj(:,c), nb_m(:,c), 15, col_light(c,:), 'filled','MarkerEdgeColor','none');
            end
            boxplot(nb_m,'Colors',col,'Symbol','','Widths',0.5,'Positions',1:3,...
                'Labels',{'Chosen','Unchosen','Other'});
            ylabel('Nb states/trial','FontSize',11);
            title([mk_names{m} ': Nb states'],'FontSize',12);
            set(gca,'XTick',1:3,'XTickLabel',{'Chosen','Unchosen','Other'}); ylim([0 2.2]);

            subplot(2,3,(m-1)*3+2); hold on;
            dj = (rand(nR,3)-0.5)*jitter;
            for ii=1:nR
                plot((1:3)+dj(ii,:), dur_m(ii,:), '-','Color',[0.7 0.7 0.7],'LineWidth',0.8);
            end
            for c=1:3
                scatter(c+dj(:,c), dur_m(:,c), 15, col_light(c,:), 'filled','MarkerEdgeColor','none');
            end
            boxplot(dur_m,'Colors',col,'Symbol','','Widths',0.5,'Positions',1:3,...
                'Labels',{'Chosen','Unchosen','Other'});
            ylabel('Duration (s)','FontSize',11);
            title([mk_names{m} ': Duration'],'FontSize',12);
            set(gca,'XTick',1:3,'XTickLabel',{'Chosen','Unchosen','Other'}); ylim([0.05 0.7]);

            subplot(2,3,(m-1)*3+3); hold on;
            t_m = t_states(mk_idx,:);
            dj = (rand(nR,3)-0.5)*jitter;
            for ii=1:nR
                plot((1:3)+dj(ii,:), t_m(ii,:), '-','Color',[0.7 0.7 0.7],'LineWidth',0.8);
            end
            for c=1:3
                scatter(c+dj(:,c), t_m(:,c), 15, col_light(c,:), 'filled','MarkerEdgeColor','none');
            end
            boxplot(t_m,'Colors',col,'Symbol','','Widths',0.5,'Positions',1:3,...
                'Labels',{'Chosen','Unchosen','Other'});
            ylabel('Start time (s)','FontSize',11);
            title([mk_names{m} ': Start time'],'FontSize',12);
            set(gca,'XTick',1:3,'XTickLabel',{'Chosen','Unchosen','Other'}); ylim([0 0.5]);
        end
        sgtitle(strrep(thr_tag,'_',' '),'FontSize',14,'FontWeight','bold');
        exportgraphics(fig2cd, [report_dir 'Fig_2cd_S5a_states.pdf'], 'ContentType', 'vector');
        close(fig2cd);

        %% ------------------------------------------------------------
        %% STEP 3 — Fig 2e: Ch/Unch ratio vs proba diff + LME
        %% ------------------------------------------------------------
        fprintf('  [3/7] Fig 2e (ratio vs proba diff) ...\n');

        nb_ch_unch  = NaN(nSess, length(diff_cd));
        dur_ch_unch = NaN(nSess, length(diff_cd));

        for i = 1:nSess
            if isempty(nb_states_cell{i}) || size(nb_states_cell{i},1)==0, continue; end
            cds = abs(out(i).cond.chosenproba_2AFC - out(i).cond.unchosenproba_2AFC);
            for cd = 1:length(diff_cd)
                take_idx = cds == diff_cd(cd);
                if sum(take_idx) == 0, continue; end
                nb_cd  = sum(nb_states_cell{i}(take_idx,:), 1) / sum(take_idx);
                dur_tmp= st_dur_list(dur_states_cell{i}(take_idx,:) + 1);
                dur_tmp(nb_states_cell{i}(take_idx,:) == 0) = NaN;
                dur_cd = mean(dur_tmp, 1, 'omitnan');
                if nb_cd(2) > 0
                    nb_ch_unch(i,cd)  = nb_cd(1)  / nb_cd(2);
                end
                if ~isnan(dur_cd(2)) && dur_cd(2) > 0
                    dur_ch_unch(i,cd) = dur_cd(1) / dur_cd(2);
                end
            end
        end

        temp_mk2   = repmat(monks', 1, length(diff_cd));
        temp_diff  = repmat(diff_cd, nSess, 1);
        temp_sess2 = repmat(sess_num, 1, length(diff_cd));
        ratio_tbl = table(nb_ch_unch(:), dur_ch_unch(:), temp_mk2(:), ...
            categorical(temp_diff(:)), categorical(temp_sess2(:)), ...
            'VariableNames',{'nb_ratio','dur_ratio','monkey','diff','session'});

        tbl_nb_e  = ratio_tbl(~isnan(ratio_tbl.nb_ratio),  :);
        tbl_dur_e = ratio_tbl(~isnan(ratio_tbl.dur_ratio), :);

        if height(tbl_nb_e) > 4
            mdl_nb_e = fitlme(tbl_nb_e, ...
                'nb_ratio ~ 1 + diff + (1|monkey) + (1|monkey:session)');
            utils_diary(fid_log, '\n=== Fig 2e: nb_ratio vs proba diff LME ===\n');
            utils_diary(fid_log, '%s', evalc('disp(mdl_nb_e.Coefficients)'));
            utils_diary(fid_log, '%s', evalc('anova(mdl_nb_e)'));
        end
        if height(tbl_dur_e) > 4
            mdl_dur_e = fitlme(tbl_dur_e, ...
                'dur_ratio ~ 1 + diff + (1|monkey) + (1|monkey:session)');
            utils_diary(fid_log, '\n=== Fig 2e: dur_ratio vs proba diff LME ===\n');
            utils_diary(fid_log, '%s', evalc('disp(mdl_dur_e.Coefficients)'));
            utils_diary(fid_log, '%s', evalc('anova(mdl_dur_e)'));
        end

        fig2e = figure('Position',[50 50 960 560],'Color','w');
        for m = 1:length(mk_names)
            mk_idx = strcmp(monks, mk_names{m});
            nb_m  = nb_ch_unch(mk_idx,:);
            dur_m = dur_ch_unch(mk_idx,:);

            subplot(2,2,(m-1)*2+1); hold on;
            valid = ~all(isnan(nb_m),2);
            if any(valid)
                nRv = sum(valid); dj = (rand(nRv,3)-0.5)*0.3;
                nb_m_v = nb_m(valid,:);
                for ii=1:nRv
                    plot((1:3)+dj(ii,:), nb_m_v(ii,:), '-','Color',[0.7 0.7 0.7],'LineWidth',0.8);
                end
                for c=1:3
                    scatter(c+dj(:,c), nb_m_v(:,c), 18, ...
                        repmat(mk_col(m,:)*0.4+0.6,nRv,1),'filled','MarkerEdgeColor','none');
                end
                boxplot(nb_m_v,'Colors',mk_col(m,:),'Symbol','','Widths',0.5,...
                    'Positions',1:3,'Labels',arrayfun(@num2str,diff_cd,'UniformOutput',false));
            end
            ylabel('Ratio Ch/Unch states','FontSize',11);
            xlabel('|Proba diff| (%)','FontSize',10);
            title([mk_names{m} ': Nb ratio vs proba diff'],'FontSize',11); xlim([0.5 3.5]);

            subplot(2,2,(m-1)*2+2); hold on;
            valid = ~all(isnan(dur_m),2);
            if any(valid)
                nRv = sum(valid); dj = (rand(nRv,3)-0.5)*0.3;
                dur_m_v = dur_m(valid,:);
                for ii=1:nRv
                    plot((1:3)+dj(ii,:), dur_m_v(ii,:), '-','Color',[0.7 0.7 0.7],'LineWidth',0.8);
                end
                for c=1:3
                    scatter(c+dj(:,c), dur_m_v(:,c), 18, ...
                        repmat(mk_col(m,:)*0.4+0.6,nRv,1),'filled','MarkerEdgeColor','none');
                end
                boxplot(dur_m_v,'Colors',mk_col(m,:),'Symbol','','Widths',0.5,...
                    'Positions',1:3,'Labels',arrayfun(@num2str,diff_cd,'UniformOutput',false));
            end
            ylabel('Ratio Ch/Unch dur (s)','FontSize',11);
            xlabel('|Proba diff| (%)','FontSize',10);
            title([mk_names{m} ': Dur ratio vs proba diff'],'FontSize',11); xlim([0.5 3.5]);
        end
        sgtitle(strrep(thr_tag,'_',' '),'FontSize',13,'FontWeight','bold');
        exportgraphics(fig2e, [report_dir 'Fig_2e_S5b_ratio_vs_probadiff.pdf'], 'ContentType', 'vector');
        close(fig2e);

        %% ------------------------------------------------------------
        %% STEP 4 — Fig 2f: Quick vs Hesit Ch/Unch ratio + LME
        %%          Uses probability-matched trial pairs (same as main_002)
        %% ------------------------------------------------------------
        fprintf('  [4/7] Fig 2f (quick vs hesit) ...\n');

        nb_q = []; nb_h = []; dur_q = []; dur_h = []; animal_f = {};
        x_f = 0;

        for i = 1:nSess
            if isempty(out(i).saccades) || isempty(nb_states_cell{i}) || ...
                    size(nb_states_cell{i},1)==0
                continue
            end

            quick_idx = find(out(i).saccades == 1);
            hesit_idx = find(out(i).saccades >  1);
            nb2take   = min([length(quick_idx) length(hesit_idx)]);
            if nb2take == 0, continue; end

            x_f = x_f + 1;
            animal_f{x_f} = out(i).session(1);

            % Probability-matching (same heuristic as main_002)
            qpb = [out(i).cond.chosenproba_2AFC(quick_idx), ...
                   out(i).cond.unchosenproba_2AFC(quick_idx)];
            hpb = [out(i).cond.chosenproba_2AFC(hesit_idx), ...
                   out(i).cond.unchosenproba_2AFC(hesit_idx)];

            used_q = false(length(quick_idx), 1);
            sel_q  = NaN(nb2take, 1);
            for h = 1:nb2take
                dists = sqrt(sum((qpb - hpb(h,:)).^2, 2));
                dists(used_q) = Inf;
                [~, pick] = min(dists);
                sel_q(h)  = quick_idx(pick);
                used_q(pick) = true;
            end
            sel_h = hesit_idx(1:nb2take);

            % Mean nb/dur over matched trials
            nb_q(x_f,:) = mean(nb_states_cell{i}(sel_q,:), 1);
            nb_h(x_f,:) = mean(nb_states_cell{i}(sel_h,:), 1);

            dq = st_dur_list(dur_states_cell{i}(sel_q,:) + 1);
            dh = st_dur_list(dur_states_cell{i}(sel_h,:) + 1);
            dq(nb_states_cell{i}(sel_q,:) == 0) = NaN;
            dh(nb_states_cell{i}(sel_h,:) == 0) = NaN;
            dur_q(x_f,:) = mean(dq, 1, 'omitnan');
            dur_h(x_f,:) = mean(dh, 1, 'omitnan');
        end

        if x_f > 0
            nS_f = x_f;

            % Ch/Unch ratio
            nb_ratio_q  = nb_q(:,1)  ./ nb_q(:,2);
            nb_ratio_h  = nb_h(:,1)  ./ nb_h(:,2);
            dur_ratio_q = dur_q(:,1) ./ dur_q(:,2);
            dur_ratio_h = dur_h(:,1) ./ dur_h(:,2);

            % LME: ratio ~ type + (1|monkey) + (1|monkey:session)
            nb_r_all  = [nb_ratio_q;  nb_ratio_h];
            dur_r_all = [dur_ratio_q; dur_ratio_h];
            type_all  = [repmat({'quick'},nS_f,1); repmat({'hesit'},nS_f,1)];
            monk_all  = [animal_f(:); animal_f(:)];
            sess_all  = categorical([(1:nS_f)'; (1:nS_f)']);

            qh_tbl = table(nb_r_all, dur_r_all, type_all, monk_all, sess_all, ...
                'VariableNames',{'nb_ratio','dur_ratio','type','monkey','session'});

            tbl_nb_qh  = qh_tbl(~isnan(qh_tbl.nb_ratio)  & ~isinf(qh_tbl.nb_ratio),  :);
            tbl_dur_qh = qh_tbl(~isnan(qh_tbl.dur_ratio) & ~isinf(qh_tbl.dur_ratio), :);

            if height(tbl_nb_qh) > 4
                lme_nb_qh = fitlme(tbl_nb_qh, ...
                    'nb_ratio ~ 1 + type + (1|monkey) + (1|monkey:session)');
                utils_diary(fid_log, '\n=== Fig 2f: nb_ratio Quick vs Hesit LME ===\n');
                utils_diary(fid_log, '%s', evalc('disp(lme_nb_qh.Coefficients)'));
                utils_diary(fid_log, '%s', evalc('anova(lme_nb_qh)'));
            end
            if height(tbl_dur_qh) > 4
                lme_dur_qh = fitlme(tbl_dur_qh, ...
                    'dur_ratio ~ 1 + type + (1|monkey) + (1|monkey:session)');
                utils_diary(fid_log, '\n=== Fig 2f: dur_ratio Quick vs Hesit LME ===\n');
                utils_diary(fid_log, '%s', evalc('disp(lme_dur_qh.Coefficients)'));
                utils_diary(fid_log, '%s', evalc('anova(lme_dur_qh)'));
            end

            % Figure 2f
            data_nb_qh  = [nb_ratio_q   nb_ratio_h];
            data_dur_qh = [dur_ratio_q  dur_ratio_h];
            valid_nb  = ~any(isnan(data_nb_qh),2)  & ~any(isinf(data_nb_qh),2);
            valid_dur = ~any(isnan(data_dur_qh),2) & ~any(isinf(data_dur_qh),2);
            col_qh = [0.2 0.5 0.8; 0.8 0.3 0.3];

            fig2f = figure('Position',[50 50 840 400],'Color','w');
            rng(1); jitter_qh = 0.25;

            subplot(1,2,1); hold on;
            dj = (rand(nS_f,2)-0.5)*jitter_qh;
            for ii=1:nS_f
                if valid_nb(ii)
                    plot([1+dj(ii,1) 2+dj(ii,2)],[data_nb_qh(ii,1) data_nb_qh(ii,2)],...
                        '-','Color',[0.7 0.7 0.7 0.5]);
                end
            end
            for g=1:2
                scatter(g+dj(valid_nb,g), data_nb_qh(valid_nb,g), 20, col_qh(g,:), ...
                    'filled','MarkerEdgeColor','none','MarkerFaceAlpha',0.6);
            end
            if any(valid_nb)
                boxplot(data_nb_qh(valid_nb,:),'Colors',col_qh,'Symbol','','Widths',0.4,...
                    'Positions',[1 2],'Labels',{'Quick','Hesit'});
            end
            yline(1,'k--'); ylabel('Ratio Ch/Unch nb states','FontSize',13);
            title('Nb states: Quick vs Hesit','FontSize',13); xlim([0.5 2.5]);

            subplot(1,2,2); hold on;
            dj2 = (rand(nS_f,2)-0.5)*jitter_qh;
            for ii=1:nS_f
                if valid_dur(ii)
                    plot([1+dj2(ii,1) 2+dj2(ii,2)],[data_dur_qh(ii,1) data_dur_qh(ii,2)],...
                        '-','Color',[0.7 0.7 0.7 0.5]);
                end
            end
            for g=1:2
                scatter(g+dj2(valid_dur,g), data_dur_qh(valid_dur,g), 20, col_qh(g,:), ...
                    'filled','MarkerEdgeColor','none','MarkerFaceAlpha',0.6);
            end
            if any(valid_dur)
                boxplot(data_dur_qh(valid_dur,:),'Colors',col_qh,'Symbol','','Widths',0.4,...
                    'Positions',[1 2],'Labels',{'Quick','Hesit'});
            end
            yline(1,'k--'); ylabel('Ratio Ch/Unch duration','FontSize',13);
            title('Duration: Quick vs Hesit','FontSize',13); xlim([0.5 2.5]);

            sgtitle(strrep(thr_tag,'_',' '),'FontSize',13,'FontWeight','bold');
            % exportgraphics(fig2f, [report_dir 'Fig_extra_hesit_ratio.pdf'], 'ContentType', 'vector');
            close(fig2f);
        else
            fprintf('     No sessions with both quick and hesit trials — Fig 2f skipped.\n');
        end

        if ~exist(savefile, 'file')   % outer guard: skip STEP 5+6 entirely if CCGP already exists

        %% ------------------------------------------------------------
        %% STEP 5 — Build FR cache
        %%          Uses new states_cell + existing fr_heldout
        %% ------------------------------------------------------------
        fprintf('  [5/7] FR cache ...\n');

        if ~exist(fr_cache, 'file')
            monkey_name_cell = cellfun(@(s) s(1), sess_names, 'UniformOutput', false);

            all_fr_stacked    = [];
            all_behav_stacked = table();
            sess_list         = unique(sess_names);
            xx = 0;

            for s = 1:length(sess_list)
                % Find the entry for this session in out
                u = find(strcmp(sess_names, sess_list{s}), 1);
                if isempty(u), continue; end
                if isempty(states_cell{u}) || size(out(u).fr_heldout,1)==0, continue; end

                M_matched = states_cell{u};   % [nTrials x nBins], re-computed states

                for n = 1:size(out(u).fr_heldout, 1)
                    xx = xx + 1;
                    fr_n = NaN(size(out(u).fr_heldout, 2), 3, 'single');
                    for c_st = 1:3
                        tmp = squeeze(out(u).fr_heldout(n, :, :));  % [nTrials x nBins]
                        tmp(M_matched ~= c_st) = NaN;
                        fr_n(:, c_st) = single(nanmean(tmp, 2));
                    end
                    ar4unit = out(u).area4unit_heldout(n);
                    all_fr_stacked = [all_fr_stacked; fr_n];
                    all_behav_stacked = [all_behav_stacked; ...
                        table(single(xx * ones(size(fr_n,1),1)), 'VariableNames',{'Unit'}), ...
                        table(repmat(ar4unit(1),         size(fr_n,1),1), 'VariableNames',{'Area'}), ...
                        table(repmat(monkey_name_cell{u},size(fr_n,1),1), 'VariableNames',{'Monkey'}), ...
                        table(repmat(out(u).session,     size(fr_n,1),1), 'VariableNames',{'Session'}), ...
                        out(u).cond];
                end
            end

            fprintf('     FR cache: %d rows, %d neurons.\n', size(all_fr_stacked,1), xx);
            save(fr_cache, 'all_fr_stacked', 'all_behav_stacked', 'sess_list', '-v7.3');
        else
            fprintf('     Loading existing FR cache: %s\n', fr_cache);
            load(fr_cache, 'all_fr_stacked', 'all_behav_stacked', 'sess_list');
        end

        % Preference bias (matched by session name)
        pref = NaN(height(all_behav_stacked), 1);
        for s = 1:length(preference.session)
            row_match = strcmp(cellstr(all_behav_stacked.Session), preference.session{s});
            pref(row_match) = preference.bias_point(s);
        end
        bias_thr = 0;

        %% ------------------------------------------------------------
        %% STEP 6 — CCGP 2x2 decoding
        %%          Diagonal regularised LDA, same as main_003_crossdecoding.m
        %% ------------------------------------------------------------
        fprintf('  [6/7] CCGP decoding (Repetition=%d) ...\n', param_ccgp.Repetition);

            acc_overall = cell(length(mks), 1);
            for mk_idx = 1:length(mks)
                acc_overall{mk_idx} = cell(nAreas, length(param_ccgp.param2decode));
            end

            for p = 1:length(param_ccgp.param2decode)
                for m = 1:length(mks)
                    for ar = 1:nAreas

                        diff_label = all_behav_stacked.chosenflavor_2AFC ~= ...
                                     all_behav_stacked.unchosenflavor_2AFC;
                        row_nan_ok = ~isnan(all_fr_stacked(:,1)) & ~isnan(all_fr_stacked(:,2));
                        row_area   = ismember(all_behav_stacked.Area, area2test_name{ar});
                        row_pref   = diff_label & abs(pref) > bias_thr;
                        row_ok     = row_nan_ok & row_area & row_pref & ...
                                     ismember(all_behav_stacked.Monkey, mks{m});

                        fr_clean    = all_fr_stacked(row_ok, [1 2]);
                        behav_clean = all_behav_stacked(row_ok, :);

                        unit_id = unique(behav_clean.Unit);
                        cond_id = unique(behav_clean.(param_ccgp.param2decode{p}));
                        ntr = zeros(length(unit_id), length(cond_id));
                        for u_cnt = 1:length(unit_id)
                            for cd_cnt = 1:length(cond_id)
                                ntr(u_cnt,cd_cnt) = sum( ...
                                    behav_clean.Unit == unit_id(u_cnt) & ...
                                    behav_clean.(param_ccgp.param2decode{p}) == cond_id(cd_cnt));
                            end
                        end
                        unit2take = find(sum(ntr >= param_ccgp.minTr(p), 2) == length(cond_id));

                        fr_clean2    = fr_clean(ismember(behav_clean.Unit, unit_id(unit2take)), :);
                        behav_clean2 = behav_clean(ismember(behav_clean.Unit, unit_id(unit2take)), :);
                        behav_vals2  = [double(behav_clean2.(param_ccgp.param2decode{p})), ...
                                        double(behav_clean2.Unit)];
                        unit_id = unit_id(unit2take);

                        acc_overall{m}{ar,p} = NaN(length(param_ccgp.pseudopop), ...
                                                   param_ccgp.Repetition, 2, 2);

                        for u = 1:length(param_ccgp.pseudopop)
                            if length(unit2take) <= param_ccgp.pseudopop(u), continue; end
                            fprintf('     p=%d m=%d %s N=%d\n', p, m, area2test_name{ar}, ...
                                param_ccgp.pseudopop(u));

                            for pp = 1:param_ccgp.Repetition

                                unit_sub = randperm(length(unit2take), param_ccgp.pseudopop(u));
                                sel      = ismember(behav_vals2(:,2), unit_id(unit_sub));
                                data_pp  = fr_clean2(sel, :);
                                fact_pp  = behav_vals2(sel, :);

                                [XX, YY, ~] = utils_createpseudopop(data_pp, fact_pp, [1 2], ...
                                    'minTr', param_ccgp.minTr(p), 'perm', 1, 'pop', false);

                                nFolds  = 10;
                                nTrials = size(XX{1}, 1);
                                cv_pp   = cvpartition(nTrials, 'KFold', nFolds);

                                for trainState = 1:2
                                for testState  = 1:2
                                    scores_p = []; labels_p = [];

                                    for f = 1:nFolds
                                        trI = cv_pp.training(f);
                                        teI = cv_pp.test(f);

                                        X_tr = XX{trainState}(trI,:);
                                        y_tr = YY{trainState}(trI);
                                        X_te = XX{testState}(teI,:);
                                        y_te = YY{testState}(teI);

                                        mu_tr = mean(X_tr, 1);
                                        S_tr  = X_tr - mu_tr;
                                        S_te  = X_te - mu_tr;

                                        classes = unique(y_tr);
                                        if numel(classes) ~= 2, continue; end
                                        m1 = y_tr == classes(1);
                                        m2 = y_tr == classes(2);
                                        n1 = sum(m1); n2 = sum(m2);
                                        if n1 < 2 || n2 < 2, continue; end

                                        % Regularised diagonal LDA
                                        mu1 = mean(S_tr(m1,:), 1);
                                        mu2 = mean(S_tr(m2,:), 1);
                                        pv  = ((n1-1)*var(S_tr(m1,:),0,1) + ...
                                               (n2-1)*var(S_tr(m2,:),0,1)) / (n1+n2-2);
                                        pv_reg = pv + 0.01*mean(pv);
                                        w = ((mu1 - mu2) ./ pv_reg)';

                                        d_tr = S_tr * w;
                                        d_te = S_te * w;

                                        dmu1 = mean(d_tr(m1));
                                        dmu2 = mean(d_tr(m2));
                                        hsep = 0.5*(dmu1-dmu2);
                                        thr0 = 0.5*(dmu1+dmu2);
                                        if abs(hsep) < 1e-10, continue; end

                                        d_te_n   = (d_te - thr0) / hsep;
                                        y_binary = double(y_te == classes(1));

                                        scores_p = [scores_p; d_te_n(:)];
                                        labels_p = [labels_p; y_binary(:)];
                                    end  % folds

                                    if isempty(scores_p), continue; end
                                    acc_overall{m}{ar,p}(u,pp,trainState,testState) = ...
                                        mean(double(scores_p > 0) == labels_p);

                                end  % testState
                                end  % trainState

                            end  % repetitions
                        end  % pseudopop sizes

                    end  % areas
                end  % monkeys
            end  % parameters

            save(savefile, 'acc_overall', 'area2test_name', 'N_eval_global', '-v7.3');
            fprintf('     Saved: %s\n', savefile);

        else   % savefile already exists — skip STEP 5 and STEP 6 entirely
            fprintf('  [5-6/7] CCGP file exists — skipping FR build and decoding.\n');
            load(savefile, 'acc_overall');
        end   % if ~exist(savefile) outer guard

        %% ------------------------------------------------------------
        %% STEP 7 — CCGP LME + Fig 3c/3d-equivalent figures
        %% ------------------------------------------------------------
        fprintf('  [7/7] CCGP LME + figures ...\n');

        utils_diary(fid_log, '\n=== CCGP LME (thr=%.2f  dur=%d) ===\n', thr_now, dur_now);

        LogN_common = log2(N_eval_global);
        LogN_vec    = log2(param_ccgp.pseudopop);

        all_contrasts = cell(length(param_ccgp.param2decode), 1);

        for p = 1:length(param_ccgp.param2decode)

            param_name = strrep(param_ccgp.param2decode{p},'_2AFC','');
            fig_ttl    = strrep(param_ccgp.param2decode{p},'_',' ');

            % ---- Build LME table ----
            tab_Perf=[]; tab_Train={}; tab_Test={}; tab_Area={}; tab_Monkey={}; tab_N=[];

            for mk = 1:length(mks)
                for ar = 1:nAreas
                    if isempty(acc_overall{mk}) || isempty(acc_overall{mk}{ar,p}), continue; end
                    d = acc_overall{mk}{ar,p};
                    for ui = 1:length(param_ccgp.pseudopop)
                        if ui > size(d,1), break; end
                        for trnS = 1:2
                            for tstS = 1:2
                                v = squeeze(d(ui,:,trnS,tstS));
                                v = mean(v(~isnan(v)));
                                if isnan(v), continue; end
                                tab_Perf  = [tab_Perf;  v];
                                tab_Train = [tab_Train; state_labels(trnS)];
                                tab_Test  = [tab_Test;  state_labels(tstS)];
                                tab_Area  = [tab_Area;  area2test_name(ar)];
                                tab_Monkey= [tab_Monkey;mks(mk)];
                                tab_N     = [tab_N;     LogN_vec(ui)];
                            end
                        end
                    end
                end
            end

            lme_tab = table(tab_Perf, ...
                categorical(tab_Train, state_labels), ...
                categorical(tab_Test,  state_labels), ...
                categorical(tab_Area,  area2test_name), ...
                categorical(tab_Monkey,{'M','X'}), ...
                tab_N, ...
                'VariableNames',{'Perf','TrainState','TestState','Area','Monkey','NbNeuronsSat'});

            utils_diary(fid_log, '\n--- %s ---\n', param_ccgp.param2decode{p});
            lme = fitlme(lme_tab, ...
                'Perf ~ 1 + TrainState*TestState*Area*NbNeuronsSat + (1|Monkey)');
            utils_diary(fid_log, '%s', evalc('anova(lme)'));
            r2_cond = 1 - sum(residuals(lme).^2) / sum((lme_tab.Perf - mean(lme_tab.Perf)).^2);
            utils_diary(fid_log, 'Conditional R^2: %.4f\n', r2_cond);

            coef_names = lme.Coefficients.Name;
            coef_vals  = lme.Coefficients.Estimate;
            CovMat     = lme.CoefficientCovariance;
            dfe        = lme.DFE;

            % Detect reference levels
            is_area_m  = startsWith(coef_names,'Area_')       & ~contains(coef_names,':');
            is_train_m = startsWith(coef_names,'TrainState_') & ~contains(coef_names,':');
            is_test_m  = startsWith(coef_names,'TestState_')  & ~contains(coef_names,':');
            area_seen  = cellfun(@(s) s(6:end),  coef_names(is_area_m),  'UniformOutput',false);
            train_seen = cellfun(@(s) s(12:end), coef_names(is_train_m), 'UniformOutput',false);
            test_seen  = cellfun(@(s) s(11:end), coef_names(is_test_m),  'UniformOutput',false);
            ref_area   = setdiff(area2test_name, area_seen);  ref_area  = ref_area{1};
            ref_train  = setdiff(state_labels, train_seen);   ref_train = ref_train{1};
            ref_test   = setdiff(state_labels, test_seen);    ref_test  = ref_test{1};

            % Contrasts per area
            geomspec_est=NaN(nAreas,1); geomspec_se=NaN(nAreas,1);
            gainmod_est =NaN(nAreas,1); gainmod_se =NaN(nAreas,1);
            genrz_est   =NaN(nAreas,1); genrz_se   =NaN(nAreas,1);
            mu=NaN(nAreas,4); mu_se=NaN(nAreas,4);

            for ar = 1:nAreas
                al = area2test_name{ar};
                x11 = build_x(coef_names,'Chosen',  'Chosen',  al,LogN_common,ref_train,ref_test,ref_area);
                x12 = build_x(coef_names,'Chosen',  'Unchosen',al,LogN_common,ref_train,ref_test,ref_area);
                x21 = build_x(coef_names,'Unchosen','Chosen',  al,LogN_common,ref_train,ref_test,ref_area);
                x22 = build_x(coef_names,'Unchosen','Unchosen',al,LogN_common,ref_train,ref_test,ref_area);

                xvec = {x11,x12,x21,x22};
                for cc = 1:4
                    mu(ar,cc)    = xvec{cc}' * coef_vals;
                    mu_se(ar,cc) = sqrt(max(0, xvec{cc}' * CovMat * xvec{cc}));
                end

                xB = ((x11-x12)+(x22-x21))/2;
                geomspec_est(ar) = xB'*coef_vals;
                geomspec_se(ar)  = sqrt(max(0,xB'*CovMat*xB));

                xC = ((x11+x21)-(x12+x22))/2;
                gainmod_est(ar) = xC'*coef_vals;
                gainmod_se(ar)  = sqrt(max(0,xC'*CovMat*xC));

                xD = (x12+x21)/2;
                genrz_est(ar) = xD'*coef_vals - 0.5;
                genrz_se(ar)  = sqrt(max(0,xD'*CovMat*xD));
            end

            t_gs = geomspec_est./geomspec_se;  p_gs = 2*tcdf(-abs(t_gs),dfe);
            t_gm = gainmod_est ./gainmod_se;   p_gm = 2*tcdf(-abs(t_gm),dfe);
            t_gr = genrz_est   ./genrz_se;    p_gr = 2*tcdf(-abs(t_gr),dfe);
            [~,~,p_gs_fdr] = utils_fdr_bh(p_gs);
            [~,~,p_gm_fdr] = utils_fdr_bh(p_gm);
            [~,~,p_gr_fdr] = utils_fdr_bh(p_gr);

            % Log tables
            tbl_gs = table(area2test_name',geomspec_est,geomspec_se,t_gs,p_gs,p_gs_fdr,...
                'VariableNames',{'Area','Est','SE','t','p_raw','p_FDR'});
            utils_diary(fid_log,'\n--- Geometry specificity ---\n');
            utils_diary(fid_log,'%s',regexprep(evalc('disp(tbl_gs)'),'</?strong>',''));
            tbl_gm = table(area2test_name',gainmod_est,gainmod_se,t_gm,p_gm,p_gm_fdr,...
                'VariableNames',{'Area','Est','SE','t','p_raw','p_FDR'});
            utils_diary(fid_log,'\n--- Gain modulation ---\n');
            utils_diary(fid_log,'%s',regexprep(evalc('disp(tbl_gm)'),'</?strong>',''));
            tbl_gr = table(area2test_name',genrz_est,genrz_se,t_gr,p_gr,p_gr_fdr,...
                'VariableNames',{'Area','Est','SE','t','p_raw','p_FDR'});
            utils_diary(fid_log,'\n--- Generalization (off-diag - 0.5) ---\n');
            utils_diary(fid_log,'%s',regexprep(evalc('disp(tbl_gr)'),'</?strong>',''));

            % Store for summary
            ct.geomspec_est=geomspec_est; ct.geomspec_se=geomspec_se; ct.p_geomspec_fdr=p_gs_fdr;
            ct.gainmod_est =gainmod_est;  ct.gainmod_se =gainmod_se;  ct.p_gainmod_fdr =p_gm_fdr;
            ct.genrz_est   =genrz_est;    ct.genrz_se   =genrz_se;    ct.p_genrz_fdr   =p_gr_fdr;
            ct.mu=mu; ct.mu_se=mu_se;
            all_contrasts{p} = ct;

            % ---- Figure: heatmaps + contrast bars  (Fig 3c/3d style) ----
            mu_now = mu;
            fig_main = figure('Position',[50 50 841 750],'Color','w');

            ml=0.030; mr=0.018; mt=0.09; mb=0.10;
            rw=0.25; gap_mid=0.035; bc_gap=0.07;
            rx=1-mr-rw; rh=(1-mt-mb-bc_gap)/2;

            ax_gm_fig = axes('Parent',fig_main,'Units','normalized',...
                'Position',[rx, mb+rh+bc_gap, rw, rh]);
            ax_gs_fig = axes('Parent',fig_main,'Units','normalized',...
                'Position',[rx, mb, rw, rh]);

            lw=rx-gap_mid-ml;
            nhm_c=3; nhm_r=3;
            hm_gap_x=0.020; hm_gap_y=0.055; cb_h=0.022; cb_gap=0.032;
            hm_w=(lw-hm_gap_x*(nhm_c-1))/nhm_c;
            hm_h=(1-mt-mb-cb_h-cb_gap-hm_gap_y*(nhm_r-1))/nhm_r;
            cax=[min(0.45,min(mu_now(:))) max(mu_now(:))+0.01];
            hm_order=[1 2 3 9 8 4 7 6 5];

            for hm_idx = 1:nAreas
                ar = hm_order(hm_idx);
                ar_row = floor((hm_idx-1)/nhm_c);
                ar_col = mod(hm_idx-1,nhm_c);
                ax_hm = axes('Parent',fig_main,'Units','normalized',...
                    'Position',[ml+ar_col*(hm_w+hm_gap_x), ...
                                mb+cb_h+cb_gap+(nhm_r-1-ar_row)*(hm_h+hm_gap_y), hm_w, hm_h]);
                mat22 = [mu_now(ar,1) mu_now(ar,2); mu_now(ar,3) mu_now(ar,4)];
                imagesc(ax_hm, mat22, cax);
                colormap(ax_hm, flipud(hot(256)));
                is_bot = (ar_row==nhm_r-1); is_lft = (ar_col==0);
                if is_bot, set(ax_hm,'XTick',[1 2],'XTickLabel',{'C','U'},'FontSize',11);
                else,       set(ax_hm,'XTick',[1 2],'XTickLabel',{},'FontSize',11); end
                if is_lft, set(ax_hm,'YTick',[1 2],'YTickLabel',{'C','U'},'FontSize',11);
                else,       set(ax_hm,'YTick',[1 2],'YTickLabel',{},'FontSize',11); end
                set(ax_hm,'TickLength',[0 0]);
                if is_bot, xlabel(ax_hm,'Test state','FontSize',10); end
                if is_lft, ylabel(ax_hm,'Train state','FontSize',10); end
                title(ax_hm, area2test_name{ar}, 'Color',colorareas(ar,:)/255, ...
                    'FontWeight','bold','FontSize',12);
                for rr=1:2
                    for ci=1:2
                        v=mat22(rr,ci);
                        text(ax_hm,ci,rr,sprintf('%.3f',v),'HorizontalAlignment','center',...
                            'FontSize',9,'FontWeight','bold','Color',[1 1 1]*(v>mean(cax)));
                    end
                end
            end

            ax_cb = axes('Parent',fig_main,'Units','normalized',...
                'Position',[ml,mb,lw,cb_h],'Visible','off');
            colormap(ax_cb,flipud(hot(256))); clim(ax_cb,cax);
            cb = colorbar(ax_cb,'Location','South','AxisLocation','out');
            cb.Position=[ml,mb,lw,cb_h]; cb.Label.String='Decoding accuracy';
            cb.Label.FontSize=11; cb.FontSize=10; cb.TickLength=0.02;

            annotation(fig_main,'textbox',[0 0.93 1 0.06],...
                'String',['CCGP - ' fig_ttl ' | thr=' num2str(thr_now,'%.2f') ...
                          '  dur=' num2str(dur_now) '  (Rep=' num2str(param_ccgp.Repetition) ')'],...
                'EdgeColor','none','HorizontalAlignment','center','FontSize',14,'FontWeight','bold');

            % Geometry specificity bars
            [~,sidx_gs] = sort(geomspec_est,'descend');
            hold(ax_gs_fig,'on');
            for kk=1:nAreas
                ii=sidx_gs(kk); cb_=colorareas(ii,:)/255;
                if p_gs_fdr(ii)>=fdr_alpha, cb_=cb_*0.5+0.5; end
                bar(ax_gs_fig,kk,geomspec_est(ii),'FaceColor',cb_,'EdgeColor','none');
                errorbar(ax_gs_fig,kk,geomspec_est(ii),1.96*geomspec_se(ii),'k.','LineWidth',1.5);
                if p_gs_fdr(ii)<fdr_alpha
                    text(ax_gs_fig,kk,geomspec_est(ii)+sign(geomspec_est(ii))*(1.96*geomspec_se(ii)+0.003),...
                        '*','HorizontalAlignment','center','FontSize',13,'FontWeight','bold');
                end
            end
            set(ax_gs_fig,'XTick',1:nAreas,'XTickLabel',area2test_name(sidx_gs),'FontSize',10);
            xtickangle(ax_gs_fig,45); ylabel(ax_gs_fig,'LME estimate','FontSize',11);
            title(ax_gs_fig,'Geometry specificity','FontSize',12,'FontWeight','bold');
            yline(ax_gs_fig,0,'k--','LineWidth',1); box(ax_gs_fig,'off');

            % Gain modulation bars
            [~,sidx_gm] = sort(gainmod_est,'descend');
            hold(ax_gm_fig,'on');
            for kk=1:nAreas
                ii=sidx_gm(kk); cb_=colorareas(ii,:)/255;
                if p_gm_fdr(ii)>=fdr_alpha, cb_=cb_*0.5+0.5; end
                bar(ax_gm_fig,kk,gainmod_est(ii),'FaceColor',cb_,'EdgeColor','none');
                errorbar(ax_gm_fig,kk,gainmod_est(ii),1.96*gainmod_se(ii),'k.','LineWidth',1.5);
                if p_gm_fdr(ii)<fdr_alpha
                    text(ax_gm_fig,kk,gainmod_est(ii)+sign(gainmod_est(ii))*(1.96*gainmod_se(ii)+0.003),...
                        '*','HorizontalAlignment','center','FontSize',13,'FontWeight','bold');
                end
            end
            set(ax_gm_fig,'XTick',1:nAreas,'XTickLabel',area2test_name(sidx_gm),'FontSize',10);
            xtickangle(ax_gm_fig,45); ylabel(ax_gm_fig,'LME estimate','FontSize',11);
            title(ax_gm_fig,'Gain modulation','FontSize',12,'FontWeight','bold');
            yline(ax_gm_fig,0,'k--','LineWidth',1); box(ax_gm_fig,'off');

            drawnow;
            % Name files to match main_003 convention (3c = flavor, 3d = side)
            if strcmp(param_name,'chosenflavor')
                fig_letter = '3c';
            elseif strcmp(param_name,'chosenside')
                fig_letter = '3d';
            else
                fig_letter = ['3_' param_name];
            end
            fname = [report_dir 'Fig_' fig_letter '_CCGP_' param_name '.pdf'];
            exportgraphics(fig_main, fname, 'ContentType', 'vector');
            close(fig_main);
            fprintf('     Saved: %s\n', fname);

        end  % param loop

        summary_ccgp{ti,di} = all_contrasts;

        % Append contrasts to the savefile
        save(savefile, 'all_contrasts', '-append');

        clear log_cleanup;  % closes fid_log via onCleanup
        fprintf('  Combo %s done.\n', thr_tag);

    end  % di
    end  % ti

    save(summary_file, 'summary_nb', 'summary_dur', 'summary_ccgp', 'summary_valid_frac', ...
        'summary_pval_nb_state', 'summary_pval_dur_state', 'sess_names', 'monks', 'nSess', '-v7.3');
    fprintf('Saved: %s\n', summary_file);

else
    fprintf('Loading existing summary: %s\n', summary_file);
    load(summary_file, 'summary_nb', 'summary_dur', 'summary_ccgp', 'summary_valid_frac', ...
        'summary_pval_nb_state', 'summary_pval_dur_state', 'sess_names', 'monks', 'nSess');
    summary_dir = [currentPath '\report_005\'];
    if ~exist(summary_dir, 'dir'), mkdir(summary_dir); end
end

%% ============================================================
%% SUMMARY FIGURES — cross-combo comparison (line plots)
%% ============================================================

fprintf('\nBuilding cross-combo summary figures ...\n');

% Combo index layout: ci = (di-1)*nThr + ti
% Groups: dur=dur_vals(1) -> ci 1..nThr | dur=dur_vals(2) -> ci nThr+1..2*nThr | etc.
nCombos = nThr * nDur;
combo_labels = cell(nCombos, 1);
ci = 0;
for di2 = 1:nDur
    for ti2 = 1:nThr
        ci = ci + 1;
        combo_labels{ci} = sprintf('%.2f/%d', thr_vals(ti2), dur_vals(di2));
    end
end
ref_combo = (find(dur_vals==5)-1)*nThr + find(thr_vals==0.30);

%% --- Valid trial fraction heatmap ---
fig_vt = figure('Position', [30 30 580 380], 'Color', 'w');
imagesc(thr_vals, dur_vals, summary_valid_frac');
colormap(hot); clim([0 1]);
cb_vt = colorbar; cb_vt.Label.String = 'Fraction of valid trials'; cb_vt.FontSize = 10;
hold on;
for ti2 = 1:nThr
    for di2 = 1:nDur
        val = summary_valid_frac(ti2, di2);
        if ~isnan(val)
            text(thr_vals(ti2), dur_vals(di2), sprintf('%.0f%%', 100*val), ...
                'HorizontalAlignment','center','FontSize',12,...
                'Color',[1 1 1]*(val < 0.6),'FontWeight','bold');
        end
    end
end
dx_thr = thr_vals(2) - thr_vals(1);
dy_dur = dur_vals(2)  - dur_vals(1);
ref_ti = find(thr_vals == 0.30);
ref_di = find(dur_vals  == 5);
if ~isempty(ref_ti) && ~isempty(ref_di)
    rectangle('Position',[thr_vals(ref_ti)-dx_thr/2, dur_vals(ref_di)-dy_dur/2, dx_thr, dy_dur],...
        'EdgeColor','w','LineWidth',3);
end
xlabel('thr\_state','FontSize',12);
ylabel('thr\_state\_dur (bins)','FontSize',12);
title({'Fraction of trials with \geq1 Chosen AND \geq1 Unchosen state',...
    '(white box = reference: thr=0.30, dur=5)'},'FontSize',11);
set(gca,'XTick',thr_vals,'YTick',dur_vals,'TickLength',[0 0],'FontSize',10);
exportgraphics(fig_vt, [summary_dir 'Fig_R1b_valid_trials.pdf'], 'ContentType', 'vector');
fprintf('  Saved: Fig_R1b_valid_trials.pdf\n');

%% --- Summary A: geomspec / gainmod / genrz — Fig-S8 style, 2×3 panels ---
% Top row:    thr = 0.30 fixed,  x-axis = dur_vals  (duration sweep)
% Bottom row: dur = 5 fixed,     x-axis = thr_vals  (threshold sweep)
% Columns:    Gain modulation | Geometry specificity | Generalization
% Each area = one colored line; filled dot = |t| > 1.96; shaded 95% CI band.

thr_ref_s7 = 0.30;   ti_ref_s7 = find(thr_vals == thr_ref_s7);
dur_ref_s7 = 5;      di_ref_s7 = find(dur_vals  == dur_ref_s7);

cont_flds_s7 = {'gainmod_est','gainmod_se'; ...
                'geomspec_est','geomspec_se'; ...
                'genrz_est','genrz_se'};
cont_lbs_s7  = {'Gain modulation', 'Geometry specificity', 'Generalization'};

for p_sum = 1:length(param_ccgp.param2decode)
    param_name_sum = strrep(param_ccgp.param2decode{p_sum},'_2AFC','');

    fig_s7 = figure('Position',[30 30 1450 700],'Color','w');
    tl_s7  = tiledlayout(fig_s7, 2, 3, 'TileSpacing','compact','Padding','compact');
    title(tl_s7, [strrep(param_ccgp.param2decode{p_sum},'_',' ') ...
        '  |  LME contrasts vs threshold / duration parameters'], ...
        'FontSize',14,'FontWeight','bold');

    for row = 1:2
        if row == 1
            nx         = nDur;
            x_vals_s7  = dur_vals;
            x_ref_now  = dur_ref_s7;
            x_lbl_s7   = 'thr\_state\_dur (bins)';
            row_lbl_s7 = sprintf('thr = %.2f   (dur sweep)', thr_ref_s7);
        else
            nx         = nThr;
            x_vals_s7  = thr_vals;
            x_ref_now  = thr_ref_s7;
            x_lbl_s7   = 'thr\_state';
            row_lbl_s7 = sprintf('dur = %d   (thr sweep)', dur_ref_s7);
        end

        for col = 1:3
            ax_s7 = nexttile((row-1)*3 + col);
            hold(ax_s7,'on'); box(ax_s7,'off');

            for ar = 1:nAreas
                col_ar = colorareas(ar,:) / 255;

                v = NaN(nx,1);
                s = NaN(nx,1);
                for xi = 1:nx
                    if row == 1
                        ti2 = ti_ref_s7; di2 = xi;
                    else
                        ti2 = xi; di2 = di_ref_s7;
                    end
                    ctc = summary_ccgp{ti2, di2};
                    if isempty(ctc) || numel(ctc) < p_sum || isempty(ctc{p_sum}), continue; end
                    v(xi) = ctc{p_sum}.(cont_flds_s7{col,1})(ar);
                    s(xi) = ctc{p_sum}.(cont_flds_s7{col,2})(ar);
                end

                ok_x = ~isnan(v) & ~isnan(s);
                if ~any(ok_x), continue; end

                xv = x_vals_s7(ok_x);  xv = xv(:)';   % force row
                yv = v(ok_x);          yv = yv(:)';
                ys = s(ok_x);          ys = ys(:)';

                % 95% CI shading
                fill(ax_s7, [xv, fliplr(xv)], [yv+1.96*ys, fliplr(yv-1.96*ys)], ...
                    col_ar,'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');

                % Line (no regular markers, like Fig_S8)
                plot(ax_s7, xv, yv, 'Color', col_ar, 'LineWidth', 2, ...
                    'Marker','none', 'DisplayName', area2test_name{ar});

                % Filled dot at significant points (|t| > 1.96)
                sig_m = abs(yv ./ max(ys, 1e-12)) > 1.96;
                if any(sig_m)
                    plot(ax_s7, xv(sig_m), yv(sig_m), 'o', ...
                        'Color', col_ar, 'MarkerSize', 9, ...
                        'MarkerFaceColor', col_ar, 'MarkerEdgeColor','none', ...
                        'HandleVisibility','off');
                end
            end

            yline(ax_s7, 0,'k--','LineWidth',1);
            xline(ax_s7, x_ref_now, 'Color',[0.4 0.4 0.4],'LineStyle',':', ...
                'LineWidth',2,'HandleVisibility','off');

            dx_s7 = x_vals_s7(2) - x_vals_s7(1);
            set(ax_s7,'XTick',x_vals_s7,'FontSize',11,'TickLength',[0 0]);
            xlim(ax_s7, [x_vals_s7(1)-dx_s7/2, x_vals_s7(end)+dx_s7/2]);
            xlabel(ax_s7, x_lbl_s7,'FontSize',11);

            if row == 1
                title(ax_s7, cont_lbs_s7{col},'FontSize',12,'FontWeight','bold');
            end
            if col == 1
                ylabel(ax_s7, sprintf('%s\nLME estimate', row_lbl_s7),'FontSize',10);
            end
            if row == 1 && col == 3
                legend(ax_s7,'Location','eastoutside','FontSize',8,'Box','off');
            end
        end
    end

    fname_s7 = [summary_dir sprintf('Fig_S9_robustness_%s.pdf', param_name_sum)];
    exportgraphics(fig_s7, fname_s7, 'ContentType', 'vector');
    fprintf('  Saved: %s\n', fname_s7);
end

%% --- Summary B+C: nb/dur ratio — per-monkey line plots across threshold combos ---

nb_ratio_all  = NaN(nSess, nCombos);
dur_ratio_all = NaN(nSess, nCombos);

% --- ADDED: Arrays for Chosen / Other ratio ---
nb_ratio_ch_oth  = NaN(nSess, nCombos);
dur_ratio_ch_oth = NaN(nSess, nCombos);
% ----------------------------------------------

ci = 0;
for di2 = 1:nDur
    for ti2 = 1:nThr
        ci = ci + 1;
        if isempty(summary_nb{ti2,di2}) || isempty(summary_dur{ti2,di2}), continue; end
        
        % Ch / Unch
        nb_ratio_all(:,ci)  = summary_nb{ti2,di2}(:,1)  ./ summary_nb{ti2,di2}(:,2);
        dur_ratio_all(:,ci) = summary_dur{ti2,di2}(:,1) ./ summary_dur{ti2,di2}(:,2);
        
        % --- ADDED: Ch / Other (indices 1 and 3) ---
        nb_ratio_ch_oth(:,ci)  = summary_nb{ti2,di2}(:,1)  ./ summary_nb{ti2,di2}(:,3);
        dur_ratio_ch_oth(:,ci) = summary_dur{ti2,di2}(:,1) ./ summary_dur{ti2,di2}(:,3);
        % -------------------------------------------
    end
end

[~, ~, p_fdr_nb]  = utils_fdr_bh(summary_pval_nb_state(:));
[~, ~, p_fdr_dur] = utils_fdr_bh(summary_pval_dur_state(:));
met_pvals = {p_fdr_nb, p_fdr_dur};

%% --- Fig S5E: Chosen / Unchosen and Chosen / Other state ratios ---
plot_state_ratios({nb_ratio_all, dur_ratio_all}, ...
    {{'Ratio of Chosen/Unchosen', 'state number'}, {'Ratio of Chosen/Unchosen', 'state duration'}}, ...
    monks, mks, mk_col, thr_vals, dur_vals, 0.30, 5, [summary_dir 'Fig_S5e_ratio_ch_unch']);
fprintf('  Saved: Fig_S5e_ratio_ch_unch.pdf\n');

plot_state_ratios({nb_ratio_ch_oth, dur_ratio_ch_oth}, ...
    {{'Ratio of Chosen/Other', 'state number'}, {'Ratio of Chosen/Other', 'state duration'}}, ...
    monks, mks, mk_col, thr_vals, dur_vals, 0.30, 5, [summary_dir 'Fig_S5e_ratio_ch_other']);
fprintf('  Saved: Fig_S5e_ratio_ch_other.pdf\n');

fprintf('\nDone. All output in:\n  per-combo: %sreport_005\\<thr_tag>\\\n', currentPath);
fprintf('  summary:  %s\n', summary_dir);
%% ============================================================
%% LOCAL FUNCTIONS  (copied from main_003_crossdecoding.m)
%% ============================================================

function x = build_x(coef_names, trnS_lbl, tstS_lbl, ar_lbl, Ns, ...
                      ref_train, ref_test, ref_area)
    n = length(coef_names);
    x = zeros(n, 1);
    is_unc_trn   = ~strcmp(trnS_lbl, ref_train);
    is_unc_tst   = ~strcmp(tstS_lbl, ref_test);
    is_nonref_ar = ~strcmp(ar_lbl,   ref_area);
    active = {};
    if is_unc_trn,   active{end+1} = ['TrainState_' trnS_lbl]; end
    if is_unc_tst,   active{end+1} = ['TestState_'  tstS_lbl]; end
    if is_nonref_ar, active{end+1} = ['Area_'       ar_lbl];   end
    n_act = length(active);
    for bits = 0 : (2^n_act - 1)
        subset = {};
        for k = 1:n_act
            if bitand(bits, 2^(k-1)), subset{end+1} = active{k}; end
        end
        if isempty(subset)
            idx = find_coef_exact(coef_names, {'(Intercept)'});
            if ~isempty(idx), x(idx) = x(idx) + 1; end
        else
            idx = find_coef_exact(coef_names, subset);
            if idx > 0, x(idx) = x(idx) + 1; end
        end
        subset_ns = [subset, {'NbNeuronsSat'}];
        idx_ns = find_coef_exact(coef_names, subset_ns);
        if idx_ns > 0, x(idx_ns) = x(idx_ns) + Ns; end
    end
end

function idx = find_coef_exact(coef_names, required_parts)
    n_req = length(required_parts);
    idx   = 0;
    for k = 1:length(coef_names)
        parts_k = strsplit(coef_names{k}, ':');
        if length(parts_k) ~= n_req, continue; end
        if all(ismember(required_parts, parts_k)) && ...
           all(ismember(parts_k, required_parts))
            idx = k;
            return;
        end
    end
end

function fg = plot_state_ratios(met_data, ylabs, monks, mks, mk_col, thr_vals, dur_vals, ref_thr, ref_dur, fname)
% Fig S5E row: per-monkey mean +/- SEM (across sessions) of a state ratio for
% each threshold x duration combination. One panel per metric (met_data{k}:
% sessions x combos, combo index ci = (di-1)*nThr + ti), with one block per
% duration. The arrow marks the reference combination used in the paper.
nThr = numel(thr_vals);  nDur = numel(dur_vals);
xpos = @(ti, di) (di - 1)*(nThr + 1) + ti;            % one empty slot between blocks
xall = xpos(repmat((1:nThr)', 1, nDur), repmat(1:nDur, nThr, 1));
xall = xall(:)';                                     % same order as ci

W = 18;  H = 6.8;
cm = @(x, y, w, h) [x y w h] ./ [W H W H];
fg = figure('Units', 'centimeters', 'Position', [2 2 W H], 'Color', 'w', ...
    'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial');

for k = 1 : numel(met_data)
    ax = axes(fg, 'Position', cm(2.0 + (k-1)*9.1, 1.6, 6.1, 4.9));  hold(ax, 'on');

    % Mean +/- SEM per monkey
    mu = NaN(numel(mks), numel(xall));  se = mu;
    for m = 1 : numel(mks)
        d = met_data{k}(strcmp(monks, mks{m}), :);
        mu(m,:) = mean(d, 1, 'omitnan');
        se(m,:) = std(d, 0, 1, 'omitnan') ./ sqrt(max(sum(~isnan(d), 1), 1));
    end
    lo = min(mu - se, [], 'all');  hi = max(mu + se, [], 'all');  rg = hi - lo;
    yl = [min(lo - 0.05*rg, 1), hi + 0.18*rg];
    xl = [0.4, xpos(nThr, nDur) + 0.6];               % data range (labels go to the right)

    for m = 1 : numel(mks)
        for di = 1 : nDur
            c = (di - 1)*nThr + (1:nThr);
            x = xall(c);  y = mu(m,c);  s = se(m,c);  ok = ~isnan(y);
            if ~any(ok), continue; end
            fill(ax, [x(ok) fliplr(x(ok))], [y(ok)+s(ok) fliplr(y(ok)-s(ok))], mk_col(m,:), ...
                'FaceAlpha', 0.25, 'EdgeColor', 'none');
            plot(ax, x(ok), y(ok), '-o', 'Color', mk_col(m,:), 'LineWidth', 1.5, 'MarkerSize', 4, ...
                'MarkerFaceColor', mk_col(m,:), 'MarkerEdgeColor', mk_col(m,:)*0.6);
        end
    end

    % Reference line at 1, block separators and titles
    if yl(1) < 1
        plot(ax, xl, [1 1], 'k--', 'LineWidth', 0.8);
    end
    for di = 1 : nDur
        if di > 1
            plot(ax, [1 1]*(xpos(1, di) - 1), yl, 'k--', 'LineWidth', 0.8);
        end
        text(ax, mean(xpos([1 nThr], di)), yl(2), sprintf('%d bins', dur_vals(di)), 'FontSize', 10, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
    end

    % Monkey labels at the end of each line
    for m = 1 : numel(mks)
        text(ax, xall(end) + 0.35, mu(m, end), ['mk ' mks{m}], 'Color', mk_col(m,:), 'FontSize', 10, ...
            'FontAngle', 'italic', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    end

    % Arrow below the data at the reference combination
    ci_ref = (find(dur_vals == ref_dur) - 1)*nThr + find(thr_vals == ref_thr);
    xr  = xall(ci_ref);
    top = min(mu(:, ci_ref) - se(:, ci_ref)) - 0.05*rg;
    ah  = min(0.16*rg, top - yl(1) - 0.02*rg);
    if ah > 0.04*rg
        yb = top - ah;
        px = xr + [-0.12 0.12 0.12 0.28 0 -0.28 -0.12];
        py = yb + ah*[0 0 0.55 0.55 1 0.55 0.55];
        fill(ax, px, py, [0.9 0.9 0.9], 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.6);
    end

    set(ax, 'XTick', xall, 'XTickLabel', arrayfun(@(t) sprintf('%.2f', t), repmat(thr_vals, 1, nDur), ...
        'UniformOutput', false), 'XTickLabelRotation', 45, 'FontSize', 9, 'TickDir', 'out', ...
        'TickLength', [0.015 0.015], 'Box', 'off', 'LineWidth', 0.6);
    xlim(ax, xl);  ylim(ax, yl);
    xlabel(ax, 'State threshold', 'FontSize', 11);
    ylabel(ax, ylabs{k}, 'FontSize', 11);
end

if nargin >= 10 && ~isempty(fname)
    exportgraphics(fg, [fname '.pdf'], 'ContentType', 'vector');   % for CorelDRAW / Illustrator
end
end
