%% main_003_crossdecoding.m
%
% Cross-decoding with regularized diagonal LDA (no PCA in the fold-level decoder).
%
% Pseudopopulations have zero noise correlations by construction, so the
% true Sw is diagonal.  The correct decoder is diagonal LDA in the full
% N-dimensional space:
%   w_i = (mu1_i - mu2_i) / (pv_i + alpha*mean(pv))
% where alpha=0.01 provides L2 regularization against near-silent neurons.
% No PCA is used here because pseudopop samples are already decorrelated.
%
% Full 2x2 cross-decoding producing both CCGP and shCCGP for all four cells
% (CC, CU, UC, UU), following Nogueira et al. 2026.
%
% Key design choices
%
%   CCGP  = accuracy with training-calibrated threshold 
%   shCCGP = accuracy with optimally shifted threshold 
%
%   Both are computed by pooling CV-fold scores within each (trainState,
%   testState) direction, then:
%     CCGP  = accuracy at threshold delta = 0   (training midpoint mapped to 0)
%     shCCGP = max accuracy over a fine threshold grid delta in [-3, 3]
%
%   Pooling is valid because each trial appears in exactly one test fold
%   (no train/test overlap).  Scores from different folds are normalised by
%   the within-fold training class-separation before pooling so that the
%   training threshold is always at 0 in normalised units.  With minTr = 20
%   this yields 20 test observations per class per direction (pooled over
%   10 folds x 2 trials), giving a stable threshold scan with no inflation.
%
%   NOTE on within-state cells (CC and UU):
%     The training and test distributions are from the SAME state, so the
%     training-calibrated threshold is appropriate and shCCGP ~ CCGP.  Any
%     small improvement reflects sampling noise, not a midpoint shift.  This
%     acts as a built-in negative control for the H2 test.
%
% H2 test (shift / midpoint displacement)
%
%   Improvement  = shCCGP - CCGP   (per cell)
%
%   Generalization of the improvement 2x2 = mean(CU_improv, UC_improv) - mean(CC_improv, UU_improv)
%
%   Interpretation:
%     genrz_improv > 0 & FDR-significant:
%       The optimal threshold is displaced cross-state MORE than within-state
%       -> the decision midpoint shifts between states -> H2
%     genrz_improv ~ 0:
%       No systematic threshold shift -> axis rotated (H3) or scaled (H4)
%       rather than shifted.
%
% Outputs:
%   processed/states_2afc_ccgp*.mat        acc_overall, shc_overall, imp_overall
%   report_003*/                           figures and output log
%
% Dependencies
%   utils_fdr_bh.m, utils_createpseudopop.m, utils_matchTimes.m, utils_matchBins.m
%   states_2afc_final.mat                  built by main_002_states.m
%   states_2afc_fr*.mat                    (built by this script on first run)
%   behav_pref.mat

clear
overwrite_fr       = false;   % recompute FR matrix cache
overwrite_decoding = false;   % rerun decoding (outer save file)
plot_shCCGP = false;

%% **************** Parameters **************************************************************
f = mfilename('fullpath');
if isempty(f) 
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end
addpath(genpath([currentPath '\scripts\'])); % add scripts folder to path

param.path2go        = [currentPath '\processed\'];
param.pseudopop      = round(logspace(log10(25), log10(500), 25));   % 30 log-spaced sizes
param.param2decode   = {'chosenflavor_2AFC' 'chosenside_2AFC'};
param.minTr          = [20 20];
param.Repetition     = 100;
param.match_type     = 'none';% time bin label_swap none
param.init_decoder   = '';

N_eval_global = 200;   % fixed evaluation point for reported contrasts
fdr_alpha     = 0.05;  % FDR significance threshold

% Threshold scan grid for shCCGP (in normalised units where training threshold = 0,
% class means on training data are at ~= +/-1).  101 points over [-3, 3] gives
% resolution of 0.06 - negligible bias at minTr = 20 per class.
delta_grid = linspace(-3, 3, 101);

switch param.match_type
    case 'time',        match_sfx = '_timematched';
    case 'bin',         match_sfx = '_binmatched';
    case 'label_swap',  match_sfx = '_labelswap';   % within-trial Chosen/Unchosen label-swap null
    otherwise,          match_sfx = '';
end
switch param.init_decoder
    case 'timept', init_sfx = '_timept';
    otherwise,     init_sfx = '';
end

area2test_name = {'MFC' 'PMC' 'dlPFC' 'IFG' 'vlPFC' 'AI' 'OFC' 'STR' 'AMG'};
nAreas = length(area2test_name);

colorareas = [230 171 2 ; 152 78 163 ; 237 87 90 ; 252 141 98 ; 141 160 203 ; ...
              166 216 84 ; 102 194 165 ; 180 180 180 ; 231 138 195];

state_labels = {'Chosen', 'Unchosen'};
mks = {'M', 'X'};

%% I/O paths

report_dir = [currentPath '\report_003' match_sfx init_sfx '\'];
if ~exist(report_dir, 'dir'), mkdir(report_dir); end

fr_cache = [param.path2go 'states_2afc_fr' match_sfx init_sfx '.mat'];
savefile = [param.path2go 'states_2afc_ccgp' match_sfx init_sfx '.mat'];

%% Compute 2x2 CCGP and shCCGP

if ~exist(savefile, 'file') || overwrite_decoding

    % Build or load firing rate cache
    %
    % The cache (states_2afc_fr*.mat) stores one row per trial per
    % neuron.  Columns 1-3 of all_fr_stacked are mean firing rates in the
    % Chosen, Unchosen, and Other HMM states respectively (averaged over all
    % time bins assigned to that state for that trial).
    %
    % all_behav_stacked has matching rows with per-trial behaviour columns:
    %   Unit (integer ID), Area, Monkey, Session, and all cond.* fields
    %   (chosenflavor_2AFC, unchosenflavor_2AFC, chosenside_2AFC, ...).
    %
    % Source: states_2afc_final.mat from main_002_states.m.
    % Only units with area4unit_removed == '' (i.e., not held-out area-drop
    % units) are included, matching the same inclusion filter used below.
    %
    % match_type controls optional trial/bin equalisation or null generation:
    %   'none'        - raw states (default)
    %   'time'        - equalise n_chosen / n_unchosen trials per time bin
    %   'bin'         - equalise n_chosen / n_unchosen bins per trial
    %   'label_swap'  - for each trial independently (p=0.5), swap the
    %                   Chosen<->Unchosen state labels while keeping the same
    %                   time bins; FR values come from identical windows but
    %                   the identity (Chosen vs Unchosen) is randomised.
    %                   Null for all CCGP/shCCGP contrasts.

    if ~exist(fr_cache, 'file') || overwrite_fr

        % Load the held-out pseudopopulation data if not already in workspace
        if ~exist('out_all', 'var')
            fprintf('Loading out_all from states_2afc_final%s.mat ...\n', init_sfx)
            load([param.path2go 'states_2afc_final' init_sfx '.mat'], 'out_all')
        end

        all_sess    = {out_all(:).session}';
        monkey_name = cellfun(@(x) x(1), all_sess, 'UniformOutput', false);
        areas       = {out_all(:).area4unit_removed}';
        sess_list   = unique(all_sess);
        xx          = 0;

        all_fr_stacked    = [];
        all_behav_stacked = table();

        for s = 1 : length(sess_list)
            fprintf('  Building FR cache: session %d / %d\n', s, length(sess_list))

            % All entries for this session where no area was held out
            takeme = find(ismember(all_sess, sess_list{s}) & cellfun(@isempty, areas));

            for u_idx = 1 : length(takeme)
                u = takeme(u_idx);

                % Per-unit state assignment, optionally balanced (same for all
                % neurons in this batch because states are session-level)
                switch param.match_type
                    case 'time'
                        % Equalise chosen / unchosen trial counts per time bin
                        M_matched = utils_matchTimes(out_all(u).states);
                    case 'bin'
                        % Equalise chosen / unchosen bin counts per trial
                        M_matched = utils_matchBins(out_all(u).states);
                    case 'label_swap'
                        % Null: for each trial independently (Bernoulli p=0.5),
                        % swap state labels 1 (Chosen) <-> 2 (Unchosen).
                        % The same time bins are used - only the identity of
                        % which bins are called "Chosen" vs "Unchosen" is
                        % randomised.  State 3 (Other) is untouched.
                        M_matched = utils_swapStateLabels(out_all(u).states);
                    otherwise
                        M_matched = out_all(u).states;   % no balancing
                end

                % One row per (trial x neuron); columns = HMM states 1-3
                for n = 1 : size(out_all(u).fr_heldout, 1)
                    xx = xx + 1;

                    % Average firing rate within each HMM state for this trial
                    fr_n = NaN(size(out_all(u).fr_heldout, 2), 3, 'single');
                    for c_st = 1 : 3
                        tmp = squeeze(out_all(u).fr_heldout(n, :, :));  % trials x bins
                        tmp(M_matched ~= c_st) = NaN;
                        fr_n(:, c_st) = single(nanmean(tmp, 2));
                    end

                    ar4unit = out_all(u).area4unit_heldout(n);
                    all_fr_stacked    = [all_fr_stacked; fr_n];           
                    all_behav_stacked = [all_behav_stacked; ...           
                        table(single(xx * ones(size(fr_n, 1), 1)), 'VariableNames', {'Unit'}), ...
                        table(repmat(ar4unit(1),          size(fr_n, 1), 1), 'VariableNames', {'Area'}), ...
                        table(repmat(monkey_name{u},      size(fr_n, 1), 1), 'VariableNames', {'Monkey'}), ...
                        table(repmat(out_all(u).session,  size(fr_n, 1), 1), 'VariableNames', {'Session'}), ...
                        out_all(u).cond];
                end
            end
        end

        fprintf('FR cache built: %d trial-rows across %d neurons.\n', ...
                size(all_fr_stacked, 1), xx)
        save(fr_cache, 'all_fr_stacked', 'all_behav_stacked', 'sess_list', '-v7.3');
        fprintf('Saved FR cache: %s\n', fr_cache);

    else
        fprintf('Loading existing FR cache: %s\n', fr_cache)
        load(fr_cache, 'all_fr_stacked', 'all_behav_stacked', 'sess_list')
    end

    % Load preference bias
    % preference.bias_point(s) = the delta value at which the monkey is indifferent
    % between the two flavours in session s.  Trials where the preferred and
    % non-preferred flavours differ (diff_label) AND |bias| > bias_thr are
    % used for decoding (same filter used across this script).
    load([param.path2go 'behav_pref.mat'])
    bias_thr = 0;
    pref = NaN(height(all_behav_stacked), 1);
    for s = 1 : length(sess_list)
        pref(ismember(all_behav_stacked.Session, preference.session(s))) = ...
            preference.bias_point(s);
    end

    % Output arrays: [n_psz x n_rep x 2 x 2] per monkey x area x param
    %   dim 3 = trainState (1=Chosen, 2=Unchosen)
    %   dim 4 = testState
    acc_overall = cell(length(mks), 1);   % CCGP (accuracy, fixed threshold)
    shc_overall = cell(length(mks), 1);   % shCCGP (accuracy, optimal threshold)
    % bias_curve: mean accuracy-vs-delta curve at N_eval only
    %   {mk}{ar,p}(trainState, testState, n_delta)  - averaged over repetitions
    bias_curve           = cell(length(mks), 1);
    bias_curve_N_used    = cell(length(mks), 1);   % N actually used per area/monkey/param
    % peak_loc_reps_final: per-repetition peak delta* at the largest valid N
    %   {mk}{ar,p}(trainState, testState, n_rep) - delta_grid value of shCCGP peak per rep
    peak_loc_reps_final  = cell(length(mks), 1);
    for mk_idx = 1 : length(mks)
        acc_overall{mk_idx}         = cell(nAreas, length(param.param2decode));
        shc_overall{mk_idx}         = cell(nAreas, length(param.param2decode));
        bias_curve{mk_idx}          = cell(nAreas, length(param.param2decode));
        bias_curve_N_used{mk_idx}   = cell(nAreas, length(param.param2decode));
        peak_loc_reps_final{mk_idx} = cell(nAreas, length(param.param2decode));
    end
    n_delta = length(delta_grid);

    for p = 1 : length(param.param2decode)
        for m = 1 : length(mks)
            for ar = 1 : nAreas

                % Trial selection
                diff_label = all_behav_stacked.chosenflavor_2AFC ~= ...
                             all_behav_stacked.unchosenflavor_2AFC;
                row_nan_ok = ~isnan(all_fr_stacked(:,1)) & ~isnan(all_fr_stacked(:,2));
                row_area   = ismember(all_behav_stacked.Area, area2test_name{ar});
                row_pref   = diff_label & abs(pref) > bias_thr;
                row_ok     = row_nan_ok & row_area & row_pref & ...
                             ismember(all_behav_stacked.Monkey, mks{m});

                fr_clean    = all_fr_stacked(row_ok, [1 2]);
                behav_clean = all_behav_stacked(row_ok, :);

                % Unit eligibility
                unit_id = unique(behav_clean.Unit);
                cond_id = unique(behav_clean.(param.param2decode{p}));
                ntr = zeros(length(unit_id), length(cond_id));
                for u_cnt = 1 : length(unit_id)
                    for cd = 1 : length(cond_id)
                        ntr(u_cnt, cd) = sum( ...
                            behav_clean.Unit == unit_id(u_cnt) & ...
                            behav_clean.(param.param2decode{p}) == cond_id(cd));
                    end
                end
                unit2take = find(sum(ntr >= param.minTr(p), 2) == length(cond_id));

                fr_clean2    = fr_clean(ismember(behav_clean.Unit, unit_id(unit2take)), :);
                behav_clean2 = behav_clean(ismember(behav_clean.Unit, unit_id(unit2take)), :);
                behav_clean2 = [double(behav_clean2.(param.param2decode{p})), ...
                                double(behav_clean2.Unit)];
                unit_id = unit_id(unit2take);

                acc_overall{m}{ar,p} = NaN(length(param.pseudopop), param.Repetition, 2, 2);
                shc_overall{m}{ar,p} = NaN(length(param.pseudopop), param.Repetition, 2, 2);
                % bias_curve: accumulated at the LARGEST valid N for this area/monkey.
                % Reset at each new valid N level; after the u-loop the accumulator
                % holds data from the highest N reached.
                bias_curve_sum = zeros(2, 2, n_delta);
                bias_curve_cnt = zeros(2, 2);
                bias_curve_N   = NaN;   % will record the N actually used
                peak_loc_reps  = NaN(2, 2, param.Repetition);   % init; reset at each valid N

                % Pseudopop loop
                for u = 1 : length(param.pseudopop)

                    if length(unit2take) <= param.pseudopop(u), continue; end

                    % New valid N level: reset accumulator so only this N is kept
                    bias_curve_sum = zeros(2, 2, n_delta);
                    bias_curve_cnt = zeros(2, 2);
                    bias_curve_N   = param.pseudopop(u);
                    peak_loc_reps  = NaN(2, 2, param.Repetition);   % reset for this N level

                      disp(['crossdecode: p=' num2str(p) ' m=' num2str(m) ' ' ...
                          area2test_name{ar} ' N=' num2str(param.pseudopop(u))])

                    for pp = 1 : param.Repetition

                        unit_sub = randperm(length(unit2take), param.pseudopop(u));

                        data    = fr_clean2( ...
                            ismember(behav_clean2(:,2), unit_id(unit_sub)), :);
                        factors = behav_clean2( ...
                            ismember(behav_clean2(:,2), unit_id(unit_sub)), :);

                        [XX, YY, ~] = utils_createpseudopop(data, factors, [1 2], ...
                            'minTr', param.minTr(p), 'perm', 1, 'pop', false);

                        nFolds  = 10;
                        nTrials = size(XX{1}, 1);
                        cv      = cvpartition(nTrials, 'KFold', nFolds);

                        % Compute all 4 cells of the 2x2 matrix
                        for trainState = 1 : 2
                            for testState = 1 : 2

                                % Collect normalised decision scores across all folds.
                                % Each fold contributes (nTestTrials) scores in
                                % normalised units where training threshold = 0 and
                                % training class means are at ~= +/-1.
                                scores_pooled = [];
                                labels_pooled = [];

                                for f = 1 : nFolds
                                    trainInds = cv.training(f);
                                    testInds  = cv.test(f);

                                    X_tr_raw = XX{trainState}(trainInds, :);
                                    y_tr     = YY{trainState}(trainInds);
                                    X_te_raw = XX{testState} (testInds,  :);
                                    y_te     = YY{testState} (testInds);

                                    % Mean-centre on training set (no PCA)
                                    mu_tr = mean(X_tr_raw, 1);
                                    S_tr  = X_tr_raw - mu_tr;
                                    S_te  = X_te_raw - mu_tr;

                                    classes = unique(y_tr);
                                    if numel(classes) ~= 2, continue; end
                                    mask1 = y_tr == classes(1);
                                    mask2 = y_tr == classes(2);
                                    n1 = sum(mask1);
                                    n2 = sum(mask2);
                                    if n1 < 2 || n2 < 2, continue; end

                                    % Regularized diagonal LDA (full space, no PCA)
                                    % Pseudopopulations have zero noise correlations by
                                    % construction, so the true Sw is diagonal.  We use
                                    % diagonal LDA directly in the full N-dimensional space
                                    % with L2 regularization (smooth variance floor):
                                    %   w_i = (mu1_i - mu2_i) / (pv_i + alpha*mean(pv))
                                    % alpha = 0.01 (1% of mean pooled variance).
                                    % This prevents near-silent neurons from dominating
                                    % without distorting neurons with real signal.
                                    mu1 = mean(S_tr(mask1, :), 1);   % 1xN
                                    mu2 = mean(S_tr(mask2, :), 1);   % 1xN

                                    pv = ((n1-1)*var(S_tr(mask1,:), 0, 1) + ...
                                          (n2-1)*var(S_tr(mask2,:), 0, 1)) / (n1+n2-2);

                                    alpha_reg = 0.01;
                                    pv_reg    = pv + alpha_reg * mean(pv);

                                    w = ((mu1 - mu2) ./ pv_reg)';  % Nx1

                                    % Raw discriminant scores on train and test sets
                                    d_tr = S_tr * w;
                                    d_te = S_te * w;

                                    % Normalise scores before pooling
                                    % Map so that training class means -> +/-1.
                                    % Training threshold (midpoint of class means) -> 0.
                                    % This makes delta = 0 equivalent to CCGP regardless
                                    % of fold-to-fold variation in w magnitude.
                                    d_mu1_tr = mean(d_tr(mask1));
                                    d_mu2_tr = mean(d_tr(mask2));
                                    half_sep = 0.5 * (d_mu1_tr - d_mu2_tr);
                                    thresh0  = 0.5 * (d_mu1_tr + d_mu2_tr);

                                    if abs(half_sep) < 1e-10, continue; end  % degenerate fold

                                    d_te_norm = (d_te - thresh0) / half_sep;

                                    % Label convention: class 1 has positive scores
                                    % (since w points toward class 1 and we divided
                                    % by half_sep which is positive when class 1 mean
                                    % is above class 2 mean on the axis).
                                    y_binary = double(y_te == classes(1));  % 1 for class 1, 0 for class 2

                                    scores_pooled = [scores_pooled; d_te_norm(:)];
                                    labels_pooled = [labels_pooled; y_binary(:)];

                                end  % folds

                                if isempty(scores_pooled), continue; end

                                % CCGP: accuracy at delta = 0 (training threshold)
                                pred_ccgp = double(scores_pooled > 0);
                                acc_overall{m}{ar,p}(u, pp, trainState, testState) = ...
                                    mean(pred_ccgp == labels_pooled);

                                % shCCGP: max accuracy over threshold grid
                                % Scan delta over delta_grid; for each delta, predict class 1
                                % when score > delta.  shCCGP = max accuracy over grid.
                                % With minTr = 20 per class and nFolds = 10, pooled
                                % set has 20 obs per class (each appears in exactly
                                % one test fold) - sufficient for a stable scan.
                                acc_grid = zeros(n_delta, 1);
                                for di = 1 : n_delta
                                    pred_di = double(scores_pooled > delta_grid(di));
                                    acc_grid(di) = mean(pred_di == labels_pooled);
                                end
                                shc_val = max(acc_grid);
                                shc_overall{m}{ar,p}(u, pp, trainState, testState) = shc_val;
                                % Use mean delta of ALL tied peak positions to avoid
                                % MATLAB first-occurrence bias (max() returns leftmost
                                % tie, which drags delta* toward -3 for flat curves).
                                tied_di = find(acc_grid == shc_val);
                                peak_loc_reps(trainState, testState, pp) = mean(delta_grid(tied_di));

                                % Accumulate bias curve (reset to this N level at the start of the loop)
                                bias_curve_sum(trainState, testState, :) = ...
                                    bias_curve_sum(trainState, testState, :) + ...
                                    reshape(acc_grid, 1, 1, n_delta);
                                bias_curve_cnt(trainState, testState) = ...
                                    bias_curve_cnt(trainState, testState) + 1;

                            end  % testState
                        end  % trainState

                    end  % pp (repetitions)
                end  % u (pseudopop sizes)

                % Average bias curve over repetitions at the highest valid N
                bc = NaN(2, 2, n_delta);
                for trnS = 1:2
                    for tstS = 1:2
                        if bias_curve_cnt(trnS, tstS) > 0
                            bc(trnS, tstS, :) = bias_curve_sum(trnS, tstS, :) / ...
                                                bias_curve_cnt(trnS, tstS);
                        end
                    end
                end
                bias_curve{m}{ar,p}          = bc;
                bias_curve_N_used{m}{ar,p}   = bias_curve_N;   % record N for figure title
                peak_loc_reps_final{m}{ar,p} = peak_loc_reps;   % per-rep peak delta* at max valid N

            end  % ar (areas)
        end  % m (monkeys)
    end  % p (parameters)

    % Improvement = shCCGP - CCGP  (element-wise, same indexing)
    imp_overall = cell(length(mks), 1);
    for mk_idx = 1 : length(mks)
        imp_overall{mk_idx} = cell(nAreas, length(param.param2decode));
        for ar = 1 : nAreas
            for p = 1 : length(param.param2decode)
                if ~isempty(acc_overall{mk_idx}{ar,p})
                    imp_overall{mk_idx}{ar,p} = ...
                        shc_overall{mk_idx}{ar,p} - acc_overall{mk_idx}{ar,p};
                end
            end
        end
    end

    save(savefile, 'acc_overall', 'shc_overall', 'imp_overall', 'bias_curve', ...
         'bias_curve_N_used', 'peak_loc_reps_final', 'param', 'area2test_name', ...
         'N_eval_global', 'delta_grid', '-v7.3');
    fprintf('Saved: %s\n', savefile);

else
    disp('Loading existing shCCGP 2x2 results...')
    load(savefile)
    param.param2decode = {'chosenflavor_2AFC' 'chosenside_2AFC'};
    if ~exist('bias_curve_N_used', 'var')
        bias_curve_N_used = [];   % field missing in earlier saves, default to empty
    end
    if ~exist('peak_loc_reps_final', 'var')
        peak_loc_reps_final = [];  % peak location data absent in earlier saves, default to empty
    end
end


%% ** Peak location statistics *****************************************************
%
% For each area / monkey / parameter / cell (trainState x testState):
%   Mean and SEM of peak delta* across repetitions at the largest valid N.
%   Wilcoxon signed-rank: H0: median(delta*) = 0  (training threshold is optimal).
%   Raw p-values stored; FDR correction applied jointly across all areas/monkeys
%   in the figure section (36 tests per parameter).
%
% Interpretation:
%   delta*_CU and delta*_UC FDR-significant with opposite signs -> H2.
%   delta*_CU ~= delta*_UC ~= 0                                 -> H3 / no signal.

pk_stat = cell(length(mks), 1);
for mk_idx = 1 : length(mks)
    pk_stat{mk_idx} = cell(nAreas, length(param.param2decode));
end

if ~isempty(peak_loc_reps_final)
    for p_pk = 1 : length(param.param2decode)
        for m = 1 : length(mks)
            for ar = 1 : nAreas
                pklr = peak_loc_reps_final{m}{ar, p_pk};  % 2x2xRepetition or NaNs
                if isempty(pklr), continue; end
                st.mean_pk = NaN(2, 2);
                st.sem_pk  = NaN(2, 2);
                st.pval_pk = NaN(2, 2);   % Wilcoxon signed-rank: H0 median(delta*)=0
                for trnS = 1 : 2
                    for tstS = 1 : 2
                        v = squeeze(pklr(trnS, tstS, :));
                        v = v(~isnan(v));
                        if numel(v) < 2, continue; end
                        st.mean_pk(trnS, tstS) = mean(v);
                        st.sem_pk(trnS, tstS)  = std(v) / sqrt(numel(v));
                        st.pval_pk(trnS, tstS) = signrank(v, 0);
                    end
                end
                pk_stat{m}{ar, p_pk} = st;
            end
        end
    end
end

%% ** LME statistics *******************************************************
%
% Metrics 1-3: full 2x2 LME - Perf ~ TrainState * TestState * Area * NbNeuronsSat + (1|Monkey)
%   (1) acc_overall -> CCGP 2x2  -> generalization = CCGP above chance
%   (2) shc_overall -> shCCGP 2x2 -> generalization = shCCGP above chance
%   (3) imp_overall -> Improvement 2x2 -> generalization = H2 test; geometry specificity = cross-specificity
%
% All evaluated at N = N_eval_global.

diary off;
fid_log = fopen([report_dir 'output_003.txt'], 'w');
log_cleanup = onCleanup(@() fclose(fid_log));
utils_diary(fid_log, '\n============================================================\n');
utils_diary(fid_log, 'Report generated: %s\n', datestr(now));
utils_diary(fid_log, 'N_eval = %d  (log2 = %.3f)\n', N_eval_global, log2(N_eval_global));

LogN_common = log2(N_eval_global);
LogN_vec    = log2(param.pseudopop);

perf_labels  = {'CCGP (accuracy, fixed threshold)', ...
                'shCCGP (accuracy, optimal threshold)', ...
                'Improvement (shCCGP - CCGP)'};
perf_cells   = {acc_overall, shc_overall, imp_overall};
perf_tags    = {'ccgp', 'shccgp', 'improv'};
metric_idx_stats = 1:(1 + 2*double(plot_shCCGP));

all_contrasts  = cell(length(param.param2decode), length(perf_cells));
lme_ccgp_info  = cell(length(param.param2decode), 1);   % saved for posthoc area tests

for p = 1 : length(param.param2decode)

    for metric = metric_idx_stats

        perf_now = perf_cells{metric};

        utils_diary(fid_log, '\n');
        utils_diary(fid_log, '%s\n', repmat('=', 1, 72));
        utils_diary(fid_log, '  %s  -  %s\n', param.param2decode{p}, perf_labels{metric});
        utils_diary(fid_log, '%s\n', repmat('=', 1, 72));

        if metric < 4
        % Build long table for LME fits
        tab_Perf         = [];
        tab_TrainState   = {};
        tab_TestState    = {};
        tab_Area         = {};
        tab_Monkey       = {};
        tab_NbNeuronsSat = [];

        for mk = 1 : length(mks)
            for ar = 1 : nAreas
                if isempty(perf_now{mk}) || isempty(perf_now{mk}{ar,p}), continue; end
                d = perf_now{mk}{ar,p};  % [n_psz x n_rep x 2 x 2]

                for ui = 1 : length(param.pseudopop)
                    if ui > size(d,1), break; end
                    for trnS = 1 : 2
                        for tstS = 1 : 2
                            v = squeeze(d(ui, :, trnS, tstS));
                            v = mean(v(~isnan(v)));   % average over repetitions -> one row per (area,mk,psz,states)
                            if isnan(v), continue; end

                            tab_Perf         = [tab_Perf;         v];
                            tab_TrainState   = [tab_TrainState;   state_labels(trnS)];
                            tab_TestState    = [tab_TestState;    state_labels(tstS)];
                            tab_Area         = [tab_Area;         area2test_name(ar)];
                            tab_Monkey       = [tab_Monkey;       mks(mk)];
                            tab_NbNeuronsSat = [tab_NbNeuronsSat; LogN_vec(ui)];
                        end
                    end
                end
            end
        end

        tab = table(tab_Perf, ...
                    categorical(tab_TrainState,  state_labels), ...
                    categorical(tab_TestState,   state_labels), ...
                    categorical(tab_Area,        area2test_name), ...
                    categorical(tab_Monkey,      {'M','X'}), ...
                    tab_NbNeuronsSat, ...
                    'VariableNames', {'Perf','TrainState','TestState','Area','Monkey','NbNeuronsSat'});
        utils_diary(fid_log, 'Fitting LME: Perf ~ TrainState * TestState * Area * NbNeuronsSat + (1|Monkey) ...\n');
        lme = fitlme(tab, 'Perf ~ 1 + TrainState * TestState * Area * NbNeuronsSat + (1|Monkey)');
        % disp(lme.Coefficients);
        utils_diary(fid_log, '%s', evalc('anova(lme)'));
        utils_diary(fid_log, '%s', evalc('disp(lme.Rsquared)'));
        % Conditional R^2 (fixed + random effects, Nakagawa & Schielzeth 2013)
        r2_res = sum(residuals(lme).^2);
        r2_tot = sum((tab.Perf - mean(tab.Perf)).^2);
        r2_conditional = 1 - (r2_res / r2_tot);
        utils_diary(fid_log, 'Conditional R^2 (fixed + random effects): %.4f\n', r2_conditional);

        coef_names = lme.Coefficients.Name;
        coef_vals  = lme.Coefficients.Estimate;
        CovMat     = lme.CoefficientCovariance;
        dfe        = lme.DFE;

        % Detect reference levels from which level is absent in coef names
        is_area_main  = startsWith(coef_names, 'Area_')       & ~contains(coef_names, ':');
        is_train_main = startsWith(coef_names, 'TrainState_') & ~contains(coef_names, ':');
        is_test_main  = startsWith(coef_names, 'TestState_')  & ~contains(coef_names, ':');
        area_seen     = cellfun(@(s) s(6:end),  coef_names(is_area_main),  'UniformOutput', false);
        train_seen    = cellfun(@(s) s(12:end), coef_names(is_train_main), 'UniformOutput', false);
        test_seen     = cellfun(@(s) s(11:end), coef_names(is_test_main),  'UniformOutput', false);
        ref_area      = setdiff(area2test_name, area_seen);  ref_area  = ref_area{1};
        ref_train     = setdiff(state_labels,   train_seen); ref_train = ref_train{1};
        ref_test  = setdiff(state_labels, test_seen);  ref_test  = ref_test{1};
        utils_diary(fid_log, 'Reference: Area=%s, Train=%s, Test=%s\n', ref_area, ref_train, ref_test);

        % Contrasts per area
        geomspec_est = NaN(nAreas, 1);  geomspec_se = NaN(nAreas, 1);
        gainmod_est = NaN(nAreas, 1);  gainmod_se = NaN(nAreas, 1);
        genrz_est = NaN(nAreas, 1);  genrz_se = NaN(nAreas, 1);
        mu = NaN(nAreas, 4);         mu_se = NaN(nAreas, 4);

        for ar = 1 : nAreas
            ar_lbl = area2test_name{ar};

            x11 = build_x(coef_names, 'Chosen',   'Chosen',   ar_lbl, LogN_common, ref_train, ref_test, ref_area);
            x12 = build_x(coef_names, 'Chosen',   'Unchosen', ar_lbl, LogN_common, ref_train, ref_test, ref_area);
            x21 = build_x(coef_names, 'Unchosen', 'Chosen',   ar_lbl, LogN_common, ref_train, ref_test, ref_area);
            x22 = build_x(coef_names, 'Unchosen', 'Unchosen', ar_lbl, LogN_common, ref_train, ref_test, ref_area);

            xvec = {x11, x12, x21, x22};
            for cc = 1 : 4
                mu(ar,cc)    = xvec{cc}' * coef_vals;
                mu_se(ar,cc) = sqrt(max(0, xvec{cc}' * CovMat * xvec{cc}));
            end

            % Geometry specificity = [(CC-CU) + (UU-UC)] / 2
            xB = ((x11 - x12) + (x22 - x21)) / 2;
            geomspec_est(ar) = xB' * coef_vals;
            geomspec_se(ar)  = sqrt(max(0, xB' * CovMat * xB));

            % Gain modulation = [(CC+UC) - (CU+UU)] / 2
            xC = ((x11 + x21) - (x12 + x22)) / 2;
            gainmod_est(ar) = xC' * coef_vals;
            gainmod_se(ar)  = sqrt(max(0, xC' * CovMat * xC));

            % Generalization:
            if metric < 3
                xD = (x12 + x21) / 2;
                genrz_est(ar) = xD' * coef_vals - 0.5;
            else  % metric == 3: improvement off-diag - on-diag
                xD = (x12 + x21)/2 - (x11 + x22)/2;
                genrz_est(ar) = xD' * coef_vals;
            end
            genrz_se(ar) = sqrt(max(0, xD' * CovMat * xD));
        end

        t_geomspec = geomspec_est ./ geomspec_se;
        t_gainmod = gainmod_est ./ gainmod_se;
        t_genrz = genrz_est ./ genrz_se;
        p_geomspec = 2 * tcdf(-abs(t_geomspec), dfe);
        p_gainmod = 2 * tcdf(-abs(t_gainmod), dfe);
        p_genrz = 2 * tcdf(-abs(t_genrz), dfe);
        [~,~,p_geomspec_fdr] = utils_fdr_bh(p_geomspec);
        [~,~,p_gainmod_fdr] = utils_fdr_bh(p_gainmod);
        [~,~,p_genrz_fdr] = utils_fdr_bh(p_genrz);

        %  Display 
        utils_diary(fid_log, '\n--- Geometry specificity (within - cross) ---\n');
        tbl_geomspec = table(area2test_name', geomspec_est, geomspec_se, t_geomspec, p_geomspec, p_geomspec_fdr, ...
            'VariableNames', {'Area','Est','SE','t','p_raw','p_FDR'});
        utils_diary(fid_log, '%s', regexprep(evalc('disp(tbl_geomspec)'), '</?strong>', ''));

        utils_diary(fid_log, '\n--- Gain modulation (test-chosen - test-unchosen) ---\n');
        tbl_gainmod = table(area2test_name', gainmod_est, gainmod_se, t_gainmod, p_gainmod, p_gainmod_fdr, ...
            'VariableNames', {'Area','Est','SE','t','p_raw','p_FDR'});
        utils_diary(fid_log, '%s', regexprep(evalc('disp(tbl_gainmod)'), '</?strong>', ''));

        utils_diary(fid_log, '\n');
        if metric < 3
            utils_diary(fid_log, '--- Generalization: mean(off-diag) - 0.5 ---\n');
        else
            utils_diary(fid_log, '--- Generalization (H2 test): Improvement_off-diag - Improvement_on-diag ---\n');
            utils_diary(fid_log, '    Positive = threshold shift larger cross-state than within-state -> H2\n');
        end
        tbl_genrz = table(area2test_name', genrz_est, genrz_se, t_genrz, p_genrz, p_genrz_fdr, ...
            'VariableNames', {'Area','Est','SE','t','p_raw','p_FDR'});
        utils_diary(fid_log, '%s', regexprep(evalc('disp(tbl_genrz)'), '</?strong>', ''));

        utils_diary(fid_log, '\n--- Marginal means at N=%d (log2=%.3f) ---\n', N_eval_global, LogN_common);
        tbl_mu = table(area2test_name', mu(:,1), mu(:,2), mu(:,3), mu(:,4), ...
            'VariableNames', {'Area','CC','CU','UC','UU'});
        utils_diary(fid_log, '%s', regexprep(evalc('disp(tbl_mu)'), '</?strong>', ''));

        % Save contrasts for figures
        all_contrasts{p, metric}.geomspec_est = geomspec_est; all_contrasts{p, metric}.geomspec_se = geomspec_se; all_contrasts{p, metric}.p_geomspec_fdr = p_geomspec_fdr;
        all_contrasts{p, metric}.gainmod_est = gainmod_est; all_contrasts{p, metric}.gainmod_se = gainmod_se; all_contrasts{p, metric}.p_gainmod_fdr = p_gainmod_fdr;
        all_contrasts{p, metric}.genrz_est = genrz_est; all_contrasts{p, metric}.genrz_se = genrz_se; all_contrasts{p, metric}.p_genrz_fdr = p_genrz_fdr;
        all_contrasts{p, metric}.mu = mu; all_contrasts{p, metric}.mu_se = mu_se;

        % Robustness: evaluate geometry specificity, gain modulation, and generalization at multiple N values
        %
        % The LME is fit to all N, so it can be queried at any log2(N) in the
        % training range.  We evaluate at N = 50, 100, 200, 350, 500 to show
        % that the reported contrast estimates are stable across N (not an
        % artefact of the specific N_eval_global = 200 choice).
        N_evals_rob = [50 100 200 350 500];
        N_evals_rob = N_evals_rob(N_evals_rob <= param.pseudopop(end));
        n_N_rob     = length(N_evals_rob);

        rob_geomspec_est = NaN(nAreas, n_N_rob);  rob_geomspec_se = NaN(nAreas, n_N_rob);
        rob_gainmod_est = NaN(nAreas, n_N_rob);  rob_gainmod_se = NaN(nAreas, n_N_rob);
        rob_genrz_est = NaN(nAreas, n_N_rob);  rob_genrz_se = NaN(nAreas, n_N_rob);

        for n_idx = 1 : n_N_rob
            LogN_rob = log2(N_evals_rob(n_idx));
            for ar_rob = 1 : nAreas
                ar_lbl = area2test_name{ar_rob};
                x12 = build_x(coef_names, 'Chosen',   'Unchosen', ar_lbl, LogN_rob, ref_train, ref_test, ref_area);
                x21 = build_x(coef_names, 'Unchosen', 'Chosen',   ar_lbl, LogN_rob, ref_train, ref_test, ref_area);
                x11 = build_x(coef_names, 'Chosen',   'Chosen',   ar_lbl, LogN_rob, ref_train, ref_test, ref_area);
                x22 = build_x(coef_names, 'Unchosen', 'Unchosen', ar_lbl, LogN_rob, ref_train, ref_test, ref_area);

                xB = ((x11 - x12) + (x22 - x21)) / 2;
                rob_geomspec_est(ar_rob, n_idx) = xB' * coef_vals;
                rob_geomspec_se(ar_rob, n_idx)  = sqrt(max(0, xB' * CovMat * xB));

                xC = ((x11 + x21) - (x12 + x22)) / 2;
                rob_gainmod_est(ar_rob, n_idx) = xC' * coef_vals;
                rob_gainmod_se(ar_rob, n_idx)  = sqrt(max(0, xC' * CovMat * xC));

                if metric < 3
                    xD = (x12 + x21) / 2;
                    rob_genrz_est(ar_rob, n_idx) = xD' * coef_vals - 0.5;
                else
                    xD = (x12 + x21)/2 - (x11 + x22)/2;
                    rob_genrz_est(ar_rob, n_idx) = xD' * coef_vals;
                end
                rob_genrz_se(ar_rob, n_idx) = sqrt(max(0, xD' * CovMat * xD));
            end
        end

        all_contrasts{p, metric}.N_evals_rob   = N_evals_rob;
        all_contrasts{p, metric}.rob_geomspec_est = rob_geomspec_est;  all_contrasts{p, metric}.rob_geomspec_se = rob_geomspec_se;
        all_contrasts{p, metric}.rob_gainmod_est = rob_gainmod_est;  all_contrasts{p, metric}.rob_gainmod_se = rob_gainmod_se;
        all_contrasts{p, metric}.rob_genrz_est = rob_genrz_est;  all_contrasts{p, metric}.rob_genrz_se = rob_genrz_se;

        % Save CCGP LME info for pairwise area posthoc
        if metric == 1
            lme_ccgp_info{p}.coef_names = coef_names;
            lme_ccgp_info{p}.coef_vals  = coef_vals;
            lme_ccgp_info{p}.CovMat     = CovMat;
            lme_ccgp_info{p}.dfe        = dfe;
            lme_ccgp_info{p}.ref_train  = ref_train;
            lme_ccgp_info{p}.ref_test   = ref_test;
            lme_ccgp_info{p}.ref_area   = ref_area;
        end

        end  % if metric < 4

    end  % metric loop

end  % parameter loop

fclose(fid_log);

%% **************** Figures ***************************************************************
%
% For each parameter:
%   Fig A - CCGP 2x2 heatmaps + geometry specificity + gain modulation
%   Fig B - shCCGP 2x2 heatmaps + geometry specificity + gain modulation
%   Fig C - Improvement 2x2 heatmaps + generalization (H2 test) bar chart

for p = 1 : length(param.param2decode)

    param_name = strrep(param.param2decode{p}, '_2AFC', '');
    fig_ttl    = strrep(param.param2decode{p}, '_', ' ');

    metric_names = {'CCGP', 'shCCGP', 'Improvement'};
    metric_idx_fig = 1:(1 + 2*double(plot_shCCGP));

    for metric = metric_idx_fig

        ct = all_contrasts{p, metric};
        mu_now = ct.mu;

        % Heatmap + geometry specificity + gain modulation composite (for CCGP and shCCGP)
        if metric <= 2

            fig_main = figure('Position', [358   329   841   750], 'Color', 'w');

            % --- Layout for 9 heatmaps plus two contrast bar panels ---
            ml = 0.030; mr = 0.018; mt = 0.09; mb = 0.10;
            rw = 0.25; gap_mid = 0.035; bc_gap = 0.07;
            rx = 1 - mr - rw;
            rh = (1 - mt - mb - bc_gap) / 2;
            ax_gainmod_fig = axes('Parent', fig_main, 'Units', 'normalized', ...
                            'Position', [rx, mb + rh + bc_gap, rw, rh]);
            ax_geomspec_fig = axes('Parent', fig_main, 'Units', 'normalized', ...
                            'Position', [rx, mb, rw, rh]);

            lw = rx - gap_mid - ml;
            nhm_c = 3; nhm_r = 3;
            hm_gap_x = 0.020; hm_gap_y = 0.055; cb_h = 0.022; cb_gap = 0.032;
            hm_w = (lw  - hm_gap_x*(nhm_c-1)) / nhm_c;
            hm_h = (1 - mt - mb - cb_h - cb_gap - hm_gap_y*(nhm_r-1)) / nhm_r;

            % Set color axis: CCGP/shCCGP = auto, Improvement = [-0.15 0.15]
            if metric == 3
                caxis_lim = [-0.15 0.15];
            else
                caxis_lim = [min(0.45, min(mu_now(:))) max(mu_now(:)) + 0.01];
            end

            hm_order = [1 2 3 9 8 4 7 6 5];
            for hm_idx = 1 : nAreas
                ar = hm_order(hm_idx);
                ar_row = floor((hm_idx-1)/nhm_c);
                ar_col = mod(hm_idx-1, nhm_c);
                ax_x = ml + ar_col * (hm_w + hm_gap_x);
                ax_y = mb + cb_h + cb_gap + (nhm_r-1 - ar_row) * (hm_h + hm_gap_y);
                ax_hm = axes('Parent', fig_main, 'Units', 'normalized', ...
                             'Position', [ax_x, ax_y, hm_w, hm_h]);
                mat2x2 = [mu_now(ar,1) mu_now(ar,2); mu_now(ar,3) mu_now(ar,4)];
                imagesc(ax_hm, mat2x2, caxis_lim);
                colormap(ax_hm, flipud(hot(256)));
                is_bottom = (ar_row == nhm_r-1); is_left = (ar_col == 0);
                if is_bottom, set(ax_hm, 'XTick', [1 2], 'XTickLabel', {'C','U'}, 'FontSize', 11);
                else,          set(ax_hm, 'XTick', [1 2], 'XTickLabel', {}, 'FontSize', 11); end
                if is_left,   set(ax_hm, 'YTick', [1 2], 'YTickLabel', {'C','U'}, 'FontSize', 11);
                else,          set(ax_hm, 'YTick', [1 2], 'YTickLabel', {}, 'FontSize', 11); end
                set(ax_hm, 'TickLength', [0 0]);
                if is_bottom, xlabel(ax_hm, 'Test state', 'FontSize', 12); end
                if is_left,   ylabel(ax_hm, 'Train state', 'FontSize', 12); end
                title(ax_hm, area2test_name{ar}, 'Color', colorareas(ar,:)/255, ...
                      'FontWeight', 'bold', 'FontSize', 13);
                for rr = 1:2
                    for cc_idx = 1:2
                        v = mat2x2(rr, cc_idx);
                        text(ax_hm, cc_idx, rr, sprintf('%.3f', v), ...
                             'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
                             'Color', [1 1 1] * (v > mean(caxis_lim)));
                    end
                end
            end


            ax_cb = axes('Parent', fig_main, 'Units', 'normalized', ...
                         'Position', [ml, mb, lw, cb_h], 'Visible', 'off');
            colormap(ax_cb, flipud(hot(256))); clim(ax_cb, caxis_lim);
            cb = colorbar(ax_cb, 'Location', 'South', 'AxisLocation', 'out');
            cb.Position = [ml, mb, lw, cb_h];
            if metric == 3
                cb.Label.String = 'shCCGP shift';
            else
                cb.Label.String = 'Decoding accuracy';
            end
            cb.Label.FontSize = 12;
            cb.FontSize = 11; cb.TickLength = 0.02;

            annotation(fig_main, 'textbox', [0 0.93 1 0.06], ...
                'String', [metric_names{metric} ' - ' fig_ttl], ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
                'FontSize', 20, 'FontWeight', 'bold');

            % Geometry specificity
            [~, sidx_geomspec] = sort(ct.geomspec_est, 'descend');
            hold(ax_geomspec_fig, 'on');
            for kk = 1 : nAreas
                ii = sidx_geomspec(kk); col = colorareas(ii,:)/255;
                if ct.p_geomspec_fdr(ii) >= fdr_alpha, col = col*0.5+0.5; end
                bar(ax_geomspec_fig, kk, ct.geomspec_est(ii), 'FaceColor', col, 'EdgeColor', 'none');
                errorbar(ax_geomspec_fig, kk, ct.geomspec_est(ii), 1.96*ct.geomspec_se(ii), 'k.', 'LineWidth', 1.5);
                if ct.p_geomspec_fdr(ii) < fdr_alpha
                    text(ax_geomspec_fig, kk, ct.geomspec_est(ii)+sign(ct.geomspec_est(ii))*(1.96*ct.geomspec_se(ii)+0.003), ...
                         '*', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
                end
            end
            set(ax_geomspec_fig, 'XTick', 1:nAreas, 'XTickLabel', area2test_name(sidx_geomspec), 'FontSize', 11);
            xtickangle(ax_geomspec_fig, 45);
            ylabel(ax_geomspec_fig, 'LME estimate', 'FontSize', 12);
            title(ax_geomspec_fig, 'Geometry specificity', 'FontSize', 13, 'FontWeight', 'bold');
            yline(ax_geomspec_fig, 0, 'k--', 'LineWidth', 1); box(ax_geomspec_fig, 'off'); hold(ax_geomspec_fig, 'off');

            % Gain modulation
            [~, sidx_gainmod] = sort(ct.gainmod_est, 'descend');
            hold(ax_gainmod_fig, 'on');
            for kk = 1 : nAreas
                ii = sidx_gainmod(kk); col = colorareas(ii,:)/255;
                if ct.p_gainmod_fdr(ii) >= fdr_alpha, col = col*0.5+0.5; end
                bar(ax_gainmod_fig, kk, ct.gainmod_est(ii), 'FaceColor', col, 'EdgeColor', 'none');
                errorbar(ax_gainmod_fig, kk, ct.gainmod_est(ii), 1.96*ct.gainmod_se(ii), 'k.', 'LineWidth', 1.5);
                if ct.p_gainmod_fdr(ii) < fdr_alpha
                    text(ax_gainmod_fig, kk, ct.gainmod_est(ii)+sign(ct.gainmod_est(ii))*(1.96*ct.gainmod_se(ii)+0.003), ...
                         '*', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
                end
            end
            set(ax_gainmod_fig, 'XTick', 1:nAreas, 'XTickLabel', area2test_name(sidx_gainmod), 'FontSize', 11);
            xtickangle(ax_gainmod_fig, 45);
            ylabel(ax_gainmod_fig, 'LME estimate', 'FontSize', 12);
            title(ax_gainmod_fig, 'Gain modulation', 'FontSize', 13, 'FontWeight', 'bold');
            yline(ax_gainmod_fig, 0, 'k--', 'LineWidth', 1); box(ax_gainmod_fig, 'off'); hold(ax_gainmod_fig, 'off');

            drawnow;
            % Keep manuscript names for CCGP main panels while preserving
            % generic naming for all other metrics.
            if metric == 1 && strcmp(param_name, 'chosenflavor')
                fname_main = [report_dir 'Fig_3c_CCGP.png'];
            elseif metric == 1 && strcmp(param_name, 'chosenside')
                fname_main = [report_dir 'Fig_3d_CCGP.png'];
            else
                fname_main = [report_dir 'Fig_main_' metric_names{metric} '_' param_name '.png'];
            end
            saveas(fig_main, fname_main);
            fprintf('  Saved: %s\n', fname_main);

        elseif metric == 3
            % Improvement heatmaps + generalization bar
            fig_improv = figure('Position', [50 50 1600 700], 'Color', 'w');
            tl = tiledlayout(fig_improv, 3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
            title(tl, ['shCCGP - CCGP Improvement (H2 test) - ' fig_ttl], ...
                  'FontSize', 20, 'FontWeight', 'bold');

            % Generalization bar (H2 test): create FIRST so it claims column 4 (tiles 4, 8, 12)
            % before the heatmap loop fills the remaining tiles.
            ax_genrz_fig = nexttile(4, [3 1]);
            [~, sidx_genrz] = sort(ct.genrz_est, 'descend');
            hold(ax_genrz_fig, 'on');
            for kk = 1 : nAreas
                ii = sidx_genrz(kk); col = colorareas(ii,:)/255;
                if ct.p_genrz_fdr(ii) >= fdr_alpha, col = col*0.5+0.5; end
                bar(ax_genrz_fig, kk, ct.genrz_est(ii), 'FaceColor', col, 'EdgeColor', 'none');
                errorbar(ax_genrz_fig, kk, ct.genrz_est(ii), 1.96*ct.genrz_se(ii), 'k.', 'LineWidth', 1.5);
                if ct.p_genrz_fdr(ii) < fdr_alpha
                    text(ax_genrz_fig, kk, ct.genrz_est(ii)+sign(ct.genrz_est(ii))*(1.96*ct.genrz_se(ii)+0.002), ...
                         '*', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
                end
            end
            set(ax_genrz_fig, 'XTick', 1:nAreas, 'XTickLabel', area2test_name(sidx_genrz), 'FontSize', 12);
            xtickangle(ax_genrz_fig, 45);
            ylabel(ax_genrz_fig, 'Improvement off-diag - on-diag (LME)', 'FontSize', 13);
            title(ax_genrz_fig, 'Generalization of Improvement (H2 test)', 'FontSize', 14, 'FontWeight', 'bold');
            yline(ax_genrz_fig, 0, 'k--', 'LineWidth', 1.5); box(ax_genrz_fig, 'off'); hold(ax_genrz_fig, 'off');

            % 9 heatmaps in the first 3 columns (tiles 1-3, 5-7, 9-11)
            hm_tiles  = [1 2 3 5 6 7 9 10 11];
            hm_order = [1 2 3 9 8 4 7 6 5];

            clim_imp = [0 0.15]; % improvement: shCCGP - CCGP
            cmap_imp = flipud(gray(256));

            for hm_idx = 1 : nAreas
                ar = hm_order(hm_idx);
                ax_hm = nexttile(hm_tiles(hm_idx));
                mat2x2 = [mu_now(ar,1) mu_now(ar,2); mu_now(ar,3) mu_now(ar,4)];
                imagesc(ax_hm, mat2x2, clim_imp);
                colormap(ax_hm, cmap_imp);
                set(ax_hm, 'XTick', [1 2], 'XTickLabel', {'C','U'}, ...
                         'YTick', [1 2], 'YTickLabel', {'C','U'}, ...
                         'TickLength', [0 0], 'FontSize', 11);
                xlabel(ax_hm, 'Test'); ylabel(ax_hm, 'Train');
                title(ax_hm, area2test_name{ar}, 'Color', colorareas(ar,:)/255, ...
                      'FontWeight', 'bold', 'FontSize', 12);
                % Text color: black if abs(value) < 0.08, else white (centered at 0)
                for rr = 1:2
                    for cc_idx = 1:2
                        v = mat2x2(rr, cc_idx);
                        if abs(v) < 0.08
                            txt_col = [0 0 0];
                        else
                            txt_col = [1 1 1];
                        end
                        text(ax_hm, cc_idx, rr, sprintf('%.3f', v), ...
                             'HorizontalAlignment', 'center', 'FontSize', 9, ...
                             'FontWeight', 'bold', 'Color', txt_col);
                    end
                end
            end  % heatmap loop

            % Add colorbar below all heatmaps
            cb_ax = axes('Parent', fig_improv, 'Position', [0.08, 0.08, 0.62, 0.025], 'Visible', 'off');
            colormap(cb_ax, cmap_imp); caxis(cb_ax, clim_imp);
            cb = colorbar(cb_ax, 'Location', 'South', 'AxisLocation', 'out');
            cb.Position = [0.08, 0.08, 0.62, 0.025];
            cb.Label.String = 'shCCGP shift';
            cb.Label.FontSize = 12;
            cb.FontSize = 11; cb.TickLength = 0.02;

            drawnow;
            saveas(fig_improv, [report_dir 'Fig_' metric_names{metric} '_' param_name '.png']);
            fprintf('  Saved: Fig_%s_%s.png\n', metric_names{metric}, param_name);

        end  % metric figure branch

    end  % metric

    if plot_toggle_enabled(plot_shCCGP)
        % Summary: generalization(CCGP), generalization(shCCGP), generalization(Improvement), scatter geomspec_shCCGP vs geomspec_CCGP
        % Panel 4: scatter of LME-estimated geomspec_shCCGP (y) vs geomspec_CCGP (x) per area.
        % No division -> no instability.  Areas near origin = weak geometry specificity (correctly
        % uninterpretable).  Reference line y=x = no recovery; y=0 = complete recovery.
        % Iso-recovery lines at 25 / 50 / 75 % to guide interpretation of points away diagonal
        fig_sum = figure('Position', [50 50 2400 650], 'Color', 'w');
        tl_sum = tiledlayout(fig_sum, 1, 4, 'TileSpacing', 'loose', 'Padding', 'compact');
        title(tl_sum, ['Cross-state generalization - ' fig_ttl], 'FontSize', 26, 'FontWeight', 'bold');

        % Panels 1-3: Generalization for each metric
        panel_labels = {'CCGP - chance', 'shCCGP - chance', 'Improvement (H2 test)'};
        for metric = 1 : 3
            ct = all_contrasts{p, metric};
            ax_s = nexttile(metric);
            [~, sidx_genrz] = sort(ct.genrz_est, 'descend');
            hold(ax_s, 'on');
            for kk = 1 : nAreas
                ii = sidx_genrz(kk); col = colorareas(ii,:)/255;
                if ct.p_genrz_fdr(ii) >= fdr_alpha, col = col*0.5+0.5; end
                bar(ax_s, kk, ct.genrz_est(ii), 'FaceColor', col, 'EdgeColor', 'none');
                errorbar(ax_s, kk, ct.genrz_est(ii), 1.96*ct.genrz_se(ii), 'k.', 'LineWidth', 1.5);
                if ct.p_genrz_fdr(ii) < fdr_alpha
                    text(ax_s, kk, ct.genrz_est(ii)+sign(ct.genrz_est(ii))*(1.96*ct.genrz_se(ii)+0.003), ...
                         '*', 'HorizontalAlignment', 'center', 'FontSize', 20, 'FontWeight', 'bold');
                end
            end
            set(ax_s, 'XTick', 1:nAreas, 'XTickLabel', area2test_name(sidx_genrz), 'FontSize', 16);
            xtickangle(ax_s, 45);
            ylabel(ax_s, panel_labels{metric}, 'FontSize', 17);
            title(ax_s, [metric_names{metric} ' - Generalization'], 'FontSize', 18, 'FontWeight', 'bold');
            yline(ax_s, 0, 'k--', 'LineWidth', 1.2); box(ax_s, 'off'); hold(ax_s, 'off');
        end

        % Panel 4: geomspec_shCCGP vs geomspec_CCGP scatter - geometry specificity before/after optimal shift
        % x = geomspec_CCGP  (specificity at fixed threshold = baseline)
        % y = geomspec_shCCGP (specificity at optimal threshold = after shift)
        % y = x     -> no recovery (pure rotation / scaling)
        % y = 0     -> complete recovery (pure bias shift)
        % y = (1-r)*x for r = 0.25 / 0.50 / 0.75 = iso-recovery lines
        ct_ccgp   = all_contrasts{p, 1};   % metric 1: CCGP
        ct_shccgp = all_contrasts{p, 2};   % metric 2: shCCGP
        ct_imp    = all_contrasts{p, 3};   % metric 3: Improvement
        % Significance of deviation from diagonal: geomspec_Improvement = geomspec_shCCGP - geomspec_CCGP
        % Negative & FDR-significant -> point lies below y=x -> geometry specificity reduced -> recovery
        geomspec_ccgp   = ct_ccgp.geomspec_est;
        geomspec_shccgp = ct_shccgp.geomspec_est;
        se_geomspec_ccgp  = ct_ccgp.geomspec_se;
        se_geomspec_shccgp = ct_shccgp.geomspec_se;
        p_diag_fdr = ct_imp.p_geomspec_fdr;   % FDR p for geomspec_Improvement ≠ 0 (= deviation from diagonal)

        ax_s4 = nexttile(4);
        hold(ax_s4, 'on');
        set(ax_s4, 'FontSize', 15);

        % Axis range: include error bars so nothing is clipped
        ax_max = max([geomspec_ccgp + 1.96*se_geomspec_ccgp; geomspec_shccgp + 1.96*se_geomspec_shccgp; 0.02]) * 1.18;
        ax_max = max(ax_max, 0.02);
        ax_min = min([geomspec_ccgp - 1.96*se_geomspec_ccgp; geomspec_shccgp - 1.96*se_geomspec_shccgp; -0.005]) * 1.1;

        % Reference lines
        plot(ax_s4, [0 ax_max], [0 ax_max],   'k-',  'LineWidth', 1.2, 'HandleVisibility', 'off');  % y=x (no recovery)
        plot(ax_s4, [0 ax_max], [0 0],        'k--', 'LineWidth', 1.0, 'HandleVisibility', 'off');  % y=0 (full recovery)
        % Iso-recovery lines: 25%, 50%, 75%
        rec_levels  = [0.25 0.50 0.75];
        rec_colors  = {[0.75 0.75 0.75], [0.55 0.55 0.55], [0.35 0.35 0.35]};
        rec_labels  = {'25%', '50%', '75%'};
        for ri = 1 : 3
            yl = (1 - rec_levels(ri)) * ax_max;
            plot(ax_s4, [0 ax_max], [0 yl], '--', 'Color', rec_colors{ri}, 'LineWidth', 0.8, ...
                 'HandleVisibility', 'off');
            text(ax_s4, ax_max * 0.98, yl * 0.98, rec_labels{ri}, ...
                 'Color', rec_colors{ri}, 'FontSize', 14, 'HorizontalAlignment', 'right');
        end
        text(ax_s4, ax_max * 0.98, ax_max * 0.98, 'no recovery', ...
             'FontSize', 14, 'HorizontalAlignment', 'right', 'Color', [0 0 0]);
        text(ax_s4, ax_max * 0.98, ax_max * 0.015, 'full recovery', ...
             'FontSize', 14, 'HorizontalAlignment', 'right', 'Color', [0 0 0]);

        % Area scatter points with error bars
        % Colour: saturated if geomspec_CCGP is significant (has real geometry specificity).
        % Star (*): geomspec_Improvement FDR-significant = point deviates from diagonal = recovery.
        for ar = 1 : nAreas
            col = colorareas(ar,:) / 255;
            if ct_ccgp.p_geomspec_fdr(ar) >= fdr_alpha, col = col*0.4 + 0.6; end
            errorbar(ax_s4, geomspec_ccgp(ar), geomspec_shccgp(ar), ...
                     1.96*se_geomspec_shccgp(ar), 1.96*se_geomspec_shccgp(ar), ...
                     1.96*se_geomspec_ccgp(ar),  1.96*se_geomspec_ccgp(ar), ...
                     'o', 'Color', col, 'MarkerFaceColor', col, ...
                     'MarkerSize', 8, 'LineWidth', 1.2, 'CapSize', 4);
            lbl = area2test_name{ar};
            if p_diag_fdr(ar) < fdr_alpha
                lbl = [lbl '*'];   % deviation from diagonal = significant recovery
            end
            text(ax_s4, geomspec_ccgp(ar), geomspec_shccgp(ar) + ax_max*0.03, lbl, ...
                 'Color', col, 'FontSize', 15, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end

        xlim(ax_s4, [ax_min  ax_max]);
        ylim(ax_s4, [ax_min  ax_max]);
        axis(ax_s4, 'square');
        xlabel(ax_s4, 'Geometry specificity (CCGP, fixed \delta)', 'FontSize', 16);
        ylabel(ax_s4, 'Geometry specificity (shCCGP, optimal \delta)', 'FontSize', 16);
        title(ax_s4, 'Geometry specificity: before vs after optimal shift', 'FontSize', 18, 'FontWeight', 'bold');
        box(ax_s4, 'off');  hold(ax_s4, 'off');

        drawnow;
        if strcmp(param_name, 'chosenflavor')
            fname_genrz = [report_dir 'Fig_S6b_generalization.png'];
        elseif strcmp(param_name, 'chosenside')
            fname_genrz = [report_dir 'Fig_S6c_generalization.png'];
        else
            fname_genrz = [report_dir 'Fig_genrz_summary_' param_name '.png'];
        end
        saveas(fig_sum, fname_genrz);
        fprintf('  Saved: %s\n', fname_genrz);

        % Recovery vs N: geomspec_CCGP and geomspec_shCCGP on same axes
        % Plots the two geometry-specificity estimates per area as a function of N.
        % Gap between solid (CCGP) and dashed (shCCGP) line = recovery.
        % No division -> no denominator noise.  Gap shaded in area colour.
        ct1_rob = all_contrasts{p, 1};
        ct2_rob = all_contrasts{p, 2};
        if isfield(ct1_rob, 'rob_geomspec_est') && isfield(ct2_rob, 'rob_geomspec_est')
            N_rob_vec    = ct1_rob.N_evals_rob;
            LogN_rob_vec = log2(N_rob_vec);

            bB1  = ct1_rob.rob_geomspec_est;   % [nAreas x n_N_r]  CCGP
            bB2  = ct2_rob.rob_geomspec_est;   % [nAreas x n_N_r]  shCCGP

            rob_tick_N_traj = [50 100 200 500];
            rob_tick_N_traj = rob_tick_N_traj(rob_tick_N_traj <= max(N_rob_vec));

            fig_traj = figure('Name', ['Geometry specificity vs N - ' fig_ttl], ...
                              'Position', [50 50 1100 700], 'Color', 'w');
            ax_tr = axes(fig_traj);
            hold(ax_tr, 'on');
            set(ax_tr, 'FontSize', 16);

            yline(ax_tr, 0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
            xline(ax_tr, log2(N_eval_global), 'Color', [0.4 0.4 0.4], ...
                  'LineStyle', ':', 'LineWidth', 2, 'HandleVisibility', 'off');

            for ar = 1 : nAreas
                col = colorareas(ar,:) / 255;
                v1 = bB1(ar,:);
                v2 = bB2(ar,:);

                % CCGP: solid line
                plot(ax_tr, LogN_rob_vec, v1, '-',  'Color', col, 'LineWidth', 2.5, ...
                     'DisplayName', area2test_name{ar});

                % shCCGP: dashed line
                plot(ax_tr, LogN_rob_vec, v2, '--', 'Color', col, 'LineWidth', 2.0, ...
                     'HandleVisibility', 'off');
            end

            set(ax_tr, 'XTick', log2(rob_tick_N_traj), ...
                       'XTickLabel', arrayfun(@num2str, rob_tick_N_traj, 'UniformOutput', false));
            xlim(ax_tr, [log2(min(N_rob_vec)*0.85)  log2(max(N_rob_vec)*1.1)]);
            xlabel(ax_tr, 'Pseudopop size  N', 'FontSize', 18);
            ylabel(ax_tr, 'Geometry specificity', 'FontSize', 18);
            title(ax_tr, ['Geom. specificity: CCGP (-) vs shCCGP (- -) per area  -  ' fig_ttl], ...
                  'FontSize', 18, 'FontWeight', 'bold');
            % Manual legend note
            text(ax_tr, log2(min(N_rob_vec)*0.87), 0, ...
                 '  solid = CCGP   dashed = shCCGP   gap = recovery', ...
                 'FontSize', 12, 'Color', [0.4 0.4 0.4], 'VerticalAlignment', 'bottom');
            legend(ax_tr, 'Location', 'best', 'FontSize', 13);
            box(ax_tr, 'off');  hold(ax_tr, 'off');

            drawnow;
            saveas(fig_traj, [report_dir 'Fig_geomspec_recovery_' param_name '.png']);
            fprintf('  Saved: Fig_geomspec_recovery_%s.png\n', param_name);
        end
    else
        ct = all_contrasts{p, 1};
        fig_genrz = figure('Name', ['CCGP Generalization - ' fig_ttl], ...
                           'Position', [80 240 450 450], 'Color', 'w');
        ax_genrz = axes(fig_genrz);
        [~, sidx_genrz] = sort(ct.genrz_est, 'descend');
        hold(ax_genrz, 'on');
        for kk = 1 : nAreas
            ii = sidx_genrz(kk);
            col = colorareas(ii,:) / 255;
            if ct.p_genrz_fdr(ii) >= fdr_alpha
                col = col * 0.5 + 0.5;
            end
            bar(ax_genrz, kk, ct.genrz_est(ii), 'FaceColor', col, 'EdgeColor', 'none');
            errorbar(ax_genrz, kk, ct.genrz_est(ii), 1.96 * ct.genrz_se(ii), 'k.', 'LineWidth', 1.5);
            if ct.p_genrz_fdr(ii) < fdr_alpha
                text(ax_genrz, kk, ct.genrz_est(ii) + sign(ct.genrz_est(ii)) * (1.96 * ct.genrz_se(ii) + 0.003), ...
                     '*', 'HorizontalAlignment', 'center', 'FontSize', 20, 'FontWeight', 'bold');
            end
        end
        set(ax_genrz, 'XTick', 1:nAreas, 'XTickLabel', area2test_name(sidx_genrz), 'FontSize', 14);
        xtickangle(ax_genrz, 45);
        ylabel(ax_genrz, 'CCGP - chance', 'FontSize', 16);
        title(ax_genrz, ['CCGP - Generalization - ' fig_ttl], 'FontSize', 18, 'FontWeight', 'bold');
        yline(ax_genrz, 0, 'k--', 'LineWidth', 1.2);
        box(ax_genrz, 'off');
        hold(ax_genrz, 'off');

        drawnow;
        if strcmp(param_name, 'chosenflavor')
            fname_genrz = [report_dir 'Fig_S6b_generalization.png'];
        elseif strcmp(param_name, 'chosenside')
            fname_genrz = [report_dir 'Fig_S6c_generalization.png'];
        else
            fname_genrz = [report_dir 'Fig_genrz_summary_' param_name '.png'];
        end
        saveas(fig_genrz, fname_genrz);
        fprintf('  Saved: %s\n', fname_genrz);
    end

end  % parameter loop

%% **************** Figure S5-style: accuracy-vs-bias curves per area ***************************************************************
%
% One figure per parameter x monkey.
% 3x3 grid of subplots (one per area), each showing 4 curves:
%   CC (within chosen), UU (within unchosen), CU (cross from chosen), UC (cross from unchosen)
% Vertical dashed line at delta = 0 = training threshold = CCGP.
% Peak of CU and UC curves = shCCGP.
% A displaced peak (from delta=0) indicates H2 (midpoint shift between states).
% A flat curve near 0.5 indicates no generalisation (H3 or no signal).

cell_names  = {'CC (within chosen)', 'CU (cross)', 'UC (cross)', 'UU (within unchosen)'};
cell_colors = {[0.2 0.4 0.8], [0.9 0.4 0.1], [0.9 0.4 0.1], [0.2 0.4 0.8]};
cell_styles = {'-', '-', '--', '--'};
% trainState x testState index pairs: CC=1,1  CU=1,2  UC=2,1  UU=2,2
cell_idx    = {[1 1], [1 2], [2 1], [2 2]};

if plot_toggle_enabled(plot_shCCGP)
    for p = 1 : length(param.param2decode)
        param_name = strrep(param.param2decode{p}, '_2AFC', '');

        for m = 1 : length(mks)

        fig_s5 = figure('Name', ['Bias curves - ' param_name ' - Monkey ' mks{m}], ...
                        'Position', [50 50 1400 1100], 'Color', 'w');
        tl_s5  = tiledlayout(fig_s5, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        title(tl_s5, ['Accuracy vs. bias offset - ' param_name ', Monkey ' mks{m} ...
                       ' (each area at its max valid N)'], ...
              'FontSize', 18, 'FontWeight', 'bold');

        hm_order = [1 2 3 9 8 4 7 6 5];   % same layout as heatmap figures
        for hm_idx = 1 : nAreas
            ar = hm_order(hm_idx);
            ax = nexttile(hm_idx);
            hold(ax, 'on');

            bc = bias_curve{m}{ar, p};   % 2 x 2 x n_delta, or empty

            % Get the N actually used for this area/monkey/param
            if ~isempty(bias_curve_N_used) && ~isempty(bias_curve_N_used{m}{ar,p})
                N_used_str = ['N=' num2str(bias_curve_N_used{m}{ar,p})];
            else
                N_used_str = '';
            end

            if isempty(bc) || all(isnan(bc(:)))
                text(0.5, 0.5, 'no data', 'Units', 'normalized', ...
                     'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', [0.6 0.6 0.6]);
            else
                for ci = 1 : 4
                    trnS = cell_idx{ci}(1);
                    tstS = cell_idx{ci}(2);
                    curve = squeeze(bc(trnS, tstS, :));
                    if all(isnan(curve)), continue; end

                    plot(ax, delta_grid, curve, cell_styles{ci}, ...
                         'Color', cell_colors{ci}, 'LineWidth', 2.0, ...
                         'DisplayName', cell_names{ci});

                    % Mark peak of cross-state curves with 95% CI on peak location
                    if trnS ~= tstS
                        [pk_val, pk_idx] = max(curve);
                        plot(ax, delta_grid(pk_idx), pk_val, 'o', ...
                             'Color', cell_colors{ci}, 'MarkerFaceColor', cell_colors{ci}, ...
                             'MarkerSize', 7, 'HandleVisibility', 'off');
                        % Horizontal error bar showing 95% CI of peak location
                        if ~isempty(pk_stat{m}{ar,p})
                            pk_sem_v = pk_stat{m}{ar,p}.sem_pk(trnS, tstS);
                            if ~isnan(pk_sem_v) && pk_sem_v > 0
                                errorbar(ax, delta_grid(pk_idx), pk_val, ...
                                         0, 0, 1.96*pk_sem_v, 1.96*pk_sem_v, ...
                                         'Color', cell_colors{ci}, 'LineWidth', 1.5, ...
                                         'CapSize', 5, 'HandleVisibility', 'off');
                            end
                            if ~isnan(pk_stat{m}{ar,p}.pval_pk(trnS, tstS)) && ...
                               pk_stat{m}{ar,p}.pval_pk(trnS, tstS) < 0.05
                                text(ax, delta_grid(pk_idx), pk_val + 0.03, '*', ...
                                     'Color', cell_colors{ci}, 'FontSize', 12, ...
                                     'HorizontalAlignment', 'center', ...
                                     'HandleVisibility', 'off');
                            end
                        end
                    end
                end
            end

            xline(ax, 0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
            yline(ax, 0.5, 'Color', [0.7 0.7 0.7], 'LineStyle', ':', ...
                  'LineWidth', 1.0, 'HandleVisibility', 'off');
            xlim(ax, [delta_grid(1) delta_grid(end)]);
            ylim(ax, [0.4 1.02]);
            xlabel(ax, 'Bias offset (\delta)', 'FontSize', 11);
            ylabel(ax, 'Accuracy', 'FontSize', 11);
            % Area name + N used in subtitle
            title(ax, [area2test_name{ar} '  ' N_used_str], ...
                  'Color', colorareas(ar,:)/255, 'FontWeight', 'bold', 'FontSize', 13);
            set(ax, 'FontSize', 10); box(ax, 'off');

            if hm_idx == 1
                legend(ax, 'Location', 'southeast', 'FontSize', 9);
            end

            hold(ax, 'off');
        end

            drawnow;
            fname = [report_dir 'Fig_bias_curves_' param_name '_Monkey' mks{m} '.png'];
            saveas(fig_s5, fname);
            fprintf('  Saved: %s\n', fname);

        end  % monkey
    end  % parameter
end

%% Figure: Peak displacement (delta* per area, cross-state conditions)
%
% For each parameter, one figure (3x3 areas) showing mean peak location delta*
% for CU and UC conditions.  Circles = monkey M, squares = monkey X.
% Error bars = 95% CI across repetitions.  * = FDR-corrected p<fdr_alpha vs delta*=0.
% H2 predicts: delta*_CU and delta*_UC significant with opposite signs.
% H3 predicts: delta*_CU ~= delta*_UC ~= 0.

if plot_toggle_enabled(plot_shCCGP) && ~isempty(peak_loc_reps_final)
    for p = 1 : length(param.param2decode)
        param_name = strrep(param.param2decode{p}, '_2AFC', '');
        fig_ttl    = strrep(param.param2decode{p}, '_', ' ');

        fig_pk = figure('Name', ['Peak displacement \delta^* \u2014 ' param_name], ...
                        'Position', [50 50 1400 1100], 'Color', 'w');
        tl_pk  = tiledlayout(fig_pk, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        title(tl_pk, ['\delta^* of cross-state shCCGP peak - ' param_name ...
                       '  (circles=M, inv.triangles=X;  mean \pm 95% CI;  *: FDR p<' num2str(fdr_alpha) ' vs \delta^*=0)'], ...
              'FontSize', 17, 'FontWeight', 'bold');

        hm_order   = [1 2 3 9 8 4 7 6 5];
        mk_markers = {'o', 'v'};   % circle = M, inverted triangle = X
        mk_xoff    = [-0.12, 0.12];

        % Collect all cross-state p-values for FDR correction
        p_cross_vec = [];
        p_cross_idx = zeros(0, 3);   % rows: [m ar ci]
        for m = 1 : length(mks)
            for ar = 1 : nAreas
                if isempty(pk_stat{m}{ar,p}), continue; end
                for ci = 1 : 2   % 1=CU (trainC testU), 2=UC (trainU testC)
                    if ci == 1
                        trnS2 = 1; tstS2 = 2; x_pos = 1 + mk_xoff(m);
                    else
                        trnS2 = 2; tstS2 = 1; x_pos = 2 + mk_xoff(m);
                    end
                    pv = pk_stat{m}{ar,p}.pval_pk(trnS2, tstS2);
                    p_cross_vec(end+1)   = pv;         %#ok<AGROW>
                    p_cross_idx(end+1,:) = [m ar ci];  %#ok<AGROW>
                end
            end
        end
        sig_cross_fdr = false(size(p_cross_vec));
        if ~isempty(p_cross_vec)
            valid_p = ~isnan(p_cross_vec);
            if any(valid_p)
                [~, ~, pfdr] = utils_fdr_bh(p_cross_vec(valid_p));
                sig_cross_fdr(valid_p) = pfdr < fdr_alpha;
            end
        end

        for hm_idx = 1 : nAreas
            ar = hm_order(hm_idx);
            ax = nexttile(hm_idx);
            hold(ax, 'on');
            col = colorareas(ar, :) / 255;

            for m = 1 : length(mks)
                if isempty(pk_stat{m}{ar,p}), continue; end
                st = pk_stat{m}{ar,p};

                for ci = 1 : 2
                    if ci == 1
                        trnS2 = 1; tstS2 = 2; x_pos = 1 + mk_xoff(m);
                    else
                        trnS2 = 2; tstS2 = 1; x_pos = 2 + mk_xoff(m);
                    end
                    mn_v  = st.mean_pk(trnS2, tstS2);
                    sem_v = st.sem_pk(trnS2, tstS2);
                    if isnan(mn_v), continue; end

                    % Extract individual rep peaks for later star placement, but don't plot them
                    v_dots = [];
                    pklr = peak_loc_reps_final{m}{ar, p};
                    if ~isempty(pklr)
                        v_dots = squeeze(pklr(trnS2, tstS2, :));
                        v_dots = v_dots(~isnan(v_dots));
                    end

                    % Plot mean +/- 95% CI
                    errorbar(ax, x_pos, mn_v, 1.96*sem_v, mk_markers{m}, ...
                             'Color', col, 'LineWidth', 2, 'CapSize', 5, ...
                             'MarkerSize', 8, 'MarkerFaceColor', col, ...
                             'HandleVisibility', 'off');

                    % FDR significance * (median delta* != 0)
                    sig_mask = p_cross_idx(:,1)==m & p_cross_idx(:,2)==ar & p_cross_idx(:,3)==ci;
                    if any(sig_cross_fdr(sig_mask))
                        dir_v = sign(mn_v); if dir_v == 0, dir_v = 1; end
                        % Place star above/below the error bar
                        ystar = mn_v + dir_v * (1.96*sem_v + 0.30);
                        ystar = max(-0.85, min(0.85, ystar));  % clamp inside ylim [-1,1] with room for star
                        text(ax, x_pos, ystar, '*', ...
                             'Color', 'k', 'FontSize', 20, 'FontWeight', 'bold', ...
                             'HorizontalAlignment', 'center', 'HandleVisibility', 'off', ...
                             'Clipping', 'off');
                    end
                end

                % Connect CU-UC with a line to visualise symmetry
                mn_cu = st.mean_pk(1, 2);
                mn_uc = st.mean_pk(2, 1);
                if ~isnan(mn_cu) && ~isnan(mn_uc)
                    plot(ax, [1+mk_xoff(m) 2+mk_xoff(m)], [mn_cu mn_uc], '-', ...
                         'Color', [col 0.5], 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    % NOTE: s* (symmetry test) removed - confounded by shared-pseudopop
                    % correlation: CU and UC use the same neurons in each rep, so their
                    % sum is non-zero by sampling noise, not genuine same-direction shift.
                end
            end

            yline(ax, 0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
            set(ax, 'XTick', [1 2], 'XTickLabel', {'CU', 'UC'}, 'FontSize', 11);
            ylabel(ax, '\delta^* (norm. units)', 'FontSize', 11);
            title(ax, area2test_name{ar}, 'Color', col, 'FontWeight', 'bold', 'FontSize', 13);
            xlim(ax, [0.5 2.5]);
            ylim(ax, [-1 1]);   % CI of mean is much tighter than raw spread
            box(ax, 'off'); hold(ax, 'off');
        end

        drawnow;
        fname = [report_dir 'Fig_peak_disp_' param_name '.png'];
        saveas(fig_pk, fname);
        fprintf('  Saved: %s\n', fname);
    end  % parameter loop
end  % ~isempty peak_loc_reps_final

%% Within-state decoding curves per area (log2 N axis)
%
% Single 2x2 figure: rows = parameter (flavor top, side bottom),
%                    cols = monkey (M left, X right).
%
% y-axis: chosenflavor [0.45 0.85],  chosenside [0.4 1.0]
% Vertical dotted line at N_eval_global.
%
% Data source: acc_overall{mk}{ar,p}(:,:,1,1) = CC,  (:,:,2,2) = UU

tick_N_ws = [20 50 100 200 500];
tick_N_ws = tick_N_ws(tick_N_ws <= param.pseudopop(end));
ylims_ws  = {[0.45 0.85], [0.4 1.0]};   % {flavor, side}

fig_ws = figure('Name', 'Within-state decoding', ...
                'Position', [100 100 1100 1130], 'Color', 'w');
tl_ws  = tiledlayout(fig_ws, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl_ws, 'Within-state decoding  (avg chosen + unchosen)', ...
      'FontSize', 18, 'FontWeight', 'bold');

for p = 1 : length(param.param2decode)
    param_name_ws = strrep(param.param2decode{p}, '_2AFC', '');
    fig_ttl_ws    = strrep(param.param2decode{p}, '_', ' ');

    for m_idx = 1 : length(mks)
        % Layout: row = param, col = monkey  ->  tile (p-1)*2 + m_idx
        ax_ws = nexttile((p-1)*length(mks) + m_idx);
        hold(ax_ws, 'on');

        if isempty(acc_overall{m_idx})
            title(ax_ws, [fig_ttl_ws '  -  Monkey ' mks{m_idx}], 'FontSize', 12);
            continue;
        end

        for ar = 1 : nAreas
            if isempty(acc_overall{m_idx}{ar,p}), continue; end
            d = acc_overall{m_idx}{ar,p};   % [n_psz x n_rep x 2 x 2]

            d11   = squeeze(nanmean(d(:,:,1,1), 2));   % CC mean over reps
            d22   = squeeze(nanmean(d(:,:,2,2), 2));   % UU mean over reps
            curve = (d11 + d22) / 2;

            valid_n = find(~isnan(curve));
            if isempty(valid_n), continue; end

            col    = colorareas(ar,:) / 255;
            N_vals = param.pseudopop(valid_n);

            eval_idx = find(N_vals == N_eval_global, 1);
            if ~isempty(eval_idx)
                leg_label = sprintf('%s  %.3f @N=%d', area2test_name{ar}, ...
                                    curve(valid_n(eval_idx)), N_eval_global);
            else
                leg_label = sprintf('%s  (no data @N=%d)', area2test_name{ar}, N_eval_global);
            end

            plot(ax_ws, log2(N_vals), curve(valid_n), '-o', ...
                 'Color', col, 'LineWidth', 1.4, 'MarkerSize', 8, ...
                 'MarkerFaceColor', col, 'MarkerEdgeColor', 'none', ...
                 'DisplayName', leg_label);
        end

        yline(ax_ws, 0.5, 'k--', 'LineWidth', 0.9, 'HandleVisibility', 'off');
        xline(ax_ws, log2(N_eval_global), 'k--', 'LineWidth', 1.8, ...
              'Label', sprintf('N_{eval}=%d', N_eval_global), ...
              'LabelVerticalAlignment', 'bottom', 'FontSize', 9, ...
              'HandleVisibility', 'off');

        set(ax_ws, 'XTick', log2(tick_N_ws), ...
                   'XTickLabel', arrayfun(@num2str, tick_N_ws, 'UniformOutput', false), ...
                   'FontSize', 14);
        ylim(ax_ws, ylims_ws{p});
        if m_idx == 1
            ylabel(ax_ws, 'Within-state decoding', 'FontSize', 12);
        end
        if p == length(param.param2decode)
            xlabel(ax_ws, 'Pseudopop size N  (log_{2} scale)', 'FontSize', 12);
        end
        title(ax_ws, [fig_ttl_ws '  -  Monkey ' mks{m_idx}], ...
              'Interpreter', 'none', 'FontSize', 13);
        legend(ax_ws, 'Location', 'northwest', 'FontSize', 7);
        box(ax_ws, 'on');
    end
end

drawnow;
fname_ws = [report_dir 'Fig_S5_popsize.png'];
saveas(fig_ws, fname_ws);
fprintf('  Saved: %s\n', fname_ws);

%% Robustness: LME contrasts vs log2(N)
%
% Re-evaluates the three LME contrasts at N = 50, 100, 200, 350, 500 to confirm
% that reported results at N_eval_global = 200 are stable across pseudopop size.
%
% One 1x3-panel figure per (parameter x metric: CCGP / shCCGP / Improvement).
%
% Interpretation guide:
%   Filled circles at a given N = that area's estimate is significant (|z| > 1.96).
%   Dotted vertical line = N_eval_global (where main statistics are reported).
%   SE shading is opaque by default (alpha_rob = 1) for clean vector export;
%   set alpha_rob = 0.15 to visualise uncertainty bands.

alpha_rob    = 1;    % SE shading opacity (1 = hidden for Corel/vector export)
metric_names = {'CCGP', 'shCCGP', 'Improvement'};
metric_idx_rob = 1:(1 + 2*double(plot_shCCGP));

for p = 1 : length(param.param2decode)
    param_name_rob = strrep(param.param2decode{p}, '_2AFC', '');
    fig_ttl_rob    = strrep(param.param2decode{p}, '_', ' ');

    for metric = metric_idx_rob
        ct_rob = all_contrasts{p, metric};
        if ~isfield(ct_rob, 'N_evals_rob'), continue; end

        N_evals_rob  = ct_rob.N_evals_rob;
        LogN_rob_vec = log2(N_evals_rob);

        rob_tick_N = [50 100 200 500];
        rob_tick_N = rob_tick_N(rob_tick_N <= param.pseudopop(end));

        fig_rob = figure('Name', ['Robustness - ' metric_names{metric} ' - ' param_name_rob], ...
                         'Position', [100 100 1400 480], 'Color', 'w');
        tl_rob  = tiledlayout(fig_rob, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        title(tl_rob, [metric_names{metric} ' - ' fig_ttl_rob ...
                       '  |  Robustness: contrast vs log_{2}(N)'], ...
              'FontSize', 18, 'FontWeight', 'bold');

        % Panel order: gain modulation, geometry specificity, generalization
        cont_names      = {'Gain modulation',      'Geometry specificity', ...
                           'Generalization'};
        cont_est_fields = {'rob_gainmod_est', 'rob_geomspec_est', 'rob_genrz_est'};
        cont_se_fields  = {'rob_gainmod_se',  'rob_geomspec_se',  'rob_genrz_se'};

        for ci = 1 : 3
            ax_r = nexttile(ci);
            hold(ax_r, 'on');

            vals_mat = ct_rob.(cont_est_fields{ci});   % [nAreas x n_N_rob]
            se_mat   = ct_rob.(cont_se_fields{ci});

            for ar = 1 : nAreas
                col = colorareas(ar,:) / 255;
                v   = vals_mat(ar,:);
                s   = se_mat(ar,:);

                % 95% CI shading (opaque by default - set alpha_rob < 1 to show)
                x_fill = [LogN_rob_vec, fliplr(LogN_rob_vec)];
                y_fill = [v + 1.96*s,   fliplr(v - 1.96*s)];
                fill(ax_r, x_fill, y_fill, col, 'FaceAlpha', alpha_rob, ...
                     'EdgeColor', 'none', 'HandleVisibility', 'off');

                plot(ax_r, LogN_rob_vec, v, 'Color', col, 'LineWidth', 2, ...
                     'Marker', 'none', 'DisplayName', area2test_name{ar});

                % Filled dot at N values where estimate is significant (|z| > 1.96)
                sig_mask = abs(v ./ max(s, 1e-12)) > 1.96;
                if any(sig_mask)
                    plot(ax_r, LogN_rob_vec(sig_mask), v(sig_mask), 'o', ...
                         'Color', col, 'MarkerSize', 10, 'MarkerFaceColor', col, ...
                         'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
                end
            end

            yline(ax_r, 0, 'k--', 'LineWidth', 1);
            xline(ax_r, log2(N_eval_global), 'Color', [0.4 0.4 0.4], ...
                  'LineStyle', ':', 'LineWidth', 2, 'HandleVisibility', 'off');
            set(ax_r, 'XTick', log2(rob_tick_N), ...
                      'XTickLabel', arrayfun(@num2str, rob_tick_N, 'UniformOutput', false), ...
                      'FontSize', 20);
            xlim(ax_r, [log2(40), log2(param.pseudopop(end) + 100)]);
            xlabel(ax_r, 'Pseudopop size N  (log_{2} scale)', 'FontSize', 12);
            ylabel(ax_r, ['Contrast ' cont_names{ci}(1) ' estimate'], 'FontSize', 12);
            title(ax_r, cont_names{ci}, 'FontSize', 14, 'FontWeight', 'bold');
            if ci == 1
                legend(ax_r, 'Location', 'best', 'FontSize', 8);
            end
        end

        drawnow;
        if metric == 1 && strcmp(param_name_rob, 'chosenflavor')
            fname_rob_png = [report_dir 'Fig_S7a_robustness_flavor.png'];
            fname_rob_pdf = [report_dir 'Fig_S7a_robustness_flavor'];
        elseif metric == 1 && strcmp(param_name_rob, 'chosenside')
            fname_rob_png = [report_dir 'Fig_S7b_robustness_side.png'];
            fname_rob_pdf = [report_dir 'Fig_S7b_robustness_side'];
        else
            fname_rob_png = [report_dir 'Fig_robustness_' metric_names{metric} '_' param_name_rob '.png'];
            fname_rob_pdf = [report_dir 'Fig_robustness_' metric_names{metric} '_' param_name_rob];
        end
        saveas(fig_rob, fname_rob_png);
        % print(fig_rob, fname_rob_pdf, '-dpdf', '-vector');
        fprintf('  Saved: %s + .pdf\n', fname_rob_png);
    end
end

%% Posthoc: pairwise CCGP contrast comparisons between areas
%
% For CCGP (metric=1), compute pairwise LME-based t-tests comparing each
% contrast between every pair of areas using the full covariance
% matrix of the LME fit:
%
%   For contrast X and areas ar1, ar2:
%     xdiff = x_X(ar1) - x_X(ar2)   (contrast vectors from build_x)
%     t     = (xdiff' * coef_vals) / sqrt(xdiff' * CovMat * xdiff)
%     p     = 2 * tcdf(-|t|, dfe)
%
% FDR correction applied across all pairs within each contrast x parameter.
% Display: imagesc of |t|-statistic (colour) + FDR-adjusted p-values (text),
% mirroring the utils_areaposthoc convention (colour in upper triangle,
% text in lower triangle).
%
% Output: 1 figure per parameter (flavor / side), 3 subplots (gain modulation, geometry specificity, generalization).

% Toggle: false = show FDR-adjusted p-values; true = show '*' only when p < 0.01
if ~exist('posthoc_compact', 'var')
    posthoc_compact = true;
end
posthoc_star_thr  = 0.01;   % threshold for star in compact mode

cont_labels_ph = {'Gain modulation', ...
                  'Geometry specificity', ...
                  'Generalization'};

for p_ph = 1 : length(param.param2decode)

    info = lme_ccgp_info{p_ph};
    if isempty(info), continue; end

    cn     = info.coef_names;
    cv     = info.coef_vals;
    CM     = info.CovMat;
    dfe_ph = info.dfe;
    r_tr   = info.ref_train;
    r_te   = info.ref_test;
    r_ar   = info.ref_area;

    param_name_ph = strrep(param.param2decode{p_ph}, '_2AFC', '');

    % Pre-compute contrast vectors for every area -----------------------------
    % Geometry specificity = [(CC-CU) + (UU-UC)] / 2
    % Gain modulation = [(CC+UC) - (CU+UU)] / 2
    % Generalization = (CU+UC)/2  (-0.5 constant vanishes in diff)
    xB_vecs = zeros(nAreas, length(cn));
    xC_vecs = zeros(nAreas, length(cn));
    xD_vecs = zeros(nAreas, length(cn));

    for ar_ph = 1 : nAreas
        al   = area2test_name{ar_ph};
        x11  = build_x(cn, 'Chosen',   'Chosen',   al, LogN_common, r_tr, r_te, r_ar);
        x12  = build_x(cn, 'Chosen',   'Unchosen', al, LogN_common, r_tr, r_te, r_ar);
        x21  = build_x(cn, 'Unchosen', 'Chosen',   al, LogN_common, r_tr, r_te, r_ar);
        x22  = build_x(cn, 'Unchosen', 'Unchosen', al, LogN_common, r_tr, r_te, r_ar);
        xB_vecs(ar_ph, :) = ((x11 - x12) + (x22 - x21))' / 2;
        xC_vecs(ar_ph, :) = ((x11 + x21) - (x12 + x22))' / 2;
        xD_vecs(ar_ph, :) = ((x12 + x21) / 2)';
    end
    all_ph_xmats = {xC_vecs, xB_vecs, xD_vecs};  % order: gain modulation, geometry specificity, generalization

    % Build figure
    fig_ph = figure('Name', ['CCGP posthoc area - ' param_name_ph], ...
                    'Position', [417 416 1500 344], 'Color', 'w');
    tl_ph  = tiledlayout(fig_ph, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl_ph, ['CCGP - pairwise area posthoc - ' param_name_ph], ...
          'FontSize', 18, 'FontWeight', 'bold');

    % Purple gradient colormap (white to dark purple)
    purp_cmap = [linspace(0.97, 0.42, 100)', linspace(0.97, 0.15, 100)', linspace(0.97, 0.68, 100)'];

    for ci_ph = 1 : 3

        XM = all_ph_xmats{ci_ph};   % [nAreas x nCoefs]

        tstat_mat = zeros(nAreas, nAreas);   % lower triangle: ar1 > ar2
        pval_mat  = ones(nAreas,  nAreas);
        pval_vec  = [];
        pair_idx  = zeros(0, 2);

        for ar1 = 1 : nAreas
            for ar2 = 1 : nAreas
                if ar1 > ar2
                    xdiff = (XM(ar1,:) - XM(ar2,:))';
                    est_d = xdiff' * cv;
                    se_d  = sqrt(max(0, xdiff' * CM * xdiff));
                    if se_d < 1e-12, continue; end
                    t_val = est_d / se_d;
                    p_val = 2 * tcdf(-abs(t_val), dfe_ph);
                    tstat_mat(ar1, ar2) = t_val;
                    pval_mat(ar1,  ar2) = p_val;
                    pval_vec(end+1)     = p_val;        
                    pair_idx(end+1,:)   = [ar1 ar2];   
                end
            end
        end

        % FDR correction across all area pairs for this contrast
        pval_adj = ones(nAreas, nAreas);
        if ~isempty(pval_vec)
            [~, ~, adj_p] = utils_fdr_bh(pval_vec);
            for ip = 1 : size(pair_idx, 1)
                pval_adj(pair_idx(ip,1), pair_idx(ip,2)) = adj_p(ip);
            end
        end

        ax_ph = nexttile(ci_ph);

        % Colour: |t|-statistic.  imagesc(tstat_mat') places value for pair
        % (ar1,ar2) [ar1>ar2] at image position (x=ar1, y=ar2) - upper triangle.
        tmax = max(abs(tstat_mat(:)));
        if tmax < 0.5, tmax = 5; end
        imagesc(ax_ph, abs(tstat_mat'), [0 tmax]);
        colormap(ax_ph, purp_cmap);
        cb_ph = colorbar(ax_ph);
        cb_ph.Label.String = '|t|-statistic';
        cb_ph.Label.FontSize = 11;
        cb_ph.FontSize = 10;

        % Text overlay - same cell as the colour (x=ar1, y=ar2).
        % Compact mode: '*' only when p < posthoc_star_thr, blank otherwise.
        % Full mode: FDR-adjusted p-value always shown; bold when significant.
        % Text colour: white on dark cells (|t| > half colour range), else black.
        for ar1 = 1 : nAreas
            for ar2 = 1 : nAreas
                if ar1 > ar2
                    p_adj = pval_adj(ar1, ar2);
                    t_abs = abs(tstat_mat(ar1, ar2));
                    txt_col = [1 1 1] * double(t_abs > tmax * 0.5);

                    if posthoc_compact
                        % Compact: only draw a star if p < posthoc_star_thr
                        if p_adj < posthoc_star_thr
                            text(ax_ph, ar1, ar2, '*', ...
                                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                                 'FontSize', 25, 'FontWeight', 'bold', 'Color', txt_col);
                        end
                    else
                        % Full: always show rounded p-value
                        n_dig = 1; p_rd = 0;
                        while p_rd == 0 && n_dig < 6
                            n_dig = n_dig + 1;
                            p_rd  = round(p_adj, n_dig);
                        end
                        p_rd = round(p_adj, n_dig + 1);
                        if p_adj < fdr_alpha
                            text(ax_ph, ar1, ar2, num2str(p_rd), ...
                                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                                 'FontSize', 10, 'FontWeight', 'bold', 'Color', txt_col);
                        else
                            text(ax_ph, ar1, ar2, num2str(p_rd), ...
                                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                                 'FontSize', 9, 'Color', txt_col);
                        end
                    end
                end
            end
        end

        set(ax_ph, 'XTick', 1:nAreas, 'XTickLabel', area2test_name, ...
                   'YTick', 1:nAreas, 'YTickLabel', area2test_name, 'FontSize', 12);
        xtickangle(ax_ph, 45);
        xlim(ax_ph, [1.5  nAreas + 0.5]);
        ylim(ax_ph, [0.5  nAreas - 0.5]);
        title(ax_ph, cont_labels_ph{ci_ph}, 'FontSize', 13, 'FontWeight', 'bold');

    end  % ci_ph

    drawnow;
    if posthoc_compact
        if strcmp(param_name_ph, 'chosenflavor')
            fname_ph = [report_dir 'Fig_S6d_area_flavor.png'];
        elseif strcmp(param_name_ph, 'chosenside')
            fname_ph = [report_dir 'Fig_S6d_area_side.png'];
        else
            fname_ph = [report_dir 'Fig_posthoc_CCGP_' param_name_ph '_compact.png'];
        end
    else
        fname_ph = [report_dir 'Fig_posthoc_CCGP_' param_name_ph '.png'];
    end
    saveas(fig_ph, fname_ph);
    fprintf('  Saved: %s\n', fname_ph);

end  % p_ph

save(savefile, 'all_contrasts', '-append');
fprintf('\nDone. All results appended to %s\n', savefile);

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function x = build_x(coef_names, trnS_lbl, tstS_lbl, ar_lbl, Ns, ref_train, ref_test, ref_area)
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
        for k = 1 : n_act
            if bitand(bits, bitshift(1, k-1))
                subset{end+1} = active{k};
            end
        end
        if isempty(subset)
            idx = find(strcmp(coef_names, '(Intercept)'), 1);
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

function tf = plot_toggle_enabled(plot_shCCGP)
    tf = logical(plot_shCCGP);
end

function idx = find_coef_exact(coef_names, required_parts)
    n_req = length(required_parts);
    idx   = 0;
    for k = 1 : length(coef_names)
        parts_k = strsplit(coef_names{k}, ':');
        if length(parts_k) ~= n_req, continue; end
        if all(ismember(required_parts, parts_k)) && all(ismember(parts_k, required_parts))
            idx = k;
            return;
        end
    end
end

