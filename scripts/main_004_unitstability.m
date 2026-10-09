%% main_004_unitstability.m
%
% Unit-level stability and sign consistency summary.
%
% Figure S8A (top):    Cross-state congruency flip rate per area
%                      Left  = flavor, Right = side
% Figure S8B (bottom): Generalization vs
%                      single-neuron sign consistency (scatter)
%                      Left  = flavor, Right = side
%
% Statistical tables (congruency counts, flip scores) are printed to report_004.

clear

%% ========================================================================
% Shared parameters
% =========================================================================
f = mfilename('fullpath');
if isempty(f) 
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end
addpath(genpath([currentPath '\scripts\'])); % add scripts folder to path

path2go  = [currentPath '\processed\'];
report_dir = [currentPath '\report_004\'];
if ~exist(report_dir,'dir'), mkdir(report_dir); end

fid_log = fopen([report_dir 'output_004.txt'], 'w');
log_cleanup = onCleanup(@() fclose(fid_log));
utils_diary(fid_log, '\n============================================================\n');
utils_diary(fid_log, 'Report generated: %s\n', datestr(now));

area2test_name = {'MFC' 'PMC' 'dlPFC' 'IFG' 'vlPFC' 'AI' 'OFC' 'STR' 'AMG'};
colorareas     = [230 171 2 ; 152 78 163 ; 237 87 90 ; 252 141 98 ; 141 160 203 ; ...
                  166 216 84 ; 102 194 165 ; 180 180 180 ; 231 138 195] / 255;

nStates     = 3;
state_names = {'Chosen','Unchosen','Other'};
state_col   = [100 200 160 ; 240 90 90 ; 0 0 0] / 255;

require_sig_2afc = false;
sig_thresh       = 0.05;

nAreas = length(area2test_name);

%% ========================================================================
% Load shared data
% =========================================================================
disp('Loading shared data...')

load([path2go 'states_2afc_final.mat'], 'out_all', 'param')
out = out_all(cellfun(@isempty, {out_all(:).area4unit_removed}));
clear out_all

% create a all_units matrix with neuron number (col 1) and out.keptUnits (col 2) to be able to filter the flavor units later on.
all_units = [];
for s = 1 : length(out)
    all_units = [all_units ;  repmat(s, length(out(s).keptUnits),1) out(s).keptUnits];
end

unified_file = [path2go 'states_2afc_ccgp.mat'];

disp('  Done.')

%% ########################################################################
%% SECTION A: FLAVOR
%% ########################################################################
disp(' ')
disp('########################################')
disp('  SECTION A: FLAVOR')
disp('########################################')

%% ========================================================================
% Load flavor unit table & filter
% =========================================================================
load([path2go 'flavor_1fc.mat'], 'table_flavor_1FC')

valid_and_flavor = intersect(find(all_units(:,2) == 1), table_flavor_1FC.unit_nb);
table_filt = table_flavor_1FC(ismember(table_flavor_1FC.unit_nb, valid_and_flavor), :);
nNeurons   = height(table_filt);
fprintf('Flavor-encoding neurons after filtering: %d\n', nNeurons);

%% ========================================================================
% Compute flavor coefficient per neuron per proba state in 2AFC
% =========================================================================
disp('Computing per-neuron flavor coefficients in each proba state...')

p_flavor       = NaN(nNeurons, nStates);
coeff_flavor   = NaN(nNeurons, nStates);
ar_label       = cell(nNeurons, 1);
matched_unitnb = NaN(nNeurons, 1);

x = 0;

for s = 1 : length(out)
    if mod(s,5)==0, fprintf('  Session %d / %d\n', s, length(out)); end

    sess_unit  = find(all_units(:,1) == s & all_units(:,2) == 1);
    unit2test  = intersect(sess_unit, table_filt.unit_nb);
    if isempty(unit2test), continue; end

    flavors = [out(s).cond.chosenflavor_2AFC, out(s).cond.unchosenflavor_2AFC];
    diff_flavor_trials = find(flavors(:,1) ~= flavors(:,2));
    if length(diff_flavor_trials) < 10, continue; end

    for u = 1 : length(unit2test)
        x = x + 1;
        matched_unitnb(x) = unit2test(u);
        local_idx = find(sess_unit == unit2test(u));

        curr_fr = squeeze(out(s).fr_heldout(local_idx, :, :));
        ar_raw  = out(s).area4unit_heldout{local_idx};
        ar_label{x} = ar_raw;

        fr_states = NaN(size(curr_fr, 1), nStates);
        for c = 1 : nStates
            tmp = curr_fr;
            tmp(out(s).states ~= c) = NaN;
            fr_states(:, c) = nanmean(tmp, 2);
        end

        for c = 1 : nStates
            y = fr_states(diff_flavor_trials, c);
            grp = flavors(diff_flavor_trials, 1) - 1;
            valid_obs = ~isnan(y);
            if sum(valid_obs) < 5, continue; end
            [p, ~, stats] = anovan(y(valid_obs), grp(valid_obs), ...
                'continuous', [], 'display', 'off');
            p_flavor(x, c)     = p(1);
            coeff_flavor(x, c) = stats.coeffs(2);
        end
    end
end

% Trim to matched neurons
p_flavor       = p_flavor(1:x, :);
coeff_flavor   = coeff_flavor(1:x, :);
ar_label       = ar_label(1:x);
matched_unitnb = matched_unitnb(1:x);

[~, reindex] = ismember(matched_unitnb, table_filt.unit_nb);
assert(all(reindex > 0), 'Some matched neurons not found in table_filt.');
table_matched = table_filt(reindex, :);
fprintf('Neurons matched to sessions: %d / %d\n', x, nNeurons);

sign_2afc = sign(coeff_flavor);
sign_1fc  = table_matched.flavor_1FC_sign;

%% ========================================================================
% Count same-sign neurons per area x state + binomial tests
% =========================================================================
n_same  = NaN(nAreas, nStates);
n_total = NaN(nAreas, nStates);
prop    = NaN(nAreas, nStates);
p_binom = NaN(nAreas, nStates);

for st = 1 : nStates
    for ar = 1 : nAreas
        if require_sig_2afc
            idx = find(strcmp(ar_label, area2test_name{ar}) & p_flavor(:,st) < sig_thresh);
        else
            idx = find(strcmp(ar_label, area2test_name{ar}));
        end
        idx = idx(~isnan(sign_2afc(idx, st)));
        n_tot = length(idx);
        n_sam = sum(sign_1fc(idx) == sign_2afc(idx, st));
        n_same(ar,st)  = n_sam;
        n_total(ar,st) = n_tot;
        prop(ar,st)    = n_sam / n_tot;
        if n_tot > 0
            p_binom(ar,st) = min(1, 2 * min(1 - binocdf(n_sam-1, n_tot, 0.5), ...
                                              binocdf(n_sam, n_tot, 0.5)));
        end
    end
end

% FDR correction
all_pvals = p_binom(:);
valid_p   = ~isnan(all_pvals);
[~, ~, adj_p_tmp] = utils_fdr_bh(all_pvals(valid_p));
p_binom_fdr = NaN(size(all_pvals));
p_binom_fdr(valid_p) = adj_p_tmp;
p_binom_fdr = reshape(p_binom_fdr, nAreas, nStates);

% Chi-squared across areas
chi2_p = NaN(nStates,1);  chi2_stat = NaN(nStates,1);
for st = 1 : nStates
    obs  = [n_same(:,st), n_total(:,st) - n_same(:,st)];
    keep = n_total(:,st) > 0;  obs = obs(keep,:);
    if size(obs,1) < 2, continue; end
    row_sum = sum(obs,2);  col_sum = sum(obs,1);  grand = sum(col_sum);
    expected = row_sum * col_sum / grand;
    chi2_stat(st) = sum((obs(:) - expected(:)).^2 ./ expected(:));
    chi2_p(st) = 1 - chi2cdf(chi2_stat(st), (size(obs,1)-1)*(size(obs,2)-1));
end

%% ========================================================================
% Display flavor sign-consistency table (kept for stats reporting)
% =========================================================================
utils_diary(fid_log, '\n================================================================\n');
utils_diary(fid_log, '  SIGN CONSISTENCY (FLAVOR): 1FC identity preservation in 2AFC states\n');
utils_diary(fid_log, '  same sign = chosen-flavor coding | opp sign = unchosen-flavor coding\n');
utils_diary(fid_log, '================================================================\n');
for st = 1 : nStates
    utils_diary(fid_log, '\n--- State %d: %s ---\n', st, state_names{st});
    tbl_sc_flav = table(area2test_name', n_same(:,st), n_total(:,st), prop(:,st), ...
               p_binom(:,st), p_binom_fdr(:,st), ...
               'VariableNames', {'Area','N_same','N_total','Proportion','p_binom','p_FDR'});
    utils_diary(fid_log, '%s', regexprep(evalc('disp(tbl_sc_flav)'), '</?strong>', ''));
    utils_diary(fid_log, '  Chi-squared across areas: chi2 = %.2f, p = %.4g\n', chi2_stat(st), chi2_p(st));
end

% Create combined 2x3 figure: flavor (row 1) | side (row 2)
%   columns: sign consistency | scatter vs generalization | flip rate
fig_combined = figure('Name','Fig S11 - Sign Consistency & Flip Rate','Position',[50 50 1600 950]);

%% ========================================================================
% Subplot (2,3,1): Flavor sign consistency - deviation from chance
% =========================================================================
subplot(2,3,1); hold on;
bar_w = 0.25;  x_pos = 1 : nAreas;  h_st = gobjects(1,2);
for st = 1 : 2
    offset = (st-1.5) * bar_w;
    for ar = 1 : nAreas
        bx  = x_pos(ar) + offset;
        val = prop(ar,st);
        h_st(st) = bar(bx, val, bar_w, 'FaceColor', state_col(st,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
        if ~isnan(p_binom_fdr(ar,st)) && p_binom_fdr(ar,st) < 0.05
            text(bx, val+0.02, '*', 'HorizontalAlignment','center','FontSize',21,'Color',state_col(st,:),'FontWeight','bold');
        end
    end
end
yline(0.5,'--','Color',[0.5 0.5 0.5],'LineWidth',1.2);
set(gca,'XTick',x_pos,'XTickLabel',area2test_name,'FontSize',16); xtickangle(90);
ylim([0 1]); ylabel('P(same sign as 1FC)'); title('Flavor: sign consistency');
legend(h_st, {'Chosen','Unchosen'},'Location','northeast','FontSize',15); box off;

%% ========================================================================
% Congruency reframing table (FLAVOR)
% =========================================================================
utils_diary(fid_log, '\n================================================================\n');
utils_diary(fid_log, '  CONGRUENCY REFRAMING (FLAVOR)\n');
utils_diary(fid_log, '  P(same sign) = P(chosen-flavor coding)\n');
utils_diary(fid_log, '================================================================\n');
for st = 1 : 2
    utils_diary(fid_log, '\n--- State %d: %s-probability ---\n', st, state_names{st});
    utils_diary(fid_log, '  %-8s  N_chosen  N_unchosen  N_total  %%Chosen   %%Unchosen  p_FDR\n', 'Area');
    for ar = 1 : nAreas
        nc = n_same(ar,st);  nu = n_total(ar,st) - n_same(ar,st);  nt = n_total(ar,st);
        utils_diary(fid_log, '  %-8s  %5d     %5d       %5d    %5.1f%%     %5.1f%%     %.4g\n', ...
            area2test_name{ar}, nc, nu, nt, 100*nc/max(nt,1), 100*nu/max(nt,1), p_binom_fdr(ar,st));
    end
end

%% ========================================================================
% Cross-state flip score (FLAVOR)
% =========================================================================
utils_diary(fid_log, '\n================================================================\n');
utils_diary(fid_log, '  CROSS-STATE FLIP SCORE (FLAVOR)\n');
utils_diary(fid_log, '  Does a neuron switch chosen<->unchosen flavor coding across states?\n');
utils_diary(fid_log, '================================================================\n');

is_chosen_st1_flav = (sign_1fc == sign_2afc(:,1));
is_chosen_st2_flav = (sign_1fc == sign_2afc(:,2));
valid_both_flav    = ~isnan(sign_2afc(:,1)) & ~isnan(sign_2afc(:,2));

flip_n_total_flav  = NaN(nAreas, 1);
flip_n_flip_flav   = NaN(nAreas, 1);
flip_prop_flav     = NaN(nAreas, 1);
flip_p_binom_flav  = NaN(nAreas, 1);
flip_p_indep_flav  = NaN(nAreas, 1);
flip_exp_rate_flav = NaN(nAreas, 1);
flip_n_c2u_flav    = NaN(nAreas, 1);
flip_n_u2c_flav    = NaN(nAreas, 1);
flip_n_cc_flav     = NaN(nAreas, 1);
flip_n_uu_flav     = NaN(nAreas, 1);

for ar = 1 : nAreas
    idx = find(strcmp(ar_label, area2test_name{ar}) & valid_both_flav);
    n_tot = length(idx);
    flip_n_total_flav(ar) = n_tot;
    if n_tot < 2, continue; end

    c1 = is_chosen_st1_flav(idx);
    c2 = is_chosen_st2_flav(idx);
    n_flip = sum(c1 ~= c2);

    flip_n_flip_flav(ar) = n_flip;
    flip_prop_flav(ar)   = n_flip / n_tot;
    flip_n_c2u_flav(ar)  = sum(c1 == 1 & c2 == 0);
    flip_n_u2c_flav(ar)  = sum(c1 == 0 & c2 == 1);
    flip_n_cc_flav(ar)   = sum(c1 == 1 & c2 == 1);
    flip_n_uu_flav(ar)   = sum(c1 == 0 & c2 == 0);

    flip_p_binom_flav(ar) = min(1, 2 * min(1 - binocdf(n_flip-1, n_tot, 0.5), ...
                                            binocdf(n_flip, n_tot, 0.5)));

    p1 = prop(ar,1);  p2 = prop(ar,2);
    p_exp = p1*(1-p2) + (1-p1)*p2;
    flip_exp_rate_flav(ar) = p_exp;
    if p_exp > 0 && p_exp < 1
        flip_p_indep_flav(ar) = min(1, 2 * min(1 - binocdf(n_flip-1, n_tot, p_exp), ...
                                                binocdf(n_flip, n_tot, p_exp)));
    end
end

% FDR correction
valid_fp = ~isnan(flip_p_binom_flav);
flip_p_fdr_flav = NaN(nAreas, 1);
if sum(valid_fp) > 1
    [~, ~, adj_fp] = utils_fdr_bh(flip_p_binom_flav(valid_fp));
    flip_p_fdr_flav(valid_fp) = adj_fp;
end

valid_ip = ~isnan(flip_p_indep_flav);
flip_p_indep_fdr_flav = NaN(nAreas, 1);
if sum(valid_ip) > 1
    [~, ~, adj_ip] = utils_fdr_bh(flip_p_indep_flav(valid_ip));
    flip_p_indep_fdr_flav(valid_ip) = adj_ip;
end

utils_diary(fid_log, '\n%-8s  N_tot  Flip  P(flip)  Exp_indep  C->U  U->C  C->C  U->U  p_50%%_FDR  p_indep_FDR\n', 'Area');
for ar = 1 : nAreas
    utils_diary(fid_log, '%-8s  %4d   %4d  %5.1f%%    %5.1f%%   %4d  %4d  %4d  %4d  %.4g     %.4g\n', ...
        area2test_name{ar}, flip_n_total_flav(ar), flip_n_flip_flav(ar), ...
        100*flip_prop_flav(ar), 100*flip_exp_rate_flav(ar), ...
        flip_n_c2u_flav(ar), flip_n_u2c_flav(ar), flip_n_cc_flav(ar), flip_n_uu_flav(ar), ...
        flip_p_fdr_flav(ar), flip_p_indep_fdr_flav(ar));
end

% Pooled tests
idx_pool_flav = find(valid_both_flav);
n_flip_pool = sum(is_chosen_st1_flav(idx_pool_flav) ~= is_chosen_st2_flav(idx_pool_flav));
n_tot_pool  = length(idx_pool_flav);
p_flip_50   = min(1, 2 * min(1 - binocdf(n_flip_pool-1, n_tot_pool, 0.5), ...
                              binocdf(n_flip_pool, n_tot_pool, 0.5)));
p1_pool = mean(is_chosen_st1_flav(idx_pool_flav));
p2_pool = mean(is_chosen_st2_flav(idx_pool_flav));
p_exp_pool = p1_pool*(1-p2_pool) + (1-p1_pool)*p2_pool;
p_flip_ind = min(1, 2 * min(1 - binocdf(n_flip_pool-1, n_tot_pool, p_exp_pool), ...
                             binocdf(n_flip_pool, n_tot_pool, p_exp_pool)));
utils_diary(fid_log, '\nPooled: %d / %d flip (%.1f%%), expected under indep = %.1f%%\n', ...
    n_flip_pool, n_tot_pool, 100*n_flip_pool/n_tot_pool, 100*p_exp_pool);
utils_diary(fid_log, '  vs 50%%: p = %.4g  |  vs independence: p = %.4g\n', p_flip_50, p_flip_ind);

n_cu_flav = sum(is_chosen_st1_flav(idx_pool_flav) == 1 & is_chosen_st2_flav(idx_pool_flav) == 0);
n_uc_flav = sum(is_chosen_st1_flav(idx_pool_flav) == 0 & is_chosen_st2_flav(idx_pool_flav) == 1);
if (n_cu_flav + n_uc_flav) > 0
    mcnemar_chi2 = (n_cu_flav - n_uc_flav)^2 / (n_cu_flav + n_uc_flav);
    mcnemar_p = 1 - chi2cdf(mcnemar_chi2, 1);
    utils_diary(fid_log, 'McNemar (C->U vs U->C): %d vs %d, chi2 = %.2f, p = %.4g\n', ...
        n_cu_flav, n_uc_flav, mcnemar_chi2, mcnemar_p);
end

obs_flip_flav = [flip_n_flip_flav, flip_n_total_flav - flip_n_flip_flav];
keep_flav = flip_n_total_flav > 0 & ~isnan(flip_n_flip_flav);
if sum(keep_flav) > 1
    obs_f = obs_flip_flav(keep_flav,:);
    row_s = sum(obs_f,2);  col_s = sum(obs_f,1);  grand = sum(col_s);
    exp_f = row_s * col_s / grand;
    chi2_flip = sum((obs_f(:) - exp_f(:)).^2 ./ exp_f(:));
    df_flip = (size(obs_f,1)-1) * (size(obs_f,2)-1);
    p_chi2_flip = 1 - chi2cdf(chi2_flip, df_flip);
    utils_diary(fid_log, 'Chi-squared (flip rate across areas): chi2 = %.2f, df = %d, p = %.4g\n', ...
        chi2_flip, df_flip, p_chi2_flip);
end

%% ========================================================================
% Subplot (2,3,3): Flavor flip rate
% =========================================================================
subplot(2,3,3); hold on;
for ar = 1 : nAreas
    bar(ar, flip_prop_flav(ar), 0.7, 'FaceColor', colorareas(ar,:), 'EdgeColor','none');
    text(ar, 0.03, sprintf('n=%d',flip_n_total_flav(ar)), 'Rotation',90,'HorizontalAlignment','left','VerticalAlignment','middle','FontSize',12,'Color',[1 1 1]);
    if ~isnan(flip_p_fdr_flav(ar)) && flip_p_fdr_flav(ar) < 0.05
        text(ar, flip_prop_flav(ar)+0.025, '*', 'HorizontalAlignment','center','FontSize',21,'FontWeight','bold');
    end
    if ~isnan(flip_exp_rate_flav(ar))
        plot(ar + [-0.3 0.3], [flip_exp_rate_flav(ar) flip_exp_rate_flav(ar)], '-', ...
            'Color', colorareas(ar,:)*0.5, 'LineWidth', 1.5);
    end
end
yline(0.5,'--','Color',[0.5 0.5 0.5],'LineWidth',1);
set(gca,'XTick',1:nAreas,'XTickLabel',area2test_name,'FontSize',16); xtickangle(90);
ylim([0 1]); ylabel('Cross-state flip proportion');
title('Flavor: cross-state flip rate','FontSize',18); box off;

%% ========================================================================
% Subplot (2,3,2): Flavor scatter - sign consistency vs generalization
% =========================================================================
subplot(2,3,2); hold on;
if exist(unified_file,'file') && ~isempty(who('-file',unified_file,'all_contrasts'))
    uni = load(unified_file,'all_contrasts','param');
    p_idx = find(strcmp(uni.param.param2decode,'chosenflavor_2AFC'));
    if isempty(p_idx), p_idx = 1; end
    genrz_model  = uni.all_contrasts{p_idx, 1}.genrz_est;
    genrz_se_mod = uni.all_contrasts{p_idx, 1}.genrz_se;
    sign_consist_avg_flav = nanmean(prop(:,1:2), 2);
    for ar = 1 : nAreas
        scatter(genrz_model(ar), sign_consist_avg_flav(ar), 100, colorareas(ar,:), 'filled', 'MarkerEdgeColor',[0 0 0],'LineWidth',0.8);
        plot([genrz_model(ar)-1.96*genrz_se_mod(ar), genrz_model(ar)+1.96*genrz_se_mod(ar)], ...
             [sign_consist_avg_flav(ar), sign_consist_avg_flav(ar)], '-','Color',[0.6 0.6 0.6],'LineWidth',0.8);
        text(genrz_model(ar)+0.003, sign_consist_avg_flav(ar)+0.008, area2test_name{ar}, 'FontSize',15,'FontWeight','bold','Color',colorareas(ar,:));
    end
    xline(0,'--','Color',[0.7 0.7 0.7]); yline(0.5,'--','Color',[0.7 0.7 0.7]);
    xlabel('Generalization (cross-state CCGP) - 0.5 (LME at N=200)');
    ylabel('Sign consistency (avg Chosen & Unchosen)');
    title('Flavor: cross-state decoding vs sign preservation');
    set(gca,'FontSize',17); box off;
    valid = ~isnan(genrz_model) & ~isnan(sign_consist_avg_flav);
    if sum(valid) > 3
        [r_corr,p_corr] = corr(genrz_model(valid), sign_consist_avg_flav(valid));
        text(0.05,0.95,sprintf('r = %.3f, p = %.3f',r_corr,p_corr),'Units','normalized','FontSize',16,'VerticalAlignment','top');
    end
else
    warning('states_2afc_ccgp.mat not found - skipping generalization scatter (flavor).');
    text(0.5,0.5,'data not found','HorizontalAlignment','center','Units','normalized');
end

disp('Section A (flavor) done.')

%% ########################################################################
%% SECTION B: SIDE
%% ########################################################################
disp(' ')
disp('########################################')
disp('  SECTION B: SIDE')
disp('########################################')

%% ========================================================================
% Load side unit table & filter
% =========================================================================
load([path2go 'side_1fc.mat'], 'table_side_1FC')

valid_and_side = intersect(find(all_units(:,2) == 1), table_side_1FC.unit_nb);
table_filt = table_side_1FC(ismember(table_side_1FC.unit_nb, valid_and_side), :);
nNeurons   = height(table_filt);
fprintf('Side-encoding neurons after filtering: %d\n', nNeurons);

%% ========================================================================
% Compute side coefficient per neuron per HMM state in 2AFC
% =========================================================================
disp('Computing per-neuron side coefficients in each HMM state...')

p_side         = NaN(nNeurons, nStates);
coeff_side     = NaN(nNeurons, nStates);
ar_label       = cell(nNeurons, 1);
matched_unitnb = NaN(nNeurons, 1);

x = 0;

for s = 1 : length(out)
    if mod(s,5)==0, fprintf('  Session %d / %d\n', s, length(out)); end

    sess_unit = find(all_units(:,1) == s & all_units(:,2) == 1);
    unit2test = intersect(sess_unit, table_filt.unit_nb);
    if isempty(unit2test), continue; end

    sides = out(s).cond.chosenside_2AFC;
    if length(unique(sides)) < 2, continue; end
    grp_side = sides - 1;

    for u = 1 : length(unit2test)
        x = x + 1;
        matched_unitnb(x) = unit2test(u);
        local_idx = find(sess_unit == unit2test(u));

        curr_fr = squeeze(out(s).fr_heldout(local_idx, :, :));
        ar_raw  = out(s).area4unit_heldout{local_idx};
        ar_label{x} = ar_raw;

        fr_states = NaN(size(curr_fr,1), nStates);
        for c = 1 : nStates
            tmp = curr_fr;
            tmp(out(s).states ~= c) = NaN;
            fr_states(:,c) = nanmean(tmp, 2);
        end

        for c = 1 : nStates
            y = fr_states(:, c);
            valid_obs = ~isnan(y);
            if sum(valid_obs) < 5, continue; end
            if length(unique(grp_side(valid_obs))) < 2, continue; end
            [p,~,stats] = anovan(y(valid_obs), grp_side(valid_obs), 'continuous',[],'display','off');
            p_side(x,c)     = p(1);
            coeff_side(x,c) = stats.coeffs(2);
        end
    end
end

% Trim
p_side         = p_side(1:x, :);
coeff_side     = coeff_side(1:x, :);
ar_label       = ar_label(1:x);
matched_unitnb = matched_unitnb(1:x);

[~, reindex] = ismember(matched_unitnb, table_filt.unit_nb);
assert(all(reindex > 0), 'Some matched neurons not found in table_filt.');
table_matched = table_filt(reindex, :);
fprintf('Neurons matched to sessions: %d / %d\n', x, nNeurons);

sign_2afc = sign(coeff_side);
sign_1fc  = table_matched.side_1FC_sign;

%% ========================================================================
% Count same-sign neurons per area x state + binomial tests
% =========================================================================
n_same  = NaN(nAreas, nStates);
n_total = NaN(nAreas, nStates);
prop    = NaN(nAreas, nStates);
p_binom = NaN(nAreas, nStates);

for st = 1 : nStates
    for ar = 1 : nAreas
        if require_sig_2afc
            idx = find(strcmp(ar_label, area2test_name{ar}) & p_side(:,st) < sig_thresh);
        else
            idx = find(strcmp(ar_label, area2test_name{ar}));
        end
        idx = idx(~isnan(sign_2afc(idx,st)));
        n_tot = length(idx);  n_sam = sum(sign_1fc(idx) == sign_2afc(idx,st));
        n_same(ar,st)  = n_sam;
        n_total(ar,st) = n_tot;
        prop(ar,st)    = n_sam / n_tot;
        if n_tot > 0
            p_binom(ar,st) = min(1, 2*min(1-binocdf(n_sam-1,n_tot,0.5), binocdf(n_sam,n_tot,0.5)));
        end
    end
end

all_pvals = p_binom(:);  valid_p = ~isnan(all_pvals);
[~,~,adj_p_tmp] = utils_fdr_bh(all_pvals(valid_p));
p_binom_fdr = NaN(size(all_pvals));
p_binom_fdr(valid_p) = adj_p_tmp;
p_binom_fdr = reshape(p_binom_fdr, nAreas, nStates);

chi2_p = NaN(nStates,1);  chi2_stat = NaN(nStates,1);
for st = 1 : nStates
    obs  = [n_same(:,st), n_total(:,st)-n_same(:,st)];
    keep = n_total(:,st) > 0;  obs = obs(keep,:);
    if size(obs,1) < 2, continue; end
    row_sum = sum(obs,2);  col_sum = sum(obs,1);  grand = sum(col_sum);
    expected = row_sum * col_sum / grand;
    chi2_stat(st) = sum((obs(:)-expected(:)).^2 ./ expected(:));
    chi2_p(st) = 1 - chi2cdf(chi2_stat(st), (size(obs,1)-1)*(size(obs,2)-1));
end

%% ========================================================================
% Display side sign-consistency table
% =========================================================================
utils_diary(fid_log, '\n================================================================\n');
utils_diary(fid_log, '  SIGN CONSISTENCY (SIDE): 1FC identity preservation in 2AFC states\n');
utils_diary(fid_log, '  same sign = chosen-side coding | opp sign = unchosen-side coding\n');
utils_diary(fid_log, '================================================================\n');
for st = 1 : nStates
    utils_diary(fid_log, '\n--- State %d: %s ---\n', st, state_names{st});
    tbl_sc_side = table(area2test_name', n_same(:,st), n_total(:,st), prop(:,st), ...
               p_binom(:,st), p_binom_fdr(:,st), ...
               'VariableNames', {'Area','N_same','N_total','Proportion','p_binom','p_FDR'});
    utils_diary(fid_log, '%s', regexprep(evalc('disp(tbl_sc_side)'), '</?strong>', ''));
    utils_diary(fid_log, '  Chi-squared across areas: chi2 = %.2f, p = %.4g\n', chi2_stat(st), chi2_p(st));
end

%% ========================================================================
% Subplot (2,3,4): Side sign consistency - deviation from chance
% =========================================================================
subplot(2,3,4); hold on;
for st = 1 : 2
    offset = (st-1.5) * bar_w;
    for ar = 1 : nAreas
        bx  = x_pos(ar) + offset;
        val = prop(ar,st);
        h_st(st) = bar(bx, val, bar_w, 'FaceColor', state_col(st,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
        if ~isnan(p_binom_fdr(ar,st)) && p_binom_fdr(ar,st) < 0.05
            text(bx, val+0.02, '*', 'HorizontalAlignment','center','FontSize',21,'Color',state_col(st,:),'FontWeight','bold');
        end
    end
end
yline(0.5,'--','Color',[0.5 0.5 0.5],'LineWidth',1.2);
set(gca,'XTick',x_pos,'XTickLabel',area2test_name,'FontSize',16); xtickangle(90);
ylim([0 1]); ylabel('P(same sign as 1FC)'); title('Side: sign consistency');
legend(h_st, {'Chosen','Unchosen'},'Location','northeast','FontSize',15); box off;

%% ========================================================================
% Congruency reframing table (SIDE)
% =========================================================================
utils_diary(fid_log, '\n================================================================\n');
utils_diary(fid_log, '  CONGRUENCY REFRAMING (SIDE)\n');
utils_diary(fid_log, '  P(same sign) = P(chosen-side coding)\n');
utils_diary(fid_log, '================================================================\n');
for st = 1 : 2
    utils_diary(fid_log, '\n--- State %d: %s-probability ---\n', st, state_names{st});
    utils_diary(fid_log, '  %-8s  N_chosen  N_unchosen  N_total  %%Chosen   %%Unchosen  p_FDR\n', 'Area');
    for ar = 1 : nAreas
        nc = n_same(ar,st);  nu = n_total(ar,st) - n_same(ar,st);  nt = n_total(ar,st);
        utils_diary(fid_log, '  %-8s  %5d     %5d       %5d    %5.1f%%     %5.1f%%     %.4g\n', ...
            area2test_name{ar}, nc, nu, nt, 100*nc/max(nt,1), 100*nu/max(nt,1), p_binom_fdr(ar,st));
    end
end

%% ========================================================================
% Cross-state flip score (SIDE)
% =========================================================================
utils_diary(fid_log, '\n================================================================\n');
utils_diary(fid_log, '  CROSS-STATE FLIP SCORE (SIDE)\n');
utils_diary(fid_log, '  Does a neuron switch chosen<->unchosen side coding across states?\n');
utils_diary(fid_log, '================================================================\n');

is_chosen_st1_side = (sign_1fc == sign_2afc(:,1));
is_chosen_st2_side = (sign_1fc == sign_2afc(:,2));
valid_both_side    = ~isnan(sign_2afc(:,1)) & ~isnan(sign_2afc(:,2));

flip_n_total_side  = NaN(nAreas, 1);
flip_n_flip_side   = NaN(nAreas, 1);
flip_prop_side     = NaN(nAreas, 1);
flip_p_binom_side  = NaN(nAreas, 1);
flip_p_indep_side  = NaN(nAreas, 1);
flip_exp_rate_side = NaN(nAreas, 1);
flip_n_c2u_side    = NaN(nAreas, 1);
flip_n_u2c_side    = NaN(nAreas, 1);
flip_n_cc_side     = NaN(nAreas, 1);
flip_n_uu_side     = NaN(nAreas, 1);

for ar = 1 : nAreas
    idx = find(strcmp(ar_label, area2test_name{ar}) & valid_both_side);
    n_tot = length(idx);
    flip_n_total_side(ar) = n_tot;
    if n_tot < 2, continue; end

    c1 = is_chosen_st1_side(idx);
    c2 = is_chosen_st2_side(idx);
    n_flip = sum(c1 ~= c2);

    flip_n_flip_side(ar) = n_flip;
    flip_prop_side(ar)   = n_flip / n_tot;
    flip_n_c2u_side(ar)  = sum(c1 == 1 & c2 == 0);
    flip_n_u2c_side(ar)  = sum(c1 == 0 & c2 == 1);
    flip_n_cc_side(ar)   = sum(c1 == 1 & c2 == 1);
    flip_n_uu_side(ar)   = sum(c1 == 0 & c2 == 0);

    flip_p_binom_side(ar) = min(1, 2 * min(1 - binocdf(n_flip-1, n_tot, 0.5), ...
                                            binocdf(n_flip, n_tot, 0.5)));

    p1 = prop(ar,1);  p2 = prop(ar,2);
    p_exp = p1*(1-p2) + (1-p1)*p2;
    flip_exp_rate_side(ar) = p_exp;
    if p_exp > 0 && p_exp < 1
        flip_p_indep_side(ar) = min(1, 2 * min(1 - binocdf(n_flip-1, n_tot, p_exp), ...
                                                binocdf(n_flip, n_tot, p_exp)));
    end
end

valid_fp = ~isnan(flip_p_binom_side);
flip_p_fdr_side = NaN(nAreas, 1);
if sum(valid_fp) > 1
    [~, ~, adj_fp] = utils_fdr_bh(flip_p_binom_side(valid_fp));
    flip_p_fdr_side(valid_fp) = adj_fp;
end

valid_ip = ~isnan(flip_p_indep_side);
flip_p_indep_fdr_side = NaN(nAreas, 1);
if sum(valid_ip) > 1
    [~, ~, adj_ip] = utils_fdr_bh(flip_p_indep_side(valid_ip));
    flip_p_indep_fdr_side(valid_ip) = adj_ip;
end

utils_diary(fid_log, '\n%-8s  N_tot  Flip  P(flip)  Exp_indep  C->U  U->C  C->C  U->U  p_50%%_FDR  p_indep_FDR\n', 'Area');
for ar = 1 : nAreas
    utils_diary(fid_log, '%-8s  %4d   %4d  %5.1f%%    %5.1f%%   %4d  %4d  %4d  %4d  %.4g     %.4g\n', ...
        area2test_name{ar}, flip_n_total_side(ar), flip_n_flip_side(ar), ...
        100*flip_prop_side(ar), 100*flip_exp_rate_side(ar), ...
        flip_n_c2u_side(ar), flip_n_u2c_side(ar), flip_n_cc_side(ar), flip_n_uu_side(ar), ...
        flip_p_fdr_side(ar), flip_p_indep_fdr_side(ar));
end

idx_pool_side = find(valid_both_side);
n_flip_pool = sum(is_chosen_st1_side(idx_pool_side) ~= is_chosen_st2_side(idx_pool_side));
n_tot_pool  = length(idx_pool_side);
p_flip_50   = min(1, 2 * min(1 - binocdf(n_flip_pool-1, n_tot_pool, 0.5), ...
                              binocdf(n_flip_pool, n_tot_pool, 0.5)));
p1_pool = mean(is_chosen_st1_side(idx_pool_side));
p2_pool = mean(is_chosen_st2_side(idx_pool_side));
p_exp_pool = p1_pool*(1-p2_pool) + (1-p1_pool)*p2_pool;
p_flip_ind = min(1, 2 * min(1 - binocdf(n_flip_pool-1, n_tot_pool, p_exp_pool), ...
                             binocdf(n_flip_pool, n_tot_pool, p_exp_pool)));
utils_diary(fid_log, '\nPooled: %d / %d flip (%.1f%%), expected under indep = %.1f%%\n', ...
    n_flip_pool, n_tot_pool, 100*n_flip_pool/n_tot_pool, 100*p_exp_pool);
utils_diary(fid_log, '  vs 50%%: p = %.4g  |  vs independence: p = %.4g\n', p_flip_50, p_flip_ind);

n_cu_side = sum(is_chosen_st1_side(idx_pool_side) == 1 & is_chosen_st2_side(idx_pool_side) == 0);
n_uc_side = sum(is_chosen_st1_side(idx_pool_side) == 0 & is_chosen_st2_side(idx_pool_side) == 1);
if (n_cu_side + n_uc_side) > 0
    mcnemar_chi2 = (n_cu_side - n_uc_side)^2 / (n_cu_side + n_uc_side);
    mcnemar_p = 1 - chi2cdf(mcnemar_chi2, 1);
    utils_diary(fid_log, 'McNemar (C->U vs U->C): %d vs %d, chi2 = %.2f, p = %.4g\n', ...
        n_cu_side, n_uc_side, mcnemar_chi2, mcnemar_p);
end

obs_flip_side = [flip_n_flip_side, flip_n_total_side - flip_n_flip_side];
keep_side = flip_n_total_side > 0 & ~isnan(flip_n_flip_side);
if sum(keep_side) > 1
    obs_f = obs_flip_side(keep_side,:);
    row_s = sum(obs_f,2);  col_s = sum(obs_f,1);  grand = sum(col_s);
    exp_f = row_s * col_s / grand;
    chi2_flip = sum((obs_f(:) - exp_f(:)).^2 ./ exp_f(:));
    df_flip = (size(obs_f,1)-1) * (size(obs_f,2)-1);
    p_chi2_flip = 1 - chi2cdf(chi2_flip, df_flip);
    utils_diary(fid_log, 'Chi-squared (flip rate across areas): chi2 = %.2f, df = %d, p = %.4g\n', ...
        chi2_flip, df_flip, p_chi2_flip);
end

%% ========================================================================
% Subplot (2,3,6): Side flip rate
% =========================================================================
subplot(2,3,6); hold on;
for ar = 1 : nAreas
    bar(ar, flip_prop_side(ar), 0.7, 'FaceColor', colorareas(ar,:), 'EdgeColor','none');
    text(ar, 0.03, sprintf('n=%d',flip_n_total_side(ar)), 'Rotation',90,'HorizontalAlignment','left','VerticalAlignment','middle','FontSize',12,'Color',[1 1 1]);
    if ~isnan(flip_p_fdr_side(ar)) && flip_p_fdr_side(ar) < 0.05
        text(ar, flip_prop_side(ar)+0.025, '*', 'HorizontalAlignment','center','FontSize',21,'FontWeight','bold');
    end
    if ~isnan(flip_exp_rate_side(ar))
        plot(ar + [-0.3 0.3], [flip_exp_rate_side(ar) flip_exp_rate_side(ar)], '-', ...
            'Color', colorareas(ar,:)*0.5, 'LineWidth', 1.5);
    end
end
yline(0.5,'--','Color',[0.5 0.5 0.5],'LineWidth',1);
set(gca,'XTick',1:nAreas,'XTickLabel',area2test_name,'FontSize',16); xtickangle(90);
ylim([0 1]); ylabel('Cross-state flip proportion');
title('Side: cross-state flip rate','FontSize',18); box off;

%% ========================================================================
% Subplot (2,3,5): Side scatter - sign consistency vs generalization
% =========================================================================
subplot(2,3,5); hold on;
if exist(unified_file,'file') && ~isempty(who('-file',unified_file,'all_contrasts'))
    uni_side = load(unified_file,'all_contrasts','param');
    p_idx_side = find(strcmp(uni_side.param.param2decode,'chosenside_2AFC'));
    if isempty(p_idx_side)
        warning('chosenside_2AFC not found in param2decode - skipping scatter (side).');
        text(0.5,0.5,'data not found','HorizontalAlignment','center','Units','normalized');
    else
        genrz_model  = uni_side.all_contrasts{p_idx_side, 1}.genrz_est;
        genrz_se_mod = uni_side.all_contrasts{p_idx_side, 1}.genrz_se;
        sign_consist_avg_side = nanmean(prop(:,1:2), 2);
        for ar = 1 : nAreas
            scatter(genrz_model(ar), sign_consist_avg_side(ar), 100, colorareas(ar,:), 'filled','MarkerEdgeColor',[0 0 0],'LineWidth',0.8);
            plot([genrz_model(ar)-1.96*genrz_se_mod(ar), genrz_model(ar)+1.96*genrz_se_mod(ar)], ...
                 [sign_consist_avg_side(ar), sign_consist_avg_side(ar)], '-','Color',[0.6 0.6 0.6],'LineWidth',0.8);
            text(genrz_model(ar)+0.003, sign_consist_avg_side(ar)+0.008, area2test_name{ar},'FontSize',15,'FontWeight','bold','Color',colorareas(ar,:));
        end
        xline(0,'--','Color',[0.7 0.7 0.7]); yline(0.5,'--','Color',[0.7 0.7 0.7]);
        xlabel('Generalization (cross-state CCGP) - 0.5 (LME at N=200)');
        ylabel('Sign consistency (avg Chosen & Unchosen)');
        title('Side: cross-state decoding vs sign preservation');
        set(gca,'FontSize',17); box off;
        valid = ~isnan(genrz_model) & ~isnan(sign_consist_avg_side);
        if sum(valid) > 3
            [r_corr,p_corr] = corr(genrz_model(valid), sign_consist_avg_side(valid));
            text(0.05,0.95,sprintf('r = %.3f, p = %.3f',r_corr,p_corr),'Units','normalized','FontSize',16,'VerticalAlignment','top');
        end
    end
else
    warning('states_2afc_ccgp.mat not found - skipping generalization scatter (side).');
    text(0.5,0.5,'data not found','HorizontalAlignment','center','Units','normalized');
end

sgtitle('Fig S11 - Sign consistency (col 1) | Generalization (col 2) | Flip rate (col 3)','FontSize',18);
exportgraphics(fig_combined, [report_dir 'Fig_S11_neurons.pdf'], 'ContentType', 'vector');

disp(' ')
disp('Section B (side) done.')
disp('All Fig S11 analyses complete.')

diary off;
utils_diary(fid_log, 'Figures saved to: %s\n', report_dir);
