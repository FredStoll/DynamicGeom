%% ============================================================
% main_006_lda_vs_linear.m
% ============================================================
% Categorical (LDA) vs continuous (linear) decoding of reward probability
% on 1FC trials (Figure S4).
% - Uses the 10-fold cross-validated predictions stored in out_all.decoding_cmp
%   by utils_decoding_crosstask_rmvarea (run through main_002_states.m):
%   diagonal-covariance LDA vs OLS linear regression, whose continuous
%   predictions are mapped to probability levels using class-specific
%   Gaussian PDFs (Rich & Wallis, 2016).
% - Averages decoded distributions, confusion matrices and error-by-distance
%   profiles per session, and compares decoders with an LME
%   (Pct ~ Decoder*Distance + (1|Monkey) + (1|Monkey:Session)).
% - Writes report_006/output_006.txt and report_006/Fig_S4_lda_vs_linear.pdf.
% Dependencies: utils_diary, utils_fdr_bh
% ============================================================

clear

f = mfilename('fullpath');
if isempty(f)
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end
addpath(genpath([currentPath '\scripts\'])); % add scripts folder to path

path2proc = [currentPath '\processed\'];

report_dir = [currentPath '\report_006\'];
if ~exist(report_dir, 'dir'), mkdir(report_dir); end
fid_log = fopen([report_dir 'output_006.txt'], 'w');
log_cleanup = onCleanup(@() fclose(fid_log));
utils_diary(fid_log, '\n============================================================\n');
utils_diary(fid_log, 'Report generated: %s\n', datestr(now));

% decoding_cmp (1FC trials only) is stored in states_2afc_final.mat by main_002_states.m
load([path2proc 'states_2afc_final.mat'], 'out_all', 'param')

% keep only the sessions without dropping any area
out = out_all(cellfun(@isempty, {out_all(:).area4unit_removed}));

%% FIG S4 - LDA vs linear decoder, averaged per session

n_classes  = length(param.labels);
max_dist   = n_classes - 1;
n_sessions = length(out);

bin_edges   = -25:2.5:125;
bin_centers = bin_edges(1:end-1) + diff(bin_edges)/2;

% Pre-allocate per-session matrices
sess_hist_counts = nan(n_sessions, n_classes, length(bin_centers));
sess_conf_lda    = nan(n_sessions, n_classes, n_classes);
sess_conf_lin    = nan(n_sessions, n_classes, n_classes);
sess_pct_lda     = nan(n_sessions, max_dist + 1);
sess_pct_lin     = nan(n_sessions, max_dist + 1);
valid_sessions   = false(n_sessions, 1);

% 1. Compute metrics per session
for u = 1:n_sessions
    if isfield(out(u), 'decoding_cmp') && ~isempty(out(u).decoding_cmp)
        cmp = out(u).decoding_cmp;
        
        y_cat   = [];
        lda_cat = [];
        lin_raw = [];
        lin_cat = [];
        
        % Aggregate fold results for session u
        for k = 1:length(cmp)
            y_cat   = [y_cat;   double(cmp(k).y_test_cat)];
            lda_cat = [lda_cat; double(cmp(k).pred_lda_cat)];
            lin_raw = [lin_raw; double(cmp(k).pred_lin_raw)];
            lin_cat = [lin_cat; double(cmp(k).pred_lin_cat)];
        end
        
        if isempty(y_cat), continue; end
        valid_sessions(u) = true;
        
        % --- Distribution counts for session u ---
        for c = 1:n_classes
            idx = (y_cat == c);
            if any(idx)
                sess_hist_counts(u, c, :) = histcounts(lin_raw(idx), bin_edges);
            end
        end
        
        % --- Confusion Matrices (Row-Normalized % per session) ---
        cm_lda = confusionmat(y_cat, lda_cat, 'Order', 1:n_classes);
        cm_lin = confusionmat(y_cat, lin_cat, 'Order', 1:n_classes);
        
        row_sum_lda = sum(cm_lda, 2); row_sum_lda(row_sum_lda == 0) = 1;
        row_sum_lin = sum(cm_lin, 2); row_sum_lin(row_sum_lin == 0) = 1;
        
        sess_conf_lda(u, :, :) = (cm_lda ./ row_sum_lda) * 100;
        sess_conf_lin(u, :, :) = (cm_lin ./ row_sum_lin) * 100;
        
        % --- Error Decay Profile per session ---
        dist_lda = abs(lda_cat - y_cat);
        dist_lin = abs(lin_cat - y_cat);
        
        for d = 0:max_dist
            sess_pct_lda(u, d+1) = mean(dist_lda == d) * 100;
            sess_pct_lin(u, d+1) = mean(dist_lin == d) * 100;
        end
    end
end

% 2. Average across valid sessions
n_valid = sum(valid_sessions);
mean_hist_counts = squeeze(mean(sess_hist_counts(valid_sessions, :, :), 1, 'omitnan'));
mean_conf_lda    = squeeze(mean(sess_conf_lda(valid_sessions, :, :), 1, 'omitnan'));
mean_conf_lin    = squeeze(mean(sess_conf_lin(valid_sessions, :, :), 1, 'omitnan'));

mean_pct_lda = mean(sess_pct_lda(valid_sessions, :), 1, 'omitnan');
mean_pct_lin = mean(sess_pct_lin(valid_sessions, :), 1, 'omitnan');

sem_pct_lda = std(sess_pct_lda(valid_sessions, :), 0, 1, 'omitnan') / sqrt(n_valid);
sem_pct_lin = std(sess_pct_lin(valid_sessions, :), 0, 1, 'omitnan') / sqrt(n_valid);

% 2b. STATS: LDA vs Linear error decay profile (paired across sessions)
% One LME on the whole profile: decoder x distance offset, with monkey and
% session-within-monkey as random effects. The session term is what makes this
% paired, since both decoders are run on the same trials of the same session.
% Per-offset comparisons are contrasts on this same model (same approach as the
% posthocs further down), so there is only ever one model fit here.
sess_ids  = find(valid_sessions);
n_sess_v  = length(sess_ids);

monks_cmp = cell(n_sess_v, 1);
for s = 1:n_sess_v
    monks_cmp{s} = out(sess_ids(s)).session(1); % monkey initial from session name
end
sess_num_cmp = zeros(n_sess_v, 1);
mk_u = unique(monks_cmp);
for m = 1:length(mk_u)
    idx_mk = strcmp(monks_cmp, mk_u{m});
    sess_num_cmp(idx_mk) = 1:sum(idx_mk); % session number within monkey
end

pct_lda_v = sess_pct_lda(valid_sessions, :);
pct_lin_v = sess_pct_lin(valid_sessions, :);
dist_tmp  = repmat(0:max_dist, n_sess_v, 1);

Pct      = [pct_lda_v(:); pct_lin_v(:)];
Decoder  = [repmat({'LDA'}, numel(pct_lda_v), 1); repmat({'Linear'}, numel(pct_lin_v), 1)];
Distance = categorical([dist_tmp(:); dist_tmp(:)]);
Monkey   = repmat(monks_cmp, 2*(max_dist+1), 1);
Session  = categorical(repmat(sess_num_cmp, 2*(max_dist+1), 1));

tbl_decay = table(Pct, Decoder, Distance, Monkey, Session);
tbl_decay = tbl_decay(~isnan(tbl_decay.Pct), :);

lme_decay = fitlme(tbl_decay, 'Pct ~ 1 + Decoder*Distance + (1|Monkey) + (1|Monkey:Session)');
utils_diary(fid_log, '\n=== LME: Error decay profile, LDA vs Linear (%d sessions) ===\n', n_sess_v);
utils_diary(fid_log, '%s', evalc('disp(lme_decay.Coefficients)'));
utils_diary(fid_log, '%s', evalc('anova(lme_decay)'));

% Posthoc: Linear - LDA at each offset, as contrasts on the model above (FDR-corrected)
% Reference coding: Decoder ref = 'LDA', Distance ref = '0'.
cnames_dec = lme_decay.CoefficientNames;
beta_dec   = fixedEffects(lme_decay);
covB_dec   = lme_decay.CoefficientCovariance;
dof_dec    = lme_decay.DFE;

ph_est_decay   = NaN(max_dist+1, 1); % Linear - LDA, in % of trials
ph_tstat_decay = NaN(max_dist+1, 1);
ph_pval_decay  = NaN(max_dist+1, 1);
for d = 0:max_dist
    C = zeros(1, length(cnames_dec));
    C(strcmp(cnames_dec, 'Decoder_Linear')) = 1;
    if d > 0
        pat = sprintf('^(Decoder_Linear:Distance_%d|Distance_%d:Decoder_Linear)$', d, d);
        C(~cellfun(@isempty, regexp(cnames_dec, pat))) = 1;
    end
    ph_est_decay(d+1)   = C * beta_dec;
    se_d                = sqrt(C * covB_dec * C');
    ph_tstat_decay(d+1) = ph_est_decay(d+1) / se_d;
    ph_pval_decay(d+1)  = 2 * (1 - tcdf(abs(ph_tstat_decay(d+1)), dof_dec));
end
[~, ~, ph_adj_decay] = utils_fdr_bh(ph_pval_decay);

utils_diary(fid_log, '=== Posthoc: LDA vs Linear at each distance offset (FDR-corrected) ===\n');
for d = 0:max_dist
    utils_diary(fid_log, '  Offset %d: LDA=%5.1f%%  Linear=%5.1f%%  diff(Lin-LDA)=%6.2f  t=%6.2f  p=%.4f  (FDR adj=%.4f)\n', ...
        d, mean_pct_lda(d+1), mean_pct_lin(d+1), ph_est_decay(d+1), ...
        ph_tstat_decay(d+1), ph_pval_decay(d+1), ph_adj_decay(d+1));
end

% 3. Plotting
figure('Position', [50, 100, 1600, 380]);
colors = lines(n_classes);

% --- SUBPLOT 1 (Fig S4A): Mean Continuous Distributions ---
subplot(1, 4, 1); hold on;
for c = 1:n_classes
    plot(bin_centers, mean_hist_counts(c, :), 'Color', colors(c,:), 'LineWidth', 2);
end
xlabel('Linear Decoded Value'); ylabel('Mean Trials / Session');
title('Continuous Decoded Distributions');
legend(arrayfun(@(x) [num2str(x) '%'], param.labels, 'UniformOutput', false), 'Location', 'northwest');
set(gca, 'FontSize', 16);
box off; grid on;

% --- SUBPLOT 2 (Fig S4B, left): Mean LDA Confusion Matrix ---
subplot(1, 4, 2);
imagesc(mean_conf_lda, [0 50]); colormap(hot); colorbar; axis square;
set(gca, 'XTick', 1:n_classes, 'YTick', 1:n_classes, 'XTickLabel', param.labels, 'YTickLabel', param.labels, 'FontSize', 16);
xlabel('Predicted Category'); ylabel('Ground Truth');
title(sprintf('LDA Matrix (Mean %d Sess)', n_valid));

for i = 1:n_classes
    for j = 1:n_classes
        text(j, i, sprintf('%.1f', mean_conf_lda(i,j)), 'HorizontalAlignment', 'center', ...
            'Color', 'k', 'FontSize', 15, 'FontWeight', 'bold');
    end
end

% --- SUBPLOT 3 (Fig S4B, right): Mean Linear Confusion Matrix ---
subplot(1, 4, 3);
imagesc(mean_conf_lin, [0 50]); colormap(hot); colorbar; axis square;
set(gca, 'XTick', 1:n_classes, 'YTick', 1:n_classes, 'XTickLabel', param.labels, 'YTickLabel', param.labels, 'FontSize', 16);
xlabel('Predicted Category'); ylabel('Ground Truth');
title(sprintf('Linear Matrix (Mean %d Sess)', n_valid));

for i = 1:n_classes
    for j = 1:n_classes
        text(j, i, sprintf('%.1f', mean_conf_lin(i,j)), 'HorizontalAlignment', 'center', ...
            'Color', 'k', 'FontSize', 15, 'FontWeight', 'bold');
    end
end

% --- SUBPLOT 4 (Fig S4C): Error Decay Profile with SEM ---
subplot(1, 4, 4); hold on;
errorbar(0:max_dist, mean_pct_lda, sem_pct_lda, '-o', 'Color', [0.2 0.6 0.8], 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.2 0.6 0.8]);
errorbar(0:max_dist, mean_pct_lin, sem_pct_lin, '-s', 'Color', [0.8 0.4 0.2], 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.8 0.4 0.2]);

xlabel('Distance Offset (Categories Away)'); ylabel('% of Trials');
title('Error Decay Profile (\pmSEM)');
set(gca, 'XTick', 0:max_dist, 'XTickLabel', {'Diag (0)', 'Off-1', 'Off-2', 'Off-3', 'Off-4'}, 'FontSize', 16);
legend({'Categorical (LDA)', 'Continuous (Linear)'}, 'Location', 'northeast');
grid off;
xlim([-.5 max_dist+.5]);
% Significance markers for LDA vs Linear at each offset (FDR-corrected)
yl = ylim;
for d = 0:max_dist
    if ph_adj_decay(d+1) < 0.001,     star = '***';
    elseif ph_adj_decay(d+1) < 0.01,  star = '**';
    elseif ph_adj_decay(d+1) < 0.05,  star = '*';
    else,                             star = 'n.s.';
    end
    y_top = max(mean_pct_lda(d+1) + sem_pct_lda(d+1), mean_pct_lin(d+1) + sem_pct_lin(d+1));
    text(d, y_top + 0.05*diff(yl), star, 'HorizontalAlignment', 'center', 'FontSize', 14);
end
ylim([yl(1) yl(2) + 0.08*diff(yl)]);

% Save figure
exportgraphics(gcf, [report_dir 'Fig_S4_lda_vs_linear.pdf'], 'ContentType', 'vector');
