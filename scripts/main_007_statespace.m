%% main_007_statespace.m
%
% Figure S10: cross-validated state-space geometry. Replaces the in-sample
% projection of supp_007_statespace_lda.m.
%
% In S10 the Chosen and Unchosen LDA axes are fitted on the same condition
% means that are then projected onto them. With 200 neurons, the noise part
% of each axis is nearly orthogonal to the other axis, so a population
% without any coding still shows two separated, near-perpendicular pairs.
% Each S10 panel is also scaled on its own, which hides differences in
% coding strength between areas. This script draws the same geometry
% without these issues (monkeys pooled, 200 neurons as in S10 and main_003).
%
% Cross-validated plane: axes from one half of the trials, condition means
% from the other half (both directions, averaged over n_split random
% splits), in whitened units (d'), same scale for all areas. Each panel is
% rotated within the plane so that the held-out Chosen separation lies
% along x (distances and angles are unchanged). The noise-corrected coding
% strengths (d'_C, d'_U) are given in each panel.
%
% Whitening: each neuron is divided by sqrt(pv + alpha*mean(pv)), where pv
% is its pooled within-cell variance over the four state x condition cells.
% Euclidean distances then match the regularized diagonal LDA of main_003,
% and projections on unit vectors are in single-trial SD (d') units.
%
% Split-half noise correction: with independent trial halves A and B,
%   E[dA . dB] = |d|^2   and   E[dC_A . dU_B] = dC . dU,
% so the norms and dot products below are unbiased by trial noise. The
% halves are the same trials for both states (each trial has a Chosen and
% an Unchosen firing rate), so cross-state terms always pair A with B.
%
% The log also reports the noise-corrected coding strengths and the angle of
% the supp_007 method on the data and after shuffling condition labels.
%
% Outputs:
%   report_007/Fig_S10_cvplane.pdf             A: chosen flavor, B: response side
%   report_007/output_007.txt
%   processed/states_2afc_cvplane.mat                bootstrap / shuffle results (res)

clear
overwrite_cvplane = false;   % recompute the bootstrap results instead of loading them

%% --- Parameters --------
f = mfilename('fullpath');
if isempty(f)
    currentPath = pwd;
else
    currentPath = fileparts(fileparts(f));
end
addpath(genpath([currentPath '\scripts\']));

path2proc  = [currentPath '\processed\'];
report_dir = [currentPath '\report_007\'];
if ~exist(report_dir, 'dir'), mkdir(report_dir); end
savefile   = [path2proc 'states_2afc_cvplane.mat'];

param.param2decode = {'chosenflavor_2AFC'  'chosenside_2AFC'};
var_labels = {'Chosen flavor' 'Response side'};
minTr     = 20;     % min trials per (state x condition) cell, as in supp_007
N_units   = 200;    % neurons per bootstrap draw, as in S10 / main_003
n_rep     = 200;    % bootstrap draws (neurons with replacement + trial split)
n_null    = 100;    % label-shuffle draws
n_split   = 10;     % random trial splits averaged within each draw
alpha_reg = 0.01;   % regularization, as in main_003
bias_thr  = 0;

area2test_name = {'MFC' 'PMC' 'dlPFC' 'IFG' 'vlPFC' 'AI' 'OFC' 'STR' 'AMG'};
nAreas = length(area2test_name);
nP     = length(param.param2decode);
hm_order = [1 2 3 9 8 4 7 6 5];   % S10 panel order

colorareas = [230 171   2 ; 152  78 163 ; 237  87  90 ; 252 141  98 ; 141 160 203 ;
              166 216  84 ; 102 194 165 ; 180 180 180 ; 231 138 195] / 255;
col_state = [100 200 160 ;    % Chosen   = green
             240  90  90] / 255;  % Unchosen = red
col_null  = [0.3 0.3 0.3];   % shuffle-null circle

fid_log     = fopen([report_dir 'output_007.txt'], 'w');
log_cleanup = onCleanup(@() fclose(fid_log));
utils_diary(fid_log, 'main_007_statespace  —  %s\n', datestr(now));

if ~exist(savefile, 'file') || overwrite_cvplane

    %% --- Load FR cache and build per-unit trial matrices -------
    fprintf('Loading FR cache ...\n')
    load([path2proc 'states_2afc_fr.mat'], 'all_fr_stacked', 'all_behav_stacked', 'sess_list')
    load([path2proc 'behav_pref.mat'])

    pref = NaN(height(all_behav_stacked), 1);
    for s = 1 : length(sess_list)
        pref(ismember(all_behav_stacked.Session, preference.session(s))) = ...
            preference.bias_point(s);
    end
    diff_label = all_behav_stacked.chosenflavor_2AFC ~= all_behav_stacked.unchosenflavor_2AFC;
    row_nan_ok = ~isnan(all_fr_stacked(:,1)) & ~isnan(all_fr_stacked(:,2));
    row_pref   = diff_label & abs(pref) > bias_thr;

    % dat{p,ar}.T : unit x trial x state (1 Chosen, 2 Unchosen) x condition, NaN-padded
    dat = cell(nP, nAreas);
    for p = 1 : nP
        for ar = 1 : nAreas
            row_ok = row_nan_ok & row_pref & ismember(all_behav_stacked.Area, area2test_name{ar});
            fr   = double(all_fr_stacked(row_ok, [1 2]));
            unit = all_behav_stacked.Unit(row_ok);
            cval = all_behav_stacked.(param.param2decode{p})(row_ok);

            cond_vals = sort(unique(cval));
            if numel(cond_vals) ~= 2, continue; end

            [~, ~, g] = unique(unit);
            ci   = 1 + (cval == cond_vals(2));
            cnt  = accumarray([g ci], 1);
            keep = find(all(cnt >= minTr, 2));
            if numel(keep) < N_units / 2, continue; end

            T = NaN(numel(keep), max(max(cnt(keep, :))), 2, 2);
            for k = 1 : numel(keep)
                r = g == keep(k);
                for c = 1 : 2
                    v = fr(r & ci == c, :);
                    T(k, 1:size(v,1), :, c) = reshape(v, 1, [], 2);
                end
            end

            % Whitening SD: pooled within-cell variance over the 4 cells
            n  = reshape(sum(~isnan(T(:,:,1,:)), 2), [], 2);
            pv = zeros(numel(keep), 1);
            for s = 1 : 2
                for c = 1 : 2
                    pv = pv + (n(:,c) - 1) .* var(T(:,:,s,c), 0, 2, 'omitnan');
                end
            end
            pv = pv ./ (2 * sum(n, 2) - 4);

            dat{p,ar}.T         = T;
            dat{p,ar}.sdw       = sqrt(pv + alpha_reg * mean(pv));
            dat{p,ar}.cond_vals = cond_vals;
            fprintf('  %s %s: %d units\n', param.param2decode{p}, area2test_name{ar}, numel(keep));
        end
    end
    clear all_fr_stacked all_behav_stacked pref diff_label row_nan_ok row_pref

    %% --- Bootstrap / shuffle draws -----
    disp('Running bootstrap ...')
    res = cell(nP, nAreas);
    for p = 1 : nP
        for ar = 1 : nAreas
            if isempty(dat{p,ar}), continue; end
            nU = size(dat{p,ar}.T, 1);

            R = init_store(n_rep);
            for b = 1 : n_rep
                R = put_rep(R, b, one_rep(dat{p,ar}.T, dat{p,ar}.sdw, randi(nU, N_units, 1), false, alpha_reg, n_split));
            end
            Rn = init_store(n_null);
            for b = 1 : n_null
                Rn = put_rep(Rn, b, one_rep(dat{p,ar}.T, dat{p,ar}.sdw, randi(nU, N_units, 1), true, alpha_reg, n_split));
            end

            R.nUnits  = nU;
            R.summ    = summarize(R);
            res{p,ar} = R;
            res{p,ar}.null = Rn;
            fprintf('  %s %s done\n', param.param2decode{p}, area2test_name{ar});
        end
    end

    cv_param = struct('param2decode', {param.param2decode}, 'minTr', minTr, 'N_units', N_units, ...
        'n_rep', n_rep, 'n_null', n_null, 'n_split', n_split, 'alpha_reg', alpha_reg, 'bias_thr', bias_thr);
    save(savefile, 'res', 'cv_param', 'area2test_name', '-v7.3');
    fprintf('Saved: %s\n', savefile);

else
    fprintf('Loading existing results: %s\n', savefile)
    load(savefile, 'res', 'cv_param')
end

utils_diary(fid_log, 'N = %d neurons per draw, %d bootstrap draws, %d shuffle draws, %d trial splits per draw, minTr = %d\n', ...
    cv_param.N_units, cv_param.n_rep, cv_param.n_null, cv_param.n_split, cv_param.minTr);
utils_diary(fid_log, 'Results: %s\n', savefile);

%% --- Log: noise-corrected geometry per area -----------------------
for p = 1 : nP
    utils_diary(fid_log, '\n=== %s ===\n', var_labels{p});
    utils_diary(fid_log, ['%-6s %5s | %6s %6s | %16s %16s | %16s | %13s %13s\n'], 'Area', 'nPool', ...
        'dC''', 'dU''', 'U par (x dC)', 'U perp (x dC)', 'cos (noise-corr)', 'th in-sample', 'th null');
    for ar = 1 : nAreas
        if isempty(res{p,ar}), continue; end
        S = res{p,ar}.summ;
        utils_diary(fid_log, '%-6s %5d | %6.2f %6.2f | %5.2f [%5.2f %5.2f] %5.2f [%5.2f %5.2f] | %5.2f [%5.2f %5.2f] | %5.1f [%4.1f %4.1f] %5.1f [%4.1f %4.1f]\n', ...
            area2test_name{ar}, res{p,ar}.nUnits, S.nC(1), S.nU(1), ...
            S.r_par, S.r_perp, S.cosnc, ...
            prctile(res{p,ar}.theta_ins, [50 2.5 97.5]), prctile(res{p,ar}.null.theta_ins, [50 2.5 97.5]));
    end
end
% utils_diary(fid_log, ['\ndC'', dU'': noise-corrected coding strength (d'') in the Chosen / Unchosen state.\n' ...
%     'U par, U perp: components of the Unchosen coding vector parallel / orthogonal to the Chosen one,\n' ...
%     '               in units of |dC|. (1, 0) = identical code, (g, 0) = pure gain g, (0, 1) = rotated code of equal strength.\n' ...
%     'cos: noise-corrected cosine between Chosen and Unchosen coding vectors (can exceed [-1 1] when coding is weak).\n' ...
%     'th in-sample / null: S10 angle (supp_007 method) on the data and after shuffling condition labels.\n' ...
%     'Values: median [2.5 97.5 percentiles] over bootstrap draws.\n']);

%% --- Fig S10: chosen flavor (A), response side (B) ---------------
face_alpha = 0.35;    % ellipse transparency (1 = opaque)
scale_unit = 'SD';    % scale bar unit: single-trial standard deviation (d' units)
cond_lbl   = {{'J1' 'J2'}, {'Left' 'Right'}};

W = 18;  H = 12.3;                                    % figure size (cm)
cm = @(x, y, w, h) [x y w h] ./ [W H W H];            % cm -> normalized units
fg = figure('Units', 'centimeters', 'Position', [2 2 W H], 'Color', 'w', ...
    'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial');

% 3 x 3 areas per variable
pw = 2.5;  gap = 0.25;  vgap = 0.75;  y0 = 2.0;  x0 = [1.0 9.8];
for p = 1 : nP
    % Rotate each area so the held-out Chosen separation lies along +x
    Qr = cell(nAreas, 1);  Qnr = cell(nAreas, 1);  lim = 0;
    for ar = 1 : nAreas
        if isempty(res{p,ar}), continue; end
        a       = green_angle(squeeze(mean(res{p,ar}.cvp, 1)));
        Qr{ar}  = rotate_pts(res{p,ar}.cvp, a);
        Qnr{ar} = rotate_pts(res{p,ar}.null.cvp, a);
        lim = max(lim, max(abs(squeeze(mean(Qr{ar}, 1))) + 2*squeeze(std(Qr{ar}, 0, 1)), [], 'all'));
    end
    lim = lim * 1.02;

    for hm_idx = 1 : nAreas
        ar = hm_order(hm_idx);
        r  = ceil(hm_idx / 3);  c = mod(hm_idx - 1, 3);
        ax = axes(fg, 'Position', cm(x0(p) + c*(pw + gap), y0 + (3 - r)*(pw + vgap), pw, pw));
        if isempty(res{p,ar}), axis(ax, 'off'); continue; end
        draw_panel(ax, Qr{ar}, null_radius(Qnr{ar}), lim, res{p,ar}.summ, col_state, col_null, face_alpha);
        title(ax, area2test_name{ar}, 'FontSize', 11, 'FontWeight', 'bold');
        if hm_idx == 7
            text(ax, 0, -1.06*lim, 'Chosen-state axis', 'FontSize', 9, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
            text(ax, -1.06*lim, 0, 'Orthogonal axis', 'FontSize', 9, 'Rotation', 90, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    panel_letter(fg, cm(x0(p) - 0.9, 11.2, 1, 0.8), char('A' + p - 1));
    lbl = cond_lbl{p};
    draw_marker_legend(fg, cm(x0(p) + 0.5, 0.2, 5.0, 1.1), ...
        {['Ch x ' lbl{1}], ['Ch x ' lbl{2}], ['Unch x ' lbl{1}], ['Unch x ' lbl{2}]}, col_state, 9);
    draw_scalebar(fg, cm(x0(p) + 2*(pw + gap), 0.5, pw, 0.9), lim, scale_unit);
end

fname = [report_dir 'Fig_S10_cvplane'];
exportgraphics(fg, [fname '.pdf'], 'ContentType', 'vector');   % for CorelDRAW / Illustrator

fprintf('\nmain_007 complete.\n');


%% ---- Local functions -------

function out = one_rep(T, sdw, idx, do_shuffle, alpha_reg, n_split)
% One bootstrap draw of neurons idx: in-sample S10 angle
% and split-half cross-validated plane and dot products (whitened),
% averaged over n_split random trial splits.
T   = T(idx, :, :, :);
sdw = sdw(idx);
if do_shuffle, T = shuffle_labels(T); end
nU = numel(idx);

% In-sample S10 angle, as supp_007 (cells: C-A, C-B, U-A, U-B)
n = reshape(sum(~isnan(T(:,:,1,:)), 2), [], 2);
M = NaN(nU, 4); V = NaN(nU, 4);
for s = 1 : 2
    for c = 1 : 2
        M(:, (s-1)*2+c) = mean(T(:,:,s,c), 2, 'omitnan');
        V(:, (s-1)*2+c) = var(T(:,:,s,c), 0, 2, 'omitnan');
    end
end
w = NaN(nU, 2);
for s = 1 : 2
    pv = ((n(:,1)-1).*V(:,(s-1)*2+1) + (n(:,2)-1).*V(:,(s-1)*2+2)) ./ (sum(n, 2) - 2);
    w(:,s) = (M(:,(s-1)*2+1) - M(:,(s-1)*2+2)) ./ (pv + alpha_reg * mean(pv));
end
out.theta_ins = acosd(min(1, abs(w(:,1)' * w(:,2)) / (norm(w(:,1)) * norm(w(:,2)))));

% Split-half, whitened
Z = T ./ sdw;
out.cc = 0;  out.uu = 0;  out.cu = 0;  out.cvp = zeros(4, 2);
for k = 1 : n_split
    [MA, MB] = split_means(Z);
    dA = [MA(:,1) - MA(:,2), MA(:,3) - MA(:,4)];
    dB = [MB(:,1) - MB(:,2), MB(:,3) - MB(:,4)];
    out.cc = out.cc + dA(:,1)' * dB(:,1) / n_split;
    out.uu = out.uu + dA(:,2)' * dB(:,2) / n_split;
    out.cu = out.cu + (dA(:,1)' * dB(:,2) + dA(:,2)' * dB(:,1)) / (2 * n_split);
    out.cvp = out.cvp + (project_cv(dA, MB) + project_cv(dB, MA)) / (2 * n_split);
end
end

function P = project_cv(d, Mh)
% Axes from one half (d), condition means from the other (Mh), projected on
% the orthonormal plane (Chosen axis, orthogonalized Unchosen axis).
e1 = d(:,1) / norm(d(:,1));
eu = d(:,2) / norm(d(:,2));
e2 = eu - (eu' * e1) * e1;  e2 = e2 / norm(e2);
P  = (Mh - mean(Mh, 2))' * [e1 e2];
end

function [MA, MB] = split_means(Z)
% Random half split of each unit's trials within each condition, the same
% split for both states. Returns unit x cell means (C-A, C-B, U-A, U-B).
[nU, nT, ~, ~] = size(Z);
MA = NaN(nU, 4); MB = NaN(nU, 4);
for c = 1 : 2
    valid = ~isnan(Z(:,:,1,c));
    R = rand(nU, nT);  R(~valid) = Inf;
    [~, ord] = sort(R, 2);
    [~, rk]  = sort(ord, 2);
    inA = rk <= floor(sum(valid, 2) / 2);
    inB = valid & ~inA;
    for s = 1 : 2
        X = Z(:,:,s,c);  X(~valid) = 0;
        MA(:, (s-1)*2+c) = sum(X .* inA, 2) ./ sum(inA, 2);
        MB(:, (s-1)*2+c) = sum(X .* inB, 2) ./ sum(inB, 2);
    end
end
end

function T = shuffle_labels(T)
% Permute condition labels across each unit's trials (both states move together).
for u = 1 : size(T, 1)
    n1 = sum(~isnan(T(u,:,1,1)));
    n2 = sum(~isnan(T(u,:,1,2)));
    v  = [reshape(T(u,1:n1,:,1), n1, 2); reshape(T(u,1:n2,:,2), n2, 2)];
    v  = v(randperm(n1 + n2), :);
    T(u,1:n1,:,1) = reshape(v(1:n1, :),     1, n1, 2);
    T(u,1:n2,:,2) = reshape(v(n1+1:end, :), 1, n2, 2);
end
end

function R = init_store(n)
R.cvp = NaN(n, 4, 2);
R.theta_ins = NaN(n, 1);  R.cc = NaN(n, 1);  R.uu = NaN(n, 1);  R.cu = NaN(n, 1);
end

function R = put_rep(R, b, o)
R.cvp(b,:,:) = o.cvp;
R.theta_ins(b) = o.theta_ins;  R.cc(b) = o.cc;  R.uu(b) = o.uu;  R.cu(b) = o.cu;
end

function S = summarize(R)
% Median [2.5 97.5] over draws of the noise-corrected quantities.
cc = R.cc;  uu = R.uu;  cu = R.cu;
cc(cc <= 0) = NaN;  uu(uu <= 0) = NaN;
nC = sqrt(cc);  nU = sqrt(uu);
q  = @(x) prctile(x, [50 2.5 97.5]);
S.nC     = q(nC);
S.nU     = q(nU);
S.r_par  = q(cu ./ cc);                                    % x |dC|
S.r_perp = q(sqrt(max(uu - cu.^2 ./ cc, 0)) ./ nC);       % x |dC|
S.cosnc  = q(cu ./ (nC .* nU));
end

function a = green_angle(m)
% Angle of the Chosen separation (cell 1 - cell 2) in a 4 x 2 point set.
v = m(1, :) - m(2, :);
a = atan2(v(2), v(1));
end

function Q = rotate_pts(Q, a)
% Rotate points stored with x/y in the last dimension by -a.
sz = size(Q);
Rm = [cos(a) sin(a); -sin(a) cos(a)];
Q  = reshape(reshape(Q, [], 2) * Rm', sz);
end

function rn = null_radius(Qn)
% Radius containing 95% of the shuffle-null condition means.
rn = prctile(sqrt(sum(reshape(Qn, [], 2).^2, 2)), 95);
end

function draw_panel(ax, Q, r_null, lim, S, col_state, col_null, face_alpha)
% One area: crosshair, bootstrap ellipses (2 SD), shuffle-null circle,
% condition means (draws x 4 x 2; cells C-A, C-B, U-A, U-B) and d' values.
hold(ax, 'on');
mk = {'o', '^', 'o', '^'};  si = [1 1 2 2];
plot(ax, [-lim lim], [0 0], '--', 'Color', [0.65 0.65 0.65], 'LineWidth', 0.5);
plot(ax, [0 0], [-lim lim], '--', 'Color', [0.65 0.65 0.65], 'LineWidth', 0.5);
for pt = 1 : 4
    draw_ellipse(ax, squeeze(Q(:,pt,:)), col_state(si(pt),:), face_alpha, 2);
end
draw_circle(ax, r_null, col_null);
m = squeeze(mean(Q, 1));
plot(ax, m(1:2,1), m(1:2,2), '-', 'Color', col_state(1,:), 'LineWidth', 1.2);
plot(ax, m(3:4,1), m(3:4,2), '-', 'Color', col_state(2,:), 'LineWidth', 1.2);
for pt = 1 : 4
    plot(ax, m(pt,1), m(pt,2), mk{pt}, 'MarkerSize', 4, 'MarkerFaceColor', col_state(si(pt),:), ...
        'MarkerEdgeColor', 'none');
end
text(ax, -0.95*lim, 0.95*lim, sprintf('d''_C = %.2f', S.nC(1)), 'FontSize', 7, ...
    'Color', col_state(1,:)*0.85, 'VerticalAlignment', 'top');
text(ax, -0.95*lim, 0.75*lim, sprintf('d''_U = %.2f', S.nU(1)), 'FontSize', 7, ...
    'Color', col_state(2,:)*0.9, 'VerticalAlignment', 'top');
axis(ax, 'equal');  xlim(ax, [-lim lim]);  ylim(ax, [-lim lim]);  axis(ax, 'off');
end

function draw_circle(ax, r, col)
% Dashed unfilled circle (a fill without face exports cleanly to vector formats).
t = linspace(0, 2*pi, 100);
fill(ax, r*cos(t), r*sin(t), 'w', 'FaceColor', 'none', 'EdgeColor', col, ...
    'LineStyle', '--', 'LineWidth', 0.6, 'HandleVisibility', 'off');
end

function draw_marker_legend(fg, pos, labels, col_state, fs)
% Boxed 2 x 2 legend of the 4 cells (C-A, C-B, U-A, U-B).
ax = axes(fg, 'Position', pos, 'XLim', [0 1], 'YLim', [0 1]); hold(ax, 'on');  axis(ax, 'off');
rectangle(ax, 'Position', [0 0 1 1], 'EdgeColor', 'k', 'LineWidth', 0.5);
mk = {'o', '^', 'o', '^'};  si = [1 1 2 2];
xy = [0.06 0.70; 0.06 0.28; 0.52 0.70; 0.52 0.28];
for k = 1 : 4
    plot(ax, xy(k,1), xy(k,2), mk{k}, 'MarkerSize', 5, 'MarkerFaceColor', col_state(si(k),:), ...
        'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.5);
    text(ax, xy(k,1) + 0.06, xy(k,2), labels{k}, 'FontSize', fs, 'VerticalAlignment', 'middle');
end
end

function draw_scalebar(fg, pos, lim, unit)
% Horizontal scale bar in an axes as wide as a data panel ([-lim lim]).
len = [0.25 0.5 1 2 5];
len = len(find(len <= 0.8*lim, 1, 'last'));
ax  = axes(fg, 'Position', pos, 'XLim', [-lim lim], 'YLim', [0 1]); hold(ax, 'on');  axis(ax, 'off');
plot(ax, [lim - len, lim], [0.75 0.75], 'k-', 'LineWidth', 1);
text(ax, lim - len/2, 0.55, sprintf('%g %s', len, unit), 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end

function panel_letter(fg, pos, letter)
ax = axes(fg, 'Position', pos, 'XLim', [0 1], 'YLim', [0 1]);  axis(ax, 'off');
text(ax, 0, 0.5, letter, 'FontSize', 16, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end

function draw_ellipse(ax, pts, col, face_alpha, n_std)
pts = pts(~any(isnan(pts), 2), :);
if size(pts, 1) < 10, return; end
[V, D] = eig(cov(pts));
t   = linspace(0, 2*pi, 100);
ell = V * diag(n_std * sqrt(max(diag(D), 0))) * [cos(t); sin(t)];
mu  = mean(pts, 1);
fill(ax, mu(1) + ell(1,:), mu(2) + ell(2,:), col, 'FaceAlpha', face_alpha, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
end
