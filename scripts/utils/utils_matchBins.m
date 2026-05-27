function matched = utils_matchBins(M)
% utils_matchBins  Per-trial equalization of chosen / unchosen state bins.
%
%   matched = utils_matchBins(M)
%
%   For each trial (row), randomly keeps min(n_chosen_bins, n_unchosen_bins)
%   bins from the chosen state (1) AND the same number from the unchosen
%   state (2).  All other positions are set to NaN.
%
%   Nanmean over the resulting matrix therefore gives an unbiased per-trial
%   mean that cannot be driven by one state having more bins than the other
%   on a given trial.
%
%   This is complementary to utils_matchTimes (which equalises across
%   trials per time-bin).  Use utils_matchBins when you want each trial to
%   contribute exactly equal chosen/unchosen signal.
%
%   Input
%     M        [nTrials x nTimeBins]  state label matrix.
%              1 = Chosen, 2 = Unchosen, 3 = Other, NaN = no state.
%
%   Output
%     matched  [nTrials x nTimeBins]  same size as M.
%              Contains only the randomly kept 1s and 2s; all other
%              positions are NaN.  The count of 1s == count of 2s in
%              every row (or 0 for rows where either state is absent).
%
%   Typical usage in a cross-decoding loop
%     M_matched = utils_matchBins(out_all(takeme(u)).states);
%     st(M_matched ~= c_st) = NaN;
%     all_fr(:, c_st) = single(nanmean(st, 2));
%
%   See also  utils_matchTimes

% -------------------------------------------------------------------------
matched = nan(size(M));

for tr = 1 : size(M, 1)
    row  = M(tr, :);
    idx1 = find(row == 1);   % chosen-state bins in this trial
    idx2 = find(row == 2);   % unchosen-state bins in this trial

    nKeep = min(numel(idx1), numel(idx2));
    if nKeep == 0
        continue
    end

    keep1 = idx1(randperm(numel(idx1), nKeep));
    keep2 = idx2(randperm(numel(idx2), nKeep));

    matched(tr, keep1) = 1;
    matched(tr, keep2) = 2;
end
end
