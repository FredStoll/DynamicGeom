function [matched, keep_mask, stats] = utils_matchTimes(M)
% utils_matchTimes - Equalize occurrences of 1 and 2 per time bin in a matrix.
%
%   [matched, keep_mask, stats] = utils_matchTimes(M)
%
%   Input:
%       M : nbTrials x nbTimeBins matrix
%           Values:
%               NaN = no state
%               1   = ChState
%               2   = UnchState
%               3   = OtherState (ignored)
%
%   Output:
%       matched    : matrix same size as M, containing only the kept 1s and 2s
%                    (other values set to NaN)
%       keep_mask  : logical mask (true where a 1 or 2 was kept)
%       stats      : struct with fields
%                       .before: [2 x nbTimeBins] counts before matching
%                       .after : [2 x nbTimeBins] counts after matching
%
%   Description:
%       For each time bin (column), finds trials with 1 or 2, and keeps a
%       random subset so that the number of 1s and 2s are equal for that bin.
%       This controls for imbalances in the number of trials in each state
%       across the time dimension.
%
%   Complementary function: utils_matchBins equalises at the per-trial level
%       (same number of chosen/unchosen bins within each trial).
%
%   Example:
%       M = [NaN 1 2 2; 2 1 NaN 1; 1 1 2 NaN];
%       [matched, keep_mask, stats] = utils_matchTimes(M);
%
%   See also  utils_matchBins

rng('shuffle'); % for randomization (use rng(seed) for reproducibility)

[~, nbTimeBins] = size(M);
matched   = nan(size(M));
keep_mask = false(size(M));

before_counts = zeros(2, nbTimeBins);
after_counts  = zeros(2, nbTimeBins);

for t = 1 : nbTimeBins
    col = M(:, t);
    is1 = find(col == 1);
    is2 = find(col == 2);

    before_counts(1, t) = numel(is1);
    before_counts(2, t) = numel(is2);

    nKeep = min(numel(is1), numel(is2));
    if nKeep == 0
        continue
    end

    keep1   = is1(randperm(numel(is1), nKeep));
    keep2   = is2(randperm(numel(is2), nKeep));
    keepIdx = [keep1; keep2];

    matched(keepIdx, t)   = col(keepIdx);
    keep_mask(keepIdx, t) = true;

    after_counts(1, t) = nKeep;
    after_counts(2, t) = nKeep;
end

stats.before = before_counts;
stats.after  = after_counts;

end
