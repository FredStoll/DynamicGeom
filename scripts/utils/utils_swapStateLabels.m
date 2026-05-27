function M_swapped = utils_swapStateLabels(states)
% utils_swapStateLabels  Per-trial random swap of Chosen/Unchosen state labels.
%
%   M_swapped = utils_swapStateLabels(states)
%
%   For each trial independently (Bernoulli p = 0.5), swaps state label 1
%   (Chosen) and state label 2 (Unchosen) in every bin of that trial.
%   State label 3 (Other) and any other values are left unchanged.
%
%   Rationale:
%     The swap preserves the exact time bins occupied by each HMM state
%     within every trial (same state durations, same temporal structure).
%     Only the *identity* of which bins are called "Chosen" vs "Unchosen"
%     is randomised across trials.  This creates a null distribution for
%     CCGP, shCCGP, and all derived contrasts under the hypothesis that
%     the Chosen/Unchosen distinction carries no condition-related
%     information.
%
%   Input:
%     states    - [n_trials x n_bins] integer matrix of HMM state labels
%                 (1 = Chosen, 2 = Unchosen, 3 = Other)
%
%   Output:
%     M_swapped - same size; ~50% of trials have labels 1<->2 swapped

n_trials   = size(states, 1);
swap_mask  = rand(n_trials, 1) > 0.5;   % independent Bernoulli(0.5) per trial

M_swapped  = states;

if ~any(swap_mask), return; end

rows       = find(swap_mask);
sub        = states(rows, :);
sub_new    = sub;
sub_new(sub == 1) = 2;
sub_new(sub == 2) = 1;
M_swapped(rows, :) = sub_new;

end
