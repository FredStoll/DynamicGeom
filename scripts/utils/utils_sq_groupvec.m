function mu = utils_sq_groupvec(cnames, type_val, state_val)
% Builds a contrast row vector for the full model:
%   NbStates / DurStates ~ Type * StateType + (1|Monkey) + (1|Monkey:Session)
% Reference coding (MATLAB uses first value encountered in data as reference):
%   Type ref = 'quick'   (quick rows appear first in tbl_sq)
%   StateType ref = 'Chosen' (C < O < U alphabetically)
% Inputs:
%   cnames    - lme.CoefficientNames (1 x nCoef cell)
%   type_val  - 'quick' or 'hesit'
%   state_val - 'Chosen', 'Unchosen', or 'Other'

mu = zeros(1, length(cnames));
mu(strcmp(cnames, '(Intercept)')) = 1;

if strcmp(type_val, 'hesit')
    mu(~cellfun(@isempty, regexp(cnames, '^Type_hesit$'))) = 1;
end

if strcmp(state_val, 'Other')
    mu(~cellfun(@isempty, regexp(cnames, '^StateType_Other$'))) = 1;
    if strcmp(type_val, 'hesit')
        mu(~cellfun(@isempty, regexp(cnames, 'Type_hesit:StateType_Other|StateType_Other:Type_hesit'))) = 1;
    end
elseif strcmp(state_val, 'Unchosen')
    mu(~cellfun(@isempty, regexp(cnames, '^StateType_Unchosen$'))) = 1;
    if strcmp(type_val, 'hesit')
        mu(~cellfun(@isempty, regexp(cnames, 'Type_hesit:StateType_Unchosen|StateType_Unchosen:Type_hesit'))) = 1;
    end
end
end
