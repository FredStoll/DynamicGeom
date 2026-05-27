function mu = utils_groupvector(cnames, saccade, trialtype)
% Return a contrast vector matching the fixed effects coefficients
% cnames: cell array of coefficient names from lme.CoefficientNames
% saccade: 'quick' or 'hesit'
% trialtype: 'ch', 'unch', or 'na'

mu = zeros(1, length(cnames));
mu(strcmp(cnames, '(Intercept)')) = 1;

if strcmp(saccade, 'hesit')
    mu(strcmp(cnames, 'Saccade_hesit')) = 1;
end

if strcmp(trialtype, 'unch')
    mu(strcmp(cnames, 'TrialType_unch')) = 1;
elseif strcmp(trialtype, 'na')
    mu(strcmp(cnames, 'TrialType_na')) = 1;
end

if strcmp(saccade, 'hesit') && strcmp(trialtype, 'unch')
    mu(strcmp(cnames, 'Saccade_hesit:TrialType_unch')) = 1;
elseif strcmp(saccade, 'hesit') && strcmp(trialtype, 'na')
    mu(strcmp(cnames, 'Saccade_hesit:TrialType_na')) = 1;
end
end
