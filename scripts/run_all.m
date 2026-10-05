%% run_all.m
%
% Pipeline runner for the analyses reported in the manuscript.
% - Clears the report folders written by these scripts before running.
% - Executes the main_*.m scripts below in order (later scripts load the
%   processed/ files written by earlier ones).
% - Skips main_000_create_dataset.m (builds data_final/*_pool.mat) and the
%   supp_*.m scripts (extra analyses not reported in the manuscript).
% - Resets the random number generator before each script, so rebuilding the
%   processed/ matrices gives the same results (pseudo-populations, CV folds,
%   trial subsampling, bootstraps), whatever ran or was cached before.
% - Uses script-local path resolution (project root from this file location).

clear


f = mfilename('fullpath');
if isempty(f)
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end

scripts = {'main_000_behav.m'
           'main_001_anova_lda.m'
           'main_002_states.m'
           'main_003_crossdecoding.m'
           'main_004_unitstability.m'
           'main_005_thr_sensitivity.m'
           'main_006_lda_vs_linear.m'
           'main_007_statespace.m'};

report_dirs = {'report_000' 'report_001' 'report_002' 'report_003' ...
               'report_004' 'report_005' 'report_006' 'report_007'};

seed = 55555; % random seed, reset before each script

remove report folders written by the scripts above
for r = 1 : length(report_dirs)
    report_dir = [currentPath '\' report_dirs{r} '\'];
    if exist(report_dir,'dir')
        rmdir(report_dir,'s');
    end
end

% run all scripts in order
for i = 1:length(scripts)
    disp(['Running ' scripts{i} '...']);
    rng(seed, 'twister');
    run_script(fullfile(currentPath, 'scripts', scripts{i}));
    disp(['Finished ' scripts{i} '.']);
end

function run_script(script_path)
run(script_path);
end