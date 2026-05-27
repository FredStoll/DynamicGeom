%% run_all.m
%
% Pipeline runner for main_*.m scripts.
% - Clears report_000..report_004 before running.
% - Executes scripts/main_*.m in file-system order.
% - Skips main_000_create_dataset.m by default.
% - Uses script-local path resolution (project root from this file location).

clear


f = mfilename('fullpath');
if isempty(f) 
    currentPath = pwd; % if running section by section, use current path
else
    currentPath = fileparts(fileparts(f)); % % script run: go up from scripts/
end

% remove all report folder 
for r = 0 : 4
    report_dir = [currentPath '\report_00' num2str(r) '\'];
    if exist(report_dir,'dir')
        rmdir(report_dir,'s');
    end
end

% run all scripts in order
scripts = dir(fullfile(currentPath, 'scripts', 'main_*.m'));

for i = 1:length(scripts)   
    script_name = scripts(i).name;
    if strcmp(script_name, 'main_000_create_dataset.m')
        disp(['Skipping ' script_name '...']);
        continue;
    end
    disp(['Running ' script_name '...']);
    run_script(fullfile(currentPath, 'scripts', script_name));
    disp(['Finished ' script_name '.']);
end

function run_script(script_path)
run(script_path);
end