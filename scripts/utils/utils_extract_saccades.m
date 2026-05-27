function [saccadeTimes, saccadeTable] = utils_extract_saccades(DATA_FOLDER, OUTPUT_FOLDER)
%UTILS_EXTRACT_SACCADES Extract saccade timing/counts from EOG sessions.
%   [saccadeTimes, saccadeTable] = utils_extract_saccades(DATA_FOLDER, OUTPUT_FOLDER)
%   Loads EOG and behavioral data to identify saccades via a velocity-based threshold.

if nargin < 1 || isempty(DATA_FOLDER)
    f = mfilename('fullpath');
    currentPath = fileparts(fileparts(f));
    DATA_FOLDER = fullfile(currentPath, 'data_final');
end
if nargin < 2 || isempty(OUTPUT_FOLDER)
    f = mfilename('fullpath');
    currentPath = fileparts(fileparts(f));
    OUTPUT_FOLDER = fullfile(currentPath, 'processed');
end

if ~exist(OUTPUT_FOLDER, 'dir'), mkdir(OUTPUT_FOLDER); end

percentile = 55;
dwellTh    = 200;

Sess_col        = {};   Trial_col       = [];
NumS_col        = [];   Times_rel_col   = {};
Sides_col       = {};

FirstLeaveFix_col  = [];   EnterTimes_col     = {};
EnterDir_col       = {};   LeaveTimes_col     = {};
LeaveDir_col       = {};

files = dir(fullfile(DATA_FOLDER, '*_EOG.mat'));
fprintf('Found %d sessions.\n', numel(files));

for f = 1:numel(files)
    sessionID = files(f).name(1:7);
    fprintf('\n=== SESSION %s ===\n', sessionID);

    sid = char(sessionID);
    aPath = fullfile(DATA_FOLDER, [sid 'a_EOG.mat']);
    bPath = fullfile(DATA_FOLDER, [sid '_EOG.mat']);

    if isfile(aPath)
        eogPath = aPath;
    elseif isfile(bPath)
        eogPath = bPath;
    else
        warning('No EOG file for %s', sessionID);
        continue;
    end

    behavPath = fullfile(DATA_FOLDER, [sid '_spk.mat']);
    if ~isfile(behavPath)
        warning('No behavior file for %s', sessionID);
        continue;
    end

    E = load(eogPath, 'eog');       eog   = E.eog(:,1:2);
    B = load(behavPath, 'behav');   behav = B.behav;

    fix_on   = behav.t_evt.fixcross_on * 1000;
    stim_on  = behav.t_evt.stim_on      * 1000;
    stim_off = behav.t_evt.stim_off     * 1000;
    time_ms  = (0:size(eog,1)-1)';

    nTrials = numel(stim_on);
    if nTrials == 0
        warning('No trials in %s', sessionID);
        continue;
    end

    fixPts = [];
    for t = 1:nTrials
        si = find(time_ms >= fix_on(t),  1, 'first');
        ei = find(time_ms <= stim_on(t), 1, 'last');
        if isempty(si) || isempty(ei) || ei <= si, continue; end
        fixPts = [fixPts; eog(si:ei,:)];
    end

    if isempty(fixPts)
        warning('No fixation samples in %s', sessionID);
        continue;
    end

    cx    = median(fixPts(:,1));
    cy    = median(fixPts(:,2));
    dAll  = hypot(fixPts(:,1)-cx, fixPts(:,2)-cy);
    radius = prctile(dAll, percentile);

    for t = 1:nTrials

        si = find(time_ms >= stim_on(t),  1, 'first');
        ei = find(time_ms <= stim_off(t), 1, 'last');

        if isempty(si) || isempty(ei) || ei <= si
            numS           = 1;
            saccTimes_rel  = 0;
            saccDirs       = "";
            firstLeaveFix  = 0;
            enterTimes_rel = [];
            enterDir       = strings(0,1);
            leaveTimes_rel = [];
            leaveDir       = strings(0,1);

            Sess_col{end+1,1}       = sessionID;
            Trial_col(end+1,1)      = t;
            NumS_col(end+1,1)       = numS;
            Times_rel_col{end+1,1}  = saccTimes_rel;
            Sides_col{end+1,1}      = saccDirs;
            FirstLeaveFix_col(end+1,1) = firstLeaveFix;
            EnterTimes_col{end+1,1} = enterTimes_rel;
            EnterDir_col{end+1,1}   = enterDir;
            LeaveTimes_col{end+1,1} = leaveTimes_rel;
            LeaveDir_col{end+1,1}   = leaveDir;
            continue;
        end

        segT  = time_ms(si:ei);
        seg   = eog(si:ei,:);
        x_seg = seg(:,1);

        d_seg   = hypot(x_seg - cx, seg(:,2) - cy);
        outside = d_seg > radius;

        anyOut = any(outside);

        if anyOut
            idxLeaveFix   = find(outside, 1, 'first');
            firstLeaveFix_abs = segT(idxLeaveFix);
        else
            firstLeaveFix_abs = stim_on(t);
        end
        firstLeaveFix = firstLeaveFix_abs - stim_on(t);

        [runsR, runsL] = detectRuns(x_seg, d_seg, cx, radius, dwellTh);

        saccTimes_abs = [];
        saccDirs      = strings(0,1);

        if isempty(runsR) && isempty(runsL)
            tFirst_abs = firstLeaveFix_abs;

            if anyOut
                if x_seg(idxLeaveFix) > cx
                    saccDirs = "R";
                elseif x_seg(idxLeaveFix) < cx
                    saccDirs = "L";
                else
                    saccDirs = "";
                end
            else
                saccDirs = "";
            end
            saccTimes_abs = tFirst_abs;
            numS          = 1;

            enterTimes_rel = [];
            enterDir       = strings(0,1);
            leaveTimes_rel = [];
            leaveDir       = strings(0,1);

        else
            segs = struct('side',{},'start',{},'stop',{});
            for i = 1:numel(runsR)
                segs(end+1) = struct('side','R', ...
                                     'start',runsR(i).start, ...
                                     'stop', runsR(i).stop);
            end
            for i = 1:numel(runsL)
                segs(end+1) = struct('side','L', ...
                                     'start',runsL(i).start, ...
                                     'stop', runsL(i).stop);
            end

            if isempty(segs)
                tFirst_abs = firstLeaveFix_abs;
                saccTimes_abs = tFirst_abs;
                numS          = 1;
                saccDirs      = "";

                enterTimes_rel = [];
                enterDir       = strings(0,1);
                leaveTimes_rel = [];
                leaveDir       = strings(0,1);

            else
                [~, ord] = sort([segs.start]);
                segs = segs(ord);

                saccIdx   = segs(1).start;
                lastSide  = segs(1).side;
                saccDirs  = string(segs(1).side);

                for i = 2:numel(segs)
                    if ~strcmp(segs(i).side, lastSide)
                        saccIdx(end+1) = segs(i).start;
                        lastSide       = segs(i).side;
                        saccDirs(end+1) = string(segs(i).side);
                    end
                end

                saccTimes_abs = segT(saccIdx(:));
                numS          = numel(saccTimes_abs);

                enterTimes_abs = [];
                enterDir       = strings(0,1);
                leaveTimes_abs = [];
                leaveDir       = strings(0,1);

                for i = 1:numel(segs)
                    thisSide = segs(i).side;

                    enterTimes_abs(end+1,1) = segT(segs(i).start);
                    enterDir(end+1,1)       = string(thisSide);

                    idxBackFix = find(~outside(segs(i).stop:end), 1, 'first');
                    if ~isempty(idxBackFix)
                        leaveIdx = segs(i).stop + idxBackFix - 1;
                        leaveTimes_abs(end+1,1) = segT(leaveIdx);
                        leaveDir(end+1,1)       = string(thisSide);
                    end
                end

                enterTimes_rel = enterTimes_abs - stim_on(t);
                leaveTimes_rel = leaveTimes_abs - stim_on(t);
            end
        end

        saccTimes_rel = saccTimes_abs - stim_on(t);

        Sess_col{end+1,1}       = sessionID;
        Trial_col(end+1,1)      = t;
        NumS_col(end+1,1)       = numS;
        Times_rel_col{end+1,1}  = saccTimes_rel(:).';
        Sides_col{end+1,1}      = saccDirs(:).';

        FirstLeaveFix_col(end+1,1) = firstLeaveFix;
        EnterTimes_col{end+1,1}    = enterTimes_rel(:).';
        EnterDir_col{end+1,1}      = enterDir(:).';
        LeaveTimes_col{end+1,1}    = leaveTimes_rel(:).';
        LeaveDir_col{end+1,1}      = leaveDir(:).';
    end
end

saccadeTimes = table( ...
    string(Sess_col), ...
    uint32(Trial_col), ...
    uint32(NumS_col), ...
    Times_rel_col, ...
    Sides_col, ...
    FirstLeaveFix_col, ...
    EnterTimes_col, ...
    EnterDir_col, ...
    LeaveTimes_col, ...
    LeaveDir_col, ...
    'VariableNames', { ...
        'SessionID','Trial','NumSaccades', ...
        'SaccadeTimes_rel','SaccadeDir', ...
        'FirstLeaveFix_rel', ...
        'EnterTimes_rel','EnterDir', ...
        'LeaveTimes_rel','LeaveDir' ...
    });

saveFlagAll = '-v7';
try
    save(fullfile(OUTPUT_FOLDER, 'saccadeCounts_all.mat'), ...
         'saccadeTimes', saveFlagAll);
catch ME
    warning('%s', sprintf('Saving saccadeCounts_all.mat with -v7 failed (%s). Falling back to -v7.3.', ME.message));
    saveFlagAll = '-v7.3';
    save(fullfile(OUTPUT_FOLDER, 'saccadeCounts_all.mat'), ...
         'saccadeTimes', saveFlagAll);
end

% save a version with only counts and no times for quick access
saccadeTable = saccadeTimes(:, {'SessionID', 'Trial', 'NumSaccades'});
save(fullfile(OUTPUT_FOLDER, 'saccadeCounts_reduced.mat'), ...
     'saccadeTable', '-v7');

fprintf('\nSaved %d rows to saccadeCounts_all.mat (%s) and saccadeCounts_reduced.mat (-v7)\n', ...
    height(saccadeTimes), saveFlagAll);

end

function [runsR, runsL] = detectRuns(x_seg, d_seg, cx, radius, dwellTh)
    isR = (d_seg > radius) & (x_seg > cx);
    isL = (d_seg > radius) & (x_seg < cx);

    runsR = logicalRuns(isR, dwellTh);
    runsL = logicalRuns(isL, dwellTh);
end

function runs = logicalRuns(mask, dwellTh)
    runs = struct('start',{},'stop',{},'dur',{});
    if ~any(mask), return; end

    d = diff([0; mask; 0]);
    s = find(d == 1);
    e = find(d == -1) - 1;
    L = e - s + 1;

    keep = L >= dwellTh;
    s = s(keep); e = e(keep); L = L(keep);

    for i = 1:numel(s)
        runs(i).start = s(i);
        runs(i).stop  = e(i);
        runs(i).dur   = L(i);
    end
end
