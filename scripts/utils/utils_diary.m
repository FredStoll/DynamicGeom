function utils_diary(fid, fmt, varargin)
% UTILS_DIARY  Write formatted text to a log file and to the terminal.
%
%   utils_diary(FID, FMT, ...)
%       Writes the formatted string to both the file identified by FID
%       (opened with fopen) and to the terminal (stdout).
%       Behaves exactly like fprintf but duplicates the output.
%
%   Typical setup at the top of a script:
%       log_file    = [report_dir 'output_00N.txt'];
%       fid_log     = fopen(log_file, 'w');
%       log_cleanup = onCleanup(@() fclose(fid_log));  % safe close on error
%       utils_diary(fid_log, '\n=====\nReport generated: %s\n', datestr(now));
%
%   For disp-style objects (tables, model coefficients, etc.) capture the
%   display string with evalc first:
%       utils_diary(fid_log, '%s', evalc('disp(mdl.Coefficients)'));
%
%   See also: fopen, fclose, fprintf, evalc.

    fprintf(fid, fmt, varargin{:});
    fprintf(fmt, varargin{:});
end
