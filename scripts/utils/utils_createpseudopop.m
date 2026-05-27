function [XX,YY,param] = utils_createpseudopop(data,factors,time,varargin)

% 'data' must be a X*Y matrix, with X the trial nb * the nb of channels ;
%                                   Y the time bins.
% 'factors' must be a 2 column matrix, with the condition (col1) and the channel (col2)


%% Params selection
%- define default values
defaults = struct('minTr',20, ...   %- only consider a similar nb of trials for each condition. Different rules might be applied otherwise..            
                  'perm',20, ...    %- X number of random trial selections
                  'bin','n',...     %- apply a binning procedure
                  'BinSize',200,... %- binsize
                  'step',50,... %- binsize
                  'window',[min(time)  max(time)],... %- time to consider
                  'pop',false); %- population (true)  or pseudo-pop (false) : affect the trial randomization
                   
param = struct(varargin{:});
for f = fieldnames(defaults)'
    if ~isfield(param, f{1})
      param.(f{1}) = defaults.(f{1});
    end
end

%% extract the trial nb
nUnit = unique(factors(:,2));
% [nTr] = grpstats([1:size(data,1)],factors,{'numel'}); % without a fucking loop but did not work if there is one unit without a factor
% nTr = reshape(nTr,size(nUnit,1),size(factors,2));

fact = unique(factors(:,1));

for n = 1 : length(nUnit)
    f1=0;
    for f = 1 : length(fact)
        nTr(n,f) = sum(factors(:,1)==fact(f) & factors(:,2)==nUnit(n));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment this block to always use the minimum trial count across conditions
% instead of the value passed in via param.minTr.
% if ~param.pop | param.minTr>=1000
%    param.minTr=min(min(nTr));
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Take only the selected time window -- This is a bit problematic for dual alignments..
data = data(:,time>=param.window(1) & time<=param.window(end));
time = time(time>=param.window(1) & time<=param.window(end));

%% binning procedure
if strcmp(param.bin,'y')
    init = param.window(1) : param.step : param.window(end)-param.BinSize;
    fin = param.window(1)+param.BinSize : param.step : param.window(end);
    
    data2 = NaN(size(data,1),size(init,2));
    for w = 1 : length(init)
        data2(:,w) = mean(data(:,time>=init(w) & time<=fin(w)),2); %- on Spk Count ...
    end
    data = data2;
    param.time = init+(param.BinSize/2);
    clear data2
else
    param.time = time;
end
%% Formatting data
XX = cell(param.perm,size(data,2));
YY = cell(param.perm,size(data,2));

if length(fact)==2
    %% Reject unit (neurons or channels) with too few trials
param.reject = nUnit(nTr(:,1)<param.minTr | nTr(:,2)<param.minTr);
nUnit(nTr(:,1)<param.minTr | nTr(:,2)<param.minTr) = [] ;
param.keep = nUnit;

%    ft_progress('init', 'text',   'Extracting Trials ...')
    for p = 1 : param.perm %- trial selection permutation
%        ft_progress(p/param.perm)
        all = zeros(param.minTr*2,length(nUnit),size(data,2));
        for n = 1 : length(nUnit) %- units
            T1 = data( factors(:,1)==fact(1) & factors(:,2)==nUnit(n), :);
            T2 = data( factors(:,1)==fact(2) & factors(:,2)==nUnit(n), :);
            rd1 = randperm(length(T1(:,1)));
            rd2 = randperm(length(T2(:,1)));
            if param.pop & n==1 %- if population of neuron, only randomized the trial selection once, not for every neuron
                rd1_pop = rd1;
                rd2_pop = rd2;
            end
            if param.pop
                all(:,n,:) = [T1(rd1_pop(1:param.minTr),:) ; T2(rd2_pop(1:param.minTr),:)] ;
            else
                all(:,n,:) = [T1(rd1(1:param.minTr),:) ; T2(rd2(1:param.minTr),:)] ;
            end
            % all(:,n,:) = [T1(1:param.minTr,:) ; T2(1:param.minTr,:)] ;
        end
        XX(p,:) = squeeze(num2cell(all,[1 2]))';
        
    end
    YY(:,:)={[(-ones(param.minTr,1))*(((param.minTr*2)/param.minTr)-1) ; (ones(param.minTr,1))*(((param.minTr*2)/(param.minTr))-1)]};
    
%     for m = 1 : size(YY,1)
%             YY(m,:)= {YY{m,1}(randperm(length(YY{m,1})),:)};     %- randomize labels (1 randomization for all time bins)
%     end

    
%    ft_progress('close')
elseif length(fact)==4
    %error('Only 2 conditions can be considered for now..')
    
    nUnit = unique(factors(:,2));
    param.reject = nUnit(nTr(:,1)<param.minTr | nTr(:,2)<param.minTr | nTr(:,3)<param.minTr | nTr(:,4)<param.minTr );
    nUnit(nTr(:,1)<param.minTr | nTr(:,2)<param.minTr | nTr(:,3)<param.minTr | nTr(:,4)<param.minTr) = [] ;
    param.keep = nUnit;

%    ft_progress('init', 'text',   'Extracting Trials ...')
    for p = 1 : param.perm %- trial selection permutation
%        ft_progress(p/param.perm)
        all = zeros(param.minTr*4,length(nUnit),size(data,2));
        for n = 1 : length(nUnit) %- units
            T1 = data( factors(:,1)==fact(1) & factors(:,2)==nUnit(n), :);
            T2 = data( factors(:,1)==fact(2) & factors(:,2)==nUnit(n), :);
            T3 = data( factors(:,1)==fact(3) & factors(:,2)==nUnit(n), :);
            T4 = data( factors(:,1)==fact(4) & factors(:,2)==nUnit(n), :);
            rd1 = randperm(length(T1(:,1)));
            rd2 = randperm(length(T2(:,1)));
            rd3 = randperm(length(T3(:,1)));
            rd4 = randperm(length(T4(:,1)));
            if param.pop & n==1 %- if population of neuron, only randomized the trial selection once, not for every neuron
                rd1_pop = rd1;
                rd2_pop = rd2;
                rd3_pop = rd3;
                rd4_pop = rd4;
            end
            if param.pop
                all(:,n,:) = [T1(rd1_pop(1:param.minTr),:) ; T2(rd2_pop(1:param.minTr),:) ; T3(rd3_pop(1:param.minTr),:) ; T4(rd4_pop(1:param.minTr),:)] ;
            else
                all(:,n,:) = [T1(rd1(1:param.minTr),:) ; T2(rd2(1:param.minTr),:) ; T3(rd3(1:param.minTr),:) ; T4(rd4(1:param.minTr),:)] ;
            end
 
           % all(:,n,:) = [T1(1:param.minTr,:) ; T2(1:param.minTr,:) ; T3(1:param.minTr,:); T4(1:param.minTr,:)] ;
        end
        XX(p,:) = squeeze(num2cell(all,[1 2]))';
    end
    
elseif length(fact)==5
    %error('Only 2 conditions can be considered for now..')
    
    nUnit = unique(factors(:,2));
    param.reject = nUnit(nTr(:,1)<param.minTr | nTr(:,2)<param.minTr | nTr(:,3)<param.minTr | nTr(:,4)<param.minTr | nTr(:,5)<param.minTr);
    nUnit(nTr(:,1)<param.minTr | nTr(:,2)<param.minTr | nTr(:,3)<param.minTr | nTr(:,4)<param.minTr | nTr(:,5)<param.minTr) = [] ;
    param.keep = nUnit;

%    ft_progress('init', 'text',   'Extracting Trials ...')
    for p = 1 : param.perm %- trial selection permutation
%        ft_progress(p/param.perm)
        all = zeros(param.minTr*5,length(nUnit),size(data,2));
        for n = 1 : length(nUnit) %- units
            T1 = data( factors(:,1)==fact(1) & factors(:,2)==nUnit(n), :);
            T2 = data( factors(:,1)==fact(2) & factors(:,2)==nUnit(n), :);
            T3 = data( factors(:,1)==fact(3) & factors(:,2)==nUnit(n), :);
            T4 = data( factors(:,1)==fact(4) & factors(:,2)==nUnit(n), :);
            T5 = data( factors(:,1)==fact(5) & factors(:,2)==nUnit(n), :);
            rd1 = randperm(length(T1(:,1)));
            rd2 = randperm(length(T2(:,1)));
            rd3 = randperm(length(T3(:,1)));
            rd4 = randperm(length(T4(:,1)));
            rd5 = randperm(length(T5(:,1)));
            
            if param.pop & n==1 %- if population of neuron, only randomized the trial selection once, not for every neuron
                rd1_pop = rd1;
                rd2_pop = rd2;
                rd3_pop = rd3;
                rd4_pop = rd4;
                rd5_pop = rd5;
            end
            if param.pop
                all(:,n,:) = [T1(rd1_pop(1:param.minTr),:) ; T2(rd2_pop(1:param.minTr),:) ; T3(rd3_pop(1:param.minTr),:) ; T4(rd4_pop(1:param.minTr),:) ; T5(rd5_pop(1:param.minTr),:)] ;
            else
                all(:,n,:) = [T1(rd1(1:param.minTr),:) ; T2(rd2(1:param.minTr),:) ; T3(rd3(1:param.minTr),:) ; T4(rd4(1:param.minTr),:) ; T5(rd5(1:param.minTr),:) ] ;
            end
 
        end
        XX(p,:) = squeeze(num2cell(all,[1 2]))';
        
    end
    
end
       
if length(fact)==2
        YY(:,:)={[(-ones(param.minTr,1))*(((param.minTr*2)/param.minTr)-1) ; (ones(param.minTr,1))*(((param.minTr*2)/(param.minTr))-1)]};

else
        YY(:,:)={[sortrows(repmat(fact,param.minTr,1))]};

end
    
 %   ft_progress('close')

    
    
    
end