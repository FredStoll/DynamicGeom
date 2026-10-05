
function out = utils_decoding_crosstask_rmvarea(session,param)

load([param.path2go session '_pool.mat'],'cond','info','spkpool');
temp = load([param.path2go session '_pool.mat'],'param'); % 2 step loading cos param structure exist already! 
param.prev = temp.param; % keep the preprocessing param in prev!

labels = param.labels;
other_label = length(labels)+1; % Represent 'other' as label 4

% Neural data dimensions: neurons x trials x time bins

% Data for 1FC task
data = spkpool(:,cond.task==1,:); % neurons x trials x time bins
T = cond.proba_1FC(cond.task==1); % Labels
P = ismember(cond.flavor_1FC(cond.task==1),param.trainjuice) & ismember(cond.proba_1FC(cond.task==1),labels); % Labels

% Data for 2AFC task: 
tr_idx = cond.trial(cond.task==2); % trial index for 2AFC task
data_ins = spkpool(:,cond.task==2,:); % neurons x trials x time bins
true_values = cond.chosenproba_2AFC(cond.task==2);
other_values = cond.unchosenproba_2AFC(cond.task==2);
if param.Same_juice_only
    Samejuice_tr = (cond.chosenflavor_2AFC(cond.task==2) == cond.unchosenflavor_2AFC(cond.task==2)) & ismember(cond.chosenflavor_2AFC(cond.task==2),param.juice2keep);
else
    Samejuice_tr = ismember(cond.chosenflavor_2AFC(cond.task==2),param.juice2keep) | ismember(cond.unchosenflavor_2AFC(cond.task==2),param.juice2keep);
end
Samejuice_tr = Samejuice_tr & (true_values~=other_values); %- only different proba

%- remove the trials we don't want to consider
data = data(:,P,:); 
T = T(P);    

data_ins = data_ins(:,Samejuice_tr,:);
true_values = true_values(Samejuice_tr);    
other_values = other_values(Samejuice_tr);
tr_idx = tr_idx(Samejuice_tr); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% No extra smoothing here: spike bins are already prepared upstream.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Smooth the data into 200ms bins every 20ms
% time = param.prev.bins(1,:);
% start_bins = 1:param.step:(size(data, 3) - param.bins); % start of each bin   
% end_bins = start_bins + param.bins - 1; % end of each bin
% time_smoothed = time(start_bins); % time vector for the smoothed data

% clear data_smoothed data_ins_smoothed
% for i = 1:length(start_bins)
%     data_smoothed(:, :, i) = mean(data(:, :, start_bins(i):end_bins(i)), 3);
%     data_ins_smoothed(:, :, i) = mean(data_ins(:, :, start_bins(i):end_bins(i)), 3);
% end
% data = data_smoothed;
% data_ins = data_ins_smoothed;


% Focus on the time bins around the event selected
evt2take = find(ismember(param.prev.evt, param.alignment));
subtime = param.prev.bins(2,:)==evt2take ;

time = param.prev.bins(1,subtime);
data = data(:, :, subtime);
data_ins = data_ins(:, :, subtime);

% Remove low-firing neurons
% avg_firing_rate = mean(mean(data, 3), 2);
avg_firing_rate = mean(mean(data(:,:,time>param.bin2train(1) & time<=param.bin2train(2) ), 3), 2);
valid_neurons = avg_firing_rate >= param.min_firing_threshold;

% Define the time bins for normalization
norm_bin_start = find(time == param.baseline_norm(1)); % Start bin for normalization 
norm_bin_end = find(time == param.baseline_norm(2));  % End bin for normalization

% Normalize data
min_data = min(data(:, :, norm_bin_start:norm_bin_end), [], 3);
max_data = max(data(:, :, norm_bin_start:norm_bin_end), [], 3);
data = (data - min_data) ./ (max_data - min_data);
data(isnan(data) | isinf(data)) = 0;

min_data_ins = min(data_ins(:, :, norm_bin_start:norm_bin_end), [], 3);
max_data_ins = max(data_ins(:, :, norm_bin_start:norm_bin_end), [], 3);
data_ins = (data_ins - min_data_ins) ./ (max_data_ins - min_data_ins);
data_ins(isnan(data_ins) | isinf(data_ins)) = 0;

data = data(valid_neurons, :, :);
data_ins = data_ins(valid_neurons, :, :);
disp(['Number of low-firing neurons removed: ' num2str(sum(~valid_neurons))]);
disp(['Number of neurons remaining: ' num2str(sum(valid_neurons))]);

% Define new labels
T_new = NaN(size(T)); % Initialize with NaN for all
for i = 1:length(labels)
    T_new(T == labels(i)) = i; % Assign new labels
end
T_new(isnan(T_new)) = other_label; % Label for 'other' is 1 more than length of labels

% Map chosen and unchosen probabilities to labels
chosen_labels = NaN(size(true_values));
unchosen_labels = NaN(size(other_values));

for i = 1:length(labels)
    chosen_labels(true_values == labels(i)) = i;
    unchosen_labels(other_values == labels(i)) = i;
end
chosen_labels(isnan(chosen_labels)) = other_label; % Map "other" probabilities to length(label)+1
unchosen_labels(isnan(unchosen_labels)) = other_label; % Map "other" probabilities to length(label)+1

% keep only the labels we used to train (keep only the trials with labels )
keepTr = find((ismember(chosen_labels, length(labels)+1) | ismember(unchosen_labels, length(labels)+1))==0);

% Train multiclass classifier
bin2train = find(time >= param.bin2train(1) & time <= param.bin2train(2) ); % Specify bin range for training
cv = cvpartition(T_new, 'KFold', param.kfold);
     
all_units = find(valid_neurons);
all_areas = info.area(valid_neurons);

% change all_areas to group the one that goes together (from param.area2test, with new name in param. area2test_name)
for ar = 1:length(param.area2test)
    all_areas(ismember(all_areas, param.area2test{ar})) = param.area2test_name(ar);
end

area2rmv = unique(all_areas);

data_all = data;
data_all_ins = data_ins;

 % figure;
for u = 1 : length(area2rmv)+1
     out(1,u).session = session;
     out(1,u).keepTr = tr_idx(keepTr); % output trials considered if matching with some behavior later! 
     out(1,u).keptUnits = valid_neurons;
     out(1,u).time = time;

     if u<=length(area2rmv) % remove 1 neuron at a time
          disp(['Removing area ' area2rmv{u} ' - ' num2str(u) ' / ' num2str(length(area2rmv)) ])
          area2take = ~ismember(all_areas,area2rmv(u));
          out(1,u).area4unit_removed = area2rmv(u);

          data = data_all(area2take,:,:);
          data_ins = data_all_ins(area2take,:,:);
          data_ins_heldout = data_all_ins(~area2take,:,:);

     else % no removal!== reference perf
          disp(['Full population '])
          out(1,u).area4unit_removed = [];
          data = data_all;
          data_ins = data_all_ins;

     end
     classifiers = {};

     for fold = 1:param.kfold+1
     
         if fold == param.kfold+1 %- train on all trials for ins predictions
             trainIdx = true(size(data,2),1);
             testIdx = true(size(data,2),1);
         else
             trainIdx = training(cv, fold);
             testIdx = test(cv, fold);
         end
     
         % Filter data for training indices
         filtered_data_train = data(:, trainIdx, :);
         filtered_labels_train = T_new(trainIdx);
     
         % Filter data for testing indices
         filtered_data_test = data(:, testIdx, :);
         filtered_labels_test = T_new(testIdx);
     
         % Prepare training data by concatenating bins as separate trials
         if length(bin2train) == 1
             % Single bin case
             X_train = squeeze(filtered_data_train(:, :, bin2train))'; % Transpose for training
             y_train = filtered_labels_train;
         else
             % Concatenate bins as separate trials
             X_train = filtered_data_train(:, :, bin2train);
             X_train = reshape(X_train, size(X_train, 1), [])'; % Concatenate 2nd and 3rd dimensions
             y_train = repmat(filtered_labels_train, length(bin2train), 1); % Repeat labels for bins
         end
     
         % Prepare testing data
         if length(bin2train) == 1
             % Single bin case
             X_test = squeeze(filtered_data_test(:, :, bin2train))'; % Transpose for testing
             y_test = filtered_labels_test;
         else
             % Concatenate bins as separate trials
             X_test = filtered_data_test(:, :, bin2train);
             X_test = reshape(X_test, size(X_test, 1), [])'; % Concatenate 2nd and 3rd dimensions
             y_test = repmat(filtered_labels_test, length(bin2train), 1); % Repeat labels for bins
         end
     
         % Train classifier
         %classifier = fitcecoc(X_train, y_train);
         classifier = fitcdiscr(X_train, y_train, 'DiscrimType', 'diagLinear');
     
         classifiers{fold} = classifier;
     
         % Test classifier
         predicted_labels = predict(classifier, X_test);
     
         % Calculate accuracy
         accuracy(fold) = sum(predicted_labels == y_test) / length(y_test);

         % -----------------------------------------------------------------
         % CONTINUOUS (LINEAR) DECODER, RICH & WALLIS PDF-BASED (1FC TRIALS) - Fig S4
         % -----------------------------------------------------------------
         if fold <= param.kfold
             % Map class indices (1..5) to continuous probabilities (10..90)
             y_train_cont = labels(y_train)';
             y_test_cont  = labels(y_test)';

             % 1. Fit Linear Model (OLS via SVD / Pseudoinverse)
             X_tr_design = [ones(size(X_train, 1), 1), X_train];
             X_te_design = [ones(size(X_test, 1), 1), X_test];
             
             b_lin = pinv(X_tr_design) * y_train_cont;
             
             % 2. Get continuous predictions on training set to fit PDFs
             pred_tr_raw = X_tr_design * b_lin;

             % 3. Estimate mean and std for each class distribution (Rich & Wallis)
             mu_c = zeros(1, length(labels));
             sigma_c = zeros(1, length(labels));
             for c = 1:length(labels)
                 idx_c = (y_train == c);
                 mu_c(c) = mean(pred_tr_raw(idx_c));
                 sigma_c(c) = std(pred_tr_raw(idx_c));
                 if sigma_c(c) < 1e-4 % Guard against zero variance
                     sigma_c(c) = 1e-4;
                 end
             end

             % 4. Predict continuous values on test set
             pred_lin_raw = X_te_design * b_lin;

             % 5. Categorize test trials using Gaussian PDFs
             prob_c = zeros(length(pred_lin_raw), length(labels));
             for c = 1:length(labels)
                 prob_c(:, c) = normpdf(pred_lin_raw, mu_c(c), sigma_c(c));
             end
             [~, pred_lin_cat] = max(prob_c, [], 2); % Category with max PDF

             % Store fold outputs
             out_cmp(fold).y_test_cat   = single(y_test);
             out_cmp(fold).y_test_cont  = single(y_test_cont);
             out_cmp(fold).pred_lda_cat = single(predicted_labels);
             out_cmp(fold).pred_lin_raw = single(pred_lin_raw);
             out_cmp(fold).pred_lin_cat = single(pred_lin_cat);
         end

     end

     out(1,u).decoding_cmp = out_cmp; % LDA vs linear predictions for each fold (Fig S4, main_006)

     disp(['Mean cross-validation (1FC-1FC) accuracy: ', num2str(mean(accuracy(1:param.kfold))) ' +/- ' num2str(std(accuracy(1:param.kfold)))]);
     disp(['All trials accuracy: ', num2str(accuracy(param.kfold+1)) ]);
     
     out(1,u).accuracy = accuracy; % for output: 1FC accuracy acros crossvalidation and all trial!
     
     % Prediction for all trials and time bins
     predictions = NaN(length(keepTr), size(data_ins, 3));
     predictions_score   = NaN(length(keepTr), size(data_ins, 3), size(labels,2));
     for trial_to_plot = 1:length(keepTr)
         % disp(['Processing trial ' num2str(trial_to_plot) ' of ' num2str(length(keepTr))]);
         trial_data = squeeze(data_ins(:, keepTr(trial_to_plot), :));
         try 
             [predictions(trial_to_plot, :),predictions_score(trial_to_plot, : , :)] = predict(classifiers{end}, trial_data');
         catch ME
             disp(['Error applying classifier for trial ' num2str(trial_to_plot) ]);
             disp(ME.message);
         end
     end
     
    % output raw predictions
    out(1,u).predictions = single(predictions);
    % get accuracy
    out(1,u).accuracy_2AFC = mean(predictions(:,bin2train) == chosen_labels(keepTr),[1 2]);


    % out(1,u).predictions_score = single(predictions_score);

     % Adjust predictions: 1 for chosen, -1 for unchosen, 0 for other (only for keeptr)
     out(1,u).adjusted_prediction_scores_rand = NaN(length(keepTr), size(data_ins, 3),3);
     for trial_to_plot = 1:length(keepTr)
         rand_tr = randperm(length(labels)-2,1);
         for bin = 1:size(data_ins, 3)
             predicted_label = squeeze(predictions_score(trial_to_plot, bin,:));
     
             ch_unch_other = {chosen_labels(keepTr(trial_to_plot)) ; unchosen_labels(keepTr(trial_to_plot))};
             ch_unch_other{3} = find(~ismember(1:length(labels),[ch_unch_other{:}]));
     
             for c = 1 : length(ch_unch_other)
                 if length(ch_unch_other{c})>1
                     out(1,u).adjusted_prediction_scores_rand(trial_to_plot, bin, c) = predicted_label(ch_unch_other{c}(rand_tr));
                 else
                     out(1,u).adjusted_prediction_scores_rand(trial_to_plot, bin, c) = predicted_label(ch_unch_other{c});
                 end
             end
         end
     end
     out(1,u).adjusted_prediction_scores_rand = single(out(1,u).adjusted_prediction_scores_rand);

     % count the number and duration of chosen state, unchosen state or other state during window of interest.
     % a state is defined as a period of time (4 bins) where the prediction is above 0.4 and the state is the one with the highest
     % prediction.
     
     time_state_dectection = time>=param.time_state(1) & time<=param.time_state(2);
     subtime = time(time_state_dectection);

     out(1,u).nb_states = NaN(size(out(1,u).adjusted_prediction_scores_rand,1),3);
     out(1,u).dur_states = NaN(size(out(1,u).adjusted_prediction_scores_rand,1),3);
     out(1,u).t_states = NaN(size(out(1,u).adjusted_prediction_scores_rand,1),3); % time of first state
     % out(1,u).avgFR_heldout = NaN(size(out(1,u).adjusted_prediction_scores_rand,1),3);
     out(1,u).states = single(NaN(size(out(1,u).adjusted_prediction_scores_rand,1),sum(time_state_dectection)));
     for tr = 1 : size(out(1,u).adjusted_prediction_scores_rand,1)
         pred = squeeze(out(1,u).adjusted_prediction_scores_rand(tr,time_state_dectection,:));

         %pred = pred ./ sum(pred,2);
         pred(pred<param.thr_state) = 0;
         [~,loc_max] = max(pred');
     
         % keep only the max
         pred_max = zeros(size(pred));
         for c = 1 : 3 % 3 category (chosen/unchosen/other)
             pred_max(loc_max==c,c) =  pred(loc_max==c,c);
         end
     
         pred_thr = zeros(size(pred));
         for c = 1 : 3
             [idx,idxs] = utils_findenough(pred_max(:,c)',0,param.thr_state_dur,'>');
             pred_thr(idxs,c) = pred(idxs,c);
             out(1,u).nb_states(tr,c) = length(idx);
             out(1,u).dur_states(tr,c) = length(idxs);
             out(1,u).states(tr,idxs) = c;
             if ~isempty(idx)
                out(1,u).t_states(tr,c) = subtime(idx(1)); % average time in state
             end
         end

     end

     % LABEL SHUFFLE CONTROL - only for full population (u == length(area2rmv)+1)
     if u == length(area2rmv)+1
         T_shuffle = T_new(randperm(length(T_new)));
         if length(bin2train) == 1
             X_train_shuf = squeeze(data(:, :, bin2train))';
             y_train_shuf = T_shuffle;
         else
             X_train_shuf_raw = data(:, :, bin2train);
             X_train_shuf = reshape(X_train_shuf_raw, size(X_train_shuf_raw, 1), [])';
             y_train_shuf = repmat(T_shuffle, length(bin2train), 1);
         end
         try
             classifier_shuf = fitcdiscr(X_train_shuf, y_train_shuf, 'DiscrimType', 'diagLinear');
             predictions_score_shuf = NaN(length(keepTr), size(data_ins, 3), size(labels, 2));
             for trial_to_plot = 1:length(keepTr)
                 trial_data = squeeze(data_ins(:, keepTr(trial_to_plot), :));
                 try
                     [~, sc] = predict(classifier_shuf, trial_data');
                     predictions_score_shuf(trial_to_plot, :, :) = sc;
                 catch
                 end
             end
             out(1,u).adjusted_prediction_scores_rand_shuffle = NaN(length(keepTr), size(data_ins, 3), 3);
             for trial_to_plot = 1:length(keepTr)
                 rand_tr = randperm(length(labels)-2, 1);
                 for bin = 1:size(data_ins, 3)
                     predicted_label = squeeze(predictions_score_shuf(trial_to_plot, bin, :));
                     ch_unch_other = {chosen_labels(keepTr(trial_to_plot)); unchosen_labels(keepTr(trial_to_plot))};
                     ch_unch_other{3} = find(~ismember(1:length(labels), [ch_unch_other{:}]));
                     for c = 1:length(ch_unch_other)
                         if length(ch_unch_other{c}) > 1
                             out(1,u).adjusted_prediction_scores_rand_shuffle(trial_to_plot, bin, c) = predicted_label(ch_unch_other{c}(rand_tr));
                         else
                             out(1,u).adjusted_prediction_scores_rand_shuffle(trial_to_plot, bin, c) = predicted_label(ch_unch_other{c});
                         end
                     end
                 end
             end
             out(1,u).adjusted_prediction_scores_rand_shuffle = single(out(1,u).adjusted_prediction_scores_rand_shuffle);
             out(1,u).nb_states_shuffle  = NaN(length(keepTr), 3);
             out(1,u).dur_states_shuffle = NaN(length(keepTr), 3);
             out(1,u).t_states_shuffle   = NaN(length(keepTr), 3);
             for tr = 1:length(keepTr)
                 pred = squeeze(out(1,u).adjusted_prediction_scores_rand_shuffle(tr, time_state_dectection, :));
                 pred(pred < param.thr_state) = 0;
                 [~, loc_max] = max(pred');
                 pred_max = zeros(size(pred));
                 for c = 1:3
                     pred_max(loc_max == c, c) = pred(loc_max == c, c);
                 end
                 for c = 1:3
                     [idx, idxs] = utils_findenough(pred_max(:, c)', 0, param.thr_state_dur, '>');
                     out(1,u).nb_states_shuffle(tr, c)  = length(idx);
                     out(1,u).dur_states_shuffle(tr, c) = length(idxs);
                     if ~isempty(idx)
                         out(1,u).t_states_shuffle(tr, c) = subtime(idx(1));
                     end
                 end
             end
             disp('Label shuffle control: done.');
         catch ME
             disp(['Label shuffle control failed: ' ME.message]);
             out(1,u).adjusted_prediction_scores_rand_shuffle = [];
             out(1,u).nb_states_shuffle  = [];
             out(1,u).dur_states_shuffle = [];
             out(1,u).t_states_shuffle   = [];
         end
     else
         out(1,u).adjusted_prediction_scores_rand_shuffle = [];
         out(1,u).nb_states_shuffle  = [];
         out(1,u).dur_states_shuffle = [];
         out(1,u).t_states_shuffle   = [];
     end

     if  u<=length(area2rmv)
        out(1,u).fr_heldout = single(data_ins_heldout(:,:,time_state_dectection));
        out(1,u).area4unit_heldout = {};
     else
        out(1,u).fr_heldout = single(data_ins(:,:,time_state_dectection));
        out(1,u).area4unit_heldout = all_areas;
     end

     out(1,u).states_time = time(time_state_dectection);
     out(1,u).cond = cond(ismember(cond.trial,out(1,u).keepTr),{'trial','chosenproba_2AFC','unchosenproba_2AFC','chosenflavor_2AFC','unchosenflavor_2AFC','chosenside_2AFC'});


     %% Topology of the probability centroids (response to reviewers, Fig R5)
     if u == length(area2rmv)+1

        % --- TOPOLOGY & REPRESENTATIONAL GEOMETRY (WITHIN SESSION 'u') ---

        % 1. Extract class centroids (Means: Classes x Neurons)
        Mu = classifiers{end}.Mu; 

        % 2. Calculate neural distances between probability states
        neural_dists = pdist(Mu, 'euclidean'); 

        % 3. Calculate True Probability distances 
        % IMPORTANT: Replace these with your actual probability bin centers
        true_probs = [0.1, 0.3, 0.5, 0.7, 0.9]; 
        true_prob_dists = pdist(true_probs'); 

        % 4. Correlate neural distance with mathematical probability distance
        [rho, pval] = corr(true_prob_dists', neural_dists', 'Type', 'Spearman');

        % 5. PCA Projection for linearity check
        [~, score, ~, ~, explained] = pca(Mu);

        % 6. Save variables to the 'out' structure (Skipping figures here)
        out(1,u).topology.neural_dists = neural_dists;
        out(1,u).topology.true_prob_dists = true_prob_dists;
        out(1,u).topology.rho = rho;
        out(1,u).topology.pval = pval;
        out(1,u).topology.pca_explained = explained;

        % Optional: Save the first 2 PC scores to look at the average shape later
        out(1,u).topology.pca_score = score(:, 1:min(2, size(score,2)));
     else
        
        out(1,u).topology = [];
     end

    % subplot(7,7,u );
    % col = [100 200 160 ; 240 90 90 ; 0 0 0 ]/255;
    % for c = 1 : 3
    %     st = out(1,u).fr_heldout;
    %     st(out(1,u).states~=c)=NaN;
    %     plot(nanmean(st,1),'Color',col(c,:),'LineWidth',2,'DisplayName',['State ' num2str(c)]);hold on
    % end
    % subplot(7,7,u);
    % col = [100 200 160 ; 240 90 90 ; 0 0 0 ]/255;
    % hold on;
    % for c = 1:3
    %     for t =1 : 2
    %         for pb = 1 : 5
    %             avg_pb(c,pb,t) = nanmean(out(1,u).avgFR_heldout(ch_unch(:,t)==pb,c), 1);
    %         end
    %         if t==1 & c==1
    %             plot(squeeze(avg_pb(c,:,t)),'Color',col(c,:),'LineWidth',2,'DisplayName',['Chosen ' num2str(t)]);
    %         elseif t==2 & c==1
    %             plot(squeeze(avg_pb(c,:,t)),'Color',col(c,:),'LineWidth',1,'DisplayName',['Unchosen ' num2str(t)]);
    %         elseif t==2 & c==2
    %             plot(squeeze(avg_pb(c,:,t)),'Color',col(c,:),'LineWidth',2,'DisplayName',['Other ' num2str(t)]);
    %         elseif t==1 & c==2
    %             plot(squeeze(avg_pb(c,:,t)),'Color',col(c,:),'LineWidth',1);
    %         else
    %             plot(squeeze(avg_pb(c,:,t)),'Color',col(c,:),'LineStyle','--');
    %         end

    %     end
    % end


end

