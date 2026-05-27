%% Create example rasterplot + PSTH for preference modulation
%- Author: Fred M. Stoll, Icahn School of Medicine at Mount Sinai, NY
%- Date: 2022.12

function fighandle = utils_rasterPSTH_2bins(Unit,evt2take,evtName,cond,grp,pre,post,pathFig)

if nargin==5 %- missing time limits
    pre = 1000;
    post = 2000;
elseif nargin==7 %- no saving option
    saveit = false;
elseif nargin == 8
    saveit = true;
end

name = [Unit.session ' Ch' Unit.ch ' Clust' num2str(Unit.clust_id) ' - Area '  Unit.area];
ncd = unique(cond);
nTrCd = grpstats(ones(size(cond)),cond,'numel');
trashCd = nTrCd<2;

rastmat_all=[];psth_all=[];
for c = 1 : length(ncd)
    
    %- PSTH single trial
    [~,psth_trials,trialspx] = mpsth_v2(Unit.timestamps,evt2take(cond==ncd(c)),'fr',1,'binsz',1,'chart',0,'pre',pre,'post',post);
    psth_trials(:,2:end) = 1000*psth_trials(:,2:end);
    
    %- SDF with Gaussian 50ms
    sdf_trials = psth_trials(:,1);
    for ii = 1 : length(psth_trials(1,:))-1
        [sdf,~] = msdf(psth_trials(:,[1 ii+1]),'Gauss',50); %- 30 ms before
        sdf_trials(:,ii+1) = sdf(:,2);
    end
    
    %- average SDF + sem
    sdf_avg = mean(sdf_trials(:,2:end),2);
    if nTrCd(c)>1
        sdf_sem = std(sdf_trials(:,2:end)')/sqrt(length(sdf_trials(1,2:end)))';
    else
        sdf_sem = zeros(size(sdf_avg));
    end
    
    %- Raster
    [rastmat,timevec] = mraster(trialspx,pre,post);
    
    psth_all(:,c) = sdf_avg;
    psth_all_sem(:,c) = sdf_sem;
    rastmat_all = [rastmat_all ; rastmat];
end
sorted_cond = sortrows(cond')';

% if length(ncd)>2
%     colors =  cbrewer('div', 'RdGrBu', length(ncd));
% else
%     colors =  cbrewer('qual', 'Set2', length(ncd));
% end
colors =  cbrewer('div', 'RdGrBu', length(ncd));
if length(ncd)==6
    colors =  cbrewer('qual', 'Paired', 10);
    colors = colors([3 4 7:10],:);
elseif  length(ncd)>=8
     colors =  cbrewer('seq', 'Reds', 6);
     colors2 =  cbrewer('seq', 'Blues', 6);
     colors = [colors2(3:end,:) ; colors(3:end,:)];
   
end
% colors =  [164 205 226 ; 0 113 188 ; 250 150 150 ; 255 0 0; 177 222 138 ; 51 160 44]/255;

%- divise the 2 groups for subplots
ncd_sub1 = find(ncd<grp);
ncd_sub2 = find(ncd>grp);

%- some time average...
psth_avg = mean(psth_all(timevec>=200 & timevec<=800,:));
psth_avg_sem = mean(psth_all_sem(timevec>=200 & timevec<=800,:));

%% Make the actual figure (Raster+PSTH)

fighandle = figure('Position',[960/2   282   581*2   585],'Color','w');

for g = 1 : 2
%- plot SDF
hax1 = axes('Position',[0.13+(0.44*(g-1)) 0.11 0.775/2 0.549]);
eval(['ncd_sub = ncd_sub' num2str(g) ';']);
for c = 1 : length(ncd_sub)
    if trashCd(ncd_sub(c))~=1
        plot(timevec,psth_all(:,ncd_sub(c)),'Color',colors(c,:),'LineWidth',2);hold on
        ciplot(psth_all(:,ncd_sub(c))-psth_all_sem(:,ncd_sub(c)),psth_all(:,ncd_sub(c))+psth_all_sem(:,ncd_sub(c)),timevec,colors(c,:),.2);
    end
end
ticks_val = [-pre:500:post 0]; ticks_val=unique(ticks_val);
ticks_name=num2cell(ticks_val/1000); ticks_name{ticks_val==0}=evtName;

set(gca,'Xtick',ticks_val,'XTickLabel',ticks_name)
xlabel('Time (s)');ylabel('Firing rate (Hz)');
maxFR = ceil(max(max(psth_all(:,~trashCd)+psth_all_sem(:,~trashCd))));
line([0 0],[0 round(maxFR+.1*maxFR)],'Color','k')
ylim([0 round(maxFR+.1*maxFR)]);xlim([-pre+100 post-100]);set(gca,'FontSize',12);

eval(['rastmat_sub = rastmat_all(ismember(sorted_cond,ncd(ncd_sub' num2str(g) ' )),:);']);
eval(['sorted_cond_sub = sorted_cond(ismember(sorted_cond,ncd(ncd_sub' num2str(g) ' )));']);
ncd_sub = eval(['ncd(ncd_sub' num2str(g) ' )']);
%- plot Raster
hax2 = axes('Position',[0.13+(0.44*(g-1)) 0.6777 0.775/2 0.231]);
for i = 1 : length(rastmat_sub(:,1))
    plot(timevec(rastmat_sub(i,:)~=0),rastmat_sub(i,rastmat_sub(i,:)~=0)*i,'.','MarkerSize',4,'Color',colors(ncd_sub==sorted_cond_sub(i),:)),hold on
end

rast_lim = max([length(rastmat_sub(:,1)) length(rastmat_all(:,1))-length(rastmat_sub(:,1))]);

ylim([-2 rast_lim+2]); xlim([-pre+100 post-100]);
box on;
set(hax2,'XTick',[],'FontSize',8,'color','w','xcolor',[1 1 1]);
title(name,'FontSize',13);
ylabel('Trials');
line([0 0],[0 length(rastmat_sub(:,1))],'Color','k')

eval(['dumm = ncd_sub' num2str(g) ';']);

%- label conditions
for c = 1 : length(ncd_sub)
    if trashCd(dumm(c))~=1
        text(post-180,mean(find(sorted_cond_sub==ncd_sub(c))),[num2str(ncd_sub(c))],'Color',colors(c,:),'FontSize',14);
    end
end

end

% for g = 1 : 2
%     ncd_sub = eval(['ncd(ncd_sub' num2str(g) ' )']);
%     
%     %- extract waveform during time window of the different periods
%     timewindow_sub = evt2take(ismember(cond,ncd_sub));
%     timewindow_sub = [min(timewindow_sub) max(timewindow_sub)];
%     listspk = find(Unit.timestamps>timewindow_sub(1) & Unit.timestamps<=timewindow_sub(2));
%     wf(g,:) = mean(Unit.waveforms(listspk,:));
%     time_wf(g,:) = 1+((g-1)*10) : length(wf)+((g-1)*10) ;
% end
% 
% for g = 1 : 2
%     %- plot Waveform
%     hax3 = axes('Position',[0.141+(0.44*(g-1)) 0.581 0.089/2 0.097]);
%     plot(time_wf(abs(g-3),:),wf(abs(g-3),:),'Color',[.6 .6 .6]); hold on
%     plot(time_wf(g,:),wf(g,:),'Color','k');
%     axis off
% end
% 
% grp_sub = [1130 1150 1170 1190 ; 1230 1250 1270 1290; 2130 2150 2170 2190 ; 2230 2250 2270 2290];
% hax4 = axes('Position',[0.92 0.48 0.07 0.16]);
% colo = cbrewer('qual','Paired',6); colo = colo([1 5 2 6],:);
% 
% for g = 1 : length(grp_sub(:,1))
%     takeit = ismember(ncd,grp_sub(g,:));
%     plot(mod(ncd(takeit),100),psth_avg(takeit),'.-','LineWidth',2,'Color',colo(g,:),'MarkerSize',15);hold on
%     ciplot(psth_avg(takeit)-psth_avg_sem(takeit),psth_avg(takeit)+psth_avg_sem(takeit),mod(ncd(takeit),100),colo(g,:),0.2);
% end
% set(gca,'Xtick',mod(ncd(takeit),100),'XTickLabel',mod(ncd(takeit),100))
% xlim([20 100])

%- save Figure in eps if needed
if saveit
    set(gcf,'PaperPositionMode','auto')
    disp(['***** Figure saved in : ' pathFig ' *****' ])
    print([pathFig name(1:end)],'-depsc','-tiff')
end

