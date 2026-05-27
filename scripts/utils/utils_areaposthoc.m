% Pairwise area post-hoc contrasts from a fitted LME model, with FDR correction.
% Uses emmeans() to get estimated marginal means, then builds a Wald contrast
% for every pair of areas in area2test.  Returns raw p-values (pval), Wald
% statistics (wald), the significance threshold (thr), and FDR-adjusted p-values
% (pval_adj).  Pass fig2do = 'y' to plot the Wald matrix as a heatmap.
function [pval,wald,thr,pval_adj] = utils_areaposthoc(lme,area2test,fig2do,compact)

emm = emmeans(lme,{'area'});

thr = .05;
%h = emmip(emm,'area');
clear pval wald
pval_all=[];
for ar1 = 1 : length(area2test)
    for ar2 = 1 : length(area2test)
        if ar1>ar2
            L_post = [ strcmp(cellstr(emm.table.area),area2test{ar1})' - strcmp(cellstr(emm.table.area),area2test{ar2})'];
            H0_post = contrasts_wald(lme,emm,L_post);
            pval(ar1,ar2) = H0_post.pVal;
            wald(ar1,ar2) = H0_post.Wald;
            pval_all = [pval_all ; H0_post.pVal ar1 ar2]; 
        end
    end
end
[h, crit_p, adj_p]=utils_fdr_bh(pval_all(:,1),.05);
for i = 1 : length(adj_p)
    pval_adj(pval_all(i,2),pval_all(i,3)) = adj_p(i);
end
if strcmp(fig2do(1),'y')
    colorWald = cbrewer('seq', 'Purples', 100);
    figure
    imagesc(wald');colormap(colorWald(1:70,:));colorbar;
    for ar1 = 1 : length(area2test)
        for ar2 = 1 : length(area2test)
            if ar1>ar2
                % text(ar2,ar1,num2str(round(pval_J(ar1,ar2)*10000)/10000))
                % text(ar2,ar1,num2str(pval_J(ar1,ar2)),'HorizontalAlignment','center')
                n=1;p=0;
                while p == 0
                    n = n + 1;
                    p = round(pval_adj(ar1,ar2),n);
                end
                p = round(pval_adj(ar1,ar2),n+1);
                if pval_adj(ar1,ar2)<.05
                    if ~compact
                        text(ar1,ar2,num2str(p),'HorizontalAlignment','center','FontWeight','bold')
                    else
                        text(ar1,ar2,'*','HorizontalAlignment','center','FontWeight','bold')
                    end
                else
                    if ~compact
                        text(ar1,ar2,num2str(p),'HorizontalAlignment','center')
                    end
                end
            end
        end
    end
    set(gca,'Xtick',1:length(area2test),'XtickLabel',area2test,'Ytick',1:length(area2test),'YtickLabel',area2test)
    ylim([0.5 length(area2test)+0.5])
    xlim([1.5 length(area2test)+0.5])
end