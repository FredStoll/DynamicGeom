function [params_a,params_b,Yh,exitflag] = utils_fitDecod(meanPerf,chance)

    perf = meanPerf(:,1)-chance;
    nbNeurons = meanPerf(:,2);
    params = [1;1];
                
    [params,~,exitflag] = fminsearch(@(params) decodingfct(params,perf,nbNeurons), params);

    params_a(1,:)=1/params(1)*(params(2));
    params_b(1,:)=params(2);
            
    Yh=params(2)*nbNeurons./(params(1)+nbNeurons);
end


function f = decodingfct(params,perf,nbNeurons)
    f = norm(perf-params(2)*nbNeurons./(params(1)+nbNeurons)); 
end

%     g = fittype('b*(x/(a+x))');
%     f0 = fit(dumm(:,2),dumm(:,1)-chance_l,g,'StartPoint',[1; 1]);
%     xx = linspace(0,3000,50);
%      plot(xx,f0(xx)+chance_l,'--','Color',colorsArea(ar,:),'LineWidth',1,'MarkerSize',15);hold on
