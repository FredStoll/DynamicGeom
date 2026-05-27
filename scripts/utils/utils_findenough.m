function [idx,idxs] = utils_findenough(data,thr,nbRequired,op)
% Find the position in a matrix (data) where a given number of values (nbRequired) exceed a given threshold (thr)
    % op is a comparator string such as '>=', '<=', '>', or '<'.
    eval(['oo=[-1 find(data',op,'thr)];']); 
    %eval(['oo=[-1 find(data>thr)];']); 
    xx = [find(diff(oo)~=1) length(oo)];
    yy=[xx(1) diff(xx)];
    zz=find(yy>=nbRequired);
    %zz=find(yy>=nbRequired,1,'first');
    idx=oo(xx(zz-1)+1) ;
    
    idxs = [];parts = [];
    for i = 1 : length(idx)
        idxs = [idxs, idx(i) : idx(i)+yy(zz(i))-1];
        parts = [parts, i*ones(1,length(idx(i) : idx(i)+yy(zz(i))-1))];
    end
    