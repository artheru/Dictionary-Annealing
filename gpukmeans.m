function [label, center, val] = gpukmeans(X, k, start)

[n, dim]=size(X);

g=gpuArray(X);
if isempty(start) || max(max(start))==0
    center=X(randperm(n,k),:);
else
    center=start; 
end

%kk=gpuArray(single(repmat(1:k,n,1)));
idxes=gpuArray(single(1:k));

last = 0;label=1;
it=0; maxit=100;

b=10000;
while true
    last = label;

    label=zeros(n,1,'single','gpuArray');
    val=zeros(n,1,'single','gpuArray');
    for i=0:((n/b)-1)
        id=(1:b)+i*b;
        D=sqdist(center,g(id,:));
        [val(id),label(id)] = min(D,[],2); % assign samples to the nearest centers
    end
    
    if ~(any(label ~= last) && it<maxit)
        break
    end
    
    N=accumarray(label,1,[k 1]);
    missCluster=idxes(N==0);
    if ~isempty(missCluster)
        fprintf('dropped:%d on %d\n',length(missCluster),it);
        [~,idx] = sort(val,1,'descend');
        lbl=label(idx);
        for i=1:length(missCluster);
            nc=find(N(lbl)>1,1);
            N(lbl(nc))=N(lbl(nc))-1;
            N(missCluster(i))=N(missCluster(i))+1;
            lbl(nc)=missCluster(i);
            label(idx(nc))=missCluster(i);
        end
    end
    
    [lbl,id]=sort(label);
    dif=[diff(lbl)>0; true];
    tmp=(cumsum(g(id,:)));
    center=tmp(dif,:);
    center=[center(1,:); diff(center)];
    center=bsxfun(@rdivide,center,diff([0; find(dif)]));
    it=it+1;
end


center=gather(center);
label=gather(label);
val=gather(val);

end

function d=sqdist(a,b)

    d = bsxfun(@plus, -2*b*a', sum(a.*a,2)');
    d = bsxfun(@plus, d, sum(b.*b,2));

end