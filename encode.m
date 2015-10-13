function [idx, epsi, residue]=encode(db,centers,ls)
    cTc2=computeCTC(centers);
    [N,dim]=size(db);
    M=length(centers);
    head=1;
    idx=zeros(N,M,'single');
    epsi=zeros(N,1,'single');
    residue=zeros(N,dim,'single');
    while head<N
        tail=head+10000;
        if tail>N
            tail=N;
        end
        [idx(head:tail,:),epsi(head:tail),residue(head:tail,:)]=encodeN(db(head:tail,:),centers,cTc2,ls);
        head=tail+1;
    end
end