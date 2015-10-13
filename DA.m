function [ idx, C ] = DA( db, dict, M,K,L,I )

[~,dim]=size(db);

if (isempty(dict)) 
    C=trainRRVQ(db,M,K,L,I);
else
    C=dict;
end

for i=1:3
    sz=zeros(M,1);
    for m=1:M
        sz(m)=mean(sum((bsxfun(@minus,C{m},mean(C{m}))).^2,2));
    end
    [~,idx]=sort(-sz);
    C=C(idx);
        
    for m=1:M
        [idx, ~, residue]=encode(db, C, L);
        fprintf('Distortion: %f\n', mean(sum(residue.^2,2)));
        intermediate=residue+C{m}(idx(:,m),:);
        
        rot=runpca(intermediate);
        tX=intermediate*rot;
        tC=C{m}*rot;
        for j=ceil(dim.^linspace(0,1,I))
            [~, tC(:,1:j), ~]=gpukmeans(tX(:,1:j),K,tC(:,1:j));
        end
        C{m}=tC*rot';
    end
end

[idx,~,err]=encode(db,C,L);

fprintf('Final Distortion: %f\n', mean(sum(err.^2,2)));

end

