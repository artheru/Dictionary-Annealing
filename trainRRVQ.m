function [ centers_table ] = trainRRVQ( X, M, K, L, I)
%Initialize variables:
[N,dim] = size(X);

centers_table = cell(1, M);

pX=zeros(1,N,L);
d =cell(1,L);
uglyfix=(1-N:0)';
tX = gpuArray(X);


    function centers=coolDown()
        A = runpca(tX);
        tX=tX*A;
        [~,centers,dist]=gpukmeans(tX(:,1),K,[]);
        for i=ceil(dim.^linspace(1/I,1,I-1));
            centers=[centers zeros(K,i-size(centers,2))];
            [~,centers,dist]=gpukmeans(tX(:,1:i),K,centers);
        end
        fprintf(' mid: %f ',mean(min(dist,[],2)));

        centers=gather(centers*A');
    end
    function firstLayer()
        dMat=sqdistT(centers_table{1},X);
        for i=1:L
            [dist,idx]=min(dMat);
            pX(1,:,i)=gather(idx');
            d{i}=gather(dist');
            dMat(sub2ind(size(dMat),idx,1:N))=Inf;
        end
        dMat=0;
        tX=gpuArray(X)-centers_table{1}(pX(1,:,1),:);
    end
    function MVE(m)
        cTc2=cell(m-1,1);
        for i=1:m-1
            cTc2{i}=2*(centers_table{i})*(centers_table{m})';
        end

        tX=gpuArray(X);
        cD=sqdistT(tX,centers_table{m});
        tX=0;
        epsTemp=0;
        for i=1:m-1
            epsTemp=epsTemp+cTc2{i}(pX(i,:,:),:);
        end
        dMat=zeros(N,L,K,'single','gpuArray');
        for j=1:L
            dMat(:,j,:)=bsxfun(@plus,d{j},cD);
        end
        dMat=reshape(dMat+reshape(epsTemp,N,L,K),N,L*K);
        epsTemp=0;

        xpX=reshape(pX,m-1,L*N);
        pX=zeros(m,N,L);
        for j=1:L
            [td, idx] = min(dMat,[],2);
            d{j}=gather(td);
            w=mod(idx-1,L)*N+N; %uglyfix(idx);
            v=fix((idx-1)/L)+1; %uglyfix2(idx);
            pX(1:end-1,:,j)=xpX(:,uglyfix+w);
            pX(end,:,j)=gather(v);
            sb=sub2ind(size(dMat),1:N,idx');
            dMat(sb)=Inf;
        end
        dMat=0;
        xpX=0;
    end
for m = 1:M
    fprintf('iter %d', m);

    centers_table{m} = coolDown();  
    
    if (m==1)
        firstLayer();
    else
        MVE(m);
    end
    
    tX=gpuArray(X);
    for i=1:m
        tX=tX-centers_table{i}(pX(i,:,1),:);
    end
    fprintf('final: %f\n',mean(sum(tX.^2,2)));
end

end

