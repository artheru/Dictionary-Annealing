function [idx,epsi,err]=encodeN(db,centers,cTc2,ls)
	[N,dim]=size(db);
	M    =length(centers);
	[k,~]=size(centers{1});
    
    ls=repmat(ls,M,1);
    
	pX=zeros(1,N,ls(1),'single','gpuArray');
    d =cell(1,ls(1));
    
    GPUdb=gpuArray(db);
    gC=cell(M,1);
    for m=1:M
        gC{m}=gpuArray(centers{m});
    end
	
    D=sqdistT(gC{1},GPUdb);
	for i=1:ls(1)
        [dist,idx]=min(D);
		pX(1,:,i)=idx';
		d{i}=dist';
        D(sub2ind(size(D),idx,1:N))=Inf;
    end
	
    uglyfix=(1-N:0)';
	for m=2:M
		dMat=zeros(N,ls(m-1),k,'single','gpuArray');
		cD=sqdistT(GPUdb,gC{m});
        epsTemp=0;
        for i=1:m-1
            epsTemp=epsTemp+cTc2{m,i}(pX(i,:,:),:);
        end
		for j=1:ls(m-1)
            dMat(:,j,:)=bsxfun(@plus,d{j},cD);
        end
		dMat=reshape(dMat+reshape(epsTemp,N,ls(m-1),k),N,ls(m-1)*k);
        xpX=reshape(pX,m-1,ls(m-1)*N);
        pX=zeros(m,N,ls(m),'single','gpuArray');
		for j=1:ls(m)
            [d{j}, idx] = min(dMat,[],2);
			w=mod(idx-1,ls(m-1))*N+N; %uglyfix(idx);
            v=fix((idx-1)/ls(m-1))+1; %uglyfix2(idx);
            pX(1:end-1,:,j)=xpX(:,uglyfix+w);
            pX(end,:,j)=v;
            sb=sub2ind(size(dMat),1:N,idx');
            dMat(sb)=Inf;
        end
    end
	
    quant=zeros(N,dim,'single','gpuArray');
    for i=1:M
        quant=quant+gC{i}(pX(i,:,1),:);
    end
    epsi=sum(quant.^2,2);
    for i=1:M
        c2=sum(gC{i}.^2,2);
        epsi=epsi-c2(pX(i,:,1));
    end
    epsi=gather(epsi);
	idx=gather(squeeze(pX(:,:,1))');
    err=gather(GPUdb-quant);
end
