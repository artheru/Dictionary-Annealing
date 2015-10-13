function [ cTc2 ] = computeCTC( C )
    M=length(C);
    cTc2=cell(M); % Precomputed C<->C
    for i=2:M
        for j=1:i-1
            cTc2{i,j}=gpuArray(single(2*C{j}*C{i}'));
        end
    end
end

