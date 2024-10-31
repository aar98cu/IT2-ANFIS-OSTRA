function anfis=output2(anfis)
    st=anfis.ni+anfis.ni*anfis.mf;

    for i=st+1:st+anfis.nr
        I=find(anfis.config(:,i)==1);

        % Calculate T-Norm for lower and upper membership degree
        tmpi=cumprod(anfis.nodes(I,1));
        tmps=cumprod(anfis.nodes(I,2));
        anfis.nodes(i,1)=tmpi(end);
        anfis.nodes(i,2)=tmps(end);
    end
end