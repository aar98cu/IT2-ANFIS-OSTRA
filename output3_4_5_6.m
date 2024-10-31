function anfis=output3_4_5_6(anfis)
    st=anfis.ni+anfis.ni*anfis.mf;
    
    % OSTRA algorithm
    wi=anfis.nodes(st+1:st+anfis.nr,1)';
    ws=anfis.nodes(st+1:st+anfis.nr,2)';

    % Sort weights in increasing order
    [wi,O1]=sort(wi);
    ws=sort(ws);

    % Compute wn
    for i=1:anfis.nr
        wn(i)=(wi(i)+ws(i))/2;
    end

    % Compute yi and ys
    yi=sum(wi.*wn)/sum(wn);
    ys=sum(ws.*wn)/sum(wn);

    % Find switch points l and r
    for i=1:anfis.nr-1
        if wi(i)<=yi && yi<=wi(i+1)
           l=i;
        else
            l=0;
        end
        if ws(i)<=ys && ys<=ws(i+1)
           r=i;
         else
            r=0;
        end
    end

    % Compute Xl and Xr matrix
    Xl=[ws(1:l) wi(l+1:anfis.nr)]/sum([ws(1:l) wi(l+1:anfis.nr)]);
    Xr=[wi(1:r) ws(r+1:anfis.nr)]/sum([wi(1:r) ws(r+1:anfis.nr)]);

    % Compute X as the average of Xl and Xr
    X=(Xl+Xr)/2;
    % Unsort X
    X=X(O1);

    st=anfis.ni+anfis.ni*anfis.mf+anfis.nr;
    anfis.nodes(st+1:st+anfis.nr,1)=X;
    anfis.nodes(st+1:st+anfis.nr,2)=X;
end