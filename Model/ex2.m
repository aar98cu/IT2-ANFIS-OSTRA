function y =ex2(inputs)
    NumInputs=2;
    NumMF=2;
    NumRules=4;
    NumData=size(inputs,1);
    Nodes=zeros(19,2);
    MFparams=[8.64495      10.3709      1.17641      -21.488          0.1;
    8.67543      10.4219     0.879027      21.8429     0.658395;
    23.9982      28.8014      1.03039       4.0003     0.118968;
    23.99845      28.79578      1.085156      124.0005           0.1];
    Cparams=[0.98466   0.0090721     -1.0204;
    0.98615   0.0018278    -0.40407;
    1.0076   0.0046534    -0.23555;
    0.98509     0.01271    -0.59211];
    Rules=[3  5;
    3  6;
    4  5;
    4  6];

    for i=1:NumData
        Nodes(1:NumInputs,1)=inputs(i,:)';
        Nodes(1:NumInputs,2)=inputs(i,:)';

        for j=1:NumInputs
            for k=1:NumMF
                ind=NumInputs+(j-1)*NumMF+k;
                x = Nodes(j,1);
                b = MFparams((j-1)*NumMF+k,3);
                c = MFparams((j-1)*NumMF+k,4);
                ai = MFparams((j-1)*NumMF+k,1);
                tmp1i = (x - c)/ai;
                if tmp1i == 0
                    tmp2i=0;
                else
                    tmp2i = (tmp1i*tmp1i)^b;
                end
                Nodes(ind,1)=MFparams((j-1)*NumMF+k,5)/(1+ tmp2i);
                as = MFparams((j-1)*NumMF+k,2);
                tmp1s = (x - c)/as;
                if tmp1s == 0
                    tmp2s=0;
                else
                    tmp2s = (tmp1s*tmp1s)^b;
                end
                Nodes(ind,2)=1/(1+ tmp2s);
            end
        end

        st=NumInputs+NumInputs*NumMF;
        for j=st+1:st+NumRules
            tmpi=cumprod(Nodes(Rules(j-st,:)',1));
            tmps=cumprod(Nodes(Rules(j-st,:)',2));
            Nodes(j,1)=tmpi(end);
            Nodes(j,2)=tmps(end);
        end

        st=NumInputs+NumInputs*NumMF;
        wi=Nodes(st+1:st+NumRules,1)';
        ws=Nodes(st+1:st+NumRules,2)';
        [wi,O1]=sort(wi);
        ws=sort(ws);
        wn = (wi+ws)/2;
        yi=sum(wi.*wn)/sum(wn);
        ys=sum(ws.*wn)/sum(wn);
        for j=1:NumRules-1
            if wi(j)<=yi && yi<=wi(j+1)
               l=j;
            else
                l=0;
            end
            if ws(j)<=ys && ys<=ws(j+1)
               r=j;
             else
                r=0;
            end
        end
        Xl=[ws(1:l) wi(l+1:NumRules)]/sum([ws(1:l) wi(l+1:NumRules)]);
        Xr=[wi(1:r) ws(r+1:NumRules)]/sum([wi(1:r) ws(r+1:NumRules)]);
        X=(Xl+Xr)/2;
        X=X(O1);
        st=NumInputs+NumInputs*NumMF+NumRules;
        Nodes(st+1:st+NumRules,1)=X;
        Nodes(st+1:st+NumRules,2)=X;

        st=NumInputs+NumInputs*NumMF+2*NumRules;
        inp=Nodes(1:NumInputs,1)';
        for j=1:NumRules
            wn=Nodes(j+st-NumRules,1);
            Nodes(j+st,1)=wn*(sum(Cparams(j,1:end-1).*inp)+Cparams(j,end));
        end

        Nodes(end,1)=sum(Nodes(end-NumRules:end-1,1));
        y(i,1)=Nodes(end,1);
    end
end
