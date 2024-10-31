function anfis=output1(anfis)
    mfparams=anfis.mfparams;

    for i=1:anfis.ni
        for j=1:anfis.mf

            ind=anfis.ni+(i-1)*anfis.mf+j;

            x = anfis.nodes(i,1);
            b = mfparams((i-1)*anfis.mf+j,3);
            c = mfparams((i-1)*anfis.mf+j,4);

            % Calculate the lower membership degree
            ai = mfparams((i-1)*anfis.mf+j,1);            
            tmp1i = (x - c)/ai;
            if tmp1i == 0
                tmp2i=0;
            else
                tmp2i = (tmp1i*tmp1i)^b;
            end
            anfis.nodes(ind,1)=mfparams((i-1)*anfis.mf+j,5)/(1+ tmp2i);

            % Calculate the upper membership degree
            as = mfparams((i-1)*anfis.mf+j,2);
            tmp1s = (x - c)/as;
            if tmp1s == 0
                tmp2s=0;
            else
                tmp2s = (tmp1s*tmp1s)^b;
            end
            anfis.nodes(ind,2)=1/(1+ tmp2s);
        end 
    end
end