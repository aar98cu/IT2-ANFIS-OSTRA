function anfis=update_de_do(anfis)

    s=1;
    for i=anfis.ni+1:anfis.ni+anfis.ni*anfis.mf
        for j=1:4
            do_dp = dmf_dp(anfis,i, j);
            if j==1
                anfis.mfparam_de_do(s,j)=anfis.mfparam_de_do(s,j) + anfis.de_do(i,1)*do_dp(1);
                anfis.mfparam_de_do(s,j+1)=anfis.mfparam_de_do(s,j+1) + anfis.de_do(i,2)*do_dp(2);
            else
                anfis.mfparam_de_do(s,j+1)=anfis.mfparam_de_do(s,j+1) + sum(anfis.de_do(i,:).*do_dp)/2;
            end
        end
        s=s+1;
    end
    
    s=1;
    for i=1+anfis.ni+anfis.ni*anfis.mf+2*anfis.nr:size(anfis.config)-1
        for j=1:anfis.ni+1
            do_dp = dconsequent_dp(anfis,i, j); 
            anfis.cparam_de_do(s,j)=anfis.cparam_de_do(s,j) + anfis.de_do(i,1)*do_dp;
        end
        s=s+1;
    end
    


function tmp=dmf_dp(anfis,i, j)
    I=find(anfis.config(:,i)==1);
    x=anfis.nodes(I,1);
    ai=anfis.mfparams(i-anfis.ni,1);
    as=anfis.mfparams(i-anfis.ni,2);
    b=anfis.mfparams(i-anfis.ni,3);
    c=anfis.mfparams(i-anfis.ni,4);
    B=anfis.mfparams(i-anfis.ni,5);
    tmp1i = (x - c)/ai;
    if tmp1i == 0
        tmp2i=0;
    else
        tmp2i = (tmp1i*tmp1i)^b;
    end
    denomi = (1 + tmp2i)*(1 + tmp2i);

    tmp1s = (x - c)/as;
    if tmp1s == 0
        tmp2s=0;
    else
        tmp2s = (tmp1s*tmp1s)^b;
    end
    denoms = (1 + tmp2s)*(1 + tmp2s);

    if j==1
        tmp=[(2*b*B*tmp2i/(ai*denomi)) (2*b*tmp2s/(as*denoms))];
    elseif j==2 && tmp1i==0 && tmp1s==0
        tmp=[0 0];
    elseif j==2 && tmp1i~=0 && tmp1s~=0
        tmp=[(-log(tmp1i*tmp1i)*B*tmp2i/denomi) (-log(tmp1s*tmp1s)*tmp2s/denoms)];
    elseif j==3 && x==c
        tmp=[0 0];
    elseif j==3 && x~=c
        tmp=[(2*b*B*tmp2i/((x - c)*(denomi))) (2*b*tmp2s/((x - c)*(denoms)))];
    elseif j==4
        tmp=[B 1];
    end


function tmp=dconsequent_dp(anfis,i, j)
    wn=anfis.nodes(i-anfis.nr,1); 
    inp=anfis.nodes(1:anfis.ni,1)';
    inp=[inp 1];
    tmp=wn*inp(j);