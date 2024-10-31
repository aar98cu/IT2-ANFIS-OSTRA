function anfis=calculate_de_do(anfis,de_dout)
    anfis.de_do=zeros(size(anfis.nodes()));
    anfis.de_do(end,:)=de_dout;

    for i=length(anfis.nodes)-1:-1:anfis.ni+1
        de_do=[0 0];
        II=find(anfis.config(i,:)==1);
        I=find(II>i);
        for j=1:length(I)
            jj=II(I(j));
            tmp1=anfis.de_do(jj,:);
            tmp2=derivative_o_o(anfis,i, jj);
            de_do = de_do+tmp1.*tmp2;
        end
        anfis.de_do(i,:)=de_do;
    end


function tmp=derivative_o_o(anfis,i, j)
    if i>anfis.ni+anfis.ni*anfis.mf +2*anfis.nr
        tmp=[1 1];
    elseif i>anfis.ni+anfis.ni*anfis.mf +anfis.nr
        tmp=do4_do3(anfis,i, j);
    elseif i>anfis.ni+anfis.ni*anfis.mf 
        tmp=do3_do2(anfis,i, j);
    elseif i>anfis.ni
        tmp=[anfis.nodes(j,1)/anfis.nodes(i,1) anfis.nodes(j,2)/anfis.nodes(i,2)];
    end


function tmp=do4_do3(anfis,i, j)
    cparam=anfis.cparams;
    inp=anfis.nodes(1:anfis.ni,1)';
    jj=j-anfis.ni-anfis.ni*anfis.mf-2*anfis.nr;
    tmp=[sum(cparam(jj,1:end-1).*inp)+cparam(jj,end) sum(cparam(jj,1:end-1).*inp)+cparam(jj,end)];

function tmp=do3_do2(anfis,i, j)

    %%Checar si es que no se cancelan las sumatorias

    II=find(anfis.config(:,j)==1);
    I=find(II<j);
    totali=sum(anfis.nodes(II(I),1));
    totals=sum(anfis.nodes(II(I),2));
    if j-i==anfis.nr
        tmp=[(totali-anfis.nodes(i,1))/(totali*totali) (totals-anfis.nodes(i,2))/(totals*totals)];
    else
        tmp=[-anfis.nodes(j-anfis.nr,1)/(totali*totali) -anfis.nodes(j-anfis.nr,2)/(totals*totals)];
    end