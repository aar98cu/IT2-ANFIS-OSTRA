function anfis=kalman_filter(anfis,kalman_data,k)
%% written by Muhammet Balcilar, Franre
%  all rights reserved
k_p_n= (anfis.ni+1)*anfis.nr;
alpha = 1000000;

if k==1
    anfis.P=zeros(k_p_n,1);
    anfis.S=alpha*eye(k_p_n);
end

x=kalman_data(1:end-1);
y=kalman_data(end);

tmp1=(x'*anfis.S)';
denom=1+sum(tmp1.*x);
tmp1=(anfis.S*x);

tmp2=(x'*anfis.S)';
tmp_m=tmp1*tmp2';
tmp_m=-1/denom*tmp_m;
anfis.S=anfis.S+tmp_m;

diff=y-sum(x.*anfis.P);
tmp1=diff*(anfis.S*x);
anfis.P=anfis.P+tmp1;

anfis.cparams=reshape(anfis.P,[anfis.ni+1 anfis.nr])';