function yhat=evalmyanfis(anfis,inputs)

ndata=size(inputs,1);

for j=1:ndata
    anfis.nodes(1:anfis.ni,1)=inputs(j,:)';
    anfis.nodes(1:anfis.ni,2)=inputs(j,:)';
    anfis=output1(anfis);
    anfis=output2(anfis);
    anfis=output3_4_5_6(anfis);
    anfis=output7(anfis);
    anfis=output8(anfis);
    yhat(j,1)=anfis.nodes(end,1);
end