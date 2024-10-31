function kalman_data=get_kalman_data(anfis,target)

kalman_data=zeros((anfis.ni+1)*anfis.nr+1,1);
st=anfis.ni+anfis.ni*anfis.mf+anfis.nr;

j=1;
for i=st+1:st+anfis.nr
    for k=1:anfis.ni
        kalman_data(j)=anfis.nodes(i,1)*anfis.nodes(k,1);
        j=j+1;
    end
    kalman_data(j)=anfis.nodes(i,1);
    j=j+1;
end
kalman_data(j)=target;