function anfis=update_parameter(anfis, step_size)

    tmp=anfis.mfparam_de_do;
    tmp=tmp.*tmp;
    len=sqrt(sum(tmp(:)));
    anfis.mfparams= anfis.mfparams - step_size * anfis.mfparam_de_do/len;
    C=anfis.mfparams(:,5);
    C(C > 1) = 1;
    C(C < 0.1) = 0.1;
    anfis.mfparams(:,5)=C;