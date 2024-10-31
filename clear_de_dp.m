function  anfis=clear_de_dp(anfis)

anfis.cparam_de_do=zeros(anfis.nr,anfis.ni+1);
anfis.mfparam_de_do=zeros(anfis.ni*anfis.mf,5);