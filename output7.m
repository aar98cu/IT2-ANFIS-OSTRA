function anfis=output7(anfis)
    st=anfis.ni+anfis.ni*anfis.mf +2*anfis.nr;

    inp=anfis.nodes(1:anfis.ni,1)';
    cparam=anfis.cparams;

    for i=1:anfis.nr
        wn=anfis.nodes(i+st-anfis.nr,1);
        anfis.nodes(i+st,1)=wn*(sum(cparam(i,1:end-1).*inp)+cparam(i,end));
    end
    
    
    
    
    
    
    
    

