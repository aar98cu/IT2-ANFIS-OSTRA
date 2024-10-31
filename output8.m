function anfis=output8(anfis)
    anfis.nodes(end,1)=sum(anfis.nodes(end-anfis.nr:end-1,1));