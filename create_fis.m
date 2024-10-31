function anfisit2=create_fis(anfisit2,data,name)

fis = sugfistype2('Name',name);
for i=1:anfisit2.ni
    fis = addInput(fis,[min(data(:,i)) max(data(:,i))],'Name',join(["input",num2str(i)],''));
    for j=1:anfisit2.mf
        fis = addMF(fis,join(["input",num2str(i)],''),"gbellmf",anfisit2.mfparams((i-1)*2+j,2:4),'LowerScale',anfisit2.mfparams((i-1)*2+j,5),'LowerLag',calculate_lower_lag(anfisit2,i,j),'Name',join(["in",num2str(i),"mf",num2str(j)],''));
    end
end
fis = addOutput(fis,[min(data(:,end)) max(data(:,end))],'Name',"output");
for i=1:anfisit2.mf^anfisit2.ni
    fis = addMF(fis,"output","linear",anfisit2.cparams(i,:),'Name',join(["out1mf",num2str(i)],''));
end

x=[1;2];
n=anfisit2.ni;
d=[];

while n>1
    c=1;
    for i=1:size(x,1)
        for j=1:2
            d(c,:)=[x(i,:) j];
            c=c+1;
        end
    end
    n=n-1;
    if n>1
        x=d;
        d=[];
    end
end

for i=1:anfisit2.mf^anfisit2.ni
    fis = addRule(fis,[d(i,:) i 1 1]);
end

writeFIS(fis,strcat('Model/',name));

end

function lower_lag=calculate_lower_lag(anfisit2,i,j)
    ai=anfisit2.mfparams((i-1)*2+j,1);
    as=anfisit2.mfparams((i-1)*2+j,2);
    b=anfisit2.mfparams((i-1)*2+j,3);
    c=anfisit2.mfparams((i-1)*2+j,4);
    ex = ai*nthroot(124,2*b)+c;
    lower_lag = 1/(1+((((ex-c)/as)^(2))^b));
end