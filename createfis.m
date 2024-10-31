function createfis(it2anfis,name)
% Open a new file.
fileID = fopen(strcat('Model/',name,'.m'), 'wt');

fprintf(fileID, strcat('function y = ',name,'(inputs)\n'));
fprintf(fileID, ['    NumInputs=' num2str(it2anfis.ni) ';\n']);
fprintf(fileID, ['    NumMF=' num2str(it2anfis.mf) ';\n']);
fprintf(fileID, ['    NumRules=' num2str(it2anfis.nr) ';\n']);
fprintf(fileID, '    NumData=size(inputs,1);\n');
fprintf(fileID, ['    Nodes=zeros(' num2str(it2anfis.nn) ',2);\n']);
for i=1:it2anfis.ni*it2anfis.mf
    if i==1
        fprintf(fileID, ['    MFparams=[' num2str(it2anfis.mfparams(i,:)) ';\n']);
    elseif i==it2anfis.ni*it2anfis.mf
        fprintf(fileID, ['    ' num2str(it2anfis.mfparams(i,:)) '];\n']);
    else
        fprintf(fileID, ['    ' num2str(it2anfis.mfparams(i,:)) ';\n']);
    end
end
for i=1:it2anfis.nr
    if i==1
        fprintf(fileID, ['    Cparams=[' num2str(it2anfis.cparams(i,:)) ';\n']);
    elseif i==it2anfis.nr
        fprintf(fileID, ['    ' num2str(it2anfis.cparams(i,:)) '];\n']);
    else
        fprintf(fileID, ['    ' num2str(it2anfis.cparams(i,:)) ';\n']);
    end
end
for i=1:it2anfis.nr
    if i==1
        fprintf(fileID, ['    Rules=[' num2str(it2anfis.rules(i,:)) ';\n']);
    elseif i==it2anfis.nr
        fprintf(fileID, ['    ' num2str(it2anfis.rules(i,:)) '];\n']);
    else
        fprintf(fileID, ['    ' num2str(it2anfis.rules(i,:)) ';\n']);
    end
end
fprintf(fileID, '\n');
fprintf(fileID, '    for i=1:NumData\n');
fprintf(fileID, ['        Nodes(1:NumInputs,1)=inputs(i,:)' char(39) ';\n']);
fprintf(fileID, ['        Nodes(1:NumInputs,2)=inputs(i,:)' char(39) ';\n']);
fprintf(fileID, '\n');
fprintf(fileID, '        for j=1:NumInputs\n');
fprintf(fileID, '            for k=1:NumMF\n');
fprintf(fileID, '                ind=NumInputs+(j-1)*NumMF+k;\n');
fprintf(fileID, '                x = Nodes(j,1);\n');
fprintf(fileID, '                b = MFparams((j-1)*NumMF+k,3);\n');
fprintf(fileID, '                c = MFparams((j-1)*NumMF+k,4);\n');
fprintf(fileID, '                ai = MFparams((j-1)*NumMF+k,1);\n');        
fprintf(fileID, '                tmp1i = (x - c)/ai;\n');
fprintf(fileID, '                if tmp1i == 0\n');
fprintf(fileID, '                    tmp2i=0;\n');
fprintf(fileID, '                else\n');
fprintf(fileID, '                    tmp2i = (tmp1i*tmp1i)^b;\n');
fprintf(fileID, '                end\n');
fprintf(fileID, '                Nodes(ind,1)=MFparams((j-1)*NumMF+k,5)/(1+ tmp2i);\n');
fprintf(fileID, '                as = MFparams((j-1)*NumMF+k,2);\n');
fprintf(fileID, '                tmp1s = (x - c)/as;\n');
fprintf(fileID, '                if tmp1s == 0\n');
fprintf(fileID, '                    tmp2s=0;\n');
fprintf(fileID, '                else\n');
fprintf(fileID, '                    tmp2s = (tmp1s*tmp1s)^b;\n');
fprintf(fileID, '                end\n');
fprintf(fileID, '                Nodes(ind,2)=1/(1+ tmp2s);\n');
fprintf(fileID, '            end\n');
fprintf(fileID, '        end\n');
fprintf(fileID, '\n');
fprintf(fileID, '        st=NumInputs+NumInputs*NumMF;\n');
fprintf(fileID, '        for j=st+1:st+NumRules\n');
fprintf(fileID, ['            tmpi=cumprod(Nodes(Rules(j-st,:)' char(39) ',1));\n']);
fprintf(fileID, ['            tmps=cumprod(Nodes(Rules(j-st,:)' char(39) ',2));\n']);
fprintf(fileID, '            Nodes(j,1)=tmpi(end);\n');
fprintf(fileID, '            Nodes(j,2)=tmps(end);\n');
fprintf(fileID, '        end\n');
fprintf(fileID, '\n');
fprintf(fileID, '        st=NumInputs+NumInputs*NumMF;\n');
fprintf(fileID, ['        wi=Nodes(st+1:st+NumRules,1)' char(39) ';\n']);
fprintf(fileID, ['        ws=Nodes(st+1:st+NumRules,2)' char(39) ';\n']);
fprintf(fileID, '        [wi,O1]=sort(wi);\n');
fprintf(fileID, '        ws=sort(ws);\n');
fprintf(fileID, '        wn = (wi+ws)/2;\n');
fprintf(fileID, '        yi=sum(wi.*wn)/sum(wn);\n');
fprintf(fileID, '        ys=sum(ws.*wn)/sum(wn);\n');
fprintf(fileID, '        for j=1:NumRules-1\n');
fprintf(fileID, '            if wi(j)<=yi && yi<=wi(j+1)\n');
fprintf(fileID, '               l=j;\n');
fprintf(fileID, '            else\n');
fprintf(fileID, '                l=0;\n');
fprintf(fileID, '            end\n');
fprintf(fileID, '            if ws(j)<=ys && ys<=ws(j+1)\n');
fprintf(fileID, '               r=j;\n');
fprintf(fileID, '             else\n');
fprintf(fileID, '                r=0;\n');
fprintf(fileID, '            end\n');
fprintf(fileID, '        end\n');
fprintf(fileID, '        Xl=[ws(1:l) wi(l+1:NumRules)]/sum([ws(1:l) wi(l+1:NumRules)]);\n');
fprintf(fileID, '        Xr=[wi(1:r) ws(r+1:NumRules)]/sum([wi(1:r) ws(r+1:NumRules)]);\n');
fprintf(fileID, '        X=(Xl+Xr)/2;\n');
fprintf(fileID, '        X=X(O1);\n');
fprintf(fileID, '        st=NumInputs+NumInputs*NumMF+NumRules;\n');
fprintf(fileID, '        Nodes(st+1:st+NumRules,1)=X;\n');
fprintf(fileID, '        Nodes(st+1:st+NumRules,2)=X;\n');
fprintf(fileID, '\n');
fprintf(fileID, '        st=NumInputs+NumInputs*NumMF+2*NumRules;\n');
fprintf(fileID, ['        inp=Nodes(1:NumInputs,1)' char(39) ';\n']);
fprintf(fileID, '        for j=1:NumRules\n');
fprintf(fileID, '            wn=Nodes(j+st-NumRules,1);\n');
fprintf(fileID, '            Nodes(j+st,1)=wn*(sum(Cparams(j,1:end-1).*inp)+Cparams(j,end));\n');
fprintf(fileID, '        end\n');
fprintf(fileID, '\n');
fprintf(fileID, '        Nodes(end,1)=sum(Nodes(end-NumRules:end-1,1));\n');
fprintf(fileID, '        y(i,1)=Nodes(end,1);\n');
fprintf(fileID, '    end\n');
fprintf(fileID, 'end\n');

fclose(fileID);
% Open the file in the editor.
%edit('Model/it2_anfis_ostra.m');

end