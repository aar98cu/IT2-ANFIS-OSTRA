clear all
clc
format long

% Parameters
name='ex2';         %File name
epoch_n=100;         %Epoch number
mf_n=2;             %Membership functions number
step_size=0.1;      %Step size
decrease_rate=0.5;  %Decrease rate
increase_rate=1.1;  %Increase rate
B=0.5;              %Initial h parameter of LMF

% Load data for training
data=load(strcat('Input/',name,'.txt'));

% Train IT2-ANFIS OSTRA
[it2anfis,y_anfis,RMSE]=train_anfis(data,epoch_n,mf_n,step_size,decrease_rate,increase_rate,B);

% Evaluate the best IT2-ANFIS
y_anfis=evalmyanfis(it2anfis,data(:,1:end-1));

%Calculate the RMSE
rmse=sqrt(sum((y_anfis-data(:,end)).^2)/size(data,1))

%Generates an .m file of the IT2FLS model
createfis(it2anfis,name);

%Figures
figure;
for i=1:(size(data,2)-1)
    subplot(1,(size(data,2)-1),i);
    x = min(data(:,i)):0.1:max(data(:,i));
    for j=1:mf_n
      y1 = gbellmf(x,it2anfis.mfparams((i-1)*2+j,[2 3 4]));
      y2 = gbellmf(x,it2anfis.mfparams((i-1)*2+j,[1 3 4]))*it2anfis.mfparams((i-1)*2+j,end);
      plot(x,y1)
      hold on; 
      plot(x,y2) 
    end
    msg=['Input ' num2str(i)];
    title(msg);
end
savefig(strcat('Model/Figures/',name,'-MF.fig'))

figure;plot(RMSE,'LineWidth',2);xlabel('iteration');
ylabel('RMSE');title('Error per iteration');
savefig(strcat('Model/Figures/',name,'-RMSE.fig'))

figure;
plot(data(:,end),'r','Linewidth',0.5);
hold on;
plot(y_anfis,'b','Linewidth',0.5);
legend({'Actual output','ANFIS output'});
savefig(strcat('Model/Figures/',name,'-Approach.fig'))