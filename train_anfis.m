function [it2anfis,y_anfis,RMSE]=train_anfis(data,epoch_n,mf,step_size,decrease_rate,increase_rate,B)

% Get inputs and output set
inputs=data(:,1:end-1);
output=data(:,end);
% Get data lenght
ndata=size(data,1);
% Get number of input
ni=size(inputs,2);
% Get number of rules
nr=mf^ni;
% Compute total number of nodes of net
nn=ni+ni*mf+3*nr+1;
% Define min RMSE for compare
min_RMSE=999999999999;

% Get minimum and maximum of input to determine initial antecedent params
mn=min(inputs);
mx=max(inputs);
mm=mx-mn;

% Define initial antecedent params
mfparams=[];
for i=1:ni
    tmp=[linspace(mn(i),mx(i),mf)', ones(mf,1)*0.5];%rand(mf, 1)];
    mfparams=[mfparams;repmat([mm(i)/5 mm(i)*0.24 1],mf,1) tmp];
end

%mfparams = [8.64544920354257,10.3903179497304,1.02273649518441,-21.4920295204689,0.216231844285772;8.68811233850500,10.4163420631294,1.09707278464007,0.179749614262325,0.501472099803036;8.65151883670725,10.3901471311688,1.07619679287472,21.8541361513947,0.100000000000000;23.9974482541038,28.7979272267009,1.04074027816559,3.99874747531609,0.262003157155786;24.0066875772785,28.8051268882208,0.911905205120089,63.9986713337852,0.999109092988576;23.9982393338416,28.7978639895774,1.01239617332851,124.001364017280,0.383438245517585];

% Define initial consequent parameters in zero
cparams=zeros(nr,(ni+1));
% Define the configuration matrix and the node matrix for the output of each layer
config=zeros(nn);
nodes=zeros(nn,2);

% Layer 1 connections of configuration
st=ni;
for i=1:ni
    config(i,st+[1:mf])=1;
    st=st+mf;
end

% Layer 2 connections of configuration
st=ni+ni*mf+1;

n=ni;
x=linspace(ni+1,ni+mf,mf)';
d=[];

while n>1
    c=1;
    for i=1:size(x,1)
        for j=1:mf
            d(c,:)=[x(i,:) j+ni+mf*(ni-n+1)];
            c=c+1;
        end
    end
    n=n-1;
    if n>1
        x=d;
        d=[];
    end
end

for i=1:mf^ni
    for j=1:ni
        config(d(i,j),st)=1;
    end
    st=st+1;
end

% Layer 3 connections of configuration
for i=1:nr
    for j=1:nr
        config(ni+ni*mf+i,ni+ni*mf+nr+j)=1;
    end
end

% Layer 4 connections of configuration
for i=1:nr
    config(ni+ni*mf+nr+i,ni+ni*mf+2*nr+i)=1;
end


% Layer 5 connections of configuration
for i=1:nr
    config(ni+ni*mf+2*nr+i,end)=1;
end

% Layer 6 connections of configuration
for i=1:ni
    for j=1:nr
        config(i,ni+ni*mf+2*nr+j)=1;
    end
end

% Define global parameters
anfis.config=config;
anfis.mfparams=mfparams;
anfis.cparams=cparams;
anfis.nodes=nodes;
anfis.ni=ni;
anfis.mf=mf;
anfis.nr=nr;
anfis.nn=nn;
anfis.last_decrease_ss=1;
anfis.last_increase_ss=1;
anfis.rules=d;

% Training loop
for i=1:epoch_n
    for j=1:ndata
        % set j th input into the nodes
        anfis.nodes(1:anfis.ni,1)=inputs(j,:)';
        anfis.nodes(1:anfis.ni,2)=inputs(j,:)';
        
        % Get node outputs from layer 1 to layer 6
        anfis=output1(anfis);
        anfis=output2(anfis);
        anfis=output3_4_5_6(anfis);
        
        % Save outputs of layer 1 to 6
        output_1_to_6(:,(j*2-1):j*2)=anfis.nodes;
        
        % Calculate Kalman params
        kalman_data=get_kalman_data(anfis,output(j));
        % Update Kalman params
        anfis=kalman_filter(anfis,kalman_data,j);
    end
    
    % Clear all derivatives as zero
    anfis=clear_de_dp(anfis);
    
    for j=1:ndata
        
        % Get output of layer 1 to 6
        anfis.nodes=output_1_to_6(:,(j*2-1):j*2);
        
        % Get node output of layer 7
        anfis=output7(anfis);
        
        % Get node output of layer 8
        anfis=output8(anfis);
        
        % Calculate anfis output
        y_anfis(j,1)=anfis.nodes(end,1);
        target=output(j);
        % Calculate differential of error
        de_dout = -2*(target - y_anfis(j,1));
        % Backpropagete errors
        anfis=calculate_de_do(anfis,de_dout);
        anfis=update_de_do(anfis);
        
    end
    
    % Calculate one train loop error
    diff=y_anfis-output;
    total_squared_error=sum(diff.*diff);
    RMSE(i,1) = sqrt(total_squared_error/ndata);
    fprintf('Epoch %g. rmse error: %g \n',i,RMSE(i,1));
    % If error is the best up to now then keep it
    if RMSE(i,1)<min_RMSE        
        it2anfis=anfis;
        min_RMSE=RMSE(i,1);
    end
    
    % Update membership parameter
    anfis=update_parameter(anfis, step_size);
    % Update step size
    [anfis step_size]=update_step_size(anfis,RMSE,i,step_size,decrease_rate, increase_rate);
end


% Calculate anfis output with better results
anfis=it2anfis;
for j=1:ndata
    anfis.nodes(1:anfis.ni,1)=inputs(j,:)';
    anfis.nodes(1:anfis.ni,2)=inputs(j,:)';
    anfis=output1(anfis);
    anfis=output2(anfis);
    anfis=output3_4_5_6(anfis);
    anfis=output7(anfis);
    anfis=output8(anfis);
    y_anfis(j,1)=anfis.nodes(end,1);
end