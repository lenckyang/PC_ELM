function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = PC_elm(train_data, test_data, No_of_Output, NumberofHiddenNeurons, ActivationFunction)

% Usage: elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% No_of_Output          - Number of outputs for regression
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression

%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);
%train_data=train_data(:,2:9);

T=train_data(:,size(train_data,2)+1-No_of_Output:size(train_data,2))';
[yy,ps] = mapminmax(T);
ps.ymin = 0;
[yy,ps] = mapminmax(T,ps);
T=yy;
P=train_data(:,1:size(train_data,2)-No_of_Output)';
P=mapminmax(P);
clear train_data;                          %   Release raw training data array
yym=T;
%%%%%%%%%%% Load testing dataset
%test_data=load(TestingData_File);
%test_data=test_data(:,2:9);
TV.T=test_data(:,size(test_data,2)+1-No_of_Output:size(test_data,2))';
[y,ps] = mapminmax(TV.T);
ps.ymin = 0;
[y,ps] = mapminmax(TV.T,ps);
TV.T=y;
TV.P=test_data(:,1:size(test_data,2)-No_of_Output)';
TV.P=mapminmax(TV.P);
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%% Calculate weights & biases


%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
                                        %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';
     %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
                                               



TrainingAccuracy=sqrt(mse(T - Y))   ;            %   Calculate training accuracy (RMSE) for regression case
clear H;

InputWeight11=[];
BiasofHiddenNeurons11=[];
OutputWeight11=[];

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
start_time_train=cputime;
kkk=NumberofHiddenNeurons;
for k=1:kkk%
%intial

if k==1
   T=T-Y;
%[x_best,z,fem_min,T1]=EM(40,P,T,NumberofTrainingData);
%[x_best,z,fem_min,T1]=PCOA(40,P,T,NumberofTrainingData);
I_InputWeight=(rand(1,NumberofInputNeurons)*2-1);
I_BiasofHiddenNeurons=rand(1,1);

InputWeight11=[InputWeight;I_InputWeight];

BiasofHiddenNeurons11=[BiasofHiddenNeurons;I_BiasofHiddenNeurons];
ind=ones(1,NumberofTrainingData);
BiasMatrix=I_BiasofHiddenNeurons(:,ind);  
I_tempH=I_InputWeight*P;

II_tempH=I_tempH+BiasMatrix;
H = 1 ./ (1 + exp(-II_tempH));
T1=T;
I_OutputWeight=pinv(H') * T1';
OutputWeight11=[OutputWeight;I_OutputWeight];
Y=(H' * I_OutputWeight)';



T1=T1-Y ; 

%test
tempH_test=InputWeight11*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons11(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight11)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
 %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

Testingyym(k)=sqrt(mse(TV.T - TY))  ;          %   Calculate testing accuracy (RMSE) for regression case


    end




    
    %generate input increasment input weight
if k>1

  

[x_best,z,fem_min]=PCOA(40,P,T1,NumberofTrainingData,NumberofInputNeurons);
InputWeight=x_best(1:NumberofInputNeurons)';
BiasofHiddenNeurons=x_best(NumberofInputNeurons+1);
%InputWeight=(rand(1,NumberofInputNeurons)*2-1);
InputWeight11=[InputWeight11;InputWeight];
%BiasofHiddenNeurons=rand(1,1);
BiasofHiddenNeurons11=[BiasofHiddenNeurons11;BiasofHiddenNeurons];

tempH=InputWeight*P;
%clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix1=BiasofHiddenNeurons(:,ind); %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H

tempH_N=tempH+BiasMatrix1;

H1 = 1 ./ (1 + exp(-tempH_N));
OutputWeight1=pinv(H1') * T1';      
OutputWeight11=[OutputWeight11;OutputWeight1];
%H=[H;H1];

%
Y=(H1' * OutputWeight1)';
 T1=T1-Y;

tempH_test=InputWeight11*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons11(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight11)';                       %   TY: the actual output of the testing data
end_time_test=cputime;

Testingyym(k)=sqrt(mse(TV.T - TY))   ;         %   Calculate testing accuracy (RMSE) for regression case

end

it1=sqrt(mse(T1));

end
%删除无用节点

%for ii=1:k+1
 %   if abs(OutputWeight11(ii,1))<0.001 
  %      OutputWeight11(ii,1)=0;
   % end
%end

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train   
TrainingAccuracy=it1;
%%%%%%%%%%% Calculate the output of testing input
tempH_test=InputWeight11*P;
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons11(:,ind);     
tempH_test=tempH_test + BiasMatrix;
 H_test = 1 ./ (1 + exp(-tempH_test));
TY=(H_test' * OutputWeight11)';   
%train_data=load(TrainingData_File);
%train_data=train_data(:,4:11);
%T=train_data(:,1:No_of_Output)';
accurcy=sqrt(mse(yy - TY))  



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_time_test=cputime;
tempH_test=InputWeight11*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons11(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight11)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

TestingAccuracy=sqrt(mse(TV.T - TY))            %   Calculate testing accuracy (RMSE) for regression case
