%% BP neural network training program

%% Empty environment variable
clc
clear

%% 1.Import training data
data1=load('E:\Desktop\data\trains\0.2_4x_sim1_4_4100_trains.mat');
data2=load('E:\Desktop\data\trains\0.2_6x_sim1_6_6100_trains.mat');
data3=load('E:\Desktop\data\trains\0.3_4x_sim1_4_4100_trains.mat');
data4=load('E:\Desktop\data\trains\0.3_6x_sim1_6_6100_trains.mat');
data5=load('E:\Desktop\data\trains\0.4_4x_sim1_4_4100_trains.mat');
data6=load('E:\Desktop\data\trains\0.4_6x_sim1_6_6100_trains.mat');
data_trains1 = data1.('sim1_4_4100_read_trains');
data_trains2 = data2.('sim1_6_6100_read_trains');
data_trains3 = data3.('sim1_4_4100_read_trains');
data_trains4 = data4.('sim1_6_6100_read_trains');
data_trains5 = data5.('sim1_4_4100_read_trains');
data_trains6 = data6.('sim1_6_6100_read_trains');
data_trains=[data_trains1;data_trains2;data_trains3;data_trains4;data_trains5;data_trains6;];  %Merge six training data files into one file
column=[2,3,4,5];
[m1,n1] = size(data_trains);
trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:); %The middle four columns of data file are characteristic values
% 1.1 Random sorting from 1 to trainlines
k=rand(1,trainLines);
[m,n]=sort(k);
% 1.2 Input eigenvalue data and output label data
ginput=gdata(:,column);
goutput1 =gdata(:,6);
% 1.3 Output from one dimension to four dimensions��0����normal��1����gain��2����hemi_loss��3����homo_loss;
goutput=zeros(trainLines,4);
for i=1:trainLines
    switch goutput1(i)
        case 0
            goutput(i,:)=[1 0 0 0];
        case 1
            goutput(i,:)=[0 1 0 0];
        case 2
            goutput(i,:)=[0 0 1 0];
        case 3
            goutput(i,:)=[0 0 0 1];
    end
end
% 1.4 Training data
ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';
% 1.5 Normalization of input and output data
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

%% ��������
popsize = 200;                      % ��Ⱥ��С
bestsize = 5;                       % ��ʤ����Ⱥ����
tempsize = 5;                       % ��ʱ����Ⱥ����
SG = popsize / (bestsize+tempsize); % ��Ⱥ���С
S1 = size(ginputn,1);              % �������Ԫ����
S2 = 15;                            % ��������Ԫ����
S3 = size(outputn,1);              % �������Ԫ����
iter = 10;                          % ��������

%% ���������ʼ��Ⱥ
initpop = initpop_generate(popsize,S1,S2,S3,ginputn,outputn);

%% ������ʤ��Ⱥ�����ʱ��Ⱥ��
% �÷�����
[sort_val,index_val] = sort(initpop(:,end),'descend');
% ������ʤ����Ⱥ����ʱ����Ⱥ������
bestcenter = initpop(index_val(1:bestsize),:);
tempcenter = initpop(index_val(bestsize+1:bestsize+tempsize),:);
% ������ʤ����Ⱥ
bestpop = cell(bestsize,1);
for i = 1:bestsize
    center = bestcenter(i,:);
    bestpop{i} = subpop_generate(center,SG,S1,S2,S3,ginputn,outputn);
end
% ������ʱ����Ⱥ
temppop = cell(tempsize,1);
for i = 1:tempsize
    center = tempcenter(i,:);
    temppop{i} = subpop_generate(center,SG,S1,S2,S3,ginputn,outputn);
end
while iter > 0
    %% ��ʤ��Ⱥ����ͬ�������������Ⱥ��÷�
    best_score = zeros(1,bestsize);
    best_mature = cell(bestsize,1);
    for i = 1:bestsize
        best_mature{i} = bestpop{i}(1,:);
        best_flag = 0;                % ��ʤ��Ⱥ������־(1��ʾ���죬0��ʾδ����)
        while best_flag == 0
            % �ж���ʤ��Ⱥ���Ƿ����
            [best_flag,best_index] = ismature(bestpop{i});
            % ����ʤ��Ⱥ����δ���죬�����µ����Ĳ�������Ⱥ
            if best_flag == 0
                best_newcenter = bestpop{i}(best_index,:);
                best_mature{i} = [best_mature{i};best_newcenter];
                bestpop{i} = subpop_generate(best_newcenter,SG,S1,S2,S3,ginputn,outputn);
            end
        end
        % ���������ʤ��Ⱥ��ĵ÷�
        best_score(i) = max(bestpop{i}(:,end));
    end
    %% ��ʱ��Ⱥ����ͬ�������������Ⱥ��÷�
    temp_score = zeros(1,tempsize);
    temp_mature = cell(tempsize,1);
    for i = 1:tempsize
        temp_mature{i} = temppop{i}(1,:);
        temp_flag = 0;                % ��ʱ��Ⱥ������־(1��ʾ���죬0��ʾδ����)
        while temp_flag == 0
            % �ж���ʱ��Ⱥ���Ƿ����
            [temp_flag,temp_index] = ismature(temppop{i});
            % ����ʱ��Ⱥ����δ���죬�����µ����Ĳ�������Ⱥ
            if temp_flag == 0
                temp_newcenter = temppop{i}(temp_index,:);
                temp_mature{i} = [temp_mature{i};temp_newcenter];
                temppop{i} = subpop_generate(temp_newcenter,SG,S1,S2,S3,ginputn,outputn);
            end
        end
        % ���������ʱ��Ⱥ��ĵ÷�
        temp_score(i) = max(temppop{i}(:,end));
    end
    %% �컯����
    [score_all,index] = sort([best_score temp_score],'descend');
    % Ѱ����ʱ��Ⱥ��÷ָ�����ʤ��Ⱥ��ı��
    rep_temp = index(find(index(1:bestsize) > bestsize)) - bestsize;
    % Ѱ����ʤ��Ⱥ��÷ֵ�����ʱ��Ⱥ��ı��
    rep_best = index(find(index(bestsize+1:end) < bestsize+1) + bestsize);
    
    % �������滻����
    if ~isempty(rep_temp)
        % �÷ָߵ���ʱ��Ⱥ���滻��ʤ��Ⱥ��
        for i = 1:length(rep_best)
            bestpop{rep_best(i)} = temppop{rep_temp(i)};
        end
        % ������ʱ��Ⱥ�壬�Ա�֤��ʱ��Ⱥ��ĸ�������
        for i = 1:length(rep_temp)
            temppop{rep_temp(i)} = initpop_generate(SG,S1,S2,S3,ginputn,outputn);
        end
    else
        break;
    end
    %% �����ǰ������õ���Ѹ��弰��÷�
    if index(1) < 6
        best_individual = bestpop{index(1)}(1,:);
    else
        best_individual = temppop{index(1) - 5}(1,:);
    end

    iter = iter - 1;
    
end

%% �������Ÿ���
x = best_individual;

% ǰS1*S2������ΪW1
temp = x(1:S1*S2);
W1 = reshape(temp,S2,S1);

% ���ŵ�S2*S3������ΪW2
temp = x(S1*S2+1:S1*S2+S2*S3);
W2 = reshape(temp,S3,S2);

% ���ŵ�S2������ΪB1
temp = x(S1*S2+S2*S3+1:S1*S2+S2*S3+S2);
B1 = reshape(temp,S2,1);

%���ŵ�S3������B2
temp = x(S1*S2+S2*S3+S2+1:end-1);
B2 = reshape(temp,S3,1);

%% ����/ѵ��BP������
net_optimized = newff(ginputn,outputn,S2);
% ����ѵ������
net_optimized.trainParam.epochs = 200;
net_optimized.trainParam.show = 10;
net_optimized.trainParam.goal = 1e-4;
net_optimized.trainParam.lr = 0.1;
% ���������ʼȨֵ����ֵ
net_optimized.IW{1,1} = W1;
net_optimized.LW{2,1} = W2;
net_optimized.b{1} = B1;
net_optimized.b{2} = B2;
% �����µ�Ȩֵ����ֵ����ѵ��
net_optimized = train(net_optimized,ginputn,goutput_train);
save ('D:\Matlab\bin\matlab_procedure\MEA\MEABP','net_optimized');













% %% 2.BP neural network training
% % 2.1 Initialize network structure
% 
% %�ڵ����
% inputnum=4;
% hiddennum=15;
% outputnum=4;
% 
% net=newff(ginputn,goutput_train,hiddennum);  %5 intermediate nodes
% 
% %% �Ŵ��㷨������ʼ��
% maxgen=20;                         %��������������������
% sizepop=10;                        %��Ⱥ��ģ
% pcross=[0.2];                       %�������ѡ��0��1֮��
% pmutation=[0.1];                    %�������ѡ��0��1֮��
% 
% %�ڵ�����
% numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
% 
% lenchrom=ones(1,numsum);        
% bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %���ݷ�Χ
% 
% %------------------------------------------------------��Ⱥ��ʼ��--------------------------------------------------------
% individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ��
% avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��
% bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
% bestchrom=[];                       %��Ӧ����õ�Ⱦɫ��
% %��ʼ����Ⱥ
% for i=1:sizepop
%     %�������һ����Ⱥ
%     individuals.chrom(i,:)=Code(lenchrom,bound);    %���루binary��grey�ı�����Ϊһ��ʵ����float�ı�����Ϊһ��ʵ��������
%     x=individuals.chrom(i,:);
%     %������Ӧ��
%     individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);   %Ⱦɫ�����Ӧ��
% end
% FitRecord=[];
% %����õ�Ⱦɫ��
% [bestfitness bestindex]=min(individuals.fitness);
% bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ��
% avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��
% % ��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
% trace=[avgfitness bestfitness]; 
%  
% %% ���������ѳ�ʼ��ֵ��Ȩֵ
% % ������ʼ
% for i=1:maxgen
% 
%     % ѡ��
%     individuals=Select(individuals,sizepop); 
%     avgfitness=sum(individuals.fitness)/sizepop;
%     %����
%     individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
%     % ����
%     individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
%     
%     % ������Ӧ�� 
%     for j=1:sizepop
%         x=individuals.chrom(j,:); %����
%         individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);   
%     end
%     
%   %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
%     [newbestfitness,newbestindex]=min(individuals.fitness);
%     [worestfitness,worestindex]=max(individuals.fitness);
%     % ������һ�ν�������õ�Ⱦɫ��
%     if bestfitness>newbestfitness
%         bestfitness=newbestfitness;
%         bestchrom=individuals.chrom(newbestindex,:);
%     end
%     individuals.chrom(worestindex,:)=bestchrom;
%     individuals.fitness(worestindex)=bestfitness;
%     
%     avgfitness=sum(individuals.fitness)/sizepop;
%     
%     trace=[trace;avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
%     FitRecord=[FitRecord;individuals.fitness];
% end
% 
% %% �Ŵ��㷨������� 
% figure(1)
% [r c]=size(trace);
% plot([1:r]',trace(:,2),'b--');
% title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
% xlabel('��������');ylabel('��Ӧ��');
% legend('ƽ����Ӧ��','�����Ӧ��');
% disp('��Ӧ��                   ����');
% 
% %% �����ų�ʼ��ֵȨֵ��������Ԥ��
% % %���Ŵ��㷨�Ż���BP�������ֵԤ��
% w1=x(1:inputnum*hiddennum);
% B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
% w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
% B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
% B2=B2';
% 
% net.iw{1,1}=reshape(w1,hiddennum,inputnum);
% net.lw{2,1}=reshape(w2,outputnum,hiddennum);
% net.b{1}=reshape(B1,hiddennum,1);
% net.b{2}=B2;
% 
% % ����ѵ��
% net.trainParam.epochs=200; %The 200 iteration
% net.trainParam.lr=0.1; %Learning rate is 0.1
% net.trainParam.goal=0.00004;
% % 2.2 Network training
% net=train(net,ginputn,goutput_train);
% % 2.3 Network preservation
% save ('BPNN','net');
% 
% 
