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
% 1.3 Output from one dimension to four dimensions：0——normal，1——gain，2——hemi_loss，3——homo_loss;
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

%% 参数设置
popsize = 200;                      % 种群大小
bestsize = 5;                       % 优胜子种群个数
tempsize = 5;                       % 临时子种群个数
SG = popsize / (bestsize+tempsize); % 子群体大小
S1 = size(ginputn,1);              % 输入层神经元个数
S2 = 15;                            % 隐含层神经元个数
S3 = size(outputn,1);              % 输出层神经元个数
iter = 10;                          % 迭代次数

%% 随机产生初始种群
initpop = initpop_generate(popsize,S1,S2,S3,ginputn,outputn);

%% 产生优胜子群体和临时子群体
% 得分排序
[sort_val,index_val] = sort(initpop(:,end),'descend');
% 产生优胜子种群和临时子种群的中心
bestcenter = initpop(index_val(1:bestsize),:);
tempcenter = initpop(index_val(bestsize+1:bestsize+tempsize),:);
% 产生优胜子种群
bestpop = cell(bestsize,1);
for i = 1:bestsize
    center = bestcenter(i,:);
    bestpop{i} = subpop_generate(center,SG,S1,S2,S3,ginputn,outputn);
end
% 产生临时子种群
temppop = cell(tempsize,1);
for i = 1:tempsize
    center = tempcenter(i,:);
    temppop{i} = subpop_generate(center,SG,S1,S2,S3,ginputn,outputn);
end
while iter > 0
    %% 优胜子群体趋同操作并计算各子群体得分
    best_score = zeros(1,bestsize);
    best_mature = cell(bestsize,1);
    for i = 1:bestsize
        best_mature{i} = bestpop{i}(1,:);
        best_flag = 0;                % 优胜子群体成熟标志(1表示成熟，0表示未成熟)
        while best_flag == 0
            % 判断优胜子群体是否成熟
            [best_flag,best_index] = ismature(bestpop{i});
            % 若优胜子群体尚未成熟，则以新的中心产生子种群
            if best_flag == 0
                best_newcenter = bestpop{i}(best_index,:);
                best_mature{i} = [best_mature{i};best_newcenter];
                bestpop{i} = subpop_generate(best_newcenter,SG,S1,S2,S3,ginputn,outputn);
            end
        end
        % 计算成熟优胜子群体的得分
        best_score(i) = max(bestpop{i}(:,end));
    end
    %% 临时子群体趋同操作并计算各子群体得分
    temp_score = zeros(1,tempsize);
    temp_mature = cell(tempsize,1);
    for i = 1:tempsize
        temp_mature{i} = temppop{i}(1,:);
        temp_flag = 0;                % 临时子群体成熟标志(1表示成熟，0表示未成熟)
        while temp_flag == 0
            % 判断临时子群体是否成熟
            [temp_flag,temp_index] = ismature(temppop{i});
            % 若临时子群体尚未成熟，则以新的中心产生子种群
            if temp_flag == 0
                temp_newcenter = temppop{i}(temp_index,:);
                temp_mature{i} = [temp_mature{i};temp_newcenter];
                temppop{i} = subpop_generate(temp_newcenter,SG,S1,S2,S3,ginputn,outputn);
            end
        end
        % 计算成熟临时子群体的得分
        temp_score(i) = max(temppop{i}(:,end));
    end
    %% 异化操作
    [score_all,index] = sort([best_score temp_score],'descend');
    % 寻找临时子群体得分高于优胜子群体的编号
    rep_temp = index(find(index(1:bestsize) > bestsize)) - bestsize;
    % 寻找优胜子群体得分低于临时子群体的编号
    rep_best = index(find(index(bestsize+1:end) < bestsize+1) + bestsize);
    
    % 若满足替换条件
    if ~isempty(rep_temp)
        % 得分高的临时子群体替换优胜子群体
        for i = 1:length(rep_best)
            bestpop{rep_best(i)} = temppop{rep_temp(i)};
        end
        % 补充临时子群体，以保证临时子群体的个数不变
        for i = 1:length(rep_temp)
            temppop{rep_temp(i)} = initpop_generate(SG,S1,S2,S3,ginputn,outputn);
        end
    else
        break;
    end
    %% 输出当前迭代获得的最佳个体及其得分
    if index(1) < 6
        best_individual = bestpop{index(1)}(1,:);
    else
        best_individual = temppop{index(1) - 5}(1,:);
    end

    iter = iter - 1;
    
end

%% 解码最优个体
x = best_individual;

% 前S1*S2个编码为W1
temp = x(1:S1*S2);
W1 = reshape(temp,S2,S1);

% 接着的S2*S3个编码为W2
temp = x(S1*S2+1:S1*S2+S2*S3);
W2 = reshape(temp,S3,S2);

% 接着的S2个编码为B1
temp = x(S1*S2+S2*S3+1:S1*S2+S2*S3+S2);
B1 = reshape(temp,S2,1);

%接着的S3个编码B2
temp = x(S1*S2+S2*S3+S2+1:end-1);
B2 = reshape(temp,S3,1);

%% 创建/训练BP神经网络
net_optimized = newff(ginputn,outputn,S2);
% 设置训练参数
net_optimized.trainParam.epochs = 200;
net_optimized.trainParam.show = 10;
net_optimized.trainParam.goal = 1e-4;
net_optimized.trainParam.lr = 0.1;
% 设置网络初始权值和阈值
net_optimized.IW{1,1} = W1;
net_optimized.LW{2,1} = W2;
net_optimized.b{1} = B1;
net_optimized.b{2} = B2;
% 利用新的权值和阈值进行训练
net_optimized = train(net_optimized,ginputn,goutput_train);
save ('D:\Matlab\bin\matlab_procedure\MEA\MEABP','net_optimized');













% %% 2.BP neural network training
% % 2.1 Initialize network structure
% 
% %节点个数
% inputnum=4;
% hiddennum=15;
% outputnum=4;
% 
% net=newff(ginputn,goutput_train,hiddennum);  %5 intermediate nodes
% 
% %% 遗传算法参数初始化
% maxgen=20;                         %进化代数，即迭代次数
% sizepop=10;                        %种群规模
% pcross=[0.2];                       %交叉概率选择，0和1之间
% pmutation=[0.1];                    %变异概率选择，0和1之间
% 
% %节点总数
% numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
% 
% lenchrom=ones(1,numsum);        
% bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %数据范围
% 
% %------------------------------------------------------种群初始化--------------------------------------------------------
% individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %将种群信息定义为一个结构体
% avgfitness=[];                      %每一代种群的平均适应度
% bestfitness=[];                     %每一代种群的最佳适应度
% bestchrom=[];                       %适应度最好的染色体
% %初始化种群
% for i=1:sizepop
%     %随机产生一个种群
%     individuals.chrom(i,:)=Code(lenchrom,bound);    %编码（binary和grey的编码结果为一个实数，float的编码结果为一个实数向量）
%     x=individuals.chrom(i,:);
%     %计算适应度
%     individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);   %染色体的适应度
% end
% FitRecord=[];
% %找最好的染色体
% [bestfitness bestindex]=min(individuals.fitness);
% bestchrom=individuals.chrom(bestindex,:);  %最好的染色体
% avgfitness=sum(individuals.fitness)/sizepop; %染色体的平均适应度
% % 记录每一代进化中最好的适应度和平均适应度
% trace=[avgfitness bestfitness]; 
%  
% %% 迭代求解最佳初始阀值和权值
% % 进化开始
% for i=1:maxgen
% 
%     % 选择
%     individuals=Select(individuals,sizepop); 
%     avgfitness=sum(individuals.fitness)/sizepop;
%     %交叉
%     individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
%     % 变异
%     individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
%     
%     % 计算适应度 
%     for j=1:sizepop
%         x=individuals.chrom(j,:); %解码
%         individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);   
%     end
%     
%   %找到最小和最大适应度的染色体及它们在种群中的位置
%     [newbestfitness,newbestindex]=min(individuals.fitness);
%     [worestfitness,worestindex]=max(individuals.fitness);
%     % 代替上一次进化中最好的染色体
%     if bestfitness>newbestfitness
%         bestfitness=newbestfitness;
%         bestchrom=individuals.chrom(newbestindex,:);
%     end
%     individuals.chrom(worestindex,:)=bestchrom;
%     individuals.fitness(worestindex)=bestfitness;
%     
%     avgfitness=sum(individuals.fitness)/sizepop;
%     
%     trace=[trace;avgfitness bestfitness]; %记录每一代进化中最好的适应度和平均适应度
%     FitRecord=[FitRecord;individuals.fitness];
% end
% 
% %% 遗传算法结果分析 
% figure(1)
% [r c]=size(trace);
% plot([1:r]',trace(:,2),'b--');
% title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
% xlabel('进化代数');ylabel('适应度');
% legend('平均适应度','最佳适应度');
% disp('适应度                   变量');
% 
% %% 把最优初始阀值权值赋予网络预测
% % %用遗传算法优化的BP网络进行值预测
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
% % 网络训练
% net.trainParam.epochs=200; %The 200 iteration
% net.trainParam.lr=0.1; %Learning rate is 0.1
% net.trainParam.goal=0.00004;
% % 2.2 Network training
% net=train(net,ginputn,goutput_train);
% % 2.3 Network preservation
% save ('BPNN','net');
% 
% 
