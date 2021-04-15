%% 真实数据预测程序

%% 清空环境变量
clc
clear

%% 1.加载训练数据，作为网络的参数
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
data_trains=[data_trains1;data_trains2;data_trains3;data_trains4;data_trains5;data_trains6];
column=[2,3,4,5];
[m1,n1] = size(data_trains);
trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);
% 1.1 从1到trainLines间随机排序
k=rand(1,trainLines);
[m,n]=sort(k);
% 1.2 得到输入输出数据
ginput=gdata(:,column);
goutput1 =gdata(:,6);
% 1.3 输出从一维变成四维
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
% 1.4 找出训练数据和预测数据
ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';
% 1.5 样本输入输出数据归一化
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

%% 2.加载网络
load('-mat','D:\Matlab\bin\matlab_procedure\MEA\MEABP');

%% 3.加载真实数据
datat=load('E:\Desktop\data\tests\RealData_mat\NA19238_tests.mat');
data_tests = datat.('NA19238_tests');
[m2,n2] = size(data_tests);
testLines=m2;
gdata2(1:testLines,:) = data_tests(1:m2,:);
ginput2_bin=gdata2(:,1);
ginput2=gdata2(:,column);
goutput1 =gdata2(:,6);
goutput2=zeros(testLines,4);
for i=1:testLines
    switch goutput1(i)
        case 0
            goutput2(i,:)=[1 0 0 0];
        case 1
            goutput2(i,:)=[0 1 0 0];
        case 2
            goutput2(i,:)=[0 0 1 0];
    case 3
            goutput2(i,:)=[0 0 0 1];
    end
end
ginput_test=ginput2((1:testLines),:)';
goutput_test=goutput2((1:testLines),:)';
%% 4.BP网络预测
% 4.1 预测数据归一化
inputn_test=mapminmax('apply',ginput_test,ginputps);
% 4.2 网络预测输出
an=sim(net_optimized,inputn_test);
% 4.3 网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

%% 5.结果分析
% 5.1 预测误差
error=BPoutput-goutput_test;
abs_error=abs(error);
errorsum=sum(abs(error));
% 5.2 将检测正确的CNV编号写入文件
fid=fopen('E:\Desktop\data\NA19238bin.txt','wt'); 
TP_count=0;
P_count=0;
TPFP_count=0;
bound = 0;
count_bias_sum=0;
boundary=[]; %每个bam的边界，用于生成箱图
num_boundary=0;

k=1;
kkk=1;
binnumber=[];
binnumber_tpfp=[];
for q=1:testLines
    if ( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));
        fprintf(fid,'gain\t1');
        fprintf(fid,'\n');
    end
    if ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));
        fprintf(fid,'hemi_loss\t2');
        fprintf(fid,'\n');
    end
    if ( abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));
        fprintf(fid,'homo_loss\t3');
        fprintf(fid,'\n');

    end
    % 5.3 统计recall，precision，F1-score
    if (( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)) 
        TP_count=TP_count+1;
        binnumber(k)=ginput2_bin(q);
        k=k+1;
    end
    if ( goutput_test(2,q) == 1 || goutput_test(3,q) == 1 || goutput_test(4,q) == 1 )
        P_count=P_count+1;
    end
    if ( (abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) ) )
        TPFP_count=TPFP_count+1;
        binnumber_tpfp(kkk)=ginput2_bin(q);
        kkk=kkk+1;
    end
end

fclose(fid);

%% 6.统计最终结果
recall=TP_count./P_count;
precision=TP_count./TPFP_count;
F1_score=(2*recall*precision)./(recall+precision);
disp('sensitivity:');
disp(recall);
disp('precision:');
disp(precision);
disp('F1-score:');
disp(F1_score);
