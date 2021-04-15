%% Simulation data prediction program

%% Empty environment variable
clc
clear

%% 1.Load training data as network parameters
data_g=load('E:\Desktop\data\groundtruth.mat');  %Import groundtruth file
data_gt=data_g.('groundtruth');
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
data_trains=[data_trains1;data_trains2;data_trains3;data_trains4;data_trains5;data_trains6];%The middle four columns of data file are characteristic values
column=[2,3,4,5];
[m1,n1] = size(data_trains);
trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);
g_column=[1,2];
groundtruth=data_gt(:,g_column); 
gtRev=fliplr(groundtruth(:,2)');  %
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

%% 2.Load neural network
load('-mat','D:\Matlab\bin\matlab_procedure\MEA\MEABP');

%% 3.Load test data and count results
num=50; %Each folder contains 50 files
TP_count_sum=0;
TPFP_count_sum=0;
bound = 0;
count_bias_sum=0;
boundary=[]; %每个bam的边界，用于生成箱图
num_boundary=0;
% 3.1 循环加载测试数据
mkdir('E:\Desktop\data\','0.2_4x_binnumber');
mkdir('E:\Desktop\data\','0.2_4x_allbinnumber');

for t=1:num
    % 3.1.1 加载仿真数据
    data2=load(['E:\Desktop\data\tests\SimulationData_mat\0.2_4x_mat\sim', num2str(t) ,'_4_4100_read_trains.mat']);
    data_tests = data2.(['sim', num2str(t) ,'_4_4100_read_trains']);
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
    % 4.4 预测误差
    error=BPoutput-goutput_test;
    abs_error=abs(error);
    errorsum=sum(abs(error));
    % 4.5 创建文件保存检测正确的CNV编号
    fid=fopen(['E:\Desktop\data\0.2_4x_binnumber\sim', num2str(t) ,'_4_4100_bin_number.txt'],'wt'); 
    TP_count=0;
    P_count=0;
    TPFP_count=0;
    k=1;
    kkk=1;
    % 4.6 将检测正确的CNV编号写入文件
    for qq=1:testLines
        if ( abs_error(2,qq) < abs_error(1,qq) && abs_error(2,qq) < abs_error(3,qq) && abs_error(2,qq) < abs_error(4,qq) && goutput_test(2,qq) == 1)
            fprintf(fid,'%d\t',ginput2_bin(qq));
            fprintf(fid,'gain\t1');
            fprintf(fid,'\n');
            continue;
        end
        if ( abs_error(3,qq) < abs_error(1,qq) && abs_error(3,qq) < abs_error(2,qq) && abs_error(3,qq) < abs_error(4,qq) && goutput_test(3,qq) == 1)
            fprintf(fid,'%d\t',ginput2_bin(qq));
            fprintf(fid,'hemi_loss\t2');
            fprintf(fid,'\n');
            continue;
        end
        if ( abs_error(4,qq) < abs_error(1,qq) && abs_error(4,qq) < abs_error(2,qq) && abs_error(4,qq) < abs_error(3,qq) && goutput_test(4,qq) == 1)
            fprintf(fid,'%d\t',ginput2_bin(qq));
            fprintf(fid,'homo_loss\t3');
            fprintf(fid,'\n');
            continue;
        end
    end
    % 4.8 创建文件保存所有检测的CNV编号（无论对错都保存下来）
    fidall=fopen(['E:\Desktop\data\0.2_4x_allbinnumber\sim', num2str(t) ,'_4_4100_allbinnumber.txt'],'wt'); 
    u=1;
    for p=1:testLines
        if ( abs_error(2,p) < abs_error(1,p) && abs_error(2,p) < abs_error(3,p) && abs_error(2,p) < abs_error(4,p))
            fprintf(fidall,'%d\t',ginput2_bin(p));
            fprintf(fidall,'gain\t1');
            fprintf(fidall,'\n');
            continue;
        end
        if ( abs_error(3,p) < abs_error(1,p) && abs_error(3,p) < abs_error(2,p) && abs_error(3,p) < abs_error(4,p))
            fprintf(fidall,'%d\t',ginput2_bin(p));
            fprintf(fidall,'hemi_loss\t2');
            fprintf(fidall,'\n');
            continue;
        end
        if ( abs_error(4,p) < abs_error(1,p) && abs_error(4,p) < abs_error(2,p) && abs_error(4,p) < abs_error(3,p))
            fprintf(fidall,'%d\t',ginput2_bin(p));
            fprintf(fidall,'homo_loss\t3');
            fprintf(fidall,'\n');
            continue;
        end
    end    
    fclose(fidall);
    binnumber=[];
    binnumber_tpfp=[];
    for q=1:testLines
        % 4.7 统计TP_count,P_count,TPFP_count,boundary
        if (( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)) 
            TP_count=TP_count+1;
            binnumber(k)=ginput2_bin(q);
            k=k+1;
        end
        if ( goutput_test(2,q) == 1 || goutput_test(3,q) == 1 || goutput_test(4,q) == 1 )
            P_count=P_count+1;
            binnumber_tpfp(kkk)=ginput2_bin(q);
            kkk=kkk+1;
        end
        if (  (abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) ) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q)) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q)) )
            TPFP_count=TPFP_count+1;
        end
    end
    fclose(fid);
    TPFP_count_sum=TPFP_count_sum+TPFP_count;
    TP_count_sum=TP_count_sum+TP_count;
    boundbias = (P_count - TP_count)./14;
    bound = bound + boundbias;
    
    % 将检测到的14条cnv的bin开头结尾的编号保存下来
    binnumberRev=fliplr(binnumber);
    [m3,n3]=size(binnumber);
    jj=1;
    bin1=[];
    for ii=1:n3
        if binnumber(ii) >= groundtruth(jj,1)
            bin1(jj)=binnumber(ii);
            jj=jj+1;
        end
        if jj>14
           break; 
        end
    end
    jjj=1;
    bin22=[];
    for iii=1:n3
        if binnumberRev(iii) <= gtRev(jjj)
            bin22(jjj)=binnumberRev(iii);
            jjj=jjj+1;
        end
        if jjj>14
           break; 
        end
    end
    bin2=fliplr(bin22);
    [m5,n5]=size(bin1);
    [m6,n6]=size(bin2);
    if(n5<14 || n6<14)
       continue; 
    end
    bin=[bin1;bin2]'; 
    % 计算边界精度
    [m4,n4]=size(bin);
    c_bias1=[];
    c_bias2=[];
    for rr=1:m4
        c_bias1(rr)=bin(rr,1)-groundtruth(rr,1);
        c_bias2(rr)=groundtruth(rr,2)-bin(rr,2);
    end
    c_bias=sum(c_bias1)+sum(c_bias2);
    count_bias=c_bias./14;
    if count_bias > 50
        continue;
    end
    boundary(t)=count_bias;
    num_boundary = num_boundary + 1;
    count_bias_sum = count_bias_sum+count_bias;
    binnumber_fp=setdiff(binnumber_tpfp,binnumber);
    
end

%% 5.统计最终结果recall，precision，F1-score,boundary bias
TP_count_avg=TP_count_sum/num;
TPFP_count_avg=TPFP_count_sum/num;
recall=TP_count_avg/P_count;
precision=TP_count_avg/TPFP_count_avg;
F1_score=(2*recall*precision)/(recall+precision);
boundary_bias=bound/num;
count_bias_avg=count_bias_sum/num_boundary;
disp('sensitivity:');
disp(recall);
disp('precision:');
disp(precision);
disp('F1-score:');
disp(F1_score);
disp('all_bound_bias:');
disp(boundary_bias);
disp('14_bound_bias:');
disp(count_bias_avg);
disp('bound_bias:');
disp(boundary);
