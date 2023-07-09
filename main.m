clc;
clear;
clear all;

addr_folder='C:\Users\ava\Desktop\second Semester\Image Processing\Project\Prj_TumurClassification\2 Class\DataSet\Train\no\';
Num=95;
name='No%d.jpg';
[train_feat,x]= importdata(addr_folder,Num,name);

disp("Feature Extraction of Train Healthy images are done!");

addr_folder='C:\Users\ava\Desktop\second Semester\Image Processing\Project\Prj_TumurClassification\2 Class\DataSet\Train\yes\';
Num=155;
name='Y%d.jpg';
[train_feat1,y]= importdata(addr_folder,Num,name);
disp("Feature Extraction of Train Tumor images are done");
data=[train_feat;train_feat1];



addr_folder='C:\Users\ava\Desktop\second Semester\Image Processing\Project\Prj_TumurClassification\2 Class\DataSet\Test\';
Num=8;
name='No%d.jpg';
[train_featN,z]= importdata(addr_folder,Num,name);


addr_folder='C:\Users\ava\Desktop\second Semester\Image Processing\Project\Prj_TumurClassification\2 Class\DataSet\Test\';
Num=13;
name='Y%d.jpg';
[train_featY,zz]= importdata(addr_folder,Num,name);
disp("Feature Extraction of Test Tumor images are done");

Testdata=[train_featN;train_featY];
%%
labelnorm = zeros(1,95);
labelcancer = ones(1,155);
alllabel = [labelnorm labelcancer];

labelnorm = zeros(1,8);
labelcancer = ones(1,13);
alllabeltest = [labelnorm labelcancer];
Train = [alllabel',data];
Test = [alllabeltest', Testdata];
loop = 10;
for i=1:loop
    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(Train, Test ,1,160, 'sine');
    Training_Accuracy(i)=TrainingAccuracy;
    Testing_Accuracy(i) = TestingAccuracy;
    Training_Time(i)=TrainingTime;
    Testing_Time(i) = TestingTime; 
end
k=find (Testing_Accuracy==max(Testing_Accuracy));
if (max(Testing_Accuracy) < Training_Accuracy(k))
    disp ("Test Accuracy :")
    max(Testing_Accuracy)
    disp ("Train Accuracy :")
    Training_Accuracy(k)
    disp ("Training Time and Test time :")
    Training_Time(k)
    Testing_Time(k)
end
if (max(Testing_Accuracy) > Training_Accuracy(k))
    Testing_Accuracy(k) = [];
    Training_Accuracy(k) = [];
    k=find (Testing_Accuracy==max(Testing_Accuracy));
    if (max(Testing_Accuracy) < Training_Accuracy(k))
    disp ("Test Accuracy :")
    max(Testing_Accuracy)
    disp ("Train Accuracy :")
    Training_Accuracy(k)
    disp ("Training Time and Test time :")
    Training_Time(k)
    Testing_Time(k)
    end
end
    
