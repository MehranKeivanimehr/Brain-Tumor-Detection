function [output]= Pre_Processing(input)

input=imresize(input,[256,256]);
if size(input,3)>1
    input=rgb2gray(input);
end
[M N] = size(input);
n = zeros(M+2,N+2);
median = zeros(M+2,N+2);
[R C] = size(n);
n(2:R-1,2:C-1) = input(1:end,1:end);
median(2:R-1,2:C-1) = input(1:end,1:end);
for i = 2:R-1
    for j = 2:C-1
        temp = [median(i-1,j-1) median(i-1,j) median(i-1,j+1);median(i,j-1) median(i,j) median(i,j+1);....
                median(i+1,j-1) median(i+1,j) median(i+1,j+1)];
            mediansort = sort(temp(:),'ascend');
            median(i,j) = mediansort(5);
    end
end
resmedian = median(2:R-1,2:C-1);
output = uint8(resmedian);
end