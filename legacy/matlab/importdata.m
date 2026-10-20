function [train_feat,j]= importdata(addr_folder,Num,name)
for i=1:Num  
[img{i}]=imread([addr_folder, sprintf(name,i)]);
% Pre-Processing
inp{i}=img{i};
[result{i}]=Pre_Processing(inp{i});
inp{i} = im2double(result{i}) ;
segmented{i}=gabor_filters(inp{i});
    bw_thresh=imbinarize(segmented{i},0.5);
    tumor= imfill(bw_thresh, 'holes');
    seD = strel('diamond',3);   %creates diamond shape structure
    tumor = imerode(tumor,seD);
    se =strel('square',5);
    tumor=imdilate(tumor,se); 

    segmented{i}=tumor;
% Feature Extraction using DWT
[feat{i}]=Feature_extraction(segmented{i});
[feat1{i}] = extractLBPFeatures(result{i}); 
end
train_feat=zeros(i, 69);
for j=1:i 
       train_feat(j,:)=[feat{1,j},feat1{1,j}]; 
end

end     