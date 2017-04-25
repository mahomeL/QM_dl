%% 读取切分后的五片段txt原始文件，进行合并
% 滑窗片段:1-70,26-95,51-.,76-.,101-.

% dirpath='/Volumes/liaoliaoluo/zjh_5pianduan/';
% outpath='/Volumes/liaoliaoluo/AUNC_DeepLearning/';
% filedir=dir([dirpath,'S']);
% 
% for k=258:5:length(filedir)
%     %length(filedir)
%     
%     X=[];
%     for i=1:5
%         filename=filedir(k+i-1).name;
%         subid=filename(1:7);
%         x=importdata([dirpath,'S/',filename]);%(70,32792)
%         x=x';
%         if i>1
%           x=x(1:32792,70-25+1:70);
%         end
%         X=[X,x];
%         save([outpath,'AUNC_oridata_mat/',subid,'_oridata.mat'],'X');
%     end
% end

%%
dirpath='/Volumes/liaoliaoluo/AUNC_DeepLearning/';
outpath='/Volumes/liaoliaoluo/AUNC_DeepLearning/';
filedir=dir([dirpath,'AU20']);

for k=4:2:length(filedir)
    %length(filedir)
    filename=filedir(k).name;
    subid=filename(1:7);
    X=importdata([dirpath,'AU20/',filename]);
    save([outpath,subid,'_oridata.mat'],'X');
end

