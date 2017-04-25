%% 拓展32792数据成46，55，46的方块
addpath(genpath('/Users/l_mahome/Documents/py2_vtk/readVTK/fiber/Visualize'));
nii_path='/Volumes/liaoliaoluo/AUNC_DeepLearning';
addpath(genpath(nii_path));

nii_filenames=dir([nii_path,'/AUNC_nii_gz/NC']);

d3_size=[46,55,46];

for i=3:length(nii_filenames)
    disp(i);
    filename=nii_filenames(i).name;
    img=load_untouch_nii(filename);
    img=img.img;
    if sum(d3_size-size(img))~=0
        fprintf('ERROR');
        break;
    end
    
    sub_id=filename(1:7);
    data_name=[sub_id,'_oridata.mat'];
    load(data_name);

    data_4d=zeros(46,55,46,170);
    for k=1:32792
        [x,y,z]=ind2sub(d3_size,find(img==k));
        data_4d(x,y,z,:)=X(k,:);
    end
    save([nii_path,'/AUNC_oridata_mat_3d_46_55_46/NC/',sub_id,'_4d.mat'],'data_4d','img')
end
    
    
        
        
    
    