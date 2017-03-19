#coding=utf-8

'''Reconstruct model input-output to fit wxpython'''
import os
import datetime
from keras.models import model_from_json
import cv2
import numpy as np
import pandas as pd
from keras import backend as K
import pickle
from collections import Counter
K.set_image_dim_ordering('th')

class ClfModel():
    def __init__(self,model_archi,model_weigh,input_pic,output_path):
        """
        load model ; set input pic and output result
        :param model_archi: archi path
        :param model_weigh: weigh path
        :param input_pic: input pic path
        :param output_path: output result path
        """
        assert isinstance(model_archi,str) and isinstance(model_weigh,str)
        assert model_archi.split('.')[-1]=='json','model architecture must be *.json'
        assert model_weigh.split('.')[-1]=='h5','model weights must be *.h5'

        self.model_archi = model_archi
        self.model_weigh = model_weigh
        self.input_files_path = input_pic
        self.output_path = output_path
        self.pic_width = 350
        self.pic_height = 350

        self.input_count = 0
        self.input_len = 0
        self.input_pic_name = []

        self.output_res = []
        self.output_res_summary = []
        self.output_res_error = []

        self.be_finished = False

        self.pic_classes_str2num= dict([('lower_body', 0), ('upper_body', 1), ('whole_body', 2)])
        self.pic_classes_num2str= dict([(v,k) for k,v in self.pic_classes_str2num.items() ])

        self.model = model_from_json(open(self.model_archi).read())
        self.model.load_weights(self.model_weigh)

    def _output_reset(self):
        """
        reset input-output result
        :return:
        """
        self.input_count = 0
        self.input_len = 0
        self.input_pic_name = []
        self.output_res = []
        self.output_res_summary = []
        self.output_res_error = []
        self.be_finished = False

    def _build_model_input(self,files_path):
        """
        read pictures and resize it
        :param files_path: input pictures path
        :return: generator to produce a resized picture
        """
        self._output_reset()
        assert os.path.isdir(files_path)
        files = os.listdir(files_path)
        while '.DS_Store' in files:
            files.remove('.DS_Store')
        self.input_len =len(files)
        print ('input files len: {}'.format(self.input_len))

        for pic in files:
            self.input_count += 1
            try:
                img = cv2.imread(os.path.join(files_path,pic))
            except :
                raise IOError

            self.input_pic_name.append(pic)

            img = cv2.resize(img,(self.pic_height, self.pic_width))
            img = img.astype(np.float32, copy=False)
            for c in range(3):
                img[:, :, c] = img[:, :, c] / 255.0
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            yield img

    def _model_predict(self):
        """
        predict pic by model,not valid
        :return: generator to produce predict result[pic-name,p-class,prob]
        """
        img = self._build_model_input(self.input_files_path)

        _tmp_pred_class = []
        _tmp_data = img.next()  # in order to get input_len
        for i in range(1,self.input_len+1):
            pred_prob = self.model.predict(_tmp_data)[0]
            cur_pic_name = self.input_pic_name[-1]
            try:
                _tmp_data = img.next()
            except StopIteration:
                self.be_finished = True
            pred_prob = np.array(pred_prob).tolist()
            pred_prob = map(lambda x:round(x,4),pred_prob)
            pred_class = np.argmax(pred_prob)
            _tmp_pred_class.append(pred_class)

            _tmp_res = [cur_pic_name,self.pic_classes_num2str[pred_class],
                        pred_prob[0],pred_prob[1],pred_prob[2]]
            self.output_res.append(_tmp_res)  # pic-name,p-class,prob

            if self.be_finished:
                self.output_res_summary.append(Counter(_tmp_pred_class))

                pd_res = pd.DataFrame(data=self.output_res,
                                      columns=['pic_name', 'predict_class', 'lower_body_prob', 'upper_body_prob',
                                               'whole_body_prob'])
                now = datetime.datetime.now()
                _out_name = 'qingmu_luw_'+now.strftime('%m-%d-%H-%M-%S') + '.csv'
                pd_res.to_csv(os.path.join(self.output_path, _out_name), header=True)


            yield _tmp_res #for printing output at once




    def _model_predict_for_wx(self,pic_idx):
        """
        //TODO for wxpython
        :param pic_idx:
        :return:
        """
        pass


    def _model_clf_valid(self,file_dir_path,true_class,verbose = True,to_csv = ''):
        """
        clf pic in dir knowing it's true class
        :param file_dir_path: this file only contain 'true_class' pic
        :param true_class: picture class
        :param verbose: print each pic name and it's prob
        :param to_csv: result name to csv
        :return:
        """
        assert true_class in self.pic_classes_str2num.keys()
        img = self._build_model_input(files_path=file_dir_path)

        _tmp_pred_class = []
        _tmp_data = img.next() #in order to get input_len
        for i in range(1, self.input_len + 1):
            pred_prob = self.model.predict(_tmp_data)[0]
            cur_pic_name = self.input_pic_name[-1]
            if i<self.input_len:  #StopInter error
                _tmp_data = img.next()
            pred_prob = np.array(pred_prob).tolist()
            pred_prob = map(lambda x: round(x, 4), pred_prob)
            pred_class = np.argmax(pred_prob)
            _tmp_pred_class.append(pred_class)

            _tmp_res = [cur_pic_name, self.pic_classes_num2str[pred_class],
                        pred_prob[0], pred_prob[1], pred_prob[2]] #name,pred-c,prob
            if verbose:
                print ('{}/{} ===> {}'.format(i,self.input_len,_tmp_res))

            if pred_class != self.pic_classes_str2num[true_class]:
                self.output_res_error.append(_tmp_res)
            else :
                self.output_res.append(_tmp_res)


        self.output_res_summary.append(Counter(_tmp_pred_class))
        _tmp_accuracy = round(float(self.output_res_summary[0][self.pic_classes_str2num[true_class]])/self.input_len,4)
        self.output_res_summary.append(_tmp_accuracy)

        if verbose:
            print('\n****ERROR****')
            print(self.output_res_error)
            print('\npictures class count:{}\naccuracy:{}'.format(self.output_res_summary[0],self.output_res_summary[1]))
        if to_csv:
            pd_res = pd.DataFrame(data=self.output_res + self.output_res_error,
                                  columns=['pic_name','predict_class','lower_body_prob','upper_body_prob','whole_body_prob'])
            _out_name = to_csv +'_Tcls_' + true_class + '.csv'
            pd_res.to_csv(os.path.join(self.output_path,_out_name),header=True)



if __name__=='__main__':
    model_archi = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/0305_1try_clothes_uplow_bnft_fine_tune_model_art.json'
    model_weigh = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/0309_2try_clothes_uplow_bnft_fine_tune_model.h5'
    _tr_te_vali = ['test','validation','train']
    _my_classes = ['lower_body','upper_body','whole_body']
    output_path = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu'


    # input_pic = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/clothes/' + _tr_te_vali + '/' + _my_classes
    # to_csv = '20170316_' + _tr_te_vali + '_' + _my_classes
    # print ('cate: {} \n class: {}'.format(_tr_te_vali,_my_classes))
    # test_model = ClfModel(model_archi,model_weigh,input_pic,output_path)
    # test_model._model_clf_valid(input_pic,true_class=_my_classes,verbose=True,to_csv=to_csv)



    ##for run all files
    error_pic = []
    for _cate in _tr_te_vali:
        if _cate =='validation' or _cate =='train':
            _my_classes = ['lower_body', 'upper_body', 'whole_body']
        else:
            _my_classes = ['whole_body_2','lower_body','upper_body' ,'whole_body_1']
        for _cls in _my_classes:
            input_pic = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/clothes/' + _cate + '/' + _cls
            to_csv = '20170316_Pcls_'  + _cls + '_' +  _cate
            print ('cate: {} \nclass: {}'.format(_cate, _cls))
            test_model = ClfModel(model_archi, model_weigh, input_pic, output_path)
            true_class = _cls
            if _cls in ['whole_body_2','whole_body_1']:
                true_class = 'whole_body'
            test_model._model_clf_valid(input_pic, true_class=true_class, verbose=True, to_csv=to_csv)
            _err = [_cate,_cls,test_model.output_res_error]
            print('****{}'.format(_err))
            error_pic.append(_err)

    print ('\n\n\n**************\n\n')
    print (error_pic)
    error_pkl_file = open(output_path + '/' + 'error_pic.pkl','wb')
    pickle.dump(error_pic,error_pkl_file)
    error_pkl_file.close()


    #test:
    #valid:
    #train: