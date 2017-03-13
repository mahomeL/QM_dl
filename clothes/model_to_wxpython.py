#coding=utf-8

'''Reconstruct model input-output to fit wxpython'''
import os
from keras.models import model_from_json
import cv2
import numpy as np
from keras import backend as K
from collections import Counter
K.set_image_dim_ordering('th')

class ClfModel():
    def __init__(self,model_archi,model_weigh,input_pic,output_path):
        '''load model ; set input pic and output result'''
        assert isinstance(model_archi,str) and isinstance(model_weigh,str)
        assert model_archi.split('.')[-1]=='json','model architecture must be *.json'
        assert model_weigh.split('.')[-1]=='h5','model weights must be *.h5'

        self.model_archi = model_archi
        self.model_weigh = model_weigh
        self.input_files = input_pic
        self.output_path = output_path
        self.pic_width = 350
        self.pic_height = 350

        self.input_count = 0
        self.input_len = 0
        self.input_pic_name = []

        self.output_res = []
        self.output_res_summary = []
        self.output_res_error = []

        self.pic_classes_str2num= dict([('lower_body', 0), ('upper_body', 1), ('whole_body', 2)])
        self.pic_classes_num2str= dict([(v,k) for k,v in self.pic_classes_str2num.items() ])

        self.model = model_from_json(open(self.model_archi).read())
        self.model.load_weights(self.model_weigh)

    def _output_reset(self):
        self.input_count = 0
        self.input_len = 0
        self.input_pic_name = []
        self.output_res = []
        self.output_res_summary = []
        self.output_res_error = []

    def _build_model_input(self,files_path):
        '''read pictures and resize it '''
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
        '''predict pic by model'''
        img = self._build_model_input(self.input_files)

        _tmp_pred_class = []
        for i in range(1,self.input_len+1):
            pred_prob = self.model.predict(img.next())[0]
            pred_prob = np.array(pred_prob).tolist()
            pred_prob = map(lambda x:round(x,4),pred_prob)
            pred_class = np.argmax(pred_prob)
            _tmp_pred_class.append(pred_class)

            self.output_res.append([self.input_pic_name[-1],pred_prob,pred_class])  #picname,prob,class
        self.output_res_summary.append(Counter(_tmp_pred_class))

    def _model_clf_valid(self,file_dir_path,true_class,verbose = True):
        """ clf pic in dir knowing it's true class
        @ file_dir_path: this file only contain 'true_class' pic """
        assert true_class in self.pic_classes_str2num.keys()
        img = self._build_model_input(files_path=file_dir_path)

        _tmp_pred_class = []
        _tmp_data = img.next() #in order to get input_len
        for i in range(1, self.input_len + 1):
            pred_prob = self.model.predict(_tmp_data)[0]
            if i<self.input_len:
                _tmp_data = img.next()

            pred_prob = np.array(pred_prob).tolist()
            pred_prob = map(lambda x: round(x, 4), pred_prob)
            pred_class = np.argmax(pred_prob)
            _tmp_pred_class.append(pred_class)

            if pred_class != self.pic_classes_str2num[true_class]:
                self.output_res_error.append([self.input_pic_name[-1],
                                              self.pic_classes_num2str[pred_class],
                                              true_class,pred_prob ]) #name,pred-c,true-c,prob

            _tmp_res = [self.input_pic_name[-1],self.pic_classes_num2str[pred_class],pred_prob]  # picname,pred-c,prob
            self.output_res.append(_tmp_res)
            if verbose:
                print (_tmp_res)

        self.output_res_summary.append(Counter(_tmp_pred_class))
        _tmp_accuracy = round(float(self.output_res_summary[0][self.pic_classes_str2num[true_class]])/self.input_len,4)
        self.output_res_summary.append(_tmp_accuracy)
        if verbose:
            print('pictures class count:{}\n accuracy:{}'.format(self.output_res_summary[0],self.output_res_summary[1]))



if __name__=='__main__':
    model_archi = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/0305_1try_clothes_uplow_bnft_fine_tune_model_art.json'
    model_weigh = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/0309_2try_clothes_uplow_bnft_fine_tune_model.h5'
    input_pic = '/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/clothes/validation/lower_body'
    output_path =''
    test_model = ClfModel(model_archi,model_weigh,input_pic,output_path)
    test_model._model_clf_valid(input_pic,true_class='lower_body')
