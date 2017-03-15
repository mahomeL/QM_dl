#coding=utf-8

"""test wxpython"""

import wx
import wx.lib.buttons as buttons
import numpy as np
import os
from model_to_wxpython import *
from wx.lib.pubsub import pub as Publisher
from threading import Thread

class MyFrame(wx.Frame):
    def __init__(self,parent=None,title ='my title',size=(800,600)):
        wx.Frame.__init__(self,parent=parent,
                          title=title,size=size)
        self.Centre()
        self.panel = wx.Panel(self,-1)
        self.gauge_count = 0

        self.model_archi = ''
        self.model_weigh = ''
        self.input_files_path = ''
        self.output_path = ''

        '''text,button'''
        self.text_intro = wx.StaticText(self.panel,-1,'Clothes-classfier by deep learning',pos=(180,0))
        # self.text_intro.SetBackgroundColour('green')
        self.text_intro.SetFont(wx.Font(26,wx.SWISS,wx.NORMAL,wx.BOLD))
        self.text_line = wx.StaticLine(self.panel)

        self.text_archi = wx.StaticText(self.panel,-1,'load model architecture:')
        self.text_archi.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.text_weigh = wx.StaticText(self.panel,-1,'load model weights:')
        self.load_model_archi = wx.FilePickerCtrl(self.panel,2,wildcard = '*.json',path='path_to_model_architecture(*.json)',size=(400,-1))
        self.load_model_weigh = wx.FilePickerCtrl(self.panel,3,wildcard = '*.h5',path='path_to_model_weights(*.h5)',size=(400,-1))

        self.text_load_pic = wx.StaticText(self.panel,-1,'pictures folder(must only contain pictures):')
        self.text_res_output = wx.StaticText(self.panel, -1, 'result output folder:')
        self.load_pic = wx.DirPickerCtrl(self.panel,4,path='path_to_pic',size=(400,-1),message='abc') #message is tips when browse path
        self.output_res = wx.DirPickerCtrl(self.panel,4, path='output_result', size=(400, -1), message='abc111')

        self.gauge_training = wx.Gauge(self.panel,-1)
        # self.gauge_training.SetBezelFace(3)

        self.text_logging = wx.StaticText(self.panel,-1,'process log:')
        self.logging_output = wx.TextCtrl(self.panel,-1,style =wx.TE_MULTILINE|wx.TE_RICH,size=(400,80))


        # self.button_clf = wx.Button(self.panel,-1,'Classify',size=(100,100),)
        self.button_clf = wx.ToggleButton(self.panel,-1,'Classify')
        self.button_clf.SetFont(wx.Font(20,wx.SWISS,wx.NORMAL,wx.BOLD))
        # self.button_clf.SetBackgroundColour('AQUAMARINE')

        '''event'''
        self.load_model_archi.Bind(wx.EVT_FILEPICKER_CHANGED,self.event_getpath,self.load_model_archi)
        self.load_model_weigh.Bind(wx.EVT_FILEPICKER_CHANGED, self.event_getpath, self.load_model_weigh)

        self.load_pic.Bind(wx.EVT_DIRPICKER_CHANGED,self.event_getpath,self.load_pic)
        self.output_res.Bind(wx.EVT_DIRPICKER_CHANGED, self.event_getpath, self.output_res)

        self.gauge_training.Bind(wx.EVT_IDLE,self.do_gauge)

        self.button_clf.Bind(wx.EVT_TOGGLEBUTTON,self.begin_classify,self.button_clf)

        # wx.CallAfter(self.dologging)


        '''get info for model'''
        self.model_archi =self.load_model_archi.GetPath()
        self.model_weigh = self.load_model_weigh.GetPath()
        self.input_files_path = self.load_pic.GetPath()
        self.output_path = self.output_res.GetPath()

        """layout"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.text_intro,0,flag=wx.CENTER)
        main_sizer.Add(self.text_line,0,flag=wx.EXPAND)
        main_sizer.Add((20,20))



        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(self.sizer_model_staticbox('Load Model',text=[self.text_archi,self.text_weigh],
                                                  item=[self.load_model_archi,self.load_model_weigh]),1,flag=wx.EXPAND)
        left_sizer.Add((10,10))
        left_sizer.Add(self.sizer_model_staticbox('Input-file Output-file', text=[self.text_load_pic, self.text_res_output],
                                                  item=[self.load_pic, self.output_res]),1,flag=wx.BOTTOM|wx.EXPAND|wx.ALL)



        middle_sizer = wx.BoxSizer(wx.VERTICAL)
        middle_top_sizer = wx.BoxSizer(wx.VERTICAL)
        middle_top_sizer.Add(self.button_clf,1,flag = wx.CENTER)
        middle_top_sizer.Add(self.gauge_training, 1,flag=wx.EXPAND)
        middle_sizer.Add(middle_top_sizer,1,flag=wx.EXPAND)
        middle_sizer.Add((10,10))
        # middle_sizer.Add(self.sizer_model_staticbox('Result',text=[self.text_logging],item=[self.logging_output]) ,1,wx.EXPAND|wx.ALL)
        log_box = wx.StaticBox(self.panel, -1, 'Result')
        logsizer = wx.StaticBoxSizer(log_box, wx.VERTICAL)
        logsizer.Add(self.sizer_load_model(self.text_logging,self.logging_output,flag=wx.EXPAND|wx.ALL), 1,flag=wx.BOTTOM|wx.EXPAND|wx.ALL)
        middle_sizer.Add(logsizer,1,flag=wx.BOTTOM|wx.EXPAND|wx.ALL)

        core_sizer = wx.BoxSizer(wx.HORIZONTAL)
        core_sizer.Add(left_sizer,1,flag=wx.LEFT|wx.EXPAND|wx.ALL)
        core_sizer.Add(middle_sizer,1,flag=wx.EXPAND|wx.ALL)

        main_sizer.Add(core_sizer,1,wx.ALL,5)

        self.panel.SetSizer(main_sizer)
        main_sizer.Fit(self)

        Publisher.subscribe(self.dologging,'update')
        Publisher.subscribe(self.do_gauge, 'update_gauge')


    def event_getpath(self,event):
        '''get browse file/dir path'''
        print(event.GetPath())



    def sizer_model_staticbox(self,boxlabel,text,item):
        '''build staticbox for model and pic'''
        box = wx.StaticBox(self.panel,-1,boxlabel)
        sizer = wx.StaticBoxSizer(box,wx.VERTICAL)

        assert len(text)==len(item)
        for i in range(len(text)):
            sizer.Add(self.sizer_load_model(text[i],item[i]),0,wx.EXPAND|wx.BOTTOM)

        return sizer



    def sizer_load_model(self,label,item,flag=wx.LEFT|wx.EXPAND):
        '''build text-pickle sizer'''
        model_sizer = wx.BoxSizer(wx.VERTICAL)
        model_sizer.Add(label,0,flag = wx.LEFT)
        model_sizer.Add(item,0,flag = flag)
        return model_sizer


    def do_gauge(self,gauge_msg):
        '''process gauge '''
        if isinstance(gauge_msg,int):
            self.gauge_count = gauge_msg
            if self.gauge_count >= 100:
                self.gauge_count = 0
            self.gauge_training.SetValue(self.gauge_count)

    def begin_classify(self,event):
        # print(self.button_clf.GetValue())
        if self.button_clf.GetValue():
            self.button_clf.SetLabel('waiting...')
            # self.button_clf.Disable()
            event.GetEventObject().Disable()
            if self.valid_path():

                MyThread(self.model_archi.encode('utf-8'),self.model_weigh.encode('utf-8'),
                         self.input_files_path.encode('utf-8'),self.output_path.encode('utf-8') )
            else:
                self.button_clf.SetLabel('Classify')
                self.button_clf.Enable()

    def valid_path(self):
        _path1 = self.load_model_archi.GetPath()
        _valid_path1 = os.path.isfile(_path1) and _path1[-5:]=='.json'
        _path2 = self.load_model_weigh.GetPath()
        _valid_path2 = os.path.isfile(_path2) and _path2[-3:] == '.h5'
        _path3 = self.load_pic.GetPath()
        _valid_path3 = os.path.isdir(_path3)
        _path4 = self.output_res.GetPath()
        _valid_path4 = os.path.isdir(_path4)
        if _valid_path1 and _valid_path2 and _valid_path3 and _valid_path4:
            self.model_archi = _path1
            self.model_weigh = _path2
            self.input_files_path = _path3
            self.output_path = _path4
            return True
        else:return False

    def dologging(self,msg):
        '''print output predict result on panel'''
        if isinstance(msg,list):
            self.logging_output.AppendText(str(msg)+'\n\n') #str for print
        elif isinstance(msg,str):
            self.logging_output.AppendText('\n*********'+msg+ '\n\n')
            self.logging_output.AppendText('Predict result has been saved in \n{}'.format(self.output_path))
            self.button_clf.SetLabel('Classify')
            self.button_clf.Enable()
        else:
            self.logging_output.AppendText('Something Error!!!' + '\n\n')


class MyThread(Thread):
    '''multi thread and update model output'''
    def __init__(self,model_archi,model_weigh,input_files_path,output_path):
        Thread.__init__(self)
        self.model = ClfModel(model_archi, model_weigh, input_files_path, output_path)
        self.pic_len = 0
        self.pic_count = 0
        self.start()

    def run(self):
        output_msg = self.model._model_predict()
        pic_msg = output_msg.next()
        self.pic_len = self.model.input_len
        self.pic_count =0
        while self.pic_count<self.pic_len:
            wx.CallAfter(self.sending_msg,pic_msg)

            self.pic_count += 1
            _gauge_percent = int(float(self.pic_count) / self.pic_len * 100)
            wx.CallAfter(self.sending_gauge_msg,_gauge_percent)

            try:
                pic_msg = output_msg.next()
            except StopIteration:
                break

        wx.CallAfter(Publisher.sendMessage,'update',msg='finished!')


    def sending_msg(self,output_msg):
        Publisher.sendMessage('update',msg = output_msg)

    def sending_gauge_msg(self,g_msg):
        Publisher.sendMessage('update_gauge', gauge_msg=g_msg)


class GetClothesApp(wx.App):
    '''build APP'''
    def OnInit(self):
        frame = MyFrame(title='QingMu clothes V1.1')
        frame.Show()
        return True


if __name__ == '__main__':
    test_app = GetClothesApp()
    test_app.MainLoop()
