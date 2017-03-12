#coding=utf-8

"""test wxpython"""

import wx
import wx.lib.buttons as buttons
import numpy as np

class MyFrame(wx.Frame):
    def __init__(self,parent=None,title ='my title',size=(600,400)):
        wx.Frame.__init__(self,parent=parent,
                          title=title,size=size)
        self.Centre()
        self.panel = wx.Panel(self,-1)
        self.gauge_count = 0
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
        self.load_pic = wx.DirPickerCtrl(self.panel,4,path='path_to_pic',size=(400,-1),message='abc')
        self.output_res = wx.DirPickerCtrl(self.panel,4, path='output_result', size=(400, -1), message='abc111')

        self.gauge_training = wx.Gauge(self.panel,-1)
        # self.gauge_training.SetBezelFace(3)

        self.text_logging = wx.StaticText(self.panel,-1,'process log:')
        self.logging_output = wx.TextCtrl(self.panel,-1,style =wx.TE_MULTILINE,size=(-1,80))


        # self.button_clf = wx.Button(self.panel,-1,'Classify',size=(100,100),)
        self.button_clf = wx.ToggleButton(self.panel,-1,'Classify')
        self.button_clf.SetFont(wx.Font(20,wx.SWISS,wx.NORMAL,wx.BOLD))
        # self.button_clf.SetBackgroundColour('AQUAMARINE')

        '''event'''
        self.load_model_archi.Bind(wx.EVT_FILEPICKER_CHANGED,self.event_getpath,self.load_model_archi)

        self.load_pic.Bind(wx.EVT_DIRPICKER_CHANGED,self.event_getpath,self.load_pic)

        self.gauge_training.Bind(wx.EVT_IDLE,self.process_idle)

        self.button_clf.Bind(wx.EVT_TOGGLEBUTTON,self.begin_train,self.button_clf)

        wx.CallAfter(self.dologging)


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
                                                  item=[self.load_pic, self.output_res]),1,flag=wx.BOTTOM)



        middle_sizer = wx.BoxSizer(wx.VERTICAL)
        middle_top_sizer = wx.BoxSizer(wx.VERTICAL)
        middle_top_sizer.Add(self.button_clf,1,flag = wx.CENTER)
        middle_top_sizer.Add(self.gauge_training, 1,flag=wx.EXPAND)
        middle_sizer.Add(middle_top_sizer,1,flag=wx.EXPAND)
        middle_sizer.Add((10,10))
        middle_sizer.Add(self.sizer_model_staticbox('Result',text=[self.text_logging],item=[self.logging_output]) ,1, flag=wx.EXPAND)

        core_sizer = wx.BoxSizer(wx.HORIZONTAL)
        core_sizer.Add(left_sizer,1,flag=wx.LEFT|wx.EXPAND|wx.ALL)
        core_sizer.Add(middle_sizer,1,flag=wx.EXPAND|wx.ALL)

        main_sizer.Add(core_sizer,1,wx.ALL,5)

        self.panel.SetSizer(main_sizer)
        main_sizer.Fit(self)
        # wx.Log.SetActiveTarget(wx.LogTextCtrl(self.logging_output))


    def event_getpath(self,event):
        '''get browse file/dir path'''
        print(event.GetPath())
        event.Skip()


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



    def dologging(self):
        '''print output predict result on panel'''
        msg = np.random.randint(0,10,3)
        self.logging_output.AppendText(str(msg)+'\n')

        wx.CallLater(3000,self.dologging)

    def process_idle(self,event):
        '''process idle '''
        self.gauge_count += 1
        if self.gauge_count >= 100:
            self.gauge_count = 0
        self.gauge_training.SetValue(self.gauge_count)

    def begin_train(self,even):
        print(self.button_clf.GetValue())
        if self.button_clf.GetValue():
            self.button_clf.SetLabel('waiting...')
        else:
            self.button_clf.SetLabel('Classify')




class GetClothesApp(wx.App):
    '''build APP'''
    def OnInit(self):
        frame = MyFrame(title='QingMu clothes V1.0')
        frame.Show()
        return True





if __name__ == '__main__':
    test_app = GetClothesApp()
    test_app.MainLoop()
