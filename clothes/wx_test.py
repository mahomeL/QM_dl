#coding=utf-8

"""test wxpython"""

import wx

class MyFrame(wx.Frame):
    def __init__(self,parent=None,title ='my title',size=(600,400)):
        wx.Frame.__init__(self,parent=parent,
                          title=title,size=size)
        self.Centre()
        self.panel = wx.Panel(self,-1)

        '''text,button'''
        self.text_intro = wx.StaticText(self.panel,-1,'Clothes-classfier by deep learning',pos=(180,0))
        # self.text_intro.SetBackgroundColour('green')
        self.text_intro.SetFont(wx.Font(26,wx.SWISS,wx.NORMAL,wx.BOLD))
        self.text_line = wx.StaticLine(self.panel)

        self.text_archi = wx.StaticText(self.panel,-1,'load model architecture:')
        self.text_weigh = wx.StaticText(self.panel,-1,'load model weights:')
        self.load_model_archi = wx.FilePickerCtrl(self.panel,2,wildcard = '*.json',path='path_to_model_architecture(*.json)',size=(400,-1))
        self.load_model_weigh = wx.FilePickerCtrl(self.panel,3,wildcard = '*.h5',path='path_to_model_weights(*.h5)',size=(400,-1))

        self.text_load_pic = wx.StaticText(self.panel,-1,'pictures folder(must only contain pictures):')
        self.text_res_output = wx.StaticText(self.panel, -1, 'result output folder:')
        self.load_pic = wx.DirPickerCtrl(self.panel,4,path='path_to_pic',size=(400,-1),message='abc')
        self.output_res = wx.DirPickerCtrl(self.panel,4, path='output_result', size=(400, -1), message='abc111')

        self.gauge_training = wx.Gauge(self.panel,-1,size=(400,-1))

        self.logging_output = wx.TextCtrl(self.panel,-1,style =wx.TE_MULTILINE)
        wx.CallAfter(self.dologging)

        '''event'''
        self.load_model_archi.Bind(wx.EVT_FILEPICKER_CHANGED,self.event_getpath,self.load_model_archi)

        self.load_pic.Bind(wx.EVT_DIRPICKER_CHANGED,self.event_getpath,self.load_pic)





        """layout"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.text_intro,0,flag=wx.CENTER)
        main_sizer.Add(self.text_line,0,flag=wx.EXPAND)
        main_sizer.Add((20,20))

        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(self.sizer_model_staticbox('Load Model',text=[self.text_archi,self.text_weigh],
                                                  item=[self.load_model_archi,self.load_model_weigh]),1)
        left_sizer.Add((20,20))
        left_sizer.Add(self.sizer_model_staticbox('Input-pic Output-result', text=[self.text_load_pic, self.text_res_output],
                                                  item=[self.load_pic, self.output_res]),1)

        left_sizer.Add(self.gauge_training,1)
        left_sizer.Add(self.logging_output,1,flag=wx.EXPAND)
        main_sizer.Add(left_sizer,1,flag=wx.LEFT)

        self.panel.SetSizer(main_sizer)
        main_sizer.Fit(self)
        wx.Log.SetActiveTarget(wx.LogTextCtrl(self.logging_output))


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
            sizer.Add(self.sizer_load_model(text[i],item[i]),0,wx.ALL,5)

        return sizer



    def sizer_load_model(self,label,item,flag=wx.LEFT|wx.EXPAND):
        '''build text-pickle sizer'''
        model_sizer = wx.BoxSizer(wx.VERTICAL)
        model_sizer.Add(label,0,flag = wx.LEFT)
        model_sizer.Add(item,0,flag = flag)
        return model_sizer

    def dologging(self):
        '''print output predict result on panel'''
        print ('do logging....')
        wx.LogMessage('text message')
        wx.CallLater(3000,self.dologging)



class GetClothesApp(wx.App):
    def OnInit(self):
        frame = MyFrame(title='QingMu clothes V1.0')
        frame.Show()
        return True





if __name__ == '__main__':
    test_app = GetClothesApp()
    test_app.MainLoop()
