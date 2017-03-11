#coding=utf-8

"""test wxpython"""

import wx


class MyFrame(wx.Frame):
    def __init__(self,parent=None,title ='my title',size=(600,400)):
        wx.Frame.__init__(self,parent=parent,
                          title=title,size=size)
        self.Centre()
        panel = wx.Panel(self,-1)

        ##text,button
        self.text_intro = wx.StaticText(panel,-1,'Clothes-classfier by deep learning',pos=(180,0))
        # self.text_intro.SetBackgroundColour('green')
        self.text_intro.SetFont(wx.Font(26,wx.SWISS,wx.NORMAL,wx.BOLD))
        self.text_line = wx.StaticLine(panel)

        self.text_archi = wx.StaticText(panel,-1,'load model architecture:')
        self.text_weigh = wx.StaticText(panel,-1,'load model weights:')
        self.load_model_archi = wx.FilePickerCtrl(panel,2,wildcard = '*.json',path='path_to_json',size=(300,-1))
        self.load_model_weigh = wx.FilePickerCtrl(panel, 3,wildcard = '*.h5',path='path_to_h5',size=(300,-1))

        # self.load_model_button =

        ##layout
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.text_intro,0,flag=wx.CENTER)
        main_sizer.Add(self.text_line,0,flag=wx.EXPAND)

        # model_sizer = wx.StaticBox()
        main_sizer.Add(self.sizer_load_model(self.text_archi,self.load_model_archi),0,flag =wx.CENTER|wx.EXPAND)
        main_sizer.Add(self.sizer_load_model(self.text_weigh,self.load_model_weigh),0,flag=wx.CENTER|wx.EXPAND)

        # model_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # model_sizer.Add(self.load_model_archi,0,flag=wx.LEFT|wx.EXPAND)
        # model_sizer.Add(self.load_model_weigh,0,flag=wx.RIGHT|wx.EXPAND)
        panel.SetSizer(main_sizer)
        main_sizer.Fit(self)

    

    def sizer_load_model(self,label,item,flag=wx.LEFT|wx.EXPAND):
        model_sizer = wx.BoxSizer(wx.VERTICAL)
        model_sizer.Add(label,0,flag = wx.LEFT)
        model_sizer.Add(item,0,flag = flag)
        return model_sizer







class GetClothesApp(wx.App):
    def OnInit(self):
        frame = MyFrame(title='qingmu clothes')
        frame.Show()
        return True





if __name__ == '__main__':
    test_app = GetClothesApp()
    test_app.MainLoop()