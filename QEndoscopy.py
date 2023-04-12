#!/usr/bin/env python
import wx

#frames
from modules.main import main

#does this reduce flickering?
#USE_BUFFERED_DC = True

if __name__ == '__main__':
    #print("Loading the app:")
    app = wx.App(False)
    frame = main(None)
    frame.Show()
    app.MainLoop()
