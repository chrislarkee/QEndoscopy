#!/usr/bin/env python
import wx
from sys import argv

#frames
from modules.main import main

#does this reduce flickering?
#USE_BUFFERED_DC = True

if __name__ == '__main__':
    #print("Loading the app:")
    app = wx.App(False)
    frame = main(None)
    frame.Show()
    
    #if a video file is passed as an argument, load it automatically
    if len(argv) > 1:
        frame.openVideo(event=wx.IdleEvent(), autoload=argv[1])
       
    app.MainLoop()
