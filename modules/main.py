#!/usr/bin/env python
import wx

from os import startfile, getcwd, mkdir
from os.path import isfile, isdir
import ctypes

import modules.layouts as layouts       #all gui views.
import modules.main_dialogs as dialogs
import modules.imaging as QEi       #video image processing
import modules.analysis as QEa      #point cloud and depth map processing
import modules.logging as log

pixelScale = 0.0    #screenshot hack

class main(layouts.MainInterface):  
    def __init__(self,parent):
        #evaluate pixel scaling factor (for screenshots)
        awareness = ctypes.c_int()
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        global pixelScale
        pixelScale = wx.ScreenDC().GetPPI()[0] / 96.0
        ctypes.windll.shcore.SetProcessDpiAwareness(0)

        #initialize parent class
        layouts.MainInterface.__init__(self,parent)

        #bind keyboard shortcuts
        self.Bind(wx.EVT_CHAR_HOOK, self.shortcuts)

        #set up Vispy canvases
        QEa.VispyPCD.startup(self.vispypanel)
        QEi.VispyImg.startup(self.vidPanel)

    def shortcuts(self, event):
        if QEi.Video.ready == False:
            return

        #special keys
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_F1:
            #todo: write manual PDF
            print("F1!")  
        elif keycode == wx.WXK_LEFT:
            self.framePrevious(event)   
        elif keycode == wx.WXK_RIGHT:   
            self.frameNext(event)  

        #regular keys
        keycode = chr(event.GetKeyCode())
        if keycode == 'E':
            self.openExporter(event)  
        elif keycode == 'O':
            self.openVideo(event) 
        elif keycode == 'S':
            self.showSettings(event)  
        elif keycode == 'D':
            self.showmap(event)  
        
    def openVideo(self, event, autoload=None):
        self.b_playVideo.SetValue(False)
        
        #this shows up the 2nd time the user opens a video.
        if QEi.Video.ready == True:
            dlg = wx.MessageDialog(None, "Opening a new file will reset the program.",'QE',wx.ICON_WARNING|wx.OK|wx.CANCEL)
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
        
        #use the native file picker to select a file
        if autoload == None:
            dlg = wx.FileDialog(self, message="Select a video or image file.", style=wx.FD_FILE_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_CANCEL:
                return

        progress = wx.ProgressDialog("Please Wait", "Starting Up...", parent=self)
        
        #load the video as this persistent object
        progress.Pulse(newmsg="Loading Video Data...")
        log.startup()
        if autoload == None:            
            if dlg.GetPath()[-3:] == "csv":
                progress.Destroy()
                wx.MessageBox('CSV files cannot be loaded in this way.', 'Error.', wx.OK | wx.ICON_ERROR)
                return
            else:
                QEi.Video.loadVideo(dlg.GetPath()) 
        else:
            if (isfile(autoload) == False):
                progress.Destroy()
                wx.MessageBox('The file does not exist.', 'Error.', wx.OK | wx.ICON_ERROR)
                return
            QEi.Video.loadVideo(autoload)  

        #QEImaging.VispyImg.fixSize(self.vidPanel.GetSize())
        progress.Pulse(newmsg="Starting Depth Map...")
        #check if there's a depth map available  
        vid = QEi.Video
        depthFile1 = f"{vid._path[:-12:]}depth_{vid._path[-8:-3]}exr"
        depthFile2 = f"{vid._path[:-3]}exr"
        if isfile(depthFile1):
            self.t_statusText.AppendText("\nLoading precomputed depth map.")
            QEa.Depthmap.setupTorch(exrMap=depthFile1)
        elif isfile(depthFile2):   
            self.t_statusText.AppendText("\nLoading precomputed depth map.")
            QEa.Depthmap.setupTorch(exrMap=depthFile2)
        else: 
            QEa.Depthmap.setupTorch()
        
        #apply changes from the loaded video to the gui
        progress.Pulse(newmsg="Processing metadata...")
        log.framerate = vid._rate

        #enable the locked out buttons
        self.SetTitle("Quantitative Endoscopy: " + vid.nicename) #the title of the window
        self.t_statusText.AppendText(f"\n{vid.nicename}: {vid.getLength()}")     
        self.slider_time.SetMax(vid.endFrame)       #the limit of the main slider
        self.b_playVideo.Enable(True)
        self.b_showSettings.Enable(True)
        self.b_showmap.Enable(True)
        self.b_recordMeasurement.Enable(True)
        self.b_openExporter.Enable(True)
        self.b_openViewer.Enable(True)
        self.b_jumpFrame.Enable(True)
        self.b_vischoice.Enable(True)
        vid.guiSize = self.vidPanel.GetSize().Width
                
        progress.Pulse(newmsg="Processing Frame 1...")
        self.frameNext(wx.IdleEvent())  #load the first frame
        self.slider_distance.SetValue(self.slider_distance.GetMax())
        self.slicerDone(wx.IdleEvent())
        
        progress.Destroy()
        self.Layout()        

    ###TOP BAR FUNCTIONS###
    def openViewer(self, event):
        self.b_playVideo.SetValue(False)
        #pop up the measurement viewer, as a modal window
        dlg = dialogs.viewMeasurements(self)
        result = dlg.ShowModal()

    def openExporter(self, event):
        self.b_playVideo.SetValue(False)

        #cache a screenshot before the popup, incase the user wants to save it
        rect = self.GetRect() # pixelScale
        crop = (rect.Left, rect.Top, rect.Right, rect.Bottom)     #wx and PIL use different formats.
        QEi.Video.grabScreenshot(crop)       

        #pop up the savewizard layout, as a modal window
        dlg = dialogs.saveWizard(self)
        dlg.ShowModal()
    
    def openHelp(self, event):
        #launch the PDF of the documentation using the system default PDF loader
        startfile("includes\\QEndoscopy Documentation.pdf")

    ###TRANSPORT CONTROLS###
    def playVideo( self, event ):
        if self.b_playVideo.GetValue() == False:
            return
        self.timer = wx.Timer(self, 0)
        self.Bind(wx.EVT_TIMER, self.playVideoTick)

        self.timer.Start(int(1000.0 / 30.0))
        QEa.VispyPCD.clearPoints()

    def playVideoTick(self, event):
        if self.b_playVideo.GetValue() == False:
            self.timer.Stop()
            #self.slider_distance.SetMax(int(QEAnalysis.Depthmap.minmax[1] * 1000 - 1))
            QEa.Depthmap.processDepth(QEi.Video.imageCache)
            self.slicerDone(wx.IdleEvent)
        else:
            #left frame only
            #QEMeasurement.overlayEnabled = False
            QEi.Video.nextFrame()
            self.slider_time.SetValue(int(QEi.Video.currentFrame))

    def framePrevious(self, event):
        self.b_playVideo.SetValue(False)

        if QEi.Video.ready == False:
            return
        
        #left & right frame        
        QEi.Video.prevFrame()
        QEa.Depthmap.processDepth(QEi.Video.imageCache)
        self.slider_time.SetValue(int(QEi.Video.currentFrame))

    def frameNext(self, event):  
        self.b_playVideo.SetValue(False)
        if QEi.Video.ready == False:
            return
        #left & right frames
        #QEMeasurement.overlayEnabled = False
        QEi.Video.nextFrame()
        QEa.Depthmap.processDepth(QEi.Video.imageCache)
        self.slider_time.SetValue(int(QEi.Video.currentFrame))
        self.slider_distance.SetMin(int(QEa.Depthmap.minmax[0] * 1000 + 1))
        self.slider_distance.SetMax(int(QEa.Depthmap.minmax[1] * 1000 - 1))            

    def scrub(self, event):
        self.b_playVideo.SetValue(False)
        if QEi.Video.ready == False:
            return
        desiredFrame = int(self.slider_time.GetValue())
        QEi.Video.specificFrame(desiredFrame)
        self.slider_time.SetValue(int(QEi.Video.currentFrame))

    def scrubDone(self,event):
        if QEi.Video.ready == False:
            return        
        QEa.Depthmap.processDepth(QEi.Video.imageCache)
        self.slider_distance.SetMin(int(QEa.Depthmap.minmax[0] * 1000 + 1))
        self.slider_distance.SetMax(int(QEa.Depthmap.minmax[1] * 1000 - 1))
        self.slicerDone(event)

    def pickPoint( self, event ):
        #Don't measure if the video is playing
        if self.b_playVideo.GetValue() == True:
            return
        #Don't measure if there's no video loaded
        if QEi.Video.ready == False:
            return
        
        mouseCoords = event.GetPosition()
        targetCoords = (int(mouseCoords[0] / QEi.Video.guiSize * 720), int(mouseCoords[1] / QEi.Video.guiSize * 720))
        threshold = QEa.Depthmap.depth_Cache[targetCoords[1]][targetCoords[0]]
        self.slider_distance.SetValue(int(threshold * 1000))

        #trigger the slider change event to update the plane
        self.slicerDone(event)

    ###LEFT SIDE FUNCTIONS###
    def jumpFrame( self, event ):
        dlg = wx.GetNumberFromUser(
            message="",
            prompt="Frame: ", 
            caption="Jump to Frame", 
            value=int(QEi.Video.currentFrame), 
            min=int(QEi.Video.startFrame), 
            max=int(QEi.Video.endFrame))
        
        if dlg != -1:
            self.slider_time.SetValue(dlg)
            QEi.Video.specificFrame(dlg)
            QEa.Depthmap.processDepth(QEi.Video.imageCache)
            self.slicerDone(event)

    def showSettings(self, event):
        self.b_playVideo.SetValue(False)
        #QEMeasurement.overlayEnabled = False
        #pop up the settings layout, as a modal window
        dlg = dialogs.settings(self)
        if dlg.ShowModal() == wx.ID_OK:
            #the modal has closed; apply the changed parameters to the GUI            
            self.slider_time.SetMin(QEi.Video.startFrame)
            self.slider_time.SetMax(QEi.Video.endFrame)
            QEi.Video.currentFrame = QEi.Video.startFrame            
            QEi.Video.specificFrame(QEi.Video.startFrame)
            QEa.Depthmap.processDepth(QEi.Video.imageCache)
            self.slider_time.SetValue(int(QEi.Video.startFrame))
            self.Layout()

    def copyText(self, event):
        if self.t_statusText.GetValue() == "":
            return
        
        textPayload = wx.TextDataObject(text=self.t_statusText.GetValue())
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(textPayload)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Unable to open the clipboard", "Error")


    ###RIGHT SIDE FUNCTIONS###
    def slicerChange( self, event ):
        d, x, y = self.slider_distance.GetValue(), self.slider_planeX.GetValue(), self.slider_planeY.GetValue()
        QEi.Video.overlayEnabled = False        
        QEa.VispyPCD.changeSlice(d, x, y, measure=False)

    def slicerDone( self, event ):
        d, x, y = self.slider_distance.GetValue(), self.slider_planeX.GetValue(), self.slider_planeY.GetValue()
        QEa.VispyPCD.changeSlice(d, x, y, measure=True)

        overlayUpdate, overlayPoints = QEa.VispyPCD.getIntersection()
        if overlayUpdate == True:
            QEi.Video.overlayEnabled = True
            QEi.Video.addOverlay(overlayPoints)
        if overlayUpdate == False and QEi.Video.overlayEnabled:
            QEi.Video.overlayEnabled = False
            QEi.Video.removeOverlay()

        log.currentEntry.distance = d / 1000.0
        log.currentEntry.rotation = (x, y)
        self.t_statusText.AppendText(f"\nArea: {log.currentEntry.CSarea}")

    def changeTool(self, event):
        #self.clearMeasurements(event)   
        wx.MessageBox("Not Implemented.", "Error")

    def recordMeasurement(self, event):
        #finalize data
        global pixelScale
        log.currentEntry.frame = QEi.Video.currentFrame
        #QEMeasurement.currentEntry.frame = vid.currentFrame
    
        self.t_statusText.AppendText("\nStored.")
        
        #commit the log
        log.store()

    def showmap( self, event ):
        dlg = dialogs.depthmapViewer(self)
        originalColor = QEa.Depthmap.colormap
        dlg.ShowModal()
        self.b_colormap.SetSelection(QEa.Depthmap.colormap)
        if QEa.Depthmap.colormap != originalColor:
            QEa.Depthmap.processDepth(QEi.Video.imageCache)

    def resetCamera( self, event ):
        QEa.VispyPCD.resetCam()

    def resetPlane( self, event ):
        self.slider_planeX.SetValue(0)
        self.slider_planeY.SetValue(0)        
        QEa.VispyPCD.changeSlice(self.slider_distance.GetValue(), measure=True)
        self.t_statusText.SetLabelText("")

    def changeVisibility( self, event ):     
        QEa.VispyPCD.updateVisibility(self.b_vischoice.GetSelection())

    def setColors( self, event ):
        QEa.Depthmap.colormap = self.b_colormap.GetSelection()
        QEa.Depthmap.processDepth(QEi.Video.imageCache)

    ###WINDOW MANAGEMENT###
    #if the window has been resized, scale the images to fit the new shape.
    def setSize(self, event):        
        QEa.VispyPCD.fixSize(self.vispypanel.GetSize())
        QEi.VispyImg.fixSize(self.vidPanel.GetSize())
        
        self.Layout()
        if QEi.Video.ready == False:
            return
        self.slider_time.SetValue(int(QEi.Video.currentFrame))        

    def saveAndQuit( self, event ):
        dlg = wx.MessageDialog(None, "Are you sure you want to quit?",'QE',wx.YES_NO)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            wx.Exit()

