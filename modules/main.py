#!/usr/bin/env python
#from typing_extensions import Self
import wx
import wx.grid
from os import startfile, getcwd, mkdir
from os.path import isfile, isdir
import ctypes

#sys.path.append("..")
import modules.layouts as layouts  #all gui views.
import modules.imaging as i        #openCV image processing
import modules.analysis as mi      #Midas depth analysis
import modules.measurement as me   #image annotation

vid = None      #cache the imaging settings & buffers
midas = None    #the AI instance
pixelScale = 0.0

class main(layouts.MainInterface):  
    def __init__(self,parent):
        #evaluate pixel scaling factor (for screenshots)
        awareness = ctypes.c_int()
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        global pixelScale
        pixelScale = wx.ScreenDC().GetPPI()[0] / 96.0
        ctypes.windll.shcore.SetProcessDpiAwareness(0)

        self.vid = None
        #initialize parent class
        layouts.MainInterface.__init__(self,parent)

        #bind keyboard shortcuts
        self.Bind(wx.EVT_CHAR_HOOK, self.shortcuts)

    def shortcuts(self, event):
        if vid == None:
            return

        keycode = event.GetKeyCode()
        #print(keycode)
        if keycode == wx.WXK_F1:
            print("F1!")                #todo: write manual PDF
        elif keycode == wx.WXK_LEFT:
            self.framePrevious(event)   
        elif keycode == wx.WXK_RIGHT:   
            self.frameNext(event)  
        elif keycode == 69:             #E
            self.exportImage(event)  
        elif keycode == 79:             #O
            self.openVideo(event) 
        elif keycode == 83:             #S
            self.showSettings(event)  

        
    def openVideo(self, event):
        self.b_playVideo.SetValue(False)
        global vid, midas
        #this shows up the 2nd time the user opens a video.
        if vid != None:
            dlg = wx.MessageDialog(None, "Opening a new file will reset the program.",'QE',wx.ICON_WARNING|wx.OK|wx.CANCEL)
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
        
        #use the native file picker to select a file
        dlg = wx.FileDialog(self, message="Select a video or image file.", style=wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return

        progress = wx.ProgressDialog("Please Wait", "Starting Up...", parent=self)
        
        #load the video as this persistent object
        progress.Pulse(newmsg="Loading Video Data...")
        me.startup(dlg.GetPath())
        if dlg.GetPath()[-3:] == "csv":
            progress.Destroy()
            wx.MessageBox('CSV files cannot be loaded in this way.', 'Error.', wx.OK | wx.ICON_ERROR)
            return
        else:
            vid = i.EndoVideo(dlg.GetPath())

        progress.Pulse(newmsg="Loading Depth Map...")
        #check if there's a depth map available
        depthFile = vid._path[0:-4] + "_depth.mp4"
        if isfile(depthFile):
            midas = mi.MidasSetup(depthFile, vid)
        else: 
            progress.Destroy()
            wx.MessageBox('A depth file could not be found. Select a different file.', 'Error.', wx.OK | wx.ICON_ERROR)            
            vid = None
            return
            #midas = mi.MidasSetup(None, None)
        
        #apply changes from the loaded video to the gui
        progress.Pulse(newmsg="Processing metadata...")
        #enable the locked out buttons
        self.SetTitle("Quantitative Endoscopy: " + vid.nicename)
        self.t_status.SetLabelText(vid.nicename + vid.getLength())     #the title of the window
        self.m_Time.SetMax(vid.endFrame)                               #the limit of the main slider
        self.b_playVideo.Enable(True)
        self.b_showSettings.Enable(True)
        self.b_recordMeasurement.Enable(True)
        self.b_clearMeasurements.Enable(True)
        self.b_showExporter.Enable(True)
        self.b_exportImage.Enable(True)        
        vid.guiSize = self.i_Image.GetSize().Width
                
        progress.Pulse(newmsg="Processing Frame 1...")        
        self.frameNext(event)    
        
        progress.Destroy()
        self.Layout()        

        #immediately launch the settings panel
        #self.showSettings(event)        

    ###TRANSPORT CONTROLS###
    def playVideo( self, event ):
        if self.b_playVideo.GetValue() == False:
            return
        self.timer = wx.Timer(self, 0)
        self.Bind(wx.EVT_TIMER, self.playVideoTick)

        global vid
        self.timer.Start(int(1000.0 / vid._rate / vid.speed))
        self.i_Depth.SetBitmap(vid.blankFrame())

    def playVideoTick(self, event):
        global vid, midas
        if self.b_playVideo.GetValue() == False:
            self.timer.Stop()
            self.i_Depth.SetBitmap(midas.processDepth(vid.imageCache, vid.guiSize))
            self.m_thresholdSlider.SetMax(int(midas.minmax[1] * 100 * .95))
        else:
            #left frame only
            me.overlayEnabled = False
            self.i_Image.SetBitmap(vid.nextFrame())
            self.m_Time.SetValue(int(vid.currentFrame))

    def framePrevious(self, event):
        self.b_playVideo.SetValue(False)

        global vid, midas
        if (vid == None):
            return
        #left & right frame
        me.overlayEnabled = False
        self.i_Image.SetBitmap(vid.prevFrame())
        self.i_Depth.SetBitmap(midas.processDepth(vid.imageCache, vid.guiSize))
        self.m_Time.SetValue(int(vid.currentFrame))

    def frameNext(self, event):  
        self.b_playVideo.SetValue(False)
        global vid, midas
        if (vid == None):
            return
        #left & right frames
        me.overlayEnabled = False
        self.i_Image.SetBitmap(vid.nextFrame())
        self.i_Depth.SetBitmap(midas.processDepth(vid.imageCache, vid.guiSize))
        self.m_Time.SetValue(int(vid.currentFrame))
        self.m_thresholdSlider.SetMin(int(midas.minmax[0] * 100 + 1))
        if midas.minmax[1] >= 4000:
            self.m_thresholdSlider.SetMax(4000)
        else:
            self.m_thresholdSlider.SetMax(int(midas.minmax[1] * 100 * .95))

    def scrub(self, event):
        self.b_playVideo.SetValue(False)
        global vid
        if (vid == None):
            return
        desiredFrame = int(self.m_Time.GetValue())
        #left frame only
        self.i_Image.SetBitmap(vid.specificFrame(desiredFrame))
        self.i_Depth.SetBitmap(vid.blankFrame())
        self.m_Time.SetValue(int(vid.currentFrame))

    def scrubDone(self,event):
        global vid, midas
        if vid == None:
            return
        #right frame only
        midas.clearAccumulator()
        me.overlayEnabled = False
        self.i_Depth.SetBitmap(midas.processDepth(vid.imageCache, vid.guiSize))
        self.m_thresholdSlider.SetMin(int(midas.minmax[0] * 100 + 1))
        if midas.minmax[1] >= 4000:
            self.m_thresholdSlider.SetMax(4000)
        else:
            self.m_thresholdSlider.SetMax(int(midas.minmax[1] * 100 * .95))

    def threshChange(self,event): 
        global vid, midas
        if vid == None:
            return
        me.overlayEnabled = True
        me.updateOverlay(midas.dataCache, float(self.m_thresholdSlider.GetValue()) / 100.0)
        #REFRESH
        self.i_Depth.SetBitmap(midas.postProcess(vid.guiSize))
        self.i_Image.SetBitmap(vid.refreshAnnoation())
        status = "Threshold: {}\nDistance: {}\n\nArea (px): {}\nArea (mm^2): {}".format(me.currentEntry.threshold, me.currentEntry.distance, me.currentEntry.areaPX, me.currentEntry.areaMM)
        self.t_statusText.SetValue(status)


    ###BUTTON FUNCTIONS###
    def showSettings(self, event):
        self.b_playVideo.SetValue(False)
        me.overlayEnabled = False
        #pop up the settings layout, as a modal window
        dlg = settings(self)
        if dlg.ShowModal() == wx.ID_OK:
            #the modal has closed; apply the changed parameters to the GUI
            global vid, midas
            self.m_Time.SetMin(vid.startFrame)
            self.m_Time.SetMax(vid.endFrame)
            vid.currentFrame = vid.startFrame            
            self.i_Image.SetBitmap(vid.specificFrame(vid.startFrame))
            midas.clearAccumulator()
            self.i_Depth.SetBitmap(midas.processDepth(vid.imageCache, vid.guiSize))
            self.m_Time.SetValue(int(vid.startFrame))
            self.Layout()

    def clearMeasurements(self, event):
        me.overlayEnabled = False
        self.i_Depth.SetBitmap(midas.postProcess(vid.guiSize))
        self.i_Image.SetBitmap(vid.refreshAnnoation())
        self.t_statusText.SetValue("Annotation Cleared.")

    def recordMeasurement(self, event):
        #finalize data
        global vid, midas, pixelScale
        me.currentEntry.frame = vid.currentFrame       
        status = self.t_statusText.GetValue() + "\nStored."
        self.t_statusText.SetValue(status)
        
        #prepare screenshot metadata
        screenshotPath = getcwd() + "\\logs\\" + vid.nicename
        if isdir(screenshotPath) == False:
            mkdir(screenshotPath)
        shot_filename = screenshotPath + "\\" + vid.nicename + "_" + str(me.counter()).zfill(2) + ".png"
        rect = self.GetRect() # pixelScale
        crop = (rect.Left, rect.Top, rect.Right, rect.Bottom)     #wx and PIL use different formats.

        #take screenshot
        i.takeScreenshot(crop,shot_filename)       

        #commit the log
        me.store()

    def showExporter(self, event):
        self.b_playVideo.SetValue(False)
        #pop up the exporter layout, as a modal window
        dlg = exporter(self)
        dlg.ShowModal()

    def exportImage(self, event):
        self.b_playVideo.SetValue(False)
        global vid, midas
        defaultName = vid.nicename + "_f" + str(int(vid.currentFrame)) + ".jpg"
        dlg = wx.TextEntryDialog(self, message="Enter a filename for the image (.jpg or .png)", caption="Save Image", value=defaultName)
        if dlg.ShowModal() == wx.ID_OK:
            vid.exportImage(dlg.GetValue())
        dlg.Destroy()

        defaultName = vid.nicename + "_f" + str(int(vid.currentFrame)) + "_depth.jpg"
        dlg = wx.TextEntryDialog(self, message="Enter a filename for the depth analysis. (.jpg, .png, or .npy)", caption="Save Depth", value=defaultName)
        if dlg.ShowModal() == wx.ID_OK:
            midas.exportImage(dlg.GetValue())
        dlg.Destroy()

    def openHelp(self, event):
        #launch the PDF of the documentation using the system default PDF loader
        startfile("QEndoscopy Documentation.pdf")
        

    ###IMAGE INTERACTION###
    def pickPoint( self, event):     
        if self.b_playVideo.GetValue() == True:
            return
        global vid, midas
        if (vid == None):
            return
        #print(event.GetId())       #does it matter which one we clicked?

        mouseCoords = event.GetPosition()
        targetCoords = (int(mouseCoords[0] / vid.guiSize * vid.fullSize), int(mouseCoords[1] / vid.guiSize * vid.fullSize))
        measurement = midas.dataCache[targetCoords[0]][targetCoords[1]]
        #print(measurement)
        
        me.updateOverlay(midas.dataCache, measurement)
        self.m_thresholdSlider.SetValue(int(measurement * 100))
        #REFRESH
        me.overlayEnabled = True
        self.i_Depth.SetBitmap(midas.postProcess(vid.guiSize))
        self.i_Image.SetBitmap(vid.refreshAnnoation())
        
        #self.t_export.SetValue(measurement.report())


    ###WINDOW MANAGEMENT###
    #if the window has been resized, scale the images to fit the new shape.
    def setSize(self, event):        
        global vid, midas
        if vid == None:
            return

        if self.i_Image.GetSize().Width != vid.guiSize:
            vid.guiSize = self.i_Image.GetSize().Width
            self.i_Image.SetBitmap(vid.nextFrame())
            self.i_Depth.SetBitmap(midas.processDepth(vid.imageCache, vid.guiSize))
            self.m_Time.SetValue(int(vid.currentFrame))
        self.Layout()

    def saveAndQuit( self, event ):
        dlg = wx.MessageDialog(None, "Are you sure you want to quit?",'QE',wx.YES_NO)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            wx.Exit()


class settings(layouts.VideoSettings):  
    def __init__(self, parent):
        #initialize parent class
        layouts.VideoSettings.__init__(self,parent)
        global vid, midas

        #sync up initial values
        self.b_TrimStart.SetMax(vid._maxFrame)
        self.b_TrimStart.SetValue(vid.startFrame)
        self.b_TrimEnd.SetMin(self.b_TrimStart.GetValue() + 1)
        self.b_TrimEnd.SetMax(vid._maxFrame)
        self.b_TrimEnd.SetValue(vid.endFrame)
        self.b_speed.SetValue(vid.speed)
        self.b_cropOffset.SetValue(vid.offset)
        self.b_zoom.SetValue(vid.zoom)
        self.b_red.SetValue(midas.redMatte)


        #tweak the interaction behavior
        self.b_zoom.SetIncrement(5)
        self.b_cropOffset.SetIncrement(2)

        #load initial images
        self.i_TrimStart.SetBitmap(vid.trimmerFrame(self.b_TrimStart.GetValue()))
        self.i_TrimEnd.SetBitmap(vid.trimmerFrame(self.b_TrimEnd.GetValue()))
        
        self.Layout()
        
    def updateTrim( self, event ):
        global vid
        #refresh the images, if the need to be updated
        if self.b_TrimStart.GetValue() != vid.startFrame:
            self.i_TrimStart.SetBitmap(vid.trimmerFrame(self.b_TrimStart.GetValue()))
            vid.startFrame = self.b_TrimStart.GetValue()
            self.b_TrimEnd.SetMin(self.b_TrimStart.GetValue() + 1)
        if self.b_TrimEnd.GetValue() != vid.endFrame:
            self.i_TrimEnd.SetBitmap(vid.trimmerFrame(self.b_TrimEnd.GetValue()))
            vid.endFrame = self.b_TrimEnd.GetValue()

        #evaluate the selection's length, in seconds.
        duration = round((vid.endFrame - vid.startFrame) / vid._rate * vid.speed, 2)
        self.t_duration.SetLabelText("Clip Duration: " + str(duration) + "s.")

    def updateCrop( self, event ):
        global vid
        vid.updateCrop(self.b_cropOffset.GetValue(), self.b_zoom.GetValue())

        #refresh images
        self.i_TrimStart.SetBitmap(vid.trimmerFrame(self.b_TrimStart.GetValue()))
        self.i_TrimEnd.SetBitmap(vid.trimmerFrame(self.b_TrimEnd.GetValue()))

    def doneTrimming(self, event):
        if self.b_TrimStart.GetValue() >= self.b_TrimEnd.GetValue():
            dlg = wx.MessageDialog(None, "Clip start cannot be less than clip end.",'Error',wx.ICON_WARNING|wx.OK)
            dlg.ShowModal()
            return

        #save the values into the vid class
        global vid, midas
        vid.speed = self.b_speed.GetValue()
        vid.startFrame = self.b_TrimStart.GetValue()
        vid.endFrame = self.b_TrimEnd.GetValue()
        midas.redMatte = self.b_red.GetValue()

        self.EndModal(wx.ID_OK)


class exporter(layouts.Exporter):  
    def __init__(self, parent):
        #initialize parent class
        layouts.Exporter.__init__(self,parent)
       
        #l.sortData()       
        data = me.generatePreview()
        if (len(data) < 1):
            return
        
        #put data into the grid
        self.grid.AppendRows(len(data))
        for row in range(0, len(data)):        
            for col in range(0,9):
                cellContents = str(data[row][col])
                self.grid.SetCellValue(row + 1, col, cellContents)

        #auto choose a file name
        logPath = getcwd() + "\\logs\\" + vid.nicename
        if isdir(logPath) == False:
            mkdir(logPath)

        logName = "\\" + vid.nicename + "_measurements.xlsx"        
        self.b_exportFilename.SetPath(logPath + logName)
        self.b_exportSave.Enable(True)

        self.Layout()

    def readyToSave(self, event):
        if str.lower(self.b_exportFilename.GetPath()[-5:]) != ".xlsx":
            self.b_exportSave.Enable(False)
        else:
            self.b_exportSave.Enable(True)

    def exportSave(self, event):
        global vid
        if str.lower(self.b_exportFilename.GetPath()[-5:]) != ".xlsx":
            return

        if (me.writeLog(vid.nicename, self.b_exportFilename.GetPath())):
            self.EndModal(wx.ID_OK)
            wx.MessageBox('The file was successfully saved.', 'File Saved.', wx.OK | wx.ICON_INFORMATION)
        else: 
            self.EndModal(wx.ID_CANCEL)
            wx.MessageBox('The file was not saved due to an error.', 'Error.', wx.OK | wx.ICON_ERROR)
            
        
