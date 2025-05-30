#!/usr/bin/env python
#from typing_extensions import Self
import wx
import wx.grid
from os import startfile, getcwd, mkdir
from os.path import isfile, isdir
import ctypes

#sys.path.append("..")
import modules.layouts as layouts  #all gui views.
import modules.imaging as QEImaging        #openCV image processing
#import modules.analysis as mi      #ML depth analysis
import modules.measurement as QEMeasurement   #image annotation

vid = None      #The QEImaging instance
depth = None    #the QEMeasurement instance
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
        global vid, depth
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
        QEMeasurement.startup(dlg.GetPath())
        if dlg.GetPath()[-3:] == "csv":
            progress.Destroy()
            wx.MessageBox('CSV files cannot be loaded in this way.', 'Error.', wx.OK | wx.ICON_ERROR)
            return
        else:
            vid = QEImaging.EndoVideo(dlg.GetPath())

        progress.Pulse(newmsg="Loading Depth Libraries...")
        import modules.analysis as QEDepth      #ML depth analysis

        progress.Pulse(newmsg="Starting Depth Map...")
        #check if there's a depth map available
        depthFile = vid._path[0:-4] + "_depth.mp4"
        if isfile(depthFile):
            depth = QEDepth.Depth(depthFile, vid)            
            wx.MessageBox('A depth file was found and will be used instead of evaluation depth.', 'Error.', wx.OK | wx.ICON_ERROR)
        else: 
            depth = QEDepth.Depth(None, vid)
        
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
        self.timer.Start(int(1000.0 / 30.0))
        self.i_Depth.SetBitmap(vid.blankFrame())

    def playVideoTick(self, event):
        global vid, depth
        if self.b_playVideo.GetValue() == False:
            self.timer.Stop()
            self.i_Depth.SetBitmap(depth.processDepth(vid.imageCache, vid.guiSize))
            self.m_thresholdSlider.SetMax(int(depth.minmax[1] * 100 * .95))
        else:
            #left frame only
            QEMeasurement.overlayEnabled = False
            self.i_Image.SetBitmap(vid.nextFrame())
            self.m_Time.SetValue(int(vid.currentFrame))

    def framePrevious(self, event):
        self.b_playVideo.SetValue(False)

        global vid, depth
        if (vid == None):
            return
        #left & right frame
        QEMeasurement.overlayEnabled = False
        self.i_Image.SetBitmap(vid.prevFrame())
        self.i_Depth.SetBitmap(depth.processDepth(vid.imageCache, vid.guiSize))
        self.m_Time.SetValue(int(vid.currentFrame))

    def frameNext(self, event):  
        self.b_playVideo.SetValue(False)
        global vid, depth
        if (vid == None):
            return
        #left & right frames
        QEMeasurement.overlayEnabled = False
        self.i_Image.SetBitmap(vid.nextFrame())
        self.i_Depth.SetBitmap(depth.processDepth(vid.imageCache, vid.guiSize))
        self.m_Time.SetValue(int(vid.currentFrame))
        self.m_thresholdSlider.SetMin(int(depth.minmax[0] * 100 + 1))
        if depth.minmax[1] >= 4000:
            self.m_thresholdSlider.SetMax(4000)
        else:
            self.m_thresholdSlider.SetMax(int(depth.minmax[1] * 100 * .95))

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
        global vid, depth
        if vid == None:
            return
        #right frame only
        QEMeasurement.overlayEnabled = False
        self.i_Depth.SetBitmap(depth.processDepth(vid.imageCache, vid.guiSize))
        self.m_thresholdSlider.SetMin(int(depth.minmax[0] * 100 + 1))
        if depth.minmax[1] >= 4000:
            self.m_thresholdSlider.SetMax(4000)
        else:
            self.m_thresholdSlider.SetMax(int(depth.minmax[1] * 100 * .95))

    def threshChange(self,event): 
        global vid, depth
        if vid == None:
            return
        QEMeasurement.overlayEnabled = True
        QEMeasurement.updateOverlayCross(depth.dataCache, float(self.m_thresholdSlider.GetValue()) / 100.0, None)
        #REFRESH
        self.i_Depth.SetBitmap(depth.postProcess(vid.guiSize))
        self.i_Image.SetBitmap(vid.refreshAnnoation())
        status = f"Distance: {QEMeasurement.currentEntry.distance}\nArea (px): {QEMeasurement.currentEntry.areaPX}\nArea (mm^2): {QEMeasurement.currentEntry.areaMM}"
        self.t_statusText.SetValue(status)


    ###BUTTON FUNCTIONS###
    def showSettings(self, event):
        self.b_playVideo.SetValue(False)
        QEMeasurement.overlayEnabled = False
        #pop up the settings layout, as a modal window
        dlg = settings(self)
        if dlg.ShowModal() == wx.ID_OK:
            #the modal has closed; apply the changed parameters to the GUI
            global vid, depth
            self.m_Time.SetMin(vid.startFrame)
            self.m_Time.SetMax(vid.endFrame)
            vid.currentFrame = vid.startFrame            
            self.i_Image.SetBitmap(vid.specificFrame(vid.startFrame))
            self.i_Depth.SetBitmap(depth.processDepth(vid.imageCache, vid.guiSize))
            self.m_Time.SetValue(int(vid.startFrame))
            self.Layout()

    def clearMeasurements(self, event):
        QEMeasurement.overlayEnabled = False
        self.i_Depth.SetBitmap(depth.postProcess(vid.guiSize))
        self.i_Image.SetBitmap(vid.refreshAnnoation())
        self.t_statusText.SetValue("Annotation Cleared.")

    def recordMeasurement(self, event):
        #finalize data
        global vid, depth, pixelScale
        QEMeasurement.currentEntry.frame = vid.currentFrame       
        status = self.t_statusText.GetValue() + "\nStored."
        self.t_statusText.SetValue(status)
        
        #prepare screenshot metadata
        screenshotPath = getcwd() + "\\logs\\" + vid.nicename
        if isdir(screenshotPath) == False:
            mkdir(screenshotPath)
        shot_filename = screenshotPath + "\\" + vid.nicename + "_" + str(QEMeasurement.counter()).zfill(2) + ".png"
        rect = self.GetRect() # pixelScale
        crop = (rect.Left, rect.Top, rect.Right, rect.Bottom)     #wx and PIL use different formats.

        #take screenshot
        QEImaging.takeScreenshot(crop,shot_filename)       

        #commit the log
        QEMeasurement.store()

    def showExporter(self, event):
        self.b_playVideo.SetValue(False)
        #pop up the exporter layout, as a modal window
        dlg = exporter(self)
        dlg.ShowModal()

    def exportImage(self, event):
        self.b_playVideo.SetValue(False)
        global vid, depth
        defaultName = vid.nicename + "_f" + str(int(vid.currentFrame)) + ".jpg"
        dlg = wx.TextEntryDialog(self, message="Enter a filename for the image (.jpg or .png)", caption="Save Image", value=defaultName)
        if dlg.ShowModal() == wx.ID_OK:
            vid.exportImage(dlg.GetValue())
        dlg.Destroy()

        defaultName = vid.nicename + "_f" + str(int(vid.currentFrame)) + "_depth.jpg"
        dlg = wx.TextEntryDialog(self, message="Enter a filename for the depth analysis. (.jpg, .png, or .npy)", caption="Save Depth", value=defaultName)
        if dlg.ShowModal() == wx.ID_OK:
            depth.exportImage(dlg.GetValue())
        dlg.Destroy()

    def openHelp(self, event):
        #launch the PDF of the documentation using the system default PDF loader
        startfile("QEndoscopy Documentation.pdf")

    def copyText(self, event):
        if self.t_statusText.GetValue() == "":
            return
        
        textPayload = wx.TextDataObject(text=self.t_statusText.GetValue())
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(textPayload)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Unable to open the clipboard", "Error")

    def changeTool(self, event):
        self.clearMeasurements(event)
        

    ###IMAGE INTERACTION###
    def pickPoint(self, event):     
        if self.b_playVideo.GetValue() == True:
            return
        global vid, depth
        if (vid == None):
            return
        #print(event.GetId())       #does it matter which one we clicked?

        QEMeasurement.overlayEnabled = True
        mouseCoords = event.GetPosition()
        targetCoords = (int(mouseCoords[0] / vid.guiSize * 720), int(mouseCoords[1] / vid.guiSize * 720))
        
        if self.b_mtool.GetCurrentSelection() == 0:
            #cross section
            threshold = depth.dataCache[targetCoords[1]][targetCoords[0]]
            self.m_thresholdSlider.SetValue(int(threshold * 100))
            QEMeasurement.updateOverlayCross(depth.dataCache, threshold, targetCoords)            
            status = f"Distance: {QEMeasurement.currentEntry.distance} cm\nArea: {QEMeasurement.currentEntry.areaPX} px\nArea: {QEMeasurement.currentEntry.areaMM} cm^2"
            self.t_statusText.SetValue(status)
        elif self.b_mtool.GetCurrentSelection() == 1: 
            #line
            QEMeasurement.updateOverlayLine(depth.dataCache, (targetCoords[0],targetCoords[1]))
            status = f"{QEMeasurement.getCoordinate()}\nDistance: {QEMeasurement.currentEntry.distance} cm"
            self.t_statusText.SetValue(status)
        elif self.b_mtool.GetCurrentSelection() == 2:
            #polygon
            QEMeasurement.overlayEnabled = False
            pass

        #REFRESH        
        self.i_Depth.SetBitmap(depth.postProcess(vid.guiSize))
        self.i_Image.SetBitmap(vid.refreshAnnoation())        
        #self.t_export.SetValue(measurement.report())


    ###WINDOW MANAGEMENT###
    #if the window has been resized, scale the images to fit the new shape.
    def setSize(self, event):        
        global vid, depth
        if vid == None:
            return

        if self.i_Image.GetSize().Width != vid.guiSize:
            vid.guiSize = self.i_Image.GetSize().Width
            self.i_Image.SetBitmap(vid.nextFrame())
            self.i_Depth.SetBitmap(depth.processDepth(vid.imageCache, vid.guiSize))
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
        global vid, depth

        #sync up initial values
        self.b_TrimStart.SetMax(vid._maxFrame)
        self.b_TrimStart.SetValue(vid.startFrame)
        self.b_TrimEnd.SetMin(self.b_TrimStart.GetValue() + 1)
        self.b_TrimEnd.SetMax(vid._maxFrame)
        self.b_TrimEnd.SetValue(vid.endFrame)
        self.b_cropOffset.SetValue(vid.offset)
        self.b_zoom.SetValue(vid.zoom)
        self.b_colormap.SetValue(depth.colormap)

        #tweak the interaction behavior
        self.b_zoom.SetIncrement(5)
        self.b_cropOffset.SetIncrement(2)

        #load initial images
        self.i_TrimStart.SetBitmap(vid.trimmerFrame(self.b_TrimStart.GetValue()))
        self.i_TrimEnd.SetBitmap(vid.trimmerFrame(self.b_TrimEnd.GetValue()))
        
        self.Layout()
        
    def updateTrim( self, event ):
        global vid
        #refresh the images, if they need to be updated
        if self.b_TrimStart.GetValue() != vid.startFrame:
            self.i_TrimStart.SetBitmap(vid.trimmerFrame(self.b_TrimStart.GetValue()))
            vid.startFrame = self.b_TrimStart.GetValue()
            self.b_TrimEnd.SetMin(self.b_TrimStart.GetValue() + 1)
        if self.b_TrimEnd.GetValue() != vid.endFrame:
            self.i_TrimEnd.SetBitmap(vid.trimmerFrame(self.b_TrimEnd.GetValue()))
            vid.endFrame = self.b_TrimEnd.GetValue()

        #evaluate the selection's length, in seconds.
        duration = round((vid.endFrame - vid.startFrame) / 30.0, 2)
        self.t_duration.SetLabelText("Clip Duration: " + str(duration) + "s.")

    def updateCrop( self, event ):
        global vid
        if vid.isSquare() and self.b_zoom.GetValue() == 0:
            self.b_cropOffset.SetValue(0)
            return

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
        global vid, depth
        vid.startFrame = self.b_TrimStart.GetValue()
        vid.endFrame = self.b_TrimEnd.GetValue()
        depth.colormap = self.b_colormap.GetValue()
        #nnMod.redMatte = self.b_red.GetValue()

        self.EndModal(wx.ID_OK)


class exporter(layouts.Exporter):  
    def __init__(self, parent):
        #initialize parent class
        layouts.Exporter.__init__(self,parent)
       
        #l.sortData()       
        data = QEMeasurement.generatePreview()
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

        if (QEMeasurement.writeLog(vid.nicename, self.b_exportFilename.GetPath())):
            self.EndModal(wx.ID_OK)
            wx.MessageBox('The file was successfully saved.', 'File Saved.', wx.OK | wx.ICON_INFORMATION)
        else: 
            self.EndModal(wx.ID_CANCEL)
            wx.MessageBox('The file was not saved due to an error.', 'Error.', wx.OK | wx.ICON_ERROR)
            
        
