#!/usr/bin/env python
#from typing_extensions import Self
import wx
import wx.grid
from os import startfile, getcwd, mkdir
from os.path import isfile, isdir
import ctypes

#sys.path.append("..")
import modules.layouts as layouts       #all gui views.
import modules.imaging as QEImaging     #openCV image processing
import modules.analysis as QEAnalysis   #ML depth analysis
import modules.logging as log
#import modules.measurement as QEMeasurement   #image annotation

vid = None      #The QEImaging instance

pixelScale = 0.0

class main(layouts.MainInterface):  
    def __init__(self,parent):
        #evaluate pixel scaling factor (for screenshots)
        awareness = ctypes.c_int()
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        global pixelScale
        pixelScale = wx.ScreenDC().GetPPI()[0] / 96.0
        ctypes.windll.shcore.SetProcessDpiAwareness(0)

        #self.vid = None
        #initialize parent class
        layouts.MainInterface.__init__(self,parent)

        #bind keyboard shortcuts
        self.Bind(wx.EVT_CHAR_HOOK, self.shortcuts)

        QEAnalysis.VispyPanel.startup(self.vispypanel)

    def shortcuts(self, event):
        if vid == None:
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
        global vid
        #this shows up the 2nd time the user opens a video.
        if vid != None:
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
        if autoload == None:
            log.startup(dlg.GetPath())
            if dlg.GetPath()[-3:] == "csv":
                progress.Destroy()
                wx.MessageBox('CSV files cannot be loaded in this way.', 'Error.', wx.OK | wx.ICON_ERROR)
                return
            else:
                vid = QEImaging.EndoVideo(dlg.GetPath())    
        else:
            if (isfile(autoload) == False):
                progress.Destroy()
                wx.MessageBox('The file does not exist.', 'Error.', wx.OK | wx.ICON_ERROR)
                return
            vid = QEImaging.EndoVideo(autoload)    

        progress.Pulse(newmsg="Loading Depth Libraries...")
        QEAnalysis.Depthmap.setupTorch()

        progress.Pulse(newmsg="Starting Depth Map...")
        #check if there's a depth map available
        depthFile = vid._path[0:-4] + "_depth.mp4"
        if isfile(depthFile):
            #vispy = QEAnalysis.Depthmap.bakedDepthmap=True                     
            wx.MessageBox('A depth file was found and will be used instead of evaluation depth.', 'Error.', wx.OK | wx.ICON_ERROR)
        #else: 
             #vispy = QEDepth.Depth(None, vid)
        
        
        #apply changes from the loaded video to the gui
        progress.Pulse(newmsg="Processing metadata...")

        #enable the locked out buttons
        self.SetTitle("Quantitative Endoscopy: " + vid.nicename)
        self.t_statusText.WriteText(vid.nicename + vid.getLength())     #the title of the window
        self.slider_time.SetMax(vid.endFrame)                           #the limit of the main slider
        self.b_playVideo.Enable(True)
        self.b_showSettings.Enable(True)
        self.b_showmap.Enable(True)
        self.b_recordMeasurement.Enable(True)
        self.b_openExporter.Enable(True)
        self.b_openViewer.Enable(True)
        self.b_colormap.SetValue(False)
        vid.guiSize = self.i_Image.GetSize().Width
                
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
        dlg = viewMeasurements(self)
        result = dlg.ShowModal()

    def openExporter(self, event):
        self.b_playVideo.SetValue(False)
         #pop up the savewizard layout, as a modal window
        dlg = saveWizard(self)
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

        global vid
        self.timer.Start(int(1000.0 / 30.0))
        QEAnalysis.VispyPanel.clearPoints()

    def playVideoTick(self, event):
        global vid
        if self.b_playVideo.GetValue() == False:
            self.timer.Stop()
            QEAnalysis.Depthmap.processDepth(vid.imageCache)
            self.slider_distance.SetMax(int(QEAnalysis.Depthmap.minmax[1] * 1000 - 1))
        else:
            #left frame only
            #QEMeasurement.overlayEnabled = False
            self.i_Image.SetBitmap(vid.nextFrame())
            self.slider_time.SetValue(int(vid.currentFrame))

    def framePrevious(self, event):
        self.b_playVideo.SetValue(False)

        global vid
        if (vid == None):
            return
        #left & right frame
       #QEMeasurement.overlayEnabled = False
        self.i_Image.SetBitmap(vid.prevFrame())
        QEAnalysis.Depthmap.processDepth(vid.imageCache)
        self.slider_time.SetValue(int(vid.currentFrame))

    def frameNext(self, event):  
        self.b_playVideo.SetValue(False)
        global vid
        if (vid == None):
            return
        #left & right frames
        #QEMeasurement.overlayEnabled = False
        self.i_Image.SetBitmap(vid.nextFrame())
        QEAnalysis.Depthmap.processDepth(vid.imageCache)
        self.slider_time.SetValue(int(vid.currentFrame))
        self.slider_distance.SetMin(int(QEAnalysis.Depthmap.minmax[0] * 1000 + 1))
        self.slider_distance.SetMax(int(QEAnalysis.Depthmap.minmax[1] * 1000 - 1))            

    def scrub(self, event):
        self.b_playVideo.SetValue(False)
        global vid
        if (vid == None):
            return
        desiredFrame = int(self.slider_time.GetValue())
        #left frame only
        self.i_Image.SetBitmap(vid.specificFrame(desiredFrame))
        #self.i_Depth.SetBitmap(vid.blankFrame())
        self.slider_time.SetValue(int(vid.currentFrame))

    def scrubDone(self,event):
        global vid
        if vid == None:
            return
        #right frame only
        #QEMeasurement.overlayEnabled = False
        QEAnalysis.Depthmap.processDepth(vid.imageCache)
        self.slider_distance.SetMin(int(QEAnalysis.Depthmap.minmax[0] * 1000 + 1))
        self.slider_distance.SetMax(int(QEAnalysis.Depthmap.minmax[1] * 1000 - 1))


    ###LEFT SIDE FUNCTIONS###
    def showSettings(self, event):
        self.b_playVideo.SetValue(False)
        #QEMeasurement.overlayEnabled = False
        #pop up the settings layout, as a modal window
        dlg = settings(self)
        if dlg.ShowModal() == wx.ID_OK:
            #the modal has closed; apply the changed parameters to the GUI
            global vid
            self.slider_time.SetMin(vid.startFrame)
            self.slider_time.SetMax(vid.endFrame)
            vid.currentFrame = vid.startFrame            
            self.i_Image.SetBitmap(vid.specificFrame(vid.startFrame))
            QEAnalysis.Depthmap.processDepth(vid.imageCache)
            self.slider_time.SetValue(int(vid.startFrame))
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
        QEAnalysis.VispyPanel.changeSlice(d, x, y, measure=False)

    def slicerDone( self, event ):
        d, x, y = self.slider_distance.GetValue(), self.slider_planeX.GetValue(), self.slider_planeY.GetValue()
        QEAnalysis.VispyPanel.changeSlice(d, x, y, measure=True)

    def changeTool(self, event):
        #self.clearMeasurements(event)   
        wx.MessageBox("Not Implemented.", "Error")

    def recordMeasurement(self, event):
        #finalize data
        global vid, pixelScale
        #QEMeasurement.currentEntry.frame = vid.currentFrame       
        status = self.t_statusText.GetValue() + "\nStored."
        self.t_statusText.SetValue(status)
        
        #prepare screenshot metadata
        screenshotPath = getcwd() + "\\logs\\" + vid.nicename
        if isdir(screenshotPath) == False:
            mkdir(screenshotPath)
        #shot_filename = screenshotPath + "\\" + vid.nicename + "_" + str(QEMeasurement.counter()).zfill(2) + ".png"
        rect = self.GetRect() # pixelScale
        crop = (rect.Left, rect.Top, rect.Right, rect.Bottom)     #wx and PIL use different formats.

        #take screenshot
        #QEImaging.takeScreenshot(crop,shot_filename)       

        #commit the log
        #QEMeasurement.store()

    def showmap( self, event ):
        dlg = depthmapViewer(self)
        dlg.ShowModal()
        self.b_colormap.SetValue(QEAnalysis.Depthmap.colormap)

    def resetCamera( self, event ):
        QEAnalysis.VispyPanel.resetCam()

    def clearMeasurements(self, event):
        #QEMeasurement.overlayEnabled = False
        #self.i_Depth.SetBitmap(depth.postProcess(vid.guiSize))
        #self.i_Image.SetBitmap(vid.refreshAnnoation())
        self.t_statusText.SetValue("Annotation Cleared.")

    def resetPlane( self, event ):
        self.slider_planeX.SetValue(0)
        self.slider_planeY.SetValue(0)        
        QEAnalysis.VispyPanel.changeSlice(wx.IdleEvent(), self.slider_distance.GetValue())

    def toggleColors( self, event ):
        QEAnalysis.Depthmap.colormap = self.b_colormap.GetValue()
        QEAnalysis.Depthmap.processDepth(vid.imageCache)

    ###WINDOW MANAGEMENT###
    #if the window has been resized, scale the images to fit the new shape.
    def setSize(self, event):        
        global vid
        QEAnalysis.VispyPanel.fixSize(self.vispypanel.GetSize())
        if vid == None:
            return

        if self.i_Image.GetSize().Width != vid.guiSize:
            vid.guiSize = self.i_Image.GetSize().Width
            self.i_Image.SetBitmap(vid.nextFrame())
            #QEAnalysis.Depthmap.processDepth(vid.imageCache)
            self.slider_time.SetValue(int(vid.currentFrame))        
        self.Layout()

    def saveAndQuit( self, event ):
        dlg = wx.MessageDialog(None, "Are you sure you want to quit?",'QE',wx.YES_NO)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            wx.Exit()


class settings(layouts.VideoSettings):  
    def __init__(self, parent):
        layouts.VideoSettings.__init__(self,parent)
        global vid

        #sync up initial values
        self.b_TrimStart.SetMax(vid._maxFrame)
        self.b_TrimStart.SetValue(vid.startFrame)
        self.b_TrimEnd.SetMin(self.b_TrimStart.GetValue() + 1)
        self.b_TrimEnd.SetMax(vid._maxFrame)
        self.b_TrimEnd.SetValue(vid.endFrame)
        self.b_cropOffset.SetValue(vid.offset)
        self.b_zoom.SetValue(vid.zoom)

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
        global vid
        vid.startFrame = self.b_TrimStart.GetValue()
        vid.endFrame = self.b_TrimEnd.GetValue()
        self.EndModal(wx.ID_OK)


class viewMeasurements(layouts.Measurements):  
    def __init__(self, parent):
        #initialize parent class
        layouts.Measurements.__init__(self,parent)
       
        #l.sortData()       
        #data = QEMeasurement.generatePreview()
        data = None
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

        if (False):#QEMeasurement.writeLog(vid.nicename, self.b_exportFilename.GetPath())):
            self.EndModal(wx.ID_OK)
            wx.MessageBox('The file was successfully saved.', 'File Saved.', wx.OK | wx.ICON_INFORMATION)
        else: 
            self.EndModal(wx.ID_CANCEL)
            wx.MessageBox('The file was not saved due to an error.', 'Error.', wx.OK | wx.ICON_ERROR)
            
class saveWizard(layouts.SaveWizard):
    def __init__(self, parent):
        #initialize parent class
        layouts.SaveWizard.__init__(self,parent)

        global vid
        #populate default names
        basename = f"{vid.nicename}_f{str(int(vid.currentFrame))}"
        self.t_imageName.SetValue(basename)
        self.t_depthmapName.SetValue(f"{basename}_depth")
        self.t_pointcloudName.SetValue(f"{basename}_points")
        self.t_measurementName.SetValue(f"{basename}_log")
        self.outputDirectory.SetPath(getcwd() + "\\exports\\")

        self.Layout()
        
        #autotrigger the directory picker?
        #if self.outputDirectory.ShowModal() == wx.ID_CANCEL:
        #    self.EndModal(wx.ID_CANCEL) 
        

    def doSave(self, event):
        #is the output directory valid?
        if isdir(self.outputDirectory.GetPath()) == False:
            dlg = wx.MessageDialog(None, "The output path is not a valid directory.",'Error',wx.ICON_WARNING|wx.OK)
            dlg.ShowModal()
            return

        global vid
        saveCounter = 0
        #image
        if self.cb_image.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_imageName.GetValue()}{self.c_imageFmt.GetStringSelection()}"
            vid.exportImage(fullname)
            saveCounter += 1               

        #depth map
        if self.cb_depthmap.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_depthmapName.GetValue()}{self.c_depthmapFmt.GetStringSelection()}"
            QEAnalysis.Depthmap.saveDepthMap(fullname) 
            saveCounter += 1          

        #point cloud
        if self.cb_pointcloud.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_pointcloudName.GetValue()}{self.c_pointcloudFmt.GetStringSelection()}"
            QEAnalysis.PointCloud.savePointCloud(fullname)
            saveCounter += 1
        
        #table
        if self.cb_measurement.IsChecked():
            #can be XLSX or CSV
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_measurementName.GetValue()}{self.c_measurementFmt.GetStringSelection()}"
            #logging save function
            saveCounter += 1

        #close the save wizard, because we're done        
        self.EndModal(wx.ID_OK)
        dlg = wx.MessageDialog(None, f"Files saved: {saveCounter}",'Success',wx.ICON_INFORMATION|wx.OK)
        dlg.ShowModal()
        
        
class depthmapViewer(layouts.DepthmapViewer): 
    def __init__(self, parent):
        #initialize parent class
        layouts.DepthmapViewer.__init__(self,parent)
        self.b_colormap.SetValue(QEAnalysis.Depthmap.colormap)
        self.i_depthmap.SetBitmap(QEAnalysis.Depthmap.getImage())
        info = f"Depthmap Range: {QEAnalysis.Depthmap.minmax[0]:.4f} - {QEAnalysis.Depthmap.minmax[1]:.4f}"
        self.t_stats.SetLabelText(info)

    def toggleColors( self, event ):
        QEAnalysis.Depthmap.colormap = self.b_colormap.GetValue()
        self.i_depthmap.SetBitmap(QEAnalysis.Depthmap.getImage())

    def closeViewer( self, event ):
        self.EndModal(wx.ID_OK)