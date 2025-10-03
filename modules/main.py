#!/usr/bin/env python
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

vid = None          #The QEImaging instance
pixelScale = 0.0    #screenshot hack

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
        log.startup()
        if autoload == None:            
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
        log.framerate = vid._rate

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
        self.b_jumpFrame.Enable(True)
        self.b_vischoice.Enable(True)
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

        #cache a screenshot before the popup, incase the user wants to save it
        rect = self.GetRect() # pixelScale
        crop = (rect.Left, rect.Top, rect.Right, rect.Bottom)     #wx and PIL use different formats.
        vid.grabScreenshot(crop)       

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
            #self.slider_distance.SetMax(int(QEAnalysis.Depthmap.minmax[1] * 1000 - 1))
            QEAnalysis.Depthmap.processDepth(vid.imageCache)
            self.slicerDone(wx.IdleEvent)
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
        QEAnalysis.Depthmap.processDepth(vid.imageCache)
        self.slider_distance.SetMin(int(QEAnalysis.Depthmap.minmax[0] * 1000 + 1))
        self.slider_distance.SetMax(int(QEAnalysis.Depthmap.minmax[1] * 1000 - 1))
        self.slicerDone(event)

    def pickPoint( self, event ):
        #Don't measure if the video is playing
        if self.b_playVideo.GetValue() == True:
            return
        #Don't measure if there's no video loaded
        global vid
        if vid == None:
            return
        
        mouseCoords = event.GetPosition()
        targetCoords = (int(mouseCoords[0] / vid.guiSize * 720), int(mouseCoords[1] / vid.guiSize * 720))
        threshold = QEAnalysis.Depthmap.depth_Cache[targetCoords[1]][targetCoords[0]]
        self.slider_distance.SetValue(int(threshold * 1000))

        #trigger the slider change event to update the plane
        self.slicerDone(event)

    ###LEFT SIDE FUNCTIONS###
    def jumpFrame( self, event ):
        global vid
        dlg = wx.GetNumberFromUser(
            message="",
            prompt="Frame: ", 
            caption="Jump to Frame", 
            value=int(vid.currentFrame), 
            min=int(vid.startFrame), 
            max=int(vid.endFrame))
        
        if dlg != -1:
            self.slider_time.SetValue(dlg)
            self.i_Image.SetBitmap(vid.specificFrame(dlg))
            QEAnalysis.Depthmap.processDepth(vid.imageCache)
            self.slicerDone(event)


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
        vid.overlayEnabled = False        
        QEAnalysis.VispyPanel.changeSlice(d, x, y, measure=False)

    def slicerDone( self, event ):
        d, x, y = self.slider_distance.GetValue(), self.slider_planeX.GetValue(), self.slider_planeY.GetValue()
        QEAnalysis.VispyPanel.changeSlice(d, x, y, measure=True)

        overlayUpdate, overlayPoints = QEAnalysis.VispyPanel.getIntersection()
        if overlayUpdate == True:
            vid.overlayEnabled = True
            self.i_Image.SetBitmap(vid.addOverlay(overlayPoints))
        if overlayUpdate == False and vid.overlayEnabled:
            vid.overlayEnabled = False
            self.i_Image.SetBitmap(vid.removeOverlay())

        log.currentEntry.distance = d / 1000.0
        log.currentEntry.rotation = (x, y)
        self.t_statusText.AppendText(f"\nArea: {log.currentEntry.CSarea}")

    def changeTool(self, event):
        #self.clearMeasurements(event)   
        wx.MessageBox("Not Implemented.", "Error")

    def recordMeasurement(self, event):
        #finalize data
        global vid, pixelScale
        log.currentEntry.frame = vid.currentFrame
        #QEMeasurement.currentEntry.frame = vid.currentFrame
    
        self.t_statusText.AppendText("\nStored.")
        
        #commit the log
        log.store()

    def showmap( self, event ):
        dlg = depthmapViewer(self)
        originalColor = QEAnalysis.Depthmap.colormap
        dlg.ShowModal()
        self.b_colormap.SetSelection(QEAnalysis.Depthmap.colormap)
        if QEAnalysis.Depthmap.colormap != originalColor:
            QEAnalysis.Depthmap.processDepth(vid.imageCache)

    def resetCamera( self, event ):
        QEAnalysis.VispyPanel.resetCam()

    def resetPlane( self, event ):
        self.slider_planeX.SetValue(0)
        self.slider_planeY.SetValue(0)        
        QEAnalysis.VispyPanel.changeSlice(self.slider_distance.GetValue(), measure=True)
        self.t_statusText.SetLabelText("")

    def changeVisibility( self, event ):     
        QEAnalysis.VispyPanel.updateVisibility(self.b_vischoice.GetSelection())

    def setColors( self, event ):
        QEAnalysis.Depthmap.colormap = self.b_colormap.GetSelection()
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
        self.b_TrimStart.SetValue(vid.startFrame)
        self.b_TrimStart.SetMax(vid._maxFrame)
        self.s_TrimStart.SetValue(vid.startFrame)
        self.s_TrimStart.SetMax(vid._maxFrame)

        self.b_TrimEnd.SetMin(self.b_TrimStart.GetValue() + 1)
        self.b_TrimEnd.SetMax(vid._maxFrame)
        self.b_TrimEnd.SetValue(vid.endFrame)
        self.s_TrimEnd.SetMin(self.b_TrimStart.GetValue() + 1)
        self.s_TrimEnd.SetMax(vid._maxFrame)
        self.s_TrimEnd.SetValue(vid.endFrame)

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

        oldstart = vid.startFrame
        oldend   = vid.endFrame

        #refresh the images, if they need to be updated
        if self.b_TrimStart.GetValue() != oldstart:
            newValue = self.b_TrimStart.GetValue()
            self.i_TrimStart.SetBitmap(vid.trimmerFrame(newValue))
            self.s_TrimStart.SetValue(newValue)
            self.b_TrimEnd.SetMin(newValue + 1)
            self.s_TrimEnd.SetMin(newValue + 1)

        if self.s_TrimStart.GetValue() != oldstart:
            newValue = self.s_TrimStart.GetValue()
            self.i_TrimStart.SetBitmap(vid.trimmerFrame(newValue))
            self.b_TrimStart.SetValue(newValue)
            self.b_TrimEnd.SetMin(newValue + 1)
            self.s_TrimEnd.SetMin(newValue + 1)

        if self.b_TrimEnd.GetValue() != oldend:
            newValue = self.b_TrimEnd.GetValue()
            self.i_TrimEnd.SetBitmap(vid.trimmerFrame(newValue))
            self.s_TrimEnd.SetValue(newValue)

        if self.s_TrimEnd.GetValue() != oldend:
            newValue = self.s_TrimEnd.GetValue()
            self.i_TrimEnd.SetBitmap(vid.trimmerFrame(newValue))
            self.b_TrimEnd.SetValue(newValue)

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
        self.fillTable() 

    def fillTable(self):               
        data = log.generatePreview()
        if (len(data) < 1):
            return
        
        #put data into the grid
        self.grid.ClearGrid()
        self.grid.AppendRows(len(data))
        for row in range(0, len(data)):        
            for col in range(0,8):
                cellContents = str(data[row][col])
                self.grid.SetCellValue(row + 1, col, cellContents)

        self.grid.AutoSizeColumns()
        self.Layout()  

    def clearAll(self, event):
        log.startup()
        self.grid.ClearGrid()
        #self.Layout()

    def clearLast(self, event):
        log.clearLast()
        self.fillTable()

    def closeTable(self, event):
        self.EndModal(wx.ID_OK)

            
class saveWizard(layouts.SaveWizard):
    def __init__(self, parent):
        #initialize parent class
        layouts.SaveWizard.__init__(self,parent)

        global vid
        #populate default names
        basename = f"{vid.nicename}_f{str(int(vid.currentFrame))}"
        self.t_imageName.SetValue(basename)
        self.t_depthmapName.SetValue(f"{basename}_depth")
        self.t_screenshotName.SetValue(f"{basename}_ss")
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

        #screenshot
        if self.cb_screenshot.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_screenshotName.GetValue()}{self.c_screenshotFmt.GetStringSelection()}"
            vid.saveScreenshot(fullname)       
            saveCounter += 1   

        #point cloud
        if self.cb_pointcloud.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_pointcloudName.GetValue()}{self.c_pointcloudFmt.GetStringSelection()}"
            QEAnalysis.Depthmap.savePointCloud(fullname)
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
        self.originalColor = QEAnalysis.Depthmap.colormap
        if self.originalColor != 0:
            self.b_colormap.SetValue(True)
        self.i_depthmap.SetBitmap(QEAnalysis.Depthmap.getImage())
        info = f"Depthmap Range: {QEAnalysis.Depthmap.minmax[0]:.4f} - {QEAnalysis.Depthmap.minmax[1]:.4f}"
        self.t_stats.SetLabelText(info)

    def toggleColors( self, event ):
        if self.b_colormap.GetValue() == True:
            QEAnalysis.Depthmap.colormap = self.originalColor
        else:
            QEAnalysis.Depthmap.colormap = 0
        self.i_depthmap.SetBitmap(QEAnalysis.Depthmap.getImage())

    def closeViewer( self, event ):
        self.EndModal(wx.ID_OK)