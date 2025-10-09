import wx
import wx.grid

from os import startfile, getcwd, mkdir
from os.path import isfile, isdir
import ctypes

#sys.path.append("..")
import modules.layouts as layouts       #all gui views.
import modules.imaging as QEi     #openCV image processing
import modules.analysis as QEa   #ML depth analysis
import modules.logging as log


class settings(layouts.VideoSettings):  
    def __init__(self, parent):
        layouts.VideoSettings.__init__(self,parent)

        #sync up initial values
        vid = QEi.Video
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
        vid = QEi.Video
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
        vid = QEi.Video
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
        vid = QEi.Video
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
        vid = QEi.Video
        
        #populate default names
        basename = f"{vid.nicename}_f{str(int(vid.currentFrame))}"
        self.t_imageName.SetValue(basename)
        self.t_depthmapName.SetValue(f"{basename}_depth")
        self.t_screenshotName.SetValue(f"{basename}_ss")
        self.t_pointcloudName.SetValue(f"{basename}_points")
        self.t_measurementName.SetValue(f"{basename}_log")
        self.outputDirectory.SetPath(getcwd() + "\\exports\\")

        self.Layout()
        

    def doSave(self, event):
        #is the output directory valid?
        if isdir(self.outputDirectory.GetPath()) == False:
            dlg = wx.MessageDialog(None, "The output path is not a valid directory.",'Error',wx.ICON_WARNING|wx.OK)
            dlg.ShowModal()
            return

        vid = QEi.Video
        saveCounter = 0
        #image
        if self.cb_image.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_imageName.GetValue()}{self.c_imageFmt.GetStringSelection()}"
            vid.exportImage(fullname)
            saveCounter += 1               

        #depth map
        if self.cb_depthmap.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_depthmapName.GetValue()}{self.c_depthmapFmt.GetStringSelection()}"
            QEa.Depthmap.saveDepthMap(fullname) 
            saveCounter += 1   

        #screenshot
        if self.cb_screenshot.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_screenshotName.GetValue()}{self.c_screenshotFmt.GetStringSelection()}"
            vid.saveScreenshot(fullname)       
            saveCounter += 1   

        #point cloud
        if self.cb_pointcloud.IsChecked():
            fullname = f"{self.outputDirectory.GetPath()}\\{self.t_pointcloudName.GetValue()}{self.c_pointcloudFmt.GetStringSelection()}"
            QEa.Depthmap.savePointCloud(fullname)
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
        self.originalColor = QEa.Depthmap.colormap
        if self.originalColor != 0:
            self.b_colormap.SetValue(True)
        self.i_depthmap.SetBitmap(QEa.Depthmap.getImage())
        info = f"Depthmap Range: {QEa.Depthmap.minmax[0]:.4f} - {QEa.Depthmap.minmax[1]:.4f}"
        self.t_stats.SetLabelText(info)

    def toggleColors( self, event ):
        if self.b_colormap.GetValue() == True:
            QEa.Depthmap.colormap = self.originalColor
        else:
            QEa.Depthmap.colormap = 0
        self.i_depthmap.SetBitmap(QEa.Depthmap.getImage())

    def closeViewer( self, event ):
        self.EndModal(wx.ID_OK)
        