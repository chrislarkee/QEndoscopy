import vispy
vispy.use(app='wx')
from vispy import scene
from vispy.visuals.transforms.linear import MatrixTransform

import wx
import numpy as np
from os.path import isfile
import cv2 
#from cv2 import convexHull, contourArea, drawContours, cvtColor

import torch as torch
#import torch.nn as nn
import modules.depthmap as dm
import modules.logging as log

class VispyPanel():
    @classmethod  
    def startup(self, panel):
        #start up the vispy scene
        self.canvas = scene.SceneCanvas(parent=panel, app="wx", keys='interactive', resizable=True)
        self.canvas.bgcolor = (0, 0, 0, 1) #indirectly enable alpha
        self.view = self.canvas.central_widget.add_view()

        self.view.camera = 'turntable'
        self.resetCam()
        
        #parent everything to this node so +Z is 'forward' in screen space
        self.rootNode = scene.node.Node(parent=self.view.scene, name="root")
        rot1 = MatrixTransform()
        rot1.rotate(90, (0, 0, 1))
        rot1.rotate(-90, (1, 0, 0))
        self.rootNode.transform = rot1
        self.view.add(self.rootNode)        

        #create persistent scene objects
        scene.visuals.XYZAxis(parent=self.view.scene, width=3, name='axis')        
        self.slicer = scene.visuals.Plane(width=4, height=4, direction='+z', color=(0.1, 0.1, 0.8, 0.4), parent=self.rootNode, name="slicer")
        self.slicer.order = 1   #fixes alpha sorting
        self.allPoints = scene.visuals.Markers(parent=self.rootNode, scaling='scene', alpha=0.5)
        self.allPoints.antialias = 0
        self.planePoints = scene.visuals.Markers(parent=self.rootNode, scaling='fixed')
        self.planePoints.order = -1

        self.canvas.show() 

    @classmethod
    def updatePoints(self, points, colors='white'):
        points = np.reshape(points, (-1, 3))
        #this replaces the old points but keeps the same object
        self.allPoints.set_data(pos=points, symbol='disc', size=0.002, face_color=colors,
            edge_width_rel=0)
    
    @classmethod
    def clearPoints(self):
        nopoints = np.zeros((1, 3), dtype=np.float32)
        self.allPoints.set_data(pos=nopoints)

    @classmethod
    def changeSlice(self, sliceDepth, sliceX=0, sliceY=0, measure=False):
        slicerMatrix = MatrixTransform()        
        slicerMatrix.rotate(sliceX, (1, 0, 0))
        slicerMatrix.rotate(sliceY, (0, 1, 0))
        planeDepth = float(sliceDepth / 1000.0)
        slicerMatrix.translate((0, 0, planeDepth))
        self.slicer.transform = slicerMatrix
        
        #optimization: exit early if the user is holding the mouse button on the slider
        if measure == False:
            return
        
        #this function measures and updates the points on the slicing plane
        dm.PointCloud.measureIntersection(slicerMatrix)

        #caches are updated:
        self.planePoints.set_data(pos=dm.PointCloud.pointsOnPlane, symbol='diamond', size=7, face_color=(0, 0.843, 0.235), 
                                  edge_width_rel=0, edge_color=(1,1,1,0.1)) 

        #update log
        log.currentEntry.distance = planeDepth
        log.currentEntry.rotation = (sliceX, sliceY)
        log.currentEntry.points = len(dm.PointCloud.pointsOnPlane)
        log.currentEntry.CSarea = str(round(dm.PointCloud.area,5))

    @classmethod
    def getIntersection(self):
        truecount = np.count_nonzero(dm.PointCloud.slice_mask)
        if truecount == 0:
            return False, dm.PointCloud.slice_mask
        else:
            return True, dm.PointCloud.slice_mask

    
    @classmethod
    def resetCam(self):
        self.view.camera.center = (0, 0, 0)
        #emptyMatrix = MatrixTransform() 
        #self.view.camera.transform = emptyMatrix

        self.view.camera.elevation = 0
        self.view.camera.azimuth   = 0
        self.view.camera.fov       = 30.0
        self.view.camera.distance  = 2.0

    @classmethod
    def resetPlane(self,event):
        #remove the rotation of the slicing plane
        self.slice_x.SetValue(0)
        self.slice_y.SetValue(0)
        self.changeSlice(wx.IdleEvent())

    @classmethod
    def fixSize(self, panelsize):
        #hack to fix vispy canvas size when embedded in wx
        self.canvas.size = (panelsize.x, panelsize.y)

class Depthmap():
    #cache parameters
    bakedDepthmap = False
    colormap = 0            
    dataCache = np.zeros(1)    #a slot for last generated depthmap
    minmax = (0,1)
    
    _maskData = np.empty(0)     #the black circle that is sometimes used
    _torchReady = False

    def colorEnum(l):
        if l == 1:
            return cv2.COLORMAP_INFERNO            
        elif l == 2:
            return cv2.COLORMAP_BONE
        elif l == 3:
            return cv2.COLORMAP_RAINBOW
        else:
            return cv2.COLOR_GRAY2BGR   #aka 0
        

    @classmethod
    def setupTorch(self):
        #initialize Depthmap NN
        self.model = dm.DepthUtilities.build_unet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isfile("includes\\model.pth"):
            self.model.load_state_dict(torch.load("includes\\model.pth", map_location=self.device, weights_only=True))
        else:
            self.model.load_state_dict(torch.load("model.pth", map_location=self.device, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        self._torchReady = True    

    @classmethod
    def processDepth(self, image):
        if self._torchReady == False:
            self.setupTorch()

        nnInput = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if nnInput.shape[0] != 720:
            nnInput = cv2.resize(nnInput,(720,720))
        nnInput = torch.from_numpy(nnInput[np.newaxis, ...]).float()/255 - 0.5
        nnInput = nnInput.unsqueeze(1).to(self.device)
        self.dataCache = self.model(nnInput)
        self.dataCache = self.dataCache.squeeze().detach().cpu().numpy()

        self.minmax = (self.dataCache.min(), self.dataCache.max())
        log.currentEntry.minmax = (str(round(self.minmax[0], 4)), str(round(self.minmax[1], 4)))

        dm.PointCloud.generatePointCloud(self.dataCache)
        
        if self.colormap != 0:
            colorized = np.interp(self.dataCache, (self.minmax[0], self.minmax[1]), (255,0)).astype('uint8')
            colorized = cv2.applyColorMap(colorized, self.colorEnum(self.colormap))
            colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
            colorized = np.reshape(colorized, (-1,3))
            colorized = (colorized / 255.0).astype(np.float32)
            VispyPanel.updatePoints(dm.PointCloud.pointCloudCache, colorized)
        else:
            VispyPanel.updatePoints(dm.PointCloud.pointCloudCache)
  
    @classmethod        
    def getImage(self, guiSize=720):
        #create the circle if it hasn't been drawn yet
        if self._maskData.size == 0:                
                self._maskData = np.full((720, 720), 0.0, dtype=np.uint8)
                cv2.circle(self._maskData, (360, 360), 352, 1.0, -1, lineType=cv2.LINE_AA)

        if self.colormap != 0:
            post = np.interp(self.dataCache, (self.minmax[0], self.minmax[1]), (255,0)).astype('uint8')
            post = post * self._maskData      #black circle matte
            post = cv2.applyColorMap(post, self.colorEnum(self.colormap))
            post = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)
        else:
            post = np.interp(self.dataCache, (self.minmax[0], self.minmax[1] * 0.99), (0,255)).astype('uint8')            
            post = cv2.cvtColor(post, cv2.COLOR_GRAY2RGB)
            np.clip(post, 0, 254, post)

        if np.shape(post) != (guiSize, guiSize, 3):
            post = cv2.resize(post, (guiSize, guiSize), interpolation=cv2.INTER_CUBIC)

        finalImage = wx.Bitmap.FromBuffer(guiSize, guiSize, post.tobytes())
        return finalImage
    
    @classmethod  
    def saveDepthMap(self, filename):
        #data formats
        if str.lower(filename[-3:]) == "npy":
            np.save(filename, self.dataCache)
            return
        if str.lower(filename[-3:]) == "csv":
            np.savetxt(filename, self.dataCache, fmt="%.5", delimiter=",")            
            return
        
        #image formats
        if self._maskData.size == 0:                
            self._maskData = np.full((720, 720), 0.0, dtype=np.uint8)
            cv2.circle(self._maskData, (360, 360), 352, 1.0, -1, lineType=cv2.LINE_AA)

        if self.colormap != 0:
            post = np.interp(self.dataCache, (self.minmax[0], self.minmax[1]), (255,0)).astype('uint8')
            post = post * self._maskData      #black circle matte
            post = cv2.applyColorMap(post, self.colorEnum(self.colormap))
        else:
            post = np.interp(self.dataCache, (self.minmax[0], self.minmax[1] * 0.99), (0,255)).astype('uint8')            
            post = cv2.cvtColor(post, cv2.COLOR_GRAY2RGB)
            np.clip(post, 0, 254, post)

        if str.lower(filename[-3:]) == "png":
            cv2.imwrite(filename, post, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
        elif str.lower(filename[-3:]) == "jpg":
            cv2.imwrite(filename, post, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    @classmethod        
    def savePointCloud(self, filename):
        dm.PointCloud.savePointCloud(filename)
        