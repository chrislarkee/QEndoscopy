import numpy as np
import os.path as p

from wx import Bitmap as wxb
import cv2 
from PIL import ImageGrab
from math import floor

import vispy
vispy.use(app='wx')
from vispy import scene

class VispyImg:
    @classmethod
    def startup(self, panel):
        self.vispyCanvas = scene.SceneCanvas(parent=panel, app="wx", keys=None, bgcolor='black', resizable=True)
        self.view = self.vispyCanvas.central_widget.add_view()
        
        #initialize blank image placeholder
        imgdata = np.zeros((720, 720, 3)).astype(np.uint8)
        
        #parent=self.vispyCanvas.central_widget, 
        self.mainImage = scene.visuals.Image(imgdata, parent=self.view.scene, texture_format=np.uint8, method='subdivide')
        
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.interactive = False  # Disable user control
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        #self.view.camera.zoom(1, (0, 0))
        #self.view.camera.rect = Rect(0,0,0.5,0.5)

        # Disable all camera/mouse interaction
        #self.vispyCanvas.events.mouse_move.disconnect()   # Disable panning
        #self.vispyCanvas.events.mouse_wheel.disconnect()  # Disable zooming
        #self.vispyCanvas.events.mouse_press.disconnect()
        #self.vispyCanvas.events.mouse_release.disconnect()

        #self.vispyImg.transform = scene.transforms.STTransform(translate=(0, 0), scale=(1, 1))
        self.vispyCanvas.show() 

    @classmethod
    def fixSize(self, panelsize):
        #hack to fix vispy canvas size when embedded in wx
        self.vispyCanvas.size = (panelsize.x, panelsize.y)

        #h, w, _ = self.imageCache.shape
        #self.view.camera.set_range(x=(0, panelsize.x), y=(0, panelsize.y))
        self.view.camera.set_range()
        self.view.camera.flip = (0, 1, 0)  # Y-up correction

class Video:
    ready = False
    offset = 0
    zoom = 0
    overlayEnabled = False
    _kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]], dtype=np.uint8)

    @classmethod
    def loadVideo(self, filename: str):
        if p.exists(filename) == False:
            print("picked file doesn't exist.")
            return None

        #load the data
        self._path = str(filename)
        self.vid = cv2.VideoCapture(self._path)

        #defaults for changeable values
        self.currentFrame = 1
        self.startFrame = 1
        self.endFrame = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))      #it's trimmable

        #parameters whose values we can automatically determine
        #https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        self.nicename = str(p.basename(self._path)).replace(".","_")
        self.imageCache = None
        self._rate = self.vid.get(cv2.CAP_PROP_FPS)                     #the native rate. never changes.        
        #QEMeasurement.framerate = self._rate
        self._maxFrame = self.endFrame                                  #the end of the file        
        self.res = (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.endFrame <= 2:  #hack to determine if he have a video or a photo
            self._maxFrame = 1
            self._singleImage = cv2.imread(self._path)
            self.res = (self._singleImage.shape[1], self._singleImage.shape[0])
        else:
            self._singleImage = None
        self._crop = (0, self.res[1], int(self.res[0] / 2 - self.res[1] / 2), int(self.res[0] / 2 + self.res[1] / 2))  

        self.ready = True

    @classmethod    
    def isSquare(self):
        if self.res[0] == self.res[1]:
            return True
        else:
            return False

    @classmethod
    def getLength(self):
        # converts the length of the video from a frame number into readable timecode    
        frames = self._maxFrame % self._rate
        total_seconds = self._maxFrame // self._rate
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60

        return f" ({hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}:{frames:02.0f})"

    @classmethod
    def nextFrame(self):
        #read the next frame
        if self._maxFrame == 1: 
            ret, rawframe = True, self._singleImage               
        else:
            ret, rawframe = self.vid.read()                   
        
        if ret == False:
            #the end of the video has been reached
            return
        
        self.overlayEnabled = False

        #update the frame caches and evaluate them
        self.currentFrame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
        self.imageCache = rawframe[self._crop[0]:self._crop[1], self._crop[2]:self._crop[3]]
                
        #export
        leftImage = cv2.cvtColor(self.imageCache, cv2.COLOR_BGR2RGB)
        VispyImg.mainImage.set_data(leftImage)
        VispyImg.vispyCanvas.update()        

    @classmethod
    def specificFrame(self, frame):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame - 1))
        #self.overlayEnabled = False
        return self.nextFrame()

    @classmethod
    def prevFrame(self):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(self.currentFrame - 2))
        #self.overlayEnabled = False
        return self.nextFrame()

    @classmethod
    def trimmerFrame(self, frame):
        if self._maxFrame == 1: 
            step1 = self._singleImage  
        else:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame - 1))
            step1 = self.vid.read()[1]

        step2 = step1[self._crop[0]:self._crop[1], self._crop[2]:self._crop[3]]
        step2 = cv2.cvtColor(step2, cv2.COLOR_BGR2RGB)
        step2 = cv2.resize(step2, (500,500))
        step3 = wxb.FromBuffer(500, 500, step2.tobytes())    
        return step3

    @classmethod
    def removeOverlay(self):
        cleanImg = cv2.cvtColor(self.imageCache, cv2.COLOR_BGR2RGB)
        VispyImg.mainImage.set_data(cleanImg)
        VispyImg.vispyCanvas.update()

    @classmethod
    def addOverlay(self, points):         
        #make a blank image buffer
        overlayImg = np.zeros((720, 720, 3), dtype=np.uint8)
        
        #set the points array to a green color
        col = np.array([0,215,60], dtype=np.uint8)
        overlayImg[points] = col

        #dilate the image to make them more visible
        overlayImg = cv2.dilate(overlayImg, self._kernel, iterations=1)
        
        #composite and convert
        finalImage = cv2.cvtColor(self.imageCache, cv2.COLOR_BGR2RGB)
        overlayImg+=finalImage; overlayImg[overlayImg<finalImage]=255

        VispyImg.mainImage.set_data(overlayImg)
        VispyImg.vispyCanvas.update()
        
    @classmethod
    def updateCrop(self, offset, zoom):
        self.offset = offset
        self.zoom = zoom
        self._crop = (int(0 + zoom),
            int(self.res[1] - zoom), 
            int(self.res[0] / 2 - self.res[1] / 2) - offset + zoom,
            int(self.res[0] / 2 + self.res[1] / 2) - offset - zoom) 
    
    @classmethod
    def exportImage(self, filename):
        #cv2.imwrite automatically converts it out of the BGR colorspace, so no conversion is needed.
        if str.lower(filename[-3:]) == "png":
            cv2.imwrite(filename, self.imageCache, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
        elif str.lower(filename[-3:]) == "jpg":
            cv2.imwrite(filename, self.imageCache, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif str.lower(filename[-4:]) == "webp":
            cv2.imwrite(filename, self.imageCache, [int(cv2.IMWRITE_WEBP_QUALITY), 80])

    @classmethod
    def grabScreenshot(self, bbox):
        self.screenshot = ImageGrab.grab(bbox)

    @classmethod       
    def saveScreenshot(self, filepath):
        if str.lower(filepath[-3:]) == "jpg":
            self.screenshot.save(filepath, 'jpeg', quality=90)
        elif str.lower(filepath[-3:]) == "png":
            self.screenshot.save(filepath, 'png', optimize=True)
        elif filepath[-4:] == "webp":
            self.screenshot.save(filepath, 'webp')
        del(self.screenshot)



