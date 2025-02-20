import cv2 
from wx import Bitmap as wxb
from PIL import Image, ImageGrab
from math import floor
import os.path as p

import modules.measurement as QEMeasurement

class EndoVideo:
    def __init__(self, filename: str):
        if p.exists(filename) == False:
            print("picked file doesn't exist.")
            return None

        #load the data
        self._path = str(filename)
        self.vid = cv2.VideoCapture(self._path)
        #opens and sets up the first frame      

        #defaults for changeable values
        self.currentFrame = 1
        self.startFrame = 1
        self.endFrame = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))      #it's trimmable
        self.offset = 0
        self.zoom = 0
        self.guiSize = 500

        #parameters whose values we can automatically determine
        #https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        self.nicename = str(p.basename(self._path)).replace(".","_")
        self.imageCache = None
        self._rate = self.vid.get(cv2.CAP_PROP_FPS)                     #the native rate. never changes.        
        QEMeasurement.framerate = self._rate
        self._maxFrame = self.endFrame                                  #the end of the file        
        self.res = (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.endFrame == 1:  #hack to determine if he have a video or a photo
            self._singleImage = cv2.imread(self._path)
            self.res = (self._singleImage.shape[1], self._singleImage.shape[0])   
        self._crop = (0, self.res[1], int(self.res[0] / 2 - self.res[1] / 2), int(self.res[0] / 2 + self.res[1] / 2))  
        
    def isSquare(self):
        if self.res[0] == self.res[1]:
            return True
        else:
            return False

    def getLength(self):
        # returns nice looking string: (5:00.00)
        minutes = str(floor((self._maxFrame / self._rate) / 60)).zfill(2)
        seconds = str(round((self._maxFrame / self._rate) % 60, 2)).zfill(5)
        return " (" + minutes + ":" + seconds + ")"        

    def nextFrame(self):
        #read the next frame
        if self._maxFrame == 1: 
            ret, rawframe = True, self._singleImage               
        else:
            ret, rawframe = self.vid.read()                   
        
        if ret == False:
            #the end of the video has been reached
            return

        #update the frame caches and evaluate them
        self.currentFrame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
        self.imageCache = rawframe[self._crop[0]:self._crop[1], self._crop[2]:self._crop[3]]
                
        #export
        leftImage = cv2.cvtColor(self.imageCache, cv2.COLOR_BGR2RGB)
        leftImage = Image.fromarray(leftImage).resize((self.guiSize ,self.guiSize))                     
        leftImage = wxb.FromBuffer(self.guiSize, self.guiSize, leftImage.tobytes())       
        return leftImage

    def refreshAnnoation(self):
        leftImage = cv2.cvtColor(self.imageCache, cv2.COLOR_BGR2RGB)
        leftImage = QEMeasurement.addOverlay(leftImage)
        leftImage = Image.fromarray(leftImage).resize((self.guiSize ,self.guiSize))                     
        leftImage = wxb.FromBuffer(self.guiSize, self.guiSize, leftImage.tobytes())       
        return leftImage

    def specificFrame(self, frame):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame - 1))
        return self.nextFrame()

    def prevFrame(self):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(self.currentFrame - 2))
        return self.nextFrame()

    def trimmerFrame(self, frame):
        if self._maxFrame == 1: 
            step1 = self._singleImage  
        else:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame - 1))
            step1 = self.vid.read()[1]

        step2 = step1[self._crop[0]:self._crop[1], self._crop[2]:self._crop[3]]
        step2 = cv2.cvtColor(step2, cv2.COLOR_BGR2RGB)
        step3 = Image.fromarray(step2).resize((500,500))  
        step3 = wxb.FromBuffer(500, 500, step3.tobytes())    
        return step3
    
    def blankFrame(self):
        black = Image.new(mode="RGB", size=(self.guiSize, self.guiSize))
        blackWX = wxb.FromBuffer(self.guiSize, self.guiSize, black.tobytes())
        return blackWX

    def updateCrop(self, offset, zoom):
        self.offset = offset
        self.zoom = zoom
        self._crop = (int(0 + zoom),
            int(self.res[1] - zoom), 
            int(self.res[0] / 2 - self.res[1] / 2) - offset + zoom,
            int(self.res[0] / 2 + self.res[1] / 2) - offset - zoom) 

    def exportImage(self, filename):
        #cv2.imwrite automatically converts it out of the BGR colorspace, so no conversion is needed.
        if str.lower(filename[-3:]) == "png":
            cv2.imwrite("exports//" + filename, self.imageCache, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
        elif str.lower(filename[-3:]) == "jpg":
            cv2.imwrite("exports//" + filename, self.imageCache, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

def takeScreenshot(bbox, filepath):
    screenshot = ImageGrab.grab(bbox)
    screenshot.save(filepath, 'PNG')



