import cv2 
from wx import Bitmap as wxb
from PIL import Image
from os import getcwd
from numpy import interp, save, zeros

try:
    import torch            #midas
    import urllib.request   #midas
    import midas            #hints for Nuitka
    import midas.midas      #hints for Nuitka
except:
    pass

import modules.measurement as me
import modules.depthmap as dm

class MidasSetup:
    def __init__(self, depthmap, vid):
        if depthmap != None:
            self.depthmapEnabled = True
            self._dm = dm.DepthmapSetup(depthmap, vid)
        else:
            self.depthmapEnabled = False
            self.vid = None
            #initialize midas
            #model_type = "DPT_BEiT_L_512"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
            model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
            #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

            try:
                self._midas = torch.hub.load('midas', model_type, source='local')
                self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                self._midas.to(self._device)
                self._midas.eval()

                midas_transforms = torch.hub.load('midas', "transforms", source='local')
                if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                    self._transform = midas_transforms.dpt_transform
                else:
                    self._transform = midas_transforms.small_transform
            except:
                pass

        #parameters to cache
        self.dataCache = zeros(1)             #a slot for the data cache
        self.redMatte = True    
        self.minmax = (0,50)
        

    def processDepth(self, imageCache, guiSize):  
        if self.depthmapEnabled:
            self.dataCache = self._dm.lookupDepth()
            return self.postProcess(guiSize)
        else:
            pass
            ##choose a method:
            #return self._process_old(imageCache, guiSize)
            ##return self._process_2way(imageCache, guiSize)
            #return self._process_4way(imageCache, guiSize)
            #return self._process_accumulate(imageCache, guiSize)

    def _process_old(self, imageCache, guiSize):
        #blurring it filters out the pixel screen
        imageCache = cv2.GaussianBlur(imageCache,(3,3),0)

        #flood fill?
        #cv2.floodFill(imageCache, None, (1,1), (255, 255, 255))

        #the midas function
        input_batch = self._transform(imageCache).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=imageCache.shape[:2],mode="bicubic",align_corners=True,).squeeze()

        #cache these actual values for analysis later
        self.dataCache  = prediction.cpu().numpy()

        rightImage = self.postProcess(guiSize)
        return rightImage

    def _process_2way(self, imageCache, guiSize):       
        #generate masks
        if self.redMatte == True:
            targetSize = int(imageCache.shape[0])
            mask = zeros((targetSize,targetSize), dtype="uint8")
            mask = cv2.circle(mask, (int(targetSize/2), int(targetSize/2)), int(targetSize/2 - 1), 255, cv2.FILLED)
            red  = zeros((targetSize, targetSize, 3), dtype="uint8")
            red[:] = (70, 68, 148)      #the red color, BGR uint8
            red = cv2.circle(red, (int(targetSize/2), int(targetSize/2)), int(targetSize/2 - 1), 0, cv2.FILLED)        

            #composite masks
            imageCache = cv2.bitwise_and(imageCache, imageCache, mask=mask)
            imageCache = cv2.add( imageCache, red)            

        #blurring it filters out the pixel screen
        imageCache = cv2.GaussianBlur(imageCache,(3,3),0)
        #cv2.imshow("Validation", imageCache)
        
        #the midas function 1
        input_batch = self._transform(imageCache).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=imageCache.shape[:2],mode="bicubic",align_corners=True,).squeeze()
        analysis1  = prediction.cpu().numpy()

        ##DOUBLEFLIP##
        flipped = cv2.rotate(imageCache, cv2.ROTATE_180)
        input_batch = self._transform(flipped).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=flipped.shape[:2],mode="bicubic",align_corners=True,).squeeze()
        analysis2  = prediction.cpu().numpy()
        analysis2 = cv2.rotate(analysis2, cv2.ROTATE_180)
        self.dataCache = cv2.addWeighted(analysis1, 0.5, analysis2, 0.5, 0.0)
        
        rightImage = self.postProcess(guiSize)
        return rightImage

    def _process_4way(self, imageCache, guiSize):       
        #generate masks
        if self.redMatte == True:
            targetSize = int(imageCache.shape[0])
            mask = zeros((targetSize,targetSize), dtype="uint8")
            mask = cv2.circle(mask, (int(targetSize/2), int(targetSize/2)), int(targetSize/2 - 1), 255, cv2.FILLED)
            red = zeros((targetSize, targetSize, 3), dtype="uint8")
            red[:] = (148, 68, 70)
            red = cv2.circle(red, (int(targetSize/2), int(targetSize/2)), int(targetSize/2 - 1), 0, cv2.FILLED)        

            #composite masks
            imageCache = cv2.bitwise_and(imageCache, imageCache, mask=mask)
            imageCache = cv2.add( imageCache, red)

        #blurring it filters out the pixel screen
        imageCache = cv2.medianBlur(imageCache, 3)
        
        #the midas function 1
        input_batch = self._transform(imageCache).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=imageCache.shape[:2],mode="bicubic",align_corners=True,).squeeze()
        self.dataCache = prediction.cpu().numpy()

        #the midas function 2
        flipped = cv2.rotate(imageCache, cv2.ROTATE_90_CLOCKWISE)
        input_batch = self._transform(flipped).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=flipped.shape[:2],mode="bicubic",align_corners=True,).squeeze()
        newAnalysis  = prediction.cpu().numpy()
        newAnalysis = cv2.rotate(newAnalysis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.dataCache = cv2.add(self.dataCache, newAnalysis)

        #the midas function 3
        flipped = cv2.rotate(imageCache, cv2.ROTATE_180)
        input_batch = self._transform(flipped).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=flipped.shape[:2],mode="bicubic",align_corners=True,).squeeze()
        newAnalysis  = prediction.cpu().numpy()
        newAnalysis = cv2.rotate(newAnalysis, cv2.ROTATE_180)
        self.dataCache = cv2.add(self.dataCache, newAnalysis)

        #the midas function 4
        flipped = cv2.rotate(imageCache, cv2.ROTATE_90_COUNTERCLOCKWISE)
        input_batch = self._transform(flipped).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=flipped.shape[:2],mode="bicubic",align_corners=True,).squeeze()
        newAnalysis  = prediction.cpu().numpy()
        newAnalysis = cv2.rotate(newAnalysis, cv2.ROTATE_90_CLOCKWISE)
        self.dataCache = cv2.add(self.dataCache, newAnalysis)
        
        rightImage = self.postProcess(guiSize)
        return rightImage

    def _process_accumulate(self, imageCache, guiSize):       
        #generate masks
        if self.redMatte == True:
            targetSize = int(imageCache.shape[0])
            mask = zeros((targetSize,targetSize), dtype="uint8")
            mask = cv2.circle(mask, (int(targetSize/2), int(targetSize/2)), int(targetSize/2 - 1), 255, cv2.FILLED)
            red = zeros((targetSize, targetSize, 3), dtype="uint8")
            red[:] = (148, 68, 70)
            red = cv2.circle(red, (int(targetSize/2), int(targetSize/2)), int(targetSize/2 - 1), 0, cv2.FILLED)        

            #composite masks
            imageCache = cv2.bitwise_and(imageCache, imageCache, mask=mask)
            imageCache = cv2.add( imageCache, red)

        #blurring it filters out the pixel screen
        imageCache = cv2.medianBlur(imageCache, 3)
        
        #the midas function 
        input_batch = self._transform(imageCache).to(self._device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=imageCache.shape[:2],mode="bicubic",align_corners=True,).squeeze()
        newAnalysis  = prediction.cpu().numpy()

        if newAnalysis.shape != self.dataCache.shape:
            #this triggers reset on first run, and zoom changes!
            self.dataCache = newAnalysis
        else:
            #geometric decay of the previous frame
            self.dataCache = cv2.addWeighted(newAnalysis, 1.0, self.dataCache, 0.66666, 0)
        
        rightImage = self.postProcess(guiSize)
        return rightImage

    def postProcess(self, guiSize):
        self.minmax = (round(self.dataCache.min(),3), round(self.dataCache.max(),3))
        me.currentEntry.minmax = self.minmax    #report this calc to the log
        if self.depthmapEnabled:
            #depth maps are already normalized
            post = self.dataCache.astype('uint8')
        else:
            post = interp(self.dataCache, (self.minmax[0], self.minmax[1]), (0,254)).astype('uint8')

        #color it, resize it https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
        post = cv2.applyColorMap(post, cv2.COLORMAP_INFERNO)
        post = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)
        post = me.addOverlay(post)

        post = Image.fromarray(post).resize((guiSize, guiSize))
        finalImage = wxb.FromBuffer(guiSize, guiSize, post.tobytes())
        return finalImage
  
    def clearAccumulator(self):
        self.dataCache = zeros(1)

    def exportImage(self, filename):
        if str.lower(filename[-3:]) == "npy":
            save("exports//" + filename, self.dataCache)
        else:
            post = interp(self.dataCache, (self.dataCache.min(), self.dataCache.max()), (0,254)).astype('uint8')
            post = cv2.applyColorMap(post, cv2.COLORMAP_INFERNO)
            if str.lower(filename[-3:]) == "png":
                cv2.imwrite("exports//" + filename, post, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
            elif str.lower(filename[-3:]) == "jpg":
                cv2.imwrite("exports//" + filename, post, [int(cv2.IMWRITE_JPEG_QUALITY), 95])