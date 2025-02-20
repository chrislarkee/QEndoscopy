import cv2 
from wx import Bitmap as wxb
from PIL import Image
from os import getcwd
from os.path import isfile
import numpy as np
  
#ML
import torch
import torch.nn as nn

import modules.measurement as QEMeasurement
import modules.depthmap as dm

class DepthUtilities:
    class conv_block(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()

            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_c)

            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_c)

            self.relu = nn.ReLU()

        def forward(self, inputs):
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            return x

    class encoder_block(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()

            self.conv = DepthUtilities.conv_block(in_c, out_c)
            self.pool = nn.MaxPool2d((2, 2))

        def forward(self, inputs):
            x = self.conv(inputs)
            p = self.pool(x)

            return x, p

    class decoder_block(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()

            self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
            self.conv = DepthUtilities.conv_block(out_c+out_c, out_c)

        def forward(self, inputs, skip):
            x = self.up(inputs)
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)

            return x

    class build_unet(nn.Module):
        def __init__(self):
            super().__init__()

            """ Encoder """
            self.e1 = DepthUtilities.encoder_block(1, 64)
            self.e2 = DepthUtilities.encoder_block(64, 128)
            self.e3 = DepthUtilities.encoder_block(128, 256)
            self.e4 = DepthUtilities.encoder_block(256, 512)

            """ Bottleneck """
            self.b = DepthUtilities.conv_block(512, 1024)

            """ Decoder """
            self.d1 = DepthUtilities.decoder_block(1024, 512)
            self.d2 = DepthUtilities.decoder_block(512, 256)
            self.d3 = DepthUtilities.decoder_block(256, 128)
            self.d4 = DepthUtilities.decoder_block(128, 64)

            """ Classifier """
            self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        def forward(self, inputs):
            """ Encoder """
            s1, p1 = self.e1(inputs)
            s2, p2 = self.e2(p1)
            s3, p3 = self.e3(p2)
            s4, p4 = self.e4(p3)
    
            """ Bottleneck """
            b = self.b(p4)
            """ Decoder """
            d1 = self.d1(b, s4)
            d2 = self.d2(d1, s3)
            d3 = self.d3(d2, s2)
            d4 = self.d4(d3, s1)

            """ Classifier """
            outputs = self.outputs(d4)

            return outputs

class Depth:
    def __init__(self, depthmap, vid):
        if depthmap != None:
            self.bakedDepthmap = True
            self._dm = dm.DepthmapSetup(depthmap, vid)
        else:
            self.bakedDepthmap = False
            self.vid = None
            
            #initialize Depthmap NN
            self.model = DepthUtilities.build_unet();
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if isfile("includes\\model.pth"):
                self.model.load_state_dict(torch.load("includes\\model.pth", map_location=self.device, weights_only=True))
            else:
                self.model.load_state_dict(torch.load("model.pth", map_location=self.device, weights_only=True))
            self.model.eval()
            self.model.to(self.device)
            

        #parameters to cache
        self._maskData = np.empty(0)
        self.dataCache = np.zeros(1)             #a slot for the data cache
        self.minmax = (0,1)
        self.colormap = True

    def processDepth(self, imageCache, guiSize):  
        if self.bakedDepthmap:
            self.dataCache = self._dm.lookupDepth()
            return self.postProcess(guiSize)
        else:
            nnInput = cv2.cvtColor(imageCache, cv2.COLOR_BGR2GRAY)
            if nnInput.shape[0] != 720:
                nnInput = cv2.resize(nnInput,(720,720))
            nnInput = torch.from_numpy(nnInput[np.newaxis, ...]).float()/255 - 0.5
            nnInput = nnInput.unsqueeze(1).to(self.device)
            self.dataCache = self.model(nnInput)
            self.dataCache = self.dataCache.squeeze().detach().cpu().numpy()

            if self._maskData.size == 0:
                #draw the circle if it hasn't been drawn yet
                self._maskData = np.full((720, 720), 0.0, dtype=np.float16)
                cv2.circle(self._maskData, (360, 360), 352, 1.0, -1, lineType=cv2.LINE_AA)

            #modify the data(!) to flatten the values outside a circle
            self.dataCache = self.dataCache * self._maskData

            rightImage = self.postProcess(guiSize)
            return rightImage

    def postProcess(self, guiSize):
        self.minmax = (0.0, self.dataCache.max())
        QEMeasurement.currentEntry.maxDistance = self.minmax[1]    #report this calc to the log
        if self.bakedDepthmap:
            #depth maps are already normalized
            post = self.dataCache.astype('uint8')
        else:
            if self.colormap:
                post = np.interp(self.dataCache, (self.minmax[0], self.minmax[1]), (255,0)).astype('uint8')
            else:
                post = np.interp(self.dataCache, (self.minmax[0], self.minmax[1] * 0.99), (0,255)).astype('uint8')

        #color it, resize it https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
        if self.colormap:
            post = cv2.applyColorMap(post, cv2.COLORMAP_INFERNO)
            post = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)            
            #black circle matte
            circleMatte = (self._maskData * 255.0).astype('uint8')
            post = cv2.bitwise_and(post, post, mask=circleMatte)
        else:            
            post = cv2.cvtColor(post, cv2.COLOR_GRAY2RGB)
            np.clip(post, 0, 254, post)

        post = QEMeasurement.addOverlay(post)

        post = Image.fromarray(post).resize((guiSize, guiSize))
        finalImage = wxb.FromBuffer(guiSize, guiSize, post.tobytes())
        return finalImage

    def exportImage(self, filename):
        if str.lower(filename[-3:]) == "npy":
            np.save("exports//" + filename, self.dataCache)
        else:
            post = np.interp(self.dataCache, (0.0, self.dataCache.max()), (0,255)).astype('uint8')
            if self.colormap:
                post = cv2.applyColorMap(post, cv2.COLORMAP_INFERNO)
                #black circle matte
                circleMatte = (self._maskData * 255.0).astype('uint8')
                post = cv2.bitwise_and(post, post, mask=circleMatte)
            if str.lower(filename[-3:]) == "png":
                cv2.imwrite("exports//" + filename, post, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
            elif str.lower(filename[-3:]) == "jpg":
                cv2.imwrite("exports//" + filename, post, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

