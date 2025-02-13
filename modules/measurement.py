import cv2
#from numpy import interp, zeros, clip
import numpy as np
from math import floor, pow, log
from datetime import datetime
import xlsxwriter
from os.path import isfile

#logging cache
currentEntry = None         #a LogEntry that is currently collecting data
_database = {}              #a dict of LogEntrys
_entryCounter = -1

#image info
overlayEnabled = False
_overlayImage = None
framerate = 29.97
_previouspoint = None

csv = None

#to log, fill in the details on currentEntry as you go, then store() it.
class LogEntry:
    def __init__(self):
        global _entryCounter
        _entryCounter += 1
        self.counter = _entryCounter
        self._frame = -1
        self.timecode = "n/a"
        self.maxDistance = 0      
        self.distance = 0
        self.areaPX = 0 
        self.areaMM = 0
        self.cameraPos = "0,0,0"

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, newframe):
        global framerate, csv
        self._frame = int(newframe)
        minutes = str(floor((newframe / framerate) / 60)).zfill(2)
        seconds = str(round((newframe / framerate) % 60, 2)).zfill(5)
        self.timecode = minutes + ":" + seconds
        if csv != None:
            self.cameraPos = csv[int(newframe)]

def counter():
    global currentEntry
    return currentEntry.counter

def store():
    global _database, currentEntry
    _database[currentEntry.counter] = currentEntry  
    currentEntry = LogEntry()

class Lensmath:
    #constant lens parameters (from Blender)
    IMG_SIZE     = 720.0    #px
    FOV          = np.radians(125.0)
    SENSOR_WIDTH = 3.6      #mm. square sensor.
    FOCAL_LENGTH = 2.5      #mm
    
    @classmethod
    def convertPoint(self, pixel_coords, depth):    
        # Compute the effective focal length in pixels
        f_px = self.FOCAL_LENGTH * (self.IMG_SIZE / self.SENSOR_WIDTH)

        # Compute the maximum field angle θ_max from the equisolid model
        theta_max = self.FOV / 2

        # Convert pixel coordinates to normalized image coordinates
        x = (pixel_coords[0] - (self.IMG_SIZE / 2))
        y = (pixel_coords[1] - (self.IMG_SIZE / 2))

        # Compute radius in pixels
        r = np.sqrt(x**2 + y**2)

        # Convert radius to field angle θ using the equisolid model
        theta = 2 * np.arcsin(r / (2 * f_px))
        theta = np.clip(theta, 0, theta_max)

        # Compute azimuth angle φ
        phi = np.arctan2(y, x)

        # Convert spherical coordinates to Cartesian 3D coordinates
        X = depth * np.sin(theta) * np.cos(phi)
        Y = depth * np.sin(theta) * np.sin(phi)
        Z = depth * np.cos(theta)

        return (X, Y, Z)
    
    @classmethod
    def convertArea(self, area, depth):
        # Compute the effective focal length in pixels
        f_px = self.FOCAL_LENGTH * (self.IMG_SIZE / self.SENSOR_WIDTH)

        # Approximate the field angle per pixel using equisolid projection
        theta_per_pixel = (self.FOV / 2) / (f_px / 2)  # Approximate small-angle field resolution

        # Convert pixel area to angular area
        angular_area = area * (theta_per_pixel ** 2)

        # Convert angular area to real-world area (approximation)
        real_area = (depth ** 2) * angular_area

        return real_area

def startup(filename):
    global _database, currentEntry, _entryCounter
    #initialize values, as well as reset the pre-existing ones
    _database = {}
    _entryCounter = -1
    currentEntry = LogEntry()

    #can we load a CSV?
    global csv
    csvFile = filename[:-4] + ".csv"    
    if isfile(csvFile):
        f = open(csvFile, "r")
        data = f.read().split("\n")
        f.close()
        
        csv = []
        for line in range(1, len(data) - 2):
            coords = data[line].split(',')[1:]
            csv.append(coords)
    else:
        csv = None

def selectCenterContour(contours):
    if len(contours) == 0:
        return -1
    elif len(contours) == 1:
        return 0
    else:
        biggest = -1
        selectedContour = -1
        for i in range(0, len(contours)):
            if contours[i].size > biggest:
                biggest = contours[i].size
                selectedContour = i
        return selectedContour

#this input is the b&w depth map
def updateOverlayCross(depthmap, threshold):
    global _overlayImage, currentEntry
    #calculate the threshold    
    ret, thresholdImg = cv2.threshold(depthmap, threshold, depthmap.max(), cv2.THRESH_BINARY_INV)
    if len(thresholdImg.shape) == 3:
        thresholdImg = cv2.split(thresholdImg)[1]
    #thresholdImg = interp(thresholdImg, (depthmap.min(), depthmap.max()), (0,255)).astype('uint8')
    thresholdImg = thresholdImg.astype('uint8')
    contours,hierarchy = cv2.findContours(thresholdImg, 1, 2)

    #which contour is the biggest?
    idealContour = selectCenterContour(contours) 

    #prepare the overlay Image
    _overlayImage = np.zeros((depthmap.shape[0],depthmap.shape[1],3), dtype="uint8")
    _overlayImage = cv2.drawContours(_overlayImage, contours, idealContour, (0, 215, 60), 2, cv2.LINE_AA)

    #recordData
    currentEntry.distance = round(threshold,3)

    area = cv2.contourArea(contours[idealContour])
    currentEntry.areaPX = round(area,1)    
    currentEntry.areaMM = round(Lensmath.convertArea(area, threshold), 4)
    
        
def updateOverlayLine(depthMap, mouseCoord):
    pass

def addOverlay(image):
    global _overlayImage
    if overlayEnabled == False:
        return image
    newImage = cv2.add(image, _overlayImage)
    newImage = np.clip(newImage, 0, 255)
    #cv2.imshow('image', newImage)    
    return newImage

##DISPLAY MEASUREMENTS##
def generatePreview():
    global _database

    #sort by frame
    myKeys = list(_database.keys())
    myKeys.sort()
    _database = {i: _database[i] for i in myKeys}

    previewTable = []
    for line in _database.values():
        previewLine = [line.counter,
            line.frame,
            line.timecode,
            line.maxDistance,
            line.distance,
            line.areaPX,
            line.areaMM,
            line.cameraPos]
        previewTable.append(previewLine)
    return previewTable

def writeLog(videoname, filename):
    #this is triggered when the export save button is pushed.

    #start up the sheet writer
    workbook = xlsxwriter.Workbook(filename, {'in_memory': True})
    worksheet = workbook.add_worksheet()
    cell_format = workbook.add_format()
    cell_format.set_bold()

    global header, database
    #write the metadata headers
    worksheet.write_row(0, 0, ("Measurement log for ", videoname), cell_format)
    currentTime = datetime.now().strftime("%Y-%m-%d %H:%M")
    worksheet.write_row(1, 0, ("Log generated on ", currentTime) ,cell_format)
    worksheet.write_row(3, 0, (
        "Counter", "Frame", "Timecode", "MaxDistance", "Distance", "Area (px)", "Area (mm^2)", "CameraPos"), cell_format)

    #iterate through the entries    
    rowOffset = 4
    data = generatePreview()
    for row in range(0, len(data)):        
        for col in range(0,9):
            cellContents = str(data[row][col])
            worksheet.write(row + rowOffset, col, cellContents)

    #write the file
    workbook.close()
    return True
