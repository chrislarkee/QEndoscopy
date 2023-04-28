import cv2
from numpy import interp, zeros, clip
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

csv = None

#to log, fill in the details on currentEntry as you go, then store() it.
class LogEntry:
    def __init__(self):
        global _entryCounter
        _entryCounter += 1
        self.counter = _entryCounter
        self._frame = -1
        self.timecode = "n/a"
        self.minmax = 0
        self.threshold = 0        
        self.distance = 0
        self.areaPX = 0 
        self.areaMM = 0
        self.cameraPos = "?"        

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


#this input is the b&w depth map
def updateOverlay(depthmap, newThreshold):
    global _overlayImage, currentEntry
    #calculate the threshold    
    ret, thresholded = cv2.threshold(depthmap, newThreshold, depthmap.max(), cv2.THRESH_BINARY_INV)
    if len(thresholded.shape) == 3:
        thresholded = cv2.split(thresholded)[1]
    #thresholded = interp(thresholded, (depthmap.min(), depthmap.max()), (0,255)).astype('uint8')
    thresholded = thresholded.astype('uint8')
    contours,hierarchy = cv2.findContours(thresholded, 1, 2)
    #print("break")

    #which contour is the biggest?
    idealContour = selectCenterContour(contours) 

    #prepare the overlay Image
    #blank image
    _overlayImage = zeros((depthmap.shape[0],depthmap.shape[1],3), dtype="uint8")
    _overlayImage = cv2.drawContours(_overlayImage, contours, idealContour, (0, 215, 60), 2, cv2.LINE_AA)

    #recordData
    currentEntry.threshold = newThreshold    
    currentEntry.distance = round(log(newThreshold / 255.0) / log(0.25),5)
    currentEntry.areaPX = round(cv2.contourArea(contours[idealContour]),2)
    
    #k0, k1, k2, k3 = -9.5657e-7, 6.5697e-6, 1.8917e-6, -1.8411e-7
    k0, k1, k2, k3 = -1.05e-6, 6.74e-6, 1.98e-6, -1.86e-7
    k = k0 + (k1 * currentEntry.distance) + (k2 * pow(currentEntry.distance,2)) + (k3 * pow(currentEntry.distance,3))
    currentEntry.areaMM = round(k * currentEntry.areaPX, 5)
    #a,b = -1.8543e-7, 4.2585e-5
    #currentEntry.areaMM = round((a * newThreshold + b) * currentEntry.areaPX,4)    
        

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



def addOverlay(image):
    global _overlayImage
    if overlayEnabled == False:
        return image
    newImage = cv2.add(image, _overlayImage)
    newImage = clip(newImage, 0, 255)
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
            line.minmax,
            line.threshold,
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
        "Counter", "Frame", "Timecode", "MinMax", "Threshold", "Distance", "Area (px)", "Area (mm^2)", "CameraPos"), cell_format)

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
