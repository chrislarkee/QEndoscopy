#from numpy import interp, zeros, clip
import numpy as np
from datetime import datetime
import xlsxwriter
from os.path import isfile
from math import floor

#logging cache
currentEntry = None         #a LogEntry that is currently collecting data
_database = {}              #a dict of LogEntrys
_entryCounter = -1

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

def startup(filename):   
    global _database, currentEntry, _entryCounter
    #initialize values, as well as reset the pre-existing ones
    _database = {}
    _entryCounter = -1
    currentEntry = LogEntry()

    #can we load a CSV containing camera positions?
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