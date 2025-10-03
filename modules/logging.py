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
framerate = 29.97

#to log, fill in the details on currentEntry as you go, then store() it.
class LogEntry:
    def __init__(self):
        #"i" "Frame No." "Timecode" "Pointcloud Range" "Plane Distance" "Plane Rotation" "Matched Points" "Cross Section Area" "" ""
        global _entryCounter
        defaultstring = "?"
        _entryCounter += 1
        self.counter = _entryCounter
        self._frame = -1
        self.timecode = defaultstring
        self.minmax = defaultstring   
        self.distance = defaultstring
        self.rotation = (0,0) 
        self.points = defaultstring
        self.CSarea = defaultstring

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, newframe):
        global framerate
        self._frame = int(newframe)
        minutes = str(floor((newframe / framerate) / 60)).zfill(2)
        seconds = str(round((newframe / framerate) % 60, 2)).zfill(5)
        self.timecode = minutes + ":" + seconds
        # if csv != None:
        #     self.cameraPos = csv[int(newframe)]

def counter():
    global currentEntry
    return currentEntry.counter

def store():
    global _database, currentEntry
    _database[currentEntry.counter] = currentEntry  
    currentEntry = LogEntry()

def startup():   
    global _database, currentEntry, _entryCounter
    #initialize values, as well as reset the pre-existing ones
    _database = {}
    _entryCounter = -1
    currentEntry = LogEntry()

def clearLast():
    global _database
    if (len(_database) > 0):
        removed_item = _database.popitem()

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
            line.distance,
            line.rotation,
            line.points,
            line.CSarea]
        previewTable.append(previewLine)
    return previewTable


#this is triggered when the export save button is pushed.
def writeLog(videoname, filename):
    #shared vars
    header = (
        f"Measurement log for {videoname}",
        f"Log generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}",
        "i, Frame No., Timecode, Pointcloud Range, Plane Distance, Plane Rotation, Matched Points, Cross Section Area",
        "\n")
    data = generatePreview()

    #write csv file
    if filename[-4:] == ".csv":
        with open(filename, 'w+') as f:
            for v in header:
                f.write(v)
            for v in data:            
                f.write(str(v))
            f.write("\n")
        return
    
    #write Excel doc using XLSX Writer
    if filename[-4:] == "xlsx":
        workbook = xlsxwriter.Workbook(filename, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        cell_format = workbook.add_format()
        cell_format.set_bold()
        
        #write the metadata headers
        worksheet.write_row(0, 0, (header[0], cell_format))
        worksheet.write_row(1, 0, (header[1] ,cell_format))
        worksheet.write_row(3, 0, (header[2].split(','), cell_format))

        #iterate through the entries    
        rowOffset = 4
        data = generatePreview()
        for row in range(0, len(data)):        
            for col in range(0,9):
                cellContents = str(data[row][col])
                worksheet.write(row + rowOffset, col, cellContents)

        #write the file
        workbook.close()
        return
