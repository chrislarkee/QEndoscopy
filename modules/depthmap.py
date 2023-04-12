import cv2 
#from wx import Bitmap as wxb
#from PIL import Image

class DepthmapSetup:
    def __init__(self, filename: str, vid):        
        #load the video
        self._path = str(filename)
        self.vidDepth = cv2.VideoCapture(self._path)
        self.vid = vid

    def lookupDepth(self):
        #seek and retrieve
        self.vidDepth.set(cv2.CAP_PROP_POS_FRAMES, self.vid.currentFrame - 1)
        ret, rawframe = self.vidDepth.read()

        post = cv2.split(rawframe)[1]   #convert to 1 channel; take just the green
        crop = self.vid._crop           #get crop settings from the main vid object
        post = post[crop[0]:crop[1], crop[2]:crop[3]]

        #convert from uint8 to float32
        #post = np.invert(post)
        #post = np.log(post / 255.0) / np.log(0.25)
        #post = np.clip(post, 0.0, 5.0)
        
        return post
