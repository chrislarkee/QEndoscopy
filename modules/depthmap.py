import numpy as np
from cv2 import convexHull, contourArea, drawContours

#ML
import torch
import torch.nn as nn

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

class PointCloud:
    #constant lens parameters (from Blender)
    IMG_SIZE     = 720.0    #px
    FOV          = np.radians(125.0)
    SENSOR_WIDTH = 3.6      #mm. square sensor.
    FOCAL_LENGTH = 2.5      #mm
    DEPTH_CALIB = 0.918     #adjustment factor for depth (from testing)
    pointCloudCache = None
    pointsOnPlane = None
    
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
        trueDepth = depth * self.DEPTH_CALIB
        X = trueDepth * np.sin(theta) * np.cos(phi)
        Y = trueDepth * np.sin(theta) * np.sin(phi)
        Z = trueDepth * np.cos(theta)

        return (X, Y, Z)
    
    @classmethod    
    def calculateArea(self, contour, depth):
        newContour = contour.copy().astype('float32')
        for p in range(0, newContour.shape[0]):
            coord = (contour[p,0,0].item(), contour[p,0,1].item())
            newPoint = PointCloud.convertPoint(coord, depth)[:2]
            newContour[p,0,0] = newPoint[0]
            newContour[p,0,1] = newPoint[1]

        return contourArea(newContour)

    @classmethod
    def generatePointCloud(self, depthmap):
        # Compute the effective focal length in pixels
        f_px = self.FOCAL_LENGTH * (depthmap.shape[0] / self.SENSOR_WIDTH)

        # Compute the maximum field angle θ_max from the equisolid model
        theta_max = self.FOV / 2

        # Convert pixel coordinates to normalized image coordinates
        xy = np.mgrid[self.IMG_SIZE / -2:self.IMG_SIZE /2, self.IMG_SIZE / -2:self.IMG_SIZE / 2].transpose(1,2,0)
        
        # Compute radius in pixels
        r = np.sqrt(xy[:,:,0]**2 + xy[:,:,1]**2)
        
        # Convert radius to field angle θ using the equisolid model       
        theta = np.clip(2 * np.arcsin(r / (2 * f_px)), 0, theta_max)
        
        # Compute azimuth angle φ
        phi = np.arctan2(xy[:,:,1], xy[:,:,0])

        # Convert spherical coordinates to Cartesian 3D coordinates
        out = np.zeros((depthmap.shape[0],depthmap.shape[1],3), dtype=np.double)   
        #xy[:,:,2] = xy[:,:,2] * self.DEPTH_CALIB
        depthmap = depthmap * self.DEPTH_CALIB
        out[:,:,0] =  depthmap * np.sin(theta) * np.cos(phi)
        out[:,:,1] =  depthmap * np.sin(theta) * np.sin(phi) * -1
        out[:,:,2] =  depthmap * np.cos(theta)
        
        self.pointCloudCache = out
    
    @classmethod
    def measureIntersection(self, slicerMatrix):        
        #compute prerequisites for the intersecting points formulas
        rotation_matrix = np.linalg.inv(slicerMatrix.matrix[:3, :3])
        plane_normal = rotation_matrix @ np.array([0, 0, 1])
        plane_normal /= np.linalg.norm(plane_normal)
        plane_point = slicerMatrix.matrix[3, :3]

        # Calculate the signed distance from each point to the plane        
        distances = np.dot(self.pointCloudCache - plane_point, plane_normal)
        slice_mask = np.abs(distances) < 0.002        #define threshold
        self.pointsOnPlane = self.pointCloudCache[slice_mask]

        if len(self.pointsOnPlane) <= 4:
            #self.t_output.SetValue("No points found.")
            self.area = None
            return
            
        a = np.array([0, 0, 1]) # Use Z-axis            
        u = np.cross(plane_normal, a)
        u /= np.linalg.norm(u)
        v = np.cross(plane_normal, u) # v is already normalized

        #Project the 3D points onto the new 2D basis (u, v)
        points_2d = np.dot(self.pointsOnPlane, np.array([u, v]).T)

        #repackage points for CV2 contour
        contour_points = points_2d.astype(np.float32).reshape(-1, 1, 2)

        ##which method is better?? They never match
        self.area = contourArea(contour_points)
        #hull = convexHull(contour_points)
        #areaCH = contourArea(hull)        


    @classmethod
    def savePointCloud(self, filename):
        if self.pointCloudCache == None:
            return
        
        if filename[-4:] == ".npy":
            np.save(filename, self.pointCloudCache)
        elif filename[-4:] == ".xyz":
            xyzData = np.reshape(self.pointCloudCache, (-1,3))
            np.savetxt(filename, xyzData, fmt='%.6f', delimiter=' ')

    @classmethod    
    def convertDistance(self, depth):
        return depth * self.DEPTH_CALIB

