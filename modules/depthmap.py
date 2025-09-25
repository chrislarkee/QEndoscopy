import numpy as np
#from cv2 import convexHull, contourArea, drawContours

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
    pointsOnPlane   = None
    slice_mask      = np.zeros((720,720), dtype=np.bool) #used for the overlay
    
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
        plane_normal = np.linalg.inv(slicerMatrix.matrix[:3, :3]) @ np.array([0, 0, 1])
        plane_normal /= np.linalg.norm(plane_normal)
        plane_point = slicerMatrix.matrix[3, :3]

        # Calculate the signed distance from each point to the plane        
        distances = np.dot(self.pointCloudCache - plane_point, plane_normal)
        self.slice_mask = np.abs(distances) < 0.004        #define threshold here     
        self.pointsOnPlane = self.pointCloudCache[self.slice_mask]

        if len(self.pointsOnPlane) <= 4:
            self.area = 0
            self.slice_mask = np.zeros((720,720), dtype=np.bool)
            return

        #gui's method
        P1 = np.mean(self.pointsOnPlane, axis=0)

        # Compute distance from point P1 to all other points
        vectors = self.pointsOnPlane - P1
        distances = np.linalg.norm(vectors, axis=1)

        min_index = np.argmin(distances)
        max_index = np.argmax(distances)

        # Point P2 = point nearest to point P1
        P2 = self.pointsOnPlane[min_index]

        # Point P3 = point farthest from point P1
        P3 = self.pointsOnPlane[max_index]

        # Compute normal vector
        V1 = P2 - P1
        # U1 = cross product of N and V1
        U1 = np.cross(plane_normal, V1)
        U1 = U1 / np.linalg.norm(U1)

        # U2 = cross product of N and U1
        U2 = np.cross(plane_normal, U1)
        U2 = U2 / np.linalg.norm(U2)

        # Compute coordinates of points on plane defined by U1 and U2
        # Using vectorization for efficiency instead of a loop
        V_vectors = self.pointsOnPlane - P1
        X_point_on_plane = np.dot(V_vectors, U1)
        Y_point_on_plane = np.dot(V_vectors, U2)


        # Translate origin to the center of the cross-section
        X_max = np.max(X_point_on_plane)
        Y_max = np.max(Y_point_on_plane)
        X_min = np.min(X_point_on_plane)
        Y_min = np.min(Y_point_on_plane)
        X_new_origin = X_min + 0.5 * (X_max - X_min)
        Y_new_origin = Y_min + 0.5 * (Y_max - Y_min)
        X_point_on_plane = X_point_on_plane - X_new_origin
        Y_point_on_plane = Y_point_on_plane - Y_new_origin
        Points_on_plane = np.vstack((X_point_on_plane, Y_point_on_plane)).T


        # *************************************************************************#
        #                       SMOOTH THE PERIMETER CURVE                         #
        # *************************************************************************#
        # This section re-orders and down-samples the 2D points to create a
        # smoother, more evenly spaced perimeter.

        radius = 0.05  # Target spacing between points in smooth curve

        # Make a copy of the points to modify
        remaining_points = Points_on_plane.copy()
        smooth_perimeter_list = []

        # Start with the first point in the array
        current_point = remaining_points[0, :]
        smooth_perimeter_list.append(current_point)
        remaining_points = np.delete(remaining_points, 0, axis=0)

        while remaining_points.shape[0] > 0:
            # Find the point in the remaining list closest to the current_point
            vectors = remaining_points - current_point
            distances = np.linalg.norm(vectors, axis=1)
            
            # Get the index of the closest point
            closest_index = np.argmin(distances)
            
            # Update the current point to this new closest point
            current_point = remaining_points[closest_index, :]
            
            # Add this point to the smooth curve
            smooth_perimeter_list.append(current_point)
            
            # Remove all points from the remaining list that are too close to the *new* current_point
            vectors_from_new_current = remaining_points - current_point
            distances_from_new_current = np.linalg.norm(vectors_from_new_current, axis=1)
            
            # Keep only the points that are far enough away
            points_to_keep = distances_from_new_current >= radius
            remaining_points = remaining_points[points_to_keep, :]

        Points_smooth_perimeter = np.array(smooth_perimeter_list)

        # *************************************************************************#
        #                            AREA CALCULATION                             #
        # *************************************************************************#
        # This section computes the cross-sectional area using the shoelace formula.
        # The points must be sorted sequentially around the perimeter for this to work,
        # which was accomplished in the "SMOOTH THE PERIMETER CURVE" step.

        N_points_smooth_curve = Points_smooth_perimeter.shape[0]
        X_smooth_curve = Points_smooth_perimeter[:, 0]
        Y_smooth_curve = Points_smooth_perimeter[:, 1]

        # Compute Delta X
        # This is the difference between each x-coordinate and the next one in the sequence
        Delta_X = np.roll(X_smooth_curve, -1) - X_smooth_curve

        # Compute area using the shoelace formula variation
        self.area = np.abs(np.sum(Y_smooth_curve * Delta_X))

    @classmethod
    def savePointCloud(self, filename):                
        if filename[-4:] == ".npy":
            np.save(filename, self.pointCloudCache)
        elif filename[-4:] == ".xyz":
            xyzData = np.reshape(self.pointCloudCache, (-1,3))
            np.savetxt(filename, xyzData, fmt='%.6f', delimiter=' ')

    @classmethod    
    def convertDistance(self, depth):
        return depth * self.DEPTH_CALIB

