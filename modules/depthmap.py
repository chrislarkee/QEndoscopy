import numpy as np

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
    #DEPTH_CALIB = #0.918     #adjustment factor for depth (from testing)
    pointCloudCache = None
    pointsOnPlane   = None
    _slice_mask      = np.zeros((720,720), dtype=np.bool) #used for the overlay
    
    @classmethod
    def convertPoint(self, pixel_coords, depth):    
        # Compute the effective focal length in pixels
        f_px = self.FOCAL_LENGTH * (self.IMG_SIZE / self.SENSOR_WIDTH)

        # Compute the maximum field angle θ_max from the equisolid model
        theta_max = self.FOV / 2

        # Convert pixel coordinates to normalized image coordinates
        #x = (pixel_coo - (self.IMG_SIZE / 2))
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
        #depth = depth * self.DEPTH_CALIB
        X = depth * np.sin(theta) * np.cos(phi)
        Y = depth * np.sin(theta) * np.sin(phi)
        Z = depth * np.cos(theta)

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
        #depthmap = depthmap * self.DEPTH_CALIB
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
        planeDistances = np.dot(self.pointCloudCache - plane_point, plane_normal)
        self._slice_mask = np.abs(planeDistances) < 0.006        #define threshold here     
        self.pointsOnPlane = self.pointCloudCache[self._slice_mask]

        if len(self.pointsOnPlane) <= 4:
            self.area = 0
            self._slice_mask = np.zeros((720,720), dtype=np.bool)
            return

        #gui's method
        N_points = self.pointsOnPlane.shape[0]  # Number of points on curve perimeter
        # *************************************************************************
        #                   FIND VECTOR NORMAL TO CROSS-SECTION
        # *************************************************************************

        # Point P1 = centroid of perimeter points
        P1 = np.mean(self.pointsOnPlane, axis=0)

        # Compute distance from point P1 to all other points (vectorized)
        distance = np.linalg.norm(self.pointsOnPlane - P1, axis=1)
        
        min_index = np.argmin(distance)
        max_index = np.argmax(distance)

        # Point P2 = point nearest to point P1
        P2 = self.pointsOnPlane[min_index, :]

        # Point P3 = point farthest from point P1
        P3 = self.pointsOnPlane[max_index, :]

        # Compute normal vector
        V1 = P2 - P1
        V2 = P3 - P1
        N = np.cross(V1, V2)
        N = N / np.linalg.norm(N)

        # *************************************************************************
        #               PROJECT POINTS FROM 3D SPACE ONTO 2D PLANE
        # *************************************************************************
        
        # U1 = cross product of N and V1
        U1 = np.cross(N, V1)
        U1 = U1 / np.linalg.norm(U1)

        # U2 = cross product of N and U1
        U2 = np.cross(N, U1)
        U2 = U2 / np.linalg.norm(U2)

        # Compute coordinates of points on plane defined by U1 and U2 (vectorized)
        V = self.pointsOnPlane - P1
        X_point_on_plane = np.dot(V, U1)
        Y_point_on_plane = np.dot(V, U2)

        # Translate origin to the center of the cross-section
        X_new_origin = np.min(X_point_on_plane) + 0.5 * (np.max(X_point_on_plane) - np.min(X_point_on_plane))
        Y_new_origin = np.min(Y_point_on_plane) + 0.5 * (np.max(Y_point_on_plane) - np.min(Y_point_on_plane))
        X_point_on_plane = X_point_on_plane - X_new_origin
        Y_point_on_plane = Y_point_on_plane - Y_new_origin
        
        Points_on_plane = np.column_stack((X_point_on_plane, Y_point_on_plane))

        # *************************************************************************
        #                       SMOOTH THE PERIMETER CURVE
        # *************************************************************************

        radius = 0.02  # Target spacing between points in smooth curve
        Points_smooth_perimeter = []
        
        # Create a boolean mask for points that haven't been removed yet
        available_points_mask = np.ones(N_points, dtype=bool)
        current_index = 0 # Start with the first point

        while np.any(available_points_mask):
            # Choose the next available point
            # Find the first 'True' value in the mask from the current_index
            available_indices = np.where(available_points_mask)[0]
            if not np.any(available_indices >= current_index):
                current_index = available_indices[0]
            else:
                current_index = available_indices[available_indices >= current_index][0]

            P1 = Points_on_plane[current_index, :]
            
            # Add this point to the smooth curve
            Points_smooth_perimeter.append(P1)
            
            # Compute distance from point P1 to all other available points
            distances = np.linalg.norm(Points_on_plane[available_points_mask, :] - P1, axis=1)
            
            # Get indices of points to remove (relative to the available points)
            indices_to_remove_relative = np.where(distances < radius)[0]

            # Convert relative indices to original indices
            original_indices_to_remove = np.where(available_points_mask)[0][indices_to_remove_relative]
            
            # Mark points for removal
            available_points_mask[original_indices_to_remove] = False

        Points_smooth_perimeter = np.array(Points_smooth_perimeter)
        N_points_smooth_curve = Points_smooth_perimeter.shape[0]

        # *************************************************************************
        #           ORDER ARRAY SO THAT NEIGHBOR POINTS ARE IN SEQUENCE
        # *************************************************************************
        # This complex section for re-ordering and fixing crossing segments has been
        # replaced by a simpler and more robust nearest-neighbor sorting approach.
        
        # Start with the first point and find the next closest point until all points are ordered.
        Points_in_order = np.zeros_like(Points_smooth_perimeter)
        remaining_points = list(range(N_points_smooth_curve))
        
        current_idx = remaining_points.pop(0) # Start with point 0
        Points_in_order[0, :] = Points_smooth_perimeter[current_idx, :]

        for i in range(1, N_points_smooth_curve):
            last_point = Points_in_order[i-1, :]
            
            # Calculate distances from the last ordered point to all remaining points
            dist_to_remaining = np.linalg.norm(Points_smooth_perimeter[remaining_points, :] - last_point, axis=1)
            
            # Find the index of the closest point among the remaining ones
            closest_in_remaining_idx = np.argmin(dist_to_remaining)
            
            # Get the original index of that point
            original_idx = remaining_points.pop(closest_in_remaining_idx)
            
            # Add it to the ordered list
            Points_in_order[i, :] = Points_smooth_perimeter[original_idx, :]

        Points_smooth_perimeter = Points_in_order
        N_points_smooth_curve = Points_smooth_perimeter.shape[0]

        # *************************************************************************
        #                            AREA CALCULATION
        # *************************************************************************
        # This section computes the cross-sectional area using the Shoelace formula.
        
        X_smooth_curve = Points_smooth_perimeter[:, 0]
        Y_smooth_curve = Points_smooth_perimeter[:, 1]

        # Shoelace formula for area of a polygon
        # Area = 0.5 * |(x1*y2 + x2*y3 + ... + xn*y1) - (y1*x2 + y2*x3 + ... + yn*x1)|
        self.area = 0.5 * np.abs(np.dot(X_smooth_curve, np.roll(Y_smooth_curve, -1)) - np.dot(Y_smooth_curve, np.roll(X_smooth_curve, -1)))

    


    @classmethod
    def measureIntersection2(self, slicerMatrix):        
        #compute prerequisites for the intersecting points formulas
        plane_normal = np.linalg.inv(slicerMatrix.matrix[:3, :3]) @ np.array([0, 0, 1])
        plane_normal /= np.linalg.norm(plane_normal)
        plane_point = slicerMatrix.matrix[3, :3]

        # Calculate the signed distance from each point to the plane        
        distances = np.dot(self.pointCloudCache - plane_point, plane_normal)
        self._slice_mask = np.abs(distances) < 0.004        #define threshold here     
        self.pointsOnPlane = self.pointCloudCache[self._slice_mask]

        if len(self.pointsOnPlane) <= 4:
            self.area = 0
            self._slice_mask = np.zeros((720,720), dtype=np.bool)
            return
        

    @classmethod
    def savePointCloud(self, filename, visibility=0):                
        if visibility == 2:
            points = self.pointsOnPlane
        else:
            points = self.pointCloudCache

        if filename[-4:] == ".npy":
            np.save(filename, points)
        elif filename[-4:] == ".xyz":
            xyzData = np.reshape(points, (-1,3))
            np.savetxt(filename, xyzData, fmt='%.6f', delimiter=' ')

    @classmethod    
    def convertDistance(self, depth):
        return depth * self.DEPTH_CALIB

