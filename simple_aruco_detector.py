#!/usr/bin/env python3

"""
Simple ArUco Marker Detection Script
Detects ArUco markers in a static image file and displays the results.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_rectangle_area(coordinates):
    """
    Calculate area and width of detected ArUco marker
    
    Args:
        coordinates (list): 4 sets of (x,y) coordinates from ArUco detection
    
    Returns:
        area (float): Area of detected ArUco marker
        width (float): Width of detected ArUco marker
    """
    # Ensure that there are exactly four coordinates
    if len(coordinates) != 4:
        raise ValueError("Coordinates list must contain exactly four (x, y) coordinates.")

    # Sort the coordinates in clockwise order (starting from top-left)
    sorted_coordinates = sorted(coordinates, key=lambda xy: (xy[1], xy[0]))

    # Calculate the width and height of the rectangle
    width = np.linalg.norm(np.array(sorted_coordinates[0]) - np.array(sorted_coordinates[1]))
    height = np.linalg.norm(np.array(sorted_coordinates[1]) - np.array(sorted_coordinates[2]))

    # Calculate the area of the rectangle
    area = width * height
    
    return area, width

def detect_aruco_in_image(image_path):
    """
    Detect ArUco markers in a static image file
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing detection results
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Detection parameters
    aruco_area_threshold = 1500  # Minimum area threshold for detection
    
    # Camera matrix (you may need to adjust this based on your camera)
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], 
                       [0.0, 931.1829833984375, 360.0], 
                       [0.0, 0.0, 1.0]])
    
    # Distortion matrix (set to zero for simulation)
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # ArUco marker size in meters (150mm x 150mm)
    size_of_aruco_m = 0.15
    
    # Initialize result lists
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    filtered_ids = []
    
    # Convert BGR image to grayscale for ArUco detection
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Set up ArUco detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_para = cv2.aruco.DetectorParameters_create()
    
    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=aruco_para)
    
    # Draw detected markers on the image
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        print(f"Detected {len(ids)} ArUco markers")
        
        # Process each detected marker
        for i in range(len(ids)):
            marker_corners = corners[i][0]
            area, width = calculate_rectangle_area(marker_corners)
            
            print(f"Marker ID {ids[i][0]}: Area = {area:.2f}, Width = {width:.2f}")
            
            # Filter out markers that are too small (likely too far away)
            if area < aruco_area_threshold:
                print(f"Marker ID {ids[i][0]} filtered out (area too small)")
                continue
                
            filtered_ids.append(ids[i])
            
            # Estimate pose of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_of_aruco_m, cam_mat, dist_mat)
            
            # Extract distance and position information
            distance_from_rgb = tvec[0][0][2]
            Xc, Yc, Zc = tvec[0][0]
            
            # Calculate center point in image coordinates
            x_c = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) * 0.25
            y_c = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) * 0.25
            
            # Apply rotation correction for proper Z-axis alignment
            rot_mat1 = R.from_rotvec(rvec[0]).as_matrix()
            
            # Correction rotation (pitch = -90 degrees)
            roll, pitch, yaw = 0, -1.57, 0
            rot_mat2 = R.from_euler('zxy', [roll, pitch, yaw], degrees=False).as_matrix()
            
            # Combined rotation
            rotation_matrix = np.dot(rot_mat1, rot_mat2)
            r = R.from_matrix(rotation_matrix)
            r_euler = list(r.as_euler('zxy', degrees=False)[0])
            print(r_euler)
            
            # Draw frame axes on the marker
            frame_length = size_of_aruco_m * 6.7
            cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, frame_length)
            
            # Mark center point on image
            cv2.circle(image, (int(x_c), int(y_c)), 5, (0, 255, 0), -1)
            cv2.putText(image, f'ID:{ids[i][0]}', (int(x_c-25), int(y_c-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Store results
            center_aruco_list.append([x_c, y_c])
            distance_from_rgb_list.append(distance_from_rgb)
            angle_aruco_list.append(r_euler)
            width_aruco_list.append(width)
            
            # Print detection details
            print(f"Marker ID {ids[i][0]}:")
            print(f"  Center: ({x_c:.2f}, {y_c:.2f})")
            print(f"  Distance: {distance_from_rgb:.3f} m")
            print(f"  Position (X,Y,Z): ({Xc:.3f}, {Yc:.3f}, {Zc:.3f})")
            # print(f"  Rotation (roll, pitch, yaw): ({r_euler[0]:.3f}, {r_euler[1]:.3f}, {r_euler[2]:.3f})")
            print("-" * 40)
    
    else:
        print("No ArUco markers detected in the image")
    
    # Prepare results dictionary
    results = {
        'image': image,
        'detected_count': len(filtered_ids) if filtered_ids else 0,
        'marker_ids': [id[0] for id in filtered_ids] if filtered_ids else [],
        'centers': center_aruco_list,
        'distances': distance_from_rgb_list,
        'angles': angle_aruco_list,
        'widths': width_aruco_list
    }
    
    return results

def main():
    """
    Main function to run ArUco detection on an image file
    """
    # Image file path
    image_path = "image.jpg"
    
    print(f"Detecting ArUco markers in: {image_path}")
    print("=" * 50)
    
    # Detect ArUco markers
    results = detect_aruco_in_image(image_path)
    
    if results is not None:
        print(f"\nDetection Summary:")
        print(f"Total markers detected: {results['detected_count']}")
        if results['marker_ids']:
            print(f"Marker IDs: {results['marker_ids']}")
        
        # Display the result image
        cv2.imshow("ArUco Detection Results", results['image'])
        print("\nPress any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optionally save the result image
        output_path = "aruco_detection_result.jpg"
        cv2.imwrite(output_path, results['image'])
        print(f"Result image saved as: {output_path}")
    
    else:
        print("Failed to process the image")

if __name__ == "__main__":
    main()
