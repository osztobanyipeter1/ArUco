import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Kalibrációs adatok betöltése
calib_data_path = "../calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"] #3x3 belső kamera mátrix (fókusztávolság, főpont koordináták)
dist_coef = calib_data["distCoef"] #lencsedisztorzió együtthatók

MARKER_SIZE = 10  # centiméter
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250) #5x5 pixeles marker szótár, 250 különböző marker
param_markers = aruco.DetectorParameters() #detektálási paraméterek (küszöbök, stb stb)
detector = aruco.ArucoDetector(marker_dict, param_markers) #ténylegese detektor objektum

# 3D vizualizáció
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('3D Kamera és Marker pozíció')

# 3D marker pontok marker koordináta-rendszerben. Ezek a marker fizikai pontjai, és a ( , ,0) Z=0 azt jelenti, hogy a marker a síkban van
marker_points = np.array([
    [-MARKER_SIZE/2, MARKER_SIZE/2, 0],  #bal felső (-5, 5, 0)
    [MARKER_SIZE/2, MARKER_SIZE/2, 0], #jobb felső (5, 5, 0)
    [MARKER_SIZE/2, -MARKER_SIZE/2, 0], #jobb alsó (5, -5, 0)
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0] #bal alsó (-5, -5, 0)
])

camera_positions = [] # Kamera trajektória
cap = cv.VideoCapture(4)

# OpenCV ablak létrehozása
cv.namedWindow("ArUco Detekció", cv.WINDOW_NORMAL)

def update_3d_plot(rvec, tvec):
    ax.clear() #3D vizualizáció frissítése

    #Trajektória
    ax.scatter(marker_points[:,0], marker_points[:,1], marker_points[:,2], 
              c='blue', s=50, label='Marker')
    
    R, _ = cv.Rodrigues(rvec)
    # Javított mátrix szorzás
    camera_position = (-R.T @ tvec).flatten()
    camera_positions.append(camera_position)
    
    if len(camera_positions) > 1:
        cam_pos_array = np.array(camera_positions)
        ax.plot(cam_pos_array[:,0], cam_pos_array[:,1], cam_pos_array[:,2], 
              'r-', alpha=0.3, label='Kamera útvonal')
    
    #aktuális kamera pozíció
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], 
             c='red', s=100, label='Kamera')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
             R[2,0], R[2,1], R[2,2], length=5, color='green', label='Nézés iránya')
    
    #tengelyek és címkék
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('3D Kamera és Marker pozíció')
    ax.legend()
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([0, 100])
    
    plt.draw()
    plt.pause(0.01)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, rejected = detector.detectMarkers(gray_frame)
        
        if marker_corners:
            rVec, tVec = [], []
            for corners in marker_corners:
                obj_points = np.array([
                    [-MARKER_SIZE/2, MARKER_SIZE/2, 0],
                    [MARKER_SIZE/2, MARKER_SIZE/2, 0],
                    [MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
                ], dtype=np.float32)
                
                # A corners-t is float32-re kell konvertálni
                corners = corners.reshape(-1, 2).astype(np.float32)
                ret, rvec, tvec = cv.solvePnP(obj_points, corners, cam_mat, dist_coef)
                if ret:
                    rVec.append(rvec)
                    tVec.append(tvec)
            
            for i, (ids, corners) in enumerate(zip(marker_IDs, marker_corners)):
                corners_int = corners.astype(np.int32)
                cv.polylines(frame, [corners_int], True, (0, 255, 255), 4, cv.LINE_AA)
                
                # Csak akkor rajzoljunk tengelyeket, ha van érvényes pozíció
                if i < len(rVec):
                    cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                    update_3d_plot(rVec[i], tVec[i])
                    
                    distance = np.linalg.norm(tVec[i])
                    cv.putText(frame, f"ID: {ids[0]} Dist: {distance:.1f}cm", 
                              tuple(corners_int[0][0]), 
                              cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
        
        cv.imshow("ArUco Detekció", frame)
        
        key = cv.waitKey(1)
        if key == ord('q'):
            break

finally:
    cap.release()
    cv.destroyAllWindows()
    plt.ioff()
    plt.close()