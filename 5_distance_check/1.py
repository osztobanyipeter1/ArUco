import cv2 as cv
from cv2 import aruco
import numpy as np

# Kalibrációs adat betöltése
calib_data_path = "/home/buvr_tp4/Downloads/OpenCV-main/Distance Estimation/4_calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

MARKER_SIZE = 10  # cm, a marker mérete

# Marker dictionary és detektor paraméterek beállítása
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
param_markers = aruco.DetectorParameters()
detector = aruco.ArucoDetector(marker_dict, param_markers)

# Marker sarkok valós térbeli koordinátái (feltételezzük, hogy a marker síkja XY sík, Z=0)
marker_corners_3d = np.array([
    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

cap = cv.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, _ = detector.detectMarkers(gray_frame)

    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            # Marker lapjának sarkai (2D képpont koordináták)
            corners = corners.reshape(4, 2).astype(np.float32)

            # Pozíció és orientáció becslése solvePnP-vel
            success, rVec, tVec = cv.solvePnP(marker_corners_3d, corners, cam_mat, dist_coef)

            if success:
                # Távolság kiszámítása a tVec alapján
                distance = np.linalg.norm(tVec)

                # Marker körberajzolása
                cv.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)

                # Marker azonosító és távolság kiírása
                top_right = corners[0].astype(int)
                cv.putText(frame, f"id: {ids[0]} Dist: {distance:.2f} cm",
                           tuple(top_right), cv.FONT_HERSHEY_PLAIN,
                           1.3, (0, 0, 255), 2, cv.LINE_AA)

                # Marker tengelyének kirajzolása a frame-re
                cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec, tVec, 4)

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
