import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import math
import socket
import threading
import time
import json

class MultiArUcoSLAM:

    def __init__(self, calib_data_path, marker_size=10.5, host='127.0.0.1', port=12345):
        # Kalibrációs adatok
        calib_data = np.load(calib_data_path)
        self.cam_mat = calib_data["camMatrix"]
        self.dist_coef = calib_data["distCoef"]
        
        self.MARKER_SIZE = marker_size
        
        # ArUco beállítások
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.param_markers = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.marker_dict, self.param_markers)
        
        # Marker pozíciók tárolása világkoordináta-rendszerben
        self.marker_world_positions = {}
        
        # Kamera trajektória
        self.camera_positions = []
        self.camera_orientations = []  # Quaternion orientációk tárolása
        
        # 3D marker pontok
        self.marker_points = np.array([
            [-self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],
            [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]
        ], dtype=np.float32)
        
        # Socket kommunikáció a PointCloudMesh-hez
        self.socket_host = host
        self.socket_port = port
        self.socket = None
        self.setup_socket_connection()
        
        # Előre definiált marker térkép betöltése
        self.load_predefined_map("predefined_marker_map.json")
        
        # Vizualizáció inicializálása
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Időzítés az FPS számoláshoz
        self.prev_time = cv.getTickCount()
        self.fps = 0
        
        # Kamera orientáció követése
        self.current_orientation = np.eye(3)  # Kezdeti orientáció (identitás mátrix)

    def setup_socket_connection(self):
        """Socket kapcsolat létrehozása a PointCloudMesh-hez"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.socket_host, self.socket_port))
            print(f"Kapcsolódva a PointCloudMesh-hez: {self.socket_host}:{self.socket_port}")
        except Exception as e:
            print(f"Hiba a socket kapcsolat létrehozásakor: {e}")
            self.socket = None

    def send_pose_data(self, position, orientation_matrix):
        if self.socket is None:
            return
        
        # Csak minden 2. képkockánál küldj adatot
        if hasattr(self, 'send_count'):
            self.send_count += 1
            if self.send_count % 2 != 0:
                return
        else:
            self.send_count = 1
        
        try:
            # Egyszerűsített számítások
            quaternion = self.rotation_matrix_to_quaternion(orientation_matrix)
            
            corrected_quaternion = np.array([
                quaternion[0],
                -quaternion[1], 
                -quaternion[2],
                quaternion[3]
            ])
            
            transformed_position = np.array([
                position[0],
                -position[1],
                -position[2]
            ]) / 100.0
            
            pose_data = {
                'position': {
                    'x': float(transformed_position[0]),
                    'y': float(transformed_position[1]), 
                    'z': float(transformed_position[2])
                },
                'orientation': {
                    'w': float(corrected_quaternion[0]),
                    'x': float(corrected_quaternion[1]),
                    'y': float(corrected_quaternion[2]),
                    'z': float(corrected_quaternion[3])
                }
            }
            
            data_str = json.dumps(pose_data) + '\n'
            self.socket.send(data_str.encode())
            
        except Exception as e:
            print(f"Hiba az adatküldéskor: {e}")
            self.socket = None

    def rotation_matrix_to_quaternion(self, R):
        """Forgási mátrix átalakítása quaternionná"""
        # Biztosítjuk, hogy a mátrix ortogonális legyen
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # Quaternion számítás
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        quaternion = np.array([w, x, y, z])
        # Normalizálás
        norm = np.linalg.norm(quaternion)
        if norm > 0:
            quaternion /= norm
        
        return quaternion

    def load_predefined_map(self, filename="predefined_marker_map.json"):
        try:
            with open(filename, 'r') as f:
                map_data = json.load(f)
            
            self.MARKER_SIZE = map_data.get('marker_size', 10.5)
            
            for marker_id_str, data in map_data['markers'].items():
                marker_id = int(marker_id_str)
                R = np.array(data['rotation_matrix'])
                t = np.array(data['translation_vector'])
                self.marker_world_positions[marker_id] = (R, t)
                        
        except Exception as e:
            print(f"Hiba a marker térkép betöltésekor: {e}")

    def detect_and_estimate_poses(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, _ = self.detector.detectMarkers(gray_frame)
        
        detected_markers = {}
        
        if marker_corners and marker_IDs is not None:
            for i, (corners, marker_id) in enumerate(zip(marker_corners, marker_IDs.flatten())):
                corners = corners.reshape(-1, 2).astype(np.float32)
                ret, rvec, tvec = cv.solvePnP(
                    self.marker_points, corners,
                    self.cam_mat, self.dist_coef
                )
                
                if ret:
                    distance = np.linalg.norm(tvec)
                    confidence = self.calculate_marker_confidence(corners)
                    view_angle = self.calculate_view_angle(rvec, tvec)
                    
                    detected_markers[marker_id] = {
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': corners,
                        'distance': distance,
                        'confidence': confidence,
                        'view_angle': view_angle
                    }
        
        return detected_markers

    def calculate_view_angle(self, rvec, tvec):
        R_marker, _ = cv.Rodrigues(rvec)
        marker_normal = np.array([0, 0, 1])
        normal_in_camera = R_marker @ marker_normal
        camera_view_direction = np.array([0, 0, 1])
        
        dot_product = np.dot(normal_in_camera, camera_view_direction)
        norms = np.linalg.norm(normal_in_camera) * np.linalg.norm(camera_view_direction)
        
        if norms == 0:
            return 90.0
        
        cos_angle = dot_product / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        view_angle = min(angle_deg, 180 - angle_deg)
        
        return view_angle

    def calculate_marker_confidence(self, corners):
        if corners is None or len(corners) < 4:
            return 0.0
        
        width = np.linalg.norm(corners[0] - corners[1])
        height = np.linalg.norm(corners[1] - corners[2])
        avg_size = (width + height) / 2
        
        min_size = 20
        max_size = 200
        
        confidence = (avg_size - min_size) / (max_size - min_size)
        confidence = np.clip(confidence, 0.1, 1.0)
        
        return confidence

    def calculate_single_marker_position(self, marker_id, data):
        rvec = data['rvec']
        tvec = data['tvec'].reshape(3,1)
        
        R_m_c, _ = cv.Rodrigues(rvec)
        R_w_m, t_w_m = self.marker_world_positions[marker_id]
        t_w_m = np.asarray(t_w_m).reshape(3, 1)
        
        R_w_c = R_w_m @ R_m_c.T
        t_w_c = t_w_m - R_w_m @ R_m_c.T @ tvec
        
        return R_w_c, t_w_c.flatten()

    def calculate_distance_weight(self, distance):
        max_distance = 200
        min_distance = 20
        
        if distance <= min_distance:
            return 1.0
        elif distance >= max_distance:
            return 0.1
        else:
            weight = 1.0 - (distance - min_distance) / (max_distance - min_distance) * 0.9
            return max(weight, 0.1)

    def calculate_view_angle_weight(self, view_angle, distance):
        max_angle = 90.0
        min_angle = 20.0
        
        if view_angle <= min_angle:
            if distance <= 50:
                return 1.0
            else:
                return 0.0
        elif view_angle >= max_angle:
            return 0.1
        else:
            weight = 1.0 - (view_angle - min_angle) / (max_angle - min_angle) * 0.9
            return max(weight, 0.1)

    def calculate_camera_pose(self, detected_markers):
        if not detected_markers:
            return None, None

        camera_positions = []
        camera_orientations = []
        weights = []
        
        for marker_id, data in detected_markers.items():
            if marker_id not in self.marker_world_positions:
                continue

            R_w_c, position = self.calculate_single_marker_position(marker_id, data)
            
            distance = data['distance']
            confidence = data.get('confidence', 0.5)
            view_angle = data.get('view_angle', 45.0)
            distance_weight = self.calculate_distance_weight(distance)
            angle_weight = self.calculate_view_angle_weight(view_angle, distance)
            
            if angle_weight == 0.0:
                print(f"Marker {marker_id} kihagyva - túl kicsi betekintési szög vagy túl messze van: {view_angle:.1f}°, {distance:.1f}cm")
                continue
                
            final_weight = distance_weight * angle_weight * confidence
            
            camera_positions.append(position)
            camera_orientations.append(R_w_c)
            weights.append(final_weight)

            MAX_MARKERS = 3
            if len(weights) > MAX_MARKERS:
                sorted_indices = np.argsort(weights)[::-1]
                top_indices = sorted_indices[:MAX_MARKERS]
                
                camera_positions = [camera_positions[i] for i in top_indices]
                camera_orientations = [camera_orientations[i] for i in top_indices]
                weights = [weights[i] for i in top_indices]
                print(f"A {MAX_MARKERS} legjobb marker használata")
        
        if not camera_positions:
            return None, None
        
        # Súlyozott átlag a pozícióra és orientációra
        if weights:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            cam_pos = np.zeros(3)
            cam_orient = np.zeros((3, 3))
            
            for i, (pos, orient) in enumerate(zip(camera_positions, camera_orientations)):
                cam_pos += pos * weights[i]
                cam_orient += orient * weights[i]
            
            # Orientáció normalizálása
            U, S, Vt = np.linalg.svd(cam_orient)
            cam_orient = U @ Vt
        else:
            cam_pos = np.mean(camera_positions, axis=0)
            cam_orient = np.mean(camera_orientations, axis=0)
        
        return cam_orient, cam_pos

    def calculate_fps(self):
        current_time = cv.getTickCount()
        time_diff = (current_time - self.prev_time) / cv.getTickFrequency()
        self.prev_time = current_time
        
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        else:
            self.fps = 0
        
        return self.fps

    def update_visualization(self, camera_position=None, camera_orientation=None, detected_markers=None):
        self.ax.clear()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.marker_world_positions)))
        
        # MARKER NÉGYZETEK MEGJELENÍTÉSE - VÁLTOZATLAN
        for i, (marker_id, (R, t)) in enumerate(self.marker_world_positions.items()):
            world_corners = (R @ self.marker_points.T + t).T
            self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2],
                      c=[colors[i]], s=50, 
                      label=f'Marker {marker_id}')
            
            corners_plot = np.vstack([world_corners, world_corners[0]])
            self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2],
                    c=colors[i], linewidth=1, alpha=0.8)
        
        # Kamera pozíció kezelése - OPTIMALIZÁLT
        if camera_position is not None:
            self.camera_positions.append(camera_position)
            if camera_orientation is not None:
                self.camera_orientations.append(camera_orientation)
            
            # Korlátozd a trajektória hosszát
            if len(self.camera_positions) > 50:
                self.camera_positions.pop(0)
                if self.camera_orientations:
                    self.camera_orientations.pop(0)
        
        if self.camera_positions:
            cam_array = np.array(self.camera_positions)
            
            if len(self.camera_positions) > 1:
                # Csak az utolsó 30 pontot kösd össze
                recent_positions = cam_array[-30:]
                self.ax.plot(recent_positions[:,0], recent_positions[:,1], recent_positions[:,2],
                       'r-', alpha=0.6, linewidth=2, label='Kamera trajektória')
            
            self.ax.scatter(cam_array[-1,0], cam_array[-1,1], cam_array[-1,2],
                       c='red', s=100, marker='o', label='Kamera')
            
            # Kamera orientáció megjelenítése - OPTIMALIZÁLT
            if camera_orientation is not None and len(self.camera_positions) > 0:
                current_pos = self.camera_positions[-1]
                axis_length = 8
                
                # Kamera tengelyek
                x_axis = current_pos + camera_orientation[:, 0] * axis_length
                y_axis = current_pos + camera_orientation[:, 1] * axis_length  
                z_axis = current_pos + camera_orientation[:, 2] * axis_length
                
                self.ax.plot([current_pos[0], x_axis[0]], [current_pos[1], x_axis[1]], [current_pos[2], x_axis[2]], 
                       'r-', linewidth=1, label='X axis')
                self.ax.plot([current_pos[0], y_axis[0]], [current_pos[1], y_axis[1]], [current_pos[2], y_axis[2]], 
                       'g-', linewidth=1, label='Y axis')
                self.ax.plot([current_pos[0], z_axis[0]], [current_pos[1], z_axis[1]], [current_pos[2], z_axis[2]], 
                       'b-', linewidth=1, label='Z axis')
        
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title(f'Multi-ArUco SLAM - {len(self.marker_world_positions)} marker\nFPS: {self.fps:.1f}')
        
        # Egyszerűsített legenda
        if len(self.marker_world_positions) > 0:
            self.ax.legend(loc='upper left', fontsize='small')
        
        # Határok beállítása - OPTIMALIZÁLT
        all_positions = []
        for _, (_, t) in self.marker_world_positions.items():
            all_positions.append(t.flatten())
        if self.camera_positions:
            all_positions.extend(self.camera_positions[-20:])
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 15
            self.ax.set_xlim([all_positions[:,0].min()-margin, all_positions[:,0].max()+margin])
            self.ax.set_ylim([all_positions[:,1].min()-margin, all_positions[:,1].max()+margin])
            self.ax.set_zlim([max(0, all_positions[:,2].min()-margin), all_positions[:,2].max()+margin])
        
        plt.draw()
        plt.pause(0.001)

def main():
    slam = MultiArUcoSLAM("../calib_data/MultiMatrix.npz", marker_size=10.5)
    
    cap = cv.VideoCapture(4)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Számlálók a ritkított frissítésekhez
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            fps = slam.calculate_fps()
            detected_markers = slam.detect_and_estimate_poses(frame)
            camera_orientation, camera_position = slam.calculate_camera_pose(detected_markers)
            
            # Ritkított adatküldés (minden 2. frame)
            if frame_count % 2 == 0 and camera_position is not None and camera_orientation is not None:
                slam.send_pose_data(camera_position, camera_orientation)
            
            # Ritkított vizualizáció (minden 3. frame)
            if frame_count % 3 == 0:
                slam.update_visualization(camera_position, camera_orientation, detected_markers)
            
            # Egyszerűsített képkiírás
            for marker_id, data in detected_markers.items():
                corners = data['corners'].astype(np.int32)
                cv.polylines(frame, [corners], True, (0, 255, 255), 2, cv.LINE_AA)
                
                distance = np.linalg.norm(data['tvec'])
                text = f"ID: {marker_id} | {distance:.1f}cm"
                cv.putText(frame, text, tuple(corners[0]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Egyszerűbb status szöveg
            cv.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow("Multi-ArUco SLAM", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv.destroyAllWindows()
        if slam.socket:
            slam.socket.close()

if __name__ == "__main__":
    main()