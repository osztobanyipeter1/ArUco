import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import json

class MultiArUcoSLAM:
    def __init__(self, calib_data_path, marker_size=10):
        # Kalibrációs adatok betöltése
        calib_data = np.load(calib_data_path)
        self.cam_mat = calib_data["camMatrix"]
        self.dist_coef = calib_data["distCoef"]
        
        self.MARKER_SIZE = marker_size
        
        # ArUco beállítások
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.param_markers = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.marker_dict, self.param_markers)
        
        # Marker pozíciók tárolása világkoordináta-rendszerben
        self.marker_world_positions = {}  # {id: (R, t)}
        self.marker_confidence = {}  # {id: confidence_score}
        
        # Kamera trajektória
        self.camera_positions = []
        
        # Referencia marker ID (az első észlelt marker lesz)
        self.reference_marker_id = None
        
        # 3D marker pontok marker koordináta-rendszerben
        self.marker_points = np.array([
            [-self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],
            [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]
        ], dtype=np.float32)
        
        # Vizualizáció inicializálása
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
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
                    detected_markers[marker_id] = {
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': corners
                    }
        
        return detected_markers
    
    def update_marker_map(self, detected_markers):
        """JAVÍTOTT marker térkép frissítése"""
        if not detected_markers:
            return None
            
        # Referencia marker beállítása (első észlelt marker)
        if self.reference_marker_id is None:
            self.reference_marker_id = min(detected_markers.keys())
            self.marker_world_positions[self.reference_marker_id] = (np.eye(3), np.zeros((3, 1)))
            self.marker_confidence[self.reference_marker_id] = 1.0
            
        # Kamera pozíció számítása referencia marker alapján
        if self.reference_marker_id in detected_markers:
            ref_data = detected_markers[self.reference_marker_id]
            R_cam_to_ref = cv.Rodrigues(ref_data['rvec'])[0]
            t_cam_to_ref = ref_data['tvec']
            
            # Kamera pozíció világkoordinátákban - JAVÍTOTT
            camera_position = (-R_cam_to_ref.T @ t_cam_to_ref).flatten()
            
            # Új markerek pozíciójának számítása
            for marker_id, data in detected_markers.items():
                if marker_id not in self.marker_world_positions:
                    R_cam_to_marker = cv.Rodrigues(data['rvec'])[0]
                    t_cam_to_marker = data['tvec']
                    
                    # Transzformáció világkoordinátákba
                    R_world_to_marker = R_cam_to_marker @ R_cam_to_ref.T
                    t_world_to_marker = R_cam_to_ref.T @ (t_cam_to_marker - t_cam_to_ref)
                    
                    self.marker_world_positions[marker_id] = (R_world_to_marker, t_world_to_marker)
                    self.marker_confidence[marker_id] = 0.1
                
                # JAVÍTOTT confidence frissítés
                if marker_id in self.marker_confidence:
                    self.marker_confidence[marker_id] = min(1.0, 
                        self.marker_confidence[marker_id] + 0.1)
            
            return camera_position
        
        # JAVÍTOTT: Ha nincs referencia marker a láthatóságban
        for marker_id in detected_markers:
            if marker_id in self.marker_world_positions:
                data = detected_markers[marker_id]
                R_cam_to_marker = cv.Rodrigues(data['rvec'])[0]
                t_cam_to_marker = data['tvec']
                
                R_world_to_marker, t_world_to_marker = self.marker_world_positions[marker_id]
                
                # HELYES kamera pozíció számítás
                R_marker_to_world = R_world_to_marker.T
                t_marker_to_world = -R_world_to_marker.T @ t_world_to_marker
                
                # KRITIKUS JAVÍTÁS: flatten() hozzáadása
                camera_position = (R_marker_to_world @ t_cam_to_marker + t_marker_to_world.flatten()).flatten()
                
                # Confidence frissítés
                if marker_id in self.marker_confidence:
                    self.marker_confidence[marker_id] = min(1.0, 
                        self.marker_confidence[marker_id] + 0.1)
                
                return camera_position
                
        return None
    
    def bundle_adjustment(self):
        if len(self.marker_world_positions) < 2:
            return
        pass
    
    def update_visualization(self, camera_position=None, detected_markers=None):
        """JAVÍTOTT 3D vizualizáció"""
        self.ax.clear()
        
        # JAVÍTOTT: Színek kezelése üres lista esetére
        if len(self.marker_world_positions) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.marker_world_positions)))
        else:
            colors = []
        
        for i, (marker_id, (R, t)) in enumerate(self.marker_world_positions.items()):
            # Marker sarkok világkoordinátákban
            world_corners = (R @ self.marker_points.T + t).T
            
            # Marker megjelenítése
            color_idx = i % len(colors) if len(colors) > 0 else 0
            marker_color = colors[color_idx] if len(colors) > 0 else 'blue'
            
            self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2], 
                          c=[marker_color], s=100, 
                          label=f'Marker {marker_id} (conf: {self.marker_confidence[marker_id]:.1f})')
            
            # Marker kontúr
            corners_plot = np.vstack([world_corners, world_corners[0]])
            self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2], 
                        c=marker_color, linewidth=2)
            
            # Marker normálvektor
            normal = R @ np.array([0, 0, 1])
            center = t.flatten()
            self.ax.quiver(center[0], center[1], center[2],
                          normal[0], normal[1], normal[2],
                          length=5, color=marker_color, alpha=0.7)
        
        # JAVÍTOTT: Kamera pozíció kezelése
        if camera_position is not None:
            # KRITIKUS: Biztosítjuk, hogy camera_position 1D tömb
            if camera_position.ndim > 1:
                camera_position = camera_position.flatten()
            self.camera_positions.append(camera_position)
            
        if self.camera_positions:
            # JAVÍTOTT: Biztonságos tömb konverzió
            try:
                cam_pos_array = np.array(self.camera_positions)
                
                # Trajektória
                if len(self.camera_positions) > 1:
                    self.ax.plot(cam_pos_array[:,0], cam_pos_array[:,1], cam_pos_array[:,2], 
                               'r-', alpha=0.5, linewidth=2, label='Kamera trajektória')
                
                # Aktuális kamera pozíció
                self.ax.scatter(cam_pos_array[-1,0], cam_pos_array[-1,1], cam_pos_array[-1,2], 
                               c='red', s=150, label='Kamera')
                
            except ValueError as e:
                print(f"Kamera pozíció tömb hiba: {e}")
                # Utolsó pozíció megjelenítése fallback-ként
                if len(self.camera_positions) > 0:
                    last_pos = self.camera_positions[-1]
                    if hasattr(last_pos, '__len__') and len(last_pos) >= 3:
                        self.ax.scatter(last_pos[0], last_pos[1], last_pos[2], 
                                       c='red', s=150, label='Kamera')
            
            # JAVÍTOTT: Kamera orientáció
            if camera_position is not None and detected_markers:
                camera_z_axis = None
                
                if self.reference_marker_id in detected_markers:
                    ref_data = detected_markers[self.reference_marker_id]
                    R_cam_to_ref = cv.Rodrigues(ref_data['rvec'])[0]
                    camera_z_axis = R_cam_to_ref.T @ np.array([0, 0, 1])
                    
                else:
                    # Bármely ismert marker használata
                    for marker_id, data in detected_markers.items():
                        if marker_id in self.marker_world_positions:
                            R_cam_to_marker = cv.Rodrigues(data['rvec'])[0]
                            R_world_to_marker, _ = self.marker_world_positions[marker_id]
                            
                            # Kamera orientáció világkoordinátákban
                            R_world_to_cam = R_cam_to_marker.T @ R_world_to_marker.T
                            camera_z_axis = R_world_to_cam @ np.array([0, 0, 1])
                            break
                
                # Kamera irány megjelenítése
                if camera_z_axis is not None:
                    self.ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                                  camera_z_axis[0], camera_z_axis[1], camera_z_axis[2],
                                  length=8, color='green', label='Kamera nézés iránya')
        
        # Tengelyek és címkék
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title(f'Multi-ArUco SLAM - {len(self.marker_world_positions)} marker észlelve')
        
        # Referencia marker jelölése
        if self.reference_marker_id is not None:
            self.ax.text2D(0.05, 0.95, f'Referencia marker: {self.reference_marker_id}', 
                          transform=self.ax.transAxes)
        
        # JAVÍTOTT: Legend csak akkor, ha van mit megjeleníteni
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()
        
        # Dinamikus határok
        all_positions = []
        for _, (_, t) in self.marker_world_positions.items():
            all_positions.append(t.flatten())
        if self.camera_positions:
            for pos in self.camera_positions:
                if hasattr(pos, '__len__') and len(pos) >= 3:
                    all_positions.append(pos[:3] if hasattr(pos, '__getitem__') else [pos[0], pos[1], pos[2]])
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 20
            self.ax.set_xlim([all_positions[:,0].min()-margin, all_positions[:,0].max()+margin])
            self.ax.set_ylim([all_positions[:,1].min()-margin, all_positions[:,1].max()+margin])
            self.ax.set_zlim([max(0, all_positions[:,2].min()-margin), all_positions[:,2].max()+margin])
        
        plt.draw()
        plt.pause(0.01)
    
    def save_map(self, filename="aruco_map.json"):
        map_data = {
            'reference_marker_id': int(self.reference_marker_id) if self.reference_marker_id else None,
            'markers': {},
            'marker_size': self.MARKER_SIZE
        }
        
        for marker_id, (R, t) in self.marker_world_positions.items():
            map_data['markers'][str(marker_id)] = {
                'rotation_matrix': R.tolist(),
                'translation_vector': t.tolist(),
                'confidence': self.marker_confidence[marker_id]
            }
        
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"Marker térkép elmentve: {filename}")
    
    def load_map(self, filename="aruco_map.json"):
        try:
            with open(filename, 'r') as f:
                map_data = json.load(f)
            
            self.reference_marker_id = map_data.get('reference_marker_id')
            self.MARKER_SIZE = map_data.get('marker_size', 10)
            
            for marker_id_str, data in map_data['markers'].items():
                marker_id = int(marker_id_str)
                R = np.array(data['rotation_matrix'])
                t = np.array(data['translation_vector'])
                self.marker_world_positions[marker_id] = (R, t)
                self.marker_confidence[marker_id] = data['confidence']
            
            print(f"Marker térkép betöltve: {filename}")
            print(f"Betöltött markerek: {list(self.marker_world_positions.keys())}")
            
        except FileNotFoundError:
            print(f"Marker térkép fájl nem található: {filename}")
        except Exception as e:
            print(f"Hiba a marker térkép betöltésekor: {e}")

def main():
    # SLAM rendszer inicializálása
    slam = MultiArUcoSLAM("../calib_data/MultiMatrix.npz", marker_size=10)
    
    # korábbi térkép betöltése
    # slam.load_map("aruco_map.json")
    
    # Kamera
    cap = cv.VideoCapture(4)
    cv.namedWindow("Multi-ArUco SLAM", cv.WINDOW_NORMAL)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Markerek észlelése
            detected_markers = slam.detect_and_estimate_poses(frame)
            
            # Marker térkép frissítése és kamera pozíció számítása
            camera_position = slam.update_marker_map(detected_markers)
            
            # Vizualizáció frissítése
            slam.update_visualization(camera_position, detected_markers)
            
            # Markerek rajzolása az eredeti képre
            for marker_id, data in detected_markers.items():
                corners = data['corners'].astype(np.int32)
                
                # Marker kontúr
                cv.polylines(frame, [corners], True, (0, 255, 255), 4, cv.LINE_AA)
                
                # Tengelyek rajzolása
                cv.drawFrameAxes(frame, slam.cam_mat, slam.dist_coef, 
                               data['rvec'], data['tvec'], 4, 4)
                
                # Marker ID és távolság
                distance = np.linalg.norm(data['tvec'])
                confidence = slam.marker_confidence.get(marker_id, 0)
                
                text = f"ID: {marker_id} | Dist: {distance:.1f}cm | Conf: {confidence:.1f}"
                cv.putText(frame, text, tuple(corners[0]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Státusz információk
            status_text = [
                f"Markerek száma: {len(slam.marker_world_positions)}",
                f"Referencia marker: {slam.reference_marker_id}",
                f"Kamera pozíciók: {len(slam.camera_positions)}"
            ]
            
            for i, text in enumerate(status_text):
                cv.putText(frame, text, (10, 30 + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.imshow("Multi-ArUco SLAM", frame)
            
            # Bundle adjustment időnként
            frame_count += 1
            if frame_count % 100 == 0:
                slam.bundle_adjustment()
            
            # Kilépés és mentés
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                slam.save_map("aruco_map.json")
                print("Térkép mentve!")
            elif key == ord('l'):
                slam.load_map("aruco_map.json")
                print("Térkép betöltve!")
            elif key == ord('c'):
                slam.camera_positions.clear()
                print("Kamera trajektória törölve!")
    
    finally:
        cap.release()
        cv.destroyAllWindows()
        plt.ioff()
        
        # Automatikus mentés kilépéskor
        if slam.marker_world_positions:
            slam.save_map("aruco_map_final.json")
            print("Végleges térkép automatikusan mentve!")
        
        plt.close()

if __name__ == "__main__":
    main()