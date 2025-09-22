import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import json
import os


class KalmanFilter3D:
    """3D Kálmán szűrő a kamera pozíció simításához"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # Állapot: [x, y, z, vx, vy, vz]
        self.state_dim = 6
        self.measurement_dim = 3
        
        # Kálmán szűrő inicializálása
        self.kf = cv.KalmanFilter(self.state_dim, self.measurement_dim)
        
        # Átmeneti mátrix (F) - állapot átmenet
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        dt = 1.0  # időlépés
        for i in range(3):
            self.kf.transitionMatrix[i, i+3] = dt
        
        # Mérési mátrix (H) - csak pozíciót mérünk
        self.kf.measurementMatrix = np.zeros((self.measurement_dim, self.state_dim), dtype=np.float32)
        for i in range(3):
            self.kf.measurementMatrix[i, i] = 1
        
        # Process zaj kovariancia (Q)
        self.kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * process_noise
        
        # Mérési zaj kovariancia (R)
        self.kf.measurementNoiseCov = np.eye(self.measurement_dim, dtype=np.float32) * measurement_noise
        
        # Állapot kovariancia (P)
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32)
        
        # Kezdeti állapot
        self.kf.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)
        
        self.is_initialized = False
    
    def init(self, initial_position):
        """Kálmán szűrő inicializálása kezdeti pozícióval"""
        initial_position = initial_position.astype(np.float32)
        for i in range(3):
            self.kf.statePost[i] = initial_position[i]
        self.is_initialized = True
    
    def predict(self):
        """Előrejelzés"""
        return self.kf.predict()
    
    def update(self, measurement):
        """Frissítés méréssel"""
        if not self.is_initialized:
            self.init(measurement)
            return measurement
        
        predicted = self.predict()
        # Biztosítjuk, hogy a mérés megfelelő típusú legyen
        measurement_float32 = measurement.astype(np.float32).reshape(-1, 1)
        corrected = self.kf.correct(measurement_float32)
        
        return corrected[:3].flatten()
    
    def get_current_state(self):
        """Jelenlegi állapot lekérése"""
        return self.kf.statePost[:3].flatten()


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
        self.filtered_camera_positions = []
        
        # Kálmán szűrő a kamera pozíció simításához
        self.kalman_filter = KalmanFilter3D(process_noise=0.1, measurement_noise=1.0)
        
        # Referencia marker ID
        self.reference_marker_id = None
        
        # 3D marker pontok marker koordináta-rendszerben
        self.marker_points = np.array([
            [-self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],
            [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]
        ], dtype=np.float32)
        
        # Előre definiált marker térkép betöltése
        self.load_predefined_map("predefined_marker_map.json")
        
        # Korábbi kamera pozíció outlier szűréshez
        self.last_valid_position = None
        self.position_history = []
        self.max_position_change = 50.0  # Maximum megengedett pozíció változás (cm)
        
        # Vizualizáció inicializálása
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Időzítés az FPS számoláshoz
        self.prev_time = cv.getTickCount()
        self.fps = 0
    
    def create_predefined_map(self, filename="predefined_marker_map.json"):
        """Előre definiált marker térkép létrehozása 2-es sorban növekvő sorrendben"""
        map_data = {
            'reference_marker_id': 0,
            'markers': {},
            'marker_size': self.MARKER_SIZE
        }
        
        # Markerek elhelyezése 2-es sorban növekvő sorrendben, 30 cm távolsággal
        for i in range(20):
            row = i // 2
            col = i % 2
            
            # Pozíciók centiméterben
            x = col * 30  # 0 vagy 30 cm
            y = row * 30  # 0, 30, 60, ... cm
            z = 0         # mind a földön
            
            # Orientáció (síkban fekszenek, normál felfelé mutat)
            R = np.eye(3)  # identitás mátrix - nincs forgatás
            
            map_data['markers'][str(i)] = {
                'rotation_matrix': R.tolist(),
                'translation_vector': [[x], [y], [z]],
                'confidence': 1.0  # maximális bizalom
            }
        
        # Fájl mentése
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"Előre definiált marker térkép létrehozva: {filename}")
        return map_data
    
    def load_predefined_map(self, filename="predefined_marker_map.json"):
        """Előre definiált marker térkép betöltése"""
        # Ha a fájl nem létezik, létrehozzuk
        if not os.path.exists(filename):
            print("Előre definiált marker térkép nem található, létrehozás...")
            self.create_predefined_map(filename)
        
        try:
            with open(filename, 'r') as f:
                map_data = json.load(f)
            
            self.reference_marker_id = map_data.get('reference_marker_id')
            self.MARKER_SIZE = map_data.get('marker_size', 10)
            
            # Marker pozíciók betöltése
            for marker_id_str, data in map_data['markers'].items():
                marker_id = int(marker_id_str)
                R = np.array(data['rotation_matrix'])
                t = np.array(data['translation_vector'])
                self.marker_world_positions[marker_id] = (R, t)
                self.marker_confidence[marker_id] = data['confidence']
            
            print(f"Előre definiált marker térkép betöltve: {filename}")
            print(f"Betöltött markerek: {list(self.marker_world_positions.keys())}")
            
        except Exception as e:
            print(f"Hiba a marker térkép betöltésekor: {e}")
            # Hiba esetén is létrehozzuk az alapértelmezett térképet
            map_data = self.create_predefined_map(filename)
            self.load_predefined_map(filename)
    
    def detect_and_estimate_poses(self, frame):
        """Markerek észlelése és pózok becslése"""
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
                    # Számoljuk a re-projection errort a bizalom számolásához
                    projected_points, _ = cv.projectPoints(
                        self.marker_points, rvec, tvec, 
                        self.cam_mat, self.dist_coef
                    )
                    reprojection_error = np.mean(np.linalg.norm(corners - projected_points.reshape(-1, 2), axis=1))
                    
                    # Bizalom számolása (kisebb error = nagyobb bizalom)
                    confidence = max(0.1, 1.0 - reprojection_error / 10.0)
                    
                    detected_markers[marker_id] = {
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': corners,
                        'confidence': confidence,
                        'reprojection_error': reprojection_error
                    }
        
        return detected_markers
    
    def is_position_valid(self, new_position, detected_markers):
        """Ellenőrzi, hogy az új pozíció valid-e (outlier szűrés)"""
        if self.last_valid_position is None:
            return True
        
        # Pozíció változás mértéke
        position_change = np.linalg.norm(new_position - self.last_valid_position)
        
        # Ha túl nagy a változás, valószínűleg outlier
        if position_change > self.max_position_change:
            print(f"Outlier detected! Position change: {position_change:.1f}cm > {self.max_position_change}cm")
            return False
        
        # További ellenőrzések...
        # Például: legalább 2 marker kell a megbízható pozícióhoz
        if len(detected_markers) < 2:
            print(f"Warning: Only {len(detected_markers)} markers detected")
            return position_change < (self.max_position_change / 2)  # Szigorúbb limit kevesebb marker esetén
        
        return True
    
    def update_marker_map(self, detected_markers):
        """Marker térkép frissítése – súlyozott átlag a marker konfidenciák alapján"""
        if not detected_markers:
            return None

        camera_positions = []
        weights = []
        
        for marker_id, data in detected_markers.items():
            if marker_id not in self.marker_world_positions:
                continue

            rvec = data['rvec']
            tvec = data['tvec'].reshape(3,1)
            
            R_m_c, _ = cv.Rodrigues(rvec)
            R_w_m, t_w_m = self.marker_world_positions[marker_id]
            t_w_m = np.asarray(t_w_m).reshape(3, 1)
            
            R_w_c = R_w_m @ R_m_c.T
            t_w_c = t_w_m - R_w_m @ R_m_c.T @ tvec
            
            camera_positions.append(t_w_c.flatten())
            
            # Súly számolása a marker konfidencia és re-projection error alapján
            marker_confidence = self.marker_confidence.get(marker_id, 0.5)
            detection_confidence = data.get('confidence', 0.5)
            weight = marker_confidence * detection_confidence
            
            # Közelebbi markerek nagyobb súllyal
            distance = np.linalg.norm(tvec)
            distance_weight = 1.0 / max(1.0, distance / 10.0)  # Normalizálás
            
            final_weight = weight * distance_weight
            weights.append(final_weight)
            
        if not camera_positions:
            return None
            
        # Normalizáljuk a súlyokat
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)  # Egyenletes súlyozás
        
        # Súlyozott átlag
        cam_pos = np.zeros(3)
        for i, pos in enumerate(camera_positions):
            cam_pos += pos * weights[i]
        
        # Outlier szűrés
        if not self.is_position_valid(cam_pos, detected_markers):
            print("Invalid position rejected, using last valid position")
            return self.last_valid_position
        
        self.last_valid_position = cam_pos.astype(np.float32)
        return self.last_valid_position
    
    def smooth_camera_position(self, raw_position):
        """Kamera pozíció simítása Kálmán szűrővel"""
        if raw_position is None:
            # Ha nincs mérés, csak előrejelzést végzünk
            if self.kalman_filter.is_initialized:
                predicted = self.kalman_filter.predict()
                return predicted[:3].flatten()
            return None
        
        # Kálmán szűrő frissítése a mért pozícióval
        smoothed_position = self.kalman_filter.update(raw_position)
        
        # Pozíció történet tárolása
        self.position_history.append(smoothed_position.copy())
        if len(self.position_history) > 10:  # Tartsuk csak az utolsó 10 pozíciót
            self.position_history.pop(0)
        
        return smoothed_position
    
    def bundle_adjustment(self):
        """Bundle adjustment - jelenleg nem csinál semmit"""
        pass
    
    def calculate_fps(self):
        """FPS számolása"""
        current_time = cv.getTickCount()
        time_diff = (current_time - self.prev_time) / cv.getTickFrequency()
        self.prev_time = current_time
        
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        else:
            self.fps = 0
        
        return self.fps
    
    def update_visualization(self, camera_position=None, filtered_position=None, detected_markers=None):
        """3D vizualizáció frissítése"""
        self.ax.clear()
        
        # Markerek megjelenítése
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.marker_world_positions)))
        
        for i, (marker_id, (R, t)) in enumerate(self.marker_world_positions.items()):
            # Marker sarkok világkoordináta-rendszerben
            world_corners = (R @ self.marker_points.T + t).T
            
            # Marker megjelenítése
            self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2], 
                          c=[colors[i]], s=100, 
                          label=f'Marker {marker_id} (conf: {self.marker_confidence[marker_id]:.1f})')
            
            # Marker kontúr
            corners_plot = np.vstack([world_corners, world_corners[0]])
            self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2], 
                        c=colors[i], linewidth=2)
            
            # Marker normálvektor megjelenítése
            normal = R @ np.array([0, 0, 1])
            center = t.flatten()
            self.ax.quiver(center[0], center[1], center[2],
                          normal[0], normal[1], normal[2],
                          length=5, color=colors[i], alpha=0.7)
        
        # Kamera pozíció és trajektória
        if filtered_position is not None:
            self.filtered_camera_positions.append(filtered_position)
            
        if self.filtered_camera_positions:
            filtered_cam_array = np.array(self.filtered_camera_positions)
            
            # Szűrt trajektória
            if len(self.filtered_camera_positions) > 1:
                self.ax.plot(filtered_cam_array[:,0], filtered_cam_array[:,1], filtered_cam_array[:,2], 
                           'r-', alpha=0.8, linewidth=3, label='Szűrt kamera trajektória')
            
            # Nyers trajektória (ha van)
            if camera_position is not None and len(self.camera_positions) > 0:
                raw_cam_array = np.array(self.camera_positions)
                if len(self.camera_positions) > 1:
                    self.ax.plot(raw_cam_array[:,0], raw_cam_array[:,1], raw_cam_array[:,2], 
                               'b--', alpha=0.3, linewidth=1, label='Nyers kamera trajektória')
            
            # Aktuális szűrt kamera pozíció
            self.ax.scatter(filtered_cam_array[-1,0], filtered_cam_array[-1,1], filtered_cam_array[-1,2], 
                           c='red', s=200, marker='o', label='Szűrt kamera')
            
            # Aktuális nyers kamera pozíció
            if camera_position is not None:
                self.ax.scatter(camera_position[0], camera_position[1], camera_position[2], 
                               c='blue', s=100, marker='x', label='Nyers kamera', alpha=0.5)
            
            # Kamera nézés irányának megjelenítése
            if filtered_position is not None and detected_markers and len(detected_markers) > 0:
                # Használjuk a legmagasabb konfidenciájú markert
                best_marker_id = max(detected_markers.keys(), 
                                   key=lambda id: detected_markers[id].get('confidence', 0))
                data = detected_markers[best_marker_id]
                R_cam_to_marker = cv.Rodrigues(data['rvec'])[0]
                
                camera_z_axis = R_cam_to_marker.T @ np.array([0, 0, 1])
                self.ax.quiver(filtered_position[0], filtered_position[1], filtered_position[2],
                              camera_z_axis[0], camera_z_axis[1], camera_z_axis[2],
                              length=8, color='green', label='Kamera nézés iránya')
        
        # Tengelyek és címkék
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title(f'Multi-ArUco SLAM - {len(self.marker_world_positions)} marker betöltve\nFPS: {self.fps:.1f}')
        
        # Referencia marker jelölése
        if self.reference_marker_id is not None:
            self.ax.text2D(0.05, 0.95, f'Referencia marker: {self.reference_marker_id}', 
                          transform=self.ax.transAxes)
        
        self.ax.legend()
        
        # Dinamikus határok
        all_positions = []
        for _, (_, t) in self.marker_world_positions.items():
            all_positions.append(t.flatten())
        if self.filtered_camera_positions:
            all_positions.extend(self.filtered_camera_positions)
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 20
            self.ax.set_xlim([all_positions[:,0].min()-margin, all_positions[:,0].max()+margin])
            self.ax.set_ylim([all_positions[:,1].min()-margin, all_positions[:,1].max()+margin])
            self.ax.set_zlim([max(0, all_positions[:,2].min()-margin), all_positions[:,2].max()+margin])
        
        plt.draw()
        plt.pause(0.01)
    
    def save_map(self, filename="aruco_map.json"):
        """Marker térkép mentése"""
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
        """Marker térkép betöltése"""
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
    
    # Kamera
    cap = cv.VideoCapture(4)
    if not cap.isOpened():
        print("Hiba: Kamera nem elérhető!")
        return
    
    cv.namedWindow("Multi-ArUco SLAM", cv.WINDOW_NORMAL)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Hiba: Kép beolvasása sikertelen!")
                break
            
            # FPS számolása
            fps = slam.calculate_fps()
            
            # Markerek észlelése
            detected_markers = slam.detect_and_estimate_poses(frame)
            
            # Kamera pozíció számítása az előre definiált térkép alapján
            raw_camera_position = slam.update_marker_map(detected_markers)
            
            # Kálmán szűrő alkalmazása a kamera pozíció simításához
            filtered_camera_position = None
            if raw_camera_position is not None:
                slam.camera_positions.append(raw_camera_position)
                filtered_camera_position = slam.smooth_camera_position(raw_camera_position)
            
            # Vizualizáció frissítése
            slam.update_visualization(raw_camera_position, filtered_camera_position, detected_markers)
            
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
                confidence = data.get('confidence', 0)
                error = data.get('reprojection_error', 0)
                
                text = f"ID: {marker_id} | Dist: {distance:.1f}cm | Conf: {confidence:.2f} | Err: {error:.2f}"
                cv.putText(frame, text, tuple(corners[0]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Státusz információk
            status_text = [
                f"FPS: {fps:.1f}",
                f"Markerek száma: {len(slam.marker_world_positions)}",
                f"Referencia marker: {slam.reference_marker_id}",
                f"Kamera pozíciók: {len(slam.camera_positions)}",
                f"Észlelt markerek: {list(detected_markers.keys()) if detected_markers else 'Nincs'}",
            ]
            
            if filtered_camera_position is not None:
                status_text.append(f"Szűrt pozíció: [{filtered_camera_position[0]:.1f}, {filtered_camera_position[1]:.1f}, {filtered_camera_position[2]:.1f}]")
            
            for i, text in enumerate(status_text):
                cv.putText(frame, text, (10, 30 + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow("Multi-ArUco SLAM", frame)
            
            # Bundle adjustment időnként
            frame_count += 1
            if frame_count % 100 == 0:
                slam.bundle_adjustment()
            
            # Kilépés és egyéb parancsok
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
                slam.filtered_camera_positions.clear()
                slam.last_valid_position = None
                slam.kalman_filter = KalmanFilter3D(process_noise=0.1, measurement_noise=1.0)
                print("Kamera trajektória törölve és Kálmán szűrő resetelve!")
            elif key == ord('r'):
                # Újra betöltjük az előre definiált térképet
                slam.load_predefined_map("predefined_marker_map.json")
                slam.camera_positions.clear()
                slam.filtered_camera_positions.clear()
                slam.last_valid_position = None
                slam.kalman_filter = KalmanFilter3D(process_noise=0.1, measurement_noise=1.0)
                print("Előre definiált térkép újratöltve!")
    
    except Exception as e:
        print(f"Hiba történt: {e}")
        import traceback
        traceback.print_exc()
    
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