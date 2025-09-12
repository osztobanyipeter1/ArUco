import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import json
import time


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
        self.marker_observations = defaultdict(list)  # {id: [(R, t, timestamp)]}
        self.marker_confidence = {}  # {id: confidence_score}
        self.marker_first_seen = {}  # {id: first_detection_time}
        
        # Kamera trajektória
        self.camera_positions = []
        self.camera_orientations = []
        
        # Referencia marker ID
        self.reference_marker_id = None
        
        # 3D marker pontok
        self.marker_points = np.array([
            [-self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],
            [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]
        ], dtype=np.float32)
        
        # Időzítés
        self.start_time = time.time()
        
        # Vizualizáció inicializálása
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def detect_and_estimate_poses(self, frame):
        # Markerek észlelése és pózok becslése
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, _ = self.detector.detectMarkers(gray_frame)
        
        detected_markers = {}
        current_time = time.time() - self.start_time

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
                        'corners': corners,
                        'timestamp': current_time
                    }
        
        return detected_markers
    
    def is_marker_on_wall(self, R):
        """Determine if marker is on wall (vertical) or ground (horizontal)"""
        # Get the normal vector (Z-axis of marker)
        normal = R @ np.array([0, 0, 1])
        
        # Check if marker is mostly vertical (wall) or horizontal (ground)
        if abs(normal[2]) > 0.7:  # Z component is dominant (ground)
            return False
        else:  # Wall marker
            return True
    
    def can_still_adjust(self, marker_id, current_time):
        """Check if marker can still be adjusted (within 5 seconds of first detection)"""
        if marker_id not in self.marker_first_seen:
            return True
        
        time_since_first_seen = current_time - self.marker_first_seen[marker_id]
        return time_since_first_seen <= 5.0  # 5 másodperc az igazítási idő
    
    def smooth_marker_position(self, marker_id, new_R, new_t, current_time):
        """Smooth marker position using multiple observations"""
        if marker_id not in self.marker_observations:
            return new_R, new_t
        
        # Ha már letelt az 5 másodperc, ne simítsunk tovább
        if not self.can_still_adjust(marker_id, current_time):
            return self.marker_world_positions[marker_id]
        
        observations = self.marker_observations[marker_id]
        
        # Keep only recent observations (last 2 seconds)
        recent_observations = [obs for obs in observations if current_time - obs[2] < 2.0]
        
        if not recent_observations:
            return new_R, new_t
        
        # Use weighted average based on confidence and recency
        weights = []
        Rs = []
        ts = []
        
        for R_obs, t_obs, timestamp in recent_observations:
            # Weight based on recency (more recent = higher weight)
            recency_weight = 1.0 / (1.0 + (current_time - timestamp))
            weights.append(recency_weight)
            Rs.append(R_obs)
            ts.append(t_obs)
        
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize weights
        
        # Weighted average for translation
        avg_t = np.zeros_like(new_t)
        for i, t in enumerate(ts):
            avg_t += weights[i] * t
        
        # For rotation, use spherical linear interpolation or average quaternions
        # Simple approach: use the most confident observation for now
        if len(recent_observations) > 3:  # Only smooth after several observations
            most_confident_idx = np.argmax(weights)
            avg_R = Rs[most_confident_idx]
        else:
            avg_R = new_R
        
        return avg_R, avg_t
    
    def update_marker_map(self, detected_markers):
        # Marker térkép frissítése
        if not detected_markers:
            return None
            
        current_time = time.time() - self.start_time
        
        # Referencia marker beállítása (első észlelt marker)
        if self.reference_marker_id is None:
            self.reference_marker_id = min(detected_markers.keys())
            ref_data = detected_markers[self.reference_marker_id]
            R_ref = cv.Rodrigues(ref_data['rvec'])[0]
            t_ref = ref_data['tvec']
            
            # Store initial position
            self.marker_world_positions[self.reference_marker_id] = (np.eye(3), np.zeros((3, 1)))
            self.marker_confidence[self.reference_marker_id] = 1.0
            self.marker_first_seen[self.reference_marker_id] = current_time
            self.marker_observations[self.reference_marker_id].append(
                (np.eye(3), np.zeros((3, 1)), current_time)
            )
            
            print(f"Referencia marker beállítva: {self.reference_marker_id}")
        
        # Kamera pozíció számítása referencia marker alapján
        camera_position = None
        
        if self.reference_marker_id in detected_markers:
            ref_data = detected_markers[self.reference_marker_id]
            R_cam_to_ref = cv.Rodrigues(ref_data['rvec'])[0]
            t_cam_to_ref = ref_data['tvec']
            
            # Kamera pozíció világkoordináta-rendszerben
            camera_position = (-R_cam_to_ref.T @ t_cam_to_ref).flatten()
            
            # Kamera orientáció tárolása
            camera_orientation = R_cam_to_ref.T
            self.camera_orientations.append(camera_orientation)
            
            # Új markerek pozíciójának számítása
            for marker_id, data in detected_markers.items():
                R_cam_to_marker = cv.Rodrigues(data['rvec'])[0]
                t_cam_to_marker = data['tvec']
                
                # Transzformáció világkoordináta-rendszerbe
                R_world_to_marker = R_cam_to_marker @ R_cam_to_ref.T
                t_world_to_marker = R_cam_to_ref.T @ (t_cam_to_marker - t_cam_to_ref)
                
                # Store observation
                self.marker_observations[marker_id].append(
                    (R_world_to_marker.copy(), t_world_to_marker.copy(), current_time)
                )
                
                # Első észlelés időpontjának rögzítése
                if marker_id not in self.marker_first_seen:
                    self.marker_first_seen[marker_id] = current_time
                    print(f"Marker {marker_id} első észlelése: {current_time:.1f}s")
                
                # Ellenőrizzük, hogy még lehet-e igazítani
                if not self.can_still_adjust(marker_id, current_time):
                    # Ha letelt az 5 másodperc, ne frissítsük a pozíciót
                    time_since_first_seen = current_time - self.marker_first_seen[marker_id]
                    if marker_id not in self.marker_world_positions:
                        # Ha még nincs a térképen, de letelt az idő, akkor is adjuk hozzá
                        self.marker_world_positions[marker_id] = (R_world_to_marker, t_world_to_marker)
                        self.marker_confidence[marker_id] = 0.8  # Közepes bizalom
                    continue
                
                # Apply smoothing for existing markers
                if marker_id in self.marker_world_positions:
                    R_smoothed, t_smoothed = self.smooth_marker_position(
                        marker_id, R_world_to_marker, t_world_to_marker, current_time
                    )
                    self.marker_world_positions[marker_id] = (R_smoothed, t_smoothed)
                    
                    # Increase confidence gradually (csak ha még lehet igazítani)
                    if self.can_still_adjust(marker_id, current_time):
                        self.marker_confidence[marker_id] = min(1.0, 
                            self.marker_confidence[marker_id] + 0.05)
                else:
                    # New marker - start with low confidence
                    self.marker_world_positions[marker_id] = (R_world_to_marker, t_world_to_marker)
                    self.marker_confidence[marker_id] = 0.1
                    
                    print(f"Új marker hozzáadva: {marker_id}, confidence: 0.1")
        
        # Ha nincs referencia marker a láthatáron, használj más ismert markereket
        elif any(mid in self.marker_world_positions for mid in detected_markers):
            known_markers = [mid for mid in detected_markers if mid in self.marker_world_positions]
            
            # Use the marker with highest confidence
            best_marker_id = max(known_markers, key=lambda mid: self.marker_confidence.get(mid, 0))
            best_data = detected_markers[best_marker_id]
            
            R_cam_to_marker = cv.Rodrigues(best_data['rvec'])[0]
            t_cam_to_marker = best_data['tvec']
            
            R_world_to_marker, t_world_to_marker = self.marker_world_positions[best_marker_id]
            
            # Kamera pozíció világkoordináta-rendszerben
            R_world_to_cam = R_cam_to_marker.T @ R_world_to_marker.T
            t_world_to_cam = R_world_to_marker.T @ (-t_world_to_marker) - R_cam_to_marker.T @ t_cam_to_marker
            
            camera_position = t_world_to_cam.flatten()
            camera_orientation = R_world_to_cam
            self.camera_orientations.append(camera_orientation)
            
            # Frissítsd a többi marker pozícióját
            for marker_id, data in detected_markers.items():
                if marker_id != best_marker_id:
                    R_cam_to_other = cv.Rodrigues(data['rvec'])[0]
                    t_cam_to_other = data['tvec']
                    
                    # Transzformáció világkoordináta-rendszerbe
                    R_world_to_other = R_cam_to_other @ camera_orientation.T
                    t_world_to_other = camera_orientation.T @ (t_cam_to_other - t_world_to_cam)
                    
                    # Store observation
                    self.marker_observations[marker_id].append(
                        (R_world_to_other.copy(), t_world_to_other.copy(), current_time)
                    )
                    
                    # Első észlelés időpontjának rögzítése
                    if marker_id not in self.marker_first_seen:
                        self.marker_first_seen[marker_id] = current_time
                        print(f"Marker {marker_id} első észlelése: {current_time:.1f}s")
                    
                    # Ellenőrizzük, hogy még lehet-e igazítani
                    if not self.can_still_adjust(marker_id, current_time):
                        continue
                    
                    if marker_id in self.marker_world_positions:
                        # Apply smoothing
                        R_smoothed, t_smoothed = self.smooth_marker_position(
                            marker_id, R_world_to_other, t_world_to_other, current_time
                        )
                        self.marker_world_positions[marker_id] = (R_smoothed, t_smoothed)
                        
                        # Increase confidence gradually (csak ha még lehet igazítani)
                        if self.can_still_adjust(marker_id, current_time):
                            self.marker_confidence[marker_id] = min(1.0, 
                                self.marker_confidence[marker_id] + 0.03)
                    else:
                        # New marker with lower initial confidence
                        self.marker_world_positions[marker_id] = (R_world_to_other, t_world_to_other)
                        self.marker_confidence[marker_id] = 0.05
                        
                        print(f"Új marker hozzáadva (alternatív): {marker_id}, confidence: 0.05")
        
        # Clean up old observations
        for marker_id in list(self.marker_observations.keys()):
            self.marker_observations[marker_id] = [
                obs for obs in self.marker_observations[marker_id] 
                if current_time - obs[2] < 10.0  # Keep observations for visualization
            ]
            
        return camera_position
    
    def bundle_adjustment(self):
        if len(self.marker_world_positions) < 2:
            return
        
        current_time = time.time() - self.start_time
        
        # Egyszerű bundle adjustment: átlagolj több megfigyelést
        for marker_id in self.marker_world_positions:
            # Csak akkor végezz bundle adjustment-ot, ha még lehet igazítani
            if not self.can_still_adjust(marker_id, current_time):
                continue
                
            if len(self.marker_observations[marker_id]) > 1:
                observations = self.marker_observations[marker_id]
                
                # Átlagos transzláció
                avg_t = np.mean([obs[1] for obs in observations], axis=0)
                
                # Használd a legutóbbi rotációt
                latest_R = observations[-1][0]
                
                self.marker_world_positions[marker_id] = (latest_R, avg_t)
    
    def update_visualization(self, camera_position=None, detected_markers=None):
        # 3D vizualizáció frissítése
        self.ax.clear()
        
        # Markerek megjelenítése
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.marker_world_positions)))
        
        for i, (marker_id, (R, t)) in enumerate(self.marker_world_positions.items()):
            confidence = self.marker_confidence.get(marker_id, 0)
            current_time = time.time() - self.start_time
            
            # Szín és átlátszóság a bizalom és időállapot alapján
            if marker_id in self.marker_first_seen:
                time_since_first_seen = current_time - self.marker_first_seen[marker_id]
                if time_since_first_seen > 5.0:
                    alpha = 1.0  # Teljesen átlátszatlan, ha lejárt az idő
                    color = 'blue'  # Kék szín a rögzített markereknek
                    label_suffix = " (RÖGZÍTETT)"
                else:
                    alpha = max(0.3, confidence)
                    color = colors[i]
                    label_suffix = f" (conf: {confidence:.2f})"
            else:
                alpha = max(0.3, confidence)
                color = colors[i]
                label_suffix = f" (conf: {confidence:.2f})"
            
            # Marker sarkok világkoordináta-rendszerben
            world_corners = (R @ self.marker_points.T + t).T
            
            # Marker megjelenítése
            self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2], 
                          c=[color], s=100, alpha=alpha,
                          label=f'Marker {marker_id}{label_suffix}')
            
            # Marker kontúr
            corners_plot = np.vstack([world_corners, world_corners[0]])
            self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2], 
                        c=color, linewidth=2, alpha=alpha)
            
            # Marker normálvektor megjelenítése
            normal = R @ np.array([0, 0, 1])
            center = t.flatten()
            self.ax.quiver(center[0], center[1], center[2],
                          normal[0], normal[1], normal[2],
                          length=5, color=color, alpha=alpha)
            
            # Marker típus szöveg (fal/föld)
            marker_type = "Fal" if self.is_marker_on_wall(R) else "Föld"
            status = "Rögzített" if marker_id in self.marker_first_seen and (current_time - self.marker_first_seen[marker_id] > 5.0) else "Aktív"
            self.ax.text(center[0], center[1], center[2] + 2, 
                        f"{marker_id} ({marker_type}, {status})", fontsize=8)
        
        # Kamera pozíció és trajektória
        if camera_position is not None:
            self.camera_positions.append(camera_position)
            
        if self.camera_positions:
            cam_pos_array = np.array(self.camera_positions)
            
            # Trajektória
            if len(self.camera_positions) > 1:
                self.ax.plot(cam_pos_array[:,0], cam_pos_array[:,1], cam_pos_array[:,2], 
                           'r-', alpha=0.5, linewidth=2, label='Kamera trajektória')
            
            # Aktuális kamera pozíció
            self.ax.scatter(cam_pos_array[-1,0], cam_pos_array[-1,1], cam_pos_array[-1,2], 
                           c='red', s=150, label='Kamera')
            
            # Kamera nézés iránya
            if self.camera_orientations:
                camera_z_axis = self.camera_orientations[-1] @ np.array([0, 0, 1])
                self.ax.quiver(cam_pos_array[-1,0], cam_pos_array[-1,1], cam_pos_array[-1,2],
                            camera_z_axis[0], camera_z_axis[1], camera_z_axis[2],
                            length=8, color='green', label='Kamera nézés iránya')
        
        # Tengelyek és címkék
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        
        current_time = time.time() - self.start_time
        fixed_markers = sum(1 for mid in self.marker_first_seen 
                          if current_time - self.marker_first_seen[mid] > 5.0)
        self.ax.set_title(f'Multi-ArUco SLAM - {len(self.marker_world_positions)} marker ({fixed_markers} rögzített)')
        
        # Referencia marker jelölése
        if self.reference_marker_id is not None:
            self.ax.text2D(0.05, 0.95, f'Referencia marker: {self.reference_marker_id}', 
                          transform=self.ax.transAxes)
        
        self.ax.legend()
        
        # Dinamikus határok
        all_positions = []
        for _, (_, t) in self.marker_world_positions.items():
            all_positions.append(t.flatten())
        if self.camera_positions:
            all_positions.extend(self.camera_positions)
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 20
            self.ax.set_xlim([all_positions[:,0].min()-margin, all_positions[:,0].max()+margin])
            self.ax.set_ylim([all_positions[:,1].min()-margin, all_positions[:,1].max()+margin])
            self.ax.set_zlim([max(0, all_positions[:,2].min()-margin), all_positions[:,2].max()+margin])
        
        plt.draw()
        plt.pause(0.01)
    
    def save_map(self, filename="aruco_map.json"):
        # Marker térkép mentése
        map_data = {
            'reference_marker_id': int(self.reference_marker_id) if self.reference_marker_id else None,
            'markers': {},
            'marker_size': self.MARKER_SIZE
        }
        
        for marker_id, (R, t) in self.marker_world_positions.items():
            map_data['markers'][str(marker_id)] = {
                'rotation_matrix': R.tolist(),
                'translation_vector': t.tolist(),
                'confidence': self.marker_confidence[marker_id],
                'first_seen_time': self.marker_first_seen.get(marker_id, 0)
            }
        
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"Marker térkép elmentve: {filename}")
    
    def load_map(self, filename="aruco_map.json"):
        # Marker térkép betöltése
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
                self.marker_first_seen[marker_id] = data.get('first_seen_time', 0)
            
            print(f"Marker térkép betöltve: {filename}")
            print(f"Betöltött markerek: {list(self.marker_world_positions.keys())}")
            
        except FileNotFoundError:
            print(f"Marker térkép fájl nem található: {filename}")
        except Exception as e:
            print(f"Hiba a marker térkép betöltésekor: {e}")


def main():
    # SLAM rendszer inicializálása
    slam = MultiArUcoSLAM("../calib_data/MultiMatrix.npz", marker_size=10)
    
    # Korábbi térkép betöltése (opcionális)
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
                
                # Állapot szöveg
                current_time = time.time() - slam.start_time
                if marker_id in slam.marker_first_seen:
                    time_since_first_seen = current_time - slam.marker_first_seen[marker_id]
                    if time_since_first_seen > 5.0:
                        status = "RÖGZÍTETT"
                        color = (255, 0, 0)  # Piros
                    else:
                        status = f"Aktív ({5.0 - time_since_first_seen:.1f}s)"
                        color = (0, 255, 0)  # Zöld
                else:
                    status = "ÚJ"
                    color = (0, 255, 0)  # Zöld
                
                text = f"ID: {marker_id} | Dist: {distance:.1f}cm | {status}"
                cv.putText(frame, text, tuple(corners[0]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Státusz információk
            current_time = time.time() - slam.start_time
            fixed_count = sum(1 for mid in slam.marker_first_seen 
                            if current_time - slam.marker_first_seen[mid] > 5.0)
            
            status_text = [
                f"Markerek száma: {len(slam.marker_world_positions)}",
                f"Rögzített markerek: {fixed_count}",
                f"Referencia marker: {slam.reference_marker_id}",
                f"Kamera pozíciók: {len(slam.camera_positions)}",
                f"Frame: {frame_count}"
            ]
            
            for i, text in enumerate(status_text):
                cv.putText(frame, text, (10, 30 + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.imshow("Multi-ArUco SLAM", frame)
            
            # Bundle adjustment időnként (csak aktív markerekre)
            frame_count += 1
            if frame_count % 50 == 0:
                slam.bundle_adjustment()
                print(f"Bundle adjustment végrehajtva (frame {frame_count})")
            
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
                slam.camera_orientations.clear()
                print("Kamera trajektória törölve!")
            elif key == ord('r'):
                # Reset (új referencia marker keresése)
                old_ref = slam.reference_marker_id
                if detected_markers:
                    slam.reference_marker_id = min(detected_markers.keys())
                    print(f"Referencia marker változott: {old_ref} -> {slam.reference_marker_id}")
    
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