import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

#EBBEN VANNAK A KÜLÖNBÖZŐ SZÁMÍTÁSI JAVÍTÁSI FORMÁK, AMIK A CHATGPT SZERINT SZÓBA JÖHETNEK


class MultiArUcoSLAM:

    def __init__(self, calib_data_path, marker_size=10.5):
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
        
        # Kamera trajektória
        self.camera_positions = []
        
        # 3D marker pontok marker koordináta-rendszerben
        self.marker_points = np.array([
            [-self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
            [self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],
            [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]
        ], dtype=np.float32)
        
        # Súlyozási paraméterek
        self.distance_weight_factor = 1.5  # Távolság súlyozási faktor
        self.distance_normalization = 30.0  # Távolság normalizálás
        
        # Előre definiált marker térkép betöltése
        self.load_predefined_map("predefined_marker_map.json")
        
        # Vizualizáció inicializálása
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Időzítés az FPS számoláshoz
        self.prev_time = cv.getTickCount()
        self.fps = 0
    
    def create_predefined_map(self, filename="predefined_marker_map.json"):
        #Előre definiált marker térkép létrehozása
        map_data = {
            'reference_marker_id': 0,
            'markers': {},
            'marker_size': self.MARKER_SIZE
        }
        
        # Markerek elhelyezése 2-es sorban növekvő sorrendben, 60 cm távolsággal
        for i in range(2):
            row = i // 2
            col = i % 2
            
            # Pozíciók centiméterben
            x = col * 30
            y = row * 30
            z = 0
            
            # Orientáció
            R = np.eye(3)
            
            map_data['markers'][str(i)] = {
                'rotation_matrix': R.tolist(),
                'translation_vector': [[x], [y], [z]]
            }
        
        # Fájl mentése
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"Előre definiált marker térkép létrehozva: {filename}")
        return map_data
    
    def load_predefined_map(self, filename="predefined_marker_map.json"):
        #Előre definiált marker térkép betöltése
        if not os.path.exists(filename):
            print("Előre definiált marker térkép nem található, létrehozás...")
            self.create_predefined_map(filename)
        
        try:
            with open(filename, 'r') as f:
                map_data = json.load(f)
            
            self.MARKER_SIZE = map_data.get('marker_size', 10.5)
            
            # Marker pozíciók betöltése
            for marker_id_str, data in map_data['markers'].items():
                marker_id = int(marker_id_str)
                R = np.array(data['rotation_matrix'])
                t = np.array(data['translation_vector'])
                self.marker_world_positions[marker_id] = (R, t)
            
            print(f"Előre definiált marker térkép betöltve: {filename}")
            print(f"Betöltött markerek: {list(self.marker_world_positions.keys())}")
            
        except Exception as e:
            print(f"Hiba a marker térkép betöltésekor: {e}")
            map_data = self.create_predefined_map(filename)
            self.load_predefined_map(filename)

    def calibrate_marker_map(self, measured_positions):
        """
        Térkép kalibrálása mért pozíciók alapján
        measured_positions: {marker_id: (x, y, z)} valódi pozíciók cm-ben
        """
        for marker_id, real_pos in measured_positions.items():
            if marker_id in self.marker_world_positions:
                # Frissítsük a marker pozícióját
                t_new = np.array([[real_pos[0]], [real_pos[1]], [real_pos[2]]], dtype=np.float32)
                R = np.eye(3)  # Orientáció marad
                self.marker_world_positions[marker_id] = (R, t_new)
                print(f"Marker {marker_id} pozíció frissítve: ({real_pos[0]}, {real_pos[1]}, {real_pos[2]})")
    
    def calculate_single_marker_position(self, marker_id, data):
        """Egy marker alapján kamera pozíció számítása"""
        rvec = data['rvec']
        tvec = data['tvec'].reshape(3,1)
        
        R_m_c, _ = cv.Rodrigues(rvec)
        R_w_m, t_w_m = self.marker_world_positions[marker_id]
        t_w_m = np.asarray(t_w_m).reshape(3, 1)
        
        R_w_c = R_w_m @ R_m_c.T
        t_w_c = t_w_m - R_w_m @ R_m_c.T @ tvec
        
        return t_w_c.flatten()
    
    def detect_and_estimate_poses(self, frame):
        #Markerek észlelése és pózok becslése
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
                        'distance': np.linalg.norm(tvec),
                        'confidence': confidence,
                        'reprojection_error': reprojection_error
                    }
        
        return detected_markers
    
    def calculate_distance_weight(self, distance):
        #Távolság alapú súly számolása
        # Közelebbi markerek nagyobb súllyal
        distance_weight = 1.0 / max(1.0, (distance * self.distance_weight_factor / self.distance_normalization)**2)
        return distance_weight
    
    def calculate_camera_position_ransac(self, detected_markers, max_iterations=50, inlier_threshold=5.0):
        """RANSAC algoritmus a kiugró értékek kiszűrésére"""
        if not detected_markers or len(detected_markers) < 2:
            return self.calculate_camera_position_basic(detected_markers)

        positions = []
        for marker_id, data in detected_markers.items():
            if marker_id in self.marker_world_positions:
                pos = self.calculate_single_marker_position(marker_id, data)
                positions.append((marker_id, pos))

        best_inliers = []
        best_consensus_pos = None

        for _ in range(max_iterations):
            # Véletlenszerűen válasszunk 2 mintát
            sample = np.random.choice(len(positions), 2, replace=False)
            sample_positions = [positions[i][1] for i in sample]
            
            # Középpont számítása
            consensus_pos = np.mean(sample_positions, axis=0)
            
            # Inlierek keresése
            inliers = []
            for marker_id, pos in positions:
                distance = np.linalg.norm(pos - consensus_pos)
                if distance < inlier_threshold:  # 5 cm határ
                    inliers.append((marker_id, pos))
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_consensus_pos = consensus_pos

        # Ha találtunk elég inliert, használjuk őket
        if len(best_inliers) >= 2:
            print(f"RANSAC: {len(best_inliers)} inlier / {len(positions)} marker")
            inlier_positions = [pos for _, pos in best_inliers]
            return np.mean(inlier_positions, axis=0)
        else:
            # Visszaesés a régi módszerre
            return self.calculate_camera_position_basic(detected_markers)

    def calculate_camera_position_improved(self, detected_markers):
        """Javított súlyozás a konzisztencia alapján"""
        if not detected_markers:
            return None

        # Először számoljuk ki az egyedi pozíciókat
        individual_data = []
        for marker_id, data in detected_markers.items():
            if marker_id in self.marker_world_positions:
                pos = self.calculate_single_marker_position(marker_id, data)
                individual_data.append({
                    'id': marker_id,
                    'position': pos,
                    'distance': data['distance'],
                    'confidence': data.get('confidence', 0.5),
                    'error': data.get('reprojection_error', 0)
                })

        if len(individual_data) < 2:
            return individual_data[0]['position'] if individual_data else None

        # Középpont számítása konzisztencia alapú súlyozással
        all_positions = np.array([d['position'] for d in individual_data])
        centroid = np.mean(all_positions, axis=0)
        
        # Konzisztencia súly: mennyire illeszkedik a többi markerhez
        consistency_weights = []
        for data in individual_data:
            # Alap súly (távolság + konfidencia)
            base_weight = self.calculate_distance_weight(data['distance']) * data['confidence']
            
            # Konzisztencia súly: közelebb van-e a középponthoz
            distance_to_centroid = np.linalg.norm(data['position'] - centroid)
            consistency_weight = 1.0 / (1.0 + distance_to_centroid / 10.0)  # 10 cm normalizálás
            
            # Végső súly
            final_weight = base_weight * consistency_weight
            consistency_weights.append(final_weight)

        # Súlyozott átlag
        weights = np.array(consistency_weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        final_pos = np.zeros(3)
        for i, data in enumerate(individual_data):
            final_pos += data['position'] * weights[i]

        # Kiírás a konzisztencia súlyokkal
        print("\nJAVÍTOTT SÚLYOZÁS:")
        for i, data in enumerate(individual_data):
            dist_to_centroid = np.linalg.norm(data['position'] - centroid)
            print(f"Marker {data['id']}: base_w={self.calculate_distance_weight(data['distance']) * data['confidence']:.3f}, "
                f"consistency_w={1.0/(1.0 + dist_to_centroid/10.0):.3f}, final_w={weights[i]:.3f}")

        return final_pos
    
    def calculate_camera_position_basic(self, detected_markers):
        """Alap súlyozás - részletes kiírással"""
        if not detected_markers:
            return None

        camera_positions = []
        weights = []
        marker_distances = []
        individual_positions = []
        
        print("\n" + "="*60)
        print("EGYEDI MARKER POZÍCIÓK:")
        
        for marker_id, data in detected_markers.items():
            if marker_id not in self.marker_world_positions:
                continue

            individual_position = self.calculate_single_marker_position(marker_id, data)
            camera_positions.append(individual_position)
            individual_positions.append((marker_id, individual_position))
            
            distance = data['distance']
            confidence = data.get('confidence', 0.5)
            distance_weight = self.calculate_distance_weight(distance)
            final_weight = distance_weight * confidence
            
            weights.append(final_weight)
            marker_distances.append((marker_id, distance))
            
            # Tömör kiírás
            print(f"Marker {marker_id:2d}: dist={distance:5.1f}cm, conf={confidence:.2f}, "
                f"weight={final_weight:.3f}, pos=({individual_position[0]:6.1f}, "
                f"{individual_position[1]:6.1f}, {individual_position[2]:6.1f})")
            
        if not camera_positions:
            return None
            
        # Normalizáljuk a súlyokat
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Súlyozott átlag
        cam_pos = np.zeros(3)
        for i, pos in enumerate(camera_positions):
            cam_pos += pos * weights[i]
        
        # Összehasonlítás
        print("\nELTÉRÉSEK A VÉGSŐ POZÍCIÓTÓL:")
        max_diff = 0
        min_diff = float('inf')
        
        for marker_id, pos in individual_positions:
            diff = np.linalg.norm(pos - cam_pos)
            max_diff = max(max_diff, diff)
            min_diff = min(min_diff, diff)
            print(f"Marker {marker_id:2d}: {diff:5.1f}cm eltérés")
        
        # Legközelebbi marker
        if marker_distances:
            closest_marker = min(marker_distances, key=lambda x: x[1])
            print(f"\nLegközelebbi marker: {closest_marker[0]} ({closest_marker[1]:.1f}cm)")
            print(f"Eltérések tartománya: {min_diff:.1f}cm - {max_diff:.1f}cm")
        
        print(f"\nVÉGSŐ POZÍCIÓ: X={cam_pos[0]:6.1f}, Y={cam_pos[1]:6.1f}, Z={cam_pos[2]:6.1f}")
        print("="*60)
        
        return cam_pos

    def calculate_camera_position_robust(self, detected_markers):
        """Robusztus pozíció számítás minden módszerrel"""
        if not detected_markers:
            return None

        # 1. RANSAC próbálkozás
        ransac_result = self.calculate_camera_position_ransac(detected_markers)
        
        # 2. Javított súlyozás
        improved_result = self.calculate_camera_position_improved(detected_markers)
        
        # 3. Egyszerű súlyozás (biztonsági mentés)
        simple_result = self.calculate_camera_position_basic(detected_markers)
        
        # Eredmények összehasonlítása
        positions = [ransac_result, improved_result, simple_result]
        valid_positions = [p for p in positions if p is not None]
        
        if not valid_positions:
            return None
        
        # Válasszuk a legstabilabbat (középső érték)
        if len(valid_positions) >= 3:
            # Medián a három tengelyen külön-külön
            final_pos = np.median(valid_positions, axis=0)
            print(f"ROBUST: RANSAC + Improved + Simple → Medián")
        else:
            final_pos = valid_positions[0]
        
        return final_pos

    def calculate_camera_position(self, detected_markers):
        return self.calculate_camera_position_robust(detected_markers)
        #return self.calculate_camera_position_basic(detected_markers)
        #return self.calculate_camera_position_ransac(detected_markers)
        #return self.calculate_camera_position_improved(detected_markers)
        
    def calculate_fps(self):
        #FPS számolása
        current_time = cv.getTickCount()
        time_diff = (current_time - self.prev_time) / cv.getTickFrequency()
        self.prev_time = current_time
        
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        else:
            self.fps = 0
        
        return self.fps
    
    def update_visualization(self, camera_position=None, detected_markers=None):
        #3D vizualizáció frissítése
        self.ax.clear()
        
        # Markerek megjelenítése
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.marker_world_positions)))
        
        for i, (marker_id, (R, t)) in enumerate(self.marker_world_positions.items()):
            # Marker sarkok világkoordináta-rendszerben
            world_corners = (R @ self.marker_points.T + t).T
            
            # Marker megjelenítése
            self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2], 
                          c=[colors[i]], s=100, alpha=0.7,
                          label=f'Marker {marker_id}')
            
            # Marker kontúr
            corners_plot = np.vstack([world_corners, world_corners[0]])
            self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2], 
                        c=colors[i], linewidth=2, alpha=0.7)
        
        # Kamera pozíció és trajektória
        if camera_position is not None:
            self.camera_positions.append(camera_position)
            
        if self.camera_positions:
            cam_array = np.array(self.camera_positions)
            
            # Trajektória
            if len(self.camera_positions) > 1:
                self.ax.plot(cam_array[:,0], cam_array[:,1], cam_array[:,2], 
                           'b-', alpha=0.8, linewidth=3, label='Kamera trajektória')
            
            # Aktuális kamera pozíció
            self.ax.scatter(cam_array[-1,0], cam_array[-1,1], cam_array[-1,2], 
                           c='blue', s=200, marker='o', label='Kamera')
            
            # Legközelebbi marker kijelölése
            if detected_markers:
                closest_marker_id = min(detected_markers.items(), 
                                      key=lambda x: x[1]['distance'])[0]
                if closest_marker_id in self.marker_world_positions:
                    R, t = self.marker_world_positions[closest_marker_id]
                    world_corners = (R @ self.marker_points.T + t).T
                    
                    # Legközelebbi marker kiemelése
                    self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2], 
                                  c='red', s=150, alpha=1.0,
                                  label=f'Legközelebbi marker {closest_marker_id}')
                    
                    corners_plot = np.vstack([world_corners, world_corners[0]])
                    self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2], 
                                c='red', linewidth=3, alpha=1.0)
        
        # Tengelyek és címkék
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title(f'Multi-ArUco SLAM - {len(self.marker_world_positions)} marker\nFPS: {self.fps:.1f}')
        
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


def main():
    # SLAM rendszer inicializálása
    slam = MultiArUcoSLAM("../calib_data/MultiMatrix.npz", marker_size=10.5)
        
    # Kamera
    cap = cv.VideoCapture(4)
    if not cap.isOpened():
        print("Hiba: Kamera nem elérhető!")
        return
    
    cv.namedWindow("Multi-ArUco SLAM", cv.WINDOW_NORMAL)
    
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
            
            # Kamera pozíció számítása
            camera_position = slam.calculate_camera_position(detected_markers)
            
            # Vizualizáció frissítése
            slam.update_visualization(camera_position, detected_markers)
            
            # Markerek rajzolása az eredeti képre
            if detected_markers:
                # Keressük meg a legközelebbi markert
                closest_marker_id = min(detected_markers.items(), 
                                      key=lambda x: x[1]['distance'])[0]
            
            for marker_id, data in detected_markers.items():
                corners = data['corners'].astype(np.int32)
                
                # Szín beállítása a legközelebbi markernek
                color = (0, 0, 255) if marker_id == closest_marker_id else (0, 255, 255)
                thickness = 6 if marker_id == closest_marker_id else 4
                
                # Marker kontúr
                cv.polylines(frame, [corners], True, color, thickness, cv.LINE_AA)
                
                # Tengelyek rajzolása
                cv.drawFrameAxes(frame, slam.cam_mat, slam.dist_coef, 
                               data['rvec'], data['tvec'], 4, 4)
                
                # Marker ID és távolság
                distance = data['distance']
                confidence = data.get('confidence', 0)
                
                text = f"ID: {marker_id} | Dist: {distance:.1f}cm | Conf: {confidence:.2f}"
                if marker_id == closest_marker_id:
                    text += " [CLOSEST]"
                
                cv.putText(frame, text, tuple(corners[0]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Státusz információk
            status_text = [
                f"FPS: {fps:.1f}",
                f"Markerek száma: {len(slam.marker_world_positions)}",
                f"Észlelt markerek: {list(detected_markers.keys()) if detected_markers else 'Nincs'}",
            ]
            
            if detected_markers:
                closest_marker_id = min(detected_markers.items(), 
                                      key=lambda x: x[1]['distance'])[0]
                closest_distance = detected_markers[closest_marker_id]['distance']
                status_text.append(f"Legközelebbi: Marker {closest_marker_id} ({closest_distance:.1f}cm)")
            
            if camera_position is not None:
                status_text.append(f"Pozíció: [{camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}]")
            
            for i, text in enumerate(status_text):
                cv.putText(frame, text, (10, 30 + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow("Multi-ArUco SLAM", frame)
            
            # Kilépés
            key = cv.waitKey(1)
            if key == ord('q'):
                break
    
    except Exception as e:
        print(f"Hiba történt: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv.destroyAllWindows()
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    main()