import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os


#Ebben a kódban van az, hogy a legközelebbi marker mindig a referencia, és ha csak minimális különbség van az és más marker között
#akkor 10 centis hibahatárig nem változtat másik markerre.


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
        
        # Referencia marker kezelése
        self.current_reference_marker = None
        self.reference_switch_threshold = 10.0  # 10 cm különbség határ
        self.reference_stability_counter = 0
        self.min_stability_frames = 5  # Minimum frames for stable reference
        
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
                    detected_markers[marker_id] = {
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': corners,
                        'distance': np.linalg.norm(tvec)
                    }
        
        return detected_markers
    
    def select_reference_marker(self, detected_markers):
        #Referencia marker kiválasztása a legközelebbi marker alapján
        if not detected_markers:
            return None
        
        # Rendezzük a markereket távolság szerint
        sorted_markers = sorted(detected_markers.items(), 
                              key=lambda x: x[1]['distance'])
        
        closest_marker_id, closest_data = sorted_markers[0]
        closest_distance = closest_data['distance']
        
        # Ha nincs aktuális referencia, állítsuk be a legközelebbit
        if self.current_reference_marker is None:
            self.current_reference_marker = closest_marker_id
            self.reference_stability_counter = 0
            print(f"Új referencia marker beállítva: {closest_marker_id} (távolság: {closest_distance:.1f}cm)")
            return closest_marker_id
        
        # Ha a jelenlegi referencia még látható
        if self.current_reference_marker in detected_markers:
            current_ref_distance = detected_markers[self.current_reference_marker]['distance']
            
            # Ellenőrizzük, hogy a legközelebbi marker jelentősen közelebb van-e
            distance_diff = current_ref_distance - closest_distance
            
            if distance_diff > self.reference_switch_threshold:
                # Új marker jelentősen közelebb van, váltsunk
                self.current_reference_marker = closest_marker_id
                self.reference_stability_counter = 0
                print(f"Referencia váltás: {self.current_reference_marker} → {closest_marker_id}")
                print(f"Távolságkülönbség: {distance_diff:.1f}cm")
            else:
                # Maradjunk a jelenlegi referenciánál
                self.reference_stability_counter += 1
                closest_marker_id = self.current_reference_marker
        else:
            # Jelenlegi referencia nem látható, váltsunk a legközelebbire
            self.current_reference_marker = closest_marker_id
            self.reference_stability_counter = 0
            print(f"Referencia marker eltűnt, új referencia: {closest_marker_id}")
        
        return closest_marker_id
    
    def calculate_camera_position(self, detected_markers):
        #Kamera pozíció számítása a referencia marker alapján
        if not detected_markers:
            return None

        # Válasszuk ki a referencia markert
        reference_marker_id = self.select_reference_marker(detected_markers)
        
        if reference_marker_id is None:
            return None
        
        # Csak a referencia marker alapján számoljuk a pozíciót
        data = detected_markers[reference_marker_id]
        
        rvec = data['rvec']
        tvec = data['tvec'].reshape(3,1)
        
        # Ellenőrizzük, hogy a referencia marker benne van-e a térképben
        if reference_marker_id not in self.marker_world_positions:
            print(f"Referencia marker {reference_marker_id} nincs a térképen!")
            return None
        
        R_m_c, _ = cv.Rodrigues(rvec)
        R_w_m, t_w_m = self.marker_world_positions[reference_marker_id]
        t_w_m = np.asarray(t_w_m).reshape(3, 1)
        
        # Kamera pozíció számítása
        R_w_c = R_w_m @ R_m_c.T
        t_w_c = t_w_m - R_w_m @ R_m_c.T @ tvec
        
        return t_w_c.flatten()
    
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
            
            # Szín beállítása a referencia markernek
            color = 'red' if marker_id == self.current_reference_marker else colors[i]
            marker_size = 150 if marker_id == self.current_reference_marker else 100
            alpha = 1.0 if marker_id == self.current_reference_marker else 0.7
            
            # Marker megjelenítése
            self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2], 
                          c=[color], s=marker_size, alpha=alpha,
                          label=f'Marker {marker_id}' + (' (REF)' if marker_id == self.current_reference_marker else ''))
            
            # Marker kontúr
            corners_plot = np.vstack([world_corners, world_corners[0]])
            self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2], 
                        c=color, linewidth=2, alpha=alpha)
        
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
        
        # Tengelyek és címkék
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        
        title = f'Multi-ArUco SLAM - {len(self.marker_world_positions)} marker\nFPS: {self.fps:.1f}'
        if self.current_reference_marker is not None:
            title += f'\nReferencia: Marker {self.current_reference_marker}'
        self.ax.set_title(title)
        
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
            for marker_id, data in detected_markers.items():
                corners = data['corners'].astype(np.int32)
                
                # Szín beállítása a referencia markernek
                color = (0, 0, 255) if marker_id == slam.current_reference_marker else (0, 255, 255)
                thickness = 6 if marker_id == slam.current_reference_marker else 4
                
                # Marker kontúr
                cv.polylines(frame, [corners], True, color, thickness, cv.LINE_AA)
                
                # Tengelyek rajzolása
                cv.drawFrameAxes(frame, slam.cam_mat, slam.dist_coef, 
                               data['rvec'], data['tvec'], 4, 4)
                
                # Marker ID és távolság
                distance = data['distance']
                
                text = f"ID: {marker_id} | Dist: {distance:.1f}cm"
                if marker_id == slam.current_reference_marker:
                    text += " [REF]"
                
                cv.putText(frame, text, tuple(corners[0]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Státusz információk
            status_text = [
                f"FPS: {fps:.1f}",
                f"Markerek száma: {len(slam.marker_world_positions)}",
                f"Észlelt markerek: {list(detected_markers.keys()) if detected_markers else 'Nincs'}",
            ]
            
            if slam.current_reference_marker is not None:
                status_text.append(f"Referencia: Marker {slam.current_reference_marker}")
            
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