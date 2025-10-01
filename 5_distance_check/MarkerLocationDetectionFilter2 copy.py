import cv2 as cv
from cv2 import aruco
import numpy as np


class KalmanFilter3D:
    #3D Kálmán szűrő a kamera pozíció simításához
    
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
        #Kálmán szűrő inicializálása kezdeti pozícióval
        initial_position = initial_position.astype(np.float32)
        for i in range(3):
            self.kf.statePost[i] = initial_position[i]
        self.is_initialized = True
    
    def predict(self):
        #Előrejelzés
        return self.kf.predict()
    
    def update(self, measurement):
        #Frissítés méréssel
        if not self.is_initialized:
            self.init(measurement)
            return measurement
        
        predicted = self.predict()
        measurement_float32 = measurement.astype(np.float32).reshape(-1, 1)
        corrected = self.kf.correct(measurement_float32)
        
        return corrected[:3].flatten()


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
        self.marker_confidence = {}  # {id: confidence_score}
        
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
        self.max_position_change = 100.0  # Maximum megengedett pozíció változás (cm)
    
    def create_predefined_map(self, filename="predefined_marker_map.json"):
        #Előre definiált marker térkép létrehozása 2-es sorban növekvő sorrendben
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
            x = col * 60  # 0 vagy 30 cm
            y = row * 60  # 0, 30, 60, ... cm
            z = 0         # mind a földön
            
            # Orientáció (síkban fekszenek, normál felfelé mutat)
            R = np.eye(3)  # identitás mátrix - nincs forgatás
            
            map_data['markers'][str(i)] = {
                'rotation_matrix': R.tolist(),
                'translation_vector': [[x], [y], [z]],
                'confidence': 1.0  # maximális bizalom
            }
        
        # Fájl mentése
        import json
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"Előre definiált marker térkép létrehozva: {filename}")
        return map_data
    
    def load_predefined_map(self, filename="predefined_marker_map.json"):
        #Előre definiált marker térkép betöltése
        import json
        import os
        
        # Ha a fájl nem létezik, létrehozzuk
        if not os.path.exists(filename):
            print("Előre definiált marker térkép nem található, létrehozás...")
            self.create_predefined_map(filename)
        
        try:
            with open(filename, 'r') as f:
                map_data = json.load(f)
            
            self.reference_marker_id = map_data.get('reference_marker_id')
            self.MARKER_SIZE = map_data.get('marker_size', 10.5)
            
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
                    
                    # Távolság számítása
                    distance = np.linalg.norm(tvec)
                    
                    detected_markers[marker_id] = {
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': corners,
                        'confidence': confidence,
                        'reprojection_error': reprojection_error,
                        'distance': distance
                    }
        
        return detected_markers
    
    def is_position_valid(self, new_position, detected_markers):
        #Ellenőrzi, hogy az új pozíció valid-e (outlier szűrés)
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
        #Marker térkép frissítése – súlyozott átlag a marker konfidenciák alapján
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
            distance_weight = 1.0 / max(1.0, (distance*1.5 / 10.0)**2)  # Normalizálás
            
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
        #Kamera pozíció simítása Kálmán szűrővel
        if raw_position is None:
            # Ha nincs mérés, csak előrejelzést végzünk
            if self.kalman_filter.is_initialized:
                predicted = self.kalman_filter.predict()
                return predicted[:3].flatten()
            return None
        
        # Kálmán szűrő frissítése a mért pozícióval
        smoothed_position = self.kalman_filter.update(raw_position)
        
        return smoothed_position
    
    def draw_markers_on_frame(self, frame, detected_markers):
        #Markerek rajzolása a képkockára
        for marker_id, data in detected_markers.items():
            corners = data['corners'].astype(np.int32)
            
            # Marker kontúr
            cv.polylines(frame, [corners], True, (0, 255, 255), 4, cv.LINE_AA)
            
            # Tengelyek rajzolása
            cv.drawFrameAxes(frame, self.cam_mat, self.dist_coef, 
                           data['rvec'], data['tvec'], 4, 4)
            
            # Marker ID és távolság
            distance = data['distance']
            confidence = data.get('confidence', 0)
            error = data.get('reprojection_error', 0)
            
            # Információk megjelenítése
            info_text = [
                f"ID: {marker_id}",
                f"Tav: {distance:.1f}cm",
                f"Conf: {confidence:.2f}",
                f"Error: {error:.2f}"
            ]
            
            # Szöveg pozíciója - a marker fölött
            text_x = int(corners[0][0])
            text_y = int(corners[0][1]) - 10
            
            for i, text in enumerate(info_text):
                y_pos = text_y - i * 25
                cv.putText(frame, text, (text_x, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def draw_status_info(self, frame, detected_markers, camera_position):
        #Státusz információk megjelenítése
        status_text = [
            f"Markerek szama: {len(detected_markers)}",
            f"Referencia marker: {self.reference_marker_id}",
        ]
        
        if detected_markers:
            status_text.append(f"Észlelt markerek: {list(detected_markers.keys())}")
        
        if camera_position is not None:
            status_text.append(f"Kamera pozíció: [{camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}]")
        
        # Távolság információk
        if detected_markers:
            status_text.append("--- Távolságok ---")
            for marker_id, data in detected_markers.items():
                distance = data['distance']
                status_text.append(f"Marker {marker_id}: {distance:.1f}cm")
        
        # Információk megjelenítése a képen
        for i, text in enumerate(status_text):
            cv.putText(frame, text, (10, 30 + i * 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_camera_position(self, frame):
        #Fő függvény a kamera pozíció meghatározásához
        # Markerek észlelése
        detected_markers = self.detect_and_estimate_poses(frame)
        
        # Kamera pozíció számítása az előre definiált térkép alapján
        raw_camera_position = self.update_marker_map(detected_markers)
        
        # Kálmán szűrő alkalmazása a kamera pozíció simításához
        filtered_camera_position = None
        if raw_camera_position is not None:
            filtered_camera_position = self.smooth_camera_position(raw_camera_position)
        
        # Markerek rajzolása a képre
        self.draw_markers_on_frame(frame, detected_markers)
        
        # Státusz információk rajzolása
        self.draw_status_info(frame, detected_markers, filtered_camera_position)
        
        # Eredmények kiírása konzolra
        if detected_markers:
            print(f"Észlelt markerek: {list(detected_markers.keys())}")
            for marker_id, data in detected_markers.items():
                print(f"  Marker {marker_id}: {data['distance']:.1f}cm távolság")
        
        if filtered_camera_position is not None:
            print(f"Szűrt kamera pozíció: [{filtered_camera_position[0]:.1f}, {filtered_camera_position[1]:.1f}, {filtered_camera_position[2]:.1f}]")
        else:
            print("Nincs elég marker a pozíció meghatározásához")
        
        print("-" * 50)
        
        return filtered_camera_position, detected_markers


def main():
    # SLAM rendszer inicializálása
    slam = MultiArUcoSLAM("../calib_data/MultiMatrix.npz", marker_size=10.5)
    
    # Kamera
    cap = cv.VideoCapture(4)
    if not cap.isOpened():
        print("Hiba: Kamera nem elérhető!")
        return
    
    print("Pozíció meghatározás indítva...")
    print("Kilépéshez nyomd meg az 'q' billentyűt!")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Hiba: Kép beolvasása sikertelen!")
                break
            
            # Kamera pozíció meghatározása
            camera_position, detected_markers = slam.get_camera_position(frame)
            
            # Kép megjelenítése
            cv.imshow("Multi-ArUco SLAM - Kamera nézet", frame)
            
    except Exception as e:
        print(f"Hiba történt: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv.destroyAllWindows()



if __name__ == "__main__":
    main()