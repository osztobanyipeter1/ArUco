import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

#SIMA MINDENMENTES
#SZERINTEM AZ LESZ A BAJ, HOGY AMIKOR TUL MESSZE MEGY A KAMERA, AKKOR A MARKEREK NAGYSÁGA TUL PICI LESZ, ÉS AMIATT A SÚLYOZÁS IS ELKEZD ELCSÚSZNI? MERT SOKKAL KISEBB LESZ A SÚLY?

class MultiArUcoSLAM:

    def __init__(self, calib_data_path, marker_size=10.5):
        # Kalibrációs adatok
        calib_data = np.load(calib_data_path) #kamera kalibráció elérési útja
        self.cam_mat = calib_data["camMatrix"] #kamera mátrix
        self.dist_coef = calib_data["distCoef"] #lencse torzítási együtthatók
        
        self.MARKER_SIZE = marker_size #fizikai marker méret
        
        # ArUco beállítások
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250) #max 250 5x5-ös markerrel dolgozunk
        self.param_markers = aruco.DetectorParameters() #marker detektálási paraméter
        self.detector = aruco.ArucoDetector(self.marker_dict, self.param_markers) #aruco detektor objektum a marker felismeréshez
        
        # Marker pozíciók tárolása világkoordináta-rendszerben
        self.marker_world_positions = {}  # {id: (R, t)} (R=forgási mátrix (3x3), t=transzlációs vektor (3x1)) 
        
        # Kamera trajektória
        self.camera_positions = [] #Ebben tárolódnak a kamera pozíciók
        
        # 3D marker pontok marker koordináta-rendszerben. A marker 4 sarkának 3D koordinátái, a marker középpontja az origoban van, a Z paraméter nulla, mert egy síkban van a marker
        self.marker_points = np.array([
            [-self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],#bal fent
            [self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],#jobb fent
            [self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],#jobb lent
            [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]#bal lent
        ], dtype=np.float32)
        
        # Előre definiált marker térkép betöltése
        self.load_predefined_map("predefined_marker_map.json")
        
        # Vizualizáció inicializálása
        plt.ion() #matplotlib real-time frissítés bekapcsolása
        self.fig = plt.figure(figsize=(12, 10)) #ez a méret
        self.ax = self.fig.add_subplot(111, projection='3d') #3D tengely létrehozása
        
        # Időzítés az FPS számoláshoz
        self.prev_time = cv.getTickCount() #előző képkocka időbélyege
        self.fps = 0 #ez kapja meg a kiszámolt képkocka/másodperc adatot
    
    '''
    def create_predefined_map(self, filename="predefined_marker_map.json"):
        #Előre definiált marker térkép létrehozása
        map_data = {
            'reference_marker_id': 0,
            'markers': {},
            'marker_size': self.MARKER_SIZE #itt megkapja a 10,5-es fizikai marker méretet
        }
        
        # Markerek elhelyezése 2-es sorban növekvő sorrendben, 30 cm távolsággal
        for i in range(20):
            row = i // 2
            col = i % 2
            
            # Pozíciók centiméterben
            x = col * 30
            y = row * 30
            z = 0 #mind vízszintes
            
            # Orientáció
            R = np.eye(3) #semmi forgatás
            
            map_data['markers'][str(i)] = {
                'rotation_matrix': R.tolist(),
                'translation_vector': [[x], [y], [z]]
            }
        
        # Fájl mentése
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"Előre definiált marker térkép létrehozva: {filename}")
        return map_data
    '''
    #Előre definiált marker térkép betöltése
    def load_predefined_map(self, filename="predefined_marker_map.json"):
        
        '''if not os.path.exists(filename):
            print("Előre definiált marker térkép nem található, létrehozás...")
            self.create_predefined_map(filename)'''
        
        try:
            with open(filename, 'r') as f:
                map_data = json.load(f)
            
            self.MARKER_SIZE = map_data.get('marker_size', 10.5) #itt is biztosra megyünk, hogy 10.5 legyen a méret
            
            # Marker pozíciók betöltése
            for marker_id_str, data in map_data['markers'].items():
                marker_id = int(marker_id_str)
                R = np.array(data['rotation_matrix'])
                t = np.array(data['translation_vector'])
                self.marker_world_positions[marker_id] = (R, t)
                        
        except Exception as e:
            print(f"Hiba a marker térkép betöltésekor: {e}")
            '''map_data = self.create_predefined_map(filename)
            self.load_predefined_map(filename)'''
    
    #Markerek észlelése és pózok becslése
    def detect_and_estimate_poses(self, frame):
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #átalakítja a képet szürkeárnyalatosra
        marker_corners, marker_IDs, _ = self.detector.detectMarkers(gray_frame) #detektálja a markeret, és visszaadja a cornereket, ID-t
        
        detected_markers = {} #ebben fogja tárolni a markereket
        
        if marker_corners and marker_IDs is not None:
            for i, (corners, marker_id) in enumerate(zip(marker_corners, marker_IDs.flatten())): #flattennel 2D-ről 1D-re alakít
                corners = corners.reshape(-1, 2).astype(np.float32) #(N,2) formátumba alakítás
                ret, rvec, tvec = cv.solvePnP( #kiszámolja a marker pozícióját és orientációját a kamerához képest
                    self.marker_points, corners, #észlelet marker sarkok (self.marker_points), és 2D pontok (corners)
                    self.cam_mat, self.dist_coef #kamera paraméterek
                )
                
                if ret: #itt a ret megnézi, hogy a ret kapott e sikeres eredményt az előző for ciklusból
                    # Távolság számítása
                    distance = np.linalg.norm(tvec) #vektor hosszának számolása
                    
                    # Konfidencia számítása a marker méretéből
                    confidence = self.calculate_marker_confidence(corners) #kiszámolja a megbízhatósági értéket
                    
                    detected_markers[marker_id] = { #eltároljuk ezt az összes adatot ami fel van sorolva
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': corners,
                        'distance': distance,
                        'confidence': confidence
                    }
        
        return detected_markers #visszaadjuk tároláshoz
    
    #Marker konfidencia számítása a méret alapján
    def calculate_marker_confidence(self, corners):
        
        if corners is None or len(corners) < 4: #ellenőrzi, hogy megvannak-e érvényesen a sarokpontok
            return 0.0
        
        width = np.linalg.norm(corners[0] - corners[1]) #marker méretének számolása a képen
        height = np.linalg.norm(corners[1] - corners[2])
        avg_size = (width + height) / 2
        
        # Minél nagyobb a marker a képen, annál megbízhatóbb
        min_size = 20  # minimum pixel méret
        max_size = 200  # maximum pixel méret
        
        confidence = (avg_size - min_size) / (max_size - min_size) #a távolság, vagyis méret alapján confidence értéket kap a marker
        confidence = np.clip(confidence, 0.1, 1.0)  # 0.1 és 1.0 közé korlátozás
        
        return confidence
    
    #Egyetlen marker alapján kamera pozíció számítása
    def calculate_single_marker_position(self, marker_id, data):
        
        rvec = data['rvec'] #forgási vektor (3x1), marker orientációja a kamerához képest
        tvec = data['tvec'].reshape(3,1) #pozíció vektor (3x1), marker pozíciója a kamerához képest
        
        R_m_c, _ = cv.Rodrigues(rvec) #forgási vektor átalakítása forgási mátrixxá
        R_w_m, t_w_m = self.marker_world_positions[marker_id] #vektor forgási mátrixa és világkoordináta pozíciója
        t_w_m = np.asarray(t_w_m).reshape(3, 1)
        
        R_w_c = R_w_m @ R_m_c.T #R_m_c.T a kamera forgási mátrixa transzponáltan? (@=mátrix szorzás)
        t_w_c = t_w_m - R_w_m @ R_m_c.T @ tvec #t_w_c a kamera poíciója a világkoordináta rendszerben
        
        return t_w_c.flatten() #2D array 1D-re állítása
    
    def calculate_distance_weight(self, distance):
        #Távolság alapú súly számítása
        # Minél közelebb van a marker, annál nagyobb súlyt kap
        max_distance = 200  # cm
        min_distance = 20   # cm
        
        if distance <= min_distance:
            return 1.0
        elif distance >= max_distance:
            return 0.1
        else:
            # Lineáris interpoláció
            weight = 1.0 - (distance - min_distance) / (max_distance - min_distance) * 0.9 #súlyozás, azért 0.9 a szorzó, hogy a legtávolabbi marker is kapjon egy nagyon pici súlyt.
            return max(weight, 0.1)
    
    def calculate_camera_position(self, detected_markers):
        #Kamera pozíció számítása az előre definiált térkép alapján
        if not detected_markers:
            return None

        camera_positions = [] #markerek alapjám számolt kamera pozíció, vektorok [x,y,z]
        weights = [] #számított súlyok (0 és 1 között)
        individual_positions = [] #marker id alapján a pozíciók (marker id és vektor[x,y,z]) (Tuple)
        marker_distances = [] #marker id-k és távolságok (marker id és distance) (Touple)
        
        for marker_id, data in detected_markers.items():
            if marker_id not in self.marker_world_positions: #végigmegy az összes észlelt markeren, majd kihagyjuk azokat a markereket amik nincsenek a térképen
                continue

            individual_position = self.calculate_single_marker_position(marker_id, data) #bemenet marker id és data (rvec, tvec, ...) (marker pozícióból, és kamera-marker relativ pozicióból)
            camera_positions.append(individual_position)
            individual_positions.append((marker_id, individual_position))#kimenet [x,y,z] kamera pozíciós koordináták 
            
            #ez a rész számolja a súlyt
            distance = data['distance'] #távolság kinyerése
            confidence = data.get('confidence', 0.5) #confidence értéke kinyerése
            distance_weight = self.calculate_distance_weight(distance) #távolság súly számítás
            final_weight = distance_weight * confidence #végső súly
            
            weights.append(final_weight) #adatok begyűjtése
            marker_distances.append((marker_id, distance)) #adatok begyűjtése
            
            print(f"Marker {marker_id:2d}: dist={distance:5.1f}cm, conf={confidence:.2f}, "
                f"weight={final_weight:.3f}, pos=({individual_position[0]:6.1f}, "
                f"{individual_position[1]:6.1f}, {individual_position[2]:6.1f})")
        
        if not camera_positions:
            return None
        
        # Súlyozott átlag
        if weights:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalizálás
            
            cam_pos = np.zeros(3) #kamera pozíció inicializálása
            for i, pos in enumerate(camera_positions):
                cam_pos += pos * weights[i]
        else:
            # Egyszerű átlag, ha nincsenek súlyok
            cam_pos = np.mean(camera_positions, axis=0)
        
        # Összegző kiírás
        if individual_positions:
            print(f"Összesen {len(individual_positions)} marker, súlyozott átlag: "
                  f"({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})")
        
        return cam_pos
    
    def calculate_fps(self):
        #FPS számolása
        current_time = cv.getTickCount()
        time_diff = (current_time - self.prev_time) / cv.getTickFrequency() #másodpercekben való megjelenése
        self.prev_time = current_time
        
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        else:
            self.fps = 0
        
        return self.fps
    
    #3D vizualizáció frissítése
    def update_visualization(self, camera_position=None, detected_markers=None):

        self.ax.clear()#törli az előző táblát a frissítéshez
        
        # Markerek megjelenítése
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.marker_world_positions))) #különböző szinekben
        
        for i, (marker_id, (R, t)) in enumerate(self.marker_world_positions.items()): #marker sarkok átszámolása koordináta rendszerben
            # Marker sarkok világkoordináta-rendszerben
            world_corners = (R @ self.marker_points.T + t).T
            
            # Marker megjelenítése
            self.ax.scatter(world_corners[:,0], world_corners[:,1], world_corners[:,2], #3D pontok megjelenítése
                          c=[colors[i]], s=100, 
                          label=f'Marker {marker_id}')
            
            # Marker kontúr
            corners_plot = np.vstack([world_corners, world_corners[0]]) #itt adja hozzá az első pontot a végéhez, hogy bezárja a kontúrt
            self.ax.plot(corners_plot[:,0], corners_plot[:,1], corners_plot[:,2], #vonalak rajzolása a kontúrhoz
                        c=colors[i], linewidth=2)
        
        # Kamera pozíció és trajektória
        if camera_position is not None: #hozzáadja az új kamerapozit a listához
            self.camera_positions.append(camera_position)
            
        if self.camera_positions:
            cam_array = np.array(self.camera_positions) #lista konvertálás NumPy arrayyra
            
            # Trajektória
            if len(self.camera_positions) > 1:
                self.ax.plot(cam_array[:,0], cam_array[:,1], cam_array[:,2], #piros vonallal rajzolja a trajektóriát
                           'r-', alpha=0.8, linewidth=3, label='Kamera trajektória')
            
            # Aktuális kamera pozíció
            self.ax.scatter(cam_array[-1,0], cam_array[-1,1], cam_array[-1,2], #nagy piros pont a kamera aktuális pozíciója
                           c='red', s=200, marker='o', label='Kamera')
        
        # Tengelyek és címkék
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title(f'Multi-ArUco SLAM - {len(self.marker_world_positions)} marker\nFPS: {self.fps:.1f}')
        
        self.ax.legend()
        
        # Dinamikus határok
        all_positions = [] #markerek és kamera pozi gyűjtése
        for _, (_, t) in self.marker_world_positions.items():
            all_positions.append(t.flatten())
        if self.camera_positions:
            all_positions.extend(self.camera_positions)
        
        if all_positions:
            all_positions = np.array(all_positions)
            margin = 20
            self.ax.set_xlim([all_positions[:,0].min()-margin, all_positions[:,0].max()+margin]) #automatikus tengelyhatárok a margó alapján
            self.ax.set_ylim([all_positions[:,1].min()-margin, all_positions[:,1].max()+margin])
            self.ax.set_zlim([max(0, all_positions[:,2].min()-margin), all_positions[:,2].max()+margin])
        
        plt.draw()
        plt.pause(0.01) #ennyi ideje van frissíteni a matplotlibnek a képernyőt


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
                
                # Marker kontúr
                cv.polylines(frame, [corners], True, (0, 255, 255), 4, cv.LINE_AA) #sárga kontúr a marker mögé
                
                # Tengelyek rajzolása
                cv.drawFrameAxes(frame, slam.cam_mat, slam.dist_coef, 
                               data['rvec'], data['tvec'], 4, 4)
                
                # Marker ID és távolság
                distance = data['distance']
                
                text = f"ID: {marker_id} | Dist: {distance:.1f}cm"
                cv.putText(frame, text, tuple(corners[0]), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Státusz információk
            status_text = [
                f"FPS: {fps:.1f}"
            ]
            
            if camera_position is not None:
                status_text.append(f"Pozíció: [{camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}]")
            
            for i, text in enumerate(status_text):
                cv.putText(frame, text, (10, 30 + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow("Multi-ArUco SLAM", frame)
    
    except Exception as e:
        print(f"Hiba: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv.destroyAllWindows()
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    main()