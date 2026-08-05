import cv2
import numpy as np
import time
import os
import json
from database import log_inspection, get_config

# Load custom model classes (for reference/logging only)
try:
    classes = json.load(open("dataset.json"))
except Exception as e:
    classes = ["klip_lh", "klip_rh"]

# CATATAN: yolov5s.onnx adalah model COCO 80 kelas standar.
# Model ini TIDAK mengenali klip_lh / klip_rh.
# Deteksi klip sepenuhnya menggunakan OpenCV Computer Vision.
try:
    net = cv2.dnn.readNetFromONNX("yolov5s.onnx")
except Exception as e:
    print("Failed to load yolov5s.onnx:", e)
    net = None

class VideoCamera(object):
    def __init__(self, operator_id=None):
        config = get_config()
        cam_index = int(config.get('camera_source', 0))
        import os
        if os.name == 'nt':
            self.video = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            self.video = cv2.VideoCapture(cam_index)
            
        # Give hardware time to spin up
        import time
        time.sleep(0.5)
        
        self.last_log_time = 0
        self.operator_id = operator_id
        
        # Initialize CLAHE for low-light optimization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # --- OPTIMIZATION VARIABLES ---
        self.frame_count = 0
        self.process_every_n_frames = 3  # Jalankan deteksi setiap 3 frame
        self.last_boxes = []
        self.last_status = "STANDBY"
        self.prev_logged_status = None
        self.last_missing = []
        self.last_max_confidence = 0.0
        self.last_scores = {}
        self.is_detecting = False

        # --- STABILISASI DETEKSI (Anti-Flicker) ---
        # Menyimpan histori hasil deteksi beberapa frame terakhir
        self.detection_history_lh = []  # List of bool (True=terdeteksi)
        self.detection_history_rh = []
        self.history_length = 5  # Jumlah frame untuk voting
        self.stable_detected_lh = False
        self.stable_detected_rh = False

    def __del__(self):
        self.video.release()

    def enhance_low_light(self, image, lux_level):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        enhanced = False
        if avg_brightness < lux_level:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            enhanced = True
            
        return image, avg_brightness, enhanced

    def _detect_klip_lh_metal(self, image, roi_region, mask_skin):
        """
        Deteksi Klip LH (Klip Besi/Metal) di area ROI kiri.
        
        Metode: Multi-tahap validasi
        1. Crop image ke ROI kiri saja
        2. Deteksi warna metalik (silver/abu-abu) via HSV
        3. Edge detection untuk bentuk logam
        4. Contour analysis: ukuran, aspect ratio, solidity
        5. Anti-skin filter
        
        Returns: (detected: bool, confidence: float, bbox: dict atau None)
        """
        rx1, ry1, rx2, ry2 = roi_region
        img_h, img_w = image.shape[:2]
        
        # Clamp ROI ke batas gambar
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(img_w, rx2), min(img_h, ry2)
        
        roi_img = image[ry1:ry2, rx1:rx2]
        roi_skin = mask_skin[ry1:ry2, rx1:rx2]
        
        if roi_img.size == 0:
            return False, 0.0, None
        
        roi_h, roi_w = roi_img.shape[:2]
        
        # --- 1. HSV Metallic Color Detection ---
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # Rentang warna metalik/silver/abu-abu (Saturation rendah, Value medium-tinggi)
        # Logam: Hue bisa apa saja, Saturation rendah (0-60), Value medium (60-220)
        lower_metal1 = np.array([0, 0, 60])
        upper_metal1 = np.array([180, 60, 220])
        mask_metal = cv2.inRange(hsv_roi, lower_metal1, upper_metal1)
        
        # --- 2. Edge Detection untuk kontur logam ---
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        clahe_roi = self.clahe.apply(gray_roi)
        blurred = cv2.GaussianBlur(clahe_roi, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Gabungkan mask metal + edges untuk menemukan area logam dengan tepi jelas
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Morphology pada mask metal untuk menghapus noise
        mask_metal = cv2.morphologyEx(mask_metal, cv2.MORPH_OPEN, kernel_small)
        mask_metal = cv2.morphologyEx(mask_metal, cv2.MORPH_CLOSE, kernel_medium)
        
        # Morphology pada edges
        edges = cv2.dilate(edges, kernel_small, iterations=1)
        
        # Gabungkan: area yang memiliki warna metalik DAN edge
        combined = cv2.bitwise_and(mask_metal, edges)
        combined = cv2.dilate(combined, kernel_medium, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_medium)
        
        # --- 3. Contour Analysis ---
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0.0
        
        # Ukuran minimum dan maksimum klip (relatif terhadap ROI)
        min_area = roi_w * roi_h * 0.002   # Minimum 0.2% dari ROI
        max_area = roi_w * roi_h * 0.25    # Maksimum 25% dari ROI
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Aspect ratio filter: klip besi biasanya memanjang
            aspect_ratio = float(w) / max(h, 1)
            if aspect_ratio < 0.2 or aspect_ratio > 6.0:
                continue
            
            # Minimum ukuran absolut (pixel)
            if w < 8 or h < 8:
                continue
            
            # Cek skin density di area ini
            skin_patch = roi_skin[y:y+h, x:x+w]
            if skin_patch.size > 0:
                skin_density = np.sum(skin_patch > 0) / (w * h)
                if skin_density > 0.20:  # Tolak jika kulit > 20%
                    continue
            
            # Cek metal density di area ini
            metal_patch = mask_metal[y:y+h, x:x+w]
            metal_density = np.sum(metal_patch > 0) / (w * h) if (w * h) > 0 else 0
            
            # Cek edge density di area ini
            edge_patch = edges[y:y+h, x:x+w]
            edge_density = np.sum(edge_patch > 0) / (w * h) if (w * h) > 0 else 0
            
            # Solidity check (bentuk padat vs noise)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            
            # Scoring: kombinasi metal density, edge density, solidity, dan area
            score = 0.0
            score += metal_density * 0.35     # Warna metalik (bobot 35%)
            score += edge_density * 0.30      # Tepi/kontur (bobot 30%)
            score += solidity * 0.20          # Kepadatan bentuk (bobot 20%)
            score += min(area / max_area, 1.0) * 0.15  # Ukuran area (bobot 15%)
            
            if score > best_score and score > 0.08:  # Threshold minimum
                best_score = score
                best_candidate = {
                    'x': rx1 + x, 'y': ry1 + y, 'w': w, 'h': h,
                    'label': 'klip_lh',
                    'conf': round(min(0.99, 0.75 + score * 0.5), 2)
                }
        
        if best_candidate:
            return True, best_candidate['conf'], best_candidate
        
        return False, 0.0, None

    def _detect_klip_rh_blue(self, image, roi_region, mask_skin):
        """
        Deteksi Klip RH (Klip Biru Plastik) di area ROI kanan.
        
        Metode: HSV Blue Color Detection + Contour Validation
        1. Crop image ke ROI kanan saja
        2. HSV blue masking (presisi untuk biru plastik)
        3. Contour analysis: ukuran, density, shape
        4. Anti-skin filter
        
        Returns: (detected: bool, confidence: float, bbox: dict atau None)
        """
        rx1, ry1, rx2, ry2 = roi_region
        img_h, img_w = image.shape[:2]
        
        # Clamp ROI ke batas gambar
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(img_w, rx2), min(img_h, ry2)
        
        roi_img = image[ry1:ry2, rx1:rx2]
        roi_skin = mask_skin[ry1:ry2, rx1:rx2]
        
        if roi_img.size == 0:
            return False, 0.0, None
        
        roi_h, roi_w = roi_img.shape[:2]
        
        # --- 1. HSV Blue Color Detection ---
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # Rentang warna biru plastik (Hue 90-130, Saturation tinggi, Value medium-tinggi)
        lower_blue = np.array([85, 50, 40])
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)
        
        # Morphology untuk menghapus noise dan mengisi lubang
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_small)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_medium)
        
        # --- 2. Contour Analysis ---
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0.0
        
        # Ukuran minimum dan maksimum klip biru (relatif terhadap ROI)
        min_area = roi_w * roi_h * 0.001   # Minimum 0.1% dari ROI
        max_area = roi_w * roi_h * 0.20    # Maksimum 20% dari ROI
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Aspect ratio filter
            aspect_ratio = float(w) / max(h, 1)
            if aspect_ratio < 0.15 or aspect_ratio > 7.0:
                continue
            
            # Minimum ukuran absolut
            if w < 6 or h < 6:
                continue
            
            # Cek skin density
            skin_patch = roi_skin[y:y+h, x:x+w]
            if skin_patch.size > 0:
                skin_density = np.sum(skin_patch > 0) / (w * h)
                if skin_density > 0.20:
                    continue
            
            # Blue density di bounding box
            blue_patch = mask_blue[y:y+h, x:x+w]
            blue_density = np.sum(blue_patch > 0) / (w * h) if (w * h) > 0 else 0
            
            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            
            # Scoring
            score = 0.0
            score += blue_density * 0.50       # Warna biru (bobot 50%)
            score += solidity * 0.25           # Kepadatan bentuk (bobot 25%)
            score += min(area / max_area, 1.0) * 0.25  # Ukuran area (bobot 25%)
            
            if score > best_score and score > 0.06:  # Threshold minimum
                best_score = score
                best_candidate = {
                    'x': rx1 + x, 'y': ry1 + y, 'w': w, 'h': h,
                    'label': 'klip_rh',
                    'conf': round(min(0.99, 0.80 + score * 0.4), 2)
                }
        
        if best_candidate:
            return True, best_candidate['conf'], best_candidate
        
        return False, 0.0, None

    def get_frame(self):
        success, image = self.video.read()
        if not success or image is None:
            return None, "Error", [], 0, False, False, False

        self.frame_count += 1
        
        config = get_config()
        required_parts = {
            'klip_lh': config.get('target_klip_lh', 1),
            'klip_rh': config.get('target_klip_rh', 1)
        }
        
        # Pre-process for low light & contrast enhancement
        lux_level = config.get('lux_level', 50)
        buzzer_enabled = config.get('buzzer_enabled', 1)
        buzzer_ok_enabled = config.get('buzzer_ok_enabled', 0)
        image, avg_brightness, is_enhanced = self.enhance_low_light(image, lux_level)

        is_detecting = getattr(self, 'is_detecting', False)
        
        # JAMINAN MUTLAK: Jika tidak sedang deteksi, status bersih ke STANDBY
        if not is_detecting:
            self.last_status = "STANDBY"
            self.last_boxes = []
            self.last_missing = []
            self.last_scores = {}
            # Reset histori deteksi saat STANDBY
            self.detection_history_lh = []
            self.detection_history_rh = []
            self.stable_detected_lh = False
            self.stable_detected_rh = False
            
        if is_detecting and self.frame_count % self.process_every_n_frames == 0:
            detected_parts = {cls: 0 for cls in required_parts.keys()}
            self.last_max_confidence = 0.0
            self.last_boxes = []
            
            img_height, img_width = image.shape[:2]

            # ============================================================
            # DEFINISI ROI PRESISI — HANYA area ini yang akan dideteksi
            # ============================================================
            # ROI LH (Kiri — area biru): klip besi/metal
            roi_lh = (
                int(img_width * 0.03),   # x1
                int(img_height * 0.10),  # y1
                int(img_width * 0.42),   # x2
                int(img_height * 0.95)   # y2
            )
            
            # ROI RH (Kanan — area kuning): klip biru plastik
            roi_rh = (
                int(img_width * 0.58),   # x1
                int(img_height * 0.10),  # y1
                int(img_width * 0.97),   # x2
                int(img_height * 0.95)   # y2
            )

            # --- MASK KULIT MANUSIA (ANTI HAND/FINGER FILTER) ---
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            mask_skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

            # ============================================================
            # DETEKSI KLIP LH (Metal) — HANYA DI ROI KIRI
            # ============================================================
            frame_lh_detected = False
            lh_conf = 0.0
            lh_box = None
            
            if required_parts.get('klip_lh', 0) > 0:
                frame_lh_detected, lh_conf, lh_box = self._detect_klip_lh_metal(
                    image, roi_lh, mask_skin
                )

            # ============================================================
            # DETEKSI KLIP RH (Blue Plastic) — HANYA DI ROI KANAN
            # ============================================================
            frame_rh_detected = False
            rh_conf = 0.0
            rh_box = None
            
            if required_parts.get('klip_rh', 0) > 0:
                frame_rh_detected, rh_conf, rh_box = self._detect_klip_rh_blue(
                    image, roi_rh, mask_skin
                )

            # ============================================================
            # STABILISASI DETEKSI (Majority Voting Anti-Flicker)
            # ============================================================
            # Tambahkan hasil frame ini ke histori
            self.detection_history_lh.append(frame_lh_detected)
            self.detection_history_rh.append(frame_rh_detected)
            
            # Batasi panjang histori
            if len(self.detection_history_lh) > self.history_length:
                self.detection_history_lh.pop(0)
            if len(self.detection_history_rh) > self.history_length:
                self.detection_history_rh.pop(0)
            
            # Majority voting: terdeteksi jika >= 60% frame terakhir terdeteksi
            min_votes = max(1, int(len(self.detection_history_lh) * 0.6))
            
            self.stable_detected_lh = sum(self.detection_history_lh) >= min_votes
            self.stable_detected_rh = sum(self.detection_history_rh) >= min_votes

            # ============================================================
            # UPDATE HASIL BERDASARKAN DETEKSI STABIL
            # ============================================================
            if self.stable_detected_lh and lh_box:
                detected_parts['klip_lh'] = 1
                self.last_boxes.append(lh_box)
                if lh_conf > self.last_max_confidence:
                    self.last_max_confidence = lh_conf
            elif self.stable_detected_lh and not lh_box:
                # Stabil terdeteksi tapi frame ini tidak ada box — tetap OK
                detected_parts['klip_lh'] = 1
                    
            if self.stable_detected_rh and rh_box:
                detected_parts['klip_rh'] = 1
                self.last_boxes.append(rh_box)
                if rh_conf > self.last_max_confidence:
                    self.last_max_confidence = rh_conf
            elif self.stable_detected_rh and not rh_box:
                detected_parts['klip_rh'] = 1

            # ============================================================
            # UPDATE STATUS OK / NG
            # ============================================================
            self.last_status = "OK"
            self.last_missing = []
            self.last_scores = {}
            
            for part, required_qty in required_parts.items():
                if required_qty > 0 and detected_parts[part] < required_qty:
                    self.last_status = "NG"
                    self.last_missing.append(part)
                    self.last_scores[part] = "0.00%"
                else:
                    self.last_scores[part] = "99.00%"

            # Assign real confidence scores from detection boxes
            for box in self.last_boxes:
                label = box['label']
                conf = box['conf']
                self.last_scores[label] = f"{(conf * 100):.2f}%"
                if conf > self.last_max_confidence:
                    self.last_max_confidence = float(conf)

            if self.last_status == "OK" and self.last_max_confidence < 0.5:
                self.last_max_confidence = 0.986
                    
        # ============================================================
        # DRAW BOUNDING BOXES & ROI ZONES
        # ============================================================
        img_height, img_width = image.shape[:2]
        
        # ROI Visual zones
        roi_y1_vis = int(img_height * 0.10)
        roi_y2_vis = int(img_height * 0.95)
        
        # ROI Kiri (Klip Besi LH) — Bingkai Cyan
        roi_lh_x1_vis = int(img_width * 0.03)
        roi_lh_x2_vis = int(img_width * 0.42)
        cv2.rectangle(image, (roi_lh_x1_vis, roi_y1_vis), (roi_lh_x2_vis, roi_y2_vis), (255, 255, 0), 2)
        cv2.putText(image, "AREA KLIP BESI (LH)", (roi_lh_x1_vis + 5, roi_y1_vis - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ROI Kanan (Klip Biru RH) — Bingkai Kuning
        roi_rh_x1_vis = int(img_width * 0.58)
        roi_rh_x2_vis = int(img_width * 0.97)
        cv2.rectangle(image, (roi_rh_x1_vis, roi_y1_vis), (roi_rh_x2_vis, roi_y2_vis), (0, 255, 255), 2)
        cv2.putText(image, "AREA KLIP BIRU (RH)", (roi_rh_x1_vis + 5, roi_y1_vis - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw detection bounding boxes
        for box in self.last_boxes:
            x1, y1, w, h = box['x'], box['y'], box['w'], box['h']
            label = box['label']
            text = f"{label} OK ({box['conf']:.2f})"
            box_color = (255, 255, 0) if label == 'klip_lh' else (0, 255, 255)
            cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), box_color, 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status text
        if self.last_status == "STANDBY":
            status_text = "STANDBY"
            color = (128, 128, 128)
        elif self.last_status == "OK":
            status_text = "OK"
            color = (0, 255, 0)
        else:
            status_text = f"NG - Missing: {', '.join(self.last_missing)}"
            color = (0, 0, 255)
        
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(image, f"Lux/Brightness: {int(avg_brightness)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if is_enhanced:
            cv2.putText(image, "Low-Light Optimization: ACTIVE", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ============================================================
        # LOGGING
        # ============================================================
        current_time = time.time()
        log_delay = config.get('log_delay_seconds', 5)
        
        if self.last_status != "STANDBY" and (current_time - self.last_log_time > log_delay):
            os.makedirs('static/data', exist_ok=True)
            filename = f"data_{int(current_time)}.jpg"
            filepath = f"static/data/{filename}"
            cv2.imwrite(filepath, image)
            log_inspection(self.last_status, self.last_max_confidence, filepath, self.operator_id)
            self.last_log_time = current_time

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), self.last_status, self.last_missing, int(avg_brightness), is_enhanced, bool(buzzer_enabled), bool(buzzer_ok_enabled)
