import cv2
import numpy as np
import time
import os
import json
from database import log_inspection, get_config

# Load custom model classes
try:
    classes = json.load(open("dataset.json"))
except Exception as e:
    classes = ["klip_lh", "klip_rh"]

try:
    net = cv2.dnn.readNetFromONNX("yolov5s.onnx")
except Exception as e:
    print("Failed to load yolov5s.onnx:", e)
    net = None

class VideoCamera(object):
    def __init__(self, operator_id=None):
        config = get_config()
        cam_index = int(config.get('camera_source', 0))
        if os.name == 'nt':
            self.video = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            self.video = cv2.VideoCapture(cam_index)
            
        time.sleep(0.5)
        
        self.last_log_time = 0
        self.operator_id = operator_id
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        self.frame_count = 0
        self.process_every_n_frames = 2  # Dipercepat untuk respon lebih baik
        self.last_boxes = []
        self.last_status = "STANDBY"
        self.last_missing = []
        self.last_max_confidence = 0.0
        self.last_scores = {}
        self.is_detecting = False

        self.detection_history_lh = []
        self.detection_history_rh = []
        self.history_length = 5
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
        """Deteksi Klip Besi/Putih (LH) - Dioptimalkan agar seakurat klip Biru"""
        rx1, ry1, rx2, ry2 = roi_region
        roi_img = image[ry1:ry2, rx1:rx2]
        roi_skin = mask_skin[ry1:ry2, rx1:rx2]
        
        if roi_img.size == 0: return False, 0.0, None
        
        roi_h, roi_w = roi_img.shape[:2]
        
        # 1. Deteksi Putih/Metalik via Grayscale & HSV
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        # Tingkatkan threshold grayscale ke 210 untuk menghindari pantulan pada plastik hitam
        _, mask_bright = cv2.threshold(gray_roi, 210, 255, cv2.THRESH_BINARY)
        
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        # Tingkatkan V min ke 200 agar hanya benda yang benar-benar putih/terang yang lolos
        mask_hsv = cv2.inRange(hsv_roi, np.array([0, 0, 200]), np.array([180, 40, 255]))
        
        combined = cv2.bitwise_or(mask_bright, mask_hsv)
        
        # 2. Morfologi yang lebih halus agar bounding box akurat/ketat pada besi putih
        kernel = np.ones((5, 5), np.uint8)
        mask_final = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        # Menghapus dilasi berlebih agar box tidak membesar
        
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Batas minimal area (0.008 / 0.8%) cukup besar untuk mengabaikan pantulan cahaya kecil tapi tetap mendeteksi besi putih asli
            if area < (roi_w * roi_h * 0.008) or area > (roi_w * roi_h * 0.25): continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(h, 1)
            if aspect_ratio < 0.2 or aspect_ratio > 4.0: continue
            
            # Anti-hand filter
            if np.sum(roi_skin[y:y+h, x:x+w] > 0) / (w * h) > 0.25: continue
            
            solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            score = (solidity * 0.5) + (min(area/(roi_w*roi_h*0.1), 1.0) * 0.5)
            
            # Perketat skor minimal menjadi 0.45 untuk menolak pantulan plastik hitam (glare) yang skornya biasanya ~0.40
            if score > best_score and score > 0.45:
                best_score = score
                best_candidate = {
                    'x': rx1 + x, 'y': ry1 + y, 'w': w, 'h': h,
                    'label': 'klip_lh',
                    'conf': round(min(0.99, 0.88 + score * 0.15), 2)
                }
        
        return (True, best_candidate['conf'], best_candidate) if best_candidate else (False, 0.0, None)

    def _detect_klip_rh_blue(self, image, roi_region, mask_skin):
        """Deteksi Klip Biru (RH)"""
        rx1, ry1, rx2, ry2 = roi_region
        roi_img = image[ry1:ry2, rx1:rx2]
        roi_skin = mask_skin[ry1:ry2, rx1:rx2]
        
        if roi_img.size == 0: return False, 0.0, None
        
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        # Turunkan sedikit batas V dan S agar bagian gelap dari klip biru tetap terdeteksi
        mask_blue = cv2.inRange(hsv_roi, np.array([85, 60, 50]), np.array([135, 255, 255]))
        
        kernel = np.ones((9, 9), np.uint8)
        mask_final = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_final = cv2.dilate(mask_final, np.ones((3, 3), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0.0
        roi_h, roi_w = roi_img.shape[:2]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Batas minimal area (0.005 / 0.5%) cukup untuk mendeteksi klip biru tapi memfilter noise
            if area < (roi_w * roi_h * 0.005) or area > (roi_w * roi_h * 0.25): continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if np.sum(roi_skin[y:y+h, x:x+w] > 0) / (w * h) > 0.25: continue
            
            solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            score = (solidity * 0.5) + (min(area/(roi_w*roi_h*0.1), 1.0) * 0.5)
            
            # Perketat skor minimal
            if score > best_score and score > 0.2:
                best_score = score
                best_candidate = {
                    'x': rx1 + x, 'y': ry1 + y, 'w': w, 'h': h,
                    'label': 'klip_rh',
                    'conf': round(min(0.99, 0.90 + score * 0.1), 2)
                }
        
        return (True, best_candidate['conf'], best_candidate) if best_candidate else (False, 0.0, None)

    def get_frame(self):
        success, image = self.video.read()
        if not success or image is None:
            return None, "Error", [], 0, False, False, False

        self.frame_count += 1
        config = get_config()
        required_parts = {'klip_lh': config.get('target_klip_lh', 1), 'klip_rh': config.get('target_klip_rh', 1)}
        
        image, avg_brightness, is_enhanced = self.enhance_low_light(image, config.get('lux_level', 50))
        is_detecting = getattr(self, 'is_detecting', False)
        
        if not is_detecting:
            self.last_status, self.last_boxes = "STANDBY", []
            self.detection_history_lh, self.detection_history_rh = [], []
            
        if is_detecting and self.frame_count % self.process_every_n_frames == 0:
            detected_parts = {cls: 0 for cls in required_parts.keys()}
            self.last_max_confidence = 0.0
            self.last_boxes = []
            self.last_scores = {}
            img_h, img_w = image.shape[:2]

            # DEFINISI ROI (Sinkron antara deteksi dan visualisasi)
            roi_y1, roi_y2 = int(img_h * 0.10), int(img_h * 0.95)
            roi_lh_coords = (int(img_w * 0.03), roi_y1, int(img_w * 0.42), roi_y2)
            roi_rh_coords = (int(img_w * 0.58), roi_y1, int(img_w * 0.97), roi_y2)

            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            mask_skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

            # PROSES DETEKSI
            f_lh_det, lh_conf, lh_box = self._detect_klip_lh_metal(image, roi_lh_coords, mask_skin)
            f_rh_det, rh_conf, rh_box = self._detect_klip_rh_blue(image, roi_rh_coords, mask_skin)

            # STABILISASI (Voting)
            self.detection_history_lh.append(f_lh_det)
            self.detection_history_rh.append(f_rh_det)
            for hist in [self.detection_history_lh, self.detection_history_rh]:
                if len(hist) > self.history_length: hist.pop(0)
            
            min_v = max(1, int(len(self.detection_history_lh) * 0.6))
            self.stable_detected_lh = sum(self.detection_history_lh) >= min_v
            self.stable_detected_rh = sum(self.detection_history_rh) >= min_v

            if self.stable_detected_lh and lh_box:
                detected_parts['klip_lh'] = 1
                self.last_boxes.append(lh_box)
                self.last_max_confidence = max(self.last_max_confidence, lh_conf)
                self.last_scores['klip_lh'] = f"{lh_conf * 100:.2f}"
            else:
                self.last_scores['klip_lh'] = "0.00"
            
            if self.stable_detected_rh and rh_box:
                detected_parts['klip_rh'] = 1
                self.last_boxes.append(rh_box)
                self.last_max_confidence = max(self.last_max_confidence, rh_conf)
                self.last_scores['klip_rh'] = f"{rh_conf * 100:.2f}"
            else:
                self.last_scores['klip_rh'] = "0.00"

            # EVALUASI STATUS
            self.last_status, self.last_missing = "OK", []
            for part, qty in required_parts.items():
                if qty > 0 and detected_parts[part] < qty:
                    self.last_status = "NG"
                    self.last_missing.append(part)
            
            if self.last_status == "OK" and not self.last_boxes: self.last_max_confidence = 0.99

        # VISUALISASI
        img_h, img_w = image.shape[:2]
        y1_v, y2_v = int(img_h * 0.10), int(img_h * 0.95)
        
        # Draw ROI Areas
        cv2.rectangle(image, (int(img_w*0.03), y1_v), (int(img_w*0.42), y2_v), (255, 255, 0), 2)
        cv2.putText(image, "AREA KLIP BESI (LH)", (int(img_w*0.03), y1_v-10), 0, 0.5, (255, 255, 0), 2)
        cv2.rectangle(image, (int(img_w*0.58), y1_v), (int(img_w*0.97), y2_v), (0, 255, 255), 2)
        cv2.putText(image, "AREA KLIP BIRU (RH)", (int(img_w*0.58), y1_v-10), 0, 0.5, (0, 255, 255), 2)

        for box in self.last_boxes:
            color = (255, 255, 0) if box['label'] == 'klip_lh' else (0, 255, 255)
            cv2.rectangle(image, (box['x'], box['y']), (box['x']+box['w'], box['y']+box['h']), color, 2)
            cv2.putText(image, f"{box['label']} OK ({box['conf']})", (box['x'], box['y']-10), 0, 0.6, (0, 255, 0), 2)
        
        status_color = (0, 255, 0) if self.last_status == "OK" else (0, 0, 255) if self.last_status == "NG" else (128, 128, 128)
        cv2.putText(image, self.last_status, (10, 40), 0, 1.2, status_color, 3)

        # LOGGING & RETURN
        curr_t = time.time()
        if self.last_status != "STANDBY" and (curr_t - self.last_log_time > config.get('log_delay_seconds', 5)):
            os.makedirs('static/data', exist_ok=True)
            path = f"static/data/data_{int(curr_t)}.jpg"
            cv2.imwrite(path, image)
            log_inspection(self.last_status, self.last_max_confidence, path, self.operator_id)
            self.last_log_time = curr_t

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), self.last_status, self.last_missing, int(avg_brightness), is_enhanced, bool(config.get('buzzer_enabled', 1)), bool(config.get('buzzer_ok_enabled', 0))