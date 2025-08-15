import cv2
import numpy as np
import base64
import hashlib
from datetime import datetime
from typing import Tuple, List, Dict, Any


class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_labels = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Neutro', 'Triste', 'Surpresa']

    def _softmax(self, scores: Dict[str, float], temperature: float = 1.55) -> Dict[str, float]:
        values = np.array(list(scores.values()), dtype=np.float64)
      
        values = (values - np.max(values)) / max(1e-6, float(temperature))
        exp = np.exp(values)
        probs = exp / np.sum(exp) if np.sum(exp) > 0 else np.ones_like(exp) / len(exp)
        return {k: float(v) for k, v in zip(scores.keys(), probs)}

    def _extract_features(self, face_img: np.ndarray) -> Dict[str, float]:
        is_color = len(face_img.shape) == 3 and face_img.shape[2] == 3
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if is_color else face_img
        h, w = gray.shape

        gray_eq = cv2.equalizeHist(gray)

        y_eye1, y_eye2 = int(h * 0.25), int(h * 0.45)
        x_eye1, x_eye2 = int(w * 0.2), int(w * 0.8)
        eye_region = gray_eq[y_eye1:y_eye2, x_eye1:x_eye2]

        y_brow1, y_brow2 = int(h * 0.12), int(h * 0.25)
        x_brow1, x_brow2 = int(w * 0.15), int(w * 0.85)
        brow_region = gray_eq[y_brow1:y_brow2, x_brow1:x_brow2]

        y_nose1, y_nose2 = int(h * 0.45), int(h * 0.65)
        x_nose1, x_nose2 = int(w * 0.35), int(w * 0.65)
        nose_region = gray_eq[y_nose1:y_nose2, x_nose1:x_nose2]

        y_mouth1, y_mouth2 = int(h * 0.65), int(h * 0.90)
        x_mouth1, x_mouth2 = int(w * 0.20), int(w * 0.80)
        mouth_region = gray_eq[y_mouth1:y_mouth2, x_mouth1:x_mouth2]

       
        brightness = float(np.mean(gray_eq)) / 255.0
        contrast = float(np.std(gray_eq)) / 128.0
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_norm = float(min(1.0, blur_var / 300.0))

       
        sobelx_m = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely_m = cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1, ksize=3)
        mouth_edge_h = float(np.mean(np.abs(sobelx_m)))
        mouth_edge_v = float(np.mean(np.abs(sobely_m)))
        mouth_open_score = mouth_edge_v / (mouth_edge_h + 1e-6)

        sobelx_e = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely_e = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
        eye_edge_h = float(np.mean(np.abs(sobelx_e)))
        eye_edge_v = float(np.mean(np.abs(sobely_e)))
        eye_open_score = eye_edge_v / (eye_edge_h + 1e-6)

       
        sobelx_b = cv2.Sobel(brow_region, cv2.CV_64F, 1, 0, ksize=3)
        brow_tension = float(np.mean(np.abs(sobelx_b)))

       
        nose_wrinkle = float(cv2.Laplacian(nose_region, cv2.CV_64F).var()) / 220.0

       
        lower_skin = gray_eq[y_mouth1:y_mouth2, int(w * 0.05):int(w * 0.20)]
        lip_contrast = 0.0
        teeth_score = 0.0
        smile_curve = 0.0
        if mouth_region.size > 0 and lower_skin.size > 0:
            lip_contrast = max(0.0, (float(np.mean(lower_skin)) - float(np.mean(mouth_region))) / 60.0)
           
            thr = min(255, max(0, int(np.mean(mouth_region) + 1.2 * np.std(mouth_region))))
            teeth_score = float(np.mean((mouth_region > thr).astype(np.float32)))
           
            mw = mouth_region.shape[1]
            mh = mouth_region.shape[0]
            left_corner = mouth_region[:, : max(1, int(mw * 0.15))]
            right_corner = mouth_region[:, max(1, int(mw * 0.85)) :]
            center_band = mouth_region[:, max(1, int(mw * 0.40)) : min(mw, int(mw * 0.60))]
            if left_corner.size > 0 and right_corner.size > 0 and center_band.size > 0:
                corners_mean = (float(np.mean(left_corner)) + float(np.mean(right_corner))) / 2.0
                center_mean = float(np.mean(center_band))
                smile_curve = max(0.0, (corners_mean - center_mean) / 60.0)

       
        def clamp01(v: float) -> float:
            return float(max(0.0, min(1.5, v)))

       
        low_light_scale = 0.6 + 0.8 * brightness 
        quality_scale = 0.6 + 0.7 * blur_norm    
        mouth_open_adj = mouth_open_score * low_light_scale * quality_scale
        eye_open_adj = eye_open_score * (0.7 + 0.6 * brightness) * quality_scale

        features = {
            'brightness': clamp01(brightness * 1.5),
            'contrast': clamp01(contrast),
            'mouth_open': clamp01(mouth_open_adj),
            'eye_open': clamp01(eye_open_adj),
            'brow_tension': clamp01(brow_tension / 45.0),
            'nose_wrinkle': clamp01(nose_wrinkle),
            'lip_contrast': clamp01(lip_contrast),
            'teeth_score': clamp01(teeth_score * 2.0),
            'smile_curve': clamp01(smile_curve),
            'quality': clamp01(0.5 * contrast + 0.5 * blur_norm),
        }
        return features

    def _score_emotions(self, f: Dict[str, float]) -> Dict[str, float]:
       
       
        surprise_gate = 1.0 if (f['eye_open'] > 0.68 and f['mouth_open'] > 0.66) else 0.45
       
        if f['brightness'] < 0.35:
            surprise_gate *= 0.7
        if f['quality'] < 0.55:
            surprise_gate *= 0.75
        fear_gate = 1.0 if (f['eye_open'] > 0.60 and f['teeth_score'] < 0.10) else 0.75

       
        surprise_penalty = 0.25 * f['lip_contrast'] + 0.25 * f['teeth_score'] + 0.12 * f['brow_tension']

        scores = {
            'Feliz': 1.55 * max(f['smile_curve'], f['lip_contrast']) + 0.95 * f['teeth_score'] + 0.18 * f['mouth_open'] - 0.22 * f['brow_tension'],
            'Surpresa': surprise_gate * (0.78 * f['eye_open'] + 0.56 * f['mouth_open'] + 0.16 * f['contrast']) - surprise_penalty - 0.12,
            'Raiva': 1.15 * f['brow_tension'] + 0.5 * f['contrast'] - 0.25 * f['teeth_score'] - 0.2 * f['lip_contrast'],
            'Nojo': 1.0 * f['nose_wrinkle'] + 0.34 * f['contrast'] - 0.2 * f['mouth_open'],
            'Medo': fear_gate * (0.78 * f['eye_open'] + 0.48 * f['contrast'] + 0.28 * (1.0 - f['teeth_score']) - 0.08 * f['lip_contrast']),
            'Triste': 0.9 * (1.0 - f['brightness']) + 0.5 * (1.0 - f['lip_contrast']) + 0.28 * (1.0 - f['mouth_open']),
            'Neutro': 0.9 * (1.0 - abs(f['mouth_open'] - 0.5)) + 0.9 * (1.0 - abs(f['eye_open'] - 0.5)) - 0.2 * f['contrast'] + 0.08,
        }
       
        expressiveness = max(
            f['lip_contrast'], f['mouth_open'], f['eye_open'], f['brow_tension'], f['nose_wrinkle'], f.get('smile_curve', 0.0)
        )
        scores['Neutro'] += max(0.0, 0.45 - expressiveness * 0.22)
       
        mean_score = float(np.mean(list(scores.values())))
        for k in scores:
            scores[k] -= mean_score
        return scores

    def _detect_emotion_advanced(self, face_img: np.ndarray) -> Tuple[str, float, Dict[str, float], Dict[str, float]]:
        features = self._extract_features(face_img)
        raw_scores = self._score_emotions(features)
        probs = self._softmax(raw_scores, temperature=1.3)

        calib = {
            'Surpresa': 0.85,
            'Medo': 0.95,
            'Feliz': 1.05,
            'Neutro': 1.05,
            'Raiva': 1.0,
            'Nojo': 1.0,
            'Triste': 1.0,
        }
       
        probs = {k: float(probs.get(k, 0.0) * calib.get(k, 1.0)) for k in self.emotion_labels}
        total = float(sum(probs.values())) or 1.0
        probs = {k: float(v / total) for k, v in probs.items()}
        best_emotion = max(probs.items(), key=lambda kv: kv[1])[0]
        confidence = float(probs[best_emotion])
        return best_emotion, confidence, probs, features

    def _detect_emotion_simple(self, face_img: np.ndarray) -> Tuple[str, float]:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        if mean_intensity > 120 and std_intensity > 30:
            return 'Feliz', 0.75
        elif mean_intensity < 80:
            return 'Triste', 0.65
        elif std_intensity > 40:
            return 'Surpresa', 0.60
        elif mean_intensity < 100:
            return 'Neutro', 0.55
        else:
            return 'Neutro', 0.50

    def process_image(self, image_data: str) -> Dict[str, Any]:
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError('Formato de imagem inválido')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.15, 5)

        if len(faces) == 0:
            raise ValueError('Nenhuma face detectada na imagem')

        results = []
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            try:
                emotion, confidence, probs, features = self._detect_emotion_advanced(face_img)
            except Exception:
                emotion, confidence = self._detect_emotion_simple(face_img)
                probs = {lbl: (1.0 if lbl == emotion else 0.0) for lbl in self.emotion_labels}
                features = {}

                
            top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
            results.append({
                'emotion': emotion,
                'confidence': round(confidence, 3),
                'probabilities': {k: round(v, 3) for k, v in probs.items()},
                'top3': [{'emotion': k, 'prob': round(v, 3)} for k, v in top3],
                'features': {k: round(float(v), 3) for k, v in features.items()},
                'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            })

        return {
            'faces_detected': len(faces),
            'emotions': results,
            'message': f'Detectadas {len(faces)} face(s) na imagem',
            'processed_at': datetime.now().isoformat(),
            'image_hash': hashlib.md5(image_data.encode()).hexdigest(),
            'from_cache': False
        }
