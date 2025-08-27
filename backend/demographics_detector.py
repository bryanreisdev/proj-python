import cv2
import numpy as np
import base64
from datetime import datetime
from typing import Dict, List, Any, Tuple
import hashlib
import math
try:
    from config import Config
    _tflite_only_global = bool(getattr(Config, 'GENDER_CONFIG', {}).get('tflite_only', False)) and bool(getattr(Config, 'AGE_CONFIG', {}).get('tflite_only', False))
except Exception:
    _tflite_only_global = False

if not _tflite_only_global:
    try:
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()
        if tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        
        from tensorflow import keras
        from keras_facenet import FaceNet
        from mtcnn import MTCNN
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import pickle
        import os
        
        ML_AVAILABLE = True
        print(f"✅ TensorFlow {tf.__version__} otimizado disponível")
        print("✅ ML completamente carregado com otimizações")
    except ImportError as e:
        ML_AVAILABLE = False
        print(f"❌ ML não disponível: {e}")
        print("🔄 Usando métodos tradicionais")
else:
    ML_AVAILABLE = False
    print("ℹ️ tflite_only=True para AGE e GENDER: não carregando TensorFlow completo")

class AdvancedAgeDetector:
    def __init__(self):
        # Verifica se o modo somente TFLite está ativo
        tflite_only = False
        try:
            from config import Config
            tflite_only = bool(getattr(Config, 'AGE_CONFIG', {}).get('tflite_only', False))
        except Exception:
            pass

        if ML_AVAILABLE and not tflite_only:
            try:
                self.facenet = FaceNet()
                self.age_model = self._load_or_create_age_model()
                self.feature_scaler = StandardScaler()
                self.ml_enabled = True
                print("✅ Detector de Idade ML inicializado")
            except Exception as e:
                print(f"❌ Erro ao carregar modelos de idade ML: {e}")
                self.ml_enabled = False
        else:
            self.ml_enabled = False
        
        self._age_cache = {}

        # TFLite opcional para idade
        self._tflite_age_enabled = False
        try:
            from config import Config
            age_cfg = getattr(Config, 'AGE_CONFIG', {})
            if age_cfg.get('use_tflite_age', False):
                import os
                # Fallback: tenta tflite_runtime e, se indisponível, usa TensorFlow Lite
                InterpreterClass = None
                try:
                    import tflite_runtime.interpreter as tflite
                    InterpreterClass = tflite.Interpreter
                except Exception:
                    try:
                        from tensorflow.lite.python.interpreter import Interpreter as TfInterpreter
                        InterpreterClass = TfInterpreter
                    except Exception:
                        try:
                            from tensorflow.lite import Interpreter as TfInterpreter
                            InterpreterClass = TfInterpreter
                        except Exception as _e:
                            raise ImportError(_e)
                self._tflite_age_path = age_cfg.get('tflite_age_model_path', 'models/age_regression.tflite')
                self._tflite_age_threads = int(age_cfg.get('tflite_age_threads', 2))
                # Gera modelo TFLite simples se arquivo estiver ausente/vazio
                def _ensure_tflite_age_model():
                    try:
                        need_create = (not os.path.exists(self._tflite_age_path)) or (os.path.getsize(self._tflite_age_path) == 0)
                        if need_create:
                            if ML_AVAILABLE:
                                keras_model = self._create_age_regression_model()
                                import tensorflow as tf
                                converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
                                tflite_model = converter.convert()
                                os.makedirs(os.path.dirname(self._tflite_age_path) or '.', exist_ok=True)
                                with open(self._tflite_age_path, 'wb') as f:
                                    f.write(tflite_model)
                            else:
                                print("Aviso: Modelo TFLite de idade ausente/vazio e TensorFlow indisponível; pulando auto-geração. Forneça o arquivo .tflite em models/.")
                    except Exception as _e:
                        print(f"Falha ao gerar modelo TFLite de idade: {_e}")
                _ensure_tflite_age_model()
                if os.path.exists(self._tflite_age_path) and os.path.getsize(self._tflite_age_path) > 0:
                    self._tflite_age_interpreter = InterpreterClass(model_path=self._tflite_age_path, num_threads=self._tflite_age_threads)
                    self._tflite_age_interpreter.allocate_tensors()
                    try:
                        a_in = self._tflite_age_interpreter.get_input_details()[0]
                        a_out = self._tflite_age_interpreter.get_output_details()[0]
                        print(f"[TFLite Age] loaded: {self._tflite_age_path}")
                        print(f"[TFLite Age] input shape={a_in.get('shape')} dtype={a_in.get('dtype')} output shape={a_out.get('shape')} dtype={a_out.get('dtype')}")
                    except Exception:
                        pass
                    self._tflite_age_enabled = True
        except Exception as e:
            print(f"TFLite idade desabilitado: {e}")
        
       
        self._skin_tone_gamma_min = 1.1
        self._skin_tone_gamma_max = 1.6
        self._skin_tone_dark_L_threshold = 110

        # Log de configuração carregada (diagnóstico)
        try:
            from config import Config
            age_cfg = getattr(Config, 'AGE_CONFIG', {})
            print("[AGE_CONFIG] child_cap_enabled=", age_cfg.get('child_cap_enabled'))
            print("[AGE_CONFIG] child_blend_enabled=", age_cfg.get('child_blend_enabled'))
            print("[AGE_CONFIG] ensemble_method_weights=", age_cfg.get('ensemble_method_weights'))
            print("[AGE_CONFIG] thresholds:", {
                'child_prob_vote_threshold': age_cfg.get('child_prob_vote_threshold'),
                'elderly_prob_vote_threshold': age_cfg.get('elderly_prob_vote_threshold'),
                'child_indicators_vote_threshold': age_cfg.get('child_indicators_vote_threshold'),
                'elderly_indicators_vote_threshold': age_cfg.get('elderly_indicators_vote_threshold'),
            })
        except Exception as _:
            pass

    def _align_by_eyes(self, face_img: np.ndarray) -> np.ndarray:
        """Alinha o rosto nivelando os olhos usando Haar Cascade de olhos.
        Retorna a imagem original em caso de falha para manter robustez.
        """
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            # Usa haarcascade de olhos do OpenCV
            eye_cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_eye.xml'
            if not eye_cascade_path or not os.path.exists(eye_cascade_path):
                return face_img
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
            if eyes is None or len(eyes) < 2:
                return face_img
            # Ordena por área e pega os dois maiores
            eyes_sorted = sorted(eyes, key=lambda r: r[2] * r[3], reverse=True)[:2]
            # Calcula centros
            centers = []
            for (x, y, w, h) in eyes_sorted:
                centers.append((x + w * 0.5, y + h * 0.5))
            if len(centers) < 2:
                return face_img
            (x1, y1), (x2, y2) = centers[0], centers[1]
            # Garante que x1 < x2
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            dy = y2 - y1
            dx = x2 - x1
            if abs(dx) < 1e-6:
                return face_img
            angle = math.degrees(math.atan2(dy, dx))
            # Rotação inversa para nivelar (subtrai o ângulo)
            h, w = face_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            aligned = cv2.warpAffine(face_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            return aligned
        except Exception:
            return face_img

    def _preprocess_for_age(self, face_img: np.ndarray) -> np.ndarray:
        """Pipeline de pré-processamento para idade: alinhamento por olhos + correção de iluminação.
        """
        try:
            img = self._align_by_eyes(face_img)
            img = self._preprocess_for_skin_tone(img)
            return img
        except Exception:
            return face_img

    def _load_or_create_age_model(self):
        model_path = 'models/age_model.h5'
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            return self._create_age_regression_model()

    def _preprocess_for_skin_tone(self, face_img: np.ndarray) -> np.ndarray:
        """Melhora contraste e iluminação para peles escuras mantendo tons naturais.
        - Equaliza L (LAB) com CLAHE
        - Aplica gama adaptativa se L médio for baixo
        - Faz gray-world para corrigir balanço de branco simples
        """
        try:
            if face_img is None or face_img.size == 0:
                return face_img
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_L = float(np.mean(l))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b])
            img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            if mean_L < self._skin_tone_dark_L_threshold:
                frac = max(0.0, min(1.0, (self._skin_tone_dark_L_threshold - mean_L) / self._skin_tone_dark_L_threshold))
                gamma = self._skin_tone_gamma_min + (self._skin_tone_gamma_max - self._skin_tone_gamma_min) * frac
                inv_gamma = 1.0 / max(1e-6, gamma)
                table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype('uint8')
                img_eq = cv2.LUT(img_eq, table)
            bgr = img_eq.astype(np.float32)
            means = np.mean(bgr.reshape(-1, 3), axis=0) + 1e-6
            gray_mean = float(np.mean(means))
            gains = gray_mean / means
            bgr *= gains
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)
            return bgr
        except Exception:
            return face_img
    
    def _create_age_regression_model(self):
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(160, 160, 3)
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='mae',
            metrics=['mae', 'mse']
        )
        return model
    
    def detect_age_advanced(self, face_img: np.ndarray) -> Dict[str, Any]:
        if not self.ml_enabled:
            from config import Config
            use_ens = bool(getattr(Config, 'AGE_CONFIG', {}).get('use_ensemble_with_tflite', True))
            if getattr(self, '_tflite_age_enabled', False) and not use_ens:
                try:
                    # Pré-processamento completo (alinhamento + CLAHE)
                    face_img_proc = self._preprocess_for_age(face_img)
                    return self._predict_with_tflite_age(face_img_proc)
                except Exception as e:
                    print(f"Erro TFLite idade: {e}")
                    return self._predict_traditional(face_img)
            # fallback antigo: combina para robustez
            predictions: List[Dict[str, Any]] = []
            # Pré-processamento completo
            face_img_proc = self._preprocess_for_age(face_img)
            if getattr(self, '_tflite_age_enabled', False):
                try:
                    predictions.append(self._predict_with_tflite_age(face_img_proc))
                except Exception as e:
                    print(f"Erro TFLite idade: {e}")
            try:
                predictions.append(self._predict_traditional(face_img_proc))
            except Exception as e:
                print(f"Erro tradicional idade: {e}")
            try:
                predictions.append(self._predict_with_texture_analysis(face_img_proc))
            except Exception as e:
                print(f"Erro textura idade: {e}")
            if predictions:
                return self._ensemble_age_predictions(predictions)
            return {'method': 'traditional', 'estimated_age': 30, 'confidence': 0.3}
        
        try:
            # Pré-processamento robusto para tons de pele escuras
            face_img_proc = self._preprocess_for_skin_tone(face_img)
            predictions = []
            # TFLite age (se disponível) entra como método adicional
            if getattr(self, '_tflite_age_enabled', False):
                try:
                    predictions.append(self._predict_with_tflite_age(face_img_proc))
                except Exception as e:
                    print(f"Erro TFLite idade: {e}")
            predictions.append(self._predict_with_cnn(face_img_proc))
            predictions.append(self._predict_with_facenet(face_img_proc))
            predictions.append(self._predict_traditional(face_img_proc))
            predictions.append(self._predict_with_texture_analysis(face_img_proc))
            return self._ensemble_age_predictions(predictions)
        except Exception as e:
            print(f"Erro no detector avançado de idade: {e}")
            return self._predict_traditional(face_img)

    def _predict_with_tflite_age(self, face_img: np.ndarray) -> Dict[str, Any]:
        import numpy as np
        try:
            # Gate de qualidade: tamanho mínimo, nitidez e brilho
            try:
                from config import Config
                min_size = int(getattr(Config, 'FACE_DETECTION_CONFIG', {}).get('min_face_size', 40))
            except Exception:
                min_size = 40

            h, w = face_img.shape[:2]
            if min(h, w) < max(32, min_size):
                return {'method': 'tflite_age_regression', 'estimated_age': 30, 'confidence': 0.35}

            gray_q = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            sharp_q = float(cv2.Laplacian(gray_q, cv2.CV_64F).var())
            mean_q = float(np.mean(gray_q))
            if sharp_q < 20.0 or mean_q < 35.0:
                return {'method': 'tflite_age_regression', 'estimated_age': 30, 'confidence': 0.4}

            # Ler dinamicamente o tamanho de entrada do modelo
            inp = self._tflite_age_interpreter.get_input_details()[0]
            out = self._tflite_age_interpreter.get_output_details()[0]

            input_shape = inp.get('shape', [1, 160, 160, 3])
            input_h = int(input_shape[1])
            input_w = int(input_shape[2])
            in_dtype = inp.get('dtype')

            # BGR->RGB e base 0..1 (supondo já pré-processado)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            base_resized = cv2.resize(face_rgb, (input_w, input_h)).astype('float32') / 255.0

            def run_inference(x_batch: np.ndarray) -> np.ndarray:
                self._tflite_age_interpreter.set_tensor(inp['index'], x_batch)
                self._tflite_age_interpreter.invoke()
                y = self._tflite_age_interpreter.get_tensor(out['index'])
                return np.squeeze(y)

            from config import Config
            age_cfg = getattr(Config, 'AGE_CONFIG', {})
            out_type = age_cfg.get('tflite_age_output_type', 'auto')


            # Test-time augmentation leve: shifts, center e flip
            aug_images = []
            base = base_resized.copy()
            flips = [False, True]
            shifts = [(0, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)]
            for do_flip in flips:
                img = np.flip(base, axis=1) if do_flip else base
                for dy, dx in shifts:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted = cv2.warpAffine((img * 255).astype('uint8'), M, (input_w, input_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    aug_images.append(shifted.astype('float32') / 255.0)

            def batch_predict(img_list: List[np.ndarray]) -> List[np.ndarray]:
                preds = []
                for im in img_list:
                    if in_dtype and 'uint8' in str(in_dtype).lower():
                        q_scale = float(inp.get('quantization_parameters', {}).get('scales', [1.0])[0] or 1.0)
                        q_zero = int(inp.get('quantization_parameters', {}).get('zero_points', [0])[0] or 0)
                        x_uint8 = np.clip(np.round(im / max(q_scale, 1e-8) + q_zero), 0, 255).astype('uint8')
                        x_uint8 = np.expand_dims(x_uint8, 0)
                        y = run_inference(x_uint8)
                    else:
                        x01 = np.expand_dims(im, 0)
                        x11 = np.expand_dims(im * 2.0 - 1.0, 0)
                        y01 = run_inference(x01)
                        y11 = run_inference(x11)
                        def conf_of(vec: np.ndarray) -> float:
                            v = vec.reshape(-1).astype('float32')
                            if v.size == 1:
                                return float(abs(v[0] - 0.5))
                            vv = v - float(np.max(v))
                            e = np.exp(vv)
                            p = e / max(1e-8, float(np.sum(e)))
                            return float(np.max(p))
                        y = y01 if conf_of(y01) >= conf_of(y11) else y11
                    preds.append(np.array(y))
                return preds

            preds = batch_predict(aug_images)

            first_pred = preds[0]
            first_size = int(np.size(first_pred))
            if out_type == 'regression' or (out_type == 'auto' and first_size == 1):
                # Converte lista de saídas para vetor
                y_vec = np.array([np.squeeze(p).astype('float32') for p in preds]).reshape(-1)
                reg_scale = float(age_cfg.get('tflite_age_regression_scale', 100.0))
                # Reescala se necessário
                if np.all((0.0 <= y_vec) & (y_vec <= 1.0)):
                    y_vec = y_vec * reg_scale
                # Idade final: mediana para robustez
                predicted_age = float(np.median(y_vec))
                predicted_age = max(1, min(99, int(round(predicted_age))))
                # Dispersão entre TTA
                tta_std = float(np.std(y_vec))
                # Confiança base pela distância ao prior + penalidade pela dispersão
                age_variance = abs(predicted_age - 35) / 35.0
                conf_base = max(0.55, 1.0 - age_variance * 0.28)
                conf_disp = max(0.0, 1.0 - (tta_std / 6.0))  # std >= 6 anos derruba confiança
                confidence = max(0.5, min(0.98, 0.5 * conf_base + 0.5 * conf_disp))
                # Ajuste pela qualidade da imagem (nitidez e brilho)
                try:
                    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    sharp_norm = max(0.0, min(1.0, (sharp - 30.0) / 120.0))
                    mean_b = float(np.mean(gray))
                    bright_norm = max(0.0, min(1.0, (mean_b - 60.0) / 80.0))
                    qual = 0.6 * sharp_norm + 0.4 * bright_norm
                    confidence = max(0.5, min(0.98, confidence * (0.7 + 0.3 * qual)))
                except Exception:
                    pass
                return {
                    'method': 'tflite_age_regression',
                    'estimated_age': predicted_age,
                    'confidence': float(confidence)
                }
            else:
                # Média de probabilidades nas TTA e mediana do centro
                vecs = [np.array(p).reshape(-1).astype('float32') for p in preds]
                # Softmax por amostra
                probs_list = []
                for v in vecs:
                    vv = v - float(np.max(v))
                    e = np.exp(vv)
                    probs_list.append(e / max(1e-8, float(np.sum(e))))
                probs = np.mean(np.stack(probs_list, axis=0), axis=0)
                class_idx = int(np.argmax(probs))
                class_prob = float(np.max(probs))
                class_ranges = age_cfg.get('tflite_age_class_ranges', [
                    "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"
                ])
                age_range = class_ranges[class_idx] if 0 <= class_idx < len(class_ranges) else "25-32"
                def _mid_of_range(r: str) -> int:
                    if '+' in r:
                        return int(r.replace('+', '').strip()) + 5
                    a, b = r.split('-')
                    return (int(a) + int(b)) // 2
                estimated_age = _mid_of_range(age_range)
                # Consistência entre TTA: desvio das probabilidades do topo
                top_probs = [float(np.max(p)) for p in probs_list]
                disp = float(np.std(np.array(top_probs)))
                conf_disp = max(0.0, 1.0 - disp / 0.2)
                confidence = float(max(0.6, min(0.95, 0.5 * class_prob + 0.5 * conf_disp)))
                return {
                    'method': 'tflite_age_classes',
                    'estimated_age': int(estimated_age),
                    'age_range': age_range,
                    'confidence': confidence
                }
        except Exception as e:
            print(f"Erro no TFLite age: {e}")
            return {'method': 'tflite_age_regression', 'estimated_age': 30, 'confidence': 0.3}
    
    def _predict_with_cnn(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            predicted_age = self.age_model.predict(face_expanded, verbose=0)[0][0]
            predicted_age = max(1, min(99, int(predicted_age)))
            
            age_variance = abs(predicted_age - 35) / 35.0
            
            confidence = max(0.65, 1.0 - age_variance * 0.28)
            return {
                'method': 'cnn_age_regression',
                'estimated_age': predicted_age,
                'confidence': float(confidence)
            }
        except Exception as e:
            print(f"Erro CNN idade: {e}")
            return {'method': 'cnn_age', 'estimated_age': 30, 'confidence': 0.3}
    
    def _predict_with_facenet(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            embedding = self.facenet.embeddings(face_expanded)[0]
            
            advanced_features = self._extract_advanced_age_features(embedding, face_img)
            estimated_age = self._calculate_age_with_ml_enhancement(advanced_features)
            
            child_probability = self._analyze_child_indicators(advanced_features, face_img)
            youth_probability = self._analyze_youth_indicators(advanced_features, face_img)
            elderly_probability = self._analyze_elderly_indicators(advanced_features, face_img)
            
            if child_probability > 0.5:
                estimated_age = self._refine_child_age(estimated_age, child_probability, advanced_features)
            elif youth_probability > 0.5:
                estimated_age = self._refine_youth_age(estimated_age, youth_probability, advanced_features)
            elif elderly_probability > 0.5:
                estimated_age = self._refine_elderly_age(estimated_age, elderly_probability, advanced_features)
            
            confidence = 0.9 if (child_probability > 0.7 or youth_probability > 0.7 or elderly_probability > 0.7) else 0.85
            
            return {
                'method': 'facenet_age_enhanced',
                'estimated_age': max(3, min(99, int(estimated_age))),
                'confidence': float(confidence),
                'child_probability': float(child_probability),
                'youth_probability': float(youth_probability),
                'elderly_probability': float(elderly_probability),
                'features': advanced_features
            }
        except Exception as e:
            print(f"Erro FaceNet idade avançado: {e}")
            return {'method': 'facenet_age_enhanced', 'estimated_age': 30, 'confidence': 0.3}
    
    def _extract_age_features_from_embedding(self, embedding: np.ndarray) -> Dict[str, float]:
        features = {
            'skin_texture': float(np.std(embedding[0:100])),
            'facial_structure': float(np.mean(embedding[100:200])),
            'eye_region': float(np.std(embedding[200:300])),
            'mouth_region': float(np.mean(embedding[300:400])),
            'overall_aging': float(np.var(embedding[400:512]))
        }
        return features
    
    def _extract_advanced_age_features(self, embedding: np.ndarray, face_img: np.ndarray) -> Dict[str, float]:
        """Extrai características avançadas para detecção de idade, especialmente crianças e idosos"""
        try:
            # Características avançadas do embedding
            features = {
                'skin_smoothness': float(np.mean(embedding[0:50])),
                'facial_maturity': float(np.mean(embedding[50:100])),
                'bone_development': float(np.mean(embedding[100:150])),
                'eye_socket_depth': float(np.std(embedding[150:200])),
                'cheek_fullness': float(np.mean(embedding[200:250])),
                'jaw_definition': float(np.std(embedding[250:300])),
                'forehead_prominence': float(np.mean(embedding[300:350])),
                'facial_proportions': float(np.var(embedding[350:400])),
                'skin_elasticity_markers': float(np.std(embedding[400:450])),
                'aging_patterns': float(np.mean(embedding[450:512]))
            }
            
            # Análise adicional da imagem
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            h, w = gray.shape
            
            # Características geométricas para crianças
            face_roundness = w / max(1, h)  # Crianças têm faces mais redondas
            features['face_roundness'] = float(face_roundness)
            
            # Análise de textura para idosos
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features['texture_complexity'] = float(laplacian_var)
            
            # Análise de brightness (crianças têm pele mais clara)
            avg_brightness = np.mean(gray)
            features['average_brightness'] = float(avg_brightness)
            
            # Análise de contraste (idosos têm mais contraste)
            contrast = np.std(gray)
            features['contrast_level'] = float(contrast)
            
            return features
        except Exception as e:
            print(f"Erro na extração avançada de características de idade: {e}")
            return {f'age_feature_{i}': 0.0 for i in range(15)}
    
    def _calculate_age_with_ml_enhancement(self, features: Dict[str, float]) -> float:
        """Cálculo avançado de idade usando ML"""
        try:
            child_weights = {
                'skin_smoothness': 0.25,
                'cheek_fullness': 0.22,
                'face_roundness': 0.20,
                'average_brightness': 0.15,
                'bone_development': -0.15,
                'jaw_definition': -0.12,
                'eye_socket_depth': -0.08,
                'forehead_prominence': -0.07
            }
            
            youth_weights = {
                'skin_smoothness': 0.20,
                'facial_maturity': 0.25,
                'bone_development': 0.20,
                'jaw_definition': 0.15,
                'cheek_fullness': 0.10,
                'eye_socket_depth': 0.10
            }
            
            elderly_weights = {
                'aging_patterns': 0.30,
                'texture_complexity': 0.25,
                'skin_elasticity_markers': 0.20,
                'contrast_level': 0.15,
                'eye_socket_depth': 0.10
            }
            
            adult_weights = {
                'facial_maturity': 0.30,
                'bone_development': 0.25,
                'facial_proportions': 0.20,
                'jaw_definition': 0.15,
                'eye_socket_depth': 0.10
            }
            
            child_score = 0.0
            youth_score = 0.0
            elderly_score = 0.0
            adult_score = 0.0
            
            for feature, value in features.items():
                if not np.isnan(value) and not np.isinf(value):
                    normalized_value = max(0, min(1, value))
                    
                    if feature in child_weights:
                        weight = child_weights[feature]
                        if weight > 0:
                            child_score += normalized_value * weight
                        else:
                            child_score += (1.0 - normalized_value) * abs(weight)
                    
                    if feature in youth_weights:
                        youth_score += normalized_value * youth_weights[feature]
                    
                    if feature in elderly_weights:
                        elderly_score += normalized_value * elderly_weights[feature]
                    
                    if feature in adult_weights:
                        adult_score += normalized_value * adult_weights[feature]
            
            max_score = max(child_score, youth_score, elderly_score, adult_score)
            
            if max_score == child_score and child_score > 0.4:
                age_range = (4, 12)
                estimated_age = age_range[0] + (child_score * (age_range[1] - age_range[0]))
            elif max_score == youth_score and youth_score > 0.4:
                age_range = (13, 25)
                estimated_age = age_range[0] + (youth_score * (age_range[1] - age_range[0]))
            elif max_score == elderly_score and elderly_score > 0.5:
                age_range = (65, 90)
                estimated_age = age_range[0] + (elderly_score * (age_range[1] - age_range[0]))
            else:
                age_range = (26, 64)
                estimated_age = age_range[0] + (adult_score * (age_range[1] - age_range[0]))
            
            return max(3, min(99, estimated_age))
            
        except Exception as e:
            print(f"Erro no cálculo avançado de idade: {e}")
            return 30.0
    
    def _analyze_child_indicators(self, features: Dict[str, float], face_img: np.ndarray) -> float:
       
        try:
            child_score = 0.0
            indicators = 0
            
            if 'face_roundness' in features and features['face_roundness'] > 0.85:
                child_score += 0.35
                indicators += 1
            if 'cheek_fullness' in features and features['cheek_fullness'] > 0.5:
                child_score += 0.30
                indicators += 1
            if 'skin_smoothness' in features and features['skin_smoothness'] > 0.6:
                child_score += 0.25
                indicators += 1
            if 'bone_development' in features and features['bone_development'] < 0.2:
                child_score += 0.20
                indicators += 1
            if 'jaw_definition' in features and features['jaw_definition'] < 0.25:
                child_score += 0.15
                indicators += 1
            if 'eye_socket_depth' in features and features['eye_socket_depth'] < 0.15:
                child_score += 0.10
                indicators += 1
            
            if indicators > 0:
                child_score = child_score / indicators * min(4, indicators)
            
            return max(0.0, min(1.0, child_score))
        except Exception as e:
            print(f"Erro na análise de indicadores infantis: {e}")
            return 0.0
    
    def _analyze_youth_indicators(self, features: Dict[str, float], face_img: np.ndarray) -> float:
        """Análise específica para detectar jovens (13-25 anos)"""
        try:
            youth_score = 0.0
            indicators = 0
            
            if 'skin_smoothness' in features and 0.3 < features['skin_smoothness'] < 0.6:
                youth_score += 0.25
                indicators += 1
            if 'facial_maturity' in features and 0.3 < features['facial_maturity'] < 0.7:
                youth_score += 0.25
                indicators += 1
            if 'bone_development' in features and 0.2 < features['bone_development'] < 0.8:
                youth_score += 0.20
                indicators += 1
            if 'jaw_definition' in features and 0.3 < features['jaw_definition'] < 0.7:
                youth_score += 0.15
                indicators += 1
            if 'cheek_fullness' in features and 0.2 < features['cheek_fullness'] < 0.5:
                youth_score += 0.10
                indicators += 1
            if 'eye_socket_depth' in features and 0.2 < features['eye_socket_depth'] < 0.6:
                youth_score += 0.10
                indicators += 1
            
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            
            if gray.size > 0:
                skin_variance = np.var(gray)
                if 200 < skin_variance < 800:
                    youth_score += 0.15
                    indicators += 1
                
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                if 0.05 < edge_density < 0.15:
                    youth_score += 0.10
                    indicators += 1
            
            if indicators > 0:
                youth_score = youth_score / indicators * min(5, indicators)
            
            return max(0.0, min(1.0, youth_score))
        except Exception as e:
            print(f"Erro na análise de indicadores juvenis: {e}")
            return 0.0
    
    def _analyze_elderly_indicators(self, features: Dict[str, float], face_img: np.ndarray) -> float:
     
        try:
            elderly_score = 0.0
            indicators = 0
            
            if 'aging_patterns' in features and features['aging_patterns'] > 0.5:
                elderly_score += 0.30
                indicators += 1
            if 'texture_complexity' in features and features['texture_complexity'] > 80:
                elderly_score += 0.25
                indicators += 1
            if 'skin_elasticity_markers' in features and features['skin_elasticity_markers'] > 0.5:
                elderly_score += 0.20
                indicators += 1
            if 'contrast_level' in features and features['contrast_level'] > 35:
                elderly_score += 0.15
                indicators += 1
            if 'eye_socket_depth' in features and features['eye_socket_depth'] > 0.6:
                elderly_score += 0.10
                indicators += 1
            
            if indicators > 0:
                elderly_score = elderly_score / indicators * min(3, indicators)
            
            return max(0.0, min(1.0, elderly_score))
        except Exception as e:
            print(f"Erro na análise de indicadores de idosos: {e}")
            return 0.0
    
    def _refine_child_age(self, base_age: float, child_prob: float, features: Dict[str, float]) -> float:
       
        try:
            if child_prob > 0.9:
                refined_age = 4 + (child_prob * 6)
            elif child_prob > 0.8:
                refined_age = 6 + (child_prob * 8)
            elif child_prob > 0.7:
                refined_age = 8 + (child_prob * 10)
            else:
                refined_age = 10 + (child_prob * 8)
            
            return max(3, min(16, refined_age))
        except Exception as e:
            return base_age
    
    def _refine_youth_age(self, base_age: float, youth_prob: float, features: Dict[str, float]) -> float:

        try:
            if youth_prob > 0.9:
                refined_age = 16 + (youth_prob * 6)
            elif youth_prob > 0.8:
                refined_age = 15 + (youth_prob * 8)
            elif youth_prob > 0.7:
                refined_age = 14 + (youth_prob * 10)
            elif youth_prob > 0.6:
                refined_age = 13 + (youth_prob * 12)
            else:
                refined_age = 16 + (youth_prob * 9)
            
            skin_factor = features.get('skin_smoothness', 0.5)
            maturity_factor = features.get('facial_maturity', 0.5)
            
            if skin_factor > 0.7:
                refined_age *= 0.9
            elif skin_factor < 0.3:
                refined_age *= 1.1
                
            if maturity_factor < 0.4:
                refined_age *= 0.9
            elif maturity_factor > 0.7:
                refined_age *= 1.1
            
            return max(13, min(25, refined_age))
        except Exception as e:
            return base_age
    
    def _refine_elderly_age(self, base_age: float, elderly_prob: float, features: Dict[str, float]) -> float:
        """Refina a idade para idosos"""
        try:
            if elderly_prob > 0.9:
                refined_age = 75 + (elderly_prob * 20)
            elif elderly_prob > 0.8:
                refined_age = 70 + (elderly_prob * 18)
            elif elderly_prob > 0.7:
                refined_age = 65 + (elderly_prob * 15)
            else:
                refined_age = 60 + (elderly_prob * 12)
            
            return max(60, min(95, refined_age))
        except Exception as e:
            return base_age
        try:
            if elderly_prob > 0.8:  # Muito idoso
                refined_age = 75 + (base_age * 0.3)
            elif elderly_prob > 0.7:  # Idoso
                refined_age = 65 + (base_age * 0.4)
            else:  # Meia idade avançada
                refined_age = 55 + (base_age * 0.5)
            
            return max(50, min(99, refined_age))
        except Exception as e:
            return base_age
    
    def _calculate_age_from_features(self, features: Dict[str, float]) -> int:
        age_weights = {
            'skin_texture': 25.0,
            'facial_structure': 15.0,
            'eye_region': 20.0,
            'mouth_region': 10.0,
            'overall_aging': 30.0
        }
        
        base_age = 25
        age_adjustment = 0
        
        for feature, value in features.items():
            if feature in age_weights:
                normalized_value = max(-1, min(1, value))
                age_adjustment += normalized_value * age_weights[feature]
        
        estimated_age = base_age + age_adjustment
        return max(1, min(99, int(estimated_age)))
    
    def _predict_with_texture_analysis(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            forehead_region = gray[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
            eye_region = gray[int(h*0.25):int(h*0.55), int(w*0.15):int(w*0.85)]
            cheek_region = gray[int(h*0.4):int(h*0.7), int(w*0.1):int(w*0.9)]
            
            forehead_texture = cv2.Laplacian(forehead_region, cv2.CV_64F).var() if forehead_region.size > 0 else 0
            eye_texture = cv2.Laplacian(eye_region, cv2.CV_64F).var() if eye_region.size > 0 else 0
            cheek_texture = cv2.Laplacian(cheek_region, cv2.CV_64F).var() if cheek_region.size > 0 else 0
            
            overall_texture = (forehead_texture + eye_texture + cheek_texture) / 3
            
            wrinkle_detection = self._detect_wrinkles(gray)
            skin_elasticity = self._analyze_skin_elasticity(gray)
            gray_hair_score = self._detect_gray_hair(face_img, gray)
            
            child_indicators = 0.0
            elderly_indicators = 0.0
            
            if skin_elasticity > 0.9:
                child_indicators += 3.0
            elif skin_elasticity > 0.7:
                child_indicators += 1.5
            elif skin_elasticity < 0.3:
                elderly_indicators += 3.0
            elif skin_elasticity < 0.5:
                elderly_indicators += 1.5
            
            if overall_texture < 12:
                child_indicators += 2.5
            elif overall_texture < 40:
                child_indicators += 1.0
            elif overall_texture > 150:
                elderly_indicators += 2.5
            elif overall_texture > 100:
                elderly_indicators += 1.5
            elif overall_texture > 80:
                elderly_indicators += 0.8
            
            if wrinkle_detection > 0.7:
                elderly_indicators += 3.0
            elif wrinkle_detection > 0.5:
                elderly_indicators += 2.0
            elif wrinkle_detection > 0.3:
                elderly_indicators += 1.0
            elif wrinkle_detection < 0.1:
                child_indicators += 2.0
            
            brightness = np.mean(gray)
            if brightness > 180:
                child_indicators += 1.0
            elif brightness < 110:
                elderly_indicators += 0.8
            
            face_smoothness = 1.0 - (overall_texture / 220.0)
            if face_smoothness > 0.85:
                child_indicators += 2.0
            elif face_smoothness < 0.4:
                elderly_indicators += 1.5
            
            h, w = gray.shape
            face_roundness = w / max(1, h)
            if face_roundness > 1.05:
                child_indicators += 1.5
            elif face_roundness < 0.80:
                elderly_indicators += 0.5

            if gray_hair_score > 0.28:
                # Só reforçar idoso com cabelo grisalho quando há sinais faciais coerentes
                if (wrinkle_detection >= 0.45 and skin_elasticity < 0.50) or (wrinkle_detection >= 0.55):
                    hair_boost = 0.4 + (gray_hair_score * 1.0)
                    if skin_elasticity < 0.40:
                        hair_boost += 0.3
                    elderly_indicators += max(0.0, hair_boost)
                else:
                    # Sem rugas marcantes: contribuição mínima para evitar viés pró-idoso
                    elderly_indicators += gray_hair_score * 0.05
                # Atenuar se houver sinais juvenis
                if child_indicators >= 2.0 or face_smoothness > 0.80:
                    elderly_indicators -= min(elderly_indicators * 0.6, 1.2)
            
            # Gating anti-idoso: muitos sinais juvenis reduzem idoso
            if child_indicators >= 3.0 and wrinkle_detection < 0.25:
                elderly_indicators *= 0.5

            # Decisão de criança
            if child_indicators > elderly_indicators and child_indicators > 4.0:
                if child_indicators > 6.0:
                    age_base = 8
                    age_variance = child_indicators * 0.8
                else:
                    age_base = 15
                    age_variance = child_indicators * 1.5
                
                estimated_age = max(3, min(18, int(age_base - age_variance)))
                confidence = min(0.9, 0.65 + (child_indicators * 0.05))
                
            # Decisão de idoso exige combinação de evidências mais forte
            elif (
                elderly_indicators > child_indicators
                and elderly_indicators > 5.0
                and (
                    (wrinkle_detection >= 0.40 and skin_elasticity < 0.45)
                    or wrinkle_detection >= 0.50
                )
                and child_indicators < 2.5
            ):
                if elderly_indicators > 7.5:
                    age_base = 75
                    age_variance = elderly_indicators * 2.2
                else:
                    age_base = 60
                    age_variance = elderly_indicators * 2.6
                
                estimated_age = max(55, min(95, int(age_base + age_variance)))
                confidence = min(0.9, 0.65 + (elderly_indicators * 0.04))
                
            else:
                age_score = 25 + (overall_texture / 30) + (wrinkle_detection * 15) + ((1.0 - skin_elasticity) * 18)
                # cabelo branco só com rugas
                if wrinkle_detection >= 0.35 and skin_elasticity < 0.55:
                    age_score += gray_hair_score * 3
                # salvaguarda jovem forte
                if child_indicators >= 3.0 and wrinkle_detection < 0.25 and skin_elasticity > 0.55:
                    age_score = min(age_score, 30)
                estimated_age = max(18, min(65, int(age_score)))
                confidence = 0.6

            # Aplicar salvaguarda final jovem no resultado
            if child_indicators >= 3.0 and wrinkle_detection < 0.20 and skin_elasticity > 0.60 and estimated_age > 28:
                estimated_age = 28
            
            return {
                'method': 'texture_analysis_enhanced',
                'estimated_age': estimated_age,
                'confidence': float(confidence),
                'texture_score': float(overall_texture),
                'wrinkle_score': float(wrinkle_detection),
                'skin_elasticity': float(skin_elasticity),
                'gray_hair_score': float(gray_hair_score),
                'child_indicators': float(child_indicators),
                'elderly_indicators': float(elderly_indicators)
            }
        except Exception as e:
            print(f"Erro análise de textura: {e}")
            return {'method': 'texture', 'estimated_age': 30, 'confidence': 0.3}
    
    def _detect_wrinkles(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        
        total_wrinkle_score = 0.0
        
        forehead_region = gray[int(h*0.1):int(h*0.3), int(w*0.2):int(w*0.8)]
        if forehead_region.size > 0:
            forehead_edges = cv2.Canny(forehead_region, 25, 70)
            forehead_lines = cv2.HoughLinesP(forehead_edges, 1, np.pi/180, threshold=15, minLineLength=8, maxLineGap=3)
            
            if forehead_lines is not None:
                horizontal_forehead = 0
                for line in forehead_lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 25 or angle > 155:
                        horizontal_forehead += 1
                
                forehead_score = horizontal_forehead / max(1, forehead_region.size / 1000)
                total_wrinkle_score += forehead_score * 0.4
        
        eye_left = gray[int(h*0.35):int(h*0.5), int(w*0.05):int(w*0.3)]
        eye_right = gray[int(h*0.35):int(h*0.5), int(w*0.7):int(w*0.95)]
        
        for eye_region in [eye_left, eye_right]:
            if eye_region.size > 0:
                eye_edges = cv2.Canny(eye_region, 20, 60)
                eye_lines = cv2.HoughLinesP(eye_edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=2)
                
                if eye_lines is not None:
                    crow_feet = 0
                    for line in eye_lines:
                        x1, y1, x2, y2 = line[0]
                        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                        if 30 <= angle <= 150:
                            crow_feet += 1
                    
                    eye_score = crow_feet / max(1, eye_region.size / 500)
                    total_wrinkle_score += eye_score * 0.15
        
        mouth_region = gray[int(h*0.6):int(h*0.85), int(w*0.2):int(w*0.8)]
        if mouth_region.size > 0:
            mouth_edges = cv2.Canny(mouth_region, 20, 60)
            mouth_lines = cv2.HoughLinesP(mouth_edges, 1, np.pi/180, threshold=12, minLineLength=6, maxLineGap=3)
            
            if mouth_lines is not None:
                nasolabial_lines = 0
                for line in mouth_lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 8:
                        nasolabial_lines += 1
                
                mouth_score = nasolabial_lines / max(1, mouth_region.size / 800)
                total_wrinkle_score += mouth_score * 0.3
        
        return min(1.0, total_wrinkle_score)
    
    def _detect_gray_hair(self, face_img: np.ndarray, gray: np.ndarray) -> float:
    
        try:
            h, w = gray.shape
            if h == 0 or w == 0:
                return 0.0

            y1, y2 = int(h * 0.04), int(h * 0.18)
            x1, x2 = int(w * 0.12), int(w * 0.88)
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))

            band_bgr = face_img[y1:y2, x1:x2] if len(face_img.shape) == 3 else None
            band_gray = gray[y1:y2, x1:x2]
            if band_gray.size == 0:
                return 0.0

            components = []

            if band_bgr is not None and band_bgr.size > 0:
                hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
                _, s, v = cv2.split(hsv)
              
                high_v = (v > 200).astype(np.float32)
                low_s = (s < 35).astype(np.float32)
                high_s_block = (s > 70).astype(np.float32)
                hsv_ratio = float(np.mean(high_v * low_s))
                hsv_ratio = hsv_ratio * (1.0 - float(np.mean(high_s_block)) * 0.8)
                components.append(min(1.0, max(0.0, hsv_ratio) * 1.0))

                lab = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(lab)
                high_L = (L > 210).astype(np.float32)
                near_neutral = ((np.abs(A - 128) < 8) & (np.abs(B - 128) < 8)).astype(np.float32)
                lab_ratio = float(np.mean(high_L * near_neutral))
                components.append(min(1.0, lab_ratio * 1.0))

            try:
                edges = cv2.Canny(band_gray, 60, 150)
                bright = (band_gray > 210).astype(np.uint8) * 255
                bright_edges = cv2.bitwise_and(edges, bright)
                strands_ratio = float(np.sum(bright_edges > 0) / max(1, bright_edges.size)) * 3.0
                components.append(min(1.0, strands_ratio))
            except Exception:
                pass

            if not components:
                return 0.0

            score = float(np.clip(np.mean(components), 0.0, 1.0))
            # Penaliza possíveis falsos positivos quando há muita pele/brilho homogêneo
            uniformity = float(np.std(band_gray.astype(np.float32))) if band_gray.size > 0 else 20.0
            if uniformity < 15.0:
                score *= 0.7
            return score
        except Exception:
            return 0.0

    def _analyze_skin_elasticity(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        
        elasticity_scores = []
        
        regions = [
            gray[int(h*0.2):int(h*0.4), int(w*0.3):int(w*0.7)],  
            gray[int(h*0.4):int(h*0.6), int(w*0.1):int(w*0.4)], 
            gray[int(h*0.4):int(h*0.6), int(w*0.6):int(w*0.9)],  
            gray[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]   
        ]
        
        for region in regions:
            if region.size > 100:
                kernel_sizes = [3, 5, 7]
                texture_variations = []
                
                for kernel_size in kernel_sizes:
                    blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
                    diff = np.abs(region.astype(float) - blurred.astype(float))
                    texture_variations.append(np.mean(diff))
                
                texture_consistency = 1.0 - (np.std(texture_variations) / (np.mean(texture_variations) + 1e-6))
                
                sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                gradient_uniformity = 1.0 - (np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6))
                
                local_variance = []
                for y in range(0, region.shape[0]-10, 10):
                    for x in range(0, region.shape[1]-10, 10):
                        patch = region[y:y+10, x:x+10]
                        if patch.size == 100:
                            local_variance.append(np.var(patch))
                
                if local_variance:
                    variance_consistency = 1.0 - (np.std(local_variance) / (np.mean(local_variance) + 1e-6))
                else:
                    variance_consistency = 0.5
                
                region_elasticity = (texture_consistency + gradient_uniformity + variance_consistency) / 3.0
                elasticity_scores.append(region_elasticity)
        
        if elasticity_scores:
            overall_elasticity = np.mean(elasticity_scores)
            
            if overall_elasticity > 0.9:
                return 0.9  # Pele muito elástica (criança/jovem)
            elif overall_elasticity > 0.7:
                return 0.7  # Pele elástica (adulto jovem)
            elif overall_elasticity > 0.5:
                return 0.5  # Pele moderadamente elástica (adulto)
            elif overall_elasticity > 0.3:
                return 0.3  # Pele pouco elástica (meia-idade)
            else:
                return 0.2  # Pele muito pouco elástica (idoso)
        
        return 0.5
    
    def _predict_traditional(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            eye_region = gray[int(h*0.3):int(h*0.5), int(w*0.2):int(w*0.8)]
            eye_texture = np.std(eye_region) if eye_region.size > 0 else 0
            
            forehead_region = gray[int(h*0.1):int(h*0.35), int(w*0.25):int(w*0.75)]
            forehead_texture = np.std(forehead_region) if forehead_region.size > 0 else 0
            
            age_score = 20 + (texture_variance / 40) + (eye_texture / 8) + (forehead_texture / 10)
            estimated_age = max(1, min(99, int(age_score)))
            
            confidence = min(0.8, max(0.4, 0.7 - abs(texture_variance - 400) / 1000))
            
            return {
                'method': 'traditional_enhanced',
                'estimated_age': estimated_age,
                'confidence': float(confidence)
            }
        except Exception as e:
            print(f"Erro tradicional idade: {e}")
            return {'method': 'traditional', 'estimated_age': 30, 'confidence': 0.3}
    
    def _ensemble_age_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Dar mais importância ao CNN e Facenet quando ML está disponível
            from config import Config
            age_default_weights = {
                'cnn_age_regression': 0.40,
                'facenet_age': 0.30,
                'texture_analysis_enhanced': 0.20,
                'traditional_enhanced': 0.10
            }
            age_cfg = getattr(Config, 'AGE_CONFIG', {}).get('ensemble_method_weights', age_default_weights)
            method_weights = {**age_default_weights, **age_cfg}
            
            weighted_age = 0.0
            total_weight = 0.0
            all_confidences = []
            methods_used = []
            
            child_evidence = 0.0
            elderly_evidence = 0.0
            child_probs: List[float] = []
            elderly_probs: List[float] = []
            youth_probs: List[float] = []
            child_votes = 0
            elderly_votes = 0
            adult_votes = 0
            
            for pred in predictions:
                method = pred.get('method', 'unknown')
                # Normalizar nomes de métodos para aplicar pesos configurados
                if 'facenet_age' in method:
                    method = 'facenet_age'
                if method.startswith('traditional'):
                    method = 'traditional_enhanced'
                age = pred.get('estimated_age', 30)
                confidence = pred.get('confidence', 0.3)
                
                if method in method_weights and 1 <= age <= 99:
                    base_weight = method_weights[method] * confidence
                    
                    if age <= 16:
                        child_evidence += confidence
                        if 'child_indicators' in pred and pred['child_indicators'] > 3.0:
                            base_weight *= 1.5
                    elif age >= 65:
                        elderly_evidence += confidence
                        if 'elderly_indicators' in pred and pred['elderly_indicators'] > 3.0:
                            base_weight *= 1.4
                    
                    weighted_age += age * base_weight
                    total_weight += base_weight
                    all_confidences.append(confidence)
                    methods_used.append(method)

                # Capturar probabilidades de criança/idoso quando fornecidas (ex.: Facenet)
                cp = pred.get('child_probability')
                if isinstance(cp, (int, float)) and not np.isnan(cp) and not np.isinf(cp):
                    child_probs.append(float(cp))
                ep = pred.get('elderly_probability')
                if isinstance(ep, (int, float)) and not np.isnan(ep) and not np.isinf(ep):
                    elderly_probs.append(float(ep))
                yp = pred.get('youth_probability')
                if isinstance(yp, (int, float)) and not np.isnan(yp) and not np.isinf(yp):
                    youth_probs.append(float(yp))

                # Votos por faixa etária
                from config import Config
                age_cfg = getattr(Config, 'AGE_CONFIG', {})
                child_prob_vote_thr = float(age_cfg.get('child_prob_vote_threshold', 0.62))
                elderly_prob_vote_thr = float(age_cfg.get('elderly_prob_vote_threshold', 0.65))
                child_ind_thr = float(age_cfg.get('child_indicators_vote_threshold', 2.5))
                elderly_ind_thr = float(age_cfg.get('elderly_indicators_vote_threshold', 2.5))
                if age <= 16 or (pred.get('child_indicators', 0.0) >= child_ind_thr) or (pred.get('child_probability', 0.0) >= child_prob_vote_thr):
                    child_votes += 1
                elif (
                    (age >= 68) or
                    (pred.get('elderly_indicators', 0.0) >= elderly_ind_thr) or
                    (pred.get('elderly_probability', 0.0) >= elderly_prob_vote_thr)
                ):
                    # Anti-idoso: não votar idoso se houver sinais juvenis/infantis moderados
                    if (
                        pred.get('child_indicators', 0.0) < 2.5 and
                        pred.get('child_probability', 0.0) < 0.60 and
                        pred.get('youth_probability', 0.0) < 0.65
                    ):
                        elderly_votes += 1
                    else:
                        adult_votes += 1
                else:
                    adult_votes += 1
            
            if total_weight > 0:
                final_age = int(weighted_age / total_weight)
            else:
                final_age = 30
            
            final_confidence = np.mean(all_confidences) if all_confidences else 0.5
            
            # Reforços baseados em evidência forte, com salvaguardas para não saturar 16 anos
            mean_child_prob = float(np.mean(child_probs)) if child_probs else 0.0
            mean_elderly_prob = float(np.mean(elderly_probs)) if elderly_probs else 0.0
            mean_youth_prob = float(np.mean(youth_probs)) if youth_probs else 0.0

            from config import Config
            age_cfg = getattr(Config, 'AGE_CONFIG', {})
            # Child cap config
            child_cap_enabled = bool(age_cfg.get('child_cap_enabled', True))
            child_cap_prob_thr = float(age_cfg.get('child_cap_prob_threshold', 0.68))
            child_cap_min_votes = int(age_cfg.get('child_cap_min_votes', 2))
            child_cap_strict_votes = int(age_cfg.get('child_cap_strict_votes', 3))
            child_cap_max_age = int(age_cfg.get('child_cap_max_age', 16))
            child_cap_weighted_ceiling = int(age_cfg.get('child_cap_weighted_ceiling', 22))

            # Elderly cap config
            elderly_cap_enabled = bool(age_cfg.get('elderly_cap_enabled', True))
            elderly_cap_prob_thr = float(age_cfg.get('elderly_cap_prob_threshold', 0.70))
            elderly_cap_min_votes = int(age_cfg.get('elderly_cap_min_votes', 2))
            elderly_cap_strict_votes = int(age_cfg.get('elderly_cap_strict_votes', 3))
            elderly_cap_min_age = int(age_cfg.get('elderly_cap_min_age', 65))
            elderly_cap_weighted_floor = int(age_cfg.get('elderly_cap_weighted_floor', 58))

            # Aplica child cap somente se maioria real indicar criança e idade ponderada estiver baixa
            apply_child_cap = child_cap_enabled and (
                (mean_child_prob >= child_cap_prob_thr and child_votes >= child_cap_min_votes) or (child_votes >= child_cap_strict_votes)
            ) and (final_age <= child_cap_weighted_ceiling) and (elderly_votes == 0 and mean_elderly_prob < 0.45)

            if apply_child_cap:
                if final_age > child_cap_max_age:
                    final_age = child_cap_max_age
                final_confidence = min(0.92, max(final_confidence, 0.72) + 0.10)
            else:
                # Aplica elderly cap similarmente, evitando puxar adultos
                apply_elderly_cap = elderly_cap_enabled and (
                    (mean_elderly_prob >= elderly_cap_prob_thr and elderly_votes >= elderly_cap_min_votes) or (elderly_votes >= elderly_cap_strict_votes)
                ) and (final_age >= elderly_cap_weighted_floor) and (child_votes == 0 and mean_child_prob < 0.40) and (mean_youth_prob < 0.55)
                if apply_elderly_cap:
                    if final_age < elderly_cap_min_age:
                        final_age = elderly_cap_min_age
                    # Se evidência de idoso for bem forte, permita puxar um pouco mais a idade
                    if mean_elderly_prob >= (elderly_cap_prob_thr + 0.08) and elderly_votes >= (elderly_cap_min_votes + 1):
                        final_age = max(final_age, elderly_cap_min_age + 2)
                    final_confidence = min(0.94, max(final_confidence, 0.74) + 0.08)

            # Piso adulto sem evidência infantil forte
            try:
                from config import Config
                min_adult_age = int(getattr(Config, 'AGE_CONFIG', {}).get('min_adult_age_without_strong_child_evidence', 18))
                strong_child = (mean_child_prob >= float(getattr(Config, 'AGE_CONFIG', {}).get('child_prob_vote_threshold', 0.70))) or (child_votes >= int(getattr(Config, 'AGE_CONFIG', {}).get('child_cap_strict_votes', 4)))
                if final_age < min_adult_age and not strong_child:
                    print(f"[AGE_ENS] Piso adulto aplicado: {final_age} -> {min_adult_age} (child_votes={child_votes}, mean_child_prob={mean_child_prob:.2f})")
                    final_age = min_adult_age
                    final_confidence = max(final_confidence, 0.70)
            except Exception:
                pass

            # Blend infantil suave para puxar idade quando houver consistência infantil,
            # evitando travar todos em 16 anos
            try:
                from config import Config
                age_cfg = getattr(Config, 'AGE_CONFIG', {})
                blend_enabled = bool(age_cfg.get('child_blend_enabled', True))
                child_vote_min_for_blend = int(age_cfg.get('child_vote_min_for_blend', 2))
                child_target_ceiling = int(age_cfg.get('child_target_ceiling', 20))
                child_blend_alpha = float(age_cfg.get('child_blend_alpha', 0.6))  # peso para idade infantil mediana
                child_blend_prob_thr = float(age_cfg.get('child_blend_prob_threshold', 0.60))
                child_cap_max_age = int(age_cfg.get('child_cap_max_age', 16))
                child_blend_margin = int(age_cfg.get('child_blend_margin', 1))

                if blend_enabled and (child_votes >= child_vote_min_for_blend) and (mean_child_prob >= child_blend_prob_thr) \
                        and (elderly_votes == 0 and mean_elderly_prob < 0.45):
                    child_candidate_ages = [
                        int(p.get('estimated_age', 30)) for p in predictions
                        if 1 <= int(p.get('estimated_age', 30)) <= child_target_ceiling
                        or float(p.get('child_probability', 0.0)) >= child_blend_prob_thr
                        or float(p.get('child_indicators', 0.0)) >= 2.0
                    ]
                    if child_candidate_ages:
                        child_ref = int(np.median(child_candidate_ages))
                        child_ref = max(4, min(child_cap_max_age - child_blend_margin, child_ref))
                        blended = int((1.0 - child_blend_alpha) * final_age + child_blend_alpha * (child_ref + child_blend_margin))
                        final_age = min(final_age, blended)
                        final_confidence = min(0.93, max(final_confidence, 0.72) + 0.06)
            except Exception:
                pass
            
            if len(methods_used) > 2:
                ages = [p['estimated_age'] for p in predictions if 1 <= p.get('estimated_age', 0) <= 99]
                if len(ages) > 1:
                    age_variance = np.var(ages)
                    if age_variance < 25:
                        agreement_boost = 0.15
                    elif age_variance < 50:
                        agreement_boost = 0.1
                    else:
                        agreement_boost = 0.0
                    
                    final_confidence = min(0.95, final_confidence + agreement_boost)
            
            final_age = max(1, min(99, final_age))
            age_category = self._get_age_category(final_age)
            
            # Faixa de idade mais consistente sob ML (configurável)
            from config import Config
            age_margin = getattr(Config, 'AGE_CONFIG', {}).get('age_margin_default', 2)
            if final_age <= 12:
                age_margin = getattr(Config, 'AGE_CONFIG', {}).get('age_margin_child', 2)
            elif final_age >= 70:
                age_margin = getattr(Config, 'AGE_CONFIG', {}).get('age_margin_elderly', 3)
            
            print(f"[AGE_ENS] methods={methods_used}, child_votes={child_votes}, elderly_votes={elderly_votes}, final_age={final_age}, conf={final_confidence:.2f}")
            return {
                'estimated_age': final_age,
                'age_range': f"{max(1, final_age-age_margin)}-{min(99, final_age+age_margin)}",
                'category': age_category,
                'confidence': float(final_confidence),
                'ensemble_details': {
                    'methods_used': methods_used,
                    'child_evidence': float(child_evidence),
                    'elderly_evidence': float(elderly_evidence),
                    'age_variance': float(np.var([p['estimated_age'] for p in predictions if 1 <= p.get('estimated_age', 0) <= 99]))
                }
            }
        except Exception as e:
            print(f"Erro no ensemble de idade: {e}")
            return {'estimated_age': 30, 'category': 'Jovem Adulto', 'confidence': 0.3}
    
    def _get_age_category(self, age: int) -> str:
        age_ranges = [
            (0, 5, 'Bebê'), (6, 12, 'Criança'), (13, 17, 'Adolescente'),
            (18, 25, 'Jovem'), (26, 35, 'Jovem Adulto'), (36, 45, 'Adulto'),
            (46, 55, 'Meia-idade'), (56, 65, 'Maduro'), (66, 100, 'Idoso')
        ]
        
        for min_age, max_age, category in age_ranges:
            if min_age <= age <= max_age:
                return category
        
        return 'Jovem Adulto'

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class AdvancedGenderDetector:
    def __init__(self):
   
        tflite_only = False
        try:
            from config import Config
            tflite_only = bool(getattr(Config, 'GENDER_CONFIG', {}).get('tflite_only', False))
        except Exception:
            pass

        if ML_AVAILABLE and not tflite_only:
            try:
                self.facenet = FaceNet()
                self.mtcnn = MTCNN()
                self.gender_model = self._load_or_create_gender_model()
                self.ml_enabled = True
                self._setup_predict_functions()
            except Exception as e:
                print(f"Erro ao carregar modelos ML: {e}")
                self.ml_enabled = False
        else:
            self.ml_enabled = False
        
        self._embedding_cache = {}
        self._prediction_cache = {}
        self._cache_size_limit = 50  # Limite do cache
        self._tflite_enabled = False
        try:
            from config import Config
            cfg = getattr(Config, 'GENDER_CONFIG', {})
            if cfg.get('use_tflite', False):
                import os
            
                InterpreterClass = None
                try:
                    import tflite_runtime.interpreter as tflite
                    InterpreterClass = tflite.Interpreter
                except Exception:
                    try:
                        from tensorflow.lite.python.interpreter import Interpreter as TfInterpreter
                        InterpreterClass = TfInterpreter
                    except Exception:
                        from tensorflow.lite import Interpreter as TfInterpreter
                        InterpreterClass = TfInterpreter
                self._tflite_threads = int(cfg.get('tflite_threads', 2))
                self._tflite_gender_path = cfg.get('tflite_gender_model_path', 'models/gender.tflite')
                self._tflite_embed_path = cfg.get('tflite_embedding_model_path', 'models/face_embedding.tflite')
           
                def _ensure_tflite_gender_model():
                    try:
                        need_create = (not os.path.exists(self._tflite_gender_path)) or (os.path.getsize(self._tflite_gender_path) == 0)
                        if need_create:
                            if ML_AVAILABLE:
                                keras_model = self._create_cnn_gender_model()
                                import tensorflow as tf
                                converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
                                tflite_model = converter.convert()
                                os.makedirs(os.path.dirname(self._tflite_gender_path) or '.', exist_ok=True)
                                with open(self._tflite_gender_path, 'wb') as f:
                                    f.write(tflite_model)
                            else:
                                print("Aviso: Modelo TFLite de gênero ausente/vazio e TensorFlow indisponível; pulando auto-geração. Forneça o arquivo .tflite em models/.")
                    except Exception as err:
                        print(f"Falha ao gerar modelo TFLite de gênero: {err}")
                _ensure_tflite_gender_model()

                self._tflite_gender = None
                self._tflite_embed = None
                if os.path.exists(self._tflite_gender_path) and os.path.getsize(self._tflite_gender_path) > 0:
                    self._tflite_gender = InterpreterClass(model_path=self._tflite_gender_path, num_threads=self._tflite_threads)
                    self._tflite_gender.allocate_tensors()
                    try:
                        g_in = self._tflite_gender.get_input_details()[0]
                        g_out = self._tflite_gender.get_output_details()[0]
                        print(f"[TFLite Gender] loaded: {self._tflite_gender_path}")
                        print(f"[TFLite Gender] input shape={g_in.get('shape')} dtype={g_in.get('dtype')} output shape={g_out.get('shape')} dtype={g_out.get('dtype')}")
                    except Exception:
                        pass
                if os.path.exists(self._tflite_embed_path) and os.path.getsize(self._tflite_embed_path) > 0:
                    self._tflite_embed = InterpreterClass(model_path=self._tflite_embed_path, num_threads=self._tflite_threads)
                    self._tflite_embed.allocate_tensors()
                    try:
                        e_in = self._tflite_embed.get_input_details()[0]
                        e_out = self._tflite_embed.get_output_details()[0]
                        print(f"[TFLite Embed] loaded: {self._tflite_embed_path}")
                        print(f"[TFLite Embed] input shape={e_in.get('shape')} dtype={e_in.get('dtype')} output shape={e_out.get('shape')} dtype={e_out.get('dtype')}")
                    except Exception:
                        pass
                self._tflite_enabled = (self._tflite_gender is not None) or (self._tflite_embed is not None)
                self._tflite_gender_logged_once = False
                if not self._tflite_enabled:
                    print("[TFLite] Aviso: nenhum intérprete TFLite carregado (verifique os arquivos em /app/models)")
        except Exception as e:
            print(f"TFLite desabilitado: {e}")
        
    def _cleanup_cache(self):
       
        if len(self._prediction_cache) > self._cache_size_limit:
           
            items_to_remove = len(self._prediction_cache) // 2
            cache_keys = list(self._prediction_cache.keys())
            for key in cache_keys[:items_to_remove]:
                del self._prediction_cache[key]
        
    def _setup_predict_functions(self):
       
        if self.ml_enabled and hasattr(self, 'gender_model'):
           
            import tensorflow as tf
            
            @tf.function(reduce_retracing=True)
            def optimized_predict(x):
                return self.gender_model(x, training=False)
            
            self._optimized_predict = optimized_predict
        
    def _compute_facial_hair_features_for_image(self, face_img: np.ndarray) -> Dict[str, float]:
        
        try:
            is_color = len(face_img.shape) == 3 and face_img.shape[2] == 3
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if is_color else face_img
            h, w = gray.shape if len(gray.shape) == 2 else (0, 0)
            features: Dict[str, float] = {}

           
            try:
                features['avg_brightness_norm'] = float(np.mean(gray) / 255.0)
            except Exception:
                features['avg_brightness_norm'] = 0.5

            if h <= 0 or w <= 0:
               
                features.update({
                    'beard_coverage_ratio': 0.0,
                    'stubble_score': 0.0,
                    'mustache_score': 0.0,
                    'jaw_darkness_ratio': 0.0,
                    'lip_color_contrast': 0.0,
                    'soft_tissue_distribution': 0.0,
                })
                return features

           
            jaw_region = gray[int(h*0.58):h, int(w*0.08):int(w*0.92)]
            if jaw_region.size > 0:
                try:
                    jaw_lap = cv2.Laplacian(jaw_region, cv2.CV_64F).var()
                except Exception:
                    jaw_lap = 0.0
                stubble_score = min(2.0, jaw_lap / 130.0)
                jaw_edges = cv2.Canny(jaw_region, 30, 100)
                beard_coverage_ratio = float(np.sum(jaw_edges > 0) / max(1, jaw_edges.size))
                beard_coverage_ratio = float(min(1.0, beard_coverage_ratio * 3.0))
               
                try:
                    cheek_left = gray[int(h*0.45):int(h*0.65), int(w*0.10):int(w*0.35)]
                    cheek_right = gray[int(h*0.45):int(h*0.65), int(w*0.65):int(w*0.90)]
                    cheek_means = []
                    if cheek_left.size > 0:
                        cheek_means.append(float(np.mean(cheek_left)))
                    if cheek_right.size > 0:
                        cheek_means.append(float(np.mean(cheek_right)))
                    if cheek_means:
                        cheek_mean = float(np.mean(cheek_means))
                        jaw_mean = float(np.mean(jaw_region))
                        jaw_darkness_ratio = max(0.0, (cheek_mean - jaw_mean) / max(1.0, cheek_mean))
                    else:
                        jaw_darkness_ratio = 0.0
                except Exception:
                    jaw_darkness_ratio = 0.0
            else:
                stubble_score = 0.0
                beard_coverage_ratio = 0.0
                jaw_darkness_ratio = 0.0

           
            mustache_region = gray[int(h*0.50):int(h*0.62), int(w*0.30):int(w*0.70)]
            if mustache_region.size > 0:
                try:
                    mustache_lap = cv2.Laplacian(mustache_region, cv2.CV_64F).var()
                except Exception:
                    mustache_lap = 0.0
                mustache_score = min(2.0, mustache_lap / 120.0)
            else:
                mustache_score = 0.0

           
            lip_region_g = gray[int(h*0.62):int(h*0.82), int(w*0.25):int(w*0.75)]
            lower_skin_region_g = gray[int(h*0.62):int(h*0.82), int(w*0.05):int(w*0.20)] if w > 0 else lip_region_g
            if lip_region_g.size > 0 and lower_skin_region_g.size > 0:
                lip_mean = float(np.mean(lip_region_g))
                skin_mean = float(np.mean(lower_skin_region_g))
                lip_color_contrast = max(0.0, (skin_mean - lip_mean) / 40.0)
                lip_color_contrast = float(min(2.0, lip_color_contrast))
            else:
                lip_color_contrast = 0.0


            cheek_left = gray[int(h*0.45):int(h*0.65), int(w*0.10):int(w*0.35)]
            cheek_right = gray[int(h*0.45):int(h*0.65), int(w*0.65):int(w*0.90)]
            cheek_softness = 0.0
            try:
                cheek_lap_var = 0.0
                count = 0
                for ck in [cheek_left, cheek_right]:
                    if ck.size > 0:
                        cheek_lap_var += cv2.Laplacian(ck, cv2.CV_64F).var()
                        count += 1
                cheek_lap_var = (cheek_lap_var / max(1, count))
                cheek_softness = 1.0 / (1.0 + (cheek_lap_var / 120.0))
            except Exception:
                cheek_softness = 0.0
            soft_tissue_distribution = float(min(2.0, cheek_softness * 2.0))

            features.update({
                'beard_coverage_ratio': float(beard_coverage_ratio),
                'stubble_score': float(stubble_score),
                'mustache_score': float(mustache_score),
                'jaw_darkness_ratio': float(jaw_darkness_ratio),
                'lip_color_contrast': float(lip_color_contrast),
                'soft_tissue_distribution': float(soft_tissue_distribution),
            })
            return features
        except Exception:
            return {
                'beard_coverage_ratio': 0.0,
                'stubble_score': 0.0,
                'mustache_score': 0.0,
                'jaw_darkness_ratio': 0.0,
                'lip_color_contrast': 0.0,
                'soft_tissue_distribution': 0.0,
                'avg_brightness_norm': 0.5,
            }

    def _load_or_create_gender_model(self):
        model_path = 'models/gender_model.h5'
        if os.path.exists(model_path):
            return keras.models.load_model(model_path)
        else:
            return self._create_cnn_gender_model()
    
    def _create_cnn_gender_model(self):
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(160, 160, 3)
        )
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def detect_gender_advanced(self, face_img: np.ndarray) -> Dict[str, Any]:
        if not self.ml_enabled:
           
            if getattr(self, '_tflite_enabled', False):
                try:
                    return self._predict_with_tflite(face_img)
                except Exception as e:
                    print(f"Erro TFLite: {e}")
            return self._predict_traditional(face_img)
        
        try:
            if getattr(self, '_tflite_enabled', False):
                facenet_pred = self._predict_with_tflite(face_img)
            else:
                facenet_pred = self._predict_with_facenet(face_img)
            cnn_pred = self._predict_with_cnn(face_img)
            traditional_pred = self._predict_traditional(face_img)
            
            return self._ensemble_predictions([facenet_pred, cnn_pred, traditional_pred])
        except Exception as e:
            print(f"Erro no detector avançado: {e}")
            return self._predict_traditional(face_img)

    def _predict_with_tflite(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            import numpy as np
           
            ih, iw = 160, 160
            in_dtype = None
            if getattr(self, '_tflite_gender', None) is not None:
                g_inp = self._tflite_gender.get_input_details()[0]
                shp = g_inp.get('shape', [1, 160, 160, 3])
                ih, iw = int(shp[1]), int(shp[2])
                in_dtype = g_inp.get('dtype')

           
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (iw, ih)).astype('float32') / 255.0

            def run_inference(x_batch: np.ndarray) -> float:
                self._tflite_gender.set_tensor(g_inp['index'], x_batch)
                self._tflite_gender.invoke()
                g_raw = self._tflite_gender.get_tensor(self._tflite_gender.get_output_details()[0]['index'])
                g_pred = np.squeeze(g_raw)
               
                if g_pred.shape == () or isinstance(g_pred, (float, np.floating)):
                    return float(g_pred)
               
                vec = g_pred.reshape(-1).astype('float32')
                vec = vec - float(np.max(vec))
                expv = np.exp(vec)
                probs = expv / max(1e-8, float(np.sum(expv)))
                from config import Config
                order = getattr(Config, 'GENDER_CONFIG', {}).get('tflite_output_order', 'male_female')
                if probs.size >= 2:
                    return float(probs[0] if order == 'male_female' else probs[1])
                return 0.5

            male_probability = None

            if getattr(self, '_tflite_gender', None) is not None:
               
                if in_dtype and 'uint8' in str(in_dtype).lower():
                    inp = self._tflite_gender.get_input_details()[0]
                    scale = float(inp.get('quantization_parameters', {}).get('scales', [1.0])[0] or 1.0)
                    zero = int(inp.get('quantization_parameters', {}).get('zero_points', [0])[0] or 0)
                    x_uint8 = np.clip(np.round(face_resized / max(scale, 1e-8) + zero), 0, 255).astype('uint8')
                    x_uint8 = np.expand_dims(x_uint8, 0)
                    male_probability = run_inference(x_uint8)
                else:
                   
                    x01 = np.expand_dims(face_resized, 0)
                    x11 = np.expand_dims(face_resized * 2.0 - 1.0, 0)
                    mp01 = run_inference(x01)
                    mp11 = run_inference(x11)
                    from config import Config
                    thr = getattr(Config, 'GENDER_CONFIG', {}).get('global_threshold', 0.5)
                    male_probability = mp01 if abs(mp01 - thr) >= abs(mp11 - thr) else mp11

            from config import Config
            cfg = getattr(Config, 'GENDER_CONFIG', {})
            threshold = cfg.get('global_threshold', 0.50)
            invert = bool(cfg.get('invert_output', False))
            mp = (male_probability if male_probability is not None else 0.5)
            if invert:
                mp = 1.0 - mp

            try:
                fh = self._compute_facial_hair_features_for_image(face_img)
               
                beard_thr = cfg.get('beard_coverage_threshold', 0.12)
                stubble_thr_mod = cfg.get('stubble_moderate_threshold', 0.65)
                stubble_thr_str = cfg.get('stubble_strong_threshold', 0.85)
                must_thr_mod = cfg.get('mustache_moderate_threshold', 0.60)
                must_thr_str = cfg.get('mustache_strong_threshold', 0.85)
                require_multi = cfg.get('beard_require_multi_indicators', True)
                min_evidence = int(cfg.get('beard_evidence_min', 2))
                low_light_scale = float(cfg.get('beard_low_light_scale', 1.25))
                lip_block_thr = float(cfg.get('beard_lip_contrast_block_threshold', 0.95))
                soft_block_thr = float(cfg.get('beard_soft_tissue_block_threshold', 0.95))
                jaw_dark_thr_mod = float(cfg.get('beard_jaw_darkness_threshold', 0.20))
                jaw_dark_thr_str = float(cfg.get('beard_jaw_darkness_strong_threshold', 0.25))

                avg_brightness = float(fh.get('avg_brightness_norm', 0.5))
                if avg_brightness < cfg.get('low_light_brightness_threshold', 0.35):
                    beard_thr *= low_light_scale
                    stubble_thr_mod *= low_light_scale
                    stubble_thr_str *= low_light_scale
                    must_thr_mod *= low_light_scale
                    must_thr_str *= low_light_scale

                beard_cov = float(fh.get('beard_coverage_ratio', 0.0))
                stubble = float(fh.get('stubble_score', 0.0))
                mustache = float(fh.get('mustache_score', 0.0))
                lip_contrast = float(fh.get('lip_color_contrast', 0.0))
                soft_tissue = float(fh.get('soft_tissue_distribution', 0.0))
                jaw_dark = float(fh.get('jaw_darkness_ratio', 0.0))

                evidence = [
                    beard_cov > beard_thr,
                    stubble > stubble_thr_mod,
                    mustache > must_thr_mod
                ]
                strong_evidence = [
                    (beard_cov > beard_thr * 1.3) and (jaw_dark > jaw_dark_thr_str),
                    (stubble > stubble_thr_str) and (jaw_dark > jaw_dark_thr_str),
                    (mustache > must_thr_str) and (jaw_dark > jaw_dark_thr_str)
                ]
                strong_count = sum(1 for e in strong_evidence if e)
                evidence_count = sum(1 for e in evidence if e)
                strong_beard = (strong_count >= 1 and jaw_dark > jaw_dark_thr_str) if not require_multi else (strong_count >= 1 and evidence_count >= min_evidence and jaw_dark > jaw_dark_thr_str)
                moderate_beard = (evidence_count >= 1 and jaw_dark > jaw_dark_thr_mod) if not require_multi else (evidence_count >= min_evidence and jaw_dark > jaw_dark_thr_mod)

                block_beard = (lip_contrast > lip_block_thr and soft_tissue > soft_block_thr)
                if strong_beard and not block_beard:
                    floor_str = cfg.get('beard_male_floor_strong', 0.82)
                    boost_str = cfg.get('beard_male_boost_strong', 0.08)
                    mp = min(0.98, max(mp, floor_str) + boost_str)
                elif moderate_beard and not block_beard:
                    floor_mod = cfg.get('beard_male_floor_moderate', 0.72)
                    boost_mod = cfg.get('beard_male_boost_moderate', 0.04)
                    mp = min(0.96, max(mp, floor_mod) + boost_mod)
            except Exception:
                pass

            mp = max(0.05, min(0.95, mp))
            confidence = max(0.70, min(0.95, abs(mp - 0.5) * 2))
            predicted_gender = 'Masculino' if mp > threshold else 'Feminino'
            female_probability = 1.0 - mp

            return {
                'method': 'tflite_hybrid',
                'predicted_gender': predicted_gender,
                'male_probability': float(mp),
                'female_probability': float(female_probability),
                'confidence': float(confidence),
                'gender': predicted_gender
            }
        except Exception as e:
            print(f"Erro TFLite: {e}")
            return {'method': 'tflite_hybrid', 'predicted_gender': 'Feminino', 'male_probability': 0.48, 'female_probability': 0.52, 'confidence': 0.5, 'gender': 'Feminino'}
    
    def _predict_with_facenet(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            embedding = self.facenet.embeddings(face_expanded)[0]
            
            advanced_features = self._extract_advanced_gender_features(embedding, face_img)
            male_probability = self._classify_gender_with_advanced_ml(advanced_features)
            
            geometric_adjustment = self._get_geometric_gender_adjustment(face_img)
            
           
            from config import Config
            male_w = getattr(Config, 'GENDER_CONFIG', {}).get('facenet_male_weight', 0.90)
            geo_w = getattr(Config, 'GENDER_CONFIG', {}).get('facenet_geometric_weight', 0.10)
            total_w = max(1e-6, male_w + geo_w)
            final_probability = (male_probability * male_w + geometric_adjustment * geo_w) / total_w
            
            final_probability = max(0.05, min(0.95, final_probability))
            
            confidence_raw = abs(final_probability - 0.5) * 2
            confidence = max(getattr(Config, 'GENDER_CONFIG', {}).get('confidence_min_facenet', 0.65), min(0.95, confidence_raw))
            

            threshold = getattr(Config, 'GENDER_CONFIG', {}).get('facenet_threshold', 0.56)
            gender = 'Masculino' if final_probability > threshold else 'Feminino'
            
            return {
                'method': 'facenet_advanced',
                'male_probability': float(final_probability),
                'confidence': float(confidence),
                'gender': gender,
                'debug_features': {
                    'beard_coverage_ratio': float(advanced_features.get('beard_coverage_ratio', 0.0)),
                    'stubble_score': float(advanced_features.get('stubble_score', 0.0)),
                    'mustache_score': float(advanced_features.get('mustache_score', 0.0)),
                    'edge_intensity': float(advanced_features.get('edge_intensity', 0.0)),
                    'lip_color_contrast': float(advanced_features.get('lip_color_contrast', 0.0)),
                    'soft_tissue_distribution': float(advanced_features.get('soft_tissue_distribution', 0.0))
                }
            }
        except Exception as e:
            print(f"Erro FaceNet Avançado: {e}")
            return {'method': 'facenet_advanced', 'male_probability': 0.48, 'confidence': 0.5, 'gender': 'Feminino'}
    
    def _extract_gender_features_from_embedding(self, embedding: np.ndarray) -> Dict[str, float]:
        features = {
            'jaw_strength': float(np.mean(embedding[0:50])),
            'eye_features': float(np.mean(embedding[50:100])),
            'nose_features': float(np.mean(embedding[100:150])),
            'mouth_features': float(np.mean(embedding[150:200])),
            'face_shape': float(np.mean(embedding[200:300])),
            'skin_texture': float(np.std(embedding[300:400])),
            'overall_masculinity': float(np.mean(embedding[400:512]))
        }
        return features
    
    def _extract_advanced_gender_features(self, embedding: np.ndarray, face_img: np.ndarray) -> Dict[str, float]:
        
        try:
     
            features = {
                'facial_structure': float(np.mean(embedding[0:35])),
                'jaw_definition': float(np.mean(embedding[35:70])),
                'eye_socket_depth': float(np.mean(embedding[70:105])),
                'brow_ridge_prominence': float(np.mean(embedding[105:140])),
                'cheekbone_structure': float(np.mean(embedding[140:175])),
                'nose_bridge_width': float(np.mean(embedding[175:210])),
                'lip_fullness': float(np.mean(embedding[210:245])),
                'chin_shape': float(np.mean(embedding[245:280])),
                'forehead_slope': float(np.mean(embedding[280:315])),
                'face_width_ratio': float(np.mean(embedding[315:350])),
                'skin_texture_variance': float(np.std(embedding[350:385])),
                'facial_hair_indicators': float(np.mean(embedding[385:420])),
                'overall_bone_structure': float(np.mean(embedding[420:455])),
                'soft_tissue_distribution': float(np.mean(embedding[455:490])),
                'hormonal_markers': float(np.mean(embedding[490:512]))
            }
            
           
            is_color = len(face_img.shape) == 3 and face_img.shape[2] == 3
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if is_color else face_img
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV) if is_color else None
            h, w = gray.shape if len(gray.shape) == 2 else (0, 0)
           
            try:
                features['avg_brightness_norm'] = float(np.mean(gray) / 255.0)
            except Exception:
                features['avg_brightness_norm'] = 0.5
            
           
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_intensity = np.mean(np.sqrt(sobelx**2 + sobely**2))
            features['edge_intensity'] = float(edge_intensity)

           
            if h > 0 and w > 0:
               
                jaw_region = gray[int(h*0.58):h, int(w*0.08):int(w*0.92)]
                if jaw_region.size > 0:
                    jaw_lap = cv2.Laplacian(jaw_region, cv2.CV_64F).var()
                    stubble_score = min(2.0, jaw_lap / 130.0)
                   
                    jaw_edges = cv2.Canny(jaw_region, 30, 100)
                    beard_coverage_ratio = float(np.sum(jaw_edges > 0) / max(1, jaw_edges.size))
                    features['beard_coverage_ratio'] = float(min(1.0, beard_coverage_ratio * 3.0))
                    features['stubble_score'] = float(stubble_score)
                   
                    try:
                        cheek_left = gray[int(h*0.45):int(h*0.65), int(w*0.10):int(w*0.35)]
                        cheek_right = gray[int(h*0.45):int(h*0.65), int(w*0.65):int(w*0.90)]
                        cheek_means = []
                        if cheek_left.size > 0:
                            cheek_means.append(float(np.mean(cheek_left)))
                        if cheek_right.size > 0:
                            cheek_means.append(float(np.mean(cheek_right)))
                        if cheek_means:
                            cheek_mean = float(np.mean(cheek_means))
                            jaw_mean = float(np.mean(jaw_region))
                            jaw_darkness_ratio = max(0.0, (cheek_mean - jaw_mean) / max(1.0, cheek_mean))
                            features['jaw_darkness_ratio'] = float(min(1.0, jaw_darkness_ratio))
                        else:
                            features['jaw_darkness_ratio'] = 0.0
                    except Exception:
                        features['jaw_darkness_ratio'] = 0.0
                else:
                    features['stubble_score'] = 0.0
                    features['beard_coverage_ratio'] = 0.0
                    features['jaw_darkness_ratio'] = 0.0

               
                eyebrow_region = gray[int(h*0.12):int(h*0.35), int(w*0.10):int(w*0.90)]
                eye_region = gray[int(h*0.30):int(h*0.50), int(w*0.20):int(w*0.80)]
                if eyebrow_region.size > 0:
                    eb_sx = cv2.Sobel(eyebrow_region, cv2.CV_64F, 1, 0, ksize=3)
                    eb_sy = cv2.Sobel(eyebrow_region, cv2.CV_64F, 0, 1, ksize=3)
                    eb_vert = np.mean(np.abs(eb_sy))
                    eb_horiz = np.mean(np.abs(eb_sx)) + 1e-6
                    eyebrow_verticality = min(2.0, (eb_vert / eb_horiz))
                    eyebrow_thickness = min(2.0, float(np.std(eyebrow_region) / 30.0))
                    features['eyebrow_verticality'] = float(eyebrow_verticality)
                    features['eyebrow_thickness'] = float(eyebrow_thickness)
                else:
                    features['eyebrow_verticality'] = 0.0
                    features['eyebrow_thickness'] = 0.0

               
                eye_brow_gap_ratio = 0.0
                if h > 0:
                    gap_px = max(0, int(h*0.35) - int(h*0.30))
                    eye_brow_gap_ratio = min(2.0, (gap_px / max(1.0, h)) / 0.12)
                features['eye_brow_gap_ratio'] = float(eye_brow_gap_ratio)

               
                lip_region_g = gray[int(h*0.62):int(h*0.82), int(w*0.25):int(w*0.75)]
                lower_skin_region_g = gray[int(h*0.62):int(h*0.82), int(w*0.05):int(w*0.20)] if w > 0 else lip_region_g
                if lip_region_g.size > 0 and lower_skin_region_g.size > 0:
                    lip_mean = float(np.mean(lip_region_g))
                    skin_mean = float(np.mean(lower_skin_region_g))
                    lip_color_contrast = max(0.0, (skin_mean - lip_mean) / 40.0)
                    features['lip_color_contrast'] = float(min(2.0, lip_color_contrast))
                else:
                    features['lip_color_contrast'] = 0.0


                lip_saturation_contrast = 0.0
                if hsv is not None:
                    lip_region_hsv = hsv[int(h*0.62):int(h*0.82), int(w*0.25):int(w*0.75)]
                    lower_skin_region_hsv = hsv[int(h*0.62):int(h*0.82), int(w*0.05):int(w*0.20)] if w > 0 else lip_region_hsv
                    if lip_region_hsv.size > 0 and lower_skin_region_hsv.size > 0:
                        lip_sat = float(np.mean(lip_region_hsv[:, :, 1])) / 255.0
                        skin_sat = float(np.mean(lower_skin_region_hsv[:, :, 1])) / 255.0
                        lip_val = float(np.mean(lip_region_hsv[:, :, 2])) / 255.0
                        skin_val = float(np.mean(lower_skin_region_hsv[:, :, 2])) / 255.0
                        lip_saturation_contrast = max(0.0, (lip_sat - skin_sat) + 0.3 * (lip_val - skin_val))
                features['lip_saturation_contrast'] = float(min(2.0, lip_saturation_contrast))

               
                mustache_region = gray[int(h*0.50):int(h*0.62), int(w*0.30):int(w*0.70)]
                if mustache_region.size > 0:
                    mustache_lap = cv2.Laplacian(mustache_region, cv2.CV_64F).var()
                    mustache_score = min(2.0, mustache_lap / 120.0)
                    features['mustache_score'] = float(mustache_score)
                else:
                    features['mustache_score'] = 0.0

                                        
                cheek_left = gray[int(h*0.45):int(h*0.65), int(w*0.10):int(w*0.35)]
                cheek_right = gray[int(h*0.45):int(h*0.65), int(w*0.65):int(w*0.90)]
                cheekbone_grad = 0.0
                for ck in [cheek_left, cheek_right]:
                    if ck.size > 0:
                        g = np.sqrt(cv2.Sobel(ck, cv2.CV_64F, 1, 0, 3)**2 + cv2.Sobel(ck, cv2.CV_64F, 0, 1, 3)**2)
                        cheekbone_grad += float(np.mean(g))
                cheekbone_grad = (cheekbone_grad / 2.0) if cheekbone_grad > 0 else 0.0
                features['cheekbone_prominence_grad'] = float(min(2.0, cheekbone_grad / 30.0))


                cheek_softness = 0.0
                try:
                    cheek_lap_var = 0.0
                    count = 0
                    for ck in [cheek_left, cheek_right]:
                        if ck.size > 0:
                            cheek_lap_var += cv2.Laplacian(ck, cv2.CV_64F).var()
                            count += 1
                    cheek_lap_var = (cheek_lap_var / max(1, count))
                    cheek_softness = 1.0 / (1.0 + (cheek_lap_var / 120.0))
                except Exception:
                    cheek_softness = 0.0
                features['cheek_softness'] = float(min(2.0, cheek_softness * 2.0))


                hair_band = gray[int(h*0.06):int(h*0.12), int(w*0.20):int(w*0.80)]
                if hair_band.size > 0:
                    hb_edges = cv2.Canny(hair_band, 50, 150)
                    hairline_edge_density = (np.sum(hb_edges > 0) / hb_edges.size) * 2.0
                    features['hairline_edge_density'] = float(min(2.0, hairline_edge_density))
                else:
                    features['hairline_edge_density'] = 0.0

             
                nb_region = gray[int(h*0.30):int(h*0.60), int(w*0.45):int(w*0.55)]
                if nb_region.size > 0:
                    nb_grad = np.mean(np.abs(cv2.Sobel(nb_region, cv2.CV_64F, 0, 1, 3)))
                    features['nose_bridge_contrast'] = float(min(2.0, nb_grad / 25.0))
                else:
                    features['nose_bridge_contrast'] = 0.0
          
                eye_region = gray[int(h*0.30):int(h*0.50), int(w*0.20):int(w*0.80)]
                eye_makeup_contrast = 0.0
                if eye_region.size > 0:
                    try:
                        eye_edges = cv2.Canny(eye_region, 50, 150)
                        eye_edge_density = float(np.sum(eye_edges > 0) / max(1, eye_edges.size))
                        eye_makeup_contrast = min(2.0, eye_edge_density * 4.0)
                    except Exception:
                        eye_makeup_contrast = 0.0
                features['eye_makeup_contrast'] = float(eye_makeup_contrast)

               
                face_aspect_ratio_norm = min(2.0, (h / max(1.0, float(w))) / 1.2)
                features['face_aspect_ratio_norm'] = float(face_aspect_ratio_norm)
            
         
            try:
                from config import Config
                avg_brightness = float(np.mean(gray) / 255.0)
                if avg_brightness < getattr(Config, 'GENDER_CONFIG', {}).get('low_light_brightness_threshold', 0.35):
                    lip_boost = getattr(Config, 'GENDER_CONFIG', {}).get('low_light_lip_boost', 1.30)
                    soft_boost = getattr(Config, 'GENDER_CONFIG', {}).get('low_light_soft_tissue_boost', 1.20)
                    edge_penalty = getattr(Config, 'GENDER_CONFIG', {}).get('low_light_edge_penalty', 0.40)
                   
                    features['lip_color_contrast'] = float(min(2.0, features.get('lip_color_contrast', 0.0) * lip_boost))
                    features['soft_tissue_distribution'] = float(min(2.0, features.get('soft_tissue_distribution', 0.0) * soft_boost))
                  
                    features['edge_intensity'] = float(max(0.0, features.get('edge_intensity', 0.0) * edge_penalty))
            except Exception:
                pass

            return features
        except Exception as e:
            print(f"Erro na extração avançada de características: {e}")
            return {f'feature_{i}': 0.0 for i in range(16)}
    
    def _apply_female_safeguard(self, male_prob: float, features: Dict[str, float]) -> float:
        try:
            from config import Config
            cfg = getattr(Config, 'GENDER_CONFIG', {})
            if not cfg.get('female_safeguard_enabled', True):
                return male_prob

 
            beard_cov = features.get('beard_coverage_ratio', 0.0)
            stubble = features.get('stubble_score', 0.0)
            mustache = features.get('mustache_score', 0.0)
            if beard_cov > 0.15 or stubble > 0.9 or mustache > 0.9:
                return max(male_prob, cfg.get('male_safeguard_floor', 0.65))

            strong_thr = cfg.get('strong_female_feature_threshold', 0.85)
            weak_male_thr = cfg.get('weak_male_feature_threshold', 0.45)
            cap = cfg.get('female_safeguard_cap', 0.55)
            min_strong = cfg.get('female_safeguard_min_strong', 2)
            min_weak = cfg.get('female_safeguard_min_weak', 2)

            strong_female = 0
            for f in ['lip_fullness', 'soft_tissue_distribution', 'lip_color_contrast', 'eye_brow_gap_ratio', 'skin_texture_variance']:
                v = features.get(f, 0.0)
                if not (np.isnan(v) or np.isinf(v)) and v >= strong_thr:
                    strong_female += 1

            weak_male = 0
            for f in ['stubble_score', 'facial_hair_indicators', 'jaw_definition', 'brow_ridge_prominence', 'edge_intensity']:
                v = features.get(f, 1.0)
                if not (np.isnan(v) or np.isinf(v)) and v <= weak_male_thr:
                    weak_male += 1

            if strong_female >= min_strong and weak_male >= min_weak:
                male_prob = min(male_prob, cap)

            return male_prob
        except Exception:
            return male_prob

    def _classify_gender_from_features(self, features: Dict[str, float]) -> float:
        weights = {
            'jaw_strength': 0.25, 'eye_features': 0.15, 'nose_features': 0.12,
            'mouth_features': 0.18, 'face_shape': 0.15, 'skin_texture': 0.08,
            'overall_masculinity': 0.07
        }
        
        masculinity_score = 0.5
        for feature, value in features.items():
            if feature in weights:
                normalized_value = max(-1, min(1, value))
                masculinity_score += normalized_value * weights[feature]
        
        return max(0.1, min(0.9, masculinity_score))
    
    def _classify_gender_with_advanced_ml(self, features: Dict[str, float]) -> float:
     
        try:
            masculine_weights = {
                'facial_structure': 0.10,
                'jaw_definition': 0.18,
                'eye_socket_depth': 0.07,
                'brow_ridge_prominence': 0.14,
                'cheekbone_structure': 0.05,
                'nose_bridge_width': 0.05,
                'chin_shape': 0.10,
                'forehead_slope': 0.04,
                'face_width_ratio': 0.07,
                'overall_bone_structure': 0.12,
                'facial_hair_indicators': 0.26,
                'edge_intensity': 0.12,
            
                'stubble_score': 0.28,
                'beard_coverage_ratio': 0.22,
                'eyebrow_thickness': 0.08,
                'eyebrow_verticality': 0.07,
                'cheekbone_prominence_grad': 0.08,
                'hairline_edge_density': 0.05,
                'nose_bridge_contrast': 0.08,
                'face_aspect_ratio_norm': 0.04
            }
            
            feminine_weights = {
                'lip_fullness': 0.12,
                'soft_tissue_distribution': 0.24,
                'skin_texture_variance': 0.16,
                'hormonal_markers': 0.14,
           
                'lip_color_contrast': 0.26,
                'eye_brow_gap_ratio': 0.20
            }
            
            masculine_score = 0.0
            feminine_score = 0.0
            masculine_count = 0
            feminine_count = 0
            
            for feature, weight in masculine_weights.items():
                if feature in features:
                    value = features[feature]
                    if not np.isnan(value) and not np.isinf(value):
                        normalized_value = max(0, min(2, value))
                        if normalized_value > 0.35:
                            masculine_score += normalized_value * weight
                            masculine_count += 1
            
            for feature, weight in feminine_weights.items():
                if feature in features:
                    value = features[feature]
                    if not np.isnan(value) and not np.isinf(value):
                        normalized_value = max(0, min(2, value))
                        if normalized_value > 0.28:
                            feminine_score += normalized_value * weight
                            feminine_count += 1
            
            if masculine_count == 0 and feminine_count == 0:
                return 0.5
            
            if masculine_count > 0:
                masculine_score /= masculine_count
            if feminine_count > 0:
                feminine_score /= feminine_count
            
            evidence_strength = masculine_score + feminine_score
            if evidence_strength < 0.1:
                return 0.5
            
            raw_male_prob = masculine_score / (masculine_score + feminine_score + 1e-8)
            
            confidence_multiplier = min(2.0, evidence_strength)
            adjusted_prob = 0.5 + (raw_male_prob - 0.5) * confidence_multiplier

          
            male_feature_boost = 0.0
            from config import Config
            for feature, weight in [
                ('facial_hair_indicators', 0.30),
                ('stubble_score', 0.30),
                ('beard_coverage_ratio', 0.24),
                ('jaw_definition', 0.22),
                ('brow_ridge_prominence', 0.18),
                ('overall_bone_structure', 0.16),
                ('edge_intensity', 0.12)
            ]:
                if feature in features:
                    value = features[feature]
                    if not np.isnan(value) and not np.isinf(value):
                        normalized_value = max(0.0, min(2.0, value))
                        feature_scale = getattr(Config, 'GENDER_CONFIG', {}).get('feature_scale', 0.02)
                        feature_baseline = getattr(Config, 'GENDER_CONFIG', {}).get('feature_baseline', 0.30)
                        male_feature_boost += max(0.0, (normalized_value - feature_baseline)) * weight * feature_scale

      
            female_feature_penalty = 0.0
            from config import Config
            for feature, weight in [
                ('lip_fullness', 0.15),
                ('soft_tissue_distribution', 0.18),
                ('skin_texture_variance', 0.10),
                ('hormonal_markers', 0.12)
            ]:
                if feature in features:
                    value = features[feature]
                    if not np.isnan(value) and not np.isinf(value):
                        normalized_value = max(0.0, min(2.0, value))
                        feature_scale = getattr(Config, 'GENDER_CONFIG', {}).get('feature_scale', 0.02)
                        feature_baseline = getattr(Config, 'GENDER_CONFIG', {}).get('feature_baseline', 0.30)
                        female_feature_penalty += max(0.0, (normalized_value - feature_baseline)) * weight * feature_scale

            adjusted_prob += (
                min(getattr(Config, 'GENDER_CONFIG', {}).get('male_feature_boost_max', 0.05), male_feature_boost)
                - min(getattr(Config, 'GENDER_CONFIG', {}).get('female_feature_penalty_max', 0.05), female_feature_penalty)
            )
            
         
            adjusted_prob = self._apply_female_safeguard(adjusted_prob, features)

            try:
                beard_cov = features.get('beard_coverage_ratio', 0.0)
                stubble = features.get('stubble_score', 0.0)
                mustache = features.get('mustache_score', 0.0)
                no_beard = (beard_cov < 0.08) and (stubble < 0.75) and (mustache < 0.75)
                fem_strong = 0
                for f in ['lip_fullness', 'soft_tissue_distribution', 'lip_color_contrast', 'eye_brow_gap_ratio']:
                    v = features.get(f, 0.0)
                    if not (np.isnan(v) or np.isinf(v)) and v > 0.85:
                        fem_strong += 1
                if no_beard and fem_strong >= 2:
                    adjusted_prob = max(0.05, adjusted_prob - 0.08)
            except Exception:
                pass

           
            try:
                from config import Config
                cfg = getattr(Config, 'GENDER_CONFIG', {})
                beard_thr = cfg.get('beard_coverage_threshold', 0.12)
                stubble_thr_mod = cfg.get('stubble_moderate_threshold', 0.65)
                stubble_thr_str = cfg.get('stubble_strong_threshold', 0.85)
                must_thr_mod = cfg.get('mustache_moderate_threshold', 0.60)
                must_thr_str = cfg.get('mustache_strong_threshold', 0.85)
                require_multi = cfg.get('beard_require_multi_indicators', True)
                min_evidence = int(cfg.get('beard_evidence_min', 2))
                low_light_scale = float(cfg.get('beard_low_light_scale', 1.25))
                lip_block_thr = float(cfg.get('beard_lip_contrast_block_threshold', 0.95))
                soft_block_thr = float(cfg.get('beard_soft_tissue_block_threshold', 0.95))
                jaw_dark_thr_mod = float(cfg.get('beard_jaw_darkness_threshold', 0.20))
                jaw_dark_thr_str = float(cfg.get('beard_jaw_darkness_strong_threshold', 0.25))

                beard_cov = float(features.get('beard_coverage_ratio', 0.0))
                stubble = float(features.get('stubble_score', 0.0))
                mustache = float(features.get('mustache_score', 0.0))
                lip_contrast = float(features.get('lip_color_contrast', 0.0))
                soft_tissue = float(features.get('soft_tissue_distribution', 0.0))
                jaw_dark = float(features.get('jaw_darkness_ratio', 0.0))

               
                avg_brightness = float(features.get('avg_brightness_norm', 0.5))
                if avg_brightness < getattr(Config, 'GENDER_CONFIG', {}).get('low_light_brightness_threshold', 0.35):
                    beard_thr *= low_light_scale
                    stubble_thr_mod *= low_light_scale
                    stubble_thr_str *= low_light_scale
                    must_thr_mod *= low_light_scale
                    must_thr_str *= low_light_scale

               
                evidence = [
                    beard_cov > beard_thr,
                    stubble > stubble_thr_mod,
                    mustache > must_thr_mod
                ]
                strong_evidence = [
                    (beard_cov > beard_thr * 1.3) and (jaw_dark > jaw_dark_thr_str),
                    (stubble > stubble_thr_str) and (jaw_dark > jaw_dark_thr_str),
                    (mustache > must_thr_str) and (jaw_dark > jaw_dark_thr_str)
                ]

                strong_count = sum(1 for e in strong_evidence if e)
                evidence_count = sum(1 for e in evidence if e)
                strong_beard = (strong_count >= 1 and jaw_dark > jaw_dark_thr_str) if not require_multi else (strong_count >= 1 and evidence_count >= min_evidence and jaw_dark > jaw_dark_thr_str)
                moderate_beard = (evidence_count >= 1 and jaw_dark > jaw_dark_thr_mod) if not require_multi else (evidence_count >= min_evidence and jaw_dark > jaw_dark_thr_mod)

               
                block_beard = (lip_contrast > lip_block_thr and soft_tissue > soft_block_thr)

                if strong_beard and not block_beard:
                    floor_str = cfg.get('beard_male_floor_strong', 0.82)
                    boost_str = cfg.get('beard_male_boost_strong', 0.08)
                    adjusted_prob = min(0.98, max(adjusted_prob, floor_str) + boost_str)
                elif moderate_beard and not block_beard:
                    floor_mod = cfg.get('beard_male_floor_moderate', 0.72)
                    boost_mod = cfg.get('beard_male_boost_moderate', 0.04)
                    adjusted_prob = min(0.96, max(adjusted_prob, floor_mod) + boost_mod)
            except Exception:
                pass

           
            from config import Config
            cfg = getattr(Config, 'GENDER_CONFIG', {})
            sf_thr = cfg.get('strong_female_feature_threshold', 0.9)
            wm_thr = cfg.get('weak_male_feature_threshold', 0.4)
            sf_min = cfg.get('female_safeguard_min_strong', 2)
            wm_min = cfg.get('female_safeguard_min_weak', 2)
            strong_female = 0
            for f in ['lip_fullness', 'soft_tissue_distribution', 'lip_color_contrast', 'eye_brow_gap_ratio']:
                v = features.get(f, 0.0)
                if not (np.isnan(v) or np.isinf(v)) and v > sf_thr:
                    strong_female += 1
            weak_male = 0
            for f in ['stubble_score', 'facial_hair_indicators', 'jaw_definition', 'brow_ridge_prominence']:
                v = features.get(f, 1.0)
                if not (np.isnan(v) or np.isinf(v)) and v < wm_thr:
                    weak_male += 1
            if strong_female >= sf_min and weak_male >= wm_min:
                adjusted_prob = min(adjusted_prob, cfg.get('female_safeguard_cap', 0.55))
            

            strong_male = 0
            sm_thr = cfg.get('strong_male_feature_threshold', 0.92)
            wf_thr = cfg.get('weak_female_feature_threshold', 0.45)
            for f in ['stubble_score', 'facial_hair_indicators', 'jaw_definition', 'brow_ridge_prominence']:
                v = features.get(f, 0.0)
                if not (np.isnan(v) or np.isinf(v)) and v > sm_thr:
                    strong_male += 1
            weak_female = 0
            for f in ['lip_fullness', 'soft_tissue_distribution', 'lip_color_contrast', 'eye_brow_gap_ratio']:
                v = features.get(f, 1.0)
                if not (np.isnan(v) or np.isinf(v)) and v < wf_thr:
                    weak_female += 1
            ms_min = cfg.get('male_safeguard_min_strong', 2)
            wf_min = cfg.get('male_safeguard_min_weak', 1)
            if strong_male >= ms_min and weak_female >= wf_min:
                adjusted_prob = max(adjusted_prob, cfg.get('male_safeguard_floor', 0.6))



            return max(0.05, min(0.95, adjusted_prob))
            



        except Exception as e:
            print(f"Erro na classificação avançada de gênero: {e}")
            return 0.5
    
    def _get_geometric_gender_adjustment(self, face_img: np.ndarray) -> float:
       
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            h, w = gray.shape
            
            if h < 60 or w < 60:
                return 0.5
            
           
            jaw_region = gray[int(h*0.65):h, int(w*0.1):int(w*0.9)]
            if jaw_region.size > 0:
                jaw_variance = np.var(jaw_region)
                jaw_edges = cv2.Canny(jaw_region, 30, 100)
                jaw_edge_density = np.sum(jaw_edges > 0) / jaw_edges.size
               
                try:
                    jaw_lap = cv2.Laplacian(jaw_region, cv2.CV_64F).var()
                    geo_stubble_score = min(2.0, jaw_lap / 150.0)
                except Exception:
                    geo_stubble_score = 0.0
            else:
                jaw_variance = 0.0
                jaw_edge_density = 0.0
                geo_stubble_score = 0.0
            
           
            forehead_region = gray[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
            if forehead_region.size > 0:
                forehead_smoothness = 1.0 / (1.0 + np.var(forehead_region))
            else:
                forehead_smoothness = 0.5
            
           
            geometric_male_score = (
                jaw_variance * 0.35 +
                jaw_edge_density * 0.35 +
                geo_stubble_score * 0.15 +
                (1.0 - forehead_smoothness) * 0.15
            )
            

            geometric_male_score = np.tanh(geometric_male_score) * 0.5 + 0.5
            return max(0.1, min(0.9, geometric_male_score))
            
        except Exception as e:
            print(f"Erro na análise geométrica: {e}")
            return 0.5
    
    def _predict_with_cnn(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
           
            img_hash = hash(face_img.tobytes())
            if img_hash in self._prediction_cache:
                return self._prediction_cache[img_hash]
            
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
           
            if hasattr(self, '_optimized_predict'):
                prediction = self._optimized_predict(face_expanded)[0][0].numpy()
            else:
                prediction = self.gender_model.predict(face_expanded, verbose=0)[0][0]

            from config import Config
           
            cnn_threshold = getattr(Config, 'GENDER_CONFIG', {}).get('cnn_threshold', 0.56)
            confidence = abs(prediction - 0.5) * 2
            gender = 'Masculino' if prediction > cnn_threshold else 'Feminino'
            
            result = {
                'method': 'cnn_transfer_learning',
                'male_probability': float(prediction),
                'confidence': float(min(0.92, max(0.5, confidence))),
                'gender': gender
            }
            
           
            if len(self._prediction_cache) < 100:
                self._prediction_cache[img_hash] = result
            
            return result
        except Exception as e:
            print(f"Erro CNN: {e}")
            return {'method': 'cnn', 'male_probability': 0.48, 'confidence': 0.5, 'gender': 'Feminino'}
    
    def _predict_traditional(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            if h < 50 or w < 50:
                return {'method': 'traditional', 'male_probability': 0.48, 'confidence': 0.5, 'gender': 'Feminino'}
            
            jaw_region = gray[int(h*0.62):h, int(w*0.08):int(w*0.92)]
            if jaw_region.size > 100:
                jaw_edges = cv2.Canny(jaw_region, 30, 120)
                jaw_strength = np.sum(jaw_edges > 0) / max(1, jaw_edges.size)
                
                jaw_contours, _ = cv2.findContours(jaw_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                jaw_definition = len([c for c in jaw_contours if cv2.contourArea(c) > 50]) / max(1, len(jaw_contours))
                
                jaw_moments = cv2.moments(jaw_edges)
                jaw_asymmetry = abs(jaw_moments['m10'] / max(1, jaw_moments['m00']) - w/2) / (w/2) if jaw_moments['m00'] > 0 else 0

                try:
                    jaw_lap = cv2.Laplacian(jaw_region, cv2.CV_64F).var()
                    stubble_trad = min(2.0, jaw_lap / 130.0)
                except Exception:
                    stubble_trad = 0.0
            else:
                jaw_strength = 0.0
                jaw_definition = 0.0
                jaw_asymmetry = 0.0
                stubble_trad = 0.0
            
            eyebrow_region = gray[int(h*0.12):int(h*0.35), int(w*0.1):int(w*0.9)]
            if eyebrow_region.size > 50:
                eyebrow_edges = cv2.Canny(eyebrow_region, 25, 75)
                eyebrow_density = np.sum(eyebrow_edges > 0) / max(1, eyebrow_edges.size)
                eyebrow_thickness = np.std(eyebrow_region.astype(np.float32))
                
                sobel_x = cv2.Sobel(eyebrow_region, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(eyebrow_region, cv2.CV_64F, 0, 1, ksize=3)
                eyebrow_gradient = np.sqrt(sobel_x**2 + sobel_y**2)
                eyebrow_sharpness = np.mean(eyebrow_gradient)
            else:
                eyebrow_density = 0.0
                eyebrow_thickness = 0.0
                eyebrow_sharpness = 0.0
            
            lip_region = gray[int(h*0.62):int(h*0.82), int(w*0.25):int(w*0.75)]
            if lip_region.size > 50:
                lip_edges = cv2.Canny(lip_region, 20, 60)
                lip_definition = np.sum(lip_edges > 0) / max(1, lip_edges.size)
                
                lip_hist = cv2.calcHist([lip_region], [0], None, [256], [0, 256])
                lip_contrast_score = np.std(lip_hist)
                
                lip_contours, _ = cv2.findContours(lip_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                lip_fullness = sum(cv2.contourArea(c) for c in lip_contours) / max(1, lip_region.size)
            else:
                lip_definition = 0.0
                lip_contrast_score = 0.0
                lip_fullness = 0.0
            
            nose_region = gray[int(h*0.35):int(h*0.62), int(w*0.35):int(w*0.65)]
            if nose_region.size > 50:
                nose_edges = cv2.Canny(nose_region, 40, 100)
                nose_definition = np.sum(nose_edges > 0) / max(1, nose_edges.size)
                nose_width_variance = np.var([np.sum(nose_edges[i, :] > 0) for i in range(nose_edges.shape[0])])
            else:
                nose_definition = 0.0
                nose_width_variance = 0.0
            
            cheek_left = gray[int(h*0.4):int(h*0.7), int(w*0.05):int(w*0.4)]
            cheek_right = gray[int(h*0.4):int(h*0.7), int(w*0.6):int(w*0.95)]
            
            if cheek_left.size > 50 and cheek_right.size > 50:
                cheek_texture_left = cv2.Laplacian(cheek_left, cv2.CV_64F).var()
                cheek_texture_right = cv2.Laplacian(cheek_right, cv2.CV_64F).var()
                cheek_asymmetry = abs(cheek_texture_left - cheek_texture_right) / max(1, max(cheek_texture_left, cheek_texture_right))
                cheek_prominence = (cheek_texture_left + cheek_texture_right) / 2
            else:
                cheek_asymmetry = 0.0
                cheek_prominence = 0.0
            
            skin_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            skin_smoothness = 1.0 / (1.0 + skin_variance / 150.0)
            
            face_ratio = w / max(1, h)
            
            forehead_region = gray[int(h*0.05):int(h*0.3), int(w*0.2):int(w*0.8)]
            if forehead_region.size > 50:
                forehead_texture = cv2.Laplacian(forehead_region, cv2.CV_64F).var()
                forehead_prominence = np.std(forehead_region)
            else:
                forehead_texture = 0.0
                forehead_prominence = 0.0
            
            masculine_total = 0.0
            feminine_total = 0.0
            
            if jaw_strength > 0.35:
                masculine_total += jaw_strength * 4.0
            if jaw_definition > 0.4:
                masculine_total += jaw_definition * 3.5
            if eyebrow_density > 0.3:
                masculine_total += eyebrow_density * 3.0
           
            if lip_fullness > 0.025 and skin_smoothness > 0.78:
                masculine_total -= min(0.8, eyebrow_density * 0.8)
            if eyebrow_thickness > 25:
                masculine_total += min(3.0, eyebrow_thickness / 20.0)
            if eyebrow_sharpness > 15:
                masculine_total += min(2.5, eyebrow_sharpness / 10.0)
            if skin_variance > 120:
                masculine_total += min(2.0, skin_variance / 150.0)
            if face_ratio > 0.88:
                masculine_total += (face_ratio - 0.88) * 4.0
            if nose_definition > 0.3:
                masculine_total += nose_definition * 2.0
            if forehead_prominence > 30:
                masculine_total += min(2.0, forehead_prominence / 25.0)
            
            if skin_smoothness > 0.65:
                feminine_total += skin_smoothness * 3.5
            if lip_definition > 0.25:
                feminine_total += lip_definition * 3.0
            if lip_contrast_score > 400:
                feminine_total += min(2.5, lip_contrast_score / 300.0)
            if lip_fullness > 0.02:
                feminine_total += lip_fullness * 140.0
            if jaw_strength < 0.25:
                feminine_total += (0.25 - jaw_strength) * 3.0
            if eyebrow_density < 0.2:
                feminine_total += (0.2 - eyebrow_density) * 2.2
            if face_ratio < 0.85:
                feminine_total += (0.85 - face_ratio) * 2.0
            if cheek_prominence > 50:
                feminine_total += min(2.0, cheek_prominence / 40.0)
            
           
            try:
                from config import Config
                cfg = getattr(Config, 'GENDER_CONFIG', {})
                trad_stubble_weight = cfg.get('traditional_stubble_weight', 2.5)
                masculine_total += stubble_trad * float(trad_stubble_weight)
            except Exception:
                masculine_total += stubble_trad * 2.0
           
            try:
                no_beard_trad = stubble_trad < 0.6
                fem_count = 0
                if skin_smoothness > 0.78: fem_count += 1
                if lip_definition > 0.30: fem_count += 1
                if lip_fullness > 0.022: fem_count += 1
                if eyebrow_density < 0.22: fem_count += 1
                if no_beard_trad and fem_count >= 2:
                    feminine_total += 1.0
            except Exception:
                pass

            mustache_region = gray[int(h*0.52):int(h*0.62), int(w*0.30):int(w*0.70)]
            if mustache_region.size > 0:
                try:
                    must_lap = cv2.Laplacian(mustache_region, cv2.CV_64F).var()
                    masculine_total += min(2.0, must_lap / 120.0) * 2.0
                except Exception:
                    pass

            total_evidence = masculine_total + feminine_total
            
            if total_evidence > 0.18:
                masculinity_probability = masculine_total / total_evidence
            else:
                if jaw_strength > 0.4 or eyebrow_density > 0.35:
                    masculinity_probability = 0.72
                elif skin_smoothness > 0.75 or lip_definition > 0.35:
                    masculinity_probability = 0.28
                else:
                    masculinity_probability = 0.5
            
            if jaw_strength > 0.45 and eyebrow_density > 0.35:
                masculinity_probability = min(0.95, masculinity_probability + 0.25)
            elif jaw_strength > 0.4 and eyebrow_sharpness > 20:
                masculinity_probability = min(0.92, masculinity_probability + 0.2)
            elif eyebrow_density > 0.4 and nose_definition > 0.35:
                masculinity_probability = min(0.88, masculinity_probability + 0.15)
            
            elif skin_smoothness > 0.8 and lip_definition > 0.35:
                masculinity_probability = max(0.05, masculinity_probability - 0.25)
            elif lip_contrast_score > 500 and lip_fullness > 0.025:
                masculinity_probability = max(0.08, masculinity_probability - 0.2)
            elif skin_smoothness > 0.75 and jaw_strength < 0.2:
                masculinity_probability = max(0.12, masculinity_probability - 0.15)
            
            masculinity_probability = max(0.01, min(0.99, masculinity_probability))

         
            try:
                from config import Config
                cfg = getattr(Config, 'GENDER_CONFIG', {})
                sf_thr = cfg.get('strong_female_feature_threshold', 0.88)
                wm_thr = cfg.get('weak_male_feature_threshold', 0.45)
                sm_thr = cfg.get('strong_male_feature_threshold', 0.92)
                wf_thr = cfg.get('weak_female_feature_threshold', 0.45)
                female_cap = cfg.get('female_safeguard_cap', 0.55)
                male_floor = cfg.get('male_safeguard_floor', 0.62)

                strong_female = 0
                if skin_smoothness > 0.80: strong_female += 1
                if lip_definition > 0.35: strong_female += 1
                if lip_fullness > 0.02: strong_female += 1

                weak_male = 0
                if jaw_strength < 0.25: weak_male += 1
                if eyebrow_density < 0.20: weak_male += 1

                if strong_female >= 2 and weak_male >= 1:
                    masculinity_probability = min(masculinity_probability, female_cap)

                strong_male = 0
                if jaw_strength > 0.45: strong_male += 1
                if eyebrow_density > 0.40: strong_male += 1
                if nose_definition > 0.35: strong_male += 1

                weak_female = 0
                if skin_smoothness < 0.70: weak_female += 1
                if lip_definition < 0.25: weak_female += 1

                if strong_male >= 2 and weak_female >= 1:
                    masculinity_probability = max(masculinity_probability, male_floor)
            except Exception:
                pass
            
            if np.isnan(masculinity_probability):
                masculinity_probability = 0.5
            
            female_probability = 1.0 - masculinity_probability


            try:
                from config import Config
                tie_margin = getattr(Config, 'GENDER_CONFIG', {}).get('traditional_tie_margin', 0.06)
                tie_mode = getattr(Config, 'GENDER_CONFIG', {}).get('traditional_tie_mode', 'indefinido')
            except Exception:
                tie_margin = 0.06
                tie_mode = 'indefinido'

            beard_strong_trad = stubble_trad >= 0.9
            if abs(masculinity_probability - female_probability) < tie_margin:
                if beard_strong_trad:
                    gender = 'Masculino'
                    confidence_base = masculinity_probability
                else:
                    tm = str(tie_mode).lower()
                    if tm == 'feminino':
                        gender = 'Feminino'
                        confidence_base = female_probability
                    elif tm == 'masculino':
                        gender = 'Masculino'
                        confidence_base = masculinity_probability
                    else:
                        gender = 'Indefinido'
                        confidence_base = 0.5
            else:
                if masculinity_probability > female_probability:
                    gender = 'Masculino'
                    confidence_base = masculinity_probability
                else:
                    gender = 'Feminino'
                    confidence_base = female_probability
            
            distinctive_features = 0
            if jaw_strength > 0.4: distinctive_features += 1
            if eyebrow_density > 0.35: distinctive_features += 1
            if skin_smoothness > 0.75: distinctive_features += 1
            if lip_definition > 0.3: distinctive_features += 1
            
            confidence_boost = min(0.25, distinctive_features * 0.08)
            confidence = max(0.4, min(0.95, confidence_base + confidence_boost))
            
            print(f"🔍 Enhanced Gender: M={masculinity_probability:.3f}, F={female_probability:.3f}, "
                  f"Jaw={jaw_strength:.3f}, Eyebrow={eyebrow_density:.3f}, "
                  f"Skin={skin_variance:.1f}, Result={gender}")
            
            return {
                'method': 'traditional_enhanced_v4',
                'male_probability': float(masculinity_probability),
                'female_probability': float(female_probability),
                'confidence': float(confidence),
                'gender': gender,
                'feature_analysis': {
                    'jaw_strength': float(jaw_strength),
                    'jaw_definition': float(jaw_definition),
                    'eyebrow_density': float(eyebrow_density),
                    'eyebrow_sharpness': float(eyebrow_sharpness),
                    'skin_variance': float(skin_variance),
                    'lip_definition': float(lip_definition),
                    'lip_fullness': float(lip_fullness),
                    'nose_definition': float(nose_definition),
                    'face_ratio': float(face_ratio),
                    'masculine_total': float(masculine_total),
                    'feminine_total': float(feminine_total),
                    'distinctive_features': distinctive_features
                },
                'debug_features': {
                    'stubble_trad': float(stubble_trad),
                    'mustache_var': float(min(2.0, (cv2.Laplacian(gray[int(h*0.52):int(h*0.62), int(w*0.30):int(w*0.70)], cv2.CV_64F).var() if h>0 and w>0 else 0.0) / 120.0))
                }
            }
            
        except Exception as e:
            print(f"❌ Erro tradicional melhorado: {e}")
            return {'method': 'traditional', 'male_probability': 0.48, 'confidence': 0.5, 'gender': 'Feminino'}
    
    def _ensemble_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            from config import Config
            default_weights = {
                'facenet_advanced': 0.55,
                'cnn_transfer_learning': 0.35,
                'traditional_enhanced_v4': 0.10
            }
            cfg_weights = getattr(Config, 'GENDER_CONFIG', {}).get('ensemble_method_weights', default_weights)
            method_weights = {**default_weights, **cfg_weights}
            
            weighted_probability = 0.0
            total_weight = 0.0
            all_confidences = []
            methods_used = []
            
            for pred in predictions:
                method = pred.get('method', 'unknown')
                probability = pred.get('male_probability', 0.52)
                confidence = pred.get('confidence', 0.6)
                
                if np.isnan(probability) or np.isinf(probability):
                    probability = 0.52
                if np.isnan(confidence) or np.isinf(confidence):
                    confidence = 0.6
                
                if 'facenet' in method:
                    method = 'facenet_advanced'
                if method.startswith('traditional'):
                    method = 'traditional_enhanced_v4'
                
                if method in method_weights:
                    weight = method_weights[method] * confidence
                    weighted_probability += probability * weight
                    total_weight += weight
                    all_confidences.append(confidence)
                    methods_used.append(method)
            
            if total_weight > 0:
                final_probability = weighted_probability / total_weight
            else:
                final_probability = 0.52
            
            final_confidence = np.mean(all_confidences) if all_confidences else 0.6
            
            if len(methods_used) > 1:
                probabilities = [p['male_probability'] for p in predictions if not (np.isnan(p.get('male_probability', 0.52)) or np.isinf(p.get('male_probability', 0.52)))]
                if probabilities:
                    agreement = 1.0 - np.std(probabilities)
                    from config import Config
                    agreement_mult = getattr(Config, 'GENDER_CONFIG', {}).get('ensemble_agreement_multiplier', 0.35)
                    final_confidence = min(0.98, final_confidence * (1 + agreement * agreement_mult))
            
            if np.isnan(final_probability) or np.isinf(final_probability):
                final_probability = 0.52
            if np.isnan(final_confidence) or np.isinf(final_confidence):
                final_confidence = 0.6
            
            # Salvaguarda final pró-feminino quando não há barba e há múltiplos indícios femininos fortes
            try:
                avg_beard_cov = np.mean([
                    p.get('debug_features', {}).get('beard_coverage_ratio', 0.0) for p in predictions
                ]) if predictions else 0.0
                avg_stubble = np.mean([
                    p.get('debug_features', {}).get('stubble_score', p.get('debug_features', {}).get('stubble_trad', 0.0)) for p in predictions
                ]) if predictions else 0.0
                avg_mustache = np.mean([
                    p.get('debug_features', {}).get('mustache_score', p.get('debug_features', {}).get('mustache_var', 0.0)) for p in predictions
                ]) if predictions else 0.0

                no_beard_global = (avg_beard_cov < 0.08) and (avg_stubble < 0.6) and (avg_mustache < 0.6)

                fem_votes = 0
                fem_votes += 1 if np.mean([p.get('debug_features', {}).get('lip_color_contrast', 0.0) for p in predictions if p.get('debug_features')]) > 0.9 else 0
                fem_votes += 1 if np.mean([p.get('debug_features', {}).get('soft_tissue_distribution', 0.0) for p in predictions if p.get('debug_features')]) > 0.9 else 0

                if no_beard_global and fem_votes >= 2:
                    final_probability = min(final_probability, getattr(Config, 'GENDER_CONFIG', {}).get('no_beard_female_cap', 0.48))
            except Exception:
                pass

         
            from config import Config
            if getattr(Config, 'GENDER_CONFIG', {}).get('invert_output', False):
                final_probability = 1.0 - final_probability
            final_probability = max(0.01, min(0.99, final_probability))
            final_female_probability = 1.0 - final_probability
            
            from config import Config

            threshold = getattr(Config, 'GENDER_CONFIG', {}).get('global_threshold', 0.6)

         
            sep_base = getattr(Config, 'GENDER_CONFIG', {}).get('confidence_separation_base', 0.6)
            sep_slope = getattr(Config, 'GENDER_CONFIG', {}).get('confidence_separation_slope', 0.8)
            separation = abs(final_probability - threshold)
            final_confidence = max(final_confidence, min(0.97, sep_base + separation * sep_slope))

            try:
                cfg = getattr(Config, 'GENDER_CONFIG', {})
                beard_thr = cfg.get('beard_coverage_threshold', 0.12)
                stubble_thr_str = cfg.get('stubble_strong_threshold', 0.85)
                must_thr_str = cfg.get('mustache_strong_threshold', 0.85)
                lip_block_thr = float(cfg.get('beard_lip_contrast_block_threshold', 0.95))
                soft_block_thr = float(cfg.get('beard_soft_tissue_block_threshold', 0.95))

                avg_lip = np.mean([p.get('debug_features', {}).get('lip_color_contrast', 0.0) for p in predictions if p.get('debug_features')]) if predictions else 0.0
                avg_soft = np.mean([p.get('debug_features', {}).get('soft_tissue_distribution', 0.0) for p in predictions if p.get('debug_features')]) if predictions else 0.0

                strong_beard_global = (avg_beard_cov > beard_thr * 1.3) or (avg_stubble > stubble_thr_str) or (avg_mustache > must_thr_str)
                block_global = (avg_lip > lip_block_thr and avg_soft > soft_block_thr)
                if strong_beard_global and not block_global:
                    final_probability = max(final_probability, cfg.get('beard_global_male_floor', 0.80))
            except Exception:
                pass

            if final_probability > threshold:
                gender = 'Masculino'
            else:
                gender = 'Feminino'
      
            methods_agree = len(set([pred['gender'] for pred in predictions])) == 1
            if methods_agree and final_confidence < 0.90:
                final_confidence = min(0.95, final_confidence + 0.10)
            prob_margin = abs(np.mean([p['male_probability'] for p in predictions]) - 0.5)
            if prob_margin > 0.25:
                final_confidence = min(0.95, final_confidence + 0.05)
           
            
            return {
                'predicted_gender': gender,
                'male_probability': float(final_probability),
                'female_probability': float(final_female_probability),
                'confidence': float(final_confidence),
                'ensemble_details': {
                    'methods_used': methods_used,
                    'agreement_score': float(1.0 - np.std([p['male_probability'] for p in predictions]))
                }
            }
        except Exception as e:
            print(f"Erro no ensemble: {e}")
            return {
                'predicted_gender': 'Feminino',
                'male_probability': 0.48,
                'female_probability': 0.52,
                'confidence': 0.5
            }

class AdvancedEthnicityDetector:
    def __init__(self):
       
        self.ml_enabled = False
        self._ethnicity_cache = {}
        self._prediction_cache = {}
        self._cache_size_limit = 50
        self.ethnicity_classes = [
            'Caucasiano', 'Asiático', 'Africano', 'Hispânico', 'Árabe', 'Indiano',
            'Japonês', 'Coreano', 'Chinês', 'Tailandês', 'Filipino', 'Vietnamita',
            'Brasileiro', 'Mexicano', 'Turco', 'Persa', 'Judeu', 'Eslavo',
            'Mediterrâneo', 'Nórdico', 'Aborigene', 'Polinésio', 'Misto'
        ]
        self.ethnicity_features_weights = {
            'nose_width_ratio': 0.20,
            'eye_distance_ratio': 0.18,
            'face_width_height_ratio': 0.15,
            'lip_thickness': 0.12,
            'skin_tone_analysis': 0.20,
            'cheekbone_prominence': 0.08,
            'eye_shape_analysis': 0.07
        }
  
        self.facenet = None
        self.ethnicity_model = None
        self.mtcnn = None
        self.facial_landmarks_model = None
        
    def _cleanup_cache(self):
       
        if len(self._prediction_cache) > self._cache_size_limit:
            items_to_remove = len(self._prediction_cache) // 2
            cache_keys = list(self._prediction_cache.keys())
            for key in cache_keys[:items_to_remove]:
                del self._prediction_cache[key]
        
        if ML_AVAILABLE:
            try:
                self.facenet = FaceNet()
                self.ethnicity_model = self._load_or_create_ethnicity_model()
                self.ml_enabled = True
                self.mtcnn = MTCNN()
                self.facial_landmarks_model = self._init_facial_landmarks()
                self._setup_predict_functions()
            except Exception as e:
                print(f"Erro ao carregar modelos de etnia ML: {e}")
                self.ml_enabled = False
                self.facenet = None
                self.ethnicity_model = None
                self.mtcnn = None
                self.facial_landmarks_model = None
        else:
            self.ml_enabled = False
            self.facenet = None
            self.ethnicity_model = None
            self.mtcnn = None
            self.facial_landmarks_model = None
            
    def _setup_predict_functions(self):
    
        if self.ml_enabled and hasattr(self, 'ethnicity_model'):
            import tensorflow as tf
            
            @tf.function(reduce_retracing=True)
            def optimized_ethnicity_predict(x):
                return self.ethnicity_model(x, training=False)
            
            self._optimized_ethnicity_predict = optimized_ethnicity_predict
    
    def _init_facial_landmarks(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            self.ethnicity_classifier = RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                random_state=42
            )
            self.skin_tone_classifier = LogisticRegression(
                max_iter=1000, 
                random_state=42
            )
            return True
        except:
            return False
    
    def _load_or_create_ethnicity_model(self):
        model_path = 'models/ethnicity_model.h5'
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            return self._create_cnn_ethnicity_model()
    
    def _create_cnn_ethnicity_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            alpha=1.0
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.ethnicity_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        return model
    
    def detect_ethnicity_advanced(self, face_img: np.ndarray) -> Dict[str, Any]:
        if not self.ml_enabled:
            return self._predict_traditional_enhanced(face_img)
        
        try:
            predictions = []
            
            facenet_pred = self._predict_with_facenet_enhanced(face_img)
            predictions.append(facenet_pred)
            
            cnn_pred = self._predict_with_cnn_enhanced(face_img)
            predictions.append(cnn_pred)
            
            geometric_pred = self._predict_with_geometric_analysis(face_img)
            predictions.append(geometric_pred)
            
            skin_tone_pred = self._predict_with_skin_tone_analysis(face_img)
            predictions.append(skin_tone_pred)
            
            facial_structure_pred = self._predict_with_facial_structure(face_img)
            predictions.append(facial_structure_pred)
            
            traditional_pred = self._predict_traditional_enhanced(face_img)
            predictions.append(traditional_pred)
            
            return self._ensemble_ethnicity_predictions_weighted(predictions)
        except Exception as e:
            print(f"Erro no detector de etnia avançado: {e}")
            return self._predict_traditional_enhanced(face_img)
    
    def _predict_with_geometric_analysis(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            h, w = gray.shape
            
            features = {}
            
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
            if len(eyes) >= 2:
                eye1, eye2 = eyes[0], eyes[1]
                eye_distance = abs(eye1[0] - eye2[0])
                features['eye_distance_ratio'] = eye_distance / w
                
                eye_avg_width = (eye1[2] + eye2[2]) / 2
                features['eye_width_ratio'] = eye_avg_width / w
                
                eye_y_avg = (eye1[1] + eye2[1]) / 2
                features['eye_position_ratio'] = eye_y_avg / h
            else:
                features['eye_distance_ratio'] = 0.25
                features['eye_width_ratio'] = 0.08
                features['eye_position_ratio'] = 0.35
            
            nose_region = gray[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
            if nose_region.size > 0:
                nose_edges = cv2.Canny(nose_region, 30, 100)
                nose_width = np.sum(np.any(nose_edges, axis=0))
                features['nose_width_ratio'] = nose_width / w
                
                nose_height = np.sum(np.any(nose_edges, axis=1))
                features['nose_height_ratio'] = nose_height / h
            else:
                features['nose_width_ratio'] = 0.12
                features['nose_height_ratio'] = 0.15
            
            lip_region = gray[int(h*0.65):int(h*0.85), int(w*0.3):int(w*0.7)]
            if lip_region.size > 0:
                lip_edges = cv2.Canny(lip_region, 20, 80)
                lip_thickness = np.sum(np.any(lip_edges, axis=1))
                features['lip_thickness_ratio'] = lip_thickness / h
                
                lip_width = np.sum(np.any(lip_edges, axis=0))
                features['lip_width_ratio'] = lip_width / w
            else:
                features['lip_thickness_ratio'] = 0.05
                features['lip_width_ratio'] = 0.25
            
            features['face_width_height_ratio'] = w / h
            
            cheekbone_region = gray[int(h*0.4):int(h*0.6), int(w*0.1):int(w*0.9)]
            if cheekbone_region.size > 0:
                cheekbone_prominence = np.std(cheekbone_region.astype(np.float32))
                features['cheekbone_prominence'] = cheekbone_prominence / 50.0
            else:
                features['cheekbone_prominence'] = 0.5
            
            ethnicity_scores = self._calculate_ethnicity_from_geometry(features)
            
            predicted_ethnicity = max(ethnicity_scores, key=ethnicity_scores.get)
            confidence = ethnicity_scores[predicted_ethnicity]
            
            return {
                'method': 'geometric_analysis',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in ethnicity_scores.items()},
                'features': features
            }
        except Exception as e:
            return {
                'method': 'geometric_analysis',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {cls: 1.0/len(self.ethnicity_classes) for cls in self.ethnicity_classes}
            }
    
    def _calculate_ethnicity_from_geometry(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {ethnicity: 0.0 for ethnicity in self.ethnicity_classes}
        
        eye_distance = features.get('eye_distance_ratio', 0.25)
        nose_width = features.get('nose_width_ratio', 0.12)
        lip_thickness = features.get('lip_thickness_ratio', 0.05)
        face_ratio = features.get('face_width_height_ratio', 0.8)
        cheekbone = features.get('cheekbone_prominence', 0.5)
        
        if eye_distance > 0.28:
            scores['Asiático'] += 0.3
            scores['Caucasiano'] += 0.1
        elif eye_distance < 0.22:
            scores['Africano'] += 0.2
            scores['Árabe'] += 0.15
        
        if nose_width > 0.15:
            scores['Africano'] += 0.35
            scores['Hispânico'] += 0.2
        elif nose_width < 0.1:
            scores['Asiático'] += 0.25
            scores['Caucasiano'] += 0.15
        else:
            scores['Indiano'] += 0.2
            scores['Árabe'] += 0.2
        
        if lip_thickness > 0.08:
            scores['Africano'] += 0.3
            scores['Hispânico'] += 0.15
        elif lip_thickness < 0.04:
            scores['Asiático'] += 0.2
            scores['Caucasiano'] += 0.1
        
        if face_ratio > 0.85:
            scores['Asiático'] += 0.2
        elif face_ratio < 0.75:
            scores['Africano'] += 0.15
            scores['Caucasiano'] += 0.1
        
        if cheekbone > 0.7:
            scores['Asiático'] += 0.25
            scores['Indiano'] += 0.15
        elif cheekbone > 0.5:
            scores['Caucasiano'] += 0.15
            scores['Árabe'] += 0.1
        
        for ethnicity in scores:
            if scores[ethnicity] < 0.1:
                scores[ethnicity] = 0.1
        
        scores['Misto'] = np.mean(list(scores.values())) * 0.8
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/len(scores) for k in scores}
        
        return scores
    
    def _predict_with_skin_tone_analysis(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            
            h, w = face_img.shape[:2]
            
            skin_regions = [
                face_img[int(h*0.3):int(h*0.6), int(w*0.2):int(w*0.8)],
                face_img[int(h*0.45):int(h*0.75), int(w*0.15):int(w*0.85)],
                face_img[int(h*0.2):int(h*0.5), int(w*0.1):int(w*0.9)]
            ]
            
            skin_features = {}
            
            all_pixels = []
            for region in skin_regions:
                if region.size > 0:
                    all_pixels.extend(region.reshape(-1, 3))
            
            if all_pixels:
                all_pixels = np.array(all_pixels)
                
                skin_features['mean_b'] = np.mean(all_pixels[:, 0])
                skin_features['mean_g'] = np.mean(all_pixels[:, 1])
                skin_features['mean_r'] = np.mean(all_pixels[:, 2])
                
                hsv_pixels = cv2.cvtColor(all_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)
                hsv_flat = hsv_pixels.reshape(-1, 3)
                
                skin_features['hue_mean'] = np.mean(hsv_flat[:, 0])
                skin_features['saturation_mean'] = np.mean(hsv_flat[:, 1])
                skin_features['value_mean'] = np.mean(hsv_flat[:, 2])
                
                lab_pixels = cv2.cvtColor(all_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
                lab_flat = lab_pixels.reshape(-1, 3)
                
                skin_features['l_mean'] = np.mean(lab_flat[:, 0])
                skin_features['a_mean'] = np.mean(lab_flat[:, 1])
                skin_features['b_lab_mean'] = np.mean(lab_flat[:, 2])
                
                brightness = (skin_features['mean_r'] + skin_features['mean_g'] + skin_features['mean_b']) / 3
                skin_features['brightness'] = brightness
                
                skin_features['rg_ratio'] = skin_features['mean_r'] / max(skin_features['mean_g'], 1)
                skin_features['melanin_index'] = 100 * np.log10(1/max(skin_features['mean_g'], 1))
            else:
                skin_features = {'brightness': 123, 'rg_ratio': 1.25, 'melanin_index': 50}
            
            ethnicity_scores = self._classify_by_skin_tone(skin_features)
            
            predicted_ethnicity = max(ethnicity_scores, key=ethnicity_scores.get)
            confidence = ethnicity_scores[predicted_ethnicity]
            
            return {
                'method': 'skin_tone_analysis',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in ethnicity_scores.items()},
                'skin_features': {k: float(v) for k, v in skin_features.items()}
            }
        except Exception as e:
            return {
                'method': 'skin_tone_analysis',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {cls: 1.0/len(self.ethnicity_classes) for cls in self.ethnicity_classes}
            }
    
    def _classify_by_skin_tone(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {ethnicity: 0.0 for ethnicity in self.ethnicity_classes}
        
        brightness = features.get('brightness', 123)
        rg_ratio = features.get('rg_ratio', 1.25)
        melanin_index = features.get('melanin_index', 50)
        hue = features.get('hue_mean', 15)
        saturation = features.get('saturation_mean', 50)
        
        if brightness < 80:
            scores['Africano'] += 0.6
            scores['Indiano'] += 0.2
        elif brightness < 120:
            scores['Hispânico'] += 0.4
            scores['Árabe'] += 0.3
            scores['Indiano'] += 0.3
            scores['Misto'] += 0.2
        elif brightness < 160:
            scores['Asiático'] += 0.3
            scores['Hispânico'] += 0.2
            scores['Caucasiano'] += 0.2
            scores['Árabe'] += 0.2
        else:
            scores['Caucasiano'] += 0.5
            scores['Asiático'] += 0.2
        
        if rg_ratio > 1.4:
            scores['Caucasiano'] += 0.3
            scores['Hispânico'] += 0.2
        elif rg_ratio > 1.2:
            scores['Asiático'] += 0.2
            scores['Árabe'] += 0.2
        else:
            scores['Africano'] += 0.3
            scores['Indiano'] += 0.2
        
        if melanin_index > 60:
            scores['Africano'] += 0.4
            scores['Indiano'] += 0.2
        elif melanin_index > 45:
            scores['Hispânico'] += 0.3
            scores['Árabe'] += 0.2
        elif melanin_index > 30:
            scores['Asiático'] += 0.3
            scores['Caucasiano'] += 0.1
        else:
            scores['Caucasiano'] += 0.4
        
        if 5 <= hue <= 25:
            scores['Caucasiano'] += 0.2
            scores['Hispânico'] += 0.15
        elif 0 <= hue <= 15:
            scores['Asiático'] += 0.2
            scores['Árabe'] += 0.15
        
        if saturation < 30:
            scores['Caucasiano'] += 0.1
            scores['Asiático'] += 0.1
        elif saturation > 60:
            scores['Africano'] += 0.2
            scores['Hispânico'] += 0.15
        
        for ethnicity in scores:
            if scores[ethnicity] < 0.05:
                scores[ethnicity] = 0.05
        
        scores['Misto'] = np.mean(list(scores.values())) * 0.7
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/len(scores) for k in scores}
        
        return scores
    
    def _predict_with_facenet_enhanced(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            embedding = self.facenet.embeddings(face_expanded)[0]
            
            ethnicity_features = {}
            
            embedding_segments = np.array_split(embedding, 8)
            for i, segment in enumerate(embedding_segments):
                ethnicity_features[f'segment_{i}_mean'] = np.mean(segment)
                ethnicity_features[f'segment_{i}_std'] = np.std(segment)
                ethnicity_features[f'segment_{i}_max'] = np.max(segment)
                ethnicity_features[f'segment_{i}_min'] = np.min(segment)
            
            ethnicity_features['embedding_norm'] = np.linalg.norm(embedding)
            ethnicity_features['embedding_mean'] = np.mean(embedding)
            ethnicity_features['embedding_std'] = np.std(embedding)
            ethnicity_features['positive_ratio'] = np.sum(embedding > 0) / len(embedding)
            ethnicity_features['negative_ratio'] = np.sum(embedding < 0) / len(embedding)
            
            high_activation_indices = np.where(embedding > np.percentile(embedding, 85))[0]
            ethnicity_features['high_activation_count'] = len(high_activation_indices)
            
            low_activation_indices = np.where(embedding < np.percentile(embedding, 15))[0]
            ethnicity_features['low_activation_count'] = len(low_activation_indices)
            
            ethnicity_probs = self._classify_ethnicity_from_enhanced_features(ethnicity_features)
            
            predicted_ethnicity = max(ethnicity_probs, key=ethnicity_probs.get)
            confidence = ethnicity_probs[predicted_ethnicity]
            
            return {
                'method': 'facenet_enhanced',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in ethnicity_probs.items()}
            }
        except Exception as e:
            return {
                'method': 'facenet_enhanced',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {cls: 1.0/len(self.ethnicity_classes) for cls in self.ethnicity_classes}
            }
    
    def _classify_ethnicity_from_enhanced_features(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {ethnicity: 0.0 for ethnicity in self.ethnicity_classes}
        
        embedding_norm = features.get('embedding_norm', 10.0)
        positive_ratio = features.get('positive_ratio', 0.5)
        high_activation = features.get('high_activation_count', 20)
        embedding_std = features.get('embedding_std', 1.0)
        
        segment_means = [features.get(f'segment_{i}_mean', 0) for i in range(8)]
        segment_stds = [features.get(f'segment_{i}_std', 1) for i in range(8)]
        
        if embedding_norm > 15:
            scores['Caucasiano'] += 0.3
            scores['Asiático'] += 0.2
        elif embedding_norm > 12:
            scores['Hispânico'] += 0.25
            scores['Árabe'] += 0.2
        elif embedding_norm > 8:
            scores['Indiano'] += 0.3
            scores['Misto'] += 0.2
        else:
            scores['Africano'] += 0.4
        
        if positive_ratio > 0.6:
            scores['Asiático'] += 0.25
            scores['Caucasiano'] += 0.15
        elif positive_ratio > 0.4:
            scores['Hispânico'] += 0.2
            scores['Árabe'] += 0.15
        else:
            scores['Africano'] += 0.3
            scores['Indiano'] += 0.2
        
        if high_activation > 25:
            scores['Caucasiano'] += 0.2
            scores['Asiático'] += 0.15
        elif high_activation > 15:
            scores['Hispânico'] += 0.2
            scores['Árabe'] += 0.15
        else:
            scores['Africano'] += 0.25
            scores['Indiano'] += 0.15
        
        if embedding_std > 2.0:
            scores['Africano'] += 0.2
            scores['Hispânico'] += 0.15
        elif embedding_std > 1.5:
            scores['Indiano'] += 0.2
            scores['Árabe'] += 0.15
        else:
            scores['Asiático'] += 0.25
            scores['Caucasiano'] += 0.15
        
        segment_variance = np.var(segment_means)
        if segment_variance > 1.0:
            scores['Misto'] += 0.3
            scores['Hispânico'] += 0.1
        
        for ethnicity in scores:
            if scores[ethnicity] < 0.05:
                scores[ethnicity] = 0.05
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/len(scores) for k in scores}
        
        return scores
    
    def _predict_with_facial_structure(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            h, w = gray.shape
            
            structure_features = {}
            
            forehead_region = gray[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
            if forehead_region.size > 0:
                forehead_edges = cv2.Canny(forehead_region, 30, 100)
                structure_features['forehead_prominence'] = np.sum(forehead_edges) / forehead_region.size
                structure_features['forehead_height'] = forehead_region.shape[0] / h
            else:
                structure_features['forehead_prominence'] = 0.02
                structure_features['forehead_height'] = 0.3
            
            jaw_region = gray[int(h*0.7):h, int(w*0.1):int(w*0.9)]
            if jaw_region.size > 0:
                jaw_edges = cv2.Canny(jaw_region, 25, 80)
                jaw_contours, _ = cv2.findContours(jaw_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if jaw_contours:
                    largest_contour = max(jaw_contours, key=cv2.contourArea)
                    jaw_area = cv2.contourArea(largest_contour)
                    structure_features['jaw_area_ratio'] = jaw_area / (w * h)
                    
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    structure_features['jaw_convexity'] = jaw_area / max(hull_area, 1)
                else:
                    structure_features['jaw_area_ratio'] = 0.1
                    structure_features['jaw_convexity'] = 0.8
                
                structure_features['jaw_definition'] = np.sum(jaw_edges) / jaw_region.size
            else:
                structure_features['jaw_area_ratio'] = 0.1
                structure_features['jaw_convexity'] = 0.8
                structure_features['jaw_definition'] = 0.02
            
            cheek_region = gray[int(h*0.4):int(h*0.7), int(w*0.1):int(w*0.9)]
            if cheek_region.size > 0:
                cheek_mean = np.mean(cheek_region)
                cheek_std = np.std(cheek_region)
                structure_features['cheek_fullness'] = cheek_mean / 255.0
                structure_features['cheek_variation'] = cheek_std / 255.0
                
                sobel_x = cv2.Sobel(cheek_region, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(cheek_region, cv2.CV_64F, 0, 1, ksize=3)
                cheek_gradient = np.sqrt(sobel_x**2 + sobel_y**2)
                structure_features['cheek_structure'] = np.mean(cheek_gradient) / 255.0
            else:
                structure_features['cheek_fullness'] = 0.5
                structure_features['cheek_variation'] = 0.1
                structure_features['cheek_structure'] = 0.1
            
            orbital_region = gray[int(h*0.25):int(h*0.5), int(w*0.15):int(w*0.85)]
            if orbital_region.size > 0:
                orbital_edges = cv2.Canny(orbital_region, 20, 60)
                structure_features['orbital_depth'] = np.sum(orbital_edges) / orbital_region.size
                
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                eyes = eye_cascade.detectMultiScale(orbital_region, 1.1, 3)
                
                if len(eyes) >= 2:
                    eye_areas = [w * h for x, y, w, h in eyes[:2]]
                    structure_features['eye_size_ratio'] = np.mean(eye_areas) / (w * h)
                else:
                    structure_features['eye_size_ratio'] = 0.02
            else:
                structure_features['orbital_depth'] = 0.01
                structure_features['eye_size_ratio'] = 0.02
            
            ethnicity_scores = self._classify_by_facial_structure(structure_features)
            
            predicted_ethnicity = max(ethnicity_scores, key=ethnicity_scores.get)
            confidence = ethnicity_scores[predicted_ethnicity]
            
            return {
                'method': 'facial_structure',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in ethnicity_scores.items()},
                'structure_features': {k: float(v) for k, v in structure_features.items()}
            }
        except Exception as e:
            return {
                'method': 'facial_structure',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {cls: 1.0/len(self.ethnicity_classes) for cls in self.ethnicity_classes}
            }
    
    def _classify_by_facial_structure(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {ethnicity: 0.0 for ethnicity in self.ethnicity_classes}
        
        forehead_prominence = features.get('forehead_prominence', 0.02)
        jaw_definition = features.get('jaw_definition', 0.02)
        cheek_fullness = features.get('cheek_fullness', 0.5)
        orbital_depth = features.get('orbital_depth', 0.01)
        eye_size = features.get('eye_size_ratio', 0.02)
        
        if forehead_prominence > 0.04:
            scores['Caucasiano'] += 0.3
            scores['Africano'] += 0.2
        elif forehead_prominence > 0.025:
            scores['Hispânico'] += 0.25
            scores['Árabe'] += 0.2
        else:
            scores['Asiático'] += 0.35
            scores['Indiano'] += 0.15
        
        if jaw_definition > 0.03:
            scores['Caucasiano'] += 0.25
            scores['Africano'] += 0.2
        elif jaw_definition > 0.02:
            scores['Hispânico'] += 0.2
            scores['Árabe'] += 0.15
        else:
            scores['Asiático'] += 0.3
            scores['Indiano'] += 0.15
        
        if cheek_fullness > 0.6:
            scores['Africano'] += 0.3
            scores['Hispânico'] += 0.2
        elif cheek_fullness > 0.45:
            scores['Caucasiano'] += 0.2
            scores['Árabe'] += 0.15
        else:
            scores['Asiático'] += 0.25
            scores['Indiano'] += 0.15
        
        if orbital_depth > 0.02:
            scores['Caucasiano'] += 0.2
            scores['Africano'] += 0.15
        elif orbital_depth > 0.015:
            scores['Hispânico'] += 0.15
            scores['Árabe'] += 0.1
        else:
            scores['Asiático'] += 0.25
            scores['Indiano'] += 0.1
        
        if eye_size > 0.025:
            scores['Africano'] += 0.2
            scores['Caucasiano'] += 0.15
        elif eye_size > 0.018:
            scores['Hispânico'] += 0.15
            scores['Árabe'] += 0.1
        else:
            scores['Asiático'] += 0.3
            scores['Indiano'] += 0.1
        
        for ethnicity in scores:
            if scores[ethnicity] < 0.05:
                scores[ethnicity] = 0.05
        
        scores['Misto'] = np.mean(list(scores.values())) * 0.6
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/len(scores) for k in scores}
        
        return scores
    
    def _extract_ethnicity_features_from_embedding(self, embedding: np.ndarray) -> Dict[str, float]:
        features = {
            'facial_bone_structure': float(np.mean(embedding[0:70])),
            'eye_shape_features': float(np.mean(embedding[70:140])),
            'nose_bridge_width': float(np.std(embedding[140:210])),
            'lip_fullness': float(np.mean(embedding[210:280])),
            'cheekbone_prominence': float(np.std(embedding[280:350])),
            'forehead_shape': float(np.mean(embedding[350:420])),
            'jaw_structure': float(np.std(embedding[420:490])),
            'overall_facial_harmony': float(np.mean(embedding[490:512]))
        }
        return features
    
    def _classify_ethnicity_from_features(self, features: Dict[str, float]) -> Dict[str, float]:
        ethnicity_weights = {
            'Caucasiano': {
                'facial_bone_structure': 0.15, 'eye_shape_features': 0.10,
                'nose_bridge_width': 0.20, 'lip_fullness': 0.10,
                'cheekbone_prominence': 0.15, 'forehead_shape': 0.10,
                'jaw_structure': 0.15, 'overall_facial_harmony': 0.05
            },
            'Asiático': {
                'facial_bone_structure': 0.10, 'eye_shape_features': 0.25,
                'nose_bridge_width': 0.15, 'lip_fullness': 0.08,
                'cheekbone_prominence': 0.20, 'forehead_shape': 0.12,
                'jaw_structure': 0.08, 'overall_facial_harmony': 0.02
            },
            'Africano': {
                'facial_bone_structure': 0.20, 'eye_shape_features': 0.12,
                'nose_bridge_width': 0.18, 'lip_fullness': 0.20,
                'cheekbone_prominence': 0.15, 'forehead_shape': 0.08,
                'jaw_structure': 0.05, 'overall_facial_harmony': 0.02
            },
            'Hispânico': {
                'facial_bone_structure': 0.14, 'eye_shape_features': 0.12,
                'nose_bridge_width': 0.16, 'lip_fullness': 0.15,
                'cheekbone_prominence': 0.18, 'forehead_shape': 0.10,
                'jaw_structure': 0.12, 'overall_facial_harmony': 0.03
            },
            'Árabe': {
                'facial_bone_structure': 0.18, 'eye_shape_features': 0.15,
                'nose_bridge_width': 0.22, 'lip_fullness': 0.12,
                'cheekbone_prominence': 0.15, 'forehead_shape': 0.10,
                'jaw_structure': 0.06, 'overall_facial_harmony': 0.02
            },
            'Indiano': {
                'facial_bone_structure': 0.16, 'eye_shape_features': 0.18,
                'nose_bridge_width': 0.20, 'lip_fullness': 0.14,
                'cheekbone_prominence': 0.16, 'forehead_shape': 0.08,
                'jaw_structure': 0.06, 'overall_facial_harmony': 0.02
            }
        }
        
        ethnicity_scores = {}
        for ethnicity, weights in ethnicity_weights.items():
            score = 0.0
            for feature, value in features.items():
                if feature in weights:
                    normalized_value = max(-1, min(1, value))
                    score += abs(normalized_value) * weights[feature]
            ethnicity_scores[ethnicity] = max(0.1, min(0.9, score))
        
        total_score = sum(ethnicity_scores.values())
        if total_score > 0:
            ethnicity_scores = {k: v/total_score for k, v in ethnicity_scores.items()}
        
        return ethnicity_scores
    
    def _predict_with_cnn(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            # Criar hash para cache
            img_hash = hash(face_img.tobytes())
            if img_hash in self._prediction_cache:
                return self._prediction_cache[img_hash]
            
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            # Usar função otimizada se disponível
            if hasattr(self, '_optimized_ethnicity_predict'):
                predictions = self._optimized_ethnicity_predict(face_expanded)[0].numpy()
            else:
                predictions = self.ethnicity_model.predict(face_expanded, verbose=0)[0]
            
            ethnicity_probs = {
                self.ethnicity_classes[i]: float(predictions[i]) 
                for i in range(len(self.ethnicity_classes))
            }
            
            predicted_ethnicity = max(ethnicity_probs, key=ethnicity_probs.get)
            confidence = ethnicity_probs[predicted_ethnicity]
            
            result = {
                'method': 'cnn_ethnicity',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': ethnicity_probs
            }
            
            # Cache do resultado
            if len(self._prediction_cache) < 100:
                self._prediction_cache[img_hash] = result
            
            return result
        except Exception as e:
            print(f"Erro CNN etnia: {e}")
            return {
                'method': 'cnn_ethnicity',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {}
            }
    
    def _predict_with_color_analysis(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            
            l_mean = np.mean(lab[:, :, 0])
            a_mean = np.mean(lab[:, :, 1])
            b_mean = np.mean(lab[:, :, 2])
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            
            ethnicity_scores = {}
            
            if l_mean > 165:
                ethnicity_scores['Caucasiano'] = 0.8
                ethnicity_scores['Asiático'] = 0.3
            elif l_mean > 140:
                ethnicity_scores['Asiático'] = 0.7
                ethnicity_scores['Hispânico'] = 0.6
                ethnicity_scores['Árabe'] = 0.5
            elif l_mean > 115:
                ethnicity_scores['Hispânico'] = 0.8
                ethnicity_scores['Árabe'] = 0.7
                ethnicity_scores['Indiano'] = 0.6
            elif l_mean > 85:
                ethnicity_scores['Indiano'] = 0.8
                ethnicity_scores['Árabe'] = 0.6
                ethnicity_scores['Africano'] = 0.4
            else:
                ethnicity_scores['Africano'] = 0.9
            
            if b_mean > 135:
                ethnicity_scores['Asiático'] = ethnicity_scores.get('Asiático', 0) + 0.2
            
            if a_mean > 130:
                ethnicity_scores['Caucasiano'] = ethnicity_scores.get('Caucasiano', 0) + 0.1
            
            total_score = sum(ethnicity_scores.values())
            if total_score > 0:
                ethnicity_scores = {k: v/total_score for k, v in ethnicity_scores.items()}
            
            predicted_ethnicity = max(ethnicity_scores, key=ethnicity_scores.get) if ethnicity_scores else 'Indefinido'
            confidence = ethnicity_scores.get(predicted_ethnicity, 0.3)
            
            return {
                'method': 'color_analysis',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in ethnicity_scores.items()},
                'color_metrics': {
                    'lightness': float(l_mean),
                    'a_channel': float(a_mean),
                    'b_channel': float(b_mean),
                    'hue': float(h_mean),
                    'saturation': float(s_mean)
                }
            }
        except Exception as e:
            print(f"Erro análise de cor: {e}")
            return {
                'method': 'color_analysis',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {}
            }
    
    def _predict_traditional(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            brightness = np.mean(lab[:, :, 0])
            
            if brightness > 160:
                ethnicity = 'Caucasiano'
                confidence = 0.7
            elif brightness > 130:
                ethnicity = 'Asiático'
                confidence = 0.6
            elif brightness > 100:
                ethnicity = 'Hispânico'
                confidence = 0.6
            elif brightness > 70:
                ethnicity = 'Árabe'
                confidence = 0.5
            else:
                ethnicity = 'Africano'
                confidence = 0.6
            
            return {
                'method': 'traditional_ethnicity',
                'predicted_ethnicity': ethnicity,
                'confidence': float(confidence),
                'brightness_score': float(brightness)
            }
        except Exception as e:
            print(f"Erro tradicional etnia: {e}")
            return {
                'method': 'traditional_ethnicity',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3
            }
    
    def _ensemble_ethnicity_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            method_weights = {
                'facenet_ethnicity': 0.35,
                'cnn_ethnicity': 0.30,
                'color_analysis': 0.25,
                'traditional_ethnicity': 0.10
            }
            
            combined_probs = {}
            total_weight = 0.0
            methods_used = []
            
            for pred in predictions:
                method = pred.get('method', 'unknown')
                confidence = pred.get('confidence', 0.3)
                ethnicity = pred.get('predicted_ethnicity', 'Indefinido')
                
                if method in method_weights and ethnicity != 'Indefinido':
                    weight = method_weights[method] * confidence
                    
                    if ethnicity not in combined_probs:
                        combined_probs[ethnicity] = 0.0
                    
                    combined_probs[ethnicity] += weight
                    total_weight += weight
                    methods_used.append(method)
            
            if total_weight > 0:
                combined_probs = {k: v/total_weight for k, v in combined_probs.items()}
            
            if combined_probs:
                predicted_ethnicity = max(combined_probs, key=combined_probs.get)
                final_confidence = combined_probs[predicted_ethnicity]
            else:
                predicted_ethnicity = 'Indefinido'
                final_confidence = 0.3
                combined_probs = {'Indefinido': 1.0}
            
            agreement_scores = []
            for pred in predictions:
                if pred.get('predicted_ethnicity') == predicted_ethnicity:
                    agreement_scores.append(pred.get('confidence', 0.3))
            
            if len(agreement_scores) > 1:
                agreement_boost = min(0.2, len(agreement_scores) * 0.05)
                final_confidence = min(0.95, final_confidence + agreement_boost)
            
            return {
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(final_confidence),
                'probabilities': {k: float(v) for k, v in combined_probs.items()},
                'ensemble_details': {
                    'methods_used': methods_used,
                    'agreement_count': len(agreement_scores)
                }
            }
        except Exception as e:
            print(f"Erro no ensemble de etnia: {e}")
            return {
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {'Indefinido': 1.0}
            }
    
    def _ensemble_ethnicity_predictions_weighted(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            method_weights = {
                'facenet_enhanced': 0.25,
                'cnn_enhanced': 0.25,
                'geometric_analysis': 0.20,
                'skin_tone_analysis': 0.15,
                'facial_structure': 0.10,
                'traditional_enhanced': 0.05
            }
            
            ensemble_scores = {ethnicity: 0.0 for ethnicity in self.ethnicity_classes}
            total_weight = 0.0
            confidence_sum = 0.0
            
            for pred in predictions:
                if 'probabilities' in pred and 'method' in pred:
                    method = pred['method']
                    weight = method_weights.get(method, 0.1)
                    confidence = pred.get('confidence', 0.5)
                    
                    adjusted_weight = weight * (0.5 + confidence * 0.5)
                    
                    for ethnicity, prob in pred['probabilities'].items():
                        if ethnicity in ensemble_scores:
                            ensemble_scores[ethnicity] += prob * adjusted_weight
                    
                    total_weight += adjusted_weight
                    confidence_sum += confidence
            
            if total_weight > 0:
                ensemble_scores = {k: v/total_weight for k, v in ensemble_scores.items()}
                avg_confidence = confidence_sum / len(predictions)
            else:
                ensemble_scores = {k: 1.0/len(self.ethnicity_classes) for k in self.ethnicity_classes}
                avg_confidence = 0.3
            
            predicted_ethnicity = max(ensemble_scores, key=ensemble_scores.get)
            final_confidence = ensemble_scores[predicted_ethnicity]
            
            confidence_boost = 0.0
            if final_confidence > 0.7:
                confidence_boost = 0.1
            elif final_confidence > 0.5:
                confidence_boost = 0.05
            
            final_confidence = min(0.95, final_confidence + confidence_boost)
            
            return {
                'method': 'ensemble_weighted',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(final_confidence),
                'probabilities': {k: float(v) for k, v in ensemble_scores.items()},
                'individual_predictions': predictions,
                'ensemble_confidence': float(avg_confidence)
            }
        except Exception as e:
            print(f"Erro no ensemble ponderado: {e}")
            return {
                'method': 'ensemble_weighted',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {cls: 1.0/len(self.ethnicity_classes) for cls in self.ethnicity_classes}
            }
    
    def _predict_with_cnn_enhanced(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            # Cache para evitar reprocessamento
            img_hash = hash(face_img.tobytes()) + hash('cnn_enhanced')
            if img_hash in self._prediction_cache:
                return self._prediction_cache[img_hash]
            
            face_resized = cv2.resize(face_img, (224, 224))
            face_normalized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            # Usar função otimizada se disponível
            if hasattr(self, '_optimized_ethnicity_predict'):
                predictions = self._optimized_ethnicity_predict(face_expanded)[0].numpy()
            else:
                predictions = self.ethnicity_model.predict(face_expanded, verbose=0)[0]
            
            ethnicity_probs = {}
            for i, ethnicity in enumerate(self.ethnicity_classes):
                ethnicity_probs[ethnicity] = float(predictions[i])
            
            predicted_ethnicity = max(ethnicity_probs, key=ethnicity_probs.get)
            confidence = ethnicity_probs[predicted_ethnicity]
            
            result = {
                'method': 'cnn_enhanced',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': ethnicity_probs
            }
            
            # Cache limitado
            if len(self._prediction_cache) < 100:
                self._prediction_cache[img_hash] = result
            
            return result
        except Exception as e:
            return {
                'method': 'cnn_enhanced',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {cls: 1.0/len(self.ethnicity_classes) for cls in self.ethnicity_classes}
            }
    
    def _predict_traditional_enhanced(self, face_img: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            h, w = gray.shape
            
            features = {}
            
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            features['brightness'] = mean_intensity
            features['contrast'] = std_intensity
            
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (h * w)
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['texture_variance'] = np.var(laplacian)
            
            upper_face = gray[:int(h*0.6), :]
            lower_face = gray[int(h*0.4):, :]
            
            features['upper_lower_ratio'] = np.mean(upper_face) / max(np.mean(lower_face), 1)
            
            left_face = gray[:, :int(w*0.5)]
            right_face = gray[:, int(w*0.5):]
            
            features['left_right_symmetry'] = 1 - abs(np.mean(left_face) - np.mean(right_face)) / 255.0
            
            center_region = gray[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
            if center_region.size > 0:
                features['center_brightness'] = np.mean(center_region)
            else:
                features['center_brightness'] = mean_intensity
            
            ethnicity_scores = self._classify_traditional_enhanced(features)
            
            predicted_ethnicity = max(ethnicity_scores, key=ethnicity_scores.get)
            confidence = ethnicity_scores[predicted_ethnicity]
            
            return {
                'method': 'traditional_enhanced',
                'predicted_ethnicity': predicted_ethnicity,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in ethnicity_scores.items()}
            }
        except Exception as e:
            return {
                'method': 'traditional_enhanced',
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3,
                'probabilities': {cls: 1.0/len(self.ethnicity_classes) for cls in self.ethnicity_classes}
            }
    
    def _classify_traditional_enhanced(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {ethnicity: 0.0 for ethnicity in self.ethnicity_classes}
        
        brightness = features.get('brightness', 128)
        contrast = features.get('contrast', 30)
        edge_density = features.get('edge_density', 0.1)
        texture = features.get('texture_variance', 100)
        
        if brightness < 100:
            scores['Africano'] += 0.5
            scores['Indiano'] += 0.2
        elif brightness < 140:
            scores['Hispânico'] += 0.3
            scores['Árabe'] += 0.25
            scores['Indiano'] += 0.2
        elif brightness < 180:
            scores['Asiático'] += 0.3
            scores['Caucasiano'] += 0.25
            scores['Hispânico'] += 0.15
        else:
            scores['Caucasiano'] += 0.5
            scores['Asiático'] += 0.2
        
        if contrast > 40:
            scores['Africano'] += 0.2
            scores['Caucasiano'] += 0.15
        elif contrast > 25:
            scores['Hispânico'] += 0.2
            scores['Árabe'] += 0.15
        else:
            scores['Asiático'] += 0.25
            scores['Indiano'] += 0.15
        
        if edge_density > 0.15:
            scores['Caucasiano'] += 0.2
            scores['Africano'] += 0.15
        elif edge_density > 0.08:
            scores['Hispânico'] += 0.15
            scores['Árabe'] += 0.1
        else:
            scores['Asiático'] += 0.2
            scores['Indiano'] += 0.1
        
        if texture > 150:
            scores['Africano'] += 0.2
            scores['Caucasiano'] += 0.1
        elif texture > 80:
            scores['Hispânico'] += 0.15
            scores['Árabe'] += 0.1
        else:
            scores['Asiático'] += 0.2
            scores['Indiano'] += 0.1
        
        for ethnicity in scores:
            if scores[ethnicity] < 0.05:
                scores[ethnicity] = 0.05
        
        scores['Misto'] = np.mean(list(scores.values())) * 0.8
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/len(scores) for k in scores}
        
        return scores

class AdvancedDemographicsDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        self.age_ranges = [
            (0, 5, 'Bebê'), (6, 12, 'Criança'), (13, 17, 'Adolescente'),
            (18, 25, 'Jovem'), (26, 35, 'Jovem Adulto'), (36, 45, 'Adulto'),
            (46, 55, 'Meia-idade'), (56, 65, 'Maduro'), (66, 100, 'Idoso')
        ]
        
        self._analysis_cache = {}
        
        try:
            self.advanced_gender_detector = AdvancedGenderDetector()
            self.use_advanced_gender = True
        except Exception as e:
            print(f"Detector avançado de gênero não disponível: {e}")
            self.use_advanced_gender = False
        
        try:
            self.advanced_ethnicity_detector = AdvancedEthnicityDetector()
            self.use_advanced_ethnicity = True
        except Exception as e:
            print(f"Detector avançado de etnia não disponível: {e}")
            self.use_advanced_ethnicity = False
        
        try:
            self.advanced_age_detector = AdvancedAgeDetector()
            self.use_advanced_age = True
        except Exception as e:
            print(f"Detector avançado de idade não disponível: {e}")
            self.use_advanced_age = False
        
    def analyze_demographics(self, image_data: str) -> Dict[str, Any]:
        try:
            if len(image_data) > 50 * 1024 * 1024:
                raise ValueError("Imagem muito grande (máximo 50MB)")
            
            cache_key = hashlib.md5(image_data.encode()).hexdigest()[:10]
            if cache_key in self._analysis_cache:
                print(f"📋 Resultado encontrado no cache (ID: {cache_key})")
                return self._analysis_cache[cache_key]
            
            print(f"📊 Iniciando análise demográfica (ID: {cache_key})")
            
            img = self._decode_image(image_data)
            # Downscale agressivo para economizar memória/CPU (ex.: 1024px máx)
            try:
                h, w = img.shape[:2]
                max_dim = 1024
                if max(h, w) > max_dim:
                    if h >= w:
                        new_h = max_dim
                        new_w = int(w * (max_dim / h))
                    else:
                        new_w = max_dim
                        new_h = int(h * (max_dim / w))
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"↘️ Downscale: {w}x{h} -> {new_w}x{new_h}")
            except Exception as _r:
                pass
            print(f"🖼️  Imagem decodificada: {img.shape[1]}x{img.shape[0]} pixels")
            
            if img.shape[0] < 150 or img.shape[1] < 150:
                raise ValueError("Imagem muito pequena para detecção confiável de faces (mínimo 150x150)")
            
            faces = self._detect_faces_advanced(img)
            
            if len(faces) == 0:
                print("🔄 Nenhuma face detectada, tentando com melhoramento da imagem...")
                img_enhanced = self._enhance_image(img)
                faces = self._detect_faces_advanced(img_enhanced)
            
            if len(faces) == 0:
                # Modo rápido: pular métodos alternativos e detecção ultra-sensível
                print("⚠️ Modo rápido: pulando métodos alternativos e detecção ultra-sensível")
            
            if len(faces) == 0:
                error_msg = (
                    "Nenhuma face foi detectada na imagem.\n\n"
                    "Dicas para melhorar a detecção:\n"
                    "• Certifique-se de que há faces claramente visíveis\n"
                    "• Use boa iluminação (evite contraluz)\n"
                    "• A face deve ocupar pelo menos 15% da imagem\n"
                    "• Evite ângulos muito extremos ou faces de perfil\n"
                    "• Remova objetos que obstruam o rosto (óculos escuros, máscaras)\n"
                    "• Use imagem com resolução adequada (mínimo 300x300 pixels)"
                )
                raise ValueError(error_msg)
            
            print(f"👥 {len(faces)} face(s) detectada(s) e validada(s)")
            
            # Processamento assíncrono controlado para reduzir memória
            from concurrent.futures import ThreadPoolExecutor, as_completed
            try:
                try:
                    from config import Config
                    max_workers = int(getattr(getattr(Config, 'FACE_DETECTION_CONFIG', {}), 'get', lambda *_: None)('analysis_max_workers') or getattr(Config, 'ANALYSIS_MAX_WORKERS', 2))
                except Exception:
                    max_workers = 2

                max_workers = max(1, min(4, int(max_workers)))

                results = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {}
                    for i, face_data in enumerate(faces):
                        print(f"🔍 Analisando face {i+1}/{len(faces)} (workers={max_workers})")
                        face_img = face_data['image']
                        face_gray = face_data['gray']
                        face_rect = face_data['rect']

                        future = executor.submit(self._analyze_single_face, face_img, face_gray, face_rect)
                        future_to_index[future] = (i, face_data.get('quality', 0.5))

                    for future in as_completed(future_to_index):
                        i, quality = future_to_index[future]
                        try:
                            demographics = future.result()
                        except Exception as _e:
                            print(f"Erro ao analisar face {i+1}: {_e}")
                            demographics = self._get_default_analysis()
                        demographics['face_id'] = i + 1
                        demographics['detection_quality'] = quality
                        results.append(demographics)
                        # liberar referências cedo
                        try:
                            faces[i]['image'] = None
                            faces[i]['gray'] = None
                        except Exception:
                            pass
            except Exception as _pool_e:
                print(f"Aviso: fallback para processamento sequencial: {_pool_e}")
                results = []
                for i, face_data in enumerate(faces):
                    print(f"🔍 Analisando face {i+1}/{len(faces)} (fallback)")
                    face_img = face_data['image']
                    face_gray = face_data['gray']
                    face_rect = face_data['rect']
                    demographics = self._analyze_single_face(face_img, face_gray, face_rect)
                    demographics['face_id'] = i + 1
                    demographics['detection_quality'] = face_data.get('quality', 0.5)
                    results.append(demographics)
            
            summary = self._generate_summary(results)
            
            result = {
                'success': True,
                'total_faces': len(faces),
                'demographics': results,
                'summary': summary,
                'analyzed_at': datetime.now().isoformat(),
                'image_hash': cache_key,
                'analysis_version': '3.2',
                'image_info': {
                    'width': img.shape[1],
                    'height': img.shape[0],
                    'channels': img.shape[2] if len(img.shape) > 2 else 1
                }
            }
            
            final_result = convert_numpy_types(result)
            # Libera imagem e faces (buffers grandes)
            try:
                del img, faces
            except Exception:
                pass
            self._analysis_cache[cache_key] = final_result
            
            print(f"✅ Análise concluída com sucesso (ID: {cache_key})")
            return final_result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Erro na análise demográfica: {error_msg}")
            raise ValueError(f"Erro na análise demográfica: {error_msg}")
    
    def _decode_image(self, image_data: str) -> np.ndarray:
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Formato de imagem inválido")
            if img.shape[0] < 100 or img.shape[1] < 100:
                raise ValueError("Imagem muito pequena")
            
            # Redimensionar se a imagem for muito grande para otimizar processamento
            h, w = img.shape[:2]
            max_dimension = 1920
            
            if h > max_dimension or w > max_dimension:
                if h > w:
                    new_h = max_dimension
                    new_w = int(w * (max_dimension / h))
                else:
                    new_w = max_dimension
                    new_h = int(h * (max_dimension / w))
                
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"Imagem redimensionada de {w}x{h} para {new_w}x{new_h}")
            
            return img
        except Exception as e:
            raise ValueError(f"Erro ao decodificar imagem: {str(e)}")
    
    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
        except:
            return img
    
    def _detect_faces_advanced(self, img: np.ndarray) -> List[Dict[str, Any]]:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_data = []
            
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            
            primary_configs = [
                {
                    'cascade': self.face_cascade,
                    'scale_factor': 1.1,
                    'min_neighbors': 5,
                    'min_size': (60, 60),
                    'max_size': (int(img.shape[1]*0.8), int(img.shape[0]*0.8))
                },
                {
                    'cascade': self.face_cascade,
                    'scale_factor': 1.05,
                    'min_neighbors': 4,
                    'min_size': (40, 40),
                    'max_size': (int(img.shape[1]*0.6), int(img.shape[0]*0.6))
                }
            ]
            
            all_faces = []
            
            # Modo rápido: tentar apenas a primeira configuração principal
            for config in primary_configs[:1]:
                try:
                    faces = config['cascade'].detectMultiScale(
                        gray,
                        scaleFactor=config['scale_factor'],
                        minNeighbors=config['min_neighbors'],
                        minSize=config['min_size'],
                        maxSize=config['max_size'],
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(faces) > 0:
                        all_faces.extend(faces)
                        print(f"✅ {len(faces)} face(s) detectada(s) com configuração principal")
                        break
                        
                except Exception as e:
                    print(f"Erro em configuração principal: {e}")
                    continue
            
            if len(all_faces) == 0:
                print("🔍 Tentando com imagem aprimorada (modo rápido)...")
                
                # Modo rápido: apenas a primeira configuração com imagem aprimorada
                for config in primary_configs[:1]:
                    try:
                        faces = config['cascade'].detectMultiScale(
                            gray_enhanced,
                            scaleFactor=config['scale_factor'],
                            minNeighbors=config['min_neighbors'],
                            minSize=config['min_size'],
                            maxSize=config['max_size'],
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        
                        if len(faces) > 0:
                            all_faces.extend(faces)
                            print(f"✅ {len(faces)} face(s) detectada(s) com imagem aprimorada")
                            break
                            
                    except Exception as e:
                        continue
            
            if len(all_faces) == 0:
                # Modo rápido: uma única configuração sensível e só
                print("🔍 Tentando uma configuração mais sensível (modo rápido)...")
                config = {
                    'scale_factor': 1.08,
                    'min_neighbors': 3,
                    'min_size': (30, 30),
                    'max_size': (int(img.shape[1]*0.7), int(img.shape[0]*0.7))
                }
                faces = self.face_cascade.detectMultiScale(
                    gray_enhanced,
                    scaleFactor=config['scale_factor'],
                    minNeighbors=config['min_neighbors'],
                    minSize=config['min_size'],
                    maxSize=config['max_size']
                )
                if len(faces) > 0:
                    all_faces.extend(faces)
                    print(f"✅ {len(faces)} face(s) detectada(s) com configuração sensível (modo rápido)")
            
            if len(all_faces) > 0:
                # Unir/mesclar detecções redundantes do mesmo rosto
                from config import Config
                unique_faces = self._merge_overlapping_faces(
                    all_faces,
                    iou_threshold=getattr(Config, 'FACE_DETECTION_CONFIG', {}).get('merge_iou_threshold', 0.22),
                    center_threshold=getattr(Config, 'FACE_DETECTION_CONFIG', {}).get('merge_center_threshold', 0.28)
                )
                print(f"👥 {len(unique_faces)} face(s) única(s) após mesclagem")
                
                for (x, y, w, h) in unique_faces:
                    if w < 40 or h < 40:
                        continue
                    
                    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                        continue
                    
                    margin = max(3, min(w, h) // 15)
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(img.shape[1], x + w + margin)
                    y_end = min(img.shape[0], y + h + margin)
                    
                    face_img = img[y_start:y_end, x_start:x_end]
                    face_gray_region = gray[y_start:y_end, x_start:x_end]
                    
                    if face_img.size > 0 and face_gray_region.size > 0:
                        quality = self._assess_face_region_quality_strict(face_gray_region)
                        
                        if quality >= 0.3:
                            faces_data.append({
                                'image': face_img,
                                'gray': face_gray_region,
                                'rect': (x_start, y_start, x_end - x_start, y_end - y_start),
                                'original_rect': (x, y, w, h),
                                'quality': quality
                            })
                
                faces_data.sort(key=lambda x: x['quality'], reverse=True)

                # Remover duplicatas residuais escolhendo a melhor por cluster
                from config import Config
                faces_data = self._cluster_and_select_faces(
                    faces_data,
                    iou_threshold=getattr(Config, 'FACE_DETECTION_CONFIG', {}).get('cluster_iou_threshold', 0.28),
                    center_threshold=getattr(Config, 'FACE_DETECTION_CONFIG', {}).get('cluster_center_threshold', 0.35)
                )
                
                # Se a intenção é 1 rosto dominante, mantenha somente o de melhor qualidade quando houver forte sobreposição
                if len(faces_data) > 1:
                    faces_data = self._cluster_and_select_faces(faces_data, iou_threshold=0.28, center_threshold=0.35)
                from config import Config
                max_faces = getattr(Config, 'FACE_DETECTION_CONFIG', {}).get('max_faces', 3)
                if len(faces_data) > max_faces:
                    faces_data = faces_data[:max_faces]
                    print(f"⚠️ Limitado a {len(faces_data)} faces de melhor qualidade")
            
            print(f"🎯 Total final de faces processadas: {len(faces_data)}")
            return faces_data
            
        except Exception as e:
            print(f"❌ Erro na detecção de faces: {e}")
            return []
    
    def _compute_iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x_left = max(ax, bx)
        y_top = max(ay, by)
        x_right = min(ax + aw, bx + bw)
        y_bottom = min(ay + ah, by + bh)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        inter = (x_right - x_left) * (y_bottom - y_top)
        union = aw * ah + bw * bh - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _center_distance_ratio(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx, acy = ax + aw / 2.0, ay + ah / 2.0
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        dist = math.hypot(acx - bcx, acy - bcy)
        ref = min(min(aw, ah), min(bw, bh))
        if ref <= 0:
            return 1.0
        return dist / ref

    def _is_contained(self, inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        return ix >= ox and iy >= oy and (ix + iw) <= (ox + ow) and (iy + ih) <= (oy + oh)

    def _merge_overlapping_faces(self, faces: List[Tuple], iou_threshold: float = 0.22, center_threshold: float = 0.28) -> List[Tuple]:
        if len(faces) <= 1:
            return [tuple(f) for f in faces]
        faces_sorted = sorted([tuple(f) for f in faces], key=lambda r: r[2] * r[3], reverse=True)
        merged: List[Tuple[int, int, int, int]] = []
        for rect in faces_sorted:
            merged_flag = False
            for idx, kept in enumerate(merged):
                iou = self._compute_iou(rect, kept)
                cdr = self._center_distance_ratio(rect, kept)
                if iou >= iou_threshold or cdr <= center_threshold or self._is_contained(rect, kept) or self._is_contained(kept, rect):
                    kx, ky, kw, kh = kept
                    rx, ry, rw, rh = rect
                    nx = min(kx, rx)
                    ny = min(ky, ry)
                    nx2 = max(kx + kw, rx + rw)
                    ny2 = max(ky + kh, ry + rh)
                    merged[idx] = (nx, ny, nx2 - nx, ny2 - ny)
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append(rect)
        return merged

    def _cluster_and_select_faces(self, faces_data: List[Dict[str, Any]], iou_threshold: float = 0.22, center_threshold: float = 0.28) -> List[Dict[str, Any]]:
        if len(faces_data) <= 1:
            return faces_data
        clusters: List[List[Dict[str, Any]]] = []
        for fd in faces_data:
            placed = False
            for cluster in clusters:
                r1 = cluster[0]['rect']
                r2 = fd['rect']
                iou = self._compute_iou(r1, r2)
                cdr = self._center_distance_ratio(r1, r2)
                if iou >= iou_threshold or cdr <= center_threshold or self._is_contained(r1, r2) or self._is_contained(r2, r1):
                    cluster.append(fd)
                    placed = True
                    break
            if not placed:
                clusters.append([fd])
        selected: List[Dict[str, Any]] = []
        for cluster in clusters:
            best = max(cluster, key=lambda x: x.get('quality', 0.0))
            selected.append(best)
        return selected
    
    def _assess_face_region_quality_strict(self, face_gray: np.ndarray) -> float:
        try:
            h, w = face_gray.shape
            if h < 40 or w < 40:
                return 0.1
            
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(10, 10))
            
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            sharpness = min(1.0, laplacian_var / 100.0)
            
            contrast = face_gray.std() / 255.0
            
            hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            hist_spread = np.std(hist)
            diversity_score = min(1.0, hist_spread / 1000.0)
            
            eye_bonus = 0.4 if len(eyes) >= 1 else 0.0
            eye_bonus += 0.2 if len(eyes) >= 2 else 0.0
            
            brightness = np.mean(face_gray)
            brightness_penalty = 0.0
            if brightness < 30 or brightness > 220:
                brightness_penalty = 0.3
            
            quality = (
                sharpness * 0.3 + 
                contrast * 0.2 + 
                diversity_score * 0.2 + 
                eye_bonus + 
                0.2 - 
                brightness_penalty
            )
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            print(f"Erro na avaliação de qualidade: {e}")
            return 0.2
    
    def _remove_duplicate_faces(self, faces: List[Tuple]) -> List[Tuple]:
        if len(faces) <= 1:
            return faces
        
        faces_array = np.array(faces)
        
        keep_faces = []
        for i, face1 in enumerate(faces_array):
            x1, y1, w1, h1 = face1
            is_duplicate = False
            
            for j, face2 in enumerate(faces_array):
                if i == j:
                    continue
                
                x2, y2, w2, h2 = face2
                
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    overlap_ratio = intersection_area / min(area1, area2)
                    if overlap_ratio > 0.5:
                        if area1 < area2:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                keep_faces.append(tuple(face1))
        
        return keep_faces
    
    def _assess_face_region_quality(self, face_gray: np.ndarray) -> float:
        try:
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            sharpness = min(1.0, laplacian_var / 100.0)
            
            contrast = face_gray.std() / 255.0
            
            eye_bonus = 0.3 if len(eyes) >= 1 else 0.0
            
            quality = (sharpness * 0.4) + (contrast * 0.3) + eye_bonus + 0.3
            
            return min(1.0, quality)
        except:
            return 0.5
    
    def _analyze_single_face(self, face_img: np.ndarray, face_gray: np.ndarray, face_rect: Tuple) -> Dict[str, Any]:
        try:
            if face_img.size == 0 or face_gray.size == 0:
                raise ValueError("Região facial inválida")
            
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                print(f"⚠️  Face muito pequena: {face_img.shape}")
            
            print("🔍 Analisando idade...")
            age_analysis = self._estimate_age(face_img, face_gray)
            
            print("🔍 Analisando gênero...")
            gender_analysis = self._estimate_gender(face_img, face_gray)
            
            print("🔍 Analisando características faciais...")
            facial_features = self._analyze_facial_features(face_img, face_gray)
            
            print("🔍 Analisando etnia...")
            ethnicity_analysis = self._estimate_ethnicity(face_img)
            
            print("🔍 Analisando expressão...")
            expression_analysis = self._analyze_expression(face_img, face_gray)
            
            quality_score = self._assess_image_quality(face_img)
            position = {
                'x': int(face_rect[0]), 'y': int(face_rect[1]), 
                'width': int(face_rect[2]), 'height': int(face_rect[3])
            }
            
            confidences = [
                age_analysis.get('confidence', 0.3),
                gender_analysis.get('confidence', 0.3),
                facial_features.get('confidence', 0.3),
                ethnicity_analysis.get('confidence', 0.3),
                expression_analysis.get('confidence', 0.3)
            ]
            
            confidence_score = float(np.mean([c for c in confidences if c > 0]))
            
            result = {
                'age': age_analysis,
                'gender': gender_analysis,
                'facial_features': facial_features,
                'ethnicity': ethnicity_analysis,
                'expression': expression_analysis,
                'quality_score': float(quality_score),
                'position': position,
                'confidence_score': confidence_score
            }
            
            print("✅ Análise individual concluída")
            return result
            
        except Exception as e:
            print(f"❌ Erro na análise da face: {e}")
            return self._get_default_analysis()
    
    def _estimate_age(self, face_img: np.ndarray, face_gray: np.ndarray) -> Dict[str, Any]:
        if self.use_advanced_age:
            try:
                return self.advanced_age_detector.detect_age_advanced(face_img)
            except Exception as e:
                print(f"Erro no detector avançado de idade: {e}")
        
        return self._estimate_age_traditional(face_img, face_gray)
    
    def _estimate_age_traditional(self, face_img: np.ndarray, face_gray: np.ndarray) -> Dict[str, Any]:
        try:
            h, w = face_gray.shape
            
            laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            eye_region = face_gray[int(h*0.3):int(h*0.5), int(w*0.2):int(w*0.8)]
            eye_texture = np.std(eye_region) if eye_region.size > 0 else 0
            
            forehead_region = face_gray[int(h*0.1):int(h*0.35), int(w*0.25):int(w*0.75)]
            forehead_texture = np.std(forehead_region) if forehead_region.size > 0 else 0
            
            cheek_region = face_gray[int(h*0.4):int(h*0.7), int(w*0.15):int(w*0.85)]
            cheek_texture = cv2.Laplacian(cheek_region, cv2.CV_64F).var() if cheek_region.size > 0 else 0
            
            crow_feet_region = face_gray[int(h*0.35):int(h*0.5), int(w*0.05):int(w*0.25)]
            crow_feet_right = face_gray[int(h*0.35):int(h*0.5), int(w*0.75):int(w*0.95)]
            crow_feet_score = 0
            if crow_feet_region.size > 0 and crow_feet_right.size > 0:
                crow_edges_left = cv2.Canny(crow_feet_region, 20, 60)
                crow_edges_right = cv2.Canny(crow_feet_right, 20, 60)
                crow_feet_score = (np.sum(crow_edges_left) + np.sum(crow_edges_right)) / (crow_feet_region.size + crow_feet_right.size)
            
            mouth_region = face_gray[int(h*0.65):int(h*0.85), int(w*0.25):int(w*0.75)]
            nasolabial_score = 0
            if mouth_region.size > 0:
                mouth_edges = cv2.Canny(mouth_region, 25, 75)
                nasolabial_score = np.sum(mouth_edges) / mouth_region.size
            
            neck_region = face_gray[int(h*0.85):h, int(w*0.3):int(w*0.7)]
            neck_texture = 0
            if neck_region.size > 0:
                neck_edges = cv2.Canny(neck_region, 15, 45)
                neck_texture = np.sum(neck_edges) / neck_region.size
            
            face_brightness = np.mean(face_gray)
            face_contrast = np.std(face_gray.astype(np.float32))
            
            skin_elasticity = 1.0 - min(1.0, texture_variance / 500.0)
            
            face_roundness = w / max(1, h)
            if face_roundness > 0.9:
                child_indicator = 1.5
            elif face_roundness > 0.85:
                child_indicator = 1.0
            else:
                child_indicator = 0.0
            
            eye_size_ratio = 0.0
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(eye_region, 1.1, 3)
            if len(eyes) > 0:
                avg_eye_area = np.mean([ew * eh for ex, ey, ew, eh in eyes])
                eye_size_ratio = avg_eye_area / (w * h)
                if eye_size_ratio > 0.035:
                    child_indicator += 2.0
                elif eye_size_ratio > 0.025:
                    child_indicator += 1.0
            
            forehead_height_ratio = forehead_region.shape[0] / h
            if forehead_height_ratio > 0.3:
                child_indicator += 1.0
            elif forehead_height_ratio < 0.2:
                elderly_indicator = 0.5
            else:
                elderly_indicator = 0.0
            
            elderly_indicators = 0.0
            
            if texture_variance > 800:
                elderly_indicators += 3.0
            elif texture_variance > 600:
                elderly_indicators += 2.0
            elif texture_variance > 400:
                elderly_indicators += 1.0
            
            if crow_feet_score > 0.015:
                elderly_indicators += 2.5
            elif crow_feet_score > 0.01:
                elderly_indicators += 1.5
            
            if nasolabial_score > 0.02:
                elderly_indicators += 2.0
            elif nasolabial_score > 0.015:
                elderly_indicators += 1.0
            
            if neck_texture > 0.01:
                elderly_indicators += 1.5
            
            if skin_elasticity < 0.3:
                elderly_indicators += 2.0
            elif skin_elasticity < 0.5:
                elderly_indicators += 1.0
            
            if face_contrast < 25:
                elderly_indicators += 1.0
            elif face_contrast > 45:
                child_indicator += 0.5
            
            child_indicators = child_indicator
            
            if face_brightness > 180:
                child_indicators += 1.0
            elif face_brightness < 120:
                elderly_indicators += 0.5

            try:
                quality_gate = float(self._assess_image_quality(face_img))
            except Exception:
                quality_gate = 0.5
            if quality_gate < 0.6:
                elderly_indicators *= 0.7
                child_indicators *= 0.7
            
            # Reforço por cabelo grisalho/branco
            gray_hair_score = self._detect_gray_hair(face_img, face_gray)

            base_age = 30
            
            if child_indicators > 3.8:
                if child_indicators > 5.0:
                    estimated_age = max(3, min(12, int(8 - child_indicators * 0.5)))
                else:
                    estimated_age = max(8, min(18, int(20 - child_indicators * 2)))
                confidence_modifier = 0.15
            
            elif elderly_indicators > 6.5:
                if elderly_indicators > 8.5:
                    estimated_age = max(70, min(95, int(55 + elderly_indicators * 3.5)))
                elif elderly_indicators > 7.5:
                    estimated_age = max(60, min(85, int(45 + elderly_indicators * 4)))
                else:
                    estimated_age = max(50, min(75, int(35 + elderly_indicators * 5)))
                confidence_modifier = 0.12
            
            else:
                age_score = base_age + (texture_variance / 40) + (eye_texture / 15) + (forehead_texture / 18) + (cheek_texture / 120)
                age_score += elderly_indicators * 3 - child_indicators * 2
                if crow_feet_score > 0.008 or nasolabial_score > 0.012:
                    age_score += gray_hair_score * 5
                estimated_age = max(18, min(65, int(age_score)))
                confidence_modifier = 0.0
            
            if 19 <= estimated_age <= 26 and child_indicators < 4.2:
                estimated_age = max(20, estimated_age)
            if 58 <= estimated_age <= 65 and elderly_indicators < 7.0:
                estimated_age = min(60, estimated_age)
            estimated_age = max(1, min(100, estimated_age))
            
            age_category = 'Jovem Adulto'
            for min_age, max_age, category in self.age_ranges:
                if min_age <= estimated_age <= max_age:
                    age_category = category
                    break
            
            base_confidence = 0.7 - abs(texture_variance - 350) / 1200
            confidence = max(0.35, min(0.92, base_confidence + confidence_modifier))
            
            if child_indicators > 4.0 or elderly_indicators > 6.0:
                confidence += 0.1
            
            return {
                'estimated_age': estimated_age,
                'age_range': f"{max(1, estimated_age-4)}-{min(100, estimated_age+4)}",
                'category': age_category,
                'confidence': float(confidence),
                'debug_info': {
                    'child_indicators': float(child_indicators),
                    'elderly_indicators': float(elderly_indicators),
                    'texture_variance': float(texture_variance),
                    'crow_feet_score': float(crow_feet_score),
                    'skin_elasticity': float(skin_elasticity),
                    'face_roundness': float(face_roundness),
                    'eye_size_ratio': float(eye_size_ratio),
                    'gray_hair_score': float(gray_hair_score)
                }
            }
        except Exception as e:
            print(f"Erro na estimativa de idade: {e}")
            return {'estimated_age': 30, 'category': 'Jovem Adulto', 'confidence': 0.3}
    
    def _estimate_gender(self, face_img: np.ndarray, face_gray: np.ndarray) -> Dict[str, Any]:
        if self.use_advanced_gender:
            try:
                return self.advanced_gender_detector.detect_gender_advanced(face_img)
            except Exception as e:
                print(f"Erro no detector avançado: {e}")
        
        return self._estimate_gender_traditional(face_img, face_gray)
    
    def _estimate_gender_traditional(self, face_img: np.ndarray, face_gray: np.ndarray) -> Dict[str, Any]:
        try:
            h, w = face_gray.shape
            
            if h < 50 or w < 50:
                return {
                    'predicted_gender': 'Indefinido',
                    'male_probability': 0.5,
                    'female_probability': 0.5,
                    'confidence': 0.2
                }
            
            jaw_region = face_gray[int(h*0.75):h, int(w*0.1):int(w*0.9)]
            if jaw_region.size > 100:
                jaw_edges = cv2.Canny(jaw_region, 30, 100)
                jaw_strength = np.sum(jaw_edges > 0) / max(1, jaw_edges.size)
                
                jaw_variance = np.var(jaw_region.astype(np.float32))
                jaw_definition = min(1.0, jaw_variance / 300.0)
            else:
                jaw_strength = 0.0
                jaw_definition = 0.0
            
            eyebrow_region = face_gray[int(h*0.15):int(h*0.35), int(w*0.1):int(w*0.9)]
            if eyebrow_region.size > 50:
                eyebrow_edges = cv2.Canny(eyebrow_region, 20, 60)
                eyebrow_density = np.sum(eyebrow_edges > 0) / max(1, eyebrow_edges.size)
                eyebrow_thickness = np.std(eyebrow_region.astype(np.float32))
                
                eyebrow_contours, _ = cv2.findContours(eyebrow_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                thick_eyebrows = len([c for c in eyebrow_contours if cv2.contourArea(c) > 25])
            else:
                eyebrow_density = 0.0
                eyebrow_thickness = 0.0
                thick_eyebrows = 0
            
            skin_texture = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            skin_roughness = min(1.0, skin_texture / 150.0)
            skin_smoothness = 1.0 - skin_roughness
            
            lip_region = face_gray[int(h*0.6):int(h*0.8), int(w*0.25):int(w*0.75)]
            if lip_region.size > 50:
                lip_edges = cv2.Canny(lip_region, 20, 60)
                lip_definition = np.sum(lip_edges > 0) / max(1, lip_edges.size)
                lip_contrast = np.std(lip_region.astype(np.float32))
            else:
                lip_definition = 0.0
                lip_contrast = 0.0
            
            face_width_height_ratio = w / max(1, h)
            
            eye_region = face_gray[int(h*0.25):int(h*0.5), int(w*0.15):int(w*0.85)]
            if eye_region.size > 50:
                eyes_detected = self.eye_cascade.detectMultiScale(eye_region, 1.1, 3)
                eye_sharpness = cv2.Laplacian(eye_region, cv2.CV_64F).var()
            else:
                eyes_detected = []
                eye_sharpness = 0.0
            
            masculine_score = 0.0
            
            if jaw_strength > 0.2:
                masculine_score += jaw_strength * 0.25
            if jaw_definition > 0.3:
                masculine_score += jaw_definition * 0.15
            
            if eyebrow_density > 0.15:
                masculine_score += eyebrow_density * 0.20
            if eyebrow_thickness > 20:
                masculine_score += min(1.0, eyebrow_thickness / 50.0) * 0.15
            if thick_eyebrows > 0:
                masculine_score += 0.10
            
            if skin_roughness > 0.3:
                masculine_score += skin_roughness * 0.15
            
            if face_width_height_ratio > 0.85:
                masculine_score += (face_width_height_ratio - 0.85) * 0.10
            
            feminine_score = 0.0
            
            if skin_smoothness > 0.6:
                feminine_score += skin_smoothness * 0.25
            
            if lip_definition > 0.2:
                feminine_score += lip_definition * 0.20
            if lip_contrast > 15:
                feminine_score += min(1.0, lip_contrast / 40.0) * 0.15
            
            if jaw_strength < 0.3:
                feminine_score += (0.3 - jaw_strength) * 0.20
            
            if eyebrow_density < 0.25:
                feminine_score += (0.25 - eyebrow_density) * 0.15
            
            if face_width_height_ratio < 0.85:
                feminine_score += (0.85 - face_width_height_ratio) * 0.05
            
            total_score = masculine_score + feminine_score
            
            if total_score > 0:
                male_probability = masculine_score / total_score
            else:
                male_probability = 0.5
            
            if jaw_strength > 0.4 and eyebrow_density > 0.3:
                male_probability = min(0.95, male_probability + 0.25)
            elif jaw_strength > 0.35:
                male_probability = min(0.90, male_probability + 0.20)
            elif eyebrow_density > 0.35 and eyebrow_thickness > 25:
                male_probability = min(0.85, male_probability + 0.15)
            
            elif skin_smoothness > 0.8 and lip_definition > 0.3:
                male_probability = max(0.05, male_probability - 0.25)
            elif lip_contrast > 25 and jaw_strength < 0.2:
                male_probability = max(0.10, male_probability - 0.20)
            elif skin_smoothness > 0.7:
                male_probability = max(0.15, male_probability - 0.15)
            
            male_probability = max(0.01, min(0.99, male_probability))
            female_probability = 1.0 - male_probability
            
            if np.isnan(male_probability) or np.isnan(female_probability):
                print("⚠️ NaN detectado, usando valores padrão")
                male_probability = 0.5
                female_probability = 0.5
            
            if male_probability > female_probability:
                gender = 'Masculino'
                confidence_base = male_probability
            else:
                gender = 'Feminino'
                confidence_base = female_probability
            
            if male_probability > 0.75 or male_probability < 0.25:
                confidence_raw = confidence_base + 0.2
            elif male_probability > 0.65 or male_probability < 0.35:
                confidence_raw = confidence_base + 0.1
            else:
                confidence_raw = confidence_base
            
            confidence = max(0.3, min(0.95, confidence_raw))
            
            print(f"🔍 Gender Analysis: M={male_probability:.3f}, F={female_probability:.3f}, "
                  f"Jaw={jaw_strength:.3f}, Eyebrow={eyebrow_density:.3f}, "
                  f"Skin={skin_roughness:.3f}, Result={gender}")
            
            return {
                'predicted_gender': gender,
                'male_probability': float(male_probability),
                'female_probability': float(female_probability),
                'confidence': float(confidence),
                'debug_info': {
                    'masculine_score': float(masculine_score),
                    'feminine_score': float(feminine_score),
                    'total_score': float(total_score),
                    'jaw_strength': float(jaw_strength),
                    'eyebrow_density': float(eyebrow_density),
                    'skin_roughness': float(skin_roughness),
                    'lip_definition': float(lip_definition),
                    'face_ratio': float(face_width_height_ratio)
                }
            }
            
        except Exception as e:
            print(f"❌ Erro na estimativa de gênero: {e}")
            return {
                'predicted_gender': 'Indefinido',
                'male_probability': 0.5,
                'female_probability': 0.5,
                'confidence': 0.3
            }
    
    def _analyze_facial_features(self, face_img: np.ndarray, face_gray: np.ndarray) -> Dict[str, Any]:
        try:
            h, w = face_gray.shape
            
            face_ratio = h / w
            face_shape = 'Oval'
            if face_ratio > 1.3:
                face_shape = 'Retangular'
            elif face_ratio < 0.9:
                face_shape = 'Redondo'
            elif face_ratio > 1.1:
                face_shape = 'Oval'
            else:
                face_shape = 'Quadrado'
            
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            brightness = np.mean(lab[:, :, 0])
            
            if brightness > 180:
                skin_type = 'Muito Clara'
            elif brightness > 150:
                skin_type = 'Clara'
            elif brightness > 120:
                skin_type = 'Média'
            elif brightness > 90:
                skin_type = 'Morena'
            else:
                skin_type = 'Escura'
            
            eye_color = 'Castanho'
            eye_shape = 'Normal'
            
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 5)
            if len(eyes) > 0:
                ex, ey, ew, eh = eyes[0]
                eye_ratio = ew / eh if eh > 0 else 1
                if eye_ratio > 2.5:
                    eye_shape = 'Amendoado'
                elif eye_ratio < 1.8:
                    eye_shape = 'Redondo'
                try:
                    eye_roi_bgr = face_img[max(ey,0):ey+eh, max(ex,0):ex+ew]
                    if eye_roi_bgr.size > 0:
                        eye_hsv = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(eye_hsv)
                        mask_dark = (v < 130)
                        mask_color = (s > 40)
                        mask = (mask_dark & mask_color)
                        if np.count_nonzero(mask) < 50:
                            mask = (v < 150)
                        if np.count_nonzero(mask) > 0:
                            hue_vals = h[mask]
                            sat_vals = s[mask]
                            val_vals = v[mask]
                            mean_h = float(np.mean(hue_vals))
                            mean_s = float(np.mean(sat_vals))
                            mean_v = float(np.mean(val_vals))
                            if mean_v < 70:
                                eye_color = 'Preto'
                            else:
                                if mean_s < 60:
                                    eye_color = 'Castanho'
                                else:
                                    if 15 <= mean_h <= 45:
                                        eye_color = 'Âmbar'
                                    elif 45 < mean_h <= 85:
                                        eye_color = 'Verde'
                                    elif (85 < mean_h <= 130) or (mean_h < 10):
                                        eye_color = 'Azul'
                                    else:
                                        eye_color = 'Castanho'
                except Exception:
                    pass
            
            return {
                'formato_rosto': face_shape,
                'tipo_pele': skin_type,
                'cor_olhos_estimada': eye_color,
                'formato_olhos': eye_shape,
                'nariz': 'Médio',
                'labios': 'Médios',
                'sobrancelhas': 'Médias',
                'confidence': 0.7
            }
        except Exception as e:
            print(f"Erro na análise de características: {e}")
            return {
                'formato_rosto': 'Oval',
                'tipo_pele': 'Média',
                'cor_olhos_estimada': 'Castanho',
                'formato_olhos': 'Normal',
                'nariz': 'Médio',
                'labios': 'Médios',
                'sobrancelhas': 'Médias',
                'confidence': 0.3
            }
    
    def _estimate_ethnicity(self, face_img: np.ndarray) -> Dict[str, Any]:
      
        if hasattr(self, 'use_advanced_ethnicity') and self.use_advanced_ethnicity:
            try:
                result = self.advanced_ethnicity_detector.detect_ethnicity_advanced(face_img)
                if result and 'predicted_ethnicity' in result and result['confidence'] > 0.3:
                    return {
                        'predicted_ethnicity': result['predicted_ethnicity'],
                        'confidence': float(result['confidence'])
                    }
            except Exception as e:
                print(f"Erro no detector avançado de etnia: {e}")

        return self._traditional_ethnicity_estimation(face_img)
    
    def _traditional_ethnicity_estimation(self, face_img: np.ndarray) -> Dict[str, Any]:

        try:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            brightness = np.mean(lab[:, :, 0])
            
            if brightness > 160:
                ethnicity = 'Caucasiano'
                confidence = 0.7
            elif brightness > 130:
                ethnicity = 'Asiático'
                confidence = 0.6
            elif brightness > 100:
                ethnicity = 'Hispânico'
                confidence = 0.6
            elif brightness > 70:
                ethnicity = 'Árabe'
                confidence = 0.5
            else:
                ethnicity = 'Africano'
                confidence = 0.6
            
            return {
                'predicted_ethnicity': ethnicity,
                'confidence': float(confidence)
            }
        except Exception as e:
            print(f"Erro na estimativa de etnia: {e}")
            return {
                'predicted_ethnicity': 'Indefinido',
                'confidence': 0.3
            }
    
    def _analyze_expression(self, face_img: np.ndarray, face_gray: np.ndarray) -> Dict[str, Any]:
        try:
            smiles = self.smile_cascade.detectMultiScale(face_gray, 1.3, 5)
            has_smile = len(smiles) > 0
            
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 5)
            eye_openness = 0.8 if len(eyes) > 0 else 0.3
            
            # Heurísticas simples para pontuar emoções
            # Baseadas em abertura dos olhos e presença de sorriso
            scores = {
                'Feliz': 1.2 * (1.0 if has_smile else 0.2) + 0.2 * eye_openness,
                'Surpresa': 0.3 * (1.0 if has_smile else 0.0) + 1.1 * eye_openness,
                'Neutro': 0.6 * (1.0 - abs(eye_openness - 0.5)),
                'Triste': 0.5 * (1.0 - eye_openness) + (0.1 if not has_smile else 0.0),
                'Raiva': 0.25 * (1.0 - eye_openness) + (0.05 if not has_smile else 0.0),
                'Nojo': 0.15 * (1.0 - eye_openness),
                'Medo': 0.35 * eye_openness
            }
            # Normalização tipo softmax simples
            import numpy as _np
            vals = _np.array(list(scores.values()), dtype=_np.float64)
            vals = vals - vals.max()
            exp = _np.exp(vals)
            probs_arr = exp / (exp.sum() if exp.sum() > 0 else 1.0)
            emotions = list(scores.keys())
            probabilities = {k: float(v) for k, v in zip(emotions, probs_arr)}
            top3 = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_label, top_prob = top3[0]
            expressions = [top_label] if has_smile and top_label != 'Feliz' else ([top_label] if top_prob > 0.4 else ['Neutro'])
            
            return {
                'expressions': expressions,
                'has_smile': has_smile,
                'eye_openness': float(eye_openness),
                'confidence': float(top_prob),
                'probabilities': probabilities,
                'top3': [{'emotion': e, 'prob': float(p)} for e, p in top3]
            }
        except Exception as e:
            print(f"Erro na análise de expressão: {e}")
            return {
                'expressions': ['Neutro'],
                'has_smile': False,
                'eye_openness': 0.5,
                'confidence': 0.3
            }
    
    def _assess_image_quality(self, face_img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)
            
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            return max(0.3, (sharpness_score + brightness_score) / 2.0)
        except:
            return 0.5
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            ages = [
                r['age'].get('estimated_age')
                for r in results
                if 'age' in r and isinstance(r.get('age', {}).get('estimated_age'), (int, float))
                and r.get('age', {}).get('estimated_age') not in (None,)
                and r.get('age', {}).get('confidence', 0.0) > 0.0
            ]
            
            # CORREÇÃO: Verificar múltiplos campos de gênero
            genders = []
            for r in results:
                if 'gender' in r:
                    # Verificar diferentes formatos de resposta
                    if 'predicted_gender' in r['gender']:
                        genders.append(r['gender']['predicted_gender'])
                    elif 'gender' in r['gender']:
                        genders.append(r['gender']['gender'])
                    else:
                        # Calcular baseado na probabilidade
                        male_prob = r['gender'].get('male_probability', 0.5)
                        genders.append('Masculino' if male_prob > 0.5 else 'Feminino')
        
            male_count = sum(1 for g in genders if g == 'Masculino')
            female_count = sum(1 for g in genders if g == 'Feminino')
            
            print(f"🔍 Summary Debug: Total faces={len(results)}, Genders={genders}, M={male_count}, F={female_count}")
            
            confidences = [r.get('confidence_score', 0.5) for r in results]
            
            ethnicity_list = []
            for r in results:
                if 'ethnicity' in r and 'predicted_ethnicity' in r['ethnicity']:
                    ethnicity_list.append(r['ethnicity']['predicted_ethnicity'])
            
            return {
                'idade_media': int(np.mean(ages)) if ages else 0,
                'distribuicao_genero': {
                    'masculino': male_count,
                    'feminino': female_count,
                    'total': len(genders),
                    'indefinido': len(results) - len(genders)
                },
                'confianca_media': float(np.mean(confidences)),
                'etnias_detectadas': len(set(ethnicity_list)),
                'faces_com_sorriso': sum(1 for r in results if r.get('expression', {}).get('has_smile', False))
            }
        except Exception as e:
            print(f"❌ Erro na geração do resumo: {e}")
            return {
                'idade_media': 30,
                'distribuicao_genero': {'masculino': 0, 'feminino': 0, 'total': 0, 'indefinido': 0},
                'confianca_media': 0.5,
                'etnias_detectadas': 1,
                'faces_com_sorriso': 0
            }
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            'age': {'estimated_age': None, 'category': 'Indefinido', 'confidence': 0.0},
            'gender': {'predicted_gender': 'Indefinido', 'confidence': 0.3},
            'facial_features': {'formato_rosto': 'Oval', 'confidence': 0.3},
            'ethnicity': {'predicted_ethnicity': 'Indefinido', 'confidence': 0.3},
            'expression': {'expressions': ['Neutro'], 'confidence': 0.3},
            'quality_score': 0.5,
            'confidence_score': 0.3
        }
