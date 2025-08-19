from flask import Flask, request, send_file, jsonify, redirect
import io
import base64
import hashlib
from flask_cors import CORS
from config import Config
from cache_manager import CacheManager
from emotion_detector import EmotionDetector
from url_services import QRCodeGenerator, URLShortener
from utils import InputValidator
from datetime import datetime
from analytics_manager import AnalyticsManager
from demographics_detector import AdvancedDemographicsDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
CORS(app)

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'Arquivo muito grande. Tamanho máximo permitido: 50MB',
        'error_code': 'REQUEST_TOO_LARGE'
    }), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'error': 'Requisição inválida',
        'error_code': 'BAD_REQUEST'
    }), 400

cache_manager = CacheManager()
emotion_detector = EmotionDetector()
qr_generator = QRCodeGenerator()
url_shortener = URLShortener()
analytics_manager = AnalyticsManager(cache_manager)
demographics_detector = AdvancedDemographicsDetector()

def _log_runtime_info():
    try:
        import importlib
        info = {}
 
        try:
            tflite_spec = importlib.util.find_spec('tflite_runtime')
            if tflite_spec is not None:
                import tflite_runtime as tr
                info['tflite_runtime_version'] = getattr(tr, '__version__', 'unknown')
            else:
                info['tflite_runtime_version'] = 'not installed'
        except Exception as e:
            info['tflite_runtime_error'] = str(e)

      
        model_paths = []
        try:
            from config import Config
            age_cfg = getattr(Config, 'AGE_CONFIG', {})
            gender_cfg = getattr(Config, 'GENDER_CONFIG', {})
            if age_cfg.get('tflite_age_model_path'):
                model_paths.append(('age_model', age_cfg.get('tflite_age_model_path')))
            if gender_cfg.get('tflite_gender_model_path'):
                model_paths.append(('gender_model', gender_cfg.get('tflite_gender_model_path')))
        except Exception:
            pass
        checksums = {}
        for name, path in model_paths:
            try:
                with open(path, 'rb') as f:
                    checksums[name] = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception as e:
                checksums[name] = f'error:{e}'
        info['model_checksums'] = checksums


        try:
            from config import Config
            info['age_min_adult_no_child'] = getattr(Config, 'AGE_CONFIG', {}).get('min_adult_age_without_strong_child_evidence')
            info['age_child_prob_vote_threshold'] = getattr(Config, 'AGE_CONFIG', {}).get('child_prob_vote_threshold')
            info['age_child_cap_strict_votes'] = getattr(Config, 'AGE_CONFIG', {}).get('child_cap_strict_votes')
            info['gender_global_threshold'] = getattr(Config, 'GENDER_CONFIG', {}).get('global_threshold')
        except Exception:
            pass

        print(f"[RUNTIME] {info}")
    except Exception as e:
        print(f"[RUNTIME] log error: {e}")


_log_runtime_info()


@app.route("/api/qrcode", methods=["POST"])
def generate_qr():
    try:
        data = request.get_json()
        url = InputValidator.validate_url(data.get("url"))
        
        cache_key = cache_manager.generate_key('qr', url)
        cached_qr = cache_manager.get(cache_key)
        
        if cached_qr:
            img_data = cached_qr['binary_data'] if 'binary_data' in cached_qr else base64.b64decode(cached_qr['image_data'])
            return send_file(io.BytesIO(img_data), mimetype='image/png')
        
        qr_data = qr_generator.generate(url)
        cache_manager.set(cache_key, qr_data, expire_hours=168)
        
        return send_file(io.BytesIO(qr_data['binary_data']), mimetype='image/png')
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Erro interno do servidor'}), 500


@app.route("/api/qrcode/download", methods=["POST"])
def download_qr():
    try:
        data = request.get_json()
        url = InputValidator.validate_url(data.get("url"))
        filename = data.get("filename", "qrcode")
        
        cache_key = cache_manager.generate_key('qr', url)
        cached_qr = cache_manager.get(cache_key)
        
        if cached_qr:
            img_data = cached_qr['binary_data'] if 'binary_data' in cached_qr else base64.b64decode(cached_qr['image_data'])
        else:
            qr_data = qr_generator.generate(url)
            cache_manager.set(cache_key, qr_data, expire_hours=168)
            img_data = qr_data['binary_data']
        
        return send_file(
            io.BytesIO(img_data), 
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{filename}.png"
        )
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Erro interno do servidor'}), 500


@app.route("/api/qrcode/base64", methods=["POST"])
def get_qr_base64():
    try:
        data = request.get_json()
        url = InputValidator.validate_url(data.get("url"))
        
        cache_key = cache_manager.generate_key('qr', url)
        cached_qr = cache_manager.get(cache_key)
        
        if cached_qr:
            img_data = cached_qr['binary_data'] if 'binary_data' in cached_qr else base64.b64decode(cached_qr['image_data'])
        else:
            qr_data = qr_generator.generate(url)
            cache_manager.set(cache_key, qr_data, expire_hours=168)
            img_data = qr_data['binary_data']
        
        base64_image = base64.b64encode(img_data).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image_base64': f"data:image/png;base64,{base64_image}",
            'url': url,
            'cached': cached_qr is not None
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Erro interno do servidor'}), 500


@app.route("/api/shorten", methods=["POST"])
def shorten_url():
    try:
        data = request.get_json()
        url = InputValidator.validate_url(data.get("url"))
        
        url_cache_key = cache_manager.generate_key('url_mapping', url)
        cached_short = cache_manager.get(url_cache_key)
        
        
        def _determine_base_url(req) -> str:
          
            if getattr(Config, 'SHORT_BASE_URL', None):
                base = Config.SHORT_BASE_URL.strip()
                if not base.endswith('/'):
                    base += '/'
                return base

           
            xf_host = req.headers.get('X-Forwarded-Host')
            xf_proto = req.headers.get('X-Forwarded-Proto')
            if xf_host and xf_proto:
                return f"{xf_proto}://{xf_host}/"

         
            return req.host_url

        base_url = _determine_base_url(request)

        if cached_short:
            return jsonify({
                'short_url': base_url + 's/' + cached_short['short_id'],
                'cached': True,
                'created_at': cached_short.get('created_at')
            })
        
        short_data = url_shortener.create_short_url(url, base_url)
        
        cache_manager.set(url_cache_key, short_data, expire_hours=8760)
        cache_manager.set(f"short_id:{short_data['short_id']}", short_data, expire_hours=8760)
        
        return jsonify({
            'short_url': short_data['short_url'],
            'cached': False,
            'created_at': short_data['created_at']
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Erro interno do servidor'}), 500


@app.route('/s/<short_id>')
def redirect_short_url(short_id):
    try:
        InputValidator.validate_short_id(short_id)
        
        cached_data = cache_manager.get(f"short_id:{short_id}")
        
        if cached_data:
            # Registrar analytics
            request_data = {
                'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown')),
                'user_agent': request.environ.get('HTTP_USER_AGENT', 'unknown'),
                'referer': request.environ.get('HTTP_REFERER', 'direct')
            }
            analytics_manager.log_url_access(short_id, request_data)
            
            # Atualizar contador
            updated_data = url_shortener.update_access_count(short_id, cached_data)
            cache_manager.set(f"short_id:{short_id}", updated_data, expire_hours=8760)
            
            return redirect(cached_data['original_url'])
        
        url = url_shortener.get_original_url(short_id)
        if url:
            # Registrar analytics mesmo para URLs não cacheadas
            request_data = {
                'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown')),
                'user_agent': request.environ.get('HTTP_USER_AGENT', 'unknown'),
                'referer': request.environ.get('HTTP_REFERER', 'direct')
            }
            analytics_manager.log_url_access(short_id, request_data)
            
            return redirect(url)
        
        return jsonify({'error': 'URL não encontrada'}), 404
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Erro interno do servidor'}), 500


@app.route("/api/emotion", methods=["POST"])
def detect_emotion():
    try:
        data = request.get_json()
        image_data = InputValidator.validate_image_data(data.get("image"))
        
        image_hash = hashlib.md5(image_data.encode()).hexdigest()
        cache_key = cache_manager.generate_key('emotion', image_hash)
        
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            return jsonify(cached_result)
        
        result = emotion_detector.process_image(image_data)
        cache_manager.set(cache_key, result, expire_hours=24)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Erro ao processar imagem: {str(e)}'}), 500


@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    try:
        return jsonify(cache_manager.get_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    try:
        if cache_manager.clear():
            return jsonify({
                'message': 'Cache limpo com sucesso',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Erro ao limpar cache'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/url/<short_id>/analytics", methods=["GET"])
def get_url_analytics(short_id):
    """Obtém analytics detalhadas para uma URL específica"""
    try:
        InputValidator.validate_short_id(short_id)
        
        # Verificar se a URL existe
        url_data = cache_manager.get(f"short_id:{short_id}")
        if not url_data:
            return jsonify({'error': 'URL não encontrada'}), 404
        
        # Obter analytics
        analytics = analytics_manager.get_url_analytics(short_id)
        
        # Adicionar informações da URL
        analytics['url_info'] = {
            'short_id': short_id,
            'original_url': url_data.get('original_url'),
            'created_at': url_data.get('created_at'),
            'total_access_count': url_data.get('access_count', 0)
        }
        
        return jsonify(analytics)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Erro interno do servidor'}), 500


@app.route("/api/analytics/global", methods=["GET"])
def get_global_analytics():
    """Obtém analytics globais do sistema"""
    try:
        analytics = analytics_manager.get_global_analytics()
        return jsonify(analytics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/demographics", methods=["POST"])
def detect_demographics():
    """Detecta idade, gênero e características faciais"""
    try:
        # Verificar tamanho do conteúdo da requisição
        content_length = request.content_length
        if content_length and content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'error': 'Imagem muito grande. Tamanho máximo: 50MB',
                'error_code': 'IMAGE_TOO_LARGE'
            }), 413
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'error': 'Dados de imagem não fornecidos',
                'error_code': 'NO_IMAGE_DATA'
            }), 400
            
        image_data = InputValidator.validate_image_data(data.get("image"))
        
        # Verificar tamanho da imagem em base64
        estimated_size = len(image_data) * 0.75  # Aproximação do tamanho real
        if estimated_size > 30 * 1024 * 1024:  # 30MB em base64
            return jsonify({
                'error': 'Imagem muito grande. Reduza o tamanho da imagem e tente novamente.',
                'error_code': 'IMAGE_TOO_LARGE'
            }), 413
        
        # Verificar cache
        image_hash = hashlib.md5(image_data.encode()).hexdigest()
        cache_key = cache_manager.generate_key('demographics', image_hash)
        
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            return jsonify(cached_result)
        
        # Analisar demografia
        result = demographics_detector.analyze_demographics(image_data)
        
        # Salvar no cache (24 horas)
        cache_manager.set(cache_key, result, expire_hours=24)
        
        result['from_cache'] = False
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'error_code': 'VALIDATION_ERROR'
        }), 400
    except Exception as e:
        print(f"Erro na análise demográfica: {e}")
        return jsonify({
            'error': f'Erro interno do servidor. Tente novamente.',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route("/api/analytics/dashboard", methods=["GET"])
def get_dashboard_data():
    """Endpoint para dados do dashboard completo"""
    try:
        # Analytics globais
        global_analytics = analytics_manager.get_global_analytics()
        
        # Estatísticas do cache
        cache_stats = cache_manager.get_stats()
        
        # Combinar dados
        dashboard_data = {
            'global_analytics': global_analytics,
            'cache_stats': cache_stats,
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'services_active': ['qr_generator', 'url_shortener', 'emotion_detector', 'demographics_detector'],
                'total_features': 4
            }
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml-status', methods=['GET'])
def get_ml_status():
    try:
        from demographics_detector import ML_AVAILABLE
        
        status_info = {
            'ml_available': ML_AVAILABLE,
            'components': {},
            'errors': []
        }
        
        # Testar cada componente
        try:
            import importlib
            tf_spec = importlib.util.find_spec('tensorflow')
            if tf_spec is not None:
                import tensorflow as tf
                status_info['components']['tensorflow'] = {
                    'available': True,
                    'version': tf.__version__
                }
            else:
                raise ImportError('tensorflow não instalado')
        except ImportError as e:
            status_info['components']['tensorflow'] = {
                'available': False,
                'error': str(e)
            }
            status_info['errors'].append('TensorFlow não disponível')
        
        try:
            from keras_facenet import FaceNet
            status_info['components']['facenet'] = {'available': True}
        except ImportError as e:
            status_info['components']['facenet'] = {
                'available': False,
                'error': str(e)
            }
            status_info['errors'].append('FaceNet não disponível')
        
        try:
            from mtcnn import MTCNN
            status_info['components']['mtcnn'] = {'available': True}
        except ImportError as e:
            status_info['components']['mtcnn'] = {
                'available': False,
                'error': str(e)
            }
            status_info['errors'].append('MTCNN não disponível')
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            status_info['components']['sklearn'] = {'available': True}
        except ImportError as e:
            status_info['components']['sklearn'] = {
                'available': False,
                'error': str(e)
            }
            status_info['errors'].append('scikit-learn não disponível')
        
        # Testar detector
        try:
            detector = AdvancedDemographicsDetector()
            status_info['detectors'] = {
                'gender': {
                    'advanced': hasattr(detector, 'advanced_gender_detector'),
                    'ml_enabled': detector.advanced_gender_detector.ml_enabled if hasattr(detector, 'advanced_gender_detector') else False
                },
                'ethnicity': {
                    'advanced': hasattr(detector, 'advanced_ethnicity_detector'),
                    'ml_enabled': detector.advanced_ethnicity_detector.ml_enabled if hasattr(detector, 'advanced_ethnicity_detector') else False
                }
            }
        except Exception as e:
            status_info['detectors'] = {'error': str(e)}
        
        return jsonify(status_info), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Erro ao verificar status ML: {str(e)}',
            'ml_available': False
        }), 500


if __name__ == '__main__':
    app.run(debug=Config.FLASK_DEBUG, host=Config.FLASK_HOST, port=Config.FLASK_PORT)