import os
from datetime import timedelta

class Config:
    REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    
    CACHE_FILE = 'cache_data.json'
    
    CACHE_EXPIRY = {
        'qr_code': timedelta(days=7),
        'url_mapping': timedelta(days=365),
        'emotion': timedelta(hours=24)
    }
    
    SHORT_URL_LENGTH = 6
    # Base URL opcional para o serviço de encurtamento. Se definido, será usado no lugar do host da requisição
    SHORT_BASE_URL = os.environ.get('SHORT_BASE_URL')
    
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
    EMOTION_LABELS = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Neutro', 'Triste', 'Surpresa']
    
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  
    SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'gif', 'bmp']

   
    GENDER_CONFIG = {
        'tflite_only': True,
        'global_threshold': 0.54,
        'tflite_input_size': 96, 
        'tflite_output_order': 'male_female',  
        'invert_output':False, 

        'facenet_male_weight': 0.80,
        'facenet_geometric_weight': 0.20,


        'male_feature_boost_max': 0.07,
    'female_feature_penalty_max': 0.05,


    'feature_scale': 0.02,
    'feature_baseline': 0.25,


        'ensemble_method_weights': {
            'facenet_advanced': 0.42,
            'cnn_transfer_learning': 0.38,
            'traditional_enhanced_v4': 0.10
        },


    'confidence_min_facenet': 0.80,

  
    'ensemble_agreement_multiplier': 0.35,
    'confidence_separation_base': 0.6,
    'confidence_separation_slope': 0.8,

        'low_light_brightness_threshold': 0.35,  
        'low_light_lip_boost': 1.30,
        'low_light_soft_tissue_boost': 1.20,
        'low_light_edge_penalty': 0.40,

        'female_safeguard_enabled': True,
        'strong_female_feature_threshold': 0.88,
        'weak_male_feature_threshold': 0.45,
        'female_safeguard_cap': 0.55,
        'female_safeguard_min_strong': 2,
        'female_safeguard_min_weak': 1,


        'strong_male_feature_threshold': 0.92,
        'weak_female_feature_threshold': 0.45,
        'male_safeguard_floor': 0.62,
        'male_safeguard_min_strong': 2,
        'male_safeguard_min_weak': 1,

   
        'traditional_tie_margin': 0.06,
       
        'beard_coverage_threshold': 0.20,
        'stubble_moderate_threshold': 0.70 ,
        'stubble_strong_threshold': 0.85,
        'mustache_moderate_threshold': 0.70,
        'mustache_strong_threshold': 0.85,
        'beard_male_floor_moderate': 0.75,
        'beard_male_boost_moderate': 0.04,
        'beard_male_floor_strong': 0.82,
        'beard_male_boost_strong': 0.08,
        'traditional_stubble_weight': 2.5,
        'beard_global_male_floor': 0.80,
        'no_beard_female_cap': 0.48,
        'beard_require_multi_indicators': True,
        'beard_evidence_min': 2,
        'beard_low_light_scale': 1.25,
        'beard_lip_contrast_block_threshold': 0.95,
        'beard_soft_tissue_block_threshold': 0.95,
        'beard_jaw_darkness_threshold': 0.20,
        'beard_jaw_darkness_strong_threshold': 0.25,


        'use_tflite': True,
        'tflite_gender_model_path': 'models/gender.tflite',
        'tflite_embedding_model_path': 'models/face_embedding.tflite',
        'tflite_threads': 2,
        'tflite_temperature_confidence': 0.0,
}

    AGE_CONFIG = {
        'tflite_only': True,
        'use_ensemble_with_tflite': True, 
        'age_margin_default': 2,
        'age_margin_child': 2,
        'age_margin_elderly': 3,


        'child_cap_enabled': False,
        'child_cap_prob_threshold': 0.87,
        'child_cap_min_votes': 3,
        'child_cap_strict_votes': 4,
        'child_cap_max_age': 16,
        'child_cap_weighted_ceiling': 22,

        'elderly_cap_enabled': True,
        'elderly_cap_prob_threshold': 0.75,
        'elderly_cap_min_votes': 4,
        'elderly_cap_strict_votes': 4,
        'elderly_cap_min_age': 67,
        'elderly_cap_weighted_floor': 60,

  
        'child_blend_enabled': False,
        'child_vote_min_for_blend': 2,
        'child_target_ceiling': 20,
        'child_blend_alpha': 0.6,
        'child_blend_prob_threshold': 0.78,
        'child_blend_margin': 1,


        'use_tflite_age': True,
        'tflite_age_model_path': 'models/age_regression.tflite',
        'tflite_age_input_size': 96,  
        'tflite_age_threads': 2,
        'tflite_age_regression_scale': 80,  
        'tflite_age_output_type': 'regression',  
        'tflite_age_class_ranges': [
            "0-2", "3-9", "10-19", "20-29", "30-39", 
            "40-49", "50-59", "60-69", "70+"
        ],

        # Regras do ensemble para idade: reduzir peso de textura/tradicional
        'ensemble_method_weights': {
            'tflite_age_regression': 0.70,
            'tflite_age_classes': 0.70,
            'cnn_age_regression': 0.30,
            'facenet_age': 0.30,
            'texture_analysis_enhanced': 0.02,
            'traditional_enhanced': 0.01
        },

        # Votos/limiares adicionais usados no ensemble
        'child_prob_vote_threshold': 0.78,
        'elderly_prob_vote_threshold': 0.82,
        'child_indicators_vote_threshold': 3.5,
        'elderly_indicators_vote_threshold': 6.0,

        # Salvaguarda: não retornar <18 sem evidência infantil forte
        'min_adult_age_without_strong_child_evidence': 21,
    }

    FACE_DETECTION_CONFIG = {
        'merge_iou_threshold': 0.22,
        'merge_center_threshold': 0.28,
        'cluster_iou_threshold': 0.28,
        'cluster_center_threshold': 0.35,
        'max_faces': 3,
        'min_face_size': 40,
    }