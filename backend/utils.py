import re
from urllib.parse import urlparse
from config import Config

class InputValidator:
    @staticmethod
    def validate_url(url):
        if not url or len(url.strip()) == 0:
            raise ValueError("URL não pode estar vazia")
        
        url = url.strip()
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("URL inválida")
            return url
        except Exception:
            raise ValueError("Formato de URL inválido")
    
    @staticmethod
    def validate_image_data(image_data):
        if not image_data:
            raise ValueError("Dados da imagem não fornecidos")
        
        if 'data:image' in image_data and ',' in image_data:
            header, data = image_data.split(',', 1)
            image_format = header.split(';')[0].split('/')[1].lower()
            
            if image_format not in Config.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Formato {image_format} não suportado")
            

            estimated_size = len(data) * 0.75  
            max_size = 30 * 1024 * 1024  
            
            if estimated_size > max_size:
                raise ValueError(f"Imagem muito grande. Tamanho máximo: {max_size // (1024*1024)}MB")
            
            return data
        
       
        if len(image_data) > 40 * 1024 * 1024:  
            raise ValueError("Imagem muito grande. Reduza o tamanho e tente novamente")
        
        return image_data
    

