import qrcode
import io
import base64
import string
import random
from datetime import datetime
from typing import Dict, Any, Optional


class QRCodeGenerator:
    @staticmethod
    def generate(url: str) -> Dict[str, Any]:
        img = qrcode.make(url)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        img_data = buf.getvalue()
        return {
            'url': url,
            'image_data': base64.b64encode(img_data).decode(),
            'created_at': datetime.now().isoformat(),
            'type': 'qrcode',
            'binary_data': img_data
        }


class URLShortener:
    def __init__(self):
        self.url_map = {}
    
    @staticmethod
    def _generate_short_id(num_chars: int = 6) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=num_chars))
    
    def create_short_url(self, url: str, base_url: str) -> Dict[str, Any]:
        short_id = self._generate_short_id()
        while short_id in self.url_map:
            short_id = self._generate_short_id()
        
        self.url_map[short_id] = url
        
        return {
            'short_id': short_id,
            'original_url': url,
            'short_url': f"{base_url}s/{short_id}",
            'created_at': datetime.now().isoformat(),
            'access_count': 0,
            'cached': False
        }
    
    def get_original_url(self, short_id: str) -> Optional[str]:
        return self.url_map.get(short_id)
    
    def update_access_count(self, short_id: str, cached_data: Dict[str, Any]) -> Dict[str, Any]:
        cached_data['access_count'] = cached_data.get('access_count', 0) + 1
        cached_data['last_accessed'] = datetime.now().isoformat()
        return cached_data
