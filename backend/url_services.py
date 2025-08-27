import qrcode
import io
import base64
from datetime import datetime
from typing import Dict, Any


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



