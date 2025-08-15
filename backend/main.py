

from app import app
from config import Config

if __name__ == '__main__':
    print("🚀 Iniciando Sistema Multi-Serviços...")
    print(f"🌐 Servidor rodando em http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    print("📱 Serviços disponíveis:")
    print("   • Gerador de QR Code")
    print("   • Encurtador de URL") 
    print("   • Detector de Emoções")
    print("   • Detector de Demografia (ML Avançado)")
    print("💾 Cache Redis ativo com fallback em memória")
    print("📊 Upload máximo: 50MB | Timeout: 60s")
    
    app.run(
        debug=Config.FLASK_DEBUG,
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        threaded=True,
        request_handler=None
    )
