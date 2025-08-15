import redis
import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config import Config


class CacheManager:
    def __init__(self):
        self.cache = None
        self.cache_type = None
        self.url_map = {}
        self.cache_file = 'cache_data.json'
        self._initialize_cache()
    
    def _initialize_cache(self):
        try:
            self.cache = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=True
            )
            self.cache.ping()
            print("✅ Cache Redis conectado")
            self.cache_type = 'redis'
        except:
            self.cache = {}
            self.cache_type = 'memory'
            print("⚠️ Usando cache em memória")
            self._load_from_file()
    
    def _load_from_file(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self.cache_type == 'memory':
                        self.cache.update(data.get('cache', {}))
                    self.url_map.update(data.get('url_map', {}))
                    print(f"📁 Cache carregado: {len(self.url_map)} URLs")
        except Exception as e:
            print(f"❌ Erro ao carregar cache: {e}")
    
    def _save_to_file(self):
        if self.cache_type == 'memory':
            try:
                data = {
                    'cache': self.cache,
                    'url_map': self.url_map,
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Erro ao salvar cache: {e}")
    
    def get(self, key: str) -> Optional[Dict[Any, Any]]:
        try:
            if self.cache_type == 'redis':
                value = self.cache.get(key)
                return json.loads(value) if value else None
            else:
                cached_item = self.cache.get(key)
                if cached_item and 'expires' in cached_item:
                    if datetime.now() > datetime.fromisoformat(cached_item['expires']):
                        del self.cache[key]
                        return None
                    return cached_item.get('value')
                return cached_item
        except:
            return None
    
    def set(self, key: str, value: Dict[Any, Any], expire_hours: int = 24) -> bool:
        try:
            if self.cache_type == 'redis':
                self.cache.setex(key, timedelta(hours=expire_hours), json.dumps(value))
            else:
                self.cache[key] = {
                    'value': value,
                    'expires': (datetime.now() + timedelta(hours=expire_hours)).isoformat()
                }
                self._save_to_file()
            return True
        except Exception as e:
            print(f"❌ Erro no cache: {e}")
            return False
    
    def generate_key(self, data_type: str, content: str) -> str:
        content_hash = hashlib.md5(str(content).encode()).hexdigest()
        return f"{data_type}:{content_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'cache_type': self.cache_type,
            'url_mappings': len(self.url_map),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.cache_type == 'redis':
            try:
                stats['redis_info'] = {
                    'connected': True,
                    'total_keys': self.cache.dbsize()
                }
            except:
                stats['redis_info'] = {'connected': False}
        else:
            stats['memory_cache'] = {
                'total_items': len(self.cache),
                'cache_file_exists': os.path.exists(self.cache_file)
            }
        
        return stats
    
    def clear(self) -> bool:
        try:
            if self.cache_type == 'redis':
                self.cache.flushdb()
            else:
                self.cache.clear()
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
            
            self.url_map.clear()
            return True
        except:
            return False
