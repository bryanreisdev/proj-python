import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, Counter
import geoip2.database
import geoip2.errors
from user_agents import parse
import os

class AnalyticsManager:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.geoip_db_path = 'GeoLite2-City.mmdb'
        
    def log_url_access(self, short_id: str, request_data: Dict[str, Any]) -> None:
       
        access_data = {
            'timestamp': datetime.now().isoformat(),
            'ip': request_data.get('ip', 'unknown'),
            'user_agent': request_data.get('user_agent', 'unknown'),
            'referer': request_data.get('referer', 'direct'),
            'country': self._get_country_from_ip(request_data.get('ip')),
            'device_info': self._parse_user_agent(request_data.get('user_agent'))
        }
        
       
        access_key = f"analytics:access:{short_id}:{datetime.now().timestamp()}"
        self.cache_manager.set(access_key, access_data, expire_hours=8760) 
        
       
        today = datetime.now().strftime('%Y-%m-%d')
        daily_key = f"analytics:daily:{short_id}:{today}"
        daily_count = self.cache_manager.get(daily_key) or 0
        self.cache_manager.set(daily_key, daily_count + 1, expire_hours=8760)
    
    def get_url_analytics(self, short_id: str) -> Dict[str, Any]:
       
        access_pattern = f"analytics:access:{short_id}:*"
        accesses = self._get_cache_by_pattern(access_pattern)
        
        if not accesses:
            return {
                'total_clicks': 0,
                'unique_visitors': 0,
                'countries': {},
                'devices': {},
                'browsers': {},
                'daily_clicks': {},
                'referrers': {},
                'peak_hours': {}
            }
        
        return self._analyze_accesses(accesses)
    
    def get_global_analytics(self) -> Dict[str, Any]:
       
        popular_urls = self._get_popular_urls()
        
       
        total_urls = len(self._get_cache_by_pattern("short_id:*"))
        total_clicks = sum(self._get_cache_by_pattern("analytics:daily:*").values())
        
       
        countries = self._get_global_countries()
        
        return {
            'total_urls': total_urls,
            'total_clicks': total_clicks,
            'popular_urls': popular_urls,
            'top_countries': countries,
            'urls_created_today': self._get_urls_created_today(),
            'clicks_today': self._get_clicks_today()
        }
    
    def _get_country_from_ip(self, ip: str) -> str:
       
        if not ip or ip == 'unknown':
            return 'unknown'
            
        try:
            if os.path.exists(self.geoip_db_path):
                with geoip2.database.Reader(self.geoip_db_path) as reader:
                    response = reader.city(ip)
                    return response.country.name or 'unknown'
        except (geoip2.errors.AddressNotFoundError, Exception):
            pass
        
        return 'unknown'
    
    def _parse_user_agent(self, user_agent: str) -> Dict[str, str]:
       
        if not user_agent:
            return {'browser': 'unknown', 'os': 'unknown', 'device': 'unknown'}
        
        try:
            parsed = parse(user_agent)
            return {
                'browser': f"{parsed.browser.family} {parsed.browser.version_string}",
                'os': f"{parsed.os.family} {parsed.os.version_string}",
                'device': parsed.device.family if parsed.device.family != 'Other' else 'Desktop'
            }
        except Exception:
            return {'browser': 'unknown', 'os': 'unknown', 'device': 'unknown'}
    
    def _get_cache_by_pattern(self, pattern: str) -> Dict[str, Any]:
       
        try:
            if hasattr(self.cache_manager.cache, 'keys'):
               
                keys = self.cache_manager.cache.keys(pattern)
                result = {}
                for key in keys:
                    value = self.cache_manager.get(key.decode() if isinstance(key, bytes) else key)
                    if value:
                        result[key] = value
                return result
            else:
               
                result = {}
                for key, value in self.cache_manager.cache.items():
                    if pattern.replace('*', '') in key:
                        result[key] = value.get('value') if isinstance(value, dict) and 'value' in value else value
                return result
        except Exception:
            return {}
    
    def _analyze_accesses(self, accesses: Dict[str, Any]) -> Dict[str, Any]:
       
        countries = Counter()
        devices = Counter()
        browsers = Counter()
        daily_clicks = defaultdict(int)
        referrers = Counter()
        hourly_clicks = defaultdict(int)
        unique_ips = set()
        
        for access_data in accesses.values():
            if isinstance(access_data, dict):
               
                country = access_data.get('country', 'unknown')
                countries[country] += 1
                

                ip = access_data.get('ip', 'unknown')
                if ip != 'unknown':
                    unique_ips.add(ip)
                
                
                device_info = access_data.get('device_info', {})
                if isinstance(device_info, dict):
                    devices[device_info.get('device', 'unknown')] += 1
                    browsers[device_info.get('browser', 'unknown')] += 1
                
                
                timestamp = access_data.get('timestamp')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                        hour = dt.hour
                        daily_clicks[date_str] += 1
                        hourly_clicks[hour] += 1
                    except Exception:
                        pass
                
                
                referer = access_data.get('referer', 'direct')
                referrers[referer] += 1
        
        return {
            'total_clicks': len(accesses),
            'unique_visitors': len(unique_ips),
            'countries': dict(countries.most_common(10)),
            'devices': dict(devices.most_common(5)),
            'browsers': dict(browsers.most_common(5)),
            'daily_clicks': dict(daily_clicks),
            'referrers': dict(referrers.most_common(10)),
            'peak_hours': dict(sorted(hourly_clicks.items()))
        }
    
    def _get_popular_urls(self) -> List[Dict[str, Any]]:
        
        url_clicks = defaultdict(int)
        daily_patterns = self._get_cache_by_pattern("analytics:daily:*")
        
        for key, clicks in daily_patterns.items():
            if isinstance(key, str) and 'analytics:daily:' in key:
                short_id = key.split(':')[2] if len(key.split(':')) > 2 else 'unknown'
                url_clicks[short_id] += clicks if isinstance(clicks, int) else 0
        
        
        popular = []
        for short_id, clicks in Counter(url_clicks).most_common(10):
            url_data = self.cache_manager.get(f"short_id:{short_id}")
            if url_data and isinstance(url_data, dict):
                popular.append({
                    'short_id': short_id,
                    'original_url': url_data.get('original_url', 'unknown'),
                    'clicks': clicks,
                    'created_at': url_data.get('created_at')
                })
        
        return popular
    
    def _get_global_countries(self) -> Dict[str, int]:
        
        countries = Counter()
        accesses = self._get_cache_by_pattern("analytics:access:*")
        
        for access_data in accesses.values():
            if isinstance(access_data, dict):
                country = access_data.get('country', 'unknown')
                countries[country] += 1
        
        return dict(countries.most_common(10))
    
    def _get_urls_created_today(self) -> int:
        
        today = datetime.now().strftime('%Y-%m-%d')
        count = 0
        
        url_patterns = self._get_cache_by_pattern("short_id:*")
        for url_data in url_patterns.values():
            if isinstance(url_data, dict):
                created_at = url_data.get('created_at')
                if created_at and created_at.startswith(today):
                    count += 1
        
        return count
    
    def _get_clicks_today(self) -> int:
            
        today = datetime.now().strftime('%Y-%m-%d')
        total = 0
        
        daily_patterns = self._get_cache_by_pattern(f"analytics:daily:*:{today}")
        for clicks in daily_patterns.values():
            if isinstance(clicks, int):
                total += clicks
        
        return total
