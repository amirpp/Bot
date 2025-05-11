"""
ماژول مدیریت پروکسی برای دور زدن محدودیت‌های جغرافیایی در دسترسی به API های ارزهای دیجیتال
"""

import random
import logging
import os
import requests
import json
import time
from typing import Optional, Dict, List, Union

# تنظیم لاگر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# لیست پروکسی‌های پیش‌فرض (این در عمل باید جایگزین شود)
DEFAULT_PROXIES = {
    'general': [
        'http://username:password@proxy1.example.com:8080',
        'http://username:password@proxy2.example.com:8080',
    ],
    'binance': [
        'http://username:password@proxy1.example.com:8080',
    ],
    'kucoin': [
        'http://username:password@proxy2.example.com:8080',
    ]
}

# فعال‌سازی استفاده از پروکسی - پیش‌فرض غیرفعال است
USE_PROXY = False

def get_proxy(service: str = 'general') -> Optional[str]:
    """
    دریافت آدرس پروکسی برای یک سرویس خاص
    
    Args:
        service: نام سرویس (مثلاً 'binance', 'kucoin', یا 'general')
        
    Returns:
        str یا None: آدرس پروکسی یا None اگر استفاده از پروکسی غیرفعال است
    """
    if not USE_PROXY:
        return None
    
    # سعی در دریافت پروکسی‌های اختصاصی سرویس
    proxies = DEFAULT_PROXIES.get(service.lower(), [])
    
    # اگر پروکسی اختصاصی نداشت، از پروکسی‌های عمومی استفاده کن
    if not proxies:
        proxies = DEFAULT_PROXIES.get('general', [])
    
    # اگر هیچ پروکسی‌ای در دسترس نبود
    if not proxies:
        logger.warning(f"هیچ پروکسی‌ای برای سرویس {service} پیدا نشد.")
        return None
    
    # انتخاب تصادفی یک پروکسی
    proxy = random.choice(proxies)
    logger.debug(f"پروکسی {proxy} برای سرویس {service} انتخاب شد.")
    
    return proxy

def setup_proxy_rotation(rotation_interval: int = 300) -> None:
    """
    راه‌اندازی سیستم چرخش خودکار پروکسی‌ها
    
    Args:
        rotation_interval: فاصله زمانی چرخش پروکسی‌ها به ثانیه
    """
    logger.info(f"سیستم چرخش پروکسی هر {rotation_interval} ثانیه فعال شد.")
    
    # کد راه‌اندازی چرخش پروکسی در اینجا قرار می‌گیرد
    # این کد می‌تواند یک ترد جداگانه راه‌اندازی کند که به صورت دوره‌ای پروکسی‌ها را عوض می‌کند
    
    pass

def check_proxy_connectivity(proxy: str) -> bool:
    """
    بررسی اتصال به پروکسی
    
    Args:
        proxy: آدرس پروکسی
        
    Returns:
        bool: وضعیت اتصال
    """
    try:
        response = requests.get('https://api.ipify.org', proxies={'http': proxy, 'https': proxy}, timeout=5)
        if response.status_code == 200:
            logger.info(f"اتصال به پروکسی {proxy} موفق بود. IP: {response.text}")
            return True
        else:
            logger.warning(f"اتصال به پروکسی {proxy} ناموفق بود. کد وضعیت: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"خطا در اتصال به پروکسی {proxy}: {str(e)}")
        return False

def configure_proxy_from_env() -> None:
    """
    تنظیم پروکسی از متغیرهای محیطی
    """
    global USE_PROXY, DEFAULT_PROXIES
    
    # بررسی متغیر محیطی فعال‌سازی پروکسی
    use_proxy_env = os.environ.get('USE_PROXY', 'False')
    USE_PROXY = use_proxy_env.lower() in ('true', '1', 'yes')
    
    # بررسی متغیرهای محیطی پروکسی
    proxy_config_json = os.environ.get('PROXY_CONFIG')
    if proxy_config_json:
        try:
            proxy_config = json.loads(proxy_config_json)
            DEFAULT_PROXIES = proxy_config
            logger.info("تنظیمات پروکسی از متغیرهای محیطی بارگذاری شد.")
        except Exception as e:
            logger.error(f"خطا در بارگذاری تنظیمات پروکسی از متغیرهای محیطی: {str(e)}")

def get_proxy_status() -> Dict[str, bool]:
    """
    بررسی وضعیت اتصال همه پروکسی‌ها
    
    Returns:
        dict: وضعیت اتصال هر پروکسی
    """
    results = {}
    
    # جمع‌آوری همه پروکسی‌ها
    all_proxies = set()
    for proxies in DEFAULT_PROXIES.values():
        all_proxies.update(proxies)
    
    # بررسی اتصال هر پروکسی
    for proxy in all_proxies:
        results[proxy] = check_proxy_connectivity(proxy)
    
    return results

def create_tunneled_connection(service: str) -> Dict:
    """
    ایجاد یک اتصال تونل‌زده برای دسترسی به API
    
    Args:
        service: نام سرویس (مثلاً 'binance', 'kucoin')
        
    Returns:
        dict: اطلاعات اتصال
    """
    proxy = get_proxy(service)
    
    connection_info = {
        'proxy': proxy,
        'use_proxy': proxy is not None,
        'timestamp': time.time()
    }
    
    return connection_info

# تنظیم اولیه از متغیرهای محیطی
configure_proxy_from_env()