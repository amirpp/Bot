"""
ماژول جستجوی ارزهای دیجیتال و مدیریت داده‌های آنها

این ماژول امکان جستجو و دریافت اطلاعات ارزهای دیجیتال مختلف را فراهم می‌کند
و قابلیت ذخیره‌سازی ارزهای محبوب را دارد.
"""

import pandas as pd
import numpy as np
import ccxt
import json
import os
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import requests
from datetime import datetime

# تنظیم لاگر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoSearch:
    """کلاس جستجوی ارزهای دیجیتال"""
    
    def __init__(self, cache_dir: str = './crypto_cache'):
        """
        مقداردهی اولیه جستجوگر ارزها
        
        Args:
            cache_dir (str): مسیر ذخیره کش داده‌ها
        """
        self.cache_dir = cache_dir
        
        # ایجاد دایرکتوری کش اگر وجود ندارد
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # فایل‌های کش
        self.symbols_cache_file = os.path.join(cache_dir, 'symbols_cache.json')
        self.favorites_file = os.path.join(cache_dir, 'favorite_cryptos.json')
        
        # لیست صرافی‌های پشتیبانی شده
        self.supported_exchanges = ['binance', 'kucoin', 'huobi', 'bybit', 'kraken', 'coinbase']
        
        # لیست ارزهای محبوب
        self.favorite_cryptos = self._load_favorites()
        
        # کش نمادها
        self.symbols_cache = self._load_symbols_cache()
        
        # بررسی و به‌روزرسانی کش در صورت نیاز
        self._update_cache_if_needed()
    
    def _load_symbols_cache(self) -> Dict[str, Any]:
        """
        بارگذاری کش نمادها از فایل
        
        Returns:
            dict: کش نمادها
        """
        if os.path.exists(self.symbols_cache_file):
            try:
                with open(self.symbols_cache_file, 'r') as f:
                    cache = json.load(f)
                
                # بررسی صحت ساختار کش
                if not isinstance(cache, dict) or 'timestamp' not in cache or 'data' not in cache:
                    logger.warning("فایل کش نمادها معتبر نیست. کش جدید ایجاد می‌شود.")
                    return {'timestamp': 0, 'data': {}}
                
                return cache
            except Exception as e:
                logger.error(f"خطا در بارگذاری کش نمادها: {str(e)}")
        
        return {'timestamp': 0, 'data': {}}
    
    def _save_symbols_cache(self) -> None:
        """ذخیره کش نمادها در فایل"""
        try:
            with open(self.symbols_cache_file, 'w') as f:
                json.dump(self.symbols_cache, f)
        except Exception as e:
            logger.error(f"خطا در ذخیره کش نمادها: {str(e)}")
    
    def _update_cache_if_needed(self) -> None:
        """به‌روزرسانی کش نمادها در صورت قدیمی بودن"""
        # بررسی زمان آخرین به‌روزرسانی کش
        current_time = time.time()
        cache_age = current_time - self.symbols_cache.get('timestamp', 0)
        
        # به‌روزرسانی کش اگر بیش از 24 ساعت گذشته باشد
        if cache_age > 86400:  # 24 ساعت
            logger.info("کش نمادها قدیمی است. در حال به‌روزرسانی...")
            self._update_symbols_cache()
    
    def _update_symbols_cache(self) -> None:
        """به‌روزرسانی کش نمادها با دریافت اطلاعات جدید از صرافی‌ها"""
        all_symbols = {}
        
        # دریافت نمادها از هر صرافی
        for exchange_id in self.supported_exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({'enableRateLimit': True})
                
                markets = exchange.load_markets()
                
                # استخراج نمادهای معتبر
                exchange_symbols = {}
                for symbol, market_data in markets.items():
                    # فیلتر کردن نمادهای معتبر (با USDT و یا USD)
                    if symbol.endswith('/USDT') or symbol.endswith('/USD'):
                        base_currency = symbol.split('/')[0]
                        quote_currency = symbol.split('/')[1]
                        
                        exchange_symbols[symbol] = {
                            'base': base_currency,
                            'quote': quote_currency,
                            'precision': market_data.get('precision', {}),
                            'limits': market_data.get('limits', {}),
                            'active': market_data.get('active', True)
                        }
                
                all_symbols[exchange_id] = exchange_symbols
                logger.info(f"{len(exchange_symbols)} نماد از صرافی {exchange_id} دریافت شد.")
                
            except Exception as e:
                logger.error(f"خطا در دریافت نمادها از صرافی {exchange_id}: {str(e)}")
        
        # به‌روزرسانی کش
        self.symbols_cache = {
            'timestamp': time.time(),
            'data': all_symbols
        }
        
        # ذخیره کش جدید
        self._save_symbols_cache()
        logger.info("کش نمادها با موفقیت به‌روزرسانی شد.")
    
    def _load_favorites(self) -> List[str]:
        """
        بارگذاری لیست ارزهای محبوب
        
        Returns:
            list: لیست ارزهای محبوب
        """
        if os.path.exists(self.favorites_file):
            try:
                with open(self.favorites_file, 'r') as f:
                    favorites = json.load(f)
                
                return favorites
            except Exception as e:
                logger.error(f"خطا در بارگذاری ارزهای محبوب: {str(e)}")
        
        # لیست پیش‌فرض ارزهای محبوب
        default_favorites = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
            'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT', 'DOT/USDT', 'DOGE/USDT'
        ]
        
        # ذخیره لیست پیش‌فرض
        self._save_favorites(default_favorites)
        
        return default_favorites
    
    def _save_favorites(self, favorites: List[str]) -> None:
        """
        ذخیره لیست ارزهای محبوب
        
        Args:
            favorites (list): لیست ارزهای محبوب
        """
        try:
            with open(self.favorites_file, 'w') as f:
                json.dump(favorites, f)
        except Exception as e:
            logger.error(f"خطا در ذخیره ارزهای محبوب: {str(e)}")
    
    def search_crypto(self, query: str, exchange: str = 'all') -> List[str]:
        """
        جستجوی ارزهای دیجیتال بر اساس کلمه کلیدی
        
        Args:
            query (str): کلمه کلیدی جستجو
            exchange (str): نام صرافی یا 'all' برای جستجو در همه صرافی‌ها
            
        Returns:
            list: لیست نمادهای منطبق
        """
        query = query.upper()
        results = []
        
        # جستجو در کش نمادها
        symbols_data = self.symbols_cache.get('data', {})
        
        # تعیین صرافی‌های مورد جستجو
        exchanges_to_search = self.supported_exchanges if exchange == 'all' else [exchange]
        
        for exchange_id in exchanges_to_search:
            if exchange_id in symbols_data:
                exchange_symbols = symbols_data[exchange_id]
                
                for symbol in exchange_symbols:
                    # جستجو در نام کامل نماد
                    if query in symbol:
                        if symbol not in results:
                            results.append(symbol)
                    
                    # جستجو در ارز پایه
                    base_currency = exchange_symbols[symbol].get('base', '')
                    if query in base_currency and symbol not in results:
                        results.append(symbol)
        
        return results
    
    def get_all_cryptos(self, exchange: str = 'binance', quote_currency: str = 'USDT') -> List[str]:
        """
        دریافت لیست تمام ارزهای موجود در یک صرافی
        
        Args:
            exchange (str): نام صرافی
            quote_currency (str): ارز پایه (مثل 'USDT')
            
        Returns:
            list: لیست نمادها
        """
        symbols_data = self.symbols_cache.get('data', {})
        
        if exchange not in symbols_data:
            logger.warning(f"اطلاعات صرافی {exchange} در کش موجود نیست.")
            return []
        
        exchange_symbols = symbols_data[exchange]
        
        # فیلتر کردن نمادها بر اساس ارز پایه
        filtered_symbols = [
            symbol for symbol in exchange_symbols
            if symbol.endswith(f'/{quote_currency}') and exchange_symbols[symbol].get('active', True)
        ]
        
        return filtered_symbols
    
    def get_crypto_info(self, symbol: str, exchange: str = 'binance') -> Dict[str, Any]:
        """
        دریافت اطلاعات یک ارز
        
        Args:
            symbol (str): نماد ارز
            exchange (str): نام صرافی
            
        Returns:
            dict: اطلاعات ارز
        """
        symbols_data = self.symbols_cache.get('data', {})
        
        if exchange not in symbols_data:
            logger.warning(f"اطلاعات صرافی {exchange} در کش موجود نیست.")
            return {}
        
        exchange_symbols = symbols_data[exchange]
        
        if symbol in exchange_symbols:
            return exchange_symbols[symbol]
        
        logger.warning(f"نماد {symbol} در صرافی {exchange} یافت نشد.")
        return {}
    
    def add_to_favorites(self, symbol: str) -> bool:
        """
        اضافه کردن ارز به لیست علاقه‌مندی‌ها
        
        Args:
            symbol (str): نماد ارز
            
        Returns:
            bool: نتیجه اضافه کردن
        """
        if symbol not in self.favorite_cryptos:
            self.favorite_cryptos.append(symbol)
            self._save_favorites(self.favorite_cryptos)
            logger.info(f"ارز {symbol} به لیست علاقه‌مندی‌ها اضافه شد.")
            return True
        
        logger.info(f"ارز {symbol} قبلاً در لیست علاقه‌مندی‌ها وجود دارد.")
        return False
    
    def remove_from_favorites(self, symbol: str) -> bool:
        """
        حذف ارز از لیست علاقه‌مندی‌ها
        
        Args:
            symbol (str): نماد ارز
            
        Returns:
            bool: نتیجه حذف
        """
        if symbol in self.favorite_cryptos:
            self.favorite_cryptos.remove(symbol)
            self._save_favorites(self.favorite_cryptos)
            logger.info(f"ارز {symbol} از لیست علاقه‌مندی‌ها حذف شد.")
            return True
        
        logger.info(f"ارز {symbol} در لیست علاقه‌مندی‌ها وجود ندارد.")
        return False
    
    def get_favorites(self) -> List[str]:
        """
        دریافت لیست ارزهای محبوب
        
        Returns:
            list: لیست ارزهای محبوب
        """
        return self.favorite_cryptos
    
    def get_top_traded_cryptos(self, limit: int = 20, exchange: str = 'binance') -> List[str]:
        """
        دریافت لیست ارزهای پرمعامله
        
        Args:
            limit (int): تعداد ارزها
            exchange (str): نام صرافی
            
        Returns:
            list: لیست ارزهای پرمعامله
        """
        # بررسی کش موجود
        cache_file = os.path.join(self.cache_dir, f'top_traded_{exchange}.json')
        
        # بررسی آیا کش معتبر است (کمتر از 24 ساعت)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # بررسی زمان کش
                cache_time = cache_data.get('timestamp', 0)
                if (time.time() - cache_time) < 86400:  # 24 ساعت
                    return cache_data.get('symbols', [])[:limit]
            except Exception as e:
                logger.warning(f"خطا در خواندن کش ارزهای پرمعامله: {str(e)}")

        # تلاش برای بروزرسانی داده‌ها در تمام صرافی‌ها
        for exchange_name in self.supported_exchanges:
            if exchange_name != exchange:  # صرافی‌های دیگر را امتحان کن
                try:
                    # تلاش برای دریافت داده‌های واقعی
                    exchange_class = getattr(ccxt, exchange_name)
                    exchange_obj = exchange_class({
                        'enableRateLimit': True,
                        'timeout': 30000,  # افزایش زمان انتظار
                        'options': {'defaultType': 'spot'},
                    })
                    
                    tickers = exchange_obj.fetch_tickers()
                    
                    # محاسبه حجم معاملات و مرتب‌سازی
                    volumes = {}
                    for symbol, ticker in tickers.items():
                        if '/USDT' in symbol:
                            volumes[symbol] = ticker.get('quoteVolume', 0)
                    
                    # مرتب‌سازی بر اساس حجم معاملات
                    sorted_symbols = sorted(volumes.keys(), key=lambda s: volumes[s], reverse=True)
                    
                    # ذخیره در کش
                    cache_data = {
                        'timestamp': time.time(),
                        'symbols': sorted_symbols
                    }
                    
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(cache_data, f)
                    except Exception as e:
                        logger.warning(f"خطا در ذخیره کش ارزهای پرمعامله: {str(e)}")
                    
                    return sorted_symbols[:limit]
                    
                except Exception as e:
                    logger.warning(f"خطا در دریافت ارزهای پرمعامله از {exchange_name}: {str(e)}")
                    continue  # امتحان صرافی بعدی
        
        # اگر هیچ صرافی موفق نبود، از کش قدیمی استفاده کن
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    return cache_data.get('symbols', [])[:limit]
            except Exception:
                pass
        
        # برگرداندن لیست پیش‌فرض
            default_top = [
                'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'ADA/USDT',
                'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'SHIB/USDT', 'MATIC/USDT',
                'LINK/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT',
                'XMR/USDT', 'XLM/USDT', 'NEAR/USDT', 'ALGO/USDT', 'FIL/USDT'
            ]
            return default_top[:limit]
    
    def get_trending_cryptos(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        دریافت لیست ارزهای پرطرفدار و روند
        
        Args:
            limit (int): تعداد ارزها
            
        Returns:
            list: لیست ارزهای پرطرفدار
        """
        # بررسی کش موجود
        cache_file = os.path.join(self.cache_dir, 'trending_cryptos.json')
        
        # بررسی آیا کش معتبر است (کمتر از 6 ساعت)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # بررسی زمان کش
                cache_time = cache_data.get('timestamp', 0)
                if (time.time() - cache_time) < 21600:  # 6 ساعت
                    return cache_data.get('coins', [])[:limit]
            except Exception as e:
                logger.warning(f"خطا در خواندن کش ارزهای پرطرفدار: {str(e)}")
                
        try:
            # تلاش برای دریافت داده‌های واقعی از CoinGecko API
            response = requests.get(
                'https://api.coingecko.com/api/v3/search/trending',
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'
                },
                timeout=15
            )
            
            if response.status_code == 200:
                trending_data = response.json()
                trending_coins = trending_data.get('coins', [])
                
                result = []
                for coin in trending_coins[:limit]:
                    item = coin.get('item', {})
                    result.append({
                        'id': item.get('id', ''),
                        'name': item.get('name', ''),
                        'symbol': item.get('symbol', ''),
                        'market_cap_rank': item.get('market_cap_rank', 0),
                        'price_btc': item.get('price_btc', 0),
                        'score': item.get('score', 0),
                        'slug': item.get('slug', '')
                    })
                
                # ذخیره در کش
                cache_data = {
                    'timestamp': time.time(),
                    'coins': result
                }
                
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f)
                except Exception as e:
                    logger.warning(f"خطا در ذخیره کش ارزهای پرطرفدار: {str(e)}")
                
                return result
                
        except Exception as e:
            logger.error(f"خطا در دریافت ارزهای پرطرفدار: {str(e)}")
            
            # بررسی کش قدیمی
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                        return cache_data.get('coins', [])[:limit]
                except Exception:
                    pass
        
        # برگرداندن لیست پیش‌فرض
        default_trending = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'score': 0},
            {'symbol': 'ETH', 'name': 'Ethereum', 'score': 0},
            {'symbol': 'SOL', 'name': 'Solana', 'score': 0},
            {'symbol': 'XRP', 'name': 'Ripple', 'score': 0},
            {'symbol': 'ADA', 'name': 'Cardano', 'score': 0},
            {'symbol': 'AVAX', 'name': 'Avalanche', 'score': 0},
            {'symbol': 'DOGE', 'name': 'Dogecoin', 'score': 0},
            {'symbol': 'SHIB', 'name': 'Shiba Inu', 'score': 0},
            {'symbol': 'MATIC', 'name': 'Polygon', 'score': 0},
            {'symbol': 'DOT', 'name': 'Polkadot', 'score': 0}
        ]
        
        return default_trending[:limit]
    
    def force_update_cache(self) -> bool:
        """
        به‌روزرسانی اجباری کش نمادها
        
        Returns:
            bool: نتیجه به‌روزرسانی
        """
        try:
            self._update_symbols_cache()
            return True
        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی اجباری کش: {str(e)}")
            return False
    
    def get_crypto_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        دریافت متادیتای یک ارز (اطلاعات اضافی)
        
        Args:
            symbol (str): نماد ارز (مثال: 'BTC/USDT')
            
        Returns:
            dict: متادیتای ارز
        """
        # استخراج نام اصلی ارز
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol
        
        try:
            # تلاش برای دریافت اطلاعات از CoinGecko API
            response = requests.get(
                f'https://api.coingecko.com/api/v3/coins/{base_currency.lower()}',
                headers={'Accept': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                coin_data = response.json()
                
                metadata = {
                    'name': coin_data.get('name', ''),
                    'symbol': coin_data.get('symbol', '').upper(),
                    'description': coin_data.get('description', {}).get('en', ''),
                    'homepage': coin_data.get('links', {}).get('homepage', [''])[0],
                    'github': coin_data.get('links', {}).get('repos_url', {}).get('github', []),
                    'market_data': {
                        'market_cap': coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                        'total_volume': coin_data.get('market_data', {}).get('total_volume', {}).get('usd', 0),
                        'high_24h': coin_data.get('market_data', {}).get('high_24h', {}).get('usd', 0),
                        'low_24h': coin_data.get('market_data', {}).get('low_24h', {}).get('usd', 0),
                        'price_change_percentage_24h': coin_data.get('market_data', {}).get('price_change_percentage_24h', 0),
                        'price_change_percentage_7d': coin_data.get('market_data', {}).get('price_change_percentage_7d', 0)
                    },
                    'image': coin_data.get('image', {}).get('large', '')
                }
                
                return metadata
                
        except Exception as e:
            logger.error(f"خطا در دریافت متادیتای ارز {symbol}: {str(e)}")
        
        # برگرداندن اطلاعات پیش‌فرض
        return {
            'name': base_currency,
            'symbol': base_currency,
            'description': 'اطلاعات تکمیلی موجود نیست.',
            'image': '',
            'market_data': {
                'market_cap': 0,
                'total_volume': 0,
                'price_change_percentage_24h': 0
            }
        }
    
    def get_exchange_status(self, exchange: str) -> Dict[str, Any]:
        """
        دریافت وضعیت یک صرافی
        
        Args:
            exchange (str): نام صرافی
            
        Returns:
            dict: وضعیت صرافی
        """
        try:
            exchange_class = getattr(ccxt, exchange)
            exchange_obj = exchange_class({'enableRateLimit': True})
            
            status = exchange_obj.fetch_status()
            
            return {
                'name': exchange,
                'status': status.get('status', 'unknown'),
                'updated': status.get('updated', 0),
                'eta': status.get('eta', None),
                'url': status.get('url', None)
            }
            
        except Exception as e:
            logger.error(f"خطا در دریافت وضعیت صرافی {exchange}: {str(e)}")
            
            return {
                'name': exchange,
                'status': 'unknown',
                'error': str(e)
            }


# ایجاد نمونه سراسری
_crypto_search_instance = None

def get_crypto_search() -> CryptoSearch:
    """
    دریافت نمونه سراسری از کلاس CryptoSearch
    
    Returns:
        CryptoSearch: نمونه سراسری
    """
    global _crypto_search_instance
    
    if _crypto_search_instance is None:
        _crypto_search_instance = CryptoSearch()
    
    return _crypto_search_instance