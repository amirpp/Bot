"""
ماژول خدمات API برای دسترسی به داده‌های بازار ارز دیجیتال

این ماژول توابع مورد نیاز برای دریافت داده‌های قیمت، حجم معاملات، و سایر اطلاعات بازار
از صرافی‌های مختلف ارز دیجیتال را فراهم می‌کند.
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import ccxt
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import proxy_service
from proxy_service import (
    get_proxy, check_proxy_connectivity, create_tunneled_connection,
    find_working_binance_proxy, setup_socket_proxy_for_ccxt, test_binance_with_proxy
)

# Configure ccxt to use alternative exchanges due to geographical restrictions
def configure_alternative_exchanges():
    """Configure alternative exchanges for regions with restrictions"""
    global DEFAULT_EXCHANGE
    
    # Try to detect if we're in a restricted region
    try:
        # Use a non-restricted API to check our IP region
        response = requests.get('https://api.coingecko.com/api/v3/ping', timeout=5)
        if response.status_code == 200:
            logger.info("Successfully connected to CoinGecko API")
    except Exception as e:
        logger.warning(f"Could not connect to CoinGecko API: {str(e)}")
    
    # تست دسترسی به بایننس با استفاده از پروکسی
    binance_proxy = find_working_binance_proxy()
    
    if binance_proxy:
        logger.info(f"پروکسی کارآمد برای بایننس یافت شد: {binance_proxy}")
        # همچنان از بایننس به عنوان صرافی پیش‌فرض استفاده می‌کنیم
        DEFAULT_EXCHANGE = 'binance'
        return
    
    # اگر به اینجا برسیم، یعنی محدودیت جغرافیایی داریم و پروکسی کارآمد نیافتیم
    logger.info("در حال تلاش برای استفاده از کوکوین به عنوان جایگزین...")
    DEFAULT_EXCHANGE = 'kucoin'
    
    # تست اتصال به کوکوین
    try:
        kucoin_proxy = get_proxy('kucoin')
        proxies = None
        if kucoin_proxy:
            proxies = {'http': kucoin_proxy, 'https': kucoin_proxy}
        
        response = requests.get('https://api.kucoin.com/api/v1/timestamp', proxies=proxies, timeout=5)
        if response.status_code == 200:
            logger.info("نمونه صرافی kucoin با موفقیت ایجاد شد")
        else:
            logger.warning(f"اتصال به کوکوین ناموفق بود: {response.status_code}")
            # می‌توان از سایر صرافی‌های پشتیبان استفاده کرد
    except Exception as e:
        logger.error(f"خطا در اتصال به کوکوین: {str(e)}")

def configure_proxy_from_env():
    """Configure proxy settings from environment variables"""
    global USE_PROXY
    
    # بررسی تنظیمات پروکسی از متغیرهای محیطی
    proxy_url = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
    if proxy_url:
        USE_PROXY = True
        logger.info(f"استفاده از پروکسی از متغیرهای محیطی: {proxy_url}")
        return True
    
    # بررسی اتصال مستقیم به بایننس
    try:
        direct_response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
        if direct_response.status_code == 200:
            logger.info("اتصال به API بایننس بدون پروکسی موفق بود")
            USE_PROXY = False
            return False
    except Exception as e:
        logger.warning(f"اتصال مستقیم به API بایننس ناموفق بود: {str(e)}")
    
    # اگر به اینجا برسیم، نیاز به پروکسی داریم
    USE_PROXY = True
    
    # تست سیستم پروکسی
    working_proxy = find_working_binance_proxy()
    if working_proxy:
        logger.info(f"سیستم پروکسی با موفقیت با پروکسی {working_proxy} تنظیم شد")
        return True
    
    # تست پروکسی SOCKS برای CCXT
    socks_config = setup_socket_proxy_for_ccxt()
    if socks_config:
        logger.info("پروکسی SOCKS برای CCXT با موفقیت تنظیم شد")
        return True
    
    logger.warning("هیچ پروکسی کارآمدی یافت نشد. ممکن است مشکل در دسترسی به برخی صرافی‌ها وجود داشته باشد.")
    return False

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تنظیمات پایه
DEFAULT_TIMEFRAME = '1d'  # تایم‌فریم پیش‌فرض
API_RATE_LIMIT = 1.0  # حداقل فاصله بین درخواست‌ها (ثانیه)
USE_PROXY = os.environ.get('USE_PROXY', 'False').lower() in ('true', '1', 'yes')
MAX_RETRIES = 3  # حداکثر تعداد تلاش‌ها در صورت شکست
DEFAULT_FETCH_LIMIT = 500  # تعداد پیش‌فرض کندل‌ها برای دریافت
DEFAULT_EXCHANGE = 'kucoin'  # صرافی پیش‌فرض (بایننس در برخی مناطق محدود است)

# لیست صرافی‌های پشتیبانی شده
SUPPORTED_EXCHANGES = ['binance', 'kucoin', 'bybit', 'coinbasepro', 'kraken', 'huobi']

# نگاشت تایم‌فریم‌ها بین حالت‌های مختلف
TIMEFRAME_MAP = {
    # حالت نمایشی به CCXT
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m', 
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
    '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M',
    # حالت‌های فارسی به CCXT
    'یک دقیقه': '1m', 'سه دقیقه': '3m', 'پنج دقیقه': '5m', 'پانزده دقیقه': '15m', 'سی دقیقه': '30m',
    'یک ساعت': '1h', 'دو ساعت': '2h', 'چهار ساعت': '4h', 'شش ساعت': '6h', 'هشت ساعت': '8h', 'دوازده ساعت': '12h',
    'یک روز': '1d', 'سه روز': '3d', 'یک هفته': '1w', 'یک ماه': '1M'
}

# تنظیم صرافی‌های CCXT
exchanges = {}

def normalize_symbol(symbol: str) -> str:
    """
    استاندارد‌سازی نماد ارز
    
    Args:
        symbol (str): نماد ارز (مثلاً "BTC/USDT" یا "BTCUSDT" یا "BTC")
        
    Returns:
        str: نماد استاندارد شده (مثلاً "BTC/USDT")
    """
    # پاکسازی و تبدیل به حروف بزرگ
    symbol = symbol.strip().upper()
    
    # اگر جفت ارز کامل نیست، USDT را به عنوان ارز پایه فرض می‌کنیم
    if '/' not in symbol:
        if 'USDT' in symbol:
            # اگر USDT در انتهای نماد باشد
            coin = symbol.replace('USDT', '')
            return f"{coin}/USDT"
        else:
            return f"{symbol}/USDT"
    
    return symbol

def get_exchange_instance(exchange_id: str) -> ccxt.Exchange:
    """
    دریافت نمونه صرافی CCXT
    
    Args:
        exchange_id (str): شناسه صرافی
        
    Returns:
        ccxt.Exchange: نمونه صرافی
    """
    global exchanges
    
    exchange_id = exchange_id.lower()
    
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(f"صرافی {exchange_id} پشتیبانی نمی‌شود. صرافی‌های پشتیبانی شده: {', '.join(SUPPORTED_EXCHANGES)}")
    
    if exchange_id not in exchanges:
        try:
            # تنظیمات پایه
            exchange_class = getattr(ccxt, exchange_id)
            
            # تنظیمات اختیاری
            settings = {
                'enableRateLimit': True,  # فعال‌سازی محدودیت نرخ درخواست‌ها
            }
            
            # اضافه کردن پروکسی با استراتژی بهبود یافته
            if USE_PROXY:
                # برای بایننس، ابتدا یک پروکسی کارآمد بررسی کنیم
                if exchange_id == 'binance':
                    working_proxy = find_working_binance_proxy()
                    if working_proxy:
                        settings['proxies'] = {
                            'http': working_proxy,
                            'https': working_proxy
                        }
                        logger.info(f"استفاده از پروکسی کارآمد برای بایننس: {working_proxy}")
                # برای سایر صرافی‌ها
                else:
                    proxy = get_proxy(exchange_id)
                    if proxy:
                        settings['proxies'] = {
                            'http': proxy,
                            'https': proxy
                        }
                        
                # تلاش برای استفاده از پروکسی SOCKS که بیشتر کارآمد هستند
                if exchange_id in ['binance', 'kucoin']:
                    socks_settings = setup_socket_proxy_for_ccxt()
                    if socks_settings:
                        settings.update(socks_settings)
            
            # افزودن کلیدهای API اگر در متغیرهای محیطی تنظیم شده‌اند
            api_key = os.environ.get(f'{exchange_id.upper()}_API_KEY')
            api_secret = os.environ.get(f'{exchange_id.upper()}_API_SECRET')
            if api_key and api_secret:
                settings['apiKey'] = api_key
                settings['secret'] = api_secret
            
            # ساخت نمونه صرافی
            exchange = exchange_class(settings)
            exchanges[exchange_id] = exchange
            
            # لود کردن بازارها
            exchange.load_markets()
            
            logger.info(f"نمونه صرافی {exchange_id} با موفقیت ایجاد شد")
            
        except Exception as e:
            logger.error(f"خطا در ایجاد نمونه صرافی {exchange_id}: {str(e)}")
            # اگر بایننس با خطا مواجه شد، کوکوین را امتحان کن
            if exchange_id == 'binance':
                logger.info("در حال تلاش برای استفاده از کوکوین به عنوان جایگزین...")
                try:
                    return get_exchange_instance('kucoin')
                except Exception as inner_e:
                    logger.error(f"خطا در ایجاد نمونه جایگزین کوکوین: {str(inner_e)}")
            raise
    
    return exchanges[exchange_id]

def get_ohlcv_data(
    symbol: str, 
    timeframe: str = DEFAULT_TIMEFRAME, 
    limit: int = DEFAULT_FETCH_LIMIT, 
    exchange_id: str = DEFAULT_EXCHANGE
) -> pd.DataFrame:
    """
    دریافت داده‌های OHLCV (قیمت باز، بالا، پایین، بسته و حجم) از صرافی
    
    Args:
        symbol (str): نماد ارز (مثلاً "BTC/USDT")
        timeframe (str): تایم‌فریم (مثلاً "1d", "4h", "1h")
        limit (int): تعداد کندل‌ها
        exchange_id (str): شناسه صرافی
        
    Returns:
        pd.DataFrame: دیتافریم داده‌های OHLCV
    """
    # استاندارد‌سازی نماد و تایم‌فریم
    symbol = normalize_symbol(symbol)
    timeframe = TIMEFRAME_MAP.get(timeframe, timeframe)
    
    # لیست صرافی‌ها برای امتحان کردن به ترتیب اولویت (بهبود یافته)
    exchanges_to_try = ['binance', 'kucoin', 'bybit', 'okx', 'gate', 'mexc', 'coinbasepro', 'kraken', 'huobi']
    
    # اگر صرافی درخواستی در لیست نیست، آن را در ابتدا اضافه می‌کنیم
    if exchange_id not in exchanges_to_try:
        exchanges_to_try.insert(0, exchange_id)
    else:
        # اطمینان از اینکه صرافی درخواستی در اولویت اول است
        exchanges_to_try.remove(exchange_id)
        exchanges_to_try.insert(0, exchange_id)
        
    # اگر صرافی بایننس درخواست شده و پروکسی فعال است، ابتدا یک پروکسی کارآمد برای بایننس پیدا کنیم
    if exchange_id == 'binance' and USE_PROXY:
        working_proxy = find_working_binance_proxy()
        if not working_proxy:
            logger.warning("پروکسی کارآمد برای بایننس یافت نشد. استفاده از کوکوین به عنوان اولویت اول.")
            # قرار دادن کوکوین در اولویت اول
            if 'binance' in exchanges_to_try:
                exchanges_to_try.remove('binance')
            if 'kucoin' in exchanges_to_try:
                exchanges_to_try.remove('kucoin')
            exchanges_to_try.insert(0, 'kucoin')
    
    last_error = None
    
    # بررسی هر صرافی به ترتیب
    for current_exchange in exchanges_to_try:
        try:
            logger.info(f"تلاش برای دریافت داده‌های {symbol} از صرافی {current_exchange}...")
            
            # دریافت نمونه صرافی
            exchange = get_exchange_instance(current_exchange)
            
            # بررسی پشتیبانی از تایم‌فریم
            if timeframe not in exchange.timeframes:
                logger.warning(f"تایم‌فریم {timeframe} توسط {current_exchange} پشتیبانی نمی‌شود. استفاده از تایم‌فریم پیش‌فرض {DEFAULT_TIMEFRAME}")
                timeframe = DEFAULT_TIMEFRAME
            
            # تلاش‌های مجدد در صورت خطا
            for attempt in range(MAX_RETRIES):
                try:
                    # دریافت داده‌ها
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    
                    # بررسی داده‌های دریافتی
                    if not ohlcv or len(ohlcv) == 0:
                        logger.warning(f"داده‌ای برای {symbol} از {current_exchange} دریافت نشد")
                        break  # امتحان صرافی بعدی
                    
                    # تبدیل به دیتافریم pandas
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # تبدیل timestamp به datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # تبدیل نوع داده‌های عددی
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # اضافه کردن اطلاعات صرافی
                    df.attrs['exchange'] = current_exchange
                    df.attrs['symbol'] = symbol
                    df.attrs['timeframe'] = timeframe
                    
                    logger.info(f"داده‌های {symbol} با موفقیت از {current_exchange} دریافت شد (تعداد: {len(df)})")
                    return df
                    
                except ccxt.NetworkError as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"خطای شبکه در دریافت داده‌ها از {current_exchange} (تلاش {attempt+1}/{MAX_RETRIES}): {str(e)}")
                        time.sleep(2 ** attempt)  # انتظار نمایی
                    else:
                        last_error = str(e)
                        logger.error(f"خطای شبکه پس از {MAX_RETRIES} تلاش از {current_exchange}: {str(e)}")
                        break  # امتحان صرافی بعدی
                        
                except ccxt.ExchangeError as e:
                    last_error = str(e)
                    logger.error(f"خطای صرافی {current_exchange} در دریافت داده‌ها: {str(e)}")
                    break  # امتحان صرافی بعدی
                    
        except Exception as e:
            last_error = str(e)
            logger.error(f"خطا در استفاده از صرافی {current_exchange} برای {symbol}: {str(e)}")
            continue  # امتحان صرافی بعدی
    
    # اگر هیچ صرافی موفق نبود
    error_msg = f"همه تلاش‌ها برای دریافت داده‌های {symbol} ناموفق بود"
    if last_error:
        error_msg += f". آخرین خطا: {last_error}"
    
    logger.error(error_msg)
    
    # بازگشت دیتافریم خالی با پیام خطا
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    empty_df.attrs['error'] = error_msg
    return empty_df

def get_ohlcv_data_multi_source(
    symbol: str, 
    timeframe: str = DEFAULT_TIMEFRAME, 
    limit: int = DEFAULT_FETCH_LIMIT,
    exchanges: List[str] = None
) -> pd.DataFrame:
    """
    دریافت داده‌های OHLCV از چندین منبع و ترکیب آنها
    
    Args:
        symbol (str): نماد ارز (مثلاً "BTC/USDT")
        timeframe (str): تایم‌فریم (مثلاً "1d", "4h", "1h")
        limit (int): تعداد کندل‌ها
        exchanges (List[str]): لیست صرافی‌ها
        
    Returns:
        pd.DataFrame: دیتافریم ترکیبی داده‌های OHLCV
    """
    # اگر لیست صرافی‌ها مشخص نشده باشد، از صرافی‌های پیش‌فرض استفاده می‌کنیم
    if not exchanges:
        exchanges = ['binance', 'kucoin', 'bybit']
    
    dfs = []
    
    # دریافت داده‌ها از هر صرافی
    for exchange_id in exchanges:
        try:
            df = get_ohlcv_data(symbol, timeframe, limit, exchange_id)
            if not df.empty:
                df['source'] = exchange_id
                dfs.append(df)
                logger.info(f"داده‌های {symbol} از {exchange_id} دریافت شد: {len(df)} کندل")
                # در صورت موفقیت اولین صرافی، دیگر نیازی به بقیه نیست
                break
        except Exception as e:
            logger.warning(f"خطا در دریافت داده‌های {symbol} از {exchange_id}: {str(e)}")
    
    # اگر هیچ داده‌ای دریافت نشد
    if not dfs:
        logger.error(f"هیچ داده‌ای برای {symbol} از هیچ صرافی دریافت نشد")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # ترکیب داده‌ها (فقط از اولین منبع موفق)
    result_df = dfs[0].copy()
    return result_df

def get_current_price(symbol: str, exchange_id: str = DEFAULT_EXCHANGE) -> float:
    """
    دریافت قیمت فعلی ارز
    
    Args:
        symbol (str): نماد ارز (مثلاً "BTC/USDT")
        exchange_id (str): شناسه صرافی
        
    Returns:
        float: قیمت فعلی
    """
    # استاندارد‌سازی نماد
    symbol = normalize_symbol(symbol)
    
    try:
        # دریافت نمونه صرافی
        exchange = get_exchange_instance(exchange_id)
        
        # دریافت تیکر
        ticker = exchange.fetch_ticker(symbol)
        
        return ticker['last']
        
    except Exception as e:
        logger.error(f"خطا در دریافت قیمت فعلی {symbol} از {exchange_id}: {str(e)}")
        
        # تلاش با صرافی دیگر
        if exchange_id != 'binance':
            logger.info(f"تلاش مجدد با صرافی binance")
            try:
                return get_current_price(symbol, 'binance')
            except Exception as e2:
                logger.error(f"خطا در دریافت قیمت با binance: {str(e2)}")
        
        return 0.0

def get_multiple_prices(symbols: List[str], exchange_id: str = DEFAULT_EXCHANGE) -> Dict[str, float]:
    """
    دریافت قیمت چندین ارز به صورت یکجا
    
    Args:
        symbols (List[str]): لیست نمادهای ارز
        exchange_id (str): شناسه صرافی
        
    Returns:
        Dict[str, float]: دیکشنری قیمت‌ها
    """
    # استاندارد‌سازی نمادها
    normalized_symbols = [normalize_symbol(s) for s in symbols]
    
    results = {}
    
    try:
        # دریافت نمونه صرافی
        exchange = get_exchange_instance(exchange_id)
        
        # دریافت تیکرهای همه بازارها
        all_tickers = exchange.fetch_tickers()
        
        # استخراج قیمت‌ها
        for symbol in normalized_symbols:
            if symbol in all_tickers:
                results[symbol] = all_tickers[symbol]['last']
            else:
                # تلاش مستقیم برای هر نماد
                try:
                    results[symbol] = get_current_price(symbol, exchange_id)
                except Exception as e:
                    logger.error(f"خطا در دریافت قیمت {symbol}: {str(e)}")
                    results[symbol] = 0.0
        
        return results
        
    except Exception as e:
        logger.error(f"خطا در دریافت قیمت‌های چندگانه از {exchange_id}: {str(e)}")
        
        # تلاش تک به تک
        for symbol in normalized_symbols:
            try:
                results[symbol] = get_current_price(symbol, exchange_id)
            except Exception as e2:
                logger.error(f"خطا در دریافت قیمت {symbol}: {str(e2)}")
                results[symbol] = 0.0
        
        return results

def get_top_cryptocurrencies(limit: int = 100, min_volume: float = 0) -> pd.DataFrame:
    """
    دریافت لیست ارزهای دیجیتال برتر از نظر حجم معاملات
    
    از چندین منبع استفاده می‌کند تا قابلیت اطمینان را بهبود دهد:
    1. CoinGecko
    2. KuCoin (پشتیبان)
    3. سایر صرافی‌ها (پشتیبان)
    
    Args:
        limit (int): تعداد ارزها
        min_volume (float): حداقل حجم معاملات (دلار)
        
    Returns:
        pd.DataFrame: دیتافریم اطلاعات ارزها
    """
    # ستون‌های فارسی نهایی
    persian_columns = ['نماد', 'نام', 'قیمت فعلی', 'ارزش بازار', 'رتبه بازار', 'حجم معاملات', 'تغییر قیمت 24 ساعته']
    
    # 1. تلاش برای دریافت از CoinGecko
    try:
        # آدرس API
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={limit}&page=1"
        
        # اضافه کردن پروکسی اگر لازم است
        proxies = None
        if USE_PROXY:
            proxy = get_proxy('coingecko')
            if proxy:
                proxies = {
                    'http': proxy,
                    'https': proxy
                }
        
        # ارسال درخواست
        logger.info("در حال دریافت داده‌های ارزهای برتر از CoinGecko...")
        response = requests.get(url, proxies=proxies, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # تبدیل به دیتافریم
            df = pd.DataFrame(data)
            
            # فیلتر بر اساس حجم معاملات
            if min_volume > 0 and 'total_volume' in df.columns:
                df = df[df['total_volume'] >= min_volume]
            
            # اصلاح ستون‌ها
            if not df.empty:
                # تبدیل نماد به فرمت مناسب
                if 'symbol' in df.columns:
                    df['symbol'] = df['symbol'].str.upper() + '/USDT'
                
                # انتخاب ستون‌های مورد نیاز
                english_columns = [
                    'symbol', 'name', 'current_price', 'market_cap', 'market_cap_rank',
                    'total_volume', 'price_change_percentage_24h'
                ]
                
                available_columns = [col for col in english_columns if col in df.columns]
                if available_columns:
                    df = df[available_columns]
                    
                    # تغییر نام ستون‌ها به فارسی
                    column_mapping = {
                        'symbol': 'نماد',
                        'name': 'نام',
                        'current_price': 'قیمت فعلی',
                        'market_cap': 'ارزش بازار',
                        'market_cap_rank': 'رتبه بازار',
                        'total_volume': 'حجم معاملات',
                        'price_change_percentage_24h': 'تغییر قیمت 24 ساعته'
                    }
                    
                    # تغییر نام ستون‌های موجود
                    rename_dict = {col: column_mapping[col] for col in available_columns if col in column_mapping}
                    df = df.rename(columns=rename_dict)
                    
                    # اضافه کردن ستون‌های فارسی موجود نیست
                    for col in persian_columns:
                        if col not in df.columns:
                            df[col] = None
                    
                    logger.info(f"دریافت موفق {len(df)} ارز از CoinGecko")
                    return df[persian_columns]
            
    except Exception as e:
        logger.error(f"خطا در دریافت لیست ارزهای برتر از CoinGecko: {str(e)}")
    
    # 2. تلاش برای دریافت از KuCoin
    try:
        logger.info("در حال دریافت داده‌های ارزهای برتر از KuCoin...")
        # دریافت نمونه KuCoin
        kucoin = get_exchange_instance('kucoin')
        
        # دریافت تیکرهای KuCoin
        tickers = kucoin.fetch_tickers()
        
        # محدود کردن به جفت‌های USDT
        usdt_tickers = {symbol: ticker for symbol, ticker in tickers.items() if symbol.endswith('/USDT')}
        
        if usdt_tickers:
            # تبدیل به دیتافریم
            ticker_list = []
            
            for symbol, ticker in usdt_tickers.items():
                info = {
                    'نماد': symbol,
                    'قیمت فعلی': ticker['last'],
                    'حجم معاملات': ticker.get('quoteVolume', 0),  # حجم بر حسب USDT
                    'تغییر قیمت 24 ساعته': ticker.get('percentage', None)
                }
                ticker_list.append(info)
            
            df = pd.DataFrame(ticker_list)
            
            # فیلتر براساس حجم معاملات
            if min_volume > 0 and 'حجم معاملات' in df.columns:
                df = df[df['حجم معاملات'] >= min_volume]
            
            # مرتب‌سازی براساس حجم معاملات
            if 'حجم معاملات' in df.columns:
                df = df.sort_values('حجم معاملات', ascending=False)
            
            # محدود کردن به تعداد درخواستی
            if len(df) > limit:
                df = df.head(limit)
            
            # اضافه کردن ستون‌های فارسی موجود نیست
            for col in persian_columns:
                if col not in df.columns:
                    df[col] = None
            
            logger.info(f"دریافت موفق {len(df)} ارز از KuCoin")
            return df[persian_columns]
    
    except Exception as e:
        logger.error(f"خطا در دریافت لیست ارزهای برتر از KuCoin: {str(e)}")
    
    # 3. تلاش برای دریافت از سایر صرافی‌ها
    try:
        logger.info("در حال دریافت داده‌های ارزهای برتر از سایر صرافی‌ها...")
        df = get_top_cryptocurrencies_from_exchange(limit, min_volume)
        
        if not df.empty:
            # اضافه کردن ستون‌های فارسی موجود نیست
            for col in persian_columns:
                if col not in df.columns:
                    df[col] = None
            
            logger.info(f"دریافت موفق {len(df)} ارز از صرافی‌های دیگر")
            return df[persian_columns]
            
    except Exception as e:
        logger.error(f"خطا در دریافت لیست ارزهای برتر از سایر صرافی‌ها: {str(e)}")
    
    # 4. در صورت شکست همه روش‌ها، لیست اصلی ارزهای دیجیتال را برمی‌گردانیم
    logger.warning("همه روش‌های دریافت داده شکست خورد. بازگشت لیست پایه ارزهای دیجیتال")
    
    # لیست پایه ارزهای مهم
    base_coins = [
        {'نماد': 'BTC/USDT', 'نام': 'Bitcoin', 'قیمت فعلی': None, 'رتبه بازار': 1},
        {'نماد': 'ETH/USDT', 'نام': 'Ethereum', 'قیمت فعلی': None, 'رتبه بازار': 2},
        {'نماد': 'BNB/USDT', 'نام': 'Binance Coin', 'قیمت فعلی': None, 'رتبه بازار': 3},
        {'نماد': 'SOL/USDT', 'نام': 'Solana', 'قیمت فعلی': None, 'رتبه بازار': 4},
        {'نماد': 'XRP/USDT', 'نام': 'Ripple', 'قیمت فعلی': None, 'رتبه بازار': 5},
        {'نماد': 'DOGE/USDT', 'نام': 'Dogecoin', 'قیمت فعلی': None, 'رتبه بازار': 6},
        {'نماد': 'ADA/USDT', 'نام': 'Cardano', 'قیمت فعلی': None, 'رتبه بازار': 7},
        {'نماد': 'MATIC/USDT', 'نام': 'Polygon', 'قیمت فعلی': None, 'رتبه بازار': 8},
        {'نماد': 'DOT/USDT', 'نام': 'Polkadot', 'قیمت فعلی': None, 'رتبه بازار': 9},
        {'نماد': 'LTC/USDT', 'نام': 'Litecoin', 'قیمت فعلی': None, 'رتبه بازار': 10}
    ]
    
    # تبدیل به دیتافریم
    df = pd.DataFrame(base_coins)
    
    # اضافه کردن ستون‌های فارسی موجود نیست
    for col in persian_columns:
        if col not in df.columns:
            df[col] = None
    
    return df[persian_columns]

def get_top_cryptocurrencies_from_exchange(limit: int = 100, min_volume: float = 0) -> pd.DataFrame:
    """
    دریافت لیست ارزهای دیجیتال برتر از صرافی
    
    Args:
        limit (int): تعداد ارزها
        min_volume (float): حداقل حجم معاملات (دلار)
        
    Returns:
        pd.DataFrame: دیتافریم اطلاعات ارزها
    """
    try:
        # استفاده از Binance برای دریافت تیکرها
        exchange = get_exchange_instance('binance')
        
        # دریافت تیکرهای همه بازارها
        all_tickers = exchange.fetch_tickers()
        
        # فیلتر کردن جفت‌ارزهای USDT
        usdt_tickers = [ticker for ticker in all_tickers.values() if '/USDT' in ticker['symbol']]
        
        # تبدیل به دیتافریم
        df = pd.DataFrame(usdt_tickers)
        
        # فیلتر بر اساس حجم معاملات
        if min_volume > 0 and 'quoteVolume' in df.columns:
            df = df[df['quoteVolume'] >= min_volume]
        
        # مرتب‌سازی بر اساس حجم معاملات
        if 'quoteVolume' in df.columns:
            df = df.sort_values(by='quoteVolume', ascending=False)
        
        # محدود کردن تعداد سطرها
        df = df.head(limit)
        
        # تغییر نام ستون‌ها به فارسی
        column_mapping = {
            'symbol': 'نماد',
            'last': 'قیمت فعلی',
            'quoteVolume': 'حجم معاملات',
            'percentage': 'تغییر قیمت 24 ساعته'
        }
        
        df.rename(columns={col: column_mapping.get(col, col) for col in df.columns if col in column_mapping}, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"خطا در دریافت لیست ارزهای برتر از صرافی: {str(e)}")
        return pd.DataFrame()

def get_fear_greed_index() -> Dict[str, Any]:
    """
    دریافت شاخص ترس و طمع بازار
    
    Returns:
        Dict[str, Any]: اطلاعات شاخص ترس و طمع
    """
    try:
        url = "https://api.alternative.me/fng/"
        
        # اضافه کردن پروکسی اگر لازم است
        proxies = None
        if USE_PROXY:
            proxy = get_proxy('alternative')
            if proxy:
                proxies = {
                    'http': proxy,
                    'https': proxy
                }
        
        response = requests.get(url, proxies=proxies, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('data') and len(data['data']) > 0:
                index_data = data['data'][0]
                
                # اضافه کردن تفسیر به فارسی
                value = int(index_data['value'])
                classification = index_data['value_classification']
                
                # تبدیل طبقه‌بندی به فارسی
                fa_classification = {
                    'Extreme Fear': 'ترس شدید',
                    'Fear': 'ترس',
                    'Neutral': 'خنثی',
                    'Greed': 'طمع',
                    'Extreme Greed': 'طمع شدید'
                }.get(classification, classification)
                
                index_data['value_fa'] = fa_classification
                index_data['value_int'] = value
                
                return index_data
        
        # در صورت شکست، مقدار پیش‌فرض
        return {
            'value': '0',
            'value_classification': 'Unknown',
            'value_fa': 'نامشخص',
            'value_int': 0,
            'timestamp': str(int(time.time())),
            'time_until_update': '0'
        }
            
    except Exception as e:
        logger.error(f"خطا در دریافت شاخص ترس و طمع: {str(e)}")
        
        # مقدار پیش‌فرض در صورت خطا
        return {
            'value': '0',
            'value_classification': 'Unknown',
            'value_fa': 'نامشخص',
            'value_int': 0,
            'timestamp': str(int(time.time())),
            'time_until_update': '0'
        }

def get_market_sentiment() -> Dict[str, Any]:
    """
    دریافت احساسات کلی بازار
    
    Returns:
        Dict[str, Any]: اطلاعات احساسات بازار
    """
    results = {}
    
    # دریافت شاخص ترس و طمع
    results['fear_greed'] = get_fear_greed_index()
    
    # بررسی روند قیمت بیت‌کوین در 24 ساعت گذشته
    try:
        btc_df = get_ohlcv_data('BTC/USDT', '1h', 25)
        
        if not btc_df.empty:
            # محاسبه تغییرات 24 ساعته
            start_price = btc_df['close'].iloc[0]
            end_price = btc_df['close'].iloc[-1]
            change_24h = ((end_price / start_price) - 1) * 100
            
            # میانگین حجم معاملات 24 ساعت اخیر
            avg_volume = btc_df['volume'].mean()
            
            results['btc_change_24h'] = change_24h
            results['btc_price'] = end_price
            results['btc_avg_volume'] = avg_volume
            
            # تحلیل روند
            if change_24h > 3:
                results['btc_trend'] = 'افزایشی قوی'
            elif change_24h > 1:
                results['btc_trend'] = 'افزایشی'
            elif change_24h < -3:
                results['btc_trend'] = 'کاهشی قوی'
            elif change_24h < -1:
                results['btc_trend'] = 'کاهشی'
            else:
                results['btc_trend'] = 'نسبتاً ثابت'
    except Exception as e:
        logger.error(f"خطا در محاسبه روند بیت‌کوین: {str(e)}")
        results['btc_trend'] = 'نامشخص'
    
    return results

def is_exchange_available(exchange_id: str) -> bool:
    """
    بررسی در دسترس بودن صرافی
    
    Args:
        exchange_id (str): شناسه صرافی
        
    Returns:
        bool: وضعیت دسترسی
    """
    try:
        exchange = get_exchange_instance(exchange_id)
        # تلاش برای انجام یک عملیات ساده
        exchange.fetch_ticker('BTC/USDT')
        return True
    except Exception as e:
        logger.warning(f"صرافی {exchange_id} در دسترس نیست: {str(e)}")
        return False

def get_available_exchanges() -> List[str]:
    """
    دریافت لیست صرافی‌های در دسترس
    
    Returns:
        List[str]: لیست صرافی‌های در دسترس
    """
    available = []
    
    for exchange_id in SUPPORTED_EXCHANGES:
        if is_exchange_available(exchange_id):
            available.append(exchange_id)
    
    return available
    
def get_price_from_multiple_exchanges(symbol: str, exchanges: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    دریافت قیمت یک ارز از چندین صرافی مختلف

    Args:
        symbol (str): نماد ارز دیجیتال (مثل BTC/USDT)
        exchanges (Optional[List[str]]): لیست صرافی‌های مورد نظر، اگر None باشد از صرافی‌های در دسترس استفاده می‌شود

    Returns:
        Dict[str, Any]: دیکشنری حاوی قیمت‌ها و اطلاعات مقایسه‌ای
    """
    symbol = normalize_symbol(symbol)
    
    # اگر لیست صرافی‌ها مشخص نشده، از صرافی‌های در دسترس استفاده می‌کنیم
    if not exchanges:
        exchanges = get_available_exchanges()
        # محدود کردن به حداکثر 5 صرافی برای عملکرد بهتر
        if len(exchanges) > 5:
            exchanges = exchanges[:5]
    
    results = {}
    prices = {}
    errors = {}
    
    # جمع‌آوری قیمت‌ها از هر صرافی
    for exchange_id in exchanges:
        try:
            price = get_current_price(symbol, exchange_id)
            prices[exchange_id] = price
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"خطا در دریافت قیمت {symbol} از {exchange_id}: {error_msg}")
            errors[exchange_id] = error_msg
    
    if not prices:
        raise Exception(f"هیچ قیمتی برای {symbol} از صرافی‌های درخواستی یافت نشد")
    
    # محاسبه میانگین قیمت
    avg_price = sum(prices.values()) / len(prices)
    
    # یافتن حداقل و حداکثر قیمت
    min_price = min(prices.values())
    max_price = max(prices.values())
    
    # یافتن صرافی‌های با حداقل و حداکثر قیمت
    min_exchange = [ex for ex, p in prices.items() if p == min_price][0]
    max_exchange = [ex for ex, p in prices.items() if p == max_price][0]
    
    # محاسبه درصد اختلاف قیمت
    price_difference_pct = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
    
    # نتایج را آماده می‌کنیم
    results = {
        'symbol': symbol,
        'prices': prices,
        'average_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'min_exchange': min_exchange,
        'max_exchange': max_exchange,
        'price_difference_pct': price_difference_pct,
        'exchanges_count': len(prices),
        'errors': errors,
        'timestamp': datetime.now().timestamp()
    }
    
    return results

def get_available_symbols(exchange_id: str = 'binance') -> List[str]:
    """
    دریافت لیست نمادهای قابل معامله در صرافی
    
    Args:
        exchange_id (str): شناسه صرافی
        
    Returns:
        List[str]: لیست نمادها
    """
    try:
        exchange = get_exchange_instance(exchange_id)
        markets = exchange.load_markets()
        
        # فیلتر کردن بازارهای USDT
        usdt_symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
        
        return sorted(usdt_symbols)
    except Exception as e:
        logger.error(f"خطا در دریافت نمادهای {exchange_id}: {str(e)}")
        return []

def get_historical_data(
    symbol: str, 
    timeframe: str = '1d', 
    days: int = 30,
    exchange_id: str = 'binance'
) -> pd.DataFrame:
    """
    دریافت داده‌های تاریخی قیمت
    
    Args:
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        days (int): تعداد روزهای موردنظر
        exchange_id (str): شناسه صرافی
        
    Returns:
        pd.DataFrame: دیتافریم داده‌های تاریخی
    """
    # تبدیل روز به تعداد کندل موردنیاز
    timeframe_in_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    
    # استاندارد‌سازی تایم‌فریم
    tf = TIMEFRAME_MAP.get(timeframe, timeframe)
    
    minutes_in_day = 24 * 60
    candles_per_day = minutes_in_day / timeframe_in_minutes.get(tf, 1440)
    limit = int(days * candles_per_day) + 10  # اضافه کردن تعدادی برای اطمینان
    
    # محدود کردن به حداکثر تعداد مجاز
    limit = min(limit, 1000)
    
    return get_ohlcv_data(symbol, tf, limit, exchange_id)

# تنظیم نمونه‌های صرافی با اجرای ماژول
def initialize_exchanges():
    """راه‌اندازی اولیه صرافی‌های پیش‌فرض"""
    try:
        for exchange_id in ['binance', 'kucoin']:
            get_exchange_instance(exchange_id)
        logger.info("صرافی‌های پیش‌فرض با موفقیت راه‌اندازی شدند")
    except Exception as e:
        logger.error(f"خطا در راه‌اندازی صرافی‌های پیش‌فرض: {str(e)}")

# اجرای راه‌اندازی اولیه در زمان import شدن ماژول
initialize_exchanges()
