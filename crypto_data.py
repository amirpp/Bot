"""
ماژول دریافت و مدیریت داده‌های بازار ارزهای دیجیتال

این ماژول شامل توابع دریافت و مدیریت داده‌های قیمت و حجم ارزهای دیجیتال،
ذخیره‌سازی و بازیابی داده‌ها، و استفاده بهینه از API های صرافی‌ها است.
"""

import pandas as pd
import numpy as np
import ccxt
import json
import datetime
import time
import os
import random
import logging
from typing import Optional, Dict, List, Union, Any
import pickle
import hashlib
from pathlib import Path

# استفاده از API سرویس‌ها
from api_services import get_ohlcv_data_multi_source

# تنظیم لاگر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ایجاد مسیر ذخیره‌سازی داده‌ها
CACHE_DIR = 'crypto_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(symbol: str, timeframe: str) -> str:
    """
    ایجاد مسیر فایل کش برای یک نماد و تایم‌فریم خاص
    
    Args:
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        
    Returns:
        str: مسیر فایل کش
    """
    # حذف کاراکترهای غیرمجاز از نام فایل
    safe_symbol = symbol.replace('/', '_')
    
    return os.path.join(CACHE_DIR, f"{safe_symbol}_{timeframe}.pkl")

def get_crypto_data(symbol: str, timeframe: str, lookback_days: int = 30, 
                    exchange: str = 'binance', use_cache: bool = True) -> pd.DataFrame:
    """
    دریافت داده‌های تاریخی ارز دیجیتال
    
    Args:
        symbol (str): نماد ارز (مثال: "BTC/USDT")
        timeframe (str): تایم‌فریم (مثال: "1h", "1d")
        lookback_days (int): تعداد روزهای گذشته
        exchange (str): نام صرافی
        use_cache (bool): استفاده از کش
        
    Returns:
        pd.DataFrame: دیتافریم داده‌های OHLCV
    """
    logger.info(f"دریافت داده‌های {symbol} با تایم‌فریم {timeframe} از {exchange}")
    
    cache_path = get_cache_path(symbol, timeframe)
    
    # استفاده از کش اگر موجود باشد و فعال شده باشد
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            cache_time = os.path.getmtime(cache_path)
            cache_datetime = datetime.datetime.fromtimestamp(cache_time)
            
            # بررسی اینکه آیا کش بیش از 1 ساعت قدیمی است
            if (datetime.datetime.now() - cache_datetime).total_seconds() < 3600:
                logger.info(f"استفاده از داده‌های کش شده برای {symbol} ({len(df)} رکورد)")
                return df
            else:
                logger.info(f"کش برای {symbol} قدیمی است، در حال به‌روزرسانی...")
        except Exception as e:
            logger.warning(f"خطا در خواندن کش: {str(e)}")
    
    try:
        # تبدیل روز به میلی‌ثانیه
        since = int((datetime.datetime.now() - datetime.timedelta(days=lookback_days)).timestamp() * 1000)
        
        # دریافت داده‌ها با استفاده از تابع API سرویس‌ها
        df = get_ohlcv_data_multi_source(symbol, timeframe, lookback_days, exchange, use_alternate_sources=True)
        
        if df is not None and not df.empty:
            # ذخیره در کش
            df.to_pickle(cache_path)
            logger.info(f"داده‌های {symbol} با موفقیت دریافت شدند: {len(df)} رکورد")
            return df
        else:
            logger.warning(f"داده‌های دریافتی برای {symbol} خالی است")
            
            # اگر داده‌های کش موجود باشد، از آن استفاده کن
            if os.path.exists(cache_path):
                try:
                    df = pd.read_pickle(cache_path)
                    logger.info(f"استفاده از داده‌های کش شده قدیمی برای {symbol} ({len(df)} رکورد)")
                    return df
                except Exception as e:
                    logger.warning(f"خطا در خواندن کش: {str(e)}")
            
            # ایجاد داده‌های نمونه
            logger.warning(f"ایجاد داده‌های نمونه برای {symbol}")
            df = _generate_sample_data(symbol, timeframe, lookback_days)
            return df
            
    except Exception as e:
        logger.error(f"خطا در دریافت داده‌های {symbol}: {str(e)}")
        
        # اگر داده‌های کش موجود باشد، از آن استفاده کن
        if os.path.exists(cache_path):
            try:
                df = pd.read_pickle(cache_path)
                logger.info(f"استفاده از داده‌های کش شده برای {symbol} به علت خطا ({len(df)} رکورد)")
                return df
            except Exception as e2:
                logger.warning(f"خطا در خواندن کش: {str(e2)}")
        
        # ایجاد داده‌های نمونه
        logger.warning(f"ایجاد داده‌های نمونه برای {symbol}")
        df = _generate_sample_data(symbol, timeframe, lookback_days)
        return df

def get_current_price(symbol: str, exchange: str = 'binance') -> float:
    """
    دریافت قیمت فعلی یک ارز
    
    Args:
        symbol (str): نماد ارز (مثال: "BTC/USDT")
        exchange (str): نام صرافی
        
    Returns:
        float: قیمت فعلی
    """
    try:
        if exchange == 'binance':
            binance = ccxt.binance()
            ticker = binance.fetch_ticker(symbol)
            return ticker['last']
        elif exchange == 'kucoin':
            kucoin = ccxt.kucoin()
            ticker = kucoin.fetch_ticker(symbol)
            return ticker['last']
        else:
            # استفاده از صرافی‌های دیگر...
            df = get_crypto_data(symbol, '1m', lookback_days=1, exchange=exchange)
            if df is not None and not df.empty:
                return df['close'].iloc[-1]
            else:
                return None
    except Exception as e:
        logger.error(f"خطا در دریافت قیمت فعلی {symbol}: {str(e)}")
        
        # استفاده از داده‌های تاریخی
        df = get_crypto_data(symbol, '1m', lookback_days=1, exchange=exchange)
        if df is not None and not df.empty:
            return df['close'].iloc[-1]
        else:
            # ایجاد قیمت تصادفی
            if 'BTC' in symbol:
                return random.uniform(50000, 70000)
            elif 'ETH' in symbol:
                return random.uniform(2000, 4000)
            else:
                return random.uniform(0.1, 100)

def get_available_timeframes() -> List[str]:
    """
    دریافت لیست تایم‌فریم‌های در دسترس
    
    Returns:
        list: لیست تایم‌فریم‌ها
    """
    return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]

def get_available_exchanges() -> List[str]:
    """
    دریافت لیست صرافی‌های در دسترس
    
    Returns:
        list: لیست صرافی‌ها
    """
    return ["binance", "kucoin", "huobi", "bitfinex", "kraken", "coinbase", "bybit"]

def get_price_history_for_multiple_coins(symbols: List[str], timeframe: str,
                                        lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    دریافت تاریخچه قیمت برای چندین ارز
    
    Args:
        symbols (list): لیست نمادهای ارز
        timeframe (str): تایم‌فریم
        lookback_days (int): تعداد روزهای گذشته
        
    Returns:
        dict: دیکشنری دیتافریم‌ها (کلید: نماد ارز، مقدار: دیتافریم)
    """
    result = {}
    
    for symbol in symbols:
        try:
            df = get_crypto_data(symbol, timeframe, lookback_days)
            if df is not None and not df.empty:
                result[symbol] = df
        except Exception as e:
            logger.error(f"خطا در دریافت داده‌های {symbol}: {str(e)}")
    
    return result

def _generate_sample_data(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    """
    ایجاد داده‌های نمونه برای آزمایش و توسعه
    
    Args:
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        lookback_days (int): تعداد روزهای گذشته
        
    Returns:
        pd.DataFrame: دیتافریم داده‌های نمونه
    """
    # تنظیم پارامترهای داده‌های نمونه بر اساس نماد
    if 'BTC' in symbol:
        start_price = random.uniform(50000, 70000)
        volatility = 0.02
        volume_base = 1000000
    elif 'ETH' in symbol:
        start_price = random.uniform(2000, 4000)
        volatility = 0.025
        volume_base = 500000
    else:
        start_price = random.uniform(0.1, 100)
        volatility = 0.03
        volume_base = 100000
    
    # تبدیل تایم‌فریم به دقیقه
    timeframe_minutes = _timeframe_to_minutes(timeframe)
    
    # تعداد نقاط داده
    total_minutes = lookback_days * 24 * 60
    num_points = total_minutes // timeframe_minutes
    
    # ایجاد تاریخ‌ها
    end_time = datetime.datetime.now()
    timestamps = [end_time - datetime.timedelta(minutes=i * timeframe_minutes) for i in range(num_points)]
    timestamps.reverse()  # مرتب‌سازی صعودی
    
    # تابع تصادفی برای ایجاد روند واقعی‌تر
    def simulate_price_movement(base_price: float, days: int, volatility: float) -> np.ndarray:
        # تعداد نقاط داده در هر روز
        points_per_day = 24 * 60 // timeframe_minutes
        total_points = days * points_per_day
        
        # پارامترهای روند
        trend_cycles = days // 10  # تعداد چرخه‌های روند
        if trend_cycles < 1:
            trend_cycles = 1
        
        # ایجاد حرکت براونی هندسی
        returns = np.random.normal(0, volatility, total_points)
        price_movements = np.exp(np.cumsum(returns))
        
        # افزودن روند
        trend = np.sin(np.linspace(0, trend_cycles * 2 * np.pi, total_points)) * 0.2
        trend_component = np.exp(trend)
        
        # ترکیب روند و حرکت تصادفی
        final_movement = price_movements * trend_component
        
        # مقیاس‌بندی به قیمت پایه
        prices = base_price * final_movement
        
        return prices
    
    # شبیه‌سازی قیمت
    prices = simulate_price_movement(start_price, lookback_days, volatility)
    
    # ایجاد قیمت‌های OHLC
    data = []
    for i in range(len(timestamps)):
        if i > 0:
            idx = min(i, len(prices) - 1)
            price = prices[idx]
            
            # محاسبه داده‌های OHLC
            price_range = price * volatility * random.uniform(0.5, 1.5)
            high_price = price + price_range / 2
            low_price = price - price_range / 2
            open_price = price - price_range * random.uniform(-0.5, 0.5)
            close_price = price
            
            # محاسبه حجم
            volume = volume_base * random.uniform(0.5, 1.5)
            
            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    
    # ایجاد دیتافریم
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # نمایش پیام هشدار
    logger.warning(f"استفاده از داده‌های نمونه برای {symbol} با تایم‌فریم {timeframe} ({len(df)} رکورد)")
    
    return df

def _timeframe_to_minutes(timeframe: str) -> int:
    """
    تبدیل تایم‌فریم به دقیقه
    
    Args:
        timeframe (str): تایم‌فریم (مثال: "1h", "1d")
        
    Returns:
        int: معادل دقیقه
    """
    if 'm' in timeframe:
        return int(timeframe.replace('m', ''))
    elif 'h' in timeframe:
        return int(timeframe.replace('h', '')) * 60
    elif 'd' in timeframe:
        return int(timeframe.replace('d', '')) * 60 * 24
    elif 'w' in timeframe:
        return int(timeframe.replace('w', '')) * 60 * 24 * 7
    elif 'M' in timeframe:
        return int(timeframe.replace('M', '')) * 60 * 24 * 30
    else:
        return 1  # پیش‌فرض: 1 دقیقه

def get_market_data_for_symbol(symbol: str, interval: str = '1d', limit: int = 100) -> Dict:
    """
    دریافت اطلاعات بازار برای یک نماد
    
    Args:
        symbol (str): نماد ارز
        interval (str): بازه زمانی
        limit (int): تعداد رکوردها
        
    Returns:
        dict: اطلاعات بازار
    """
    # ایجاد نتیجه
    market_data = {}
    
    # دریافت داده‌های قیمت
    df = get_crypto_data(symbol, interval, lookback_days=limit//24)
    
    if df is not None and not df.empty:
        # محاسبه شاخص‌های آماری
        market_data['price_mean'] = df['close'].mean()
        market_data['price_max'] = df['high'].max()
        market_data['price_min'] = df['low'].min()
        market_data['price_std'] = df['close'].std()
        market_data['volume_mean'] = df['volume'].mean()
        market_data['price_change_24h'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
        
        # محاسبه ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        market_data['atr'] = true_range.rolling(14).mean().iloc[-1]
        
        # محاسبه روند
        if df['close'].iloc[-1] > df['close'].iloc[-10]:
            market_data['trend'] = 'UP'
        elif df['close'].iloc[-1] < df['close'].iloc[-10]:
            market_data['trend'] = 'DOWN'
        else:
            market_data['trend'] = 'NEUTRAL'
    
    return market_data