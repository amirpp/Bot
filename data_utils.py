"""
ماژول پردازش و آماده‌سازی داده‌ها

این ماژول شامل توابع مفید برای پردازش، آماده‌سازی و تبدیل داده‌هاست.
توابع این ماژول برای کار با داده‌های بازار ارزهای دیجیتال بهینه شده‌اند.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional, Tuple

# تنظیم لاگر
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataUtils')

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    بررسی اعتبار دیتافریم و وجود ستون‌های مورد نیاز
    
    Args:
        df (pd.DataFrame): دیتافریم ورودی
        required_columns (List[str], optional): لیست ستون‌های مورد نیاز
        
    Returns:
        bool: نتیجه اعتبارسنجی
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("دیتافریم نامعتبر یا خالی")
        return False
    
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"ستون‌های مورد نیاز در دیتافریم موجود نیست: {missing_columns}")
            return False
    
    return True

def prepare_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    آماده‌سازی داده‌های قیمت و حجم (OHLCV)
    
    Args:
        df (pd.DataFrame): دیتافریم ورودی
        
    Returns:
        pd.DataFrame: دیتافریم آماده‌سازی شده
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if not validate_dataframe(df, required_columns):
        # ایجاد یک دیتافریم خالی با ستون‌های مورد نیاز
        return pd.DataFrame(columns=required_columns)
    
    # کپی دیتافریم برای جلوگیری از تغییر دیتافریم اصلی
    result_df = df.copy()
    
    # تبدیل ایندکس به تاریخ و زمان اگر نیست
    if not isinstance(result_df.index, pd.DatetimeIndex):
        try:
            if 'timestamp' in result_df.columns:
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                result_df.set_index('timestamp', inplace=True)
            else:
                # تلاش برای تبدیل ایندکس فعلی به تاریخ و زمان
                result_df.index = pd.to_datetime(result_df.index)
        except Exception as e:
            logger.warning(f"خطا در تبدیل ایندکس به تاریخ و زمان: {str(e)}")
    
    # اطمینان از اینکه تمام ستون‌های عددی هستند
    for col in required_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    # حذف ردیف‌های دارای مقادیر گمشده یا نامعتبر
    result_df.dropna(subset=required_columns, inplace=True)
    
    # مرتب‌سازی بر اساس زمان (صعودی)
    result_df.sort_index(inplace=True)
    
    return result_df

def add_technical_indicators(df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
    """
    افزودن شاخص‌های تکنیکال به دیتافریم
    
    Args:
        df (pd.DataFrame): دیتافریم ورودی
        indicators (List[str], optional): لیست شاخص‌های مورد نظر
        
    Returns:
        pd.DataFrame: دیتافریم با شاخص‌های اضافه شده
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if not validate_dataframe(df, required_columns):
        return df
    
    # کپی دیتافریم
    result_df = df.copy()
    
    # لیست پیش‌فرض شاخص‌ها
    if indicators is None:
        indicators = ['rsi', 'macd', 'bollinger_bands', 'sma_50', 'sma_200', 'ema_20']
    
    # محاسبه شاخص‌ها
    for indicator in indicators:
        try:
            if indicator.lower() == 'rsi':
                # شاخص RSI
                delta = result_df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                result_df['rsi'] = 100 - (100 / (1 + rs))
            
            elif indicator.lower() == 'macd':
                # شاخص MACD
                ema12 = result_df['close'].ewm(span=12, adjust=False).mean()
                ema26 = result_df['close'].ewm(span=26, adjust=False).mean()
                result_df['macd'] = ema12 - ema26
                result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
                result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
            
            elif indicator.lower() == 'bollinger_bands':
                # باندهای بولینگر
                rolling_mean = result_df['close'].rolling(window=20).mean()
                rolling_std = result_df['close'].rolling(window=20).std()
                result_df['bollinger_upper'] = rolling_mean + (rolling_std * 2)
                result_df['bollinger_middle'] = rolling_mean
                result_df['bollinger_lower'] = rolling_mean - (rolling_std * 2)
            
            elif indicator.lower().startswith('sma_'):
                # میانگین متحرک ساده
                window = int(indicator.split('_')[1])
                result_df[f'sma_{window}'] = result_df['close'].rolling(window=window).mean()
            
            elif indicator.lower().startswith('ema_'):
                # میانگین متحرک نمایی
                window = int(indicator.split('_')[1])
                result_df[f'ema_{window}'] = result_df['close'].ewm(span=window, adjust=False).mean()
            
            elif indicator.lower() == 'atr':
                # میانگین دامنه واقعی
                high_low = result_df['high'] - result_df['low']
                high_close = (result_df['high'] - result_df['close'].shift()).abs()
                low_close = (result_df['low'] - result_df['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                result_df['atr'] = true_range.rolling(14).mean()
            
            elif indicator.lower() == 'adx':
                # شاخص جهت متوسط
                # محاسبه True Range
                high_low = result_df['high'] - result_df['low']
                high_close = (result_df['high'] - result_df['close'].shift()).abs()
                low_close = (result_df['low'] - result_df['close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                
                # محاسبه +DI و -DI
                up_move = result_df['high'].diff()
                down_move = result_df['low'].diff(-1).abs()
                
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                
                plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
                minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr
                
                # محاسبه ADX
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
                result_df['adx'] = dx.rolling(14).mean()
            
        except Exception as e:
            logger.error(f"خطا در محاسبه شاخص {indicator}: {str(e)}")
    
    return result_df

def calculate_future_returns(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    محاسبه بازدهی آینده در بازه‌های زمانی مختلف
    
    Args:
        df (pd.DataFrame): دیتافریم ورودی
        periods (List[int], optional): لیست بازه‌های زمانی
        
    Returns:
        pd.DataFrame: دیتافریم با بازدهی‌های اضافه شده
    """
    if not validate_dataframe(df, ['close']):
        return df
    
    # کپی دیتافریم
    result_df = df.copy()
    
    # بازه‌های زمانی پیش‌فرض
    if periods is None:
        periods = [1, 3, 7, 14, 30]
    
    # محاسبه بازدهی‌ها
    for period in periods:
        try:
            # بازدهی مطلق
            result_df[f'return_{period}d'] = result_df['close'].shift(-period) / result_df['close'] - 1
            
            # بازدهی تجمعی (لگاریتمی)
            result_df[f'log_return_{period}d'] = np.log(result_df['close'].shift(-period) / result_df['close'])
            
        except Exception as e:
            logger.error(f"خطا در محاسبه بازدهی برای دوره {period}: {str(e)}")
    
    return result_df

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, target_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تقسیم داده‌ها به مجموعه آموزش و آزمون
    
    Args:
        df (pd.DataFrame): دیتافریم ورودی
        test_size (float): نسبت داده‌های آزمون
        target_col (str, optional): نام ستون هدف
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: دیتافریم‌های آموزش و آزمون
    """
    if not validate_dataframe(df):
        return pd.DataFrame(), pd.DataFrame()
    
    # تقسیم بر اساس زمان (با توجه به اینکه داده‌های سری زمانی هستند)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

def prepare_features_targets(df: pd.DataFrame, target_cols: List[str], feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    آماده‌سازی ویژگی‌ها و هدف‌ها برای آموزش مدل
    
    Args:
        df (pd.DataFrame): دیتافریم ورودی
        target_cols (List[str]): لیست ستون‌های هدف
        feature_cols (List[str], optional): لیست ستون‌های ویژگی
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: ماتریس‌های ویژگی و هدف
    """
    if not validate_dataframe(df):
        return np.array([]), np.array([])
    
    # اگر ستون‌های ویژگی مشخص نشده باشند، تمام ستون‌ها به جز هدف‌ها را انتخاب می‌کنیم
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in target_cols]
    
    # بررسی وجود تمام ستون‌های مورد نیاز
    missing_features = [col for col in feature_cols if col not in df.columns]
    missing_targets = [col for col in target_cols if col not in df.columns]
    
    if missing_features or missing_targets:
        logger.error(f"ستون‌های مفقود - ویژگی‌ها: {missing_features}, هدف‌ها: {missing_targets}")
        return np.array([]), np.array([])
    
    # حذف ردیف‌های دارای مقادیر گمشده
    clean_df = df[feature_cols + target_cols].dropna()
    
    if clean_df.empty:
        logger.error("پس از حذف داده‌های ناقص، دیتافریمی باقی نماند")
        return np.array([]), np.array([])
    
    # استخراج ویژگی‌ها و هدف‌ها
    X = clean_df[feature_cols].values
    y = clean_df[target_cols].values
    
    return X, y

def normalize_data(data: np.ndarray, scaler=None, fit: bool = True) -> Tuple[np.ndarray, Any]:
    """
    نرمال‌سازی داده‌ها
    
    Args:
        data (np.ndarray): داده‌های ورودی
        scaler: مقیاس‌دهنده (اختیاری)
        fit (bool): اجرای فیت روی مقیاس‌دهنده
        
    Returns:
        Tuple[np.ndarray, Any]: داده‌های نرمال شده و مقیاس‌دهنده
    """
    if data.size == 0:
        return data, scaler
    
    # ایجاد مقیاس‌دهنده جدید اگر ارائه نشده باشد
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    
    # نرمال‌سازی داده‌ها
    if fit:
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)
    
    return normalized_data, scaler

def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    ایجاد توالی‌ها برای مدل‌های سری زمانی
    
    Args:
        data (np.ndarray): داده‌های ورودی
        sequence_length (int): طول توالی
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: داده‌های ورودی و هدف
    """
    if len(data) <= sequence_length:
        return np.array([]), np.array([])
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    
    return np.array(X), np.array(y)

def evaluate_forecast(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """
    ارزیابی پیش‌بینی با معیارهای مختلف
    
    Args:
        actual (np.ndarray): مقادیر واقعی
        forecast (np.ndarray): مقادیر پیش‌بینی شده
        
    Returns:
        Dict[str, float]: دیکشنری معیارهای ارزیابی
    """
    if len(actual) != len(forecast) or len(actual) == 0:
        return {
            "error": "ابعاد داده‌ها نامعتبر است یا داده‌ای موجود نیست"
        }
    
    # محاسبه خطاها
    errors = actual - forecast
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # معیارهای ارزیابی
    mae = np.mean(abs_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    mape = np.mean(abs_errors / np.abs(actual)) * 100
    
    # محاسبه دقت جهت (درصد پیش‌بینی صحیح جهت تغییرات)
    direction_actual = np.diff(actual) > 0
    direction_forecast = np.diff(forecast) > 0
    direction_accuracy = np.mean(direction_actual == direction_forecast) * 100
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "direction_accuracy": direction_accuracy
    }

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    تغییر دوره زمانی داده‌های OHLCV
    
    Args:
        df (pd.DataFrame): دیتافریم ورودی
        timeframe (str): دوره زمانی جدید ('1h', '4h', '1d', ...)
        
    Returns:
        pd.DataFrame: دیتافریم با دوره زمانی جدید
    """
    if not validate_dataframe(df, ['open', 'high', 'low', 'close', 'volume']):
        return df
    
    # تبدیل دوره زمانی به فرمت pandas
    if timeframe.lower().endswith('m'):
        freq = timeframe[:-1] + 'min'
    elif timeframe.lower().endswith('h'):
        freq = timeframe[:-1] + 'H'
    elif timeframe.lower().endswith('d'):
        freq = timeframe[:-1] + 'D'
    elif timeframe.lower().endswith('w'):
        freq = timeframe[:-1] + 'W'
    else:
        logger.error(f"دوره زمانی نامعتبر: {timeframe}")
        return df
    
    # ایجاد دیتافریم با دوره زمانی جدید
    resampled = pd.DataFrame()
    
    try:
        resampled['open'] = df['open'].resample(freq).first()
        resampled['high'] = df['high'].resample(freq).max()
        resampled['low'] = df['low'].resample(freq).min()
        resampled['close'] = df['close'].resample(freq).last()
        resampled['volume'] = df['volume'].resample(freq).sum()
        
        # حذف ردیف‌های دارای مقادیر گمشده
        resampled.dropna(inplace=True)
        
    except Exception as e:
        logger.error(f"خطا در تغییر دوره زمانی: {str(e)}")
        return df
    
    return resampled

def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on: str = None, how: str = 'inner') -> pd.DataFrame:
    """
    ادغام دو دیتافریم
    
    Args:
        df1 (pd.DataFrame): دیتافریم اول
        df2 (pd.DataFrame): دیتافریم دوم
        on (str, optional): ستون مشترک برای ادغام
        how (str): نوع ادغام ('inner', 'outer', 'left', 'right')
        
    Returns:
        pd.DataFrame: دیتافریم ادغام شده
    """
    if df1 is None or df2 is None:
        return df1 if df1 is not None else df2
    
    if df1.empty:
        return df2
    
    if df2.empty:
        return df1
    
    try:
        # اگر ستون مشترک مشخص نشده و هر دو دیتافریم ایندکس زمانی دارند
        if on is None and isinstance(df1.index, pd.DatetimeIndex) and isinstance(df2.index, pd.DatetimeIndex):
            return pd.merge(df1, df2, left_index=True, right_index=True, how=how)
        
        # ادغام بر اساس ستون مشترک
        if on is not None:
            if on in df1.columns and on in df2.columns:
                return pd.merge(df1, df2, on=on, how=how)
            else:
                logger.error(f"ستون {on} در هر دو دیتافریم موجود نیست")
        
        # ادغام بر اساس ایندکس
        return pd.merge(df1, df2, left_index=True, right_index=True, how=how)
        
    except Exception as e:
        logger.error(f"خطا در ادغام دیتافریم‌ها: {str(e)}")
        return df1