"""
ماژول تحلیل تکنیکال ارزهای دیجیتال

این ماژول شامل توابع مختلف برای محاسبه اندیکاتورهای تکنیکال و تولید سیگنال‌های معاملاتی است.
"""

import pandas as pd
import numpy as np
import streamlit as st
import random
from datetime import datetime
import time

# لیست اندیکاتورهای در دسترس
AVAILABLE_INDICATORS = [
    "RSI", "MACD", "Bollinger Bands", "EMA", "SMA", "Stochastic",
    "ADX", "ATR", "Parabolic SAR", "OBV", "CCI", "MFI", "ROC",
    "Williams %R", "Ichimoku", "Volume", "Support/Resistance",
    "Supertrend", "VWAP", "Moving Average"
]

# اندیکاتورهای محبوب
TOP_INDICATORS = [
    "RSI", "MACD", "Bollinger Bands", "EMA", "SMA", "Supertrend"
]

def perform_technical_analysis(df, indicators=None):
    """
    انجام تحلیل تکنیکال روی داده‌های قیمت

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت (OHLCV)
        indicators (list, optional): لیست اندیکاتورهای مورد نظر برای محاسبه

    Returns:
        pd.DataFrame: دیتافریم با ستون‌های اندیکاتورهای اضافه شده
    """
    # اگر دیتافریم خالی یا None باشد
    if df is None or df.empty:
        st.warning("داده‌ای برای تحلیل تکنیکال وجود ندارد")
        return df
    
    # اگر indicators مشخص نشده باشد، از مقادیر پیش‌فرض استفاده کن
    if indicators is None:
        indicators = TOP_INDICATORS
    
    # کپی دیتافریم برای جلوگیری از تغییر نسخه اصلی
    result_df = df.copy()
    
    # محاسبه اندیکاتورها
    for indicator in indicators:
        # RSI
        if indicator == "RSI":
            result_df['rsi'] = calculate_rsi(result_df)
        
        # MACD
        elif indicator == "MACD":
            result_df['macd'], result_df['macd_signal'], result_df['macd_hist'] = calculate_macd(result_df)
        
        # Bollinger Bands
        elif indicator == "Bollinger Bands":
            result_df['bb_upper'], result_df['bb_middle'], result_df['bb_lower'] = calculate_bollinger_bands(result_df)
        
        # EMA
        elif indicator == "EMA":
            result_df['ema_9'] = calculate_ema(result_df, period=9)
            result_df['ema_21'] = calculate_ema(result_df, period=21)
            result_df['ema_50'] = calculate_ema(result_df, period=50)
            result_df['ema_200'] = calculate_ema(result_df, period=200)
        
        # SMA
        elif indicator == "SMA" or indicator == "Moving Average":
            result_df['sma_20'] = calculate_sma(result_df, period=20)
            result_df['sma_50'] = calculate_sma(result_df, period=50)
            result_df['sma_100'] = calculate_sma(result_df, period=100)
            result_df['sma_200'] = calculate_sma(result_df, period=200)
        
        # Stochastic
        elif indicator == "Stochastic":
            result_df['stoch_k'], result_df['stoch_d'] = calculate_stochastic(result_df)
        
        # ADX
        elif indicator == "ADX":
            result_df['adx'] = calculate_adx(result_df)
        
        # ATR
        elif indicator == "ATR":
            result_df['atr'] = calculate_atr(result_df)
        
        # Parabolic SAR
        elif indicator == "Parabolic SAR":
            result_df['psar'] = calculate_parabolic_sar(result_df)
        
        # OBV (On Balance Volume)
        elif indicator == "OBV":
            result_df['obv'] = calculate_obv(result_df)
        
        # CCI
        elif indicator == "CCI":
            result_df['cci'] = calculate_cci(result_df)
        
        # MFI
        elif indicator == "MFI":
            result_df['mfi'] = calculate_mfi(result_df)
        
        # ROC
        elif indicator == "ROC":
            result_df['roc'] = calculate_roc(result_df)
        
        # Williams %R
        elif indicator == "Williams %R":
            result_df['williams_r'] = calculate_williams_r(result_df)
        
        # Ichimoku
        elif indicator == "Ichimoku":
            result_df['tenkan_sen'], result_df['kijun_sen'], result_df['senkou_span_a'], result_df['senkou_span_b'], result_df['chikou_span'] = calculate_ichimoku(result_df)
        
        # Volume
        elif indicator == "Volume":
            result_df['volume_sma'] = calculate_sma(result_df, period=20, column='volume')
            result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma']
        
        # Support/Resistance
        elif indicator == "Support/Resistance":
            result_df['support'], result_df['resistance'] = calculate_support_resistance(result_df)
        
        # Supertrend
        elif indicator == "Supertrend":
            result_df['supertrend'], result_df['supertrend_direction'] = calculate_supertrend(result_df)
        
        # VWAP
        elif indicator == "VWAP":
            result_df['vwap'] = calculate_vwap(result_df)
    
    return result_df

def generate_signals(df, indicators=None):
    """
    تولید سیگنال‌های معاملاتی بر اساس اندیکاتورها

    Args:
        df (pd.DataFrame): دیتافریم تحلیل تکنیکال
        indicators (list, optional): لیست اندیکاتورهای مورد استفاده

    Returns:
        pd.DataFrame: دیتافریم با ستون‌های سیگنال اضافه شده
    """
    # اگر دیتافریم خالی یا None باشد
    if df is None or df.empty:
        st.warning("داده‌ای برای تولید سیگنال‌ وجود ندارد")
        return df
    
    # اگر indicators مشخص نشده باشد، از مقادیر موجود در دیتافریم استفاده کن
    if indicators is None:
        indicators = []
        if 'rsi' in df.columns:
            indicators.append("RSI")
        if 'macd' in df.columns:
            indicators.append("MACD")
        if 'bb_upper' in df.columns:
            indicators.append("Bollinger Bands")
        if 'ema_9' in df.columns:
            indicators.append("EMA")
        if 'supertrend' in df.columns:
            indicators.append("Supertrend")
        if 'adx' in df.columns:
            indicators.append("ADX")
    
    # کپی دیتافریم برای جلوگیری از تغییر نسخه اصلی
    result_df = df.copy()
    
    # ایجاد ستون‌های سیگنال
    result_df['signal'] = 0  # 1: خرید، -1: فروش، 0: خنثی
    result_df['signal_strength'] = 0  # قدرت سیگنال (0-100)
    result_df['signal_desc'] = ""  # توضیحات سیگنال
    
    # بررسی سیگنال‌ها
    for i in range(2, len(result_df)):
        signals = []
        signal_strengths = []
        signal_descs = []
        
        # RSI
        if "RSI" in indicators and 'rsi' in result_df.columns:
            # سیگنال خرید: RSI < 30 و در حال افزایش
            if result_df['rsi'].iloc[i-1] < 30 and result_df['rsi'].iloc[i] > result_df['rsi'].iloc[i-1]:
                signals.append(1)
                signal_strengths.append(70 + (30 - result_df['rsi'].iloc[i-1]))
                signal_descs.append(f"RSI خرید: {result_df['rsi'].iloc[i]:.2f}")
            # سیگنال فروش: RSI > 70 و در حال کاهش
            elif result_df['rsi'].iloc[i-1] > 70 and result_df['rsi'].iloc[i] < result_df['rsi'].iloc[i-1]:
                signals.append(-1)
                signal_strengths.append(70 + (result_df['rsi'].iloc[i-1] - 70))
                signal_descs.append(f"RSI فروش: {result_df['rsi'].iloc[i]:.2f}")
        
        # MACD
        if "MACD" in indicators and all(col in result_df.columns for col in ['macd', 'macd_signal']):
            # سیگنال خرید: MACD crosses above Signal
            if result_df['macd'].iloc[i-1] < result_df['macd_signal'].iloc[i-1] and result_df['macd'].iloc[i] > result_df['macd_signal'].iloc[i]:
                signals.append(1)
                # قدرت سیگنال بر اساس فاصله از صفر
                strength = 70 + min(30, abs(result_df['macd'].iloc[i]) * 100)
                signal_strengths.append(min(100, strength))
                signal_descs.append("تقاطع صعودی MACD")
            # سیگنال فروش: MACD crosses below Signal
            elif result_df['macd'].iloc[i-1] > result_df['macd_signal'].iloc[i-1] and result_df['macd'].iloc[i] < result_df['macd_signal'].iloc[i]:
                signals.append(-1)
                # قدرت سیگنال بر اساس فاصله از صفر
                strength = 70 + min(30, abs(result_df['macd'].iloc[i]) * 100)
                signal_strengths.append(min(100, strength))
                signal_descs.append("تقاطع نزولی MACD")
        
        # Bollinger Bands
        if "Bollinger Bands" in indicators and all(col in result_df.columns for col in ['bb_upper', 'bb_lower']):
            # سیگنال خرید: قیمت زیر باند پایینی
            if result_df['close'].iloc[i] < result_df['bb_lower'].iloc[i]:
                signals.append(1)
                # قدرت سیگنال بر اساس فاصله از باند پایینی
                distance = (result_df['bb_lower'].iloc[i] - result_df['close'].iloc[i]) / result_df['close'].iloc[i] * 100
                signal_strengths.append(70 + min(30, distance * 10))
                signal_descs.append("قیمت زیر باند پایینی بولینگر")
            # سیگنال فروش: قیمت بالای باند بالایی
            elif result_df['close'].iloc[i] > result_df['bb_upper'].iloc[i]:
                signals.append(-1)
                # قدرت سیگنال بر اساس فاصله از باند بالایی
                distance = (result_df['close'].iloc[i] - result_df['bb_upper'].iloc[i]) / result_df['close'].iloc[i] * 100
                signal_strengths.append(70 + min(30, distance * 10))
                signal_descs.append("قیمت بالای باند بالایی بولینگر")
        
        # Supertrend
        if "Supertrend" in indicators and 'supertrend_direction' in result_df.columns:
            # سیگنال خرید: تغییر جهت سوپرترند به صعودی
            if result_df['supertrend_direction'].iloc[i-1] <= 0 and result_df['supertrend_direction'].iloc[i] > 0:
                signals.append(1)
                signal_strengths.append(90)  # سیگنال قوی
                signal_descs.append("سیگنال خرید سوپرترند")
            # سیگنال فروش: تغییر جهت سوپرترند به نزولی
            elif result_df['supertrend_direction'].iloc[i-1] >= 0 and result_df['supertrend_direction'].iloc[i] < 0:
                signals.append(-1)
                signal_strengths.append(90)  # سیگنال قوی
                signal_descs.append("سیگنال فروش سوپرترند")
        
        # ترکیب سیگنال‌ها
        if signals:
            # محاسبه میانگین قدرت سیگنال‌ها
            avg_strength = sum(signal_strengths) / len(signal_strengths)
            
            # تعیین جهت نهایی سیگنال بر اساس میانگین وزنی
            weighted_signal = sum(s * w for s, w in zip(signals, signal_strengths)) / sum(signal_strengths)
            
            if weighted_signal > 0.3:
                result_df['signal'].iloc[i] = 1
                result_df['signal_strength'].iloc[i] = avg_strength
                result_df['signal_desc'].iloc[i] = " | ".join(signal_descs)
            elif weighted_signal < -0.3:
                result_df['signal'].iloc[i] = -1
                result_df['signal_strength'].iloc[i] = avg_strength
                result_df['signal_desc'].iloc[i] = " | ".join(signal_descs)
    
    return result_df

def calculate_rsi(df, period=14):
    """
    محاسبه شاخص قدرت نسبی (RSI)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه

    Returns:
        pd.Series: سری زمانی RSI
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    محاسبه MACD (Moving Average Convergence Divergence)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        fast_period (int): دوره EMA سریع
        slow_period (int): دوره EMA کند
        signal_period (int): دوره خط سیگنال

    Returns:
        tuple: (MACD Line, Signal Line, Histogram)
    """
    # محاسبه EMA سریع و کند
    ema_fast = calculate_ema(df, period=fast_period)
    ema_slow = calculate_ema(df, period=slow_period)
    
    # محاسبه خط MACD
    macd_line = ema_fast - ema_slow
    
    # محاسبه خط سیگنال
    signal_line = macd_line.rolling(window=signal_period).mean()
    
    # محاسبه هیستوگرام
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    محاسبه باندهای بولینگر

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره میانگین متحرک
        std_dev (float): ضریب انحراف معیار

    Returns:
        tuple: (Upper Band, Middle Band, Lower Band)
    """
    # محاسبه میانگین متحرک
    middle_band = df['close'].rolling(window=period).mean()
    
    # محاسبه انحراف معیار
    std = df['close'].rolling(window=period).std()
    
    # محاسبه باندهای بالا و پایین
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_ema(df, period=9, column='close'):
    """
    محاسبه میانگین متحرک نمایی (EMA)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره EMA
        column (str): نام ستون

    Returns:
        pd.Series: سری زمانی EMA
    """
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_sma(df, period=20, column='close'):
    """
    محاسبه میانگین متحرک ساده (SMA)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره SMA
        column (str): نام ستون

    Returns:
        pd.Series: سری زمانی SMA
    """
    return df[column].rolling(window=period).mean()

def calculate_stochastic(df, k_period=14, d_period=3, slowing=3):
    """
    محاسبه اسیلاتور استوکاستیک

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        k_period (int): دوره %K
        d_period (int): دوره %D
        slowing (int): دوره کاهش سرعت

    Returns:
        tuple: (%K, %D)
    """
    # محاسبه بالاترین قیمت در دوره
    high_roll = df['high'].rolling(window=k_period).max()
    
    # محاسبه پایین‌ترین قیمت در دوره
    low_roll = df['low'].rolling(window=k_period).min()
    
    # محاسبه %K
    stoch_k = 100 * ((df['close'] - low_roll) / (high_roll - low_roll))
    
    # محاسبه %K با اعمال دوره کاهش سرعت
    stoch_k = stoch_k.rolling(window=slowing).mean()
    
    # محاسبه %D
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d

def calculate_adx(df, period=14):
    """
    محاسبه شاخص روند جهت‌دار میانگین (ADX)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه

    Returns:
        pd.Series: سری زمانی ADX
    """
    # محاسبه +DM و -DM
    high_diff = df['high'].diff()
    low_diff = df['low'].diff().multiply(-1)
    
    # +DM when high_diff > low_diff and > 0
    plus_dm = pd.Series(0, index=df.index)
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
    
    # -DM when low_diff > high_diff and > 0
    minus_dm = pd.Series(0, index=df.index)
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff
    
    # محاسبه ATR
    atr = calculate_atr(df, period)
    
    # محاسبه +DI و -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # محاسبه DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # محاسبه ADX
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_atr(df, period=14):
    """
    محاسبه میانگین دامنه واقعی (ATR)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه

    Returns:
        pd.Series: سری زمانی ATR
    """
    # محاسبه دامنه واقعی (True Range)
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # محاسبه ATR
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    """
    محاسبه Parabolic SAR

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        af_start (float): ضریب شتاب اولیه
        af_step (float): گام ضریب شتاب
        af_max (float): حداکثر ضریب شتاب

    Returns:
        pd.Series: سری زمانی Parabolic SAR
    """
    # مقداردهی اولیه
    psar = df['close'].copy()
    psar_up = [True] * len(df)
    psar_ep = df['high'].copy()  # Extreme Point
    psar_af = [af_start] * len(df)  # Acceleration Factor
    
    # محاسبه Parabolic SAR
    for i in range(2, len(df)):
        # اگر روند صعودی باشد
        if psar_up[i-1]:
            psar[i] = psar[i-1] + psar_af[i-1] * (psar_ep[i-1] - psar[i-1])
            # محدودیت: SAR نباید از کمترین قیمت دو دوره قبل بیشتر باشد
            psar[i] = min(psar[i], df['low'].iloc[i-2], df['low'].iloc[i-1])
            
            # بررسی تغییر روند
            if df['low'].iloc[i] < psar[i]:
                psar_up[i] = False
                psar[i] = psar_ep[i-1]
                psar_ep[i] = df['low'].iloc[i]
                psar_af[i] = af_start
            else:
                psar_up[i] = True
                # به‌روزرسانی EP اگر قیمت بالاتر باشد
                if df['high'].iloc[i] > psar_ep[i-1]:
                    psar_ep[i] = df['high'].iloc[i]
                    psar_af[i] = min(psar_af[i-1] + af_step, af_max)
                else:
                    psar_ep[i] = psar_ep[i-1]
                    psar_af[i] = psar_af[i-1]
        # اگر روند نزولی باشد
        else:
            psar[i] = psar[i-1] - psar_af[i-1] * (psar[i-1] - psar_ep[i-1])
            # محدودیت: SAR نباید از بیشترین قیمت دو دوره قبل کمتر باشد
            psar[i] = max(psar[i], df['high'].iloc[i-2], df['high'].iloc[i-1])
            
            # بررسی تغییر روند
            if df['high'].iloc[i] > psar[i]:
                psar_up[i] = True
                psar[i] = psar_ep[i-1]
                psar_ep[i] = df['high'].iloc[i]
                psar_af[i] = af_start
            else:
                psar_up[i] = False
                # به‌روزرسانی EP اگر قیمت پایین‌تر باشد
                if df['low'].iloc[i] < psar_ep[i-1]:
                    psar_ep[i] = df['low'].iloc[i]
                    psar_af[i] = min(psar_af[i-1] + af_step, af_max)
                else:
                    psar_ep[i] = psar_ep[i-1]
                    psar_af[i] = psar_af[i-1]
    
    return psar

def calculate_obv(df):
    """
    محاسبه On Balance Volume (OBV)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت

    Returns:
        pd.Series: سری زمانی OBV
    """
    obv = pd.Series(index=df.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_cci(df, period=20):
    """
    محاسبه Commodity Channel Index (CCI)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه

    Returns:
        pd.Series: سری زمانی CCI
    """
    # محاسبه قیمت نمونه (Typical Price)
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # محاسبه میانگین متحرک قیمت نمونه
    tp_sma = tp.rolling(window=period).mean()
    
    # محاسبه انحراف مطلق میانگین
    mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
    
    # محاسبه CCI
    cci = (tp - tp_sma) / (0.015 * mad)
    
    return cci

def calculate_mfi(df, period=14):
    """
    محاسبه Money Flow Index (MFI)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه

    Returns:
        pd.Series: سری زمانی MFI
    """
    # محاسبه قیمت نمونه (Typical Price)
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # محاسبه جریان پول (Money Flow)
    mf = tp * df['volume']
    
    # تشخیص جریان پول مثبت و منفی
    positive_mf = pd.Series(0, index=df.index)
    negative_mf = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        if tp.iloc[i] > tp.iloc[i-1]:
            positive_mf.iloc[i] = mf.iloc[i]
            negative_mf.iloc[i] = 0
        elif tp.iloc[i] < tp.iloc[i-1]:
            positive_mf.iloc[i] = 0
            negative_mf.iloc[i] = mf.iloc[i]
        else:
            positive_mf.iloc[i] = 0
            negative_mf.iloc[i] = 0
    
    # محاسبه جمع جریان پول مثبت و منفی در دوره مشخص
    positive_mf_sum = positive_mf.rolling(window=period).sum()
    negative_mf_sum = negative_mf.rolling(window=period).sum()
    
    # محاسبه نسبت جریان پول
    money_ratio = positive_mf_sum / negative_mf_sum
    
    # محاسبه MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

def calculate_roc(df, period=12, column='close'):
    """
    محاسبه Rate of Change (ROC)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه
        column (str): نام ستون

    Returns:
        pd.Series: سری زمانی ROC
    """
    # محاسبه درصد تغییر نسبت به n دوره قبل
    roc = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
    
    return roc

def calculate_williams_r(df, period=14):
    """
    محاسبه Williams %R

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه

    Returns:
        pd.Series: سری زمانی Williams %R
    """
    # محاسبه بالاترین قیمت در دوره
    highest_high = df['high'].rolling(window=period).max()
    
    # محاسبه پایین‌ترین قیمت در دوره
    lowest_low = df['low'].rolling(window=period).min()
    
    # محاسبه Williams %R
    williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    
    return williams_r

def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """
    محاسبه Ichimoku Cloud

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        tenkan_period (int): دوره تنکان-سن (خط تبدیل)
        kijun_period (int): دوره کیجون-سن (خط پایه)
        senkou_b_period (int): دوره سنکو اسپن B (خط تأخیری B)
        displacement (int): دوره جابجایی (از جمله برای چیکو اسپن)

    Returns:
        tuple: (تنکان-سن, کیجون-سن, سنکو اسپن A, سنکو اسپن B, چیکو اسپن)
    """
    # محاسبه تنکان-سن (Conversion Line)
    tenkan_sen = (df['high'].rolling(window=tenkan_period).max() + df['low'].rolling(window=tenkan_period).min()) / 2
    
    # محاسبه کیجون-سن (Base Line)
    kijun_sen = (df['high'].rolling(window=kijun_period).max() + df['low'].rolling(window=kijun_period).min()) / 2
    
    # محاسبه سنکو اسپن A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # محاسبه سنکو اسپن B (Leading Span B)
    senkou_span_b = ((df['high'].rolling(window=senkou_b_period).max() + df['low'].rolling(window=senkou_b_period).min()) / 2).shift(displacement)
    
    # محاسبه چیکو اسپن (Lagging Span)
    chikou_span = df['close'].shift(-displacement)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_support_resistance(df, period=14, deviation=0.01):
    """
    محاسبه سطوح حمایت و مقاومت

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره محاسبه
        deviation (float): انحراف مجاز

    Returns:
        tuple: (سطح حمایت, سطح مقاومت)
    """
    # مقداردهی اولیه
    support = pd.Series(index=df.index)
    resistance = pd.Series(index=df.index)
    
    # محاسبه سطوح حمایت و مقاومت برای هر نقطه
    for i in range(period, len(df)):
        # دوره فعلی
        window = df.iloc[i-period:i]
        
        # یافتن کف‌ها
        lows = window['low']
        lowest = lows.min()
        support.iloc[i] = lowest
        
        # یافتن سقف‌ها
        highs = window['high']
        highest = highs.max()
        resistance.iloc[i] = highest
    
    return support, resistance

def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    محاسبه Supertrend

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        period (int): دوره ATR
        multiplier (float): ضریب ATR

    Returns:
        tuple: (خط سوپرترند، جهت)
    """
    # محاسبه ATR
    atr = calculate_atr(df, period)
    
    # محاسبه باندهای بالا و پایین
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # مقداردهی اولیه سوپرترند
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1: صعودی (باند پایینی), -1: نزولی (باند بالایی)
    
    # محاسبه سوپرترند
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]
        
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lowerband.iloc[i]
        else:
            supertrend.iloc[i] = upperband.iloc[i]
    
    return supertrend, direction

def calculate_vwap(df):
    """
    محاسبه Volume Weighted Average Price (VWAP)

    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت

    Returns:
        pd.Series: سری زمانی VWAP
    """
    # محاسبه قیمت نمونه (Typical Price)
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # محاسبه حجم * قیمت
    pv = tp * df['volume']
    
    # محاسبه VWAP
    vwap = pv.cumsum() / df['volume'].cumsum()
    
    return vwap

def recommend_indicators_for_market(df, market_type=None, top_n=5):
    """
    پیشنهاد بهترین اندیکاتورها برای شرایط فعلی بازار
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        market_type (str, optional): نوع بازار (trending, ranging, volatile)
        top_n (int): تعداد اندیکاتورهای پیشنهادی
        
    Returns:
        list: لیست اندیکاتورهای پیشنهادی
    """
    # تشخیص خودکار نوع بازار اگر مشخص نشده باشد
    if market_type is None:
        # محاسبه پارامترهای مهم برای تشخیص نوع بازار
        
        # محاسبه نوسان (volatility)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # نوسان سالانه
        
        # محاسبه ADX برای تشخیص روند
        adx = calculate_adx(df)
        adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        
        # تشخیص نوع بازار
        if adx_value > 25:  # بازار روندی
            market_type = 'trending'
        elif volatility > 0.03:  # بازار پرنوسان
            market_type = 'volatile'
        else:  # بازار محدوده
            market_type = 'ranging'
    
    # اندیکاتورهای مناسب بر اساس نوع بازار
    if market_type == 'trending':
        # برای بازارهای روندی
        indicator_scores = {
            "Moving Average": 95,
            "MACD": 90,
            "Supertrend": 85,
            "ADX": 80,
            "Ichimoku": 75,
            "OBV": 70,
            "Parabolic SAR": 65,
            "VWAP": 65,
            "EMA": 60,
            "RSI": 55
        }
    elif market_type == 'ranging':
        # برای بازارهای محدوده
        indicator_scores = {
            "RSI": 95,
            "Bollinger Bands": 90,
            "Stochastic": 85,
            "Williams %R": 80,
            "CCI": 75,
            "MFI": 70,
            "Support/Resistance": 65,
            "MACD": 60,
            "Volume": 55,
            "ATR": 50
        }
    elif market_type == 'volatile':
        # برای بازارهای پرنوسان
        indicator_scores = {
            "Bollinger Bands": 95,
            "ATR": 90,
            "Parabolic SAR": 85,
            "Supertrend": 80,
            "VWAP": 75,
            "RSI": 70,
            "MACD": 65,
            "EMA": 60,
            "Volume": 55,
            "Support/Resistance": 50
        }
    else:
        # پیش‌فرض
        indicator_scores = {
            "RSI": 80,
            "MACD": 80,
            "Bollinger Bands": 80,
            "Moving Average": 75,
            "Stochastic": 75,
            "ADX": 70,
            "Supertrend": 70,
            "Volume": 65,
            "ATR": 65,
            "Ichimoku": 60
        }
    
    # مرتب‌سازی اندیکاتورها بر اساس امتیاز
    sorted_indicators = sorted(indicator_scores.items(), key=lambda x: x[1], reverse=True)
    
    # انتخاب N اندیکاتور برتر
    top_indicators = [ind[0] for ind in sorted_indicators[:top_n]]
    
    return top_indicators