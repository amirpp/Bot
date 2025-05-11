"""
ماژول تحلیل تکنیکال برای بازار ارزهای دیجیتال

این ماژول شامل توابع و کلاس‌های مورد نیاز برای انجام تحلیل تکنیکال روی داده‌های قیمت
و تولید سیگنال‌های معاملاتی است.
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# وارد کردن اندیکاتورهای پیشرفته
from advanced_indicators import AdvancedIndicators

# تنظیم لاگر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# لیست اندیکاتورهای در دسترس
AVAILABLE_INDICATORS = [
    "RSI", "MACD", "Bollinger Bands", "Moving Average", "Stochastic", 
    "CCI", "ATR", "OBV", "Ichimoku", "SuperTrend", "Volume", 
    "MFI", "VWAP", "Fibonacci", "Pivot Points", "Elder Ray",
    "Parabolic SAR", "ADX", "Williams %R", "Chaikin Money Flow", "Awesome Oscillator",
    "TRIX", "Donchian Channel", "Hull Moving Average", "KAMA", "ZigZag", 
    "MESA", "Rainbow MA", "Elder Force Index", "PPO", "AROON",
    "Heikin-Ashi", "Guppy Multiple Moving Average", "Coppock Curve", "Momentum", "ROC"
]

# اندیکاتورهای برتر برای استفاده پیش‌فرض
TOP_INDICATORS = [
    "RSI", "MACD", "Bollinger Bands", "Moving Average", "Stochastic", 
    "ATR", "OBV", "Ichimoku", "SuperTrend", "Volume", "ADX", "MFI"
]

def perform_technical_analysis(df: pd.DataFrame, selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """
    انجام تحلیل تکنیکال روی داده‌های قیمت
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        selected_indicators (list, optional): لیست اندیکاتورهای انتخاب شده
        
    Returns:
        pd.DataFrame: دیتافریم با اندیکاتورهای اضافه شده
    """
    if df is None or df.empty:
        logger.error("دیتافریم خالی یا None برای تحلیل تکنیکال")
        return df
    
    try:
        # ایجاد یک کپی از دیتافریم
        df_copy = df.copy()
        
        # اگر اندیکاتور خاصی انتخاب نشده، از لیست پیش‌فرض استفاده می‌کنیم
        if selected_indicators is None:
            selected_indicators = TOP_INDICATORS
            
        # ایجاد نمونه از کلاس اندیکاتورهای پیشرفته
        adv_indicators = AdvancedIndicators()
        
        # محاسبه اندیکاتورهای انتخاب شده
        for indicator in selected_indicators:
            if indicator == "RSI":
                # محاسبه RSI
                df_copy['rsi'] = ta.momentum.RSIIndicator(df_copy['close'], window=14).rsi()
                
            elif indicator == "MACD":
                # محاسبه MACD
                macd = ta.trend.MACD(df_copy['close'], window_slow=26, window_fast=12, window_sign=9)
                df_copy['macd'] = macd.macd()
                df_copy['macd_signal'] = macd.macd_signal()
                df_copy['macd_diff'] = macd.macd_diff()
                
            elif indicator == "Bollinger Bands":
                # محاسبه باندهای بولینگر
                bollinger = ta.volatility.BollingerBands(df_copy['close'], window=20, window_dev=2)
                df_copy['bb_upper'] = bollinger.bollinger_hband()
                df_copy['bb_middle'] = bollinger.bollinger_mavg()
                df_copy['bb_lower'] = bollinger.bollinger_lband()
                df_copy['bb_width'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle']
                
            elif indicator == "Moving Average":
                # محاسبه میانگین‌های متحرک
                df_copy['sma_20'] = ta.trend.SMAIndicator(df_copy['close'], window=20).sma_indicator()
                df_copy['sma_50'] = ta.trend.SMAIndicator(df_copy['close'], window=50).sma_indicator()
                df_copy['sma_200'] = ta.trend.SMAIndicator(df_copy['close'], window=200).sma_indicator()
                df_copy['ema_20'] = ta.trend.EMAIndicator(df_copy['close'], window=20).ema_indicator()
                df_copy['ema_50'] = ta.trend.EMAIndicator(df_copy['close'], window=50).ema_indicator()
                df_copy['ema_200'] = ta.trend.EMAIndicator(df_copy['close'], window=200).ema_indicator()
                
            elif indicator == "Stochastic":
                # محاسبه Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(df_copy['high'], df_copy['low'], df_copy['close'], window=14, smooth_window=3)
                df_copy['stoch_k'] = stoch.stoch()
                df_copy['stoch_d'] = stoch.stoch_signal()
                
            elif indicator == "CCI":
                # محاسبه Commodity Channel Index
                df_copy['cci'] = ta.trend.CCIIndicator(df_copy['high'], df_copy['low'], df_copy['close'], window=20).cci()
                
            elif indicator == "ATR":
                # محاسبه Average True Range
                df_copy['atr'] = ta.volatility.AverageTrueRange(df_copy['high'], df_copy['low'], df_copy['close'], window=14).average_true_range()
                
            elif indicator == "OBV":
                # محاسبه On-Balance Volume
                df_copy['obv'] = ta.volume.OnBalanceVolumeIndicator(df_copy['close'], df_copy['volume']).on_balance_volume()
                
            elif indicator == "Ichimoku":
                # محاسبه Ichimoku Cloud
                ichimoku = ta.trend.IchimokuIndicator(df_copy['high'], df_copy['low'], window1=9, window2=26, window3=52)
                df_copy['ichimoku_a'] = ichimoku.ichimoku_a()
                df_copy['ichimoku_b'] = ichimoku.ichimoku_b()
                df_copy['ichimoku_base'] = ichimoku.ichimoku_base_line()
                df_copy['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
                
            elif indicator == "SuperTrend":
                # محاسبه SuperTrend
                df_copy['supertrend'], df_copy['supertrend_direction'] = adv_indicators.calculate_supertrend(df_copy, period=10, multiplier=3.0)
                
            elif indicator == "MFI":
                # محاسبه Money Flow Index
                df_copy['mfi'] = ta.volume.MFIIndicator(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14).money_flow_index()
                
            elif indicator == "VWAP":
                # محاسبه Volume Weighted Average Price
                df_copy['vwap'] = adv_indicators.calculate_vwap(df_copy)
                
            elif indicator == "ADX":
                # محاسبه Average Directional Index
                adx = ta.trend.ADXIndicator(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
                df_copy['adx'] = adx.adx()
                df_copy['adx_pos'] = adx.adx_pos()
                df_copy['adx_neg'] = adx.adx_neg()
                
            elif indicator == "Williams %R":
                # محاسبه Williams %R
                df_copy['williams_r'] = ta.momentum.WilliamsRIndicator(df_copy['high'], df_copy['low'], df_copy['close'], lbp=14).williams_r()
                
            elif indicator == "Chaikin Money Flow":
                # محاسبه Chaikin Money Flow
                df_copy['cmf'] = adv_indicators.calculate_cmf(df_copy, period=20)
                
            elif indicator == "Elder Ray":
                # محاسبه Elder Ray Index
                df_copy['bull_power'], df_copy['bear_power'] = adv_indicators.calculate_elder_ray(df_copy, period=13)
                
            elif indicator == "Pivot Points":
                # محاسبه Pivot Points
                pivot_df = adv_indicators.calculate_pivot_points(df_copy, method='standard')
                # ادغام دیتافریم‌ها
                for col in pivot_df.columns:
                    df_copy[f'pivot_{col}'] = pivot_df[col]
                
            elif indicator == "Donchian Channel":
                # محاسبه Donchian Channel
                df_copy['dc_upper'], df_copy['dc_middle'], df_copy['dc_lower'] = adv_indicators.calculate_donchian_channel(df_copy, period=20)
        
        # اضافه کردن محاسبه اندیکاتورهای ترکیبی
        if "Moving Average" in selected_indicators:
            # اضافه کردن سیگنال Golden Cross / Death Cross
            if 'sma_50' in df_copy.columns and 'sma_200' in df_copy.columns:
                df_copy['golden_cross'] = (df_copy['sma_50'] > df_copy['sma_200']) & (df_copy['sma_50'].shift(1) <= df_copy['sma_200'].shift(1))
                df_copy['death_cross'] = (df_copy['sma_50'] < df_copy['sma_200']) & (df_copy['sma_50'].shift(1) >= df_copy['sma_200'].shift(1))
        
        # پاکسازی ردیف‌های با مقادیر NaN
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        
        return df_copy
        
    except Exception as e:
        logger.error(f"خطا در انجام تحلیل تکنیکال: {str(e)}")
        return df

def generate_signals(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    تولید سیگنال‌های معاملاتی بر اساس اندیکاتورهای محاسبه شده
    همراه با قیمت‌های هدف و حد ضرر
    
    Args:
        df (pd.DataFrame): دیتافریم با اندیکاتورهای محاسبه شده
        
    Returns:
        dict: دیکشنری سیگنال‌ها
    """
    if df is None or df.empty:
        logger.error("دیتافریم خالی یا None برای تولید سیگنال")
        return {}
    
    try:
        signals = {}
        current_price = df['close'].iloc[-1]
        
        # سیگنال RSI
        if 'rsi' in df.columns:
            rsi_value = df['rsi'].iloc[-1]
            
            if rsi_value < 30:
                signal = "BUY"
                strength = 80
                description = f"RSI در ناحیه اشباع فروش (RSI = {rsi_value:.2f})"
                # تنظیم قیمت‌های هدف و حد ضرر برای سیگنال خرید RSI
                target_price = current_price * 1.05  # هدف: 5% بالاتر از قیمت فعلی
                stop_loss = current_price * 0.97  # حد ضرر: 3% پایین‌تر از قیمت فعلی
            elif rsi_value > 70:
                signal = "SELL"
                strength = 80
                description = f"RSI در ناحیه اشباع خرید (RSI = {rsi_value:.2f})"
                # تنظیم قیمت‌های هدف و حد ضرر برای سیگنال فروش RSI
                target_price = current_price * 0.95  # هدف: 5% پایین‌تر از قیمت فعلی
                stop_loss = current_price * 1.03  # حد ضرر: 3% بالاتر از قیمت فعلی
            else:
                signal = "NEUTRAL"
                strength = 50
                description = f"RSI در محدوده خنثی (RSI = {rsi_value:.2f})"
                target_price = None
                stop_loss = None
                
            signals["RSI"] = {
                "signal": signal,
                "strength": strength,
                "description": description,
                "value": rsi_value,
                "target_price": target_price,
                "stop_loss": stop_loss
            }
        
        # سیگنال MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_value = df['macd'].iloc[-1]
            macd_signal_value = df['macd_signal'].iloc[-1]
            
            if macd_value > macd_signal_value and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
                signal = "BUY"
                strength = 75
                description = "MACD قطع صعودی خط سیگنال"
            elif macd_value < macd_signal_value and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
                signal = "SELL"
                strength = 75
                description = "MACD قطع نزولی خط سیگنال"
            else:
                if macd_value > macd_signal_value:
                    signal = "BUY"
                    strength = 60
                    description = "MACD بالاتر از خط سیگنال"
                elif macd_value < macd_signal_value:
                    signal = "SELL"
                    strength = 60
                    description = "MACD پایین‌تر از خط سیگنال"
                else:
                    signal = "NEUTRAL"
                    strength = 50
                    description = "MACD برابر با خط سیگنال"
                    
            # تنظیم قیمت‌های هدف و حد ضرر برای MACD
            if signal == "BUY":
                target_price = current_price * 1.04  # هدف: 4% بالاتر از قیمت فعلی
                stop_loss = current_price * 0.98     # حد ضرر: 2% پایین‌تر از قیمت فعلی
            elif signal == "SELL":
                target_price = current_price * 0.96  # هدف: 4% پایین‌تر از قیمت فعلی
                stop_loss = current_price * 1.02     # حد ضرر: 2% بالاتر از قیمت فعلی
            else:
                target_price = None
                stop_loss = None
                
            signals["MACD"] = {
                "signal": signal,
                "strength": strength,
                "description": description,
                "value": {
                    "macd": macd_value,
                    "signal": macd_signal_value
                },
                "target_price": target_price,
                "stop_loss": stop_loss
            }
            
        # سیگنال Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            close = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            if close > bb_upper:
                signal = "SELL"
                strength = 70
                description = "قیمت بالاتر از باند بالایی بولینگر"
            elif close < bb_lower:
                signal = "BUY"
                strength = 70
                description = "قیمت پایین‌تر از باند پایینی بولینگر"
            else:
                signal = "NEUTRAL"
                strength = 50
                description = "قیمت داخل باندهای بولینگر"
                
            # بررسی باریک شدن/گسترش باندها
            if 'bb_width' in df.columns:
                bb_width = df['bb_width'].iloc[-1]
                bb_width_prev = df['bb_width'].iloc[-2] if len(df) > 1 else bb_width
                
                if bb_width < bb_width_prev:
                    description += " - باندها در حال باریک شدن هستند (احتمال حرکت قوی)"
                else:
                    description += " - باندها در حال گسترش هستند"
                    
            # تنظیم قیمت‌های هدف و حد ضرر برای Bollinger Bands
            if signal == "BUY":
                # در سیگنال خرید هدف اول باند میانی و هدف دوم باند بالایی است
                bb_middle = (bb_upper + bb_lower) / 2
                target_price = bb_middle  # هدف: میانگین باندها
                stop_loss = close * 0.98  # حد ضرر: 2% پایین‌تر از قیمت فعلی
            elif signal == "SELL":
                # در سیگنال فروش هدف اول باند میانی و هدف دوم باند پایینی است
                bb_middle = (bb_upper + bb_lower) / 2
                target_price = bb_middle  # هدف: میانگین باندها
                stop_loss = close * 1.02  # حد ضرر: 2% بالاتر از قیمت فعلی
            else:
                target_price = None
                stop_loss = None
                
            signals["Bollinger Bands"] = {
                "signal": signal,
                "strength": strength,
                "description": description,
                "value": {
                    "upper": bb_upper,
                    "lower": bb_lower,
                    "close": close
                },
                "target_price": target_price,
                "stop_loss": stop_loss
            }
            
        # سیگنال Moving Averages
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            close = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            if close > sma_20 and close > sma_50:
                signal = "BUY"
                strength = 65
                description = "قیمت بالاتر از هر دو میانگین 20 و 50"
            elif close < sma_20 and close < sma_50:
                signal = "SELL"
                strength = 65
                description = "قیمت پایین‌تر از هر دو میانگین 20 و 50"
            elif close > sma_20 and close < sma_50:
                signal = "NEUTRAL"
                strength = 55
                description = "قیمت بین میانگین‌های 20 و 50 (بالاتر از 20)"
            elif close < sma_20 and close > sma_50:
                signal = "NEUTRAL"
                strength = 45
                description = "قیمت بین میانگین‌های 20 و 50 (پایین‌تر از 20)"
            else:
                signal = "NEUTRAL"
                strength = 50
                description = "وضعیت نامشخص میانگین‌های متحرک"
                
            # بررسی تقاطع میانگین‌ها
            if 'golden_cross' in df.columns and 'death_cross' in df.columns:
                if df['golden_cross'].iloc[-1]:
                    signal = "BUY"
                    strength = 85
                    description = "تقاطع طلایی - میانگین 50 از بالا میانگین 200 را قطع کرد"
                elif df['death_cross'].iloc[-1]:
                    signal = "SELL"
                    strength = 85
                    description = "تقاطع مرگ - میانگین 50 از پایین میانگین 200 را قطع کرد"
                    
            signals["Moving Average"] = {
                "signal": signal,
                "strength": strength,
                "description": description,
                "value": {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "close": close
                }
            }
            
        # سیگنال Stochastic
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            
            if stoch_k < 20 and stoch_d < 20:
                signal = "BUY"
                strength = 70
                description = f"استوکاستیک در ناحیه اشباع فروش (K = {stoch_k:.2f}, D = {stoch_d:.2f})"
            elif stoch_k > 80 and stoch_d > 80:
                signal = "SELL"
                strength = 70
                description = f"استوکاستیک در ناحیه اشباع خرید (K = {stoch_k:.2f}, D = {stoch_d:.2f})"
            elif stoch_k > stoch_d and df['stoch_k'].iloc[-2] <= df['stoch_d'].iloc[-2]:
                signal = "BUY"
                strength = 60
                description = "تقاطع صعودی %K و %D"
            elif stoch_k < stoch_d and df['stoch_k'].iloc[-2] >= df['stoch_d'].iloc[-2]:
                signal = "SELL"
                strength = 60
                description = "تقاطع نزولی %K و %D"
            else:
                signal = "NEUTRAL"
                strength = 50
                description = f"استوکاستیک در ناحیه خنثی (K = {stoch_k:.2f}, D = {stoch_d:.2f})"
                
            signals["Stochastic"] = {
                "signal": signal,
                "strength": strength,
                "description": description,
                "value": {
                    "k": stoch_k,
                    "d": stoch_d
                }
            }
            
        # سیگنال SuperTrend
        if 'supertrend' in df.columns and 'supertrend_direction' in df.columns:
            supertrend_direction = df['supertrend_direction'].iloc[-1]
            supertrend_value = df['supertrend'].iloc[-1]
            close = df['close'].iloc[-1]
            
            if supertrend_direction == 1:  # Uptrend (Bullish)
                signal = "BUY"
                strength = 80
                description = f"SuperTrend در روند صعودی (قیمت = {close:.2f}, SuperTrend = {supertrend_value:.2f})"
            else:  # Downtrend (Bearish)
                signal = "SELL"
                strength = 80
                description = f"SuperTrend در روند نزولی (قیمت = {close:.2f}, SuperTrend = {supertrend_value:.2f})"
                
            signals["SuperTrend"] = {
                "signal": signal,
                "strength": strength,
                "description": description,
                "value": {
                    "direction": int(supertrend_direction),
                    "value": supertrend_value,
                    "close": close
                }
            }
            
        # سیگنال ADX
        if 'adx' in df.columns and 'adx_pos' in df.columns and 'adx_neg' in df.columns:
            adx_value = df['adx'].iloc[-1]
            adx_pos = df['adx_pos'].iloc[-1]
            adx_neg = df['adx_neg'].iloc[-1]
            
            if adx_value > 25:
                if adx_pos > adx_neg:
                    signal = "BUY"
                    strength = 70
                    description = f"ADX بالا با روند صعودی (ADX = {adx_value:.2f})"
                else:
                    signal = "SELL"
                    strength = 70
                    description = f"ADX بالا با روند نزولی (ADX = {adx_value:.2f})"
            else:
                signal = "NEUTRAL"
                strength = 50
                description = f"ADX پایین - روند ضعیف (ADX = {adx_value:.2f})"
                
            signals["ADX"] = {
                "signal": signal,
                "strength": strength,
                "description": description,
                "value": {
                    "adx": adx_value,
                    "adx_pos": adx_pos,
                    "adx_neg": adx_neg
                }
            }
            
        # سیگنال‌های ترکیبی
        # می‌توانید سیگنال‌های ترکیبی پیشرفته‌تری را اینجا اضافه کنید
        
        return signals
        
    except Exception as e:
        logger.error(f"خطا در تولید سیگنال‌ها: {str(e)}")
        return {}
