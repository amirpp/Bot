"""
ماژول تحلیل تکنیکال ارزهای دیجیتال

این ماژول شامل توابع محاسبه اندیکاتورهای تکنیکال و تولید سیگنال‌های معاملاتی است.
"""

import pandas as pd
import numpy as np
import streamlit as st
import ta
from advanced_indicators import AdvancedIndicators

# لیست اندیکاتورهای موجود
AVAILABLE_INDICATORS = [
    'RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'Supertrend',
    'ADX', 'ATR', 'EMA', 'SMA', 'VWAP', 'Volume', 'OBV', 'CMF',
    'Ichimoku Cloud', 'Elder Ray', 'Fibonacci Levels', 'Pivot Points',
    'Williams %R', 'Donchian Channel', 'Heikin Ashi', 'Zigzag', 'Fisher Transform'
]

# لیست اندیکاتورهای برتر برای تحلیل با دقت بالا
TOP_INDICATORS = [
    # مومنتوم
    'RSI', 'Stochastic', 'MACD', 'CCI', 'Williams %R', 'Ultimate Oscillator', 'Awesome Oscillator',
    'PPO', 'MFI', 'Stochastic RSI',
    
    # روند
    'ADX', 'EMA', 'SMA', 'Supertrend', 'DEMA', 'TEMA', 'Parabolic SAR', 'Linear Regression Slope',
    'Ichimoku Cloud', 'HMA',
    
    # نوسان
    'Bollinger Bands', 'ATR', 'Keltner Channel', 'Donchian Channel', 'Chandelier Exit', 'Volatility Stop',
    
    # حجم
    'OBV', 'Volume', 'CMF', 'MFI', 'VWAP', "Elder's Force Index",
    
    # سایکل و الگو
    'ZigZag', 'Gann Swing', 'Hurst Exponent', 'Ehlers Sine Wave', 'Hilbert Transform', 'Cycle Indicators',
    'Fourier Transform',
    
    # شاخص‌های پیشرفته
    'Fisher Transform', 'Coppock Curve', 'KST Oscillator', 'Elder Ray', 'Aroon', 'TRIX', 'Wave Trend',
    
    # مدرن
    'Squeeze Momentum', 'Vortex Indicator', 'TTM Trend', 'Relative Vigor Index', 'Elder Impulse System',
    'Kaufman Adaptive Moving Average', 'McGinley Dynamic',
    
    # هارمونیک
    'Harmonic Patterns', 'Gartley Pattern', 'Butterfly Pattern', 'Crab Pattern', 'Bat Pattern', 'Shark Pattern',
    'Cypher Pattern',
    
    # زمانی
    'Time Series Forecast', 'Time Segmented Volume', 'Market Time', 'Time at Mode', 'Time Price Opportunities'
]

def perform_technical_analysis(df, selected_indicators):
    """
    محاسبه اندیکاتورهای فنی روی دیتافریم قیمت
    
    Args:
        df (pd.DataFrame): دیتافریم با ستون‌های OHLCV
        selected_indicators (list): لیست اندیکاتورهای انتخاب شده
        
    Returns:
        pd.DataFrame: دیتافریم با اندیکاتورهای محاسبه شده
    """
    # ایجاد یک کپی از دیتافریم اصلی
    df_copy = df.copy()
    
    # بررسی وجود ستون‌های لازم
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df_copy.columns:
            st.error(f"ستون {col} در دیتافریم وجود ندارد")
            return df_copy
    
    # محاسبه اندیکاتورهای انتخاب شده
    
    # 1. RSI - شاخص قدرت نسبی
    if 'RSI' in selected_indicators or not selected_indicators:
        df_copy['rsi'] = ta.momentum.RSIIndicator(df_copy['close']).rsi()
    
    # 2. MACD - واگرایی و همگرایی میانگین متحرک
    if 'MACD' in selected_indicators or not selected_indicators:
        macd = ta.trend.MACD(df_copy['close'])
        df_copy['macd'] = macd.macd()
        df_copy['macd_signal'] = macd.macd_signal()
        df_copy['macd_diff'] = macd.macd_diff()
    
    # 3. Bollinger Bands - باندهای بولینگر
    if 'Bollinger Bands' in selected_indicators or not selected_indicators:
        bb = ta.volatility.BollingerBands(df_copy['close'])
        df_copy['bb_upper'] = bb.bollinger_hband()
        df_copy['bb_lower'] = bb.bollinger_lband()
        df_copy['bb_middle'] = bb.bollinger_mavg()
        df_copy['bb_width'] = bb.bollinger_wband()
        df_copy['bb_pct'] = (df_copy['close'] - df_copy['bb_lower']) / (df_copy['bb_upper'] - df_copy['bb_lower'])
    
    # 4. Stochastic Oscillator - نوسان‌گر استوکاستیک
    if 'Stochastic' in selected_indicators or not selected_indicators:
        stoch = ta.momentum.StochasticOscillator(df_copy['high'], df_copy['low'], df_copy['close'])
        df_copy['stoch_k'] = stoch.stoch()
        df_copy['stoch_d'] = stoch.stoch_signal()
    
    # 5. Supertrend
    if 'Supertrend' in selected_indicators or not selected_indicators:
        df_copy['supertrend'], df_copy['supertrend_direction'] = AdvancedIndicators.calculate_supertrend(df_copy)
    
    # 6. ADX - شاخص حرکت جهت‌دار میانگین
    if 'ADX' in selected_indicators or not selected_indicators:
        adx = ta.trend.ADXIndicator(df_copy['high'], df_copy['low'], df_copy['close'])
        df_copy['adx'] = adx.adx()
        df_copy['adx_pos'] = adx.adx_pos()
        df_copy['adx_neg'] = adx.adx_neg()
    
    # 7. ATR - میانگین دامنه واقعی
    if 'ATR' in selected_indicators or not selected_indicators:
        df_copy['atr'] = ta.volatility.AverageTrueRange(df_copy['high'], df_copy['low'], df_copy['close']).average_true_range()
    
    # 8. EMA - میانگین متحرک نمایی
    if 'EMA' in selected_indicators or not selected_indicators:
        df_copy['ema'] = ta.trend.EMAIndicator(df_copy['close']).ema_indicator()
        df_copy['ema50'] = ta.trend.EMAIndicator(df_copy['close'], window=50).ema_indicator()
        df_copy['ema200'] = ta.trend.EMAIndicator(df_copy['close'], window=200).ema_indicator()
    
    # 9. SMA - میانگین متحرک ساده
    if 'SMA' in selected_indicators or not selected_indicators:
        df_copy['sma'] = ta.trend.SMAIndicator(df_copy['close'], window=20).sma_indicator()  # SMA با پیش‌فرض 20 دوره
        df_copy['sma50'] = ta.trend.SMAIndicator(df_copy['close'], window=50).sma_indicator()
        df_copy['sma200'] = ta.trend.SMAIndicator(df_copy['close'], window=200).sma_indicator()
    
    # 10. VWAP - میانگین متحرک وزنی حجم
    if 'VWAP' in selected_indicators or not selected_indicators:
        df_copy['vwap'] = AdvancedIndicators.calculate_vwap(df_copy)
    
    # 11. Volume - حجم
    if 'Volume' in selected_indicators or not selected_indicators:
        df_copy['volume_sma'] = ta.trend.SMAIndicator(df_copy['volume'], window=20).sma_indicator()
        df_copy['volume_ema'] = ta.trend.EMAIndicator(df_copy['volume'], window=20).ema_indicator()
    
    # 12. OBV - حجم در تعادل
    if 'OBV' in selected_indicators or not selected_indicators:
        df_copy['obv'] = ta.volume.OnBalanceVolumeIndicator(df_copy['close'], df_copy['volume']).on_balance_volume()
    
    # 13. CMF - جریان پول چایکین
    if 'CMF' in selected_indicators or not selected_indicators:
        df_copy['cmf'] = AdvancedIndicators.calculate_cmf(df_copy)
    
    # 14. Ichimoku Cloud - ابر ایچیموکو
    if 'Ichimoku Cloud' in selected_indicators or not selected_indicators:
        ichimoku = ta.trend.IchimokuIndicator(df_copy['high'], df_copy['low'], window1=9, window2=26, window3=52)
        df_copy['ichimoku_a'] = ichimoku.ichimoku_a()
        df_copy['ichimoku_b'] = ichimoku.ichimoku_b()
        df_copy['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df_copy['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    
    # 15. Elder Ray
    if 'Elder Ray' in selected_indicators or not selected_indicators:
        df_copy['elder_bull'], df_copy['elder_bear'] = AdvancedIndicators.calculate_elder_ray(df_copy)
    
    # 16. Fibonacci Levels
    if 'Fibonacci Levels' in selected_indicators or not selected_indicators:
        fib_levels = AdvancedIndicators.calculate_fibonacci_retracement(df_copy)
        for col in fib_levels.columns:
            df_copy[f'fib_{col}'] = fib_levels[col]
    
    # 17. Pivot Points
    if 'Pivot Points' in selected_indicators or not selected_indicators:
        pivots = AdvancedIndicators.calculate_pivot_points(df_copy)
        for col in pivots.columns:
            df_copy[f'pivot_{col}'] = pivots[col]
    
    # 18. Williams %R
    if 'Williams %R' in selected_indicators or not selected_indicators:
        df_copy['williams_r'] = ta.momentum.WilliamsRIndicator(df_copy['high'], df_copy['low'], df_copy['close']).williams_r()
    
    # 19. Donchian Channel
    if 'Donchian Channel' in selected_indicators or not selected_indicators:
        df_copy['donchian_upper'], df_copy['donchian_middle'], df_copy['donchian_lower'] = AdvancedIndicators.calculate_donchian_channel(df_copy)
    
    # 20. Heikin Ashi
    if 'Heikin Ashi' in selected_indicators or not selected_indicators:
        ha = AdvancedIndicators.calculate_heikin_ashi(df_copy)
        for col in ha.columns:
            df_copy[col] = ha[col]
    
    # 21. ZigZag
    if 'Zigzag' in selected_indicators or not selected_indicators:
        df_copy['zigzag'] = AdvancedIndicators.calculate_zigzag(df_copy)
    
    # 22. Fisher Transform
    if 'Fisher Transform' in selected_indicators or not selected_indicators:
        df_copy['fisher'] = AdvancedIndicators.calculate_fisher_transform(df_copy)
    
    # اضافه کردن سایر اندیکاتورهای پیشرفته در صورت نیاز
    if len(selected_indicators) > 20 or len(selected_indicators) == 0:
        # یک نمونه از AdvancedIndicators ایجاد کنید و اندیکاتورهای پیشرفته‌تر را محاسبه کنید
        advanced_df = AdvancedIndicators.calculate_advanced_indicators(df_copy)
        
        # ترکیب نتایج با دیتافریم اصلی
        for col in advanced_df.columns:
            if col not in df_copy.columns:
                df_copy[col] = advanced_df[col]
    
    return df_copy

def generate_signals(df, selected_indicators, threshold=70, risk_level='balanced'):
    """
    تولید سیگنال‌های معاملاتی بر اساس اندیکاتورهای محاسبه شده همراه با اهداف قیمتی
    
    Args:
        df (pd.DataFrame): دیتافریم با اندیکاتورهای محاسبه شده
        selected_indicators (list): لیست اندیکاتورهای انتخابی
        threshold (int): حداقل قدرت سیگنال برای تولید سیگنال (0-100)
        risk_level (str): سطح ریسک ('conservative', 'balanced', 'aggressive')
        
    Returns:
        tuple: (نوع سیگنال، توضیحات سیگنال، قدرت سیگنال، اهداف قیمتی)
    """
    try:
        # اطمینان از وجود داده کافی
        if df.empty or len(df) < 2:
            return "NEUTRAL", "داده‌های کافی برای تحلیل وجود ندارد", 0, None
        
        # استفاده از داده‌های اخیر برای تحلیل
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # شمارش سیگنال‌های صعودی و نزولی
        bullish_signals = []
        bearish_signals = []
        neutral_signals = []
        
        # دیکشنری برای نگهداری اطلاعات اهداف قیمتی و حد ضرر
        price_targets = {}
        
        # تحلیل اندیکاتورها
        
        # 1. RSI
        if 'RSI' in selected_indicators and 'rsi' in df.columns:
            if not pd.isna(current['rsi']):
                if current['rsi'] < 30:
                    bullish_signals.append("RSI شرایط اشباع فروش را نشان می‌دهد")
                elif current['rsi'] > 70:
                    bearish_signals.append("RSI شرایط اشباع خرید را نشان می‌دهد")
                elif current['rsi'] > previous['rsi'] and previous['rsi'] < 30:
                    bullish_signals.append("RSI از شرایط اشباع فروش خارج شده است")
                elif current['rsi'] < previous['rsi'] and previous['rsi'] > 70:
                    bearish_signals.append("RSI از شرایط اشباع خرید خارج شده است")
                else:
                    if current['rsi'] > 50 and previous['rsi'] < 50:
                        bullish_signals.append("RSI از خط میانی عبور کرد (صعودی)")
                    elif current['rsi'] < 50 and previous['rsi'] > 50:
                        bearish_signals.append("RSI از خط میانی عبور کرد (نزولی)")
                    else:
                        neutral_signals.append("RSI")
        
        # 2. MACD
        if 'MACD' in selected_indicators and 'macd' in df.columns and 'macd_signal' in df.columns:
            if not pd.isna(current['macd']) and not pd.isna(current['macd_signal']):
                if current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal']:
                    bullish_signals.append("تقاطع صعودی MACD رخ داده است")
                elif current['macd'] < current['macd_signal'] and previous['macd'] >= previous['macd_signal']:
                    bearish_signals.append("تقاطع نزولی MACD رخ داده است")
                else:
                    if current['macd'] > current['macd_signal']:
                        bullish_signals.append("MACD بالاتر از خط سیگنال قرار دارد")
                    elif current['macd'] < current['macd_signal']:
                        bearish_signals.append("MACD پایین‌تر از خط سیگنال قرار دارد")
                    else:
                        neutral_signals.append("MACD")
        
        # 3. Bollinger Bands
        if 'Bollinger Bands' in selected_indicators and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
                if current['close'] > current['bb_upper']:
                    bearish_signals.append("قیمت از باند بالایی بولینگر عبور کرده است")
                elif current['close'] < current['bb_lower']:
                    bullish_signals.append("قیمت از باند پایینی بولینگر عبور کرده است")
                elif current['close'] < previous['close'] and previous['close'] > previous['bb_upper']:
                    bearish_signals.append("قیمت پس از عبور از باند بالایی بولینگر در حال بازگشت است")
                elif current['close'] > previous['close'] and previous['close'] < previous['bb_lower']:
                    bullish_signals.append("قیمت پس از عبور از باند پایینی بولینگر در حال بازگشت است")
                else:
                    neutral_signals.append("Bollinger Bands")
        
        # 4. Stochastic
        if 'Stochastic' in selected_indicators and 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            if not pd.isna(current['stoch_k']) and not pd.isna(current['stoch_d']):
                if current['stoch_k'] < 20 and current['stoch_d'] < 20:
                    bullish_signals.append("استوکاستیک در منطقه اشباع فروش قرار دارد")
                elif current['stoch_k'] > 80 and current['stoch_d'] > 80:
                    bearish_signals.append("استوکاستیک در منطقه اشباع خرید قرار دارد")
                elif current['stoch_k'] > current['stoch_d'] and previous['stoch_k'] <= previous['stoch_d']:
                    bullish_signals.append("تقاطع صعودی استوکاستیک رخ داده است")
                elif current['stoch_k'] < current['stoch_d'] and previous['stoch_k'] >= previous['stoch_d']:
                    bearish_signals.append("تقاطع نزولی استوکاستیک رخ داده است")
                else:
                    neutral_signals.append("Stochastic")
        
        # 5. Supertrend
        if 'Supertrend' in selected_indicators and 'supertrend' in df.columns and 'supertrend_direction' in df.columns:
            if not pd.isna(current['supertrend_direction']):
                if current['supertrend_direction'] == 1 and previous['supertrend_direction'] == -1:
                    bullish_signals.append("تغییر جهت سوپرترند به حالت صعودی")
                elif current['supertrend_direction'] == -1 and previous['supertrend_direction'] == 1:
                    bearish_signals.append("تغییر جهت سوپرترند به حالت نزولی")
                elif current['supertrend_direction'] == 1:
                    bullish_signals.append("سوپرترند در حالت صعودی قرار دارد")
                elif current['supertrend_direction'] == -1:
                    bearish_signals.append("سوپرترند در حالت نزولی قرار دارد")
                else:
                    neutral_signals.append("Supertrend")
        
        # 6. ADX
        if 'ADX' in selected_indicators and 'adx' in df.columns and 'adx_pos' in df.columns and 'adx_neg' in df.columns:
            if not pd.isna(current['adx']):
                if current['adx'] > 25:
                    if current['adx_pos'] > current['adx_neg']:
                        bullish_signals.append(f"ADX نشان‌دهنده روند صعودی قوی است ({current['adx']:.1f})")
                    elif current['adx_neg'] > current['adx_pos']:
                        bearish_signals.append(f"ADX نشان‌دهنده روند نزولی قوی است ({current['adx']:.1f})")
                    else:
                        neutral_signals.append("ADX")
                else:
                    neutral_signals.append("ADX نشان‌دهنده عدم وجود روند قوی است")
        
        # 7. میانگین‌های متحرک (EMA و SMA)
        # تقاطع میانگین‌های متحرک کوتاه‌مدت و بلندمدت
        if ('EMA' in selected_indicators or 'SMA' in selected_indicators) and 'ema50' in df.columns and 'ema200' in df.columns:
            if not pd.isna(current['ema50']) and not pd.isna(current['ema200']):
                if current['ema50'] > current['ema200'] and previous['ema50'] <= previous['ema200']:
                    bullish_signals.append("تقاطع طلایی میانگین‌های متحرک (Golden Cross) رخ داده است")
                elif current['ema50'] < current['ema200'] and previous['ema50'] >= previous['ema200']:
                    bearish_signals.append("تقاطع مرگ میانگین‌های متحرک (Death Cross) رخ داده است")
                elif current['ema50'] > current['ema200']:
                    bullish_signals.append("میانگین متحرک 50 بالاتر از 200 قرار دارد (روند صعودی)")
                elif current['ema50'] < current['ema200']:
                    bearish_signals.append("میانگین متحرک 50 پایین‌تر از 200 قرار دارد (روند نزولی)")
                else:
                    neutral_signals.append("Moving Averages")
        
        # 8. VWAP
        if 'VWAP' in selected_indicators and 'vwap' in df.columns:
            if not pd.isna(current['vwap']):
                if current['close'] > current['vwap'] and previous['close'] <= previous['vwap']:
                    bullish_signals.append("قیمت از VWAP عبور کرده است (صعودی)")
                elif current['close'] < current['vwap'] and previous['close'] >= previous['vwap']:
                    bearish_signals.append("قیمت از VWAP عبور کرده است (نزولی)")
                elif current['close'] > current['vwap']:
                    bullish_signals.append("قیمت بالاتر از VWAP قرار دارد")
                elif current['close'] < current['vwap']:
                    bearish_signals.append("قیمت پایین‌تر از VWAP قرار دارد")
                else:
                    neutral_signals.append("VWAP")
        
        # 9. OBV
        if 'OBV' in selected_indicators and 'obv' in df.columns:
            # بررسی روند OBV در چند روز اخیر
            obv_series = df['obv'].dropna().tail(5)
            if len(obv_series) >= 5:
                obv_slope = np.polyfit(range(len(obv_series)), obv_series.values, 1)[0]
                if obv_slope > 0:
                    bullish_signals.append("روند OBV صعودی است (ورود پول به بازار)")
                elif obv_slope < 0:
                    bearish_signals.append("روند OBV نزولی است (خروج پول از بازار)")
                else:
                    neutral_signals.append("OBV")
        
        # 10. CMF
        if 'CMF' in selected_indicators and 'cmf' in df.columns:
            if not pd.isna(current['cmf']):
                if current['cmf'] > 0.05:
                    bullish_signals.append("شاخص CMF مثبت است (جریان پول ورودی)")
                elif current['cmf'] < -0.05:
                    bearish_signals.append("شاخص CMF منفی است (جریان پول خروجی)")
                else:
                    neutral_signals.append("CMF")
        
        # 11. Ichimoku Cloud
        if 'Ichimoku Cloud' in selected_indicators and 'ichimoku_a' in df.columns and 'ichimoku_b' in df.columns:
            if not pd.isna(current['ichimoku_a']) and not pd.isna(current['ichimoku_b']):
                cloud_top = max(current['ichimoku_a'], current['ichimoku_b'])
                cloud_bottom = min(current['ichimoku_a'], current['ichimoku_b'])
                
                if current['close'] > cloud_top:
                    bullish_signals.append("قیمت بالای ابر ایچیموکو قرار دارد (روند صعودی قوی)")
                elif current['close'] < cloud_bottom:
                    bearish_signals.append("قیمت زیر ابر ایچیموکو قرار دارد (روند نزولی قوی)")
                elif current['close'] > previous['close'] and previous['close'] < cloud_bottom:
                    bullish_signals.append("قیمت در حال ورود به ابر ایچیموکو است (صعودی)")
                elif current['close'] < previous['close'] and previous['close'] > cloud_top:
                    bearish_signals.append("قیمت در حال ورود به ابر ایچیموکو است (نزولی)")
                else:
                    neutral_signals.append("Ichimoku Cloud")
        
        # محاسبه قدرت سیگنال (0-100)
        total_signals = len(bullish_signals) + len(bearish_signals) + len(neutral_signals)
        
        if total_signals == 0:
            return "NEUTRAL", "اطلاعات کافی برای تحلیل وجود ندارد", 0, None
        
        bullish_strength = len(bullish_signals) / total_signals * 100
        bearish_strength = len(bearish_signals) / total_signals * 100
        
        # ایجاد توضیحات سیگنال
        signal_description = ""
        
        # تعیین سیگنال بر اساس قدرت نسبی سیگنال‌های صعودی و نزولی
        if bullish_strength > bearish_strength and bullish_strength >= threshold:
            signal_type = "BUY"
            signal_strength = bullish_strength
            
            signal_description = "## سیگنال خرید\n\n"
            signal_description += f"قدرت سیگنال: {bullish_strength:.1f}%\n\n"
            signal_description += "### دلایل سیگنال خرید:\n"
            
            for i, signal in enumerate(bullish_signals, 1):
                signal_description += f"{i}. {signal}\n"
        
        elif bearish_strength > bullish_strength and bearish_strength >= threshold:
            signal_type = "SELL"
            signal_strength = bearish_strength
            
            signal_description = "## سیگنال فروش\n\n"
            signal_description += f"قدرت سیگنال: {bearish_strength:.1f}%\n\n"
            signal_description += "### دلایل سیگنال فروش:\n"
            
            for i, signal in enumerate(bearish_signals, 1):
                signal_description += f"{i}. {signal}\n"
        
        else:
            signal_type = "NEUTRAL"
            signal_strength = max(50, min(bullish_strength, bearish_strength))
            
            signal_description = "## سیگنال خنثی\n\n"
            signal_description += f"قدرت سیگنال صعودی: {bullish_strength:.1f}%\n"
            signal_description += f"قدرت سیگنال نزولی: {bearish_strength:.1f}%\n\n"
            
            if bullish_signals:
                signal_description += "### نشانه‌های صعودی:\n"
                for i, signal in enumerate(bullish_signals, 1):
                    signal_description += f"{i}. {signal}\n"
            
            if bearish_signals:
                signal_description += "\n### نشانه‌های نزولی:\n"
                for i, signal in enumerate(bearish_signals, 1):
                    signal_description += f"{i}. {signal}\n"
        
        # محاسبه اهداف قیمتی و حد ضرر پیشرفته
        current_price = current['close']
        
        # بررسی وجود ATR برای محاسبه نوسان قیمت
        if 'atr' in current and not pd.isna(current['atr']):
            # استفاده از ATR برای محاسبه اهداف دقیق
            volatility = current['atr']
        else:
            # تخمین نوسان بر اساس میانگین حرکات قیمتی اخیر
            recent_data = df.tail(10)
            price_movements = (recent_data['high'] / recent_data['low'] - 1).mean()
            volatility = current_price * max(0.01, price_movements)
        
        # تنظیم ضرایب ریسک بر اساس سطح ریسک‌پذیری و شرایط بازار
        if risk_level == 'conservative':
            sl_multiplier = 1.5  # حد ضرر نزدیک‌تر
            tp1_multiplier = 1.5
            tp2_multiplier = 2.5
            tp3_multiplier = 3.5
            tp4_multiplier = 5.0
        elif risk_level == 'aggressive':
            sl_multiplier = 3.0  # حد ضرر دورتر
            tp1_multiplier = 2.0
            tp2_multiplier = 3.5
            tp3_multiplier = 5.5
            tp4_multiplier = 8.0
        else:  # balanced
            sl_multiplier = 2.0  # حد ضرر متعادل
            tp1_multiplier = 1.8
            tp2_multiplier = 3.0
            tp3_multiplier = 4.5
            tp4_multiplier = 6.5
        
        # بررسی شرایط بازار و تنظیم خودکار ضرایب
        # بررسی نوسان اخیر بازار
        recent_volatility = df.tail(20)['high'].max() / df.tail(20)['low'].min() - 1
        
        # اگر بازار بسیار پرنوسان است، ضرایب را تنظیم می‌کنیم
        if recent_volatility > 0.1:  # نوسان شدید (بیش از 10%)
            # افزایش فاصله تیک پرافیت‌ها در بازار پرنوسان
            tp1_multiplier *= 1.2
            tp2_multiplier *= 1.2
            tp3_multiplier *= 1.3
            tp4_multiplier *= 1.4
            # فاصله استاپ لاس را هم بیشتر می‌کنیم
            sl_multiplier *= 1.2
        elif recent_volatility < 0.03:  # نوسان کم (کمتر از 3%)
            # کاهش فاصله تیک پرافیت‌ها در بازار کم‌نوسان
            tp1_multiplier *= 0.8
            tp2_multiplier *= 0.8
            tp3_multiplier *= 0.7
            tp4_multiplier *= 0.7
            # فاصله استاپ لاس را هم کمتر می‌کنیم
            sl_multiplier *= 0.8
        
        # ایجاد اهداف قیمتی بر اساس نوع سیگنال
        if signal_type == "BUY":
            # محاسبه حد ضرر با استفاده از سطوح حمایت اخیر
            support_levels = []
            for i in range(1, min(30, len(df) - 1)):
                if df.iloc[-i-1]['low'] < df.iloc[-i]['low'] and df.iloc[-i-1]['low'] < df.iloc[-i+1]['low']:
                    support_levels.append(df.iloc[-i-1]['low'])
                    if len(support_levels) >= 3:
                        break
            
            # استفاده از نزدیک‌ترین سطح حمایت که زیر قیمت فعلی است
            nearest_support = None
            for level in support_levels:
                if level < current_price and (nearest_support is None or level > nearest_support):
                    nearest_support = level
            
            # اگر سطح حمایت مناسبی پیدا نشد، از ATR استفاده کنیم
            if nearest_support is None or nearest_support < current_price * 0.9:
                stop_loss = current_price - (volatility * sl_multiplier)
            else:
                # قرار دادن حد ضرر کمی پایین‌تر از سطح حمایت
                stop_loss = nearest_support * 0.995
            
            # محاسبه اهداف قیمتی
            take_profit_1 = current_price + (volatility * tp1_multiplier)
            take_profit_2 = current_price + (volatility * tp2_multiplier)
            take_profit_3 = current_price + (volatility * tp3_multiplier)
            take_profit_4 = current_price + (volatility * tp4_multiplier)
            
            # اضافه کردن اهداف به دیکشنری
            price_targets = {
                'entry': current_price,
                'sl': stop_loss,
                'tp1': take_profit_1,
                'tp2': take_profit_2,
                'tp3': take_profit_3,
                'tp4': take_profit_4
            }
            
            # اضافه کردن اطلاعات به توضیحات سیگنال
            signal_description += "\n**توصیه‌های معاملاتی:**\n"
            signal_description += f"- ورود: {current_price:.2f}\n"
            signal_description += f"- حد ضرر: {stop_loss:.2f} (فاصله: {((stop_loss / current_price - 1) * 100):.2f}%)\n"
            signal_description += f"- حد سود 1: {take_profit_1:.2f} (سود: {((take_profit_1 / current_price - 1) * 100):.2f}%)\n"
            signal_description += f"- حد سود 2: {take_profit_2:.2f} (سود: {((take_profit_2 / current_price - 1) * 100):.2f}%)\n"
            signal_description += f"- حد سود 3: {take_profit_3:.2f} (سود: {((take_profit_3 / current_price - 1) * 100):.2f}%)\n"
            signal_description += f"- حد سود 4: {take_profit_4:.2f} (سود: {((take_profit_4 / current_price - 1) * 100):.2f}%)\n\n"
            
            # محاسبه نسبت ریسک به ریوارد
            reward = take_profit_1 - current_price
            risk = current_price - stop_loss
            if risk > 0:
                risk_reward_ratio = reward / risk
                signal_description += f"- نسبت ریسک به ریوارد: {risk_reward_ratio:.2f}\n\n"
            
            signal_description += "- **پیشنهاد مدیریت پوزیشن**: 30% از پوزیشن در حد سود اول، 30% در حد سود دوم، 30% در حد سود سوم و 10% در حد سود چهارم بسته شود."
        
        elif signal_type == "SELL":
            # محاسبه حد ضرر با استفاده از سطوح مقاومت اخیر
            resistance_levels = []
            for i in range(1, min(30, len(df) - 1)):
                if df.iloc[-i-1]['high'] > df.iloc[-i]['high'] and df.iloc[-i-1]['high'] > df.iloc[-i+1]['high']:
                    resistance_levels.append(df.iloc[-i-1]['high'])
                    if len(resistance_levels) >= 3:
                        break
            
            # استفاده از نزدیک‌ترین سطح مقاومت که بالای قیمت فعلی است
            nearest_resistance = None
            for level in resistance_levels:
                if level > current_price and (nearest_resistance is None or level < nearest_resistance):
                    nearest_resistance = level
            
            # اگر سطح مقاومت مناسبی پیدا نشد، از ATR استفاده کنیم
            if nearest_resistance is None or nearest_resistance > current_price * 1.1:
                stop_loss = current_price + (volatility * sl_multiplier)
            else:
                # قرار دادن حد ضرر کمی بالاتر از سطح مقاومت
                stop_loss = nearest_resistance * 1.005
            
            # محاسبه اهداف قیمتی
            take_profit_1 = current_price - (volatility * tp1_multiplier)
            take_profit_2 = current_price - (volatility * tp2_multiplier)
            take_profit_3 = current_price - (volatility * tp3_multiplier)
            take_profit_4 = current_price - (volatility * tp4_multiplier)
            
            # اضافه کردن اهداف به دیکشنری
            price_targets = {
                'entry': current_price,
                'sl': stop_loss,
                'tp1': take_profit_1,
                'tp2': take_profit_2,
                'tp3': take_profit_3,
                'tp4': take_profit_4
            }
            
            # اضافه کردن اطلاعات به توضیحات سیگنال
            signal_description += "\n**توصیه‌های معاملاتی:**\n"
            signal_description += f"- ورود: {current_price:.2f}\n"
            signal_description += f"- حد ضرر: {stop_loss:.2f} (فاصله: {((stop_loss / current_price - 1) * 100):.2f}%)\n"
            signal_description += f"- حد سود 1: {take_profit_1:.2f} (سود: {((1 - take_profit_1 / current_price) * 100):.2f}%)\n"
            signal_description += f"- حد سود 2: {take_profit_2:.2f} (سود: {((1 - take_profit_2 / current_price) * 100):.2f}%)\n"
            signal_description += f"- حد سود 3: {take_profit_3:.2f} (سود: {((1 - take_profit_3 / current_price) * 100):.2f}%)\n"
            signal_description += f"- حد سود 4: {take_profit_4:.2f} (سود: {((1 - take_profit_4 / current_price) * 100):.2f}%)\n\n"
            
            # محاسبه نسبت ریسک به ریوارد
            reward = current_price - take_profit_1
            risk = stop_loss - current_price
            if risk > 0:
                risk_reward_ratio = reward / risk
                signal_description += f"- نسبت ریسک به ریوارد: {risk_reward_ratio:.2f}\n\n"
            
            signal_description += "- **پیشنهاد مدیریت پوزیشن**: 30% از پوزیشن در حد سود اول، 30% در حد سود دوم، 30% در حد سود سوم و 10% در حد سود چهارم بسته شود."
        
        else:  # NEUTRAL
            signal_description += "\n**توصیه‌های معاملاتی:**\n"
            signal_description += "- در شرایط فعلی، بازار روند مشخصی ندارد.\n"
            signal_description += "- بهتر است صبر کنید تا سیگنال قوی‌تری ایجاد شود.\n"
            signal_description += "- می‌توانید برای معاملات کوتاه‌مدت، از تقاطع‌های قیمت با میانگین‌های متحرک کوتاه‌مدت استفاده کنید."
            
            # در حالت خنثی اهداف قیمتی نداریم
            price_targets = None
        
        return signal_type, signal_description, round(signal_strength), price_targets
    
    except Exception as e:
        print(f"خطا در تولید سیگنال: {str(e)}")
        return "NEUTRAL", "خطا در محاسبه سیگنال. لطفاً دوباره تلاش کنید.", 0, None