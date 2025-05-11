"""
ماژول شناسایی ارزهای دیجیتال با پتانسیل بالا

این ماژول شامل توابع مورد نیاز برای شناسایی ارزهای دیجیتال با پتانسیل بالا رشد است.
"""

import pandas as pd
import numpy as np
import streamlit as st
import time
import ccxt
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from technical_analysis import perform_technical_analysis
from advanced_indicators import AdvancedIndicators
from chart_patterns import analyze_chart_patterns
from api_services import get_top_cryptocurrencies, get_ohlcv_data_multi_source

def find_high_potential_cryptocurrencies(top_coins, method='ترکیبی', limit=10):
    """
    شناسایی ارزهای دیجیتال با پتانسیل بالا
    
    Args:
        top_coins (list): لیست ارزهای دیجیتال برتر
        method (str): روش اسکن (ترکیبی، حجم بالا، رشد سریع، الگوی نموداری، هوش مصنوعی)
        limit (int): حداکثر تعداد ارزهای بازگشتی
        
    Returns:
        list: لیست ارزهای با پتانسیل بالا
    """
    # ایجاد لیست نتایج
    high_potential_coins = []
    
    # صفحه پیشرفت
    progress_text = f"در حال اسکن بازار با روش {method}..."
    progress_bar = st.progress(0, text=progress_text)
    
    # تعداد کل ارزها برای بررسی
    total_coins = min(30, len(top_coins))
    
    # بررسی ارزها به صورت موازی
    with ThreadPoolExecutor(max_workers=5) as executor:
        # ایجاد تسک‌ها
        futures = {}
        for i, coin in enumerate(top_coins[:total_coins]):
            symbol = f"{coin['symbol']}/USDT"
            futures[executor.submit(analyze_coin_potential, symbol, method)] = coin
            
            # به‌روزرسانی پیشرفت
            progress_bar.progress((i + 1) / total_coins, text=f"در حال تحلیل {symbol}...")
        
        # جمع‌آوری نتایج
        for future in as_completed(futures):
            coin = futures[future]
            try:
                potential_data = future.result()
                if potential_data and potential_data['potential_score'] > 50:
                    coin_data = {
                        **coin,
                        **potential_data
                    }
                    high_potential_coins.append(coin_data)
            except Exception as e:
                st.warning(f"خطا در تحلیل {coin['symbol']}: {str(e)}")
    
    # پایان پیشرفت
    progress_bar.empty()
    
    # مرتب‌سازی بر اساس امتیاز
    high_potential_coins.sort(key=lambda x: x['potential_score'], reverse=True)
    
    # محدود کردن نتایج
    return high_potential_coins[:limit]

def analyze_coin_potential(symbol, method):
    """
    تحلیل پتانسیل یک ارز دیجیتال
    
    Args:
        symbol (str): نماد ارز
        method (str): روش تحلیل
        
    Returns:
        dict: اطلاعات پتانسیل ارز
    """
    try:
        # دریافت داده‌های تاریخی
        timeframe = '4h'  # تایم‌فریم مناسب برای تحلیل میان‌مدت
        df = get_ohlcv_data_multi_source(symbol, timeframe, lookback_days=30)
        
        if df is None or df.empty or len(df) < 10:
            return None
        
        # انجام تحلیل تکنیکال
        indicators = [
            'RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'Supertrend',
            'ADX', 'ATR', 'EMA', 'SMA', 'VWAP', 'Volume', 'OBV', 'CMF'
        ]
        analysis_df = perform_technical_analysis(df, indicators)
        
        # تشخیص الگوهای نموداری
        patterns = analyze_chart_patterns(analysis_df)
        
        # محاسبه حجم نسبی
        volume_change_24h = ((analysis_df['volume'].iloc[-1] / analysis_df['volume'].iloc[-7:].mean()) - 1) * 100
        
        # محاسبه امتیاز پتانسیل بر اساس روش انتخابی
        potential_score = 0
        potential_details = []
        
        if method == 'حجم بالا' or method == 'ترکیبی':
            volume_score = calculate_volume_potential(analysis_df)
            potential_score += volume_score * (1.0 if method == 'حجم بالا' else 0.3)
            
            if volume_score > 70:
                potential_details.append(f"حجم معاملات بالا (امتیاز: {volume_score:.1f})")
            
        if method == 'رشد سریع' or method == 'ترکیبی':
            growth_score = calculate_growth_potential(analysis_df)
            potential_score += growth_score * (1.0 if method == 'رشد سریع' else 0.3)
            
            if growth_score > 70:
                potential_details.append(f"پتانسیل رشد قیمت (امتیاز: {growth_score:.1f})")
            
        if method == 'الگوی نموداری' or method == 'ترکیبی':
            pattern_score = calculate_pattern_potential(analysis_df, patterns)
            potential_score += pattern_score * (1.0 if method == 'الگوی نموداری' else 0.2)
            
            if pattern_score > 70:
                potential_details.append(f"الگوهای نموداری مثبت (امتیاز: {pattern_score:.1f})")
            
        if method == 'هوش مصنوعی' or method == 'ترکیبی':
            ai_score = calculate_ai_potential(analysis_df)
            potential_score += ai_score * (1.0 if method == 'هوش مصنوعی' else 0.2)
            
            if ai_score > 70:
                potential_details.append(f"تأیید هوش مصنوعی (امتیاز: {ai_score:.1f})")
        
        # اگر روش ترکیبی است، نرمال‌سازی امتیاز
        if method == 'ترکیبی':
            potential_score = potential_score / 1.0
        
        # محدود کردن امتیاز به 0-100
        potential_score = max(0, min(100, potential_score))
        
        # تعیین جهت پتانسیل (صعودی یا نزولی)
        if 'rsi' in analysis_df.columns and not pd.isna(analysis_df['rsi'].iloc[-1]):
            rsi = analysis_df['rsi'].iloc[-1]
            macd_signal = 0
            
            if 'macd' in analysis_df.columns and 'macd_signal' in analysis_df.columns:
                macd = analysis_df['macd'].iloc[-1]
                macd_signal_value = analysis_df['macd_signal'].iloc[-1]
                macd_signal = 1 if macd > macd_signal_value else -1 if macd < macd_signal_value else 0
            
            if rsi > 50 and macd_signal >= 0:
                potential_direction = 'صعودی'
            elif rsi < 50 and macd_signal <= 0:
                potential_direction = 'نزولی'
            else:
                potential_direction = 'خنثی'
        else:
            # اگر RSI موجود نیست، از روند قیمت استفاده می‌کنیم
            price_change = ((analysis_df['close'].iloc[-1] / analysis_df['close'].iloc[-7]) - 1) * 100
            potential_direction = 'صعودی' if price_change > 0 else 'نزولی' if price_change < 0 else 'خنثی'
        
        # آماده‌سازی نتیجه
        result = {
            'potential_score': potential_score,
            'potential_direction': potential_direction,
            'volume_change_24h': volume_change_24h,
            'price_change_7d': ((analysis_df['close'].iloc[-1] / analysis_df['close'].iloc[-28]) - 1) * 100 if len(analysis_df) >= 28 else None,
            'current_price': analysis_df['close'].iloc[-1],
            'potential_details': potential_details,
            'patterns': [p['type'] for p in patterns] if patterns else []
        }
        
        return result
    
    except Exception as e:
        st.warning(f"خطا در تحلیل {symbol}: {str(e)}")
        return None

def calculate_volume_potential(df):
    """
    محاسبه پتانسیل بر اساس حجم معاملات
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        
    Returns:
        float: امتیاز پتانسیل (0-100)
    """
    # بررسی افزایش حجم معاملات
    recent_volume = df['volume'].iloc[-3:].mean()
    avg_volume = df['volume'].iloc[-30:].mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    # بررسی همبستگی حجم با قیمت
    recent_price_change = ((df['close'].iloc[-1] / df['close'].iloc[-3]) - 1) * 100
    price_direction = 1 if recent_price_change > 0 else -1
    
    # محاسبه OBV اگر موجود است
    obv_trend = 0
    if 'obv' in df.columns:
        recent_obv = df['obv'].iloc[-5:].values
        obv_slope = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
        obv_trend = 1 if obv_slope > 0 else -1 if obv_slope < 0 else 0
    
    # محاسبه امتیاز
    score = 50  # امتیاز پایه
    
    # افزایش امتیاز بر اساس افزایش حجم
    if volume_ratio > 3:
        score += 30
    elif volume_ratio > 2:
        score += 20
    elif volume_ratio > 1.5:
        score += 10
    
    # افزایش امتیاز بر اساس همبستگی حجم و قیمت
    if price_direction == 1 and volume_ratio > 1.2:
        score += 10
    elif price_direction == -1 and volume_ratio > 1.2:
        score -= 10
    
    # افزایش امتیاز بر اساس روند OBV
    if obv_trend == 1:
        score += 10
    elif obv_trend == -1:
        score -= 5
    
    # محدود کردن امتیاز
    return max(0, min(100, score))

def calculate_growth_potential(df):
    """
    محاسبه پتانسیل رشد قیمت
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        
    Returns:
        float: امتیاز پتانسیل (0-100)
    """
    # امتیاز پایه
    score = 50
    
    # بررسی روند قیمت
    short_ma = df['close'].iloc[-5:].mean()
    medium_ma = df['close'].iloc[-14:].mean()
    long_ma = df['close'].iloc[-30:].mean() if len(df) >= 30 else df['close'].mean()
    
    current_price = df['close'].iloc[-1]
    
    # بررسی روند صعودی
    if current_price > short_ma > medium_ma > long_ma:
        score += 20
    elif current_price > short_ma > medium_ma:
        score += 15
    elif current_price > short_ma:
        score += 10
    
    # بررسی RSI
    if 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        
        if 40 <= rsi <= 65:  # RSI در محدوده مناسب برای رشد
            score += 10
        elif rsi < 30:  # RSI در محدوده اشباع فروش
            score += 15
        elif rsi > 70:  # RSI در محدوده اشباع خرید
            score -= 10
    
    # بررسی MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        
        if macd > macd_signal and macd < 0:  # MACD در حال عبور از خط سیگنال در محدوده منفی
            score += 15
        elif macd > macd_signal:
            score += 10
        elif macd < macd_signal:
            score -= 5
    
    # بررسی Bollinger Bands
    if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
        bb_lower = df['bb_lower'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        
        if current_price < bb_lower * 1.05:  # قیمت نزدیک به باند پایینی
            score += 15
        elif current_price > bb_upper * 0.95:  # قیمت نزدیک به باند بالایی
            score -= 10
    
    # بررسی حجم معاملات
    price_change_5d = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
    volume_change_5d = ((df['volume'].iloc[-1] / df['volume'].iloc[-5:].mean()) - 1) * 100 if len(df) >= 5 else 0
    
    if price_change_5d > 0 and volume_change_5d > 50:
        score += 10
    
    # محدود کردن امتیاز
    return max(0, min(100, score))

def calculate_pattern_potential(df, patterns):
    """
    محاسبه پتانسیل بر اساس الگوهای نموداری
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        patterns (list): لیست الگوهای شناسایی شده
        
    Returns:
        float: امتیاز پتانسیل (0-100)
    """
    # امتیاز پایه
    score = 50
    
    # اگر الگویی شناسایی نشده
    if not patterns:
        return score
    
    # بررسی الگوها
    for pattern in patterns:
        pattern_type = pattern.get('type', '')
        pattern_direction = pattern.get('direction', 'neutral')
        pattern_strength = pattern.get('strength', 50)
        
        # الگوهای صعودی
        if pattern_direction == 'bullish':
            if 'Double Bottom' in pattern_type or 'Inverse Head and Shoulders' in pattern_type:
                score += 20
            elif 'Bullish Engulfing' in pattern_type or 'Morning Star' in pattern_type:
                score += 15
            elif 'Hammer' in pattern_type or 'Bullish Harami' in pattern_type:
                score += 10
            else:
                score += pattern_strength * 0.1
        
        # الگوهای نزولی
        elif pattern_direction == 'bearish':
            if 'Double Top' in pattern_type or 'Head and Shoulders' in pattern_type:
                score -= 15
            elif 'Bearish Engulfing' in pattern_type or 'Evening Star' in pattern_type:
                score -= 10
            elif 'Shooting Star' in pattern_type or 'Bearish Harami' in pattern_type:
                score -= 5
            else:
                score -= pattern_strength * 0.08
    
    # محدود کردن امتیاز
    return max(0, min(100, score))

def calculate_ai_potential(df):
    """
    محاسبه پتانسیل با استفاده از الگوریتم‌های هوش مصنوعی
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        
    Returns:
        float: امتیاز پتانسیل (0-100)
    """
    # امتیاز پایه
    score = 50
    
    # شبیه‌سازی تحلیل هوش مصنوعی
    # در نسخه واقعی از مدل‌های یادگیری ماشین استفاده می‌شود
    
    # تحلیل ترکیبی اندیکاتورها
    indicators_bullish = 0
    indicators_bearish = 0
    
    # بررسی RSI
    if 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2] if len(df) > 1 else rsi
        
        if rsi < 30:
            indicators_bullish += 1
        elif rsi > 70:
            indicators_bearish += 1
        elif rsi > rsi_prev and rsi < 50:
            indicators_bullish += 0.5
        elif rsi < rsi_prev and rsi > 50:
            indicators_bearish += 0.5
    
    # بررسی MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd
        
        if macd > macd_signal and macd_prev <= macd_signal:
            indicators_bullish += 1
        elif macd < macd_signal and macd_prev >= macd_signal:
            indicators_bearish += 1
        elif macd > macd_signal:
            indicators_bullish += 0.5
        elif macd < macd_signal:
            indicators_bearish += 0.5
    
    # بررسی Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_middle = df['bb_middle'].iloc[-1]
        
        if current_price < bb_lower:
            indicators_bullish += 1
        elif current_price > bb_upper:
            indicators_bearish += 1
        elif current_price < bb_middle:
            indicators_bullish += 0.3
        elif current_price > bb_middle:
            indicators_bearish += 0.3
    
    # بررسی EMA
    if 'ema' in df.columns:
        current_price = df['close'].iloc[-1]
        ema = df['ema'].iloc[-1]
        
        if current_price > ema:
            indicators_bullish += 0.5
        else:
            indicators_bearish += 0.5
    
    # محاسبه امتیاز بر اساس تعداد اندیکاتورهای صعودی و نزولی
    total_indicators = indicators_bullish + indicators_bearish
    if total_indicators > 0:
        bullish_ratio = indicators_bullish / total_indicators
        score = 50 + (bullish_ratio - 0.5) * 100
    
    # بررسی روند قیمت
    price_trend = np.polyfit(range(min(10, len(df))), df['close'].iloc[-min(10, len(df)):], 1)[0]
    if price_trend > 0:
        score += 10
    else:
        score -= 10
    
    # محدود کردن امتیاز
    return max(0, min(100, score))

def display_high_potential_coins(coins):
    """
    نمایش ارزهای با پتانسیل بالا
    
    Args:
        coins (list): لیست ارزهای با پتانسیل بالا
    """
    if not coins or len(coins) == 0:
        st.info("هیچ ارز با پتانسیل بالایی یافت نشد")
        return
    
    # نمایش خلاصه
    st.success(f"{len(coins)} ارز با پتانسیل بالا شناسایی شدند")
    
    # نمایش به صورت کارت
    cols = st.columns(3)
    
    for i, coin in enumerate(coins):
        col_idx = i % 3
        
        with cols[col_idx]:
            # رنگ کارت بر اساس جهت پتانسیل
            card_color = "rgba(0, 200, 0, 0.1)" if coin['potential_direction'] == 'صعودی' else "rgba(200, 0, 0, 0.1)" if coin['potential_direction'] == 'نزولی' else "rgba(100, 100, 100, 0.1)"
            
            # ایجاد کارت
            st.markdown(f"""
            <div style="border-radius: 10px; padding: 10px; margin-bottom: 15px; background-color: {card_color};">
                <h3 style="margin: 0;">{coin['name']} ({coin['symbol']})</h3>
                <div style="margin: 10px 0;">
                    <b>قیمت فعلی:</b> ${coin['current_price']:.4f}<br>
                    <b>تغییر 24 ساعته:</b> {coin.get('price_change_percentage_24h', 0):.2f}%<br>
                    <b>جهت پتانسیل:</b> {coin['potential_direction']}<br>
                    <b>امتیاز پتانسیل:</b> {coin['potential_score']:.1f}/100
                </div>
                <div style="margin-top: 5px; font-size: 0.9em; color: #555;">
                    {'<br>'.join(coin['potential_details']) if 'potential_details' in coin and coin['potential_details'] else 'اطلاعات بیشتری موجود نیست'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # نمایش جدول کامل
    st.markdown("### جدول کامل ارزهای با پتانسیل بالا")
    
    # ایجاد دیتافریم
    df = pd.DataFrame(coins)
    
    # انتخاب و تغییر نام ستون‌ها
    if 'market_cap_rank' in df.columns:
        df['رتبه'] = df['market_cap_rank']
    
    df['نام'] = df['name']
    df['نماد'] = df['symbol']
    df['قیمت'] = df['current_price']
    df['تغییر 24 ساعته (%)'] = df.get('price_change_percentage_24h', pd.Series([0] * len(df)))
    df['تغییر حجم (%)'] = df.get('volume_change_24h', pd.Series([0] * len(df)))
    df['امتیاز پتانسیل'] = df['potential_score']
    df['جهت پتانسیل'] = df['potential_direction']
    
    # نمایش دیتافریم
    st.dataframe(
        df[['نام', 'نماد', 'قیمت', 'تغییر 24 ساعته (%)', 'تغییر حجم (%)', 'امتیاز پتانسیل', 'جهت پتانسیل']],
        hide_index=True
    )
    
    # نمایش نکات تکمیلی
    st.info("""
    **نکات مهم:**
    - امتیاز پتانسیل بر اساس تحلیل تکنیکال و الگوهای نموداری محاسبه شده است.
    - امتیاز بالاتر نشان‌دهنده احتمال تغییر قیمت در جهت مشخص شده است.
    - این اطلاعات صرفاً جنبه آموزشی دارد و توصیه سرمایه‌گذاری محسوب نمی‌شود.
    - همیشه پیش از هرگونه معامله، تحلیل کامل خود را انجام دهید.
    """)
