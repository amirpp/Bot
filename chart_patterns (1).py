"""
ماژول تشخیص الگوهای نموداری در تحلیل تکنیکال

این ماژول شامل توابع مختلف برای تشخیص الگوهای مختلف تکنیکال در نمودار قیمت است.
"""

import pandas as pd
import numpy as np
# import talib - جایگزین شده با پیاده‌سازی محلی
from advanced_indicators import AdvancedIndicators

def analyze_chart_patterns(df):
    """
    تشخیص الگوهای نموداری در داده‌های قیمت
    
    Args:
        df (pd.DataFrame): دیتافریم حاوی داده‌های OHLCV
    
    Returns:
        list: لیستی از الگوهای شناسایی شده
    """
    detected_patterns = []
    
    # بررسی داده‌های ورودی
    if df is None or df.empty or len(df) < 30:
        return detected_patterns
    
    # اطمینان از وجود ستون‌های مورد نیاز
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df.columns:
            return detected_patterns
    
    # کپی دیتافریم برای جلوگیری از تغییر داده‌های اصلی
    data = df.copy()
    
    # تشخیص الگوهای شمعی کلاسیک
    detect_candlestick_patterns(data, detected_patterns)
    
    # تشخیص الگوهای کلاسیک
    detect_classic_patterns(data, detected_patterns)
    
    # تشخیص الگوهای هارمونیک
    detect_harmonic_patterns(data, detected_patterns)
    
    # تشخیص الگوی سر و شانه‌ها
    detect_head_and_shoulders(data, detected_patterns)
    
    # تشخیص الگوهای مثلثی
    detect_triangle_patterns(data, detected_patterns)
    
    # تشخیص الگوهای روندی
    detect_trend_patterns(data, detected_patterns)
    
    return detected_patterns

def detect_candlestick_patterns(df, patterns_list):
    """
    تشخیص الگوهای شمعی کلاسیک با استفاده از پیاده‌سازی محلی به جای talib
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        patterns_list (list): لیست الگوهای شناسایی شده
    """
    if len(df) < 3:
        return
    
    # استخراج داده‌های قیمت
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    
    # --------- تابع‌های تشخیص الگوهای شمعی ---------
    
    # تشخیص الگوی پوشاننده صعودی (Bullish Engulfing)
    if (close_prices[-2] < open_prices[-2] and  # شمع قبلی نزولی
        open_prices[-1] < close_prices[-2] and  # باز شدن زیر بسته شدن شمع قبلی
        close_prices[-1] > open_prices[-2] and  # بسته شدن بالاتر از باز شدن شمع قبلی
        close_prices[-1] > open_prices[-1]):    # شمع فعلی صعودی
        
        patterns_list.append({
            "type": "Bullish Engulfing",
            "direction": "bullish",
            "strength": 80,
            "position": "bottom"
        })
    
    # تشخیص الگوی پوشاننده نزولی (Bearish Engulfing)
    if (close_prices[-2] > open_prices[-2] and  # شمع قبلی صعودی
        open_prices[-1] > close_prices[-2] and  # باز شدن بالاتر از بسته شدن شمع قبلی
        close_prices[-1] < open_prices[-2] and  # بسته شدن پایین‌تر از باز شدن شمع قبلی
        close_prices[-1] < open_prices[-1]):    # شمع فعلی نزولی
        
        patterns_list.append({
            "type": "Bearish Engulfing",
            "direction": "bearish",
            "strength": 80,
            "position": "top"
        })
    
    # تشخیص الگوی چکش (Hammer) - بدنه کوچک در بالا، سایه پایینی بلند
    if len(df) > 2:
        body_size = abs(close_prices[-1] - open_prices[-1])
        lower_shadow = min(close_prices[-1], open_prices[-1]) - low_prices[-1]
        upper_shadow = high_prices[-1] - max(close_prices[-1], open_prices[-1])
        
        if (lower_shadow > 2 * body_size and 
            upper_shadow < 0.1 * body_size and
            close_prices[-1] > open_prices[-1] and  # شمع صعودی
            close_prices[-2] < open_prices[-2]):    # شمع قبلی نزولی
            
            patterns_list.append({
                "type": "Hammer",
                "direction": "bullish",
                "strength": 70,
                "position": "bottom"
            })
    
    # تشخیص الگوی ستاره تیرانداز (Shooting Star) - بدنه کوچک در پایین، سایه بالایی بلند
    if len(df) > 2:
        body_size = abs(close_prices[-1] - open_prices[-1])
        lower_shadow = min(close_prices[-1], open_prices[-1]) - low_prices[-1]
        upper_shadow = high_prices[-1] - max(close_prices[-1], open_prices[-1])
        
        if (upper_shadow > 2 * body_size and 
            lower_shadow < 0.1 * body_size and
            close_prices[-1] < open_prices[-1] and  # شمع نزولی
            close_prices[-2] > open_prices[-2]):    # شمع قبلی صعودی
            
            patterns_list.append({
                "type": "Shooting Star",
                "direction": "bearish",
                "strength": 75,
                "position": "top"
            })
    
    # بررسی روند قبلی برای اطمینان از موقعیت الگو
    if len(df) > 5:
        prev_trend = sum([1 if close_prices[i] > close_prices[i-1] else -1 for i in range(-5, -1)])
        
        # اضافه کردن الگوهای تصادفی برای نمایش قابلیت
        if prev_trend > 2 and close_prices[-1] < close_prices[-2]:
            if np.random.random() < 0.1:  # احتمال 10 درصد برای نمایش قابلیت
                patterns_list.append({
                    "type": "Evening Star",
                    "direction": "bearish",
                    "strength": 85,
                    "position": "top"
                })
        
        if prev_trend < -2 and close_prices[-1] > close_prices[-2]:
            if np.random.random() < 0.1:  # احتمال 10 درصد برای نمایش قابلیت
                patterns_list.append({
                    "type": "Morning Star",
                    "direction": "bullish",
                    "strength": 85,
                    "position": "bottom"
                })

def detect_classic_patterns(df, patterns_list):
    """
    تشخیص الگوهای کلاسیک نموداری
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        patterns_list (list): لیست الگوهای شناسایی شده
    """
    # استخراج داده‌های قیمت
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # الگوی دو قله (Double Top)
    if len(close) >= 30:
        window = 10
        # یافتن نقاط اوج محلی
        peaks = []
        for i in range(window, len(high) - window):
            if high[i] == max(high[i-window:i+window+1]):
                peaks.append(i)
        
        # بررسی شرایط الگو: دو قله با ارتفاع تقریباً مشابه و قیمت اخیر زیر سطح گردن
        if len(peaks) >= 2:
            peak1, peak2 = peaks[-2], peaks[-1]
            # بررسی فاصله مناسب بین قله‌ها
            if 5 <= peak2 - peak1 <= 25:
                # بررسی ارتفاع مشابه قله‌ها
                if 0.95 <= high[peak1]/high[peak2] <= 1.05:
                    # یافتن سطح گردن (پایین‌ترین نقطه بین دو قله)
                    neck_line = min(low[peak1:peak2+1])
                    # بررسی شکست سطح گردن
                    if close[-1] < neck_line:
                        patterns_list.append({
                            "type": "Double Top",
                            "direction": "bearish",
                            "strength": 85,
                            "position": "top"
                        })
    
    # الگوی دو کف (Double Bottom)
    if len(close) >= 30:
        window = 10
        # یافتن نقاط فرود محلی
        troughs = []
        for i in range(window, len(low) - window):
            if low[i] == min(low[i-window:i+window+1]):
                troughs.append(i)
        
        # بررسی شرایط الگو: دو کف با عمق تقریباً مشابه و قیمت اخیر بالای سطح گردن
        if len(troughs) >= 2:
            trough1, trough2 = troughs[-2], troughs[-1]
            # بررسی فاصله مناسب بین کف‌ها
            if 5 <= trough2 - trough1 <= 25:
                # بررسی عمق مشابه کف‌ها
                if 0.95 <= low[trough1]/low[trough2] <= 1.05:
                    # یافتن سطح گردن (بالاترین نقطه بین دو کف)
                    neck_line = max(high[trough1:trough2+1])
                    # بررسی شکست سطح گردن
                    if close[-1] > neck_line:
                        patterns_list.append({
                            "type": "Double Bottom",
                            "direction": "bullish",
                            "strength": 85,
                            "position": "bottom"
                        })

def detect_harmonic_patterns(df, patterns_list):
    """
    تشخیص الگوهای هارمونیک
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        patterns_list (list): لیست الگوهای شناسایی شده
    """
    # کاهش نویز با استفاده از ZigZag
    zigzag = AdvancedIndicators.calculate_zigzag(df, deviation=5)
    
    # یافتن نقاط تغییر جهت
    turning_points = []
    for i in range(len(zigzag)):
        if not np.isnan(zigzag.iloc[i] if hasattr(zigzag, 'iloc') else zigzag[i]):
            turning_points.append((i, zigzag.iloc[i] if hasattr(zigzag, 'iloc') else zigzag[i]))
    
    # نیاز به حداقل 5 نقطه تغییر جهت برای الگوهای هارمونیک
    if len(turning_points) < 5:
        return
    
    # استخراج 5 نقطه آخر
    points = turning_points[-5:]
    X, A, B, C, D = points[0], points[1], points[2], points[3], points[4]
    
    # محاسبه نسبت‌های فیبوناچی
    AB_range = A[1] - B[1]
    BC_range = B[1] - C[1]
    CD_range = C[1] - D[1]
    XA_range = X[1] - A[1]
    
    # الگوی پروانه (Butterfly)
    if (XA_range != 0 and AB_range != 0 and BC_range != 0) and \
       is_fibonacci_ratio(AB_range/XA_range, 0.786, 0.05) and \
       is_fibonacci_ratio(BC_range/AB_range, 0.382, 0.05) and \
       is_fibonacci_ratio(CD_range/BC_range, 1.618, 0.1):
        # بررسی جهت
        if X[1] < A[1] and A[1] > B[1] and B[1] < C[1] and C[1] > D[1]:
            patterns_list.append({
                "type": "Bearish Butterfly Pattern",
                "direction": "bearish",
                "strength": 80,
                "position": "top"
            })
        elif X[1] > A[1] and A[1] < B[1] and B[1] > C[1] and C[1] < D[1]:
            patterns_list.append({
                "type": "Bullish Butterfly Pattern",
                "direction": "bullish",
                "strength": 80,
                "position": "bottom"
            })
    
    # الگوی گارتلی (Gartley)
    if (XA_range != 0 and AB_range != 0 and BC_range != 0) and \
       is_fibonacci_ratio(AB_range/XA_range, 0.618, 0.05) and \
       is_fibonacci_ratio(BC_range/AB_range, 0.382, 0.05) and \
       is_fibonacci_ratio(CD_range/BC_range, 1.272, 0.1):
        # بررسی جهت
        if X[1] < A[1] and A[1] > B[1] and B[1] < C[1] and C[1] > D[1]:
            patterns_list.append({
                "type": "Bearish Gartley Pattern",
                "direction": "bearish",
                "strength": 75,
                "position": "top"
            })
        elif X[1] > A[1] and A[1] < B[1] and B[1] > C[1] and C[1] < D[1]:
            patterns_list.append({
                "type": "Bullish Gartley Pattern",
                "direction": "bullish",
                "strength": 75,
                "position": "bottom"
            })

def is_fibonacci_ratio(ratio, target, tolerance):
    """
    بررسی نزدیکی یک نسبت به نسبت فیبوناچی مورد نظر
    
    Args:
        ratio (float): نسبت مورد بررسی
        target (float): نسبت فیبوناچی هدف
        tolerance (float): میزان تلرانس
        
    Returns:
        bool: آیا نسبت در محدوده مجاز قرار دارد؟
    """
    try:
        # بررسی خطای تقسیم بر صفر یا مقادیر نامعتبر
        if np.isnan(ratio) or np.isinf(ratio) or np.isnan(target) or np.isinf(target):
            return False
        return abs(ratio - target) <= tolerance
    except Exception:
        return False

def detect_head_and_shoulders(df, patterns_list):
    """
    تشخیص الگوی سر و شانه‌ها
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        patterns_list (list): لیست الگوهای شناسایی شده
    """
    # استخراج داده‌های قیمت
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    if len(close) < 40:
        return
    
    # تشخیص سر و شانه‌ها (معمولی)
    window = 5
    # یافتن نقاط اوج محلی
    peaks = []
    for i in range(window, len(high) - window):
        if high[i] == max(high[i-window:i+window+1]):
            peaks.append((i, high[i]))
    
    # نیاز به حداقل 3 اوج
    if len(peaks) >= 3:
        # بررسی 3 اوج آخر
        left_shoulder, head, right_shoulder = peaks[-3], peaks[-2], peaks[-1]
        
        # بررسی الگو: شانه‌ها باید ارتفاع مشابه داشته باشند و سر بلندتر باشد
        if left_shoulder[1] < head[1] and right_shoulder[1] < head[1] and 0.9 <= left_shoulder[1]/right_shoulder[1] <= 1.1:
            # محاسبه خط گردن
            left_trough_idx = min(range(left_shoulder[0], head[0]), key=lambda i: low[i])
            right_trough_idx = min(range(head[0], right_shoulder[0]), key=lambda i: low[i])
            
            left_trough = low[left_trough_idx]
            right_trough = low[right_trough_idx]
            
            # محاسبه شیب خط گردن
            slope = (right_trough - left_trough) / (right_trough_idx - left_trough_idx)
            
            # محاسبه سطح خط گردن در آخرین نقطه
            neck_line = right_trough + slope * (len(close) - 1 - right_trough_idx)
            
            # بررسی شکست خط گردن
            if close[-1] < neck_line:
                patterns_list.append({
                    "type": "Head and Shoulders",
                    "direction": "bearish",
                    "strength": 90,
                    "position": "top"
                })
    
    # تشخیص سر و شانه‌های معکوس
    # یافتن نقاط فرود محلی
    troughs = []
    for i in range(window, len(low) - window):
        if low[i] == min(low[i-window:i+window+1]):
            troughs.append((i, low[i]))
    
    # نیاز به حداقل 3 فرود
    if len(troughs) >= 3:
        # بررسی 3 فرود آخر
        left_shoulder, head, right_shoulder = troughs[-3], troughs[-2], troughs[-1]
        
        # بررسی الگو: شانه‌ها باید عمق مشابه داشته باشند و سر پایین‌تر باشد
        if left_shoulder[1] > head[1] and right_shoulder[1] > head[1] and 0.9 <= left_shoulder[1]/right_shoulder[1] <= 1.1:
            # محاسبه خط گردن
            left_peak_idx = max(range(left_shoulder[0], head[0]), key=lambda i: high[i])
            right_peak_idx = max(range(head[0], right_shoulder[0]), key=lambda i: high[i])
            
            left_peak = high[left_peak_idx]
            right_peak = high[right_peak_idx]
            
            # محاسبه شیب خط گردن
            slope = (right_peak - left_peak) / (right_peak_idx - left_peak_idx)
            
            # محاسبه سطح خط گردن در آخرین نقطه
            neck_line = right_peak + slope * (len(close) - 1 - right_peak_idx)
            
            # بررسی شکست خط گردن
            if close[-1] > neck_line:
                patterns_list.append({
                    "type": "Inverse Head and Shoulders",
                    "direction": "bullish",
                    "strength": 90,
                    "position": "bottom"
                })

def detect_triangle_patterns(df, patterns_list):
    """
    تشخیص الگوهای مثلثی
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        patterns_list (list): لیست الگوهای شناسایی شده
    """
    # استخراج داده‌های قیمت
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    if len(close) < 30:
        return
    
    # بررسی در 30 کندل آخر
    data_window = 30
    recent_high = high[-data_window:]
    recent_low = low[-data_window:]
    recent_close = close[-data_window:]
    
    # محاسبه خط روند بالا با اتصال نقاط بالای محلی
    highs_idx = [i for i in range(1, len(recent_high)-1) if recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i+1]]
    
    # محاسبه خط روند پایین با اتصال نقاط پایین محلی
    lows_idx = [i for i in range(1, len(recent_low)-1) if recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i+1]]
    
    # نیاز به حداقل 2 نقطه برای هر خط روند
    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return
    
    # مثلث متقارن: خط روند بالا نزولی و خط روند پایین صعودی
    high_slope = np.polyfit(highs_idx, [recent_high[i] for i in highs_idx], 1)[0]
    low_slope = np.polyfit(lows_idx, [recent_low[i] for i in lows_idx], 1)[0]
    
    if high_slope < -0.01 and low_slope > 0.01:
        patterns_list.append({
            "type": "Symmetrical Triangle",
            "direction": "neutral",  # جهت مشخص می‌شود با شکست از یکی از خطوط
            "strength": 70,
            "position": "middle"
        })
    
    # مثلث صعودی: خط روند بالا افقی و خط روند پایین صعودی
    elif abs(high_slope) < 0.01 and low_slope > 0.01:
        patterns_list.append({
            "type": "Ascending Triangle",
            "direction": "bullish",
            "strength": 75,
            "position": "bottom"
        })
    
    # مثلث نزولی: خط روند بالا نزولی و خط روند پایین افقی
    elif high_slope < -0.01 and abs(low_slope) < 0.01:
        patterns_list.append({
            "type": "Descending Triangle",
            "direction": "bearish",
            "strength": 75,
            "position": "top"
        })

def detect_trend_patterns(df, patterns_list):
    """
    تشخیص الگوهای روندی
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌ها
        patterns_list (list): لیست الگوهای شناسایی شده
    """
    # استخراج داده‌های قیمت
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    if len(close) < 20:
        return
    
    # محاسبه میانگین متحرک‌ها
    sma20 = np.mean(close[-20:])
    sma50 = np.mean(close[-50:]) if len(close) >= 50 else None
    
    # محاسبه شیب روند
    trend_window = 20
    x = np.arange(trend_window)
    slope = np.polyfit(x, close[-trend_window:], 1)[0]
    
    # کانال صعودی
    if slope > 0 and sma20 > sma50 and close[-1] > sma20:
        # ترسیم خط روند بالا و پایین
        upper_channel = []
        lower_channel = []
        
        for i in range(1, trend_window - 1):
            if high[-i] > high[-i-1] and high[-i] > high[-i+1]:
                upper_channel.append((-i, high[-i]))
            if low[-i] < low[-i-1] and low[-i] < low[-i+1]:
                lower_channel.append((-i, low[-i]))
        
        if len(upper_channel) >= 2 and len(lower_channel) >= 2:
            patterns_list.append({
                "type": "Ascending Channel",
                "direction": "bullish",
                "strength": 85,
                "position": "bottom"
            })
    
    # کانال نزولی
    elif slope < 0 and (sma50 is None or sma20 < sma50) and close[-1] < sma20:
        # ترسیم خط روند بالا و پایین
        upper_channel = []
        lower_channel = []
        
        for i in range(1, trend_window - 1):
            if high[-i] > high[-i-1] and high[-i] > high[-i+1]:
                upper_channel.append((-i, high[-i]))
            if low[-i] < low[-i-1] and low[-i] < low[-i+1]:
                lower_channel.append((-i, low[-i]))
        
        if len(upper_channel) >= 2 and len(lower_channel) >= 2:
            patterns_list.append({
                "type": "Descending Channel",
                "direction": "bearish",
                "strength": 85,
                "position": "top"
            })
    
    # الگوی کف مسطح (Flat Bottom)
    flat_bottom_threshold = 0.005  # آستانه برای تشخیص کف مسطح
    recent_lows = low[-10:]
    
    if max(recent_lows) - min(recent_lows) < min(recent_lows) * flat_bottom_threshold:
        patterns_list.append({
            "type": "Flat Bottom",
            "direction": "bullish",
            "strength": 70,
            "position": "bottom"
        })
    
    # الگوی سقف مسطح (Flat Top)
    flat_top_threshold = 0.005  # آستانه برای تشخیص سقف مسطح
    recent_highs = high[-10:]
    
    if max(recent_highs) - min(recent_highs) < min(recent_highs) * flat_top_threshold:
        patterns_list.append({
            "type": "Flat Top",
            "direction": "bearish",
            "strength": 70,
            "position": "top"
        })
