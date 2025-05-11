"""
ماژول تشخیص و تحلیل پیشرفته سطوح حمایت و مقاومت 

این ماژول شامل توابع و کلاس‌های مورد نیاز برای تشخیص خودکار سطوح حمایت و مقاومت
با استفاده از روش‌های مختلف از جمله نقاط چرخش، حجم معاملات، مناطق تراکم قیمت، و
تحلیل فیبوناچی است.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupportResistanceDetector:
    """کلاس اصلی تشخیص سطوح حمایت و مقاومت"""
    
    def __init__(self, 
                sensitivity: float = 0.01, 
                density_threshold: float = 0.5,
                lookback_period: int = 200,
                strength_threshold: float = 0.7,
                max_points: int = 10):
        """
        مقداردهی اولیه تشخیص‌دهنده حمایت و مقاومت
        
        Args:
            sensitivity (float): حساسیت تشخیص (مقدار کمتر = سطوح بیشتر)
            density_threshold (float): آستانه تراکم برای تشخیص مناطق تراکم قیمت
            lookback_period (int): تعداد دوره‌های گذشته برای محاسبات
            strength_threshold (float): آستانه قدرت سطوح (0-1)
            max_points (int): حداکثر تعداد سطوح برای شناسایی
        """
        self.sensitivity = sensitivity
        self.density_threshold = density_threshold
        self.lookback_period = lookback_period
        self.strength_threshold = strength_threshold
        self.max_points = max_points
        
        logger.info(f"سیستم تشخیص حمایت و مقاومت با حساسیت {sensitivity} راه‌اندازی شد")
    
    def detect_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        تشخیص سطوح حمایت و مقاومت از دیتافریم قیمت
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت با ستون‌های OHLCV
            
        Returns:
            dict: دیکشنری حاوی سطوح حمایت و مقاومت و اطلاعات مرتبط
        """
        if df.empty or len(df) < 30:
            return {'support': [], 'resistance': [], 'pivot_points': {}, 'fibonacci_levels': {}}
        
        try:
            # محدود کردن داده‌ها به دوره مورد بررسی
            df_limited = df.iloc[-self.lookback_period:].copy() if len(df) > self.lookback_period else df.copy()
            
            # محاسبه سطوح با روش‌های مختلف
            pivot_levels = self._calculate_pivot_points(df_limited)
            fractal_levels = self._detect_fractals(df_limited)
            density_levels = self._detect_price_density(df_limited)
            volume_levels = self._detect_volume_clusters(df_limited)
            trend_levels = self._detect_trend_lines(df_limited)
            fib_levels = self._calculate_fibonacci_levels(df_limited)
            
            # ترکیب سطوح از روش‌های مختلف
            all_levels = self._combine_levels([
                fractal_levels, 
                density_levels, 
                volume_levels, 
                trend_levels, 
                list(pivot_levels.values())
            ])
            
            # جداسازی سطوح حمایت و مقاومت
            current_price = df_limited['close'].iloc[-1]
            support_levels = sorted([level for level in all_levels if level < current_price * 0.995], reverse=True)
            resistance_levels = sorted([level for level in all_levels if level > current_price * 1.005])
            
            # محدود کردن تعداد سطوح
            top_support = self._filter_close_levels(support_levels[:self.max_points])
            top_resistance = self._filter_close_levels(resistance_levels[:self.max_points])
            
            # ساخت نتیجه
            result = {
                'support': top_support,
                'resistance': top_resistance,
                'current_price': float(current_price),
                'pivot_points': {k: float(v) for k, v in pivot_levels.items()},
                'fibonacci_levels': {k: float(v) for k, v in fib_levels.items()}
            }
            
            # افزودن متادیتا به هر سطح
            result['support_metadata'] = [self._calculate_level_strength(df_limited, level) for level in top_support]
            result['resistance_metadata'] = [self._calculate_level_strength(df_limited, level) for level in top_resistance]
            
            return result
            
        except Exception as e:
            logger.error(f"خطا در تشخیص سطوح حمایت و مقاومت: {str(e)}")
            return {'support': [], 'resistance': [], 'pivot_points': {}, 'fibonacci_levels': {}}
    
    def _detect_fractals(self, df: pd.DataFrame) -> List[float]:
        """
        تشخیص سطوح با استفاده از فرکتال‌های ویلیامز
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            
        Returns:
            list: لیست سطوح شناسایی شده
        """
        levels = []
        
        # حداقل تعداد شمع‌ها برای تشخیص
        min_periods = 5
        if len(df) < min_periods:
            return levels
        
        # شناسایی فرکتال‌های بالا
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and 
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                levels.append(df['high'].iloc[i])
        
        # شناسایی فرکتال‌های پایین
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and 
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                levels.append(df['low'].iloc[i])
        
        return levels
    
    def _detect_price_density(self, df: pd.DataFrame) -> List[float]:
        """
        تشخیص سطوح با استفاده از تراکم قیمت
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            
        Returns:
            list: لیست سطوح شناسایی شده
        """
        levels = []
        
        # تعداد دسته‌بندی‌ها
        num_bins = int(len(df) / 10)
        if num_bins < 5:
            num_bins = 5
        
        # ایجاد هیستوگرام قیمت
        hist, bin_edges = np.histogram(df['close'], bins=num_bins)
        
        # محاسبه تراکم نسبی هر بازه
        density = hist / np.sum(hist)
        
        # شناسایی بازه‌های با تراکم بالا
        for i in range(len(density)):
            if density[i] > self.density_threshold:
                # استفاده از میانه بازه به عنوان سطح
                level = (bin_edges[i] + bin_edges[i+1]) / 2
                levels.append(level)
        
        return levels
    
    def _detect_volume_clusters(self, df: pd.DataFrame) -> List[float]:
        """
        تشخیص سطوح با استفاده از خوشه‌های حجم
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            
        Returns:
            list: لیست سطوح شناسایی شده
        """
        levels = []
        
        if 'volume' not in df.columns:
            return levels
        
        # محاسبه میانگین متحرک حجم
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        
        # محاسبه نسبت حجم به میانگین
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # شناسایی نقاط با حجم بالا (2 برابر میانگین)
        high_volume_points = df[df['volume_ratio'] > 2.0]
        
        if high_volume_points.empty:
            return levels
        
        # ایجاد هیستوگرام قیمت با وزن‌دهی حجم
        hist, bin_edges = np.histogram(high_volume_points['close'], bins=10, weights=high_volume_points['volume'])
        
        # شناسایی بازه‌های با حجم بالا
        for i in range(len(hist)):
            if hist[i] > df['volume'].mean() * 2:
                # استفاده از میانه بازه به عنوان سطح
                level = (bin_edges[i] + bin_edges[i+1]) / 2
                levels.append(level)
        
        return levels
    
    def _detect_trend_lines(self, df: pd.DataFrame) -> List[float]:
        """
        تشخیص سطوح با استفاده از خطوط روند
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            
        Returns:
            list: لیست سطوح شناسایی شده
        """
        levels = []
        
        if len(df) < 30:
            return levels
        
        try:
            # شناسایی قله‌ها و دره‌ها
            peaks = []
            troughs = []
            
            # تعداد کندل‌های مجاور برای تعیین قله و دره
            window = 5
            
            for i in range(window, len(df) - window):
                # قله
                if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
                    peaks.append((i, df['high'].iloc[i]))
                
                # دره
                if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
                    troughs.append((i, df['low'].iloc[i]))
            
            # باید حداقل دو قله یا دو دره وجود داشته باشد
            if len(peaks) >= 2:
                # خط روند نزولی از اتصال قله‌ها
                line_end = peaks[-1][1]
                levels.append(line_end)
            
            if len(troughs) >= 2:
                # خط روند صعودی از اتصال دره‌ها
                line_end = troughs[-1][1]
                levels.append(line_end)
            
            return levels
            
        except Exception as e:
            logger.warning(f"خطا در تشخیص خطوط روند: {str(e)}")
            return []
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        محاسبه نقاط پیووت
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            
        Returns:
            dict: دیکشنری نقاط پیووت
        """
        # استفاده از آخرین روز برای محاسبه نقاط پیووت
        last_idx = -1
        high = df['high'].iloc[last_idx]
        low = df['low'].iloc[last_idx]
        close = df['close'].iloc[last_idx]
        
        # محاسبه پیووت اصلی
        pivot = (high + low + close) / 3
        
        # محاسبه سطوح حمایت
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # محاسبه سطوح مقاومت
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        return {
            'pivot': pivot,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        }
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        محاسبه سطوح فیبوناچی
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            
        Returns:
            dict: دیکشنری سطوح فیبوناچی
        """
        # تعیین بالاترین و پایین‌ترین نقطه در بازه زمانی
        highest_high = df['high'].max()
        lowest_low = df['low'].min()
        price_range = highest_high - lowest_low
        
        # سطوح فیبوناچی استاندارد
        fib_23_6 = highest_high - price_range * 0.236
        fib_38_2 = highest_high - price_range * 0.382
        fib_50_0 = highest_high - price_range * 0.5
        fib_61_8 = highest_high - price_range * 0.618
        fib_78_6 = highest_high - price_range * 0.786
        
        # سطوح فیبوناچی توسعه یافته
        fib_127_2 = highest_high + price_range * 0.272
        fib_161_8 = highest_high + price_range * 0.618
        
        # سطوح بازگشتی عمیق‌تر
        fib_88_6 = highest_high - price_range * 0.886
        fib_100_0 = lowest_low
        
        return {
            'highest': highest_high,
            'lowest': lowest_low,
            'fib_0': highest_high,
            'fib_23.6': fib_23_6,
            'fib_38.2': fib_38_2,
            'fib_50.0': fib_50_0,
            'fib_61.8': fib_61_8,
            'fib_78.6': fib_78_6,
            'fib_88.6': fib_88_6,
            'fib_100.0': fib_100_0,
            'fib_127.2': fib_127_2,
            'fib_161.8': fib_161_8
        }
    
    def _combine_levels(self, level_lists: List[List[float]]) -> List[float]:
        """
        ترکیب سطوح از منابع مختلف
        
        Args:
            level_lists (list): لیست سطوح از روش‌های مختلف
            
        Returns:
            list: لیست ترکیبی سطوح
        """
        # تخت کردن لیست‌ها
        all_levels = []
        for level_list in level_lists:
            if isinstance(level_list, list):
                all_levels.extend(level_list)
            else:
                try:
                    all_levels.append(float(level_list))
                except (TypeError, ValueError):
                    pass
        
        return all_levels
    
    def _filter_close_levels(self, levels: List[float]) -> List[float]:
        """
        فیلتر کردن سطوح نزدیک به هم
        
        Args:
            levels (list): لیست سطوح
            
        Returns:
            list: لیست فیلتر شده
        """
        if not levels:
            return []
        
        filtered_levels = [levels[0]]
        
        for level in levels[1:]:
            # بررسی فاصله با سطوح قبلی
            too_close = False
            for existing_level in filtered_levels:
                # محاسبه فاصله نسبی
                relative_distance = abs(level - existing_level) / level
                
                if relative_distance < self.sensitivity:
                    too_close = True
                    break
            
            if not too_close:
                filtered_levels.append(level)
        
        return filtered_levels
    
    def _calculate_level_strength(self, df: pd.DataFrame, level: float) -> Dict[str, Any]:
        """
        محاسبه قدرت سطح حمایت/مقاومت
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            level (float): سطح حمایت/مقاومت
            
        Returns:
            dict: اطلاعات متا درباره قدرت سطح
        """
        # پارامترهای محاسبه قدرت
        touch_count = 0
        bounce_count = 0
        breakout_count = 0
        recent_touches = 0
        
        # تعریف محدوده سطح (تلرانس)
        level_tolerance = level * self.sensitivity
        
        for i in range(1, len(df)):
            # بررسی برخورد با سطح
            if ((df['high'].iloc[i-1] < level - level_tolerance and df['high'].iloc[i] >= level - level_tolerance) or
                (df['low'].iloc[i-1] > level + level_tolerance and df['low'].iloc[i] <= level + level_tolerance) or
                (df['low'].iloc[i] <= level + level_tolerance and df['high'].iloc[i] >= level - level_tolerance)):
                
                touch_count += 1
                
                # بررسی سن برخورد (30% آخر محسوب می‌شود)
                if i >= len(df) * 0.7:
                    recent_touches += 1
                
                # بررسی نوع برخورد (بازگشت یا شکست)
                if i < len(df) - 1:
                    next_candle = df.iloc[i+1]
                    
                    if (df['close'].iloc[i] < level and next_candle['close'] < level) or \
                       (df['close'].iloc[i] > level and next_candle['close'] > level):
                        # بازگشت از سطح
                        bounce_count += 1
                    else:
                        # شکست سطح
                        breakout_count += 1
        
        # محاسبه قدرت بر اساس تعداد برخوردها و بازگشت‌ها
        strength = 0.0
        
        if touch_count > 0:
            bounce_ratio = bounce_count / touch_count
            recent_ratio = recent_touches / max(1, touch_count)
            
            # فرمول قدرت: ترکیبی از تعداد برخوردها، نسبت بازگشت و تازگی برخوردها
            strength = min(1.0, (touch_count / 10) * 0.4 + bounce_ratio * 0.4 + recent_ratio * 0.2)
        
        # تعیین نوع سطح
        current_price = df['close'].iloc[-1]
        level_type = "resistance" if level > current_price else "support"
        
        return {
            'level': float(level),
            'strength': float(strength),
            'touches': touch_count,
            'bounces': bounce_count,
            'breakouts': breakout_count,
            'recent_touches': recent_touches,
            'type': level_type
        }
    
    def plot_levels(self, df: pd.DataFrame, levels: Dict[str, Any]) -> Any:
        """
        رسم نمودار با سطوح حمایت و مقاومت
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            levels (dict): اطلاعات سطوح
            
        Returns:
            matplotlib.figure.Figure: نمودار
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # رسم نمودار قیمت
        ax.plot(df.index, df['close'], label='قیمت بسته شدن', color='black')
        
        # رسم سطوح حمایت
        support_levels = levels.get('support', [])
        for level in support_levels:
            ax.axhline(y=level, color='green', linestyle='--', alpha=0.7, 
                      label=f'حمایت: {level:.2f}')
        
        # رسم سطوح مقاومت
        resistance_levels = levels.get('resistance', [])
        for level in resistance_levels:
            ax.axhline(y=level, color='red', linestyle='--', alpha=0.7, 
                      label=f'مقاومت: {level:.2f}')
        
        # رسم نقطه فعلی
        current_price = levels.get('current_price', df['close'].iloc[-1])
        ax.scatter(df.index[-1], current_price, color='blue', s=100, zorder=5, 
                  label=f'قیمت فعلی: {current_price:.2f}')
        
        ax.set_title('سطوح حمایت و مقاومت')
        ax.set_xlabel('تاریخ')
        ax.set_ylabel('قیمت')
        ax.grid(True, alpha=0.3)
        
        # حذف برچسب‌های تکراری
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        ax.legend(unique_handles, unique_labels, loc='best')
        
        plt.tight_layout()
        return fig


def get_support_resistance_levels(df: pd.DataFrame, sensitivity: float = 0.01, max_levels: int = 5) -> Dict[str, Any]:
    """
    دریافت سطوح حمایت و مقاومت برای دیتافریم قیمت
    
    Args:
        df (pd.DataFrame): دیتافریم قیمت
        sensitivity (float): حساسیت تشخیص
        max_levels (int): حداکثر تعداد سطوح
        
    Returns:
        dict: اطلاعات سطوح حمایت و مقاومت
    """
    detector = SupportResistanceDetector(sensitivity=sensitivity, max_points=max_levels)
    return detector.detect_support_resistance(df)


def get_trading_range(df: pd.DataFrame, lookback_period: int = 30) -> Dict[str, float]:
    """
    محاسبه محدوده معاملاتی بر اساس قیمت‌های اخیر
    
    Args:
        df (pd.DataFrame): دیتافریم قیمت
        lookback_period (int): تعداد دوره‌های گذشته
        
    Returns:
        dict: محدوده معاملاتی
    """
    if df.empty or len(df) < lookback_period:
        return {'upper': 0.0, 'lower': 0.0, 'middle': 0.0, 'range_percent': 0.0}
    
    # محدود کردن داده‌ها به دوره مورد بررسی
    recent_df = df.iloc[-lookback_period:]
    
    # محاسبه حد بالا و پایین
    upper = recent_df['high'].max()
    lower = recent_df['low'].min()
    middle = (upper + lower) / 2
    
    # محاسبه درصد محدوده
    range_percent = ((upper - lower) / middle) * 100
    
    # خروجی
    return {
        'upper': float(upper),
        'lower': float(lower),
        'middle': float(middle),
        'range_percent': float(range_percent)
    }


def analyze_level_breakouts(df: pd.DataFrame, levels: Dict[str, Any]) -> Dict[str, Any]:
    """
    تحلیل شکست سطوح حمایت و مقاومت
    
    Args:
        df (pd.DataFrame): دیتافریم قیمت
        levels (dict): اطلاعات سطوح
        
    Returns:
        dict: اطلاعات شکست‌ها
    """
    if df.empty or len(df) < 10:
        return {'breakouts': [], 'pending_breakouts': []}
    
    # دریافت سطوح
    support_levels = levels.get('support', [])
    resistance_levels = levels.get('resistance', [])
    current_price = df['close'].iloc[-1]
    
    # پارامترهای تحلیل
    breakout_threshold = 0.01  # 1% فاصله برای تأیید شکست
    volume_confirmation = True  # تأیید با حجم
    
    # لیست شکست‌های تأیید شده
    confirmed_breakouts = []
    
    # لیست شکست‌های در حال وقوع
    pending_breakouts = []
    
    # بررسی شکست سطوح مقاومت
    for level in resistance_levels:
        # شکست مقاومت (قیمت بالاتر از سطح)
        if current_price > level * (1 + breakout_threshold):
            # بررسی تأیید با حجم
            if 'volume' in df.columns and volume_confirmation:
                recent_volume = df['volume'].iloc[-5:].mean()
                past_volume = df['volume'].iloc[-20:-5].mean()
                
                if recent_volume > past_volume * 1.5:
                    # شکست تأیید شده با حجم
                    confirmed_breakouts.append({
                        'level': float(level),
                        'type': 'resistance',
                        'breakout_direction': 'up',
                        'confirmation': 'strong',
                        'distance_percent': float(((current_price / level) - 1) * 100)
                    })
                else:
                    # شکست بدون تأیید حجم
                    confirmed_breakouts.append({
                        'level': float(level),
                        'type': 'resistance',
                        'breakout_direction': 'up',
                        'confirmation': 'weak',
                        'distance_percent': float(((current_price / level) - 1) * 100)
                    })
        # نزدیک به شکست
        elif current_price > level * 0.99:
            pending_breakouts.append({
                'level': float(level),
                'type': 'resistance',
                'breakout_direction': 'up',
                'distance_percent': float(((current_price / level) - 1) * 100)
            })
    
    # بررسی شکست سطوح حمایت
    for level in support_levels:
        # شکست حمایت (قیمت پایین‌تر از سطح)
        if current_price < level * (1 - breakout_threshold):
            # بررسی تأیید با حجم
            if 'volume' in df.columns and volume_confirmation:
                recent_volume = df['volume'].iloc[-5:].mean()
                past_volume = df['volume'].iloc[-20:-5].mean()
                
                if recent_volume > past_volume * 1.5:
                    # شکست تأیید شده با حجم
                    confirmed_breakouts.append({
                        'level': float(level),
                        'type': 'support',
                        'breakout_direction': 'down',
                        'confirmation': 'strong',
                        'distance_percent': float(((level / current_price) - 1) * 100)
                    })
                else:
                    # شکست بدون تأیید حجم
                    confirmed_breakouts.append({
                        'level': float(level),
                        'type': 'support',
                        'breakout_direction': 'down',
                        'confirmation': 'weak',
                        'distance_percent': float(((level / current_price) - 1) * 100)
                    })
        # نزدیک به شکست
        elif current_price < level * 1.01:
            pending_breakouts.append({
                'level': float(level),
                'type': 'support',
                'breakout_direction': 'down',
                'distance_percent': float(((level / current_price) - 1) * 100)
            })
    
    # مرتب‌سازی بر اساس فاصله
    confirmed_breakouts.sort(key=lambda x: x['distance_percent'], reverse=True)
    pending_breakouts.sort(key=lambda x: abs(x['distance_percent']))
    
    return {
        'breakouts': confirmed_breakouts,
        'pending_breakouts': pending_breakouts,
        'has_confirmed_breakouts': len(confirmed_breakouts) > 0,
        'has_pending_breakouts': len(pending_breakouts) > 0
    }


def suggest_entry_points(df: pd.DataFrame, levels: Dict[str, Any], trend: str = "unknown") -> Dict[str, Any]:
    """
    پیشنهاد نقاط ورود بر اساس سطوح و روند
    
    Args:
        df (pd.DataFrame): دیتافریم قیمت
        levels (dict): اطلاعات سطوح
        trend (str): روند فعلی بازار (صعودی، نزولی، نوسانی)
        
    Returns:
        dict: پیشنهادات ورود
    """
    if df.empty or len(df) < 10:
        return {'entries': [], 'stop_levels': [], 'targets': []}
    
    # قیمت فعلی
    current_price = df['close'].iloc[-1]
    
    # دریافت سطوح
    support_levels = levels.get('support', [])
    resistance_levels = levels.get('resistance', [])
    
    # لیست پیشنهادات
    entry_points = []
    stop_levels = []
    target_levels = []
    
    # تنظیم استراتژی بر اساس روند
    if 'صعودی' in trend.lower():
        # استراتژی روند صعودی - خرید در حمایت‌ها
        for i, level in enumerate(support_levels):
            entry_points.append({
                'price': float(level),
                'type': 'buy',
                'description': f"خرید در سطح حمایت {i+1}",
                'source': 'support'
            })
            
            # حد ضرر زیر سطح حمایت بعدی
            if i + 1 < len(support_levels):
                stop_loss = support_levels[i+1] * 0.99
            else:
                stop_loss = level * 0.97  # 3% پایین‌تر
            
            stop_levels.append({
                'price': float(stop_loss),
                'type': 'stop_loss',
                'description': f"حد ضرر برای ورود در سطح حمایت {i+1}",
                'risk_percent': float(((level / stop_loss) - 1) * 100)
            })
            
            # اهداف قیمتی
            for j, res_level in enumerate(resistance_levels):
                if res_level > level * 1.02:  # حداقل 2% بالاتر
                    target_levels.append({
                        'price': float(res_level),
                        'type': 'take_profit',
                        'description': f"هدف {j+1} برای ورود در سطح حمایت {i+1}",
                        'profit_percent': float(((res_level / level) - 1) * 100)
                    })
        
        # خرید در شکست مقاومت
        if resistance_levels:
            breakout_level = resistance_levels[0] * 1.01  # 1% بالاتر برای تأیید
            entry_points.append({
                'price': float(breakout_level),
                'type': 'buy',
                'description': "خرید در شکست سطح مقاومت",
                'source': 'breakout'
            })
            
            # حد ضرر زیر سطح مقاومت شکسته شده
            stop_loss = resistance_levels[0] * 0.99
            stop_levels.append({
                'price': float(stop_loss),
                'type': 'stop_loss',
                'description': "حد ضرر برای خرید در شکست مقاومت",
                'risk_percent': float(((breakout_level / stop_loss) - 1) * 100)
            })
            
            # هدف قیمتی برای شکست (به اندازه فاصله از حمایت قبلی)
            if support_levels and resistance_levels:
                range_height = resistance_levels[0] - support_levels[0]
                target_price = breakout_level + range_height
                
                target_levels.append({
                    'price': float(target_price),
                    'type': 'take_profit',
                    'description': "هدف قیمتی پس از شکست مقاومت",
                    'profit_percent': float(((target_price / breakout_level) - 1) * 100)
                })
    
    elif 'نزولی' in trend.lower():
        # استراتژی روند نزولی - فروش در مقاومت‌ها
        for i, level in enumerate(resistance_levels):
            entry_points.append({
                'price': float(level),
                'type': 'sell',
                'description': f"فروش در سطح مقاومت {i+1}",
                'source': 'resistance'
            })
            
            # حد ضرر بالای سطح مقاومت بعدی
            if i + 1 < len(resistance_levels):
                stop_loss = resistance_levels[i+1] * 1.01
            else:
                stop_loss = level * 1.03  # 3% بالاتر
            
            stop_levels.append({
                'price': float(stop_loss),
                'type': 'stop_loss',
                'description': f"حد ضرر برای ورود در سطح مقاومت {i+1}",
                'risk_percent': float(((stop_loss / level) - 1) * 100)
            })
            
            # اهداف قیمتی
            for j, sup_level in enumerate(support_levels):
                if sup_level < level * 0.98:  # حداقل 2% پایین‌تر
                    target_levels.append({
                        'price': float(sup_level),
                        'type': 'take_profit',
                        'description': f"هدف {j+1} برای ورود در سطح مقاومت {i+1}",
                        'profit_percent': float(((level / sup_level) - 1) * 100)
                    })
        
        # فروش در شکست حمایت
        if support_levels:
            breakout_level = support_levels[0] * 0.99  # 1% پایین‌تر برای تأیید
            entry_points.append({
                'price': float(breakout_level),
                'type': 'sell',
                'description': "فروش در شکست سطح حمایت",
                'source': 'breakout'
            })
            
            # حد ضرر بالای سطح حمایت شکسته شده
            stop_loss = support_levels[0] * 1.01
            stop_levels.append({
                'price': float(stop_loss),
                'type': 'stop_loss',
                'description': "حد ضرر برای فروش در شکست حمایت",
                'risk_percent': float(((stop_loss / breakout_level) - 1) * 100)
            })
            
            # هدف قیمتی برای شکست (به اندازه فاصله از مقاومت قبلی)
            if support_levels and resistance_levels:
                range_height = resistance_levels[0] - support_levels[0]
                target_price = breakout_level - range_height
                
                if target_price > 0:  # اطمینان از مثبت بودن هدف
                    target_levels.append({
                        'price': float(target_price),
                        'type': 'take_profit',
                        'description': "هدف قیمتی پس از شکست حمایت",
                        'profit_percent': float(((breakout_level / target_price) - 1) * 100)
                    })
    
    else:  # روند نوسانی
        # استراتژی نوسانی - خرید در پایین و فروش در بالا
        if support_levels:
            # خرید در حمایت
            entry_points.append({
                'price': float(support_levels[0]),
                'type': 'buy',
                'description': "خرید در سطح حمایت اصلی",
                'source': 'support'
            })
            
            # حد ضرر
            stop_loss = support_levels[0] * 0.97
            stop_levels.append({
                'price': float(stop_loss),
                'type': 'stop_loss',
                'description': "حد ضرر برای خرید در حمایت",
                'risk_percent': float(((support_levels[0] / stop_loss) - 1) * 100)
            })
        
        if resistance_levels:
            # فروش در مقاومت
            entry_points.append({
                'price': float(resistance_levels[0]),
                'type': 'sell',
                'description': "فروش در سطح مقاومت اصلی",
                'source': 'resistance'
            })
            
            # حد ضرر
            stop_loss = resistance_levels[0] * 1.03
            stop_levels.append({
                'price': float(stop_loss),
                'type': 'stop_loss',
                'description': "حد ضرر برای فروش در مقاومت",
                'risk_percent': float(((stop_loss / resistance_levels[0]) - 1) * 100)
            })
        
        # تعیین اهداف قیمتی
        if support_levels and resistance_levels:
            # هدف برای خرید در حمایت = مقاومت
            target_levels.append({
                'price': float(resistance_levels[0]),
                'type': 'take_profit',
                'description': "هدف خرید (محدوده بالایی)",
                'profit_percent': float(((resistance_levels[0] / support_levels[0]) - 1) * 100)
            })
            
            # هدف برای فروش در مقاومت = حمایت
            target_levels.append({
                'price': float(support_levels[0]),
                'type': 'take_profit',
                'description': "هدف فروش (محدوده پایینی)",
                'profit_percent': float(((resistance_levels[0] / support_levels[0]) - 1) * 100)
            })
    
    # محاسبه نسبت‌های ریسک به پاداش
    entry_stop_target_ratios = []
    
    for entry in entry_points:
        entry_price = entry['price']
        
        # یافتن مناسب‌ترین حد ضرر
        best_stop = None
        for stop in stop_levels:
            if (entry['type'] == 'buy' and stop['price'] < entry_price) or \
               (entry['type'] == 'sell' and stop['price'] > entry_price):
                if best_stop is None or \
                   (entry['type'] == 'buy' and stop['price'] > best_stop['price']) or \
                   (entry['type'] == 'sell' and stop['price'] < best_stop['price']):
                    best_stop = stop
        
        # یافتن مناسب‌ترین هدف سود
        best_target = None
        for target in target_levels:
            if (entry['type'] == 'buy' and target['price'] > entry_price) or \
               (entry['type'] == 'sell' and target['price'] < entry_price):
                if best_target is None or \
                   (entry['type'] == 'buy' and target['price'] < best_target['price']) or \
                   (entry['type'] == 'sell' and target['price'] > best_target['price']):
                    best_target = target
        
        # محاسبه نسبت ریسک به پاداش
        if best_stop and best_target:
            if entry['type'] == 'buy':
                risk = entry_price - best_stop['price']
                reward = best_target['price'] - entry_price
            else:  # sell
                risk = best_stop['price'] - entry_price
                reward = entry_price - best_target['price']
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            entry_stop_target_ratios.append({
                'entry': entry_price,
                'stop': best_stop['price'],
                'target': best_target['price'],
                'risk_reward_ratio': float(risk_reward_ratio),
                'type': entry['type'],
                'description': entry['description']
            })
    
    # مرتب‌سازی بر اساس نسبت ریسک به پاداش
    entry_stop_target_ratios.sort(key=lambda x: x['risk_reward_ratio'], reverse=True)
    
    return {
        'entries': entry_points,
        'stop_levels': stop_levels,
        'targets': target_levels,
        'best_setups': entry_stop_target_ratios[:3],  # بهترین 3 استراتژی
        'current_price': float(current_price)
    }