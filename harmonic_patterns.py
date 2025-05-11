"""
ماژول تشخیص الگوهای هارمونیک قیمت در بازار ارزهای دیجیتال

این ماژول قادر به شناسایی خودکار الگوهای هارمونیک مانند Gartley، Butterfly، Bat، Crab و الگوهای سطوح فیبوناچی
با دقت بالا می‌باشد. برای هر الگو، اطلاعات کامل نقاط ورود، هدف و حد ضرر ارائه می‌شود.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HarmonicPatterns:
    """کلاس تشخیص الگوهای هارمونیک"""
    
    # نسبت‌های فیبوناچی استاندارد
    FIBONACCI_RATIOS = {
        'Gartley': {
            'XA': 1.0,
            'AB': 0.618,
            'BC': 0.382,
            'CD': 1.272,
            'AD': 0.786
        },
        'Butterfly': {
            'XA': 1.0,
            'AB': 0.786,
            'BC': 0.382,
            'CD': 1.618,
            'AD': 1.27
        },
        'Bat': {
            'XA': 1.0,
            'AB': 0.382,
            'BC': 0.382,
            'CD': 1.618,
            'AD': 0.886
        },
        'Crab': {
            'XA': 1.0,
            'AB': 0.382,
            'BC': 0.618,
            'CD': 3.618,
            'AD': 1.618
        },
        'Shark': {
            'XA': 1.0,
            'AB': 0.5,
            'BC': 1.13,
            'CD': 1.618,
            'AD': 0.886
        },
        'Cypher': {
            'XA': 1.0,
            'AB': 0.382,
            'BC': 1.13,
            'CD': 1.414,
            'AD': 0.786
        }
    }
    
    # محدوده قابل قبول برای هر نسبت (درصد انحراف مجاز)
    RATIO_TOLERANCE = 0.05
    
    def __init__(self, max_patterns: int = 5, tolerance: float = 0.05):
        """
        مقداردهی اولیه
        
        Args:
            max_patterns (int): حداکثر تعداد الگوهای بازگشتی
            tolerance (float): درصد انحراف مجاز از نسبت‌های دقیق
        """
        self.max_patterns = max_patterns
        self.tolerance = tolerance
        logger.info(f"سیستم تشخیص الگوهای هارمونیک با تلرانس {tolerance * 100}% راه‌اندازی شد")
    
    def _is_swing_high(self, df: pd.DataFrame, index: int, window: int = 5) -> bool:
        """
        تشخیص نقطه سوئینگ بالا
        
        Args:
            df: دیتافریم قیمت
            index: اندیس مورد بررسی
            window: پنجره بررسی اطراف نقطه
            
        Returns:
            bool: آیا نقطه سوئینگ بالاست
        """
        if index < window or index >= len(df) - window:
            return False
            
        price = df['high'].iloc[index]
        
        # بررسی اینکه آیا این قیمت از همه قیمت‌های پنجره بالاتر است
        for i in range(index - window, index + window + 1):
            if i != index and df['high'].iloc[i] >= price:
                return False
                
        return True
    
    def _is_swing_low(self, df: pd.DataFrame, index: int, window: int = 5) -> bool:
        """
        تشخیص نقطه سوئینگ پایین
        
        Args:
            df: دیتافریم قیمت
            index: اندیس مورد بررسی
            window: پنجره بررسی اطراف نقطه
            
        Returns:
            bool: آیا نقطه سوئینگ پایین است
        """
        if index < window or index >= len(df) - window:
            return False
            
        price = df['low'].iloc[index]
        
        # بررسی اینکه آیا این قیمت از همه قیمت‌های پنجره پایین‌تر است
        for i in range(index - window, index + window + 1):
            if i != index and df['low'].iloc[i] <= price:
                return False
                
        return True
    
    def _find_swing_points(self, df: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """
        پیدا کردن تمام نقاط سوئینگ
        
        Args:
            df: دیتافریم قیمت
            window: پنجره بررسی اطراف نقطه
            
        Returns:
            tuple: دو لیست از اندیس‌های نقاط سوئینگ بالا و پایین
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            if self._is_swing_high(df, i, window):
                swing_highs.append(i)
            elif self._is_swing_low(df, i, window):
                swing_lows.append(i)
                
        return swing_highs, swing_lows
    
    def _calculate_ratio(self, price1: float, price2: float) -> float:
        """
        محاسبه نسبت بین دو قیمت
        
        Args:
            price1: قیمت اول
            price2: قیمت دوم
            
        Returns:
            float: نسبت محاسبه شده
        """
        if price1 == 0 or price2 == 0:
            return 0
            
        return abs(price2 - price1) / abs(price1)
    
    def _is_valid_ratio(self, actual: float, expected: float) -> bool:
        """
        بررسی معتبر بودن نسبت با توجه به تلرانس
        
        Args:
            actual: نسبت واقعی
            expected: نسبت مورد انتظار
            
        Returns:
            bool: آیا نسبت معتبر است
        """
        return abs(actual - expected) <= self.tolerance
    
    def _is_valid_pattern(self, points: List[Tuple[int, float]], pattern_type: str) -> bool:
        """
        بررسی معتبر بودن یک الگو
        
        Args:
            points: نقاط الگو (X, A, B, C, D)
            pattern_type: نوع الگو
            
        Returns:
            bool: آیا الگو معتبر است
        """
        if pattern_type not in self.FIBONACCI_RATIOS:
            return False
            
        # استخراج قیمت‌ها
        x_price = points[0][1]
        a_price = points[1][1]
        b_price = points[2][1]
        c_price = points[3][1]
        d_price = points[4][1]
        
        # محاسبه نسبت‌ها
        xa_ratio = self._calculate_ratio(x_price, a_price)
        ab_ratio = self._calculate_ratio(a_price, b_price)
        bc_ratio = self._calculate_ratio(b_price, c_price)
        cd_ratio = self._calculate_ratio(c_price, d_price)
        ad_ratio = self._calculate_ratio(a_price, d_price)
        
        # بررسی نسبت‌ها با توجه به الگو
        pattern_ratios = self.FIBONACCI_RATIOS[pattern_type]
        
        return (self._is_valid_ratio(xa_ratio, pattern_ratios['XA']) and
                self._is_valid_ratio(ab_ratio, pattern_ratios['AB']) and
                self._is_valid_ratio(bc_ratio, pattern_ratios['BC']) and
                self._is_valid_ratio(cd_ratio, pattern_ratios['CD']) and
                self._is_valid_ratio(ad_ratio, pattern_ratios['AD']))
    
    def _validate_pattern_direction(self, points: List[Tuple[int, float]], pattern_type: str) -> bool:
        """
        بررسی صحیح بودن جهت الگو
        
        Args:
            points: نقاط الگو (X, A, B, C, D)
            pattern_type: نوع الگو
            
        Returns:
            bool: آیا جهت الگو صحیح است
        """
        # استخراج قیمت‌ها
        x_price = points[0][1]
        a_price = points[1][1]
        b_price = points[2][1]
        c_price = points[3][1]
        d_price = points[4][1]
        
        # بررسی صعودی یا نزولی بودن الگو
        is_bullish = x_price < a_price
        
        # بررسی الگوی صعودی
        if is_bullish:
            return (a_price > x_price and
                    b_price < a_price and
                    c_price > b_price and
                    d_price < c_price)
        # بررسی الگوی نزولی
        else:
            return (a_price < x_price and
                    b_price > a_price and
                    c_price < b_price and
                    d_price > c_price)
    
    def detect_harmonic_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        تشخیص الگوهای هارمونیک در دیتافریم
        
        Args:
            df: دیتافریم قیمت
            
        Returns:
            list: لیست الگوهای شناسایی شده به همراه جزئیات
        """
        if df.empty or len(df) < 20:
            logger.warning("داده‌های ناکافی برای تشخیص الگوی هارمونیک")
            return []
            
        patterns = []
        
        # پیدا کردن نقاط سوئینگ
        swing_highs, swing_lows = self._find_swing_points(df)
        
        # ترکیب نقاط سوئینگ و مرتب‌سازی بر اساس زمان
        all_swings = [(idx, df['high'].iloc[idx], 'high') for idx in swing_highs]
        all_swings.extend([(idx, df['low'].iloc[idx], 'low') for idx in swing_lows])
        all_swings.sort(key=lambda x: x[0])
        
        # بررسی ترکیبات مختلف برای یافتن الگوها
        for i in range(len(all_swings) - 4):
            for pattern_type in self.FIBONACCI_RATIOS.keys():
                # نقاط کاندید (X, A, B, C, D)
                x = (all_swings[i][0], all_swings[i][1])
                a = (all_swings[i+1][0], all_swings[i+1][1])
                b = (all_swings[i+2][0], all_swings[i+2][1])
                c = (all_swings[i+3][0], all_swings[i+3][1])
                d = (all_swings[i+4][0], all_swings[i+4][1])
                
                candidate_points = [x, a, b, c, d]
                
                # بررسی معتبر بودن الگو و جهت آن
                if (self._is_valid_pattern(candidate_points, pattern_type) and
                    self._validate_pattern_direction(candidate_points, pattern_type)):
                    
                    # نمایش با فرمت مناسب
                    points = [
                        {'name': 'X', 'index': x[0], 'price': x[1], 'time': df.index[x[0]]},
                        {'name': 'A', 'index': a[0], 'price': a[1], 'time': df.index[a[0]]},
                        {'name': 'B', 'index': b[0], 'price': b[1], 'time': df.index[b[0]]},
                        {'name': 'C', 'index': c[0], 'price': c[1], 'time': df.index[c[0]]},
                        {'name': 'D', 'index': d[0], 'price': d[1], 'time': df.index[d[0]]}
                    ]
                    
                    # تعیین جهت الگو
                    direction = "صعودی" if x[1] < a[1] else "نزولی"
                    
                    # محاسبه قیمت‌های هدف و حد ضرر
                    entry_price = d[1]
                    stop_loss = c[1]
                    target_price = a[1]  # هدف اول معمولاً نقطه A است
                    risk_reward = abs(target_price - entry_price) / abs(entry_price - stop_loss)
                    
                    # افزودن به لیست الگوها
                    pattern = {
                        'type': pattern_type,
                        'direction': direction,
                        'points': points,
                        'time_discovered': df.index[-1],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target_1': target_price,
                        'target_2': (2 * (target_price - entry_price)) + entry_price,  # هدف دوم
                        'target_3': (3 * (target_price - entry_price)) + entry_price,  # هدف سوم
                        'risk_reward': risk_reward,
                        'confidence': self._calculate_pattern_confidence(candidate_points, pattern_type)
                    }
                    
                    patterns.append(pattern)
                    
                    # محدود کردن تعداد الگوهای بازگشتی
                    if len(patterns) >= self.max_patterns:
                        break
                        
        # مرتب‌سازی الگوها بر اساس اعتماد
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"{len(patterns)} الگوی هارمونیک شناسایی شد")
        return patterns[:self.max_patterns]
    
    def _calculate_pattern_confidence(self, points: List[Tuple[int, float]], pattern_type: str) -> int:
        """
        محاسبه درصد اطمینان الگو
        
        Args:
            points: نقاط الگو (X, A, B, C, D)
            pattern_type: نوع الگو
            
        Returns:
            int: درصد اطمینان (0-100)
        """
        # استخراج قیمت‌ها
        x_price = points[0][1]
        a_price = points[1][1]
        b_price = points[2][1]
        c_price = points[3][1]
        d_price = points[4][1]
        
        # محاسبه نسبت‌ها
        xa_ratio = self._calculate_ratio(x_price, a_price)
        ab_ratio = self._calculate_ratio(a_price, b_price)
        bc_ratio = self._calculate_ratio(b_price, c_price)
        cd_ratio = self._calculate_ratio(c_price, d_price)
        ad_ratio = self._calculate_ratio(a_price, d_price)
        
        # بررسی نسبت‌ها با توجه به الگو
        pattern_ratios = self.FIBONACCI_RATIOS[pattern_type]
        
        # محاسبه دقت هر نسبت (100% منهای درصد انحراف)
        xa_accuracy = 100 * (1 - min(abs(xa_ratio - pattern_ratios['XA']) / pattern_ratios['XA'], 1))
        ab_accuracy = 100 * (1 - min(abs(ab_ratio - pattern_ratios['AB']) / pattern_ratios['AB'], 1))
        bc_accuracy = 100 * (1 - min(abs(bc_ratio - pattern_ratios['BC']) / pattern_ratios['BC'], 1))
        cd_accuracy = 100 * (1 - min(abs(cd_ratio - pattern_ratios['CD']) / pattern_ratios['CD'], 1))
        ad_accuracy = 100 * (1 - min(abs(ad_ratio - pattern_ratios['AD']) / pattern_ratios['AD'], 1))
        
        # میانگین دقت‌ها
        avg_accuracy = (xa_accuracy + ab_accuracy + bc_accuracy + cd_accuracy + ad_accuracy) / 5
        
        return int(avg_accuracy)
    
    def plot_harmonic_pattern(self, df: pd.DataFrame, pattern: Dict[str, Any]) -> Optional[io.BytesIO]:
        """
        رسم الگوی هارمونیک روی نمودار
        
        Args:
            df: دیتافریم قیمت
            pattern: دیکشنری الگوی هارمونیک
            
        Returns:
            io.BytesIO: بافر تصویر نمودار
        """
        try:
            # تنظیم نمودار
            plt.figure(figsize=(12, 8))
            plt.grid(True)
            
            # رسم نمودار قیمت
            plt.plot(df.index, df['close'], color='black', alpha=0.3, label='قیمت')
            
            # رسم کندل‌ها
            for i in range(len(df)):
                # رسم کندل‌ها با رنگ‌های مناسب
                if df['close'].iloc[i] >= df['open'].iloc[i]:
                    color = 'green'
                else:
                    color = 'red'
                    
                plt.plot([df.index[i], df.index[i]], [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1)
                plt.plot([df.index[i], df.index[i]], [df['open'].iloc[i], df['close'].iloc[i]], color=color, linewidth=3)
            
            # استخراج نقاط الگو
            points = pattern['points']
            x_point = (points[0]['time'], points[0]['price'])
            a_point = (points[1]['time'], points[1]['price'])
            b_point = (points[2]['time'], points[2]['price'])
            c_point = (points[3]['time'], points[3]['price'])
            d_point = (points[4]['time'], points[4]['price'])
            
            # رسم خطوط اتصال نقاط
            plt.plot([x_point[0], a_point[0]], [x_point[1], a_point[1]], 'b-', linewidth=2)
            plt.plot([a_point[0], b_point[0]], [a_point[1], b_point[1]], 'g-', linewidth=2)
            plt.plot([b_point[0], c_point[0]], [b_point[1], c_point[1]], 'y-', linewidth=2)
            plt.plot([c_point[0], d_point[0]], [c_point[1], d_point[1]], 'r-', linewidth=2)
            
            # رسم نقاط
            plt.scatter(x_point[0], x_point[1], color='blue', s=100, label='X')
            plt.scatter(a_point[0], a_point[1], color='green', s=100, label='A')
            plt.scatter(b_point[0], b_point[1], color='orange', s=100, label='B')
            plt.scatter(c_point[0], c_point[1], color='purple', s=100, label='C')
            plt.scatter(d_point[0], d_point[1], color='red', s=100, label='D')
            
            # افزودن متن به نقاط
            plt.annotate('X', (x_point[0], x_point[1]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate('A', (a_point[0], a_point[1]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate('B', (b_point[0], b_point[1]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate('C', (c_point[0], c_point[1]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate('D', (d_point[0], d_point[1]), textcoords="offset points", xytext=(0,10), ha='center')
            
            # افزودن خطوط هدف و حد ضرر
            plt.axhline(y=pattern['entry_price'], color='black', linestyle='--', label='قیمت ورود')
            plt.axhline(y=pattern['stop_loss'], color='red', linestyle='--', label='حد ضرر')
            plt.axhline(y=pattern['target_1'], color='green', linestyle='--', label='هدف اول')
            plt.axhline(y=pattern['target_2'], color='blue', linestyle='--', label='هدف دوم')
            plt.axhline(y=pattern['target_3'], color='purple', linestyle='--', label='هدف سوم')
            
            # تنظیمات نهایی نمودار
            title = f"الگوی هارمونیک {pattern['type']} ({pattern['direction']}) - اطمینان: {pattern['confidence']}%"
            plt.title(title)
            plt.xlabel('تاریخ')
            plt.ylabel('قیمت')
            plt.legend()
            plt.tight_layout()
            
            # ذخیره نمودار در یک بافر
            buf = io.BytesIO()
            plt.savefig(buf, format='jpg', dpi=100)
            buf.seek(0)
            plt.close()
            
            return buf
            
        except Exception as e:
            logger.error(f"خطا در رسم الگوی هارمونیک: {str(e)}")
            return None
    
    def generate_pattern_description(self, pattern: Dict[str, Any]) -> str:
        """
        تولید توضیح متنی برای الگوی هارمونیک
        
        Args:
            pattern: دیکشنری الگوی هارمونیک
            
        Returns:
            str: توضیح الگو
        """
        direction = pattern['direction']
        pattern_type = pattern['type']
        points = pattern['points']
        
        # زمان نقاط مهم
        x_time = points[0]['time'].strftime('%Y-%m-%d %H:%M')
        a_time = points[1]['time'].strftime('%Y-%m-%d %H:%M')
        b_time = points[2]['time'].strftime('%Y-%m-%d %H:%M')
        c_time = points[3]['time'].strftime('%Y-%m-%d %H:%M')
        d_time = points[4]['time'].strftime('%Y-%m-%d %H:%M')
        
        # قیمت نقاط مهم
        x_price = points[0]['price']
        a_price = points[1]['price']
        b_price = points[2]['price']
        c_price = points[3]['price']
        d_price = points[4]['price']
        
        # توضیحات هر نوع الگو
        pattern_descriptions = {
            'Gartley': "الگوی گارتلی یک الگوی هارمونیک محافظه‌کارانه با اطمینان بالاست که اغلب در بازگشت‌های میانی روند دیده می‌شود.",
            'Butterfly': "الگوی پروانه یک الگوی هارمونیک قوی است که معمولاً بازگشت‌های بزرگتر و عمیق‌تر از گارتلی را نشان می‌دهد.",
            'Bat': "الگوی بت یک الگوی هارمونیک با ریسک پایین است که برای شناسایی نقاط بازگشت احتمالی در روند مفید است.",
            'Crab': "الگوی خرچنگ قوی‌ترین الگوی هارمونیک است که اغلب بازگشت‌های شدیدی را پیش‌بینی می‌کند و بهترین نسبت ریسک به پاداش را دارد.",
            'Shark': "الگوی کوسه یک الگوی نوین در خانواده الگوهای هارمونیک است که حرکت‌های سریع قیمت را پیش‌بینی می‌کند.",
            'Cypher': "الگوی سایفر یک الگوی تهاجمی‌تر است که می‌تواند فرصت‌های معاملاتی عالی با ریسک پایین را فراهم کند."
        }
        
        # توضیحات جهت‌ها
        direction_descriptions = {
            'صعودی': "این الگو یک سیگنال صعودی نشان می‌دهد که احتمال افزایش قیمت پس از تکمیل الگو وجود دارد.",
            'نزولی': "این الگو یک سیگنال نزولی نشان می‌دهد که احتمال کاهش قیمت پس از تکمیل الگو وجود دارد."
        }
        
        # ساخت توضیح کامل
        description = f"""
الگوی هارمونیک {pattern_type} ({direction}) با {pattern['confidence']}% اطمینان شناسایی شد.

{pattern_descriptions.get(pattern_type, '')}
{direction_descriptions.get(direction, '')}

نقاط مهم الگو:
• X: قیمت {x_price:.2f} در {x_time}
• A: قیمت {a_price:.2f} در {a_time}
• B: قیمت {b_price:.2f} در {b_time}
• C: قیمت {c_price:.2f} در {c_time}
• D: قیمت {d_price:.2f} در {d_time} (نقطه ورود)

پیشنهادات معاملاتی:
• قیمت ورود: {pattern['entry_price']:.2f}
• حد ضرر: {pattern['stop_loss']:.2f} ({((pattern['stop_loss'] / pattern['entry_price']) - 1) * 100:.2f}%)
• هدف اول: {pattern['target_1']:.2f} ({((pattern['target_1'] / pattern['entry_price']) - 1) * 100:.2f}%)
• هدف دوم: {pattern['target_2']:.2f} ({((pattern['target_2'] / pattern['entry_price']) - 1) * 100:.2f}%)
• هدف سوم: {pattern['target_3']:.2f} ({((pattern['target_3'] / pattern['entry_price']) - 1) * 100:.2f}%)
• نسبت ریسک به پاداش: {pattern['risk_reward']:.2f}

نکات مهم:
• این الگو بهترین عملکرد را در تایم‌فریم‌های بالاتر نشان می‌دهد.
• برای تأیید بیشتر، از اندیکاتورهای دیگر مانند RSI، MACD، و الگوهای کندل استفاده کنید.
• در قیمت ورود (D)، توجه به شکسته شدن حمایت‌ها یا مقاومت‌ها می‌تواند به افزایش دقت الگو کمک کند.
"""
        return description


def find_harmonic_patterns(df: pd.DataFrame, max_patterns: int = 3) -> List[Dict[str, Any]]:
    """
    یافتن الگوهای هارمونیک در دیتافریم قیمت
    
    Args:
        df: دیتافریم قیمت
        max_patterns: حداکثر تعداد الگوهای بازگشتی
        
    Returns:
        list: لیست الگوهای شناسایی شده
    """
    detector = HarmonicPatterns(max_patterns=max_patterns)
    return detector.detect_harmonic_patterns(df)


def plot_harmonic_pattern(df: pd.DataFrame, pattern: Dict[str, Any]) -> Optional[io.BytesIO]:
    """
    رسم الگوی هارمونیک
    
    Args:
        df: دیتافریم قیمت
        pattern: دیکشنری الگوی هارمونیک
        
    Returns:
        io.BytesIO: بافر تصویر
    """
    detector = HarmonicPatterns()
    return detector.plot_harmonic_pattern(df, pattern)


def get_harmonic_pattern_description(pattern: Dict[str, Any]) -> str:
    """
    دریافت توضیح متنی الگوی هارمونیک
    
    Args:
        pattern: دیکشنری الگوی هارمونیک
        
    Returns:
        str: توضیح الگو
    """
    detector = HarmonicPatterns()
    return detector.generate_pattern_description(pattern)


def detect_harmonic_patterns(df: pd.DataFrame, max_patterns: int = 3) -> List[Dict[str, Any]]:
    """
    یافتن الگوهای هارمونیک در دیتافریم قیمت (تابع کمکی)
    
    Args:
        df: دیتافریم قیمت
        max_patterns: حداکثر تعداد الگوهای بازگشتی
        
    Returns:
        list: لیست الگوهای شناسایی شده
    """
    detector = HarmonicPatterns(max_patterns=max_patterns)
    return detector.detect_harmonic_patterns(df)