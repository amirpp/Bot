"""
ماژول تشخیص و تحلیل الگوهای فیبوناچی

این ماژول شامل توابع مورد نیاز برای تشخیص، تحلیل و پیش‌بینی بر اساس الگوهای فیبوناچی است.
الگوهای فیبوناچی شامل اصلاح‌های فیبوناچی، بسط‌های فیبوناچی، هارمونیک‌های فیبوناچی و ... می‌شود.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import logging
import math
from datetime import datetime, timedelta

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# سطوح فیبوناچی کلاسیک
FIBONACCI_LEVELS = {
    '0.0': 0.0,
    '23.6%': 0.236,
    '38.2%': 0.382, 
    '50.0%': 0.5,
    '61.8%': 0.618,
    '78.6%': 0.786,
    '100.0%': 1.0,
    '127.2%': 1.272,
    '161.8%': 1.618,
    '261.8%': 2.618,
    '423.6%': 4.236
}

# الگوهای هارمونیک فیبوناچی
HARMONIC_PATTERNS = {
    'گارتلی (Gartley)': {
        'XA': 1.0,
        'AB': 0.618,
        'BC': 0.382,
        'CD': 1.272,
        'type': 'گاوی/خرسی'
    },
    'پروانه (Butterfly)': {
        'XA': 1.0,
        'AB': 0.786,
        'BC': 0.382,
        'CD': 1.618,
        'type': 'گاوی/خرسی'
    },
    'خفاش (Bat)': {
        'XA': 1.0,
        'AB': 0.382,
        'BC': 0.382,
        'CD': 1.618,
        'type': 'گاوی/خرسی'
    },
    'عقاب (Eagle)': {
        'XA': 1.0,
        'AB': 0.5,
        'BC': 0.618,
        'CD': 1.23,
        'type': 'گاوی/خرسی'
    },
    'کرابی (Crab)': {
        'XA': 1.0,
        'AB': 0.382,
        'BC': 0.618,
        'CD': 3.618,
        'type': 'گاوی/خرسی'
    },
    'سرطان (Deep Crab)': {
        'XA': 1.0,
        'AB': 0.886,
        'BC': 0.382,
        'CD': 2.618,
        'type': 'گاوی/خرسی'
    },
    'کبوتر (Cypher)': {
        'XA': 1.0,
        'AB': 0.382,
        'BC': 1.272,
        'CD': 0.786,
        'type': 'گاوی/خرسی'
    },
    'لاکپشت (Shark)': {
        'XA': 1.0,
        'AB': 0.5,
        'BC': 1.618,
        'CD': 0.886,
        'type': 'گاوی/خرسی'
    },
}

class FibonacciAnalyzer:
    """کلاس تحلیل‌گر الگوهای فیبوناچی"""
    
    def __init__(self, tolerance: float = 0.05):
        """
        مقداردهی اولیه تحلیل‌گر فیبوناچی
        
        Args:
            tolerance (float): میزان انعطاف‌پذیری در تشخیص الگوها (به عنوان درصد)
        """
        self.tolerance = tolerance
        logger.info(f"تحلیل‌گر فیبوناچی با میزان انعطاف‌پذیری {tolerance * 100}% راه‌اندازی شد")
    
    def find_swing_points(self, df: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
        """
        یافتن نقاط اوج و فرود در نمودار قیمت
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            window (int): اندازه پنجره برای تشخیص نقاط اوج و فرود
            
        Returns:
            dict: دیکشنری نقاط اوج و فرود
        """
        if df.empty or len(df) < window * 2:
            return {'highs': [], 'lows': [], 'swing_points': []}
        
        # یافتن نقاط اوج
        highs = []
        for i in range(window, len(df) - window):
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
                highs.append((df.index[i], df['high'].iloc[i], 'high'))
        
        # یافتن نقاط فرود
        lows = []
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
                lows.append((df.index[i], df['low'].iloc[i], 'low'))
        
        # ترکیب و مرتب‌سازی نقاط بر اساس زمان
        swing_points = sorted(highs + lows, key=lambda x: x[0])
        
        return {
            'highs': highs,
            'lows': lows,
            'swing_points': swing_points
        }
    
    def calculate_retracement_levels(self, start_price: float, end_price: float) -> Dict[str, float]:
        """
        محاسبه سطوح اصلاحی فیبوناچی
        
        Args:
            start_price (float): قیمت شروع
            end_price (float): قیمت پایان
            
        Returns:
            dict: سطوح اصلاحی فیبوناچی
        """
        diff = end_price - start_price
        levels = {}
        
        for label, ratio in FIBONACCI_LEVELS.items():
            level = end_price - (diff * ratio)
            levels[label] = level
        
        return levels
    
    def calculate_extension_levels(self, start_price: float, end_price: float) -> Dict[str, float]:
        """
        محاسبه سطوح بسط فیبوناچی
        
        Args:
            start_price (float): قیمت شروع
            end_price (float): قیمت پایان
            
        Returns:
            dict: سطوح بسط فیبوناچی
        """
        diff = end_price - start_price
        levels = {}
        
        for label, ratio in FIBONACCI_LEVELS.items():
            level = end_price + (diff * ratio)
            levels[label] = level
        
        return levels
    
    def detect_fibonacci_patterns(self, df: pd.DataFrame, min_swings: int = 3) -> Dict[str, Any]:
        """
        تشخیص الگوهای فیبوناچی در دیتافریم قیمت
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            min_swings (int): حداقل تعداد نقاط چرخش لازم
            
        Returns:
            dict: الگوهای فیبوناچی شناسایی شده
        """
        if df.empty or len(df) < 20:
            return {'patterns': [], 'retracements': [], 'extensions': []}
        
        # یافتن نقاط چرخش
        swing_result = self.find_swing_points(df)
        swing_points = swing_result['swing_points']
        
        if len(swing_points) < min_swings:
            return {'patterns': [], 'retracements': [], 'extensions': []}
        
        # بررسی آخرین حرکت‌ها برای اصلاح‌ها و بسط‌ها
        retracements = []
        extensions = []
        patterns = []
        
        # بررسی ۵ نقطه آخر برای الگوهای فیبوناچی
        for i in range(min(len(swing_points) - 3, 10)):
            # بررسی اصلاح‌ها (از اوج به فرود یا از فرود به اوج)
            p1 = swing_points[-(i+3)]
            p2 = swing_points[-(i+2)]
            p3 = swing_points[-(i+1)]
            
            if p1[2] != p2[2]:  # اوج به فرود یا فرود به اوج
                start_price, end_price = p1[1], p2[1]
                
                # محاسبه میزان اصلاح
                full_move = abs(end_price - start_price)
                retracement = abs(p3[1] - end_price) / full_move if full_move != 0 else 0
                
                # تطبیق با سطوح فیبوناچی
                matched_level = None
                for level_name, level_ratio in FIBONACCI_LEVELS.items():
                    if abs(retracement - level_ratio) < self.tolerance:
                        matched_level = level_name
                        break
                
                if matched_level:
                    retracements.append({
                        'start_point': (p1[0], p1[1], p1[2]),
                        'end_point': (p2[0], p2[1], p2[2]),
                        'retracement_point': (p3[0], p3[1], p3[2]),
                        'retracement_ratio': retracement,
                        'fibonacci_level': matched_level,
                        'direction': 'upward' if p1[1] < p2[1] else 'downward'
                    })
            
            # بررسی بسط‌ها (ادامه حرکت در همان جهت پس از اصلاح)
            if len(swing_points) > i+4:
                p4 = swing_points[-(i+0)]
                
                if p2[2] != p3[2] and p3[2] == p1[2]:  # حرکت، اصلاح، ادامه در جهت اصلی
                    # محاسبه نسبت بسط
                    initial_move = abs(p2[1] - p1[1])
                    extension_move = abs(p4[1] - p3[1])
                    extension_ratio = extension_move / initial_move if initial_move != 0 else 0
                    
                    # تطبیق با سطوح فیبوناچی
                    matched_level = None
                    for level_name, level_ratio in FIBONACCI_LEVELS.items():
                        if abs(extension_ratio - level_ratio) < self.tolerance:
                            matched_level = level_name
                            break
                    
                    if matched_level:
                        extensions.append({
                            'start_point': (p1[0], p1[1], p1[2]),
                            'end_point': (p2[0], p2[1], p2[2]),
                            'retracement_point': (p3[0], p3[1], p3[2]),
                            'extension_point': (p4[0], p4[1], p4[2]),
                            'extension_ratio': extension_ratio,
                            'fibonacci_level': matched_level,
                            'direction': 'upward' if p1[1] < p2[1] else 'downward'
                        })
        
        # بررسی الگوهای هارمونیک
        if len(swing_points) >= 5:
            for i in range(len(swing_points) - 4):
                # انتخاب 5 نقطه متوالی (نقاط X, A, B, C, D)
                x_point = swing_points[i]
                a_point = swing_points[i+1]
                b_point = swing_points[i+2]
                c_point = swing_points[i+3]
                d_point = swing_points[i+4]
                
                # بررسی برای هر الگوی هارمونیک شناخته شده
                for pattern_name, pattern_ratios in HARMONIC_PATTERNS.items():
                    # محاسبه نسبت‌های واقعی
                    x_to_a = abs(a_point[1] - x_point[1])
                    a_to_b = abs(b_point[1] - a_point[1])
                    b_to_c = abs(c_point[1] - b_point[1])
                    c_to_d = abs(d_point[1] - c_point[1])
                    
                    # بررسی تطابق نسبت‌ها با الگو
                    ab_ratio = a_to_b / x_to_a if x_to_a != 0 else 0
                    bc_ratio = b_to_c / a_to_b if a_to_b != 0 else 0
                    cd_ratio = c_to_d / b_to_c if b_to_c != 0 else 0
                    
                    # حساب کردن میزان انحراف از نسبت‌های کلیدی الگو
                    ab_diff = abs(ab_ratio - pattern_ratios['AB'])
                    bc_diff = abs(bc_ratio - pattern_ratios['BC'])
                    cd_diff = abs(cd_ratio - pattern_ratios['CD'])
                    
                    # تشخیص الگو با میزان انعطاف معین
                    if ab_diff <= self.tolerance and bc_diff <= self.tolerance and cd_diff <= self.tolerance:
                        # تشخیص جهت الگو (گاوی یا خرسی)
                        direction = 'صعودی (گاوی)' if d_point[1] > a_point[1] else 'نزولی (خرسی)'
                        
                        # محاسبه میزان تکمیل الگو
                        completion_percentage = 100.0
                        
                        patterns.append({
                            'pattern_name': pattern_name,
                            'direction': direction,
                            'x_point': (x_point[0], x_point[1]),
                            'a_point': (a_point[0], a_point[1]),
                            'b_point': (b_point[0], b_point[1]),
                            'c_point': (c_point[0], c_point[1]),
                            'd_point': (d_point[0], d_point[1]),
                            'completion': completion_percentage,
                            'ratios': {
                                'AB': ab_ratio,
                                'BC': bc_ratio,
                                'CD': cd_ratio
                            },
                            'potential_target': None  # محاسبه در بخش بعدی
                        })
        
        # محاسبه پتانسیل قیمتی برای هر الگو
        for pattern in patterns:
            # محاسبه هدف قیمتی بر اساس الگو
            x_price = pattern['x_point'][1]
            a_price = pattern['a_point'][1]
            d_price = pattern['d_point'][1]
            
            # خط فرضی XA برای محاسبه پتانسیل
            xa_range = abs(a_price - x_price)
            
            # محاسبه هدف قیمتی (نسبت 61.8% برای برگشت از نقطه D)
            if pattern['direction'].startswith('صعودی'):
                target_price = d_price + (xa_range * 0.618)
            else:
                target_price = d_price - (xa_range * 0.618)
            
            pattern['potential_target'] = target_price
        
        return {
            'patterns': patterns,
            'retracements': retracements,
            'extensions': extensions
        }
    
    def detect_fibonacci_zones(self, df: pd.DataFrame, lookback_periods: int = 100) -> Dict[str, Any]:
        """
        تشخیص مناطق فیبوناچی مهم در نمودار
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            lookback_periods (int): تعداد دوره‌های گذشته برای بررسی
            
        Returns:
            dict: مناطق فیبوناچی
        """
        if df.empty or len(df) < lookback_periods:
            return {'zones': [], 'current_zone': None}
        
        # محدود کردن داده‌ها به دوره مورد بررسی
        df_limited = df.iloc[-lookback_periods:].copy()
        
        # یافتن اوج و فرود کلی در دوره مورد بررسی
        highest_idx = df_limited['high'].idxmax()
        lowest_idx = df_limited['low'].idxmin()
        
        highest_price = df_limited.loc[highest_idx, 'high']
        lowest_price = df_limited.loc[lowest_idx, 'low']
        
        # بررسی اینکه کدام یک زودتر اتفاق افتاده است
        if df_limited.index.get_loc(highest_idx) > df_limited.index.get_loc(lowest_idx):
            # حرکت از پایین به بالا
            start_price, end_price = lowest_price, highest_price
            direction = 'صعودی'
        else:
            # حرکت از بالا به پایین
            start_price, end_price = highest_price, lowest_price
            direction = 'نزولی'
        
        # محاسبه سطوح فیبوناچی بین اوج و فرود
        price_range = abs(end_price - start_price)
        levels = {}
        
        for label, ratio in FIBONACCI_LEVELS.items():
            if direction == 'صعودی':
                level_price = start_price + (price_range * ratio)
            else:
                level_price = start_price - (price_range * ratio)
            
            levels[label] = level_price
        
        # بررسی موقعیت فعلی قیمت نسبت به سطوح فیبوناچی
        current_price = df['close'].iloc[-1]
        current_zone = None
        
        # تبدیل levels به آرایه مرتب شده برای یافتن نزدیک‌ترین سطوح
        level_prices = [(label, price) for label, price in levels.items()]
        if direction == 'صعودی':
            level_prices.sort(key=lambda x: x[1])
        else:
            level_prices.sort(key=lambda x: x[1], reverse=True)
        
        # یافتن ناحیه فعلی قیمت
        for i in range(len(level_prices) - 1):
            level1, price1 = level_prices[i]
            level2, price2 = level_prices[i+1]
            
            if min(price1, price2) <= current_price <= max(price1, price2):
                current_zone = {
                    'lower_level': level1,
                    'upper_level': level2,
                    'lower_price': price1,
                    'upper_price': price2,
                    'direction': direction
                }
                break
        
        # ساخت مناطق فیبوناچی با توضیحات
        zones = []
        for i in range(len(level_prices) - 1):
            level1, price1 = level_prices[i]
            level2, price2 = level_prices[i+1]
            
            zones.append({
                'lower_level': level1,
                'upper_level': level2,
                'lower_price': price1,
                'upper_price': price2,
                'is_current': current_zone and current_zone['lower_level'] == level1,
                'direction': direction
            })
        
        return {
            'zones': zones,
            'current_zone': current_zone,
            'levels': levels,
            'highest_price': highest_price,
            'lowest_price': lowest_price,
            'direction': direction
        }
    
    def calculate_entry_exit_points(self, df: pd.DataFrame, fibonacci_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        محاسبه نقاط ورود و خروج بر اساس تحلیل فیبوناچی
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            fibonacci_results (dict): نتایج تحلیل فیبوناچی
            
        Returns:
            dict: نقاط ورود و خروج
        """
        current_price = df['close'].iloc[-1]
        
        entry_points = []
        exit_points = []
        stop_loss_points = []
        
        patterns = fibonacci_results.get('patterns', [])
        zones = fibonacci_results.get('zones', [])
        current_zone = fibonacci_results.get('current_zone')
        
        # استفاده از الگوهای تشخیص داده شده برای محاسبه نقاط ورود و خروج
        for pattern in patterns:
            pattern_name = pattern['pattern_name']
            direction = pattern['direction']
            a_price = pattern['a_point'][1]
            d_price = pattern['d_point'][1]
            target_price = pattern['potential_target']
            
            if target_price is not None:
                # محاسبه حد ضرر (فاصله از نقطه D بر اساس جهت الگو)
                if direction.startswith('صعودی'):
                    stop_loss = d_price - (abs(d_price - a_price) * 0.382)
                    risk_amount = d_price - stop_loss
                    reward_amount = target_price - d_price
                else:
                    stop_loss = d_price + (abs(d_price - a_price) * 0.382)
                    risk_amount = stop_loss - d_price
                    reward_amount = d_price - target_price
                
                # محاسبه نسبت ریسک به بازده
                risk_reward_ratio = reward_amount / risk_amount if risk_amount != 0 else 0
                
                # افزودن به نقاط ورود/خروج با توضیحات
                entry_type = 'خرید' if direction.startswith('صعودی') else 'فروش'
                
                entry_points.append({
                    'price': d_price,
                    'type': entry_type,
                    'source': f'الگوی {pattern_name}',
                    'confidence': 'بالا' if risk_reward_ratio >= 2.0 else 'متوسط',
                    'description': f'ورود در نقطه D الگوی {pattern_name} ({direction})'
                })
                
                exit_points.append({
                    'price': target_price,
                    'type': 'هدف قیمتی',
                    'source': f'الگوی {pattern_name}',
                    'confidence': 'متوسط',
                    'description': f'هدف قیمتی الگوی {pattern_name} (بازگشت 61.8%)'
                })
                
                stop_loss_points.append({
                    'price': stop_loss,
                    'risk_reward_ratio': risk_reward_ratio,
                    'source': f'الگوی {pattern_name}',
                    'description': f'حد ضرر الگوی {pattern_name} (38.2% فراتر از نقطه D)'
                })
        
        # استفاده از مناطق فیبوناچی برای محاسبه نقاط ورود و خروج
        if current_zone:
            direction = current_zone['direction']
            lower_level = current_zone['lower_level']
            upper_level = current_zone['upper_level']
            lower_price = current_zone['lower_price']
            upper_price = current_zone['upper_price']
            
            # ورود بر اساس منطقه فعلی قیمت
            if direction == 'صعودی':
                if lower_level in ['0.0', '23.6%', '38.2%']:
                    # در منطقه پایین - احتمال بازگشت به بالا
                    entry_points.append({
                        'price': current_price,
                        'type': 'خرید',
                        'source': f'منطقه فیبوناچی {lower_level}-{upper_level}',
                        'confidence': 'متوسط',
                        'description': f'خرید در ناحیه حمایتی فیبوناچی ({lower_level} تا {upper_level})'
                    })
                    
                    exit_points.append({
                        'price': levels.get('61.8%', upper_price * 1.2),
                        'type': 'هدف قیمتی',
                        'source': 'سطح فیبوناچی 61.8%',
                        'confidence': 'متوسط',
                        'description': 'هدف قیمتی در سطح فیبوناچی 61.8%'
                    })
                    
                    stop_loss_points.append({
                        'price': lower_price * 0.95,
                        'risk_reward_ratio': ((levels.get('61.8%', upper_price * 1.2) - current_price) / 
                                            (current_price - lower_price * 0.95)),
                        'source': 'منطقه فیبوناچی',
                        'description': f'حد ضرر 5% زیر سطح فیبوناچی {lower_level}'
                    })
                elif lower_level in ['61.8%', '78.6%']:
                    # در منطقه بالا - احتمال اصلاح به پایین
                    entry_points.append({
                        'price': current_price,
                        'type': 'فروش',
                        'source': f'منطقه فیبوناچی {lower_level}-{upper_level}',
                        'confidence': 'متوسط',
                        'description': f'فروش در ناحیه مقاومتی فیبوناچی ({lower_level} تا {upper_level})'
                    })
                    
                    exit_points.append({
                        'price': levels.get('38.2%', lower_price * 0.8),
                        'type': 'هدف قیمتی',
                        'source': 'سطح فیبوناچی 38.2%',
                        'confidence': 'متوسط',
                        'description': 'هدف قیمتی در سطح فیبوناچی 38.2%'
                    })
                    
                    stop_loss_points.append({
                        'price': upper_price * 1.05,
                        'risk_reward_ratio': ((current_price - levels.get('38.2%', lower_price * 0.8)) / 
                                            (upper_price * 1.05 - current_price)),
                        'source': 'منطقه فیبوناچی',
                        'description': f'حد ضرر 5% بالای سطح فیبوناچی {upper_level}'
                    })
            else:  # 'نزولی'
                if lower_level in ['61.8%', '78.6%']:
                    # در منطقه پایین - احتمال بازگشت به بالا
                    entry_points.append({
                        'price': current_price,
                        'type': 'خرید',
                        'source': f'منطقه فیبوناچی {lower_level}-{upper_level}',
                        'confidence': 'متوسط',
                        'description': f'خرید در ناحیه حمایتی فیبوناچی ({lower_level} تا {upper_level})'
                    })
                    
                    exit_points.append({
                        'price': levels.get('38.2%', upper_price * 1.2),
                        'type': 'هدف قیمتی',
                        'source': 'سطح فیبوناچی 38.2%',
                        'confidence': 'متوسط',
                        'description': 'هدف قیمتی در سطح فیبوناچی 38.2%'
                    })
                    
                    stop_loss_points.append({
                        'price': lower_price * 0.95,
                        'risk_reward_ratio': ((levels.get('38.2%', upper_price * 1.2) - current_price) / 
                                            (current_price - lower_price * 0.95)),
                        'source': 'منطقه فیبوناچی',
                        'description': f'حد ضرر 5% زیر سطح فیبوناچی {lower_level}'
                    })
                elif lower_level in ['0.0', '23.6%']:
                    # در منطقه بالا - احتمال اصلاح به پایین
                    entry_points.append({
                        'price': current_price,
                        'type': 'فروش',
                        'source': f'منطقه فیبوناچی {lower_level}-{upper_level}',
                        'confidence': 'متوسط',
                        'description': f'فروش در ناحیه مقاومتی فیبوناچی ({lower_level} تا {upper_level})'
                    })
                    
                    exit_points.append({
                        'price': levels.get('61.8%', lower_price * 0.8),
                        'type': 'هدف قیمتی',
                        'source': 'سطح فیبوناچی 61.8%',
                        'confidence': 'متوسط',
                        'description': 'هدف قیمتی در سطح فیبوناچی 61.8%'
                    })
                    
                    stop_loss_points.append({
                        'price': upper_price * 1.05,
                        'risk_reward_ratio': ((current_price - levels.get('61.8%', lower_price * 0.8)) / 
                                            (upper_price * 1.05 - current_price)),
                        'source': 'منطقه فیبوناچی',
                        'description': f'حد ضرر 5% بالای سطح فیبوناچی {upper_level}'
                    })
        
        return {
            'entry_points': entry_points,
            'exit_points': exit_points,
            'stop_loss_points': stop_loss_points,
            'best_risk_reward': max([p['risk_reward_ratio'] for p in stop_loss_points], default=0) if stop_loss_points else 0
        }
    
    def plot_fibonacci_patterns(self, df: pd.DataFrame, fibonacci_results: Dict[str, Any]) -> go.Figure:
        """
        رسم نمودار با الگوهای فیبوناچی
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            fibonacci_results (dict): نتایج تحلیل فیبوناچی
            
        Returns:
            go.Figure: نمودار پلاتلی
        """
        # ایجاد نمودار شمعی
        fig = go.Figure()
        
        # اضافه کردن نمودار شمعی
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="نمودار قیمت",
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ))
        
        # اضافه کردن سطوح فیبوناچی اگر در نتایج وجود دارد
        if 'levels' in fibonacci_results:
            levels = fibonacci_results['levels']
            
            for level_name, level_price in levels.items():
                fig.add_shape(
                    type="line",
                    x0=df.index[0],
                    x1=df.index[-1],
                    y0=level_price,
                    y1=level_price,
                    line=dict(
                        color="rgba(102, 102, 255, 0.5)",
                        width=1,
                        dash="dash",
                    ),
                )
                
                # اضافه کردن برچسب روی خط
                fig.add_annotation(
                    x=df.index[-1],
                    y=level_price,
                    text=f"Fib {level_name}",
                    showarrow=False,
                    xshift=10,
                    font=dict(size=10, color="rgba(102, 102, 255, 0.8)")
                )
        
        # اضافه کردن الگوهای هارمونیک
        patterns = fibonacci_results.get('patterns', [])
        
        for i, pattern in enumerate(patterns):
            # نقاط الگو
            x_date, x_price = pattern['x_point']
            a_date, a_price = pattern['a_point']
            b_date, b_price = pattern['b_point']
            c_date, c_price = pattern['c_point']
            d_date, d_price = pattern['d_point']
            
            pattern_points = [
                (x_date, x_price),
                (a_date, a_price),
                (b_date, b_price),
                (c_date, c_price),
                (d_date, d_price)
            ]
            
            # اضافه کردن خطوط بین نقاط
            dates = [x_date, a_date, b_date, c_date, d_date]
            prices = [x_price, a_price, b_price, c_price, d_price]
            
            # با رنگ متفاوت برای هر الگو
            color = f"rgba({(40 + i*70) % 255}, {(100 + i*50) % 255}, {(150 + i*40) % 255}, 0.8)"
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines+markers',
                name=f"{pattern['pattern_name']} ({pattern['direction']})",
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))
            
            # اضافه کردن برچسب‌های نقاط
            labels = ['X', 'A', 'B', 'C', 'D']
            for i, (date, price, label) in enumerate(zip(dates, prices, labels)):
                fig.add_annotation(
                    x=date,
                    y=price,
                    text=label,
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    font=dict(size=12, color=color)
                )
        
        # تنظیمات نمودار
        fig.update_layout(
            title="الگوهای فیبوناچی شناسایی شده",
            xaxis_title="تاریخ",
            yaxis_title="قیمت",
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def get_fibonacci_summary(self, fibonacci_results: Dict[str, Any]) -> str:
        """
        تولید خلاصه متنی از نتایج تحلیل فیبوناچی
        
        Args:
            fibonacci_results (dict): نتایج تحلیل فیبوناچی
            
        Returns:
            str: خلاصه متنی
        """
        patterns = fibonacci_results.get('patterns', [])
        zones = fibonacci_results.get('zones', [])
        current_zone = fibonacci_results.get('current_zone')
        entry_points = fibonacci_results.get('entry_points', [])
        
        summary = []
        
        # خلاصه الگوهای شناسایی شده
        if patterns:
            summary.append(f"**الگوهای هارمونیک شناسایی شده:** {len(patterns)} الگو")
            
            for i, pattern in enumerate(patterns[:3], 1):
                summary.append(f"{i}. **{pattern['pattern_name']}** ({pattern['direction']}): "
                              f"نقطه ورود احتمالی در قیمت {pattern['d_point'][1]:.2f} و "
                              f"هدف قیمتی {pattern['potential_target']:.2f}")
        else:
            summary.append("هیچ الگوی هارمونیک فیبوناچی در محدوده فعلی شناسایی نشد.")
        
        # خلاصه موقعیت فعلی در سطوح فیبوناچی
        if current_zone:
            direction = current_zone['direction']
            lower_level = current_zone['lower_level']
            upper_level = current_zone['upper_level']
            
            summary.append(f"\n**موقعیت فعلی قیمت:** بین سطوح فیبوناچی {lower_level} و {upper_level} "
                          f"در روند کلی {direction}")
            
            if direction == 'صعودی':
                if lower_level in ['0.0', '23.6%', '38.2%']:
                    summary.append("قیمت در ناحیه حمایتی قرار دارد و پتانسیل ادامه روند صعودی وجود دارد.")
                elif lower_level in ['61.8%', '78.6%']:
                    summary.append("قیمت در ناحیه مقاومتی قرار دارد و احتمال اصلاح وجود دارد.")
            else:
                if lower_level in ['61.8%', '78.6%']:
                    summary.append("قیمت در ناحیه حمایتی قرار دارد و احتمال بازگشت موقت وجود دارد.")
                elif lower_level in ['0.0', '23.6%']:
                    summary.append("قیمت در ناحیه مقاومتی قرار دارد و پتانسیل ادامه روند نزولی وجود دارد.")
        
        # خلاصه نقاط ورود پیشنهادی
        if entry_points:
            summary.append("\n**نقاط ورود پیشنهادی:**")
            
            for i, entry in enumerate(entry_points[:3], 1):
                summary.append(f"{i}. {entry['type']} در قیمت {entry['price']:.2f}: {entry['description']} "
                              f"(اطمینان: {entry['confidence']})")
        
        return "\n".join(summary)


def analyze_fibonacci(df: pd.DataFrame, tolerance: float = 0.05) -> Dict[str, Any]:
    """
    انجام تحلیل کامل فیبوناچی روی دیتافریم قیمت
    
    Args:
        df (pd.DataFrame): دیتافریم قیمت
        tolerance (float): میزان انعطاف‌پذیری در تشخیص الگوها
        
    Returns:
        dict: نتایج تحلیل فیبوناچی
    """
    if df.empty or len(df) < 20:
        return {'error': 'داده‌های ناکافی برای تحلیل فیبوناچی'}
    
    try:
        analyzer = FibonacciAnalyzer(tolerance=tolerance)
        
        # تشخیص الگوهای فیبوناچی
        pattern_results = analyzer.detect_fibonacci_patterns(df)
        
        # تشخیص مناطق فیبوناچی
        zone_results = analyzer.detect_fibonacci_zones(df)
        
        # ترکیب نتایج
        combined_results = {
            **pattern_results,
            **zone_results
        }
        
        # محاسبه نقاط ورود و خروج
        entry_exit_results = analyzer.calculate_entry_exit_points(df, combined_results)
        combined_results.update(entry_exit_results)
        
        # افزودن خلاصه متنی
        combined_results['summary'] = analyzer.get_fibonacci_summary(combined_results)
        
        return combined_results
        
    except Exception as e:
        logger.error(f"خطا در تحلیل فیبوناچی: {str(e)}")
        return {'error': f'خطا در تحلیل فیبوناچی: {str(e)}'}


def plot_fibonacci_levels(df: pd.DataFrame, fibonacci_results: Dict[str, Any]) -> go.Figure:
    """
    رسم نمودار سطوح فیبوناچی
    
    Args:
        df (pd.DataFrame): دیتافریم قیمت
        fibonacci_results (dict): نتایج تحلیل فیبوناچی
        
    Returns:
        go.Figure: نمودار پلاتلی
    """
    if df.empty or 'error' in fibonacci_results:
        # ایجاد نمودار خالی با پیام خطا
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="داده‌های کافی برای نمایش نمودار فیبوناچی وجود ندارد",
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
    
    # ایجاد تحلیل‌گر
    analyzer = FibonacciAnalyzer()
    
    # رسم نمودار
    return analyzer.plot_fibonacci_patterns(df, fibonacci_results)


def find_key_fibonacci_levels(df: pd.DataFrame) -> Dict[str, Any]:
    """
    یافتن سطوح کلیدی فیبوناچی برای استفاده سریع
    
    Args:
        df (pd.DataFrame): دیتافریم قیمت
        
    Returns:
        dict: سطوح کلیدی فیبوناچی
    """
    if df.empty or len(df) < 20:
        return {'levels': [], 'current_price': 0}
    
    try:
        # تحلیل‌گر فیبوناچی
        analyzer = FibonacciAnalyzer()
        
        # یافتن مناطق فیبوناچی
        zone_results = analyzer.detect_fibonacci_zones(df)
        
        # قیمت فعلی
        current_price = df['close'].iloc[-1]
        
        # ساخت لیست سطوح کلیدی
        key_levels = []
        
        if 'levels' in zone_results:
            levels = zone_results['levels']
            
            for label, price in levels.items():
                # محاسبه فاصله نسبی از قیمت فعلی
                relative_distance = abs((price - current_price) / current_price)
                
                key_levels.append({
                    'level': label,
                    'price': price,
                    'distance': relative_distance,
                    'type': 'حمایت' if price < current_price else 'مقاومت'
                })
            
            # مرتب‌سازی بر اساس فاصله از قیمت فعلی
            key_levels.sort(key=lambda x: x['distance'])
        
        return {
            'levels': key_levels,
            'current_price': current_price,
            'direction': zone_results.get('direction', 'نامشخص')
        }
        
    except Exception as e:
        logger.error(f"خطا در یافتن سطوح کلیدی فیبوناچی: {str(e)}")
        return {'levels': [], 'current_price': df['close'].iloc[-1] if not df.empty else 0}