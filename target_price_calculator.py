"""
ماژول محاسبه قیمت‌های هدف، حد ضرر و نقاط ورود/خروج

این ماژول امکان محاسبه قیمت‌های هدف و مدیریت ریسک را فراهم می‌کند
و قابلیت تعیین نقاط ورود/خروج بر اساس روش‌های مختلف را دارد.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# تنظیم لاگر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TargetPriceCalculator:
    """کلاس محاسبه قیمت‌های هدف و سطوح معاملاتی"""
    
    def __init__(self, df: pd.DataFrame, risk_percent: float = 1.0, reward_ratio: float = 3.0):
        """
        مقداردهی اولیه محاسبه‌گر قیمت هدف
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های OHLCV
            risk_percent (float): درصد ریسک (٪)
            reward_ratio (float): نسبت ریوارد به ریسک
        """
        self.df = df.copy() if df is not None else None
        self.risk_percent = risk_percent
        self.reward_ratio = reward_ratio
        
        logger.info(f"محاسبه‌گر قیمت هدف با ریسک {risk_percent}٪ و نسبت ریوارد {reward_ratio} ایجاد شد")
    
    def calculate_support_resistance(self, levels: int = 3, method: str = 'zigzag') -> List[Dict[str, float]]:
        """
        محاسبه سطوح حمایت و مقاومت
        
        Args:
            levels (int): تعداد سطوح مورد نیاز
            method (str): روش محاسبه ('zigzag', 'peaks', 'fractals', 'pivots')
            
        Returns:
            list: لیست دیکشنری‌های سطوح مقاومت و حمایت
        """
        if self.df is None or self.df.empty:
            logger.error("دیتافریم خالی برای محاسبه سطوح حمایت و مقاومت")
            return []
        
        # انتخاب روش محاسبه
        if method == 'zigzag':
            return self._calculate_zigzag_levels(levels)
        elif method == 'peaks':
            return self._calculate_peaks_levels(levels)
        elif method == 'fractals':
            return self._calculate_fractal_levels(levels)
        elif method == 'pivots':
            return self._calculate_pivot_levels()
        else:
            logger.warning(f"روش {method} نامعتبر است. استفاده از روش پیش‌فرض 'zigzag'")
            return self._calculate_zigzag_levels(levels)
    
    def _calculate_zigzag_levels(self, levels: int = 3) -> List[Dict[str, float]]:
        """
        محاسبه سطوح حمایت و مقاومت با استفاده از الگوریتم ZigZag
        
        Args:
            levels (int): تعداد سطوح مورد نیاز
            
        Returns:
            list: لیست دیکشنری‌های سطوح
        """
        # پارامترهای ZigZag
        deviation = 5.0  # درصد انحراف
        high = self.df['high'].values
        low = self.df['low'].values
        
        # آرایه‌های نقاط ZigZag
        zigzag = np.zeros(len(high))
        
        # محاسبه نقاط بالا
        max_price = high[0]
        min_price = low[0]
        trend = 0  # 1 = صعودی، -1 = نزولی
        
        for i in range(1, len(high)):
            if trend == 0:
                # تعیین روند اولیه
                if high[i] > max_price:
                    max_price = high[i]
                    trend = 1
                elif low[i] < min_price:
                    min_price = low[i]
                    trend = -1
            elif trend == 1:
                # روند صعودی
                if high[i] > max_price:
                    max_price = high[i]
                elif low[i] < min_price * (1 - deviation / 100):
                    # تغییر روند به نزولی
                    zigzag[i-1] = max_price
                    min_price = low[i]
                    trend = -1
            elif trend == -1:
                # روند نزولی
                if low[i] < min_price:
                    min_price = low[i]
                elif high[i] > max_price * (1 + deviation / 100):
                    # تغییر روند به صعودی
                    zigzag[i-1] = min_price
                    max_price = high[i]
                    trend = 1
        
        # استخراج سطوح مقاومت (نقاط بالا)
        resistance_points = np.array([p for p in zigzag if p > 0])
        resistance_points = np.sort(resistance_points)[::-1]  # مرتب‌سازی نزولی
        
        # استخراج سطوح حمایت (نقاط پایین)
        support_points = np.array([p for p in zigzag if p < 0])
        support_points = np.sort(np.abs(support_points))  # مرتب‌سازی صعودی
        
        # انتخاب سطوح برتر
        current_price = self.df['close'].iloc[-1]
        
        result = []
        
        # سطوح مقاومت (بالاتر از قیمت فعلی)
        resistance_above = resistance_points[resistance_points > current_price]
        for i, level in enumerate(resistance_above[:levels]):
            result.append({
                'type': 'resistance',
                'level': level,
                'strength': 100 - (i * 10),
                'distance': ((level / current_price) - 1) * 100  # درصد فاصله از قیمت فعلی
            })
        
        # سطوح حمایت (پایین‌تر از قیمت فعلی)
        support_below = support_points[support_points < current_price]
        for i, level in enumerate(support_below[:levels]):
            result.append({
                'type': 'support',
                'level': level,
                'strength': 100 - (i * 10),
                'distance': ((current_price / level) - 1) * 100  # درصد فاصله از قیمت فعلی
            })
        
        return result
    
    def _calculate_peaks_levels(self, levels: int = 3) -> List[Dict[str, float]]:
        """
        محاسبه سطوح حمایت و مقاومت با استفاده از پیک‌ها و دره‌ها
        
        Args:
            levels (int): تعداد سطوح مورد نیاز
            
        Returns:
            list: لیست دیکشنری‌های سطوح
        """
        high = self.df['high'].values
        low = self.df['low'].values
        window_size = 10  # اندازه پنجره برای یافتن پیک‌ها
        
        # یافتن پیک‌ها (نقاط مقاومت)
        peaks = []
        for i in range(window_size, len(high) - window_size):
            if high[i] == max(high[i - window_size:i + window_size + 1]):
                peaks.append((i, high[i]))
        
        # یافتن دره‌ها (نقاط حمایت)
        troughs = []
        for i in range(window_size, len(low) - window_size):
            if low[i] == min(low[i - window_size:i + window_size + 1]):
                troughs.append((i, low[i]))
        
        # مرتب‌سازی نقاط مقاومت بر اساس قیمت (نزولی)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # مرتب‌سازی نقاط حمایت بر اساس قیمت (صعودی)
        troughs.sort(key=lambda x: x[1])
        
        # انتخاب سطوح برتر
        current_price = self.df['close'].iloc[-1]
        
        result = []
        
        # سطوح مقاومت (بالاتر از قیمت فعلی)
        resistance_above = [(i, p) for i, p in peaks if p > current_price]
        for idx, (_, level) in enumerate(resistance_above[:levels]):
            result.append({
                'type': 'resistance',
                'level': level,
                'strength': 100 - (idx * 10),
                'distance': ((level / current_price) - 1) * 100  # درصد فاصله از قیمت فعلی
            })
        
        # سطوح حمایت (پایین‌تر از قیمت فعلی)
        support_below = [(i, p) for i, p in troughs if p < current_price]
        for idx, (_, level) in enumerate(support_below[:levels]):
            result.append({
                'type': 'support',
                'level': level,
                'strength': 100 - (idx * 10),
                'distance': ((current_price / level) - 1) * 100  # درصد فاصله از قیمت فعلی
            })
        
        return result
    
    def _calculate_fractal_levels(self, levels: int = 3) -> List[Dict[str, float]]:
        """
        محاسبه سطوح حمایت و مقاومت با استفاده از فرکتال‌های ویلیامز
        
        Args:
            levels (int): تعداد سطوح مورد نیاز
            
        Returns:
            list: لیست دیکشنری‌های سطوح
        """
        high = self.df['high'].values
        low = self.df['low'].values
        
        # یافتن فرکتال‌های بالا (مقاومت)
        up_fractals = []
        for i in range(2, len(high) - 2):
            if high[i] > high[i-2] and high[i] > high[i-1] and high[i] > high[i+1] and high[i] > high[i+2]:
                up_fractals.append((i, high[i]))
        
        # یافتن فرکتال‌های پایین (حمایت)
        down_fractals = []
        for i in range(2, len(low) - 2):
            if low[i] < low[i-2] and low[i] < low[i-1] and low[i] < low[i+1] and low[i] < low[i+2]:
                down_fractals.append((i, low[i]))
        
        # مرتب‌سازی فرکتال‌های بالا بر اساس قیمت (نزولی)
        up_fractals.sort(key=lambda x: x[1], reverse=True)
        
        # مرتب‌سازی فرکتال‌های پایین بر اساس قیمت (صعودی)
        down_fractals.sort(key=lambda x: x[1])
        
        # انتخاب سطوح برتر
        current_price = self.df['close'].iloc[-1]
        
        result = []
        
        # سطوح مقاومت (بالاتر از قیمت فعلی)
        resistance_above = [(i, p) for i, p in up_fractals if p > current_price]
        for idx, (_, level) in enumerate(resistance_above[:levels]):
            result.append({
                'type': 'resistance',
                'level': level,
                'strength': 100 - (idx * 10),
                'distance': ((level / current_price) - 1) * 100  # درصد فاصله از قیمت فعلی
            })
        
        # سطوح حمایت (پایین‌تر از قیمت فعلی)
        support_below = [(i, p) for i, p in down_fractals if p < current_price]
        for idx, (_, level) in enumerate(support_below[:levels]):
            result.append({
                'type': 'support',
                'level': level,
                'strength': 100 - (idx * 10),
                'distance': ((current_price / level) - 1) * 100  # درصد فاصله از قیمت فعلی
            })
        
        return result
    
    def _calculate_pivot_levels(self) -> List[Dict[str, float]]:
        """
        محاسبه سطوح حمایت و مقاومت با استفاده از نقاط پیووت
        
        Returns:
            list: لیست دیکشنری‌های سطوح
        """
        # استفاده از داده‌های آخرین روز
        high = self.df['high'].iloc[-1]
        low = self.df['low'].iloc[-1]
        close = self.df['close'].iloc[-1]
        
        # محاسبه نقطه پیووت
        pivot = (high + low + close) / 3
        
        # محاسبه سطوح مقاومت
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # محاسبه سطوح حمایت
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # ایجاد لیست سطوح
        levels = [
            {'type': 'resistance', 'level': r3, 'strength': 70, 'distance': ((r3 / close) - 1) * 100, 'name': 'R3'},
            {'type': 'resistance', 'level': r2, 'strength': 80, 'distance': ((r2 / close) - 1) * 100, 'name': 'R2'},
            {'type': 'resistance', 'level': r1, 'strength': 90, 'distance': ((r1 / close) - 1) * 100, 'name': 'R1'},
            {'type': 'pivot', 'level': pivot, 'strength': 100, 'distance': ((pivot / close) - 1) * 100, 'name': 'P'},
            {'type': 'support', 'level': s1, 'strength': 90, 'distance': ((close / s1) - 1) * 100, 'name': 'S1'},
            {'type': 'support', 'level': s2, 'strength': 80, 'distance': ((close / s2) - 1) * 100, 'name': 'S2'},
            {'type': 'support', 'level': s3, 'strength': 70, 'distance': ((close / s3) - 1) * 100, 'name': 'S3'}
        ]
        
        return levels
    
    def calculate_fibonacci_levels(self, trend: str = 'auto') -> List[Dict[str, float]]:
        """
        محاسبه سطوح فیبوناچی
        
        Args:
            trend (str): روند قیمت ('up', 'down', 'auto')
            
        Returns:
            list: لیست دیکشنری‌های سطوح فیبوناچی
        """
        if self.df is None or self.df.empty:
            logger.error("دیتافریم خالی برای محاسبه سطوح فیبوناچی")
            return []
        
        # تعیین روند اتوماتیک
        if trend == 'auto':
            # محاسبه میانگین متحرک برای تعیین روند
            ma = self.df['close'].rolling(window=20).mean()
            if ma.iloc[-1] > ma.iloc[-20]:
                trend = 'up'
            else:
                trend = 'down'
        
        # پیدا کردن نقاط بالا و پایین
        high = self.df['high'].values
        low = self.df['low'].values
        
        if trend == 'up':
            # روند صعودی: از پایین‌ترین به بالاترین
            lowest = np.min(low[-100:])
            highest = np.max(high[-20:])
        else:
            # روند نزولی: از بالاترین به پایین‌ترین
            lowest = np.min(low[-20:])
            highest = np.max(high[-100:])
        
        # سطوح فیبوناچی
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618]
        
        # محاسبه قیمت‌های سطوح فیبوناچی
        price_range = highest - lowest
        
        if trend == 'up':
            # سطوح ریتریسمنت (روند صعودی)
            levels = [{'level': highest - (price_range * level), 'ratio': level} for level in fib_levels]
        else:
            # سطوح ریتریسمنت (روند نزولی)
            levels = [{'level': lowest + (price_range * level), 'ratio': level} for level in fib_levels]
        
        # افزودن جزئیات بیشتر
        current_price = self.df['close'].iloc[-1]
        
        result = []
        for item in levels:
            level = item['level']
            ratio = item['ratio']
            
            # تعیین نوع سطح
            if level > current_price:
                level_type = 'resistance'
            elif level < current_price:
                level_type = 'support'
            else:
                level_type = 'current'
            
            # محاسبه فاصله از قیمت فعلی
            distance = abs((level / current_price - 1) * 100)
            
            # تعیین قدرت سطح
            if ratio in [0, 1]:
                strength = 100
            elif ratio in [0.5, 0.618]:
                strength = 90
            elif ratio in [0.382, 0.786]:
                strength = 80
            elif ratio == 0.236:
                strength = 70
            else:
                strength = 60
            
            # اضافه کردن به نتیجه
            result.append({
                'type': level_type,
                'level': level,
                'ratio': ratio,
                'strength': strength,
                'distance': distance,
                'name': f'Fib {ratio}'
            })
        
        return result
    
    def calculate_entry_exit_points(self, signal: str, technical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        محاسبه نقاط ورود و خروج بر اساس سیگنال و داده‌های تکنیکال
        
        Args:
            signal (str): سیگنال معاملاتی ('BUY', 'SELL', 'NEUTRAL')
            technical_data (dict, optional): داده‌های تکنیکال اضافی
            
        Returns:
            dict: دیکشنری نقاط ورود، خروج و حد ضرر
        """
        if self.df is None or self.df.empty:
            logger.error("دیتافریم خالی برای محاسبه نقاط ورود و خروج")
            return {}
        
        # قیمت فعلی
        current_price = self.df['close'].iloc[-1]
        
        # محاسبه سطوح حمایت و مقاومت
        levels = self.calculate_support_resistance()
        
        # فیلتر کردن سطوح بر اساس نوع
        resistance_levels = sorted([l['level'] for l in levels if l['type'] == 'resistance'])
        support_levels = sorted([l['level'] for l in levels if l['type'] == 'support'], reverse=True)
        
        # نتیجه پیش‌فرض
        result = {
            'entry_price': current_price,
            'stop_loss': 0,
            'targets': [],
            'risk_reward_ratio': self.reward_ratio,
            'risk_percent': self.risk_percent,
            'signal': signal
        }
        
        if signal == 'BUY':
            # محاسبه حد ضرر برای خرید
            stop_loss = min(support_levels) if support_levels else current_price * (1 - self.risk_percent / 100)
            
            # محاسبه اهداف قیمتی برای خرید
            if resistance_levels:
                targets = []
                for i, level in enumerate(resistance_levels):
                    risk = current_price - stop_loss
                    target_distance = risk * self.reward_ratio * (i+1)
                    targets.append({
                        'price': current_price + target_distance,
                        'potential': ((current_price + target_distance) / current_price - 1) * 100,
                        'name': f'Target {i+1}'
                    })
            else:
                # اگر سطح مقاومتی یافت نشد، از نسبت ریوارد به ریسک استفاده می‌کنیم
                risk = current_price - stop_loss
                targets = [{
                    'price': current_price + (risk * self.reward_ratio),
                    'potential': self.risk_percent * self.reward_ratio,
                    'name': 'Target 1'
                }]
            
            result['entry_price'] = current_price
            result['stop_loss'] = stop_loss
            result['targets'] = targets
            result['risk_percent'] = (current_price - stop_loss) / current_price * 100
            
        elif signal == 'SELL':
            # محاسبه حد ضرر برای فروش
            stop_loss = max(resistance_levels) if resistance_levels else current_price * (1 + self.risk_percent / 100)
            
            # محاسبه اهداف قیمتی برای فروش
            if support_levels:
                targets = []
                for i, level in enumerate(support_levels):
                    risk = stop_loss - current_price
                    target_distance = risk * self.reward_ratio * (i+1)
                    targets.append({
                        'price': current_price - target_distance,
                        'potential': ((current_price - target_distance) / current_price - 1) * -100,
                        'name': f'Target {i+1}'
                    })
            else:
                # اگر سطح حمایتی یافت نشد، از نسبت ریوارد به ریسک استفاده می‌کنیم
                risk = stop_loss - current_price
                targets = [{
                    'price': current_price - (risk * self.reward_ratio),
                    'potential': self.risk_percent * self.reward_ratio,
                    'name': 'Target 1'
                }]
            
            result['entry_price'] = current_price
            result['stop_loss'] = stop_loss
            result['targets'] = targets
            result['risk_percent'] = (stop_loss - current_price) / current_price * 100
        
        return result
    
    def calculate_risk_management(self, entry_price: float, stop_loss: float, 
                               target_prices: List[float], balance: float, 
                               max_risk_percent: float = 2.0) -> Dict[str, Any]:
        """
        محاسبه مدیریت ریسک
        
        Args:
            entry_price (float): قیمت ورود
            stop_loss (float): قیمت حد ضرر
            target_prices (list): لیست قیمت‌های هدف
            balance (float): موجودی حساب
            max_risk_percent (float): حداکثر درصد ریسک
            
        Returns:
            dict: اطلاعات مدیریت ریسک
        """
        # محاسبه ریسک
        if entry_price > stop_loss:  # پوزیشن خرید
            risk_percent = (entry_price - stop_loss) / entry_price * 100
            position_type = 'LONG'
        else:  # پوزیشن فروش
            risk_percent = (stop_loss - entry_price) / entry_price * 100
            position_type = 'SHORT'
        
        # محاسبه حجم معامله
        risk_amount = balance * max_risk_percent / 100
        position_size = risk_amount / risk_percent
        
        # محاسبه تعداد سکه/توکن
        coins = position_size / entry_price
        
        # محاسبه سود برای هر هدف
        profits = []
        for target in target_prices:
            if position_type == 'LONG':
                profit_percent = (target - entry_price) / entry_price * 100
                profit_amount = position_size * profit_percent / 100
            else:
                profit_percent = (entry_price - target) / entry_price * 100
                profit_amount = position_size * profit_percent / 100
            
            profits.append({
                'target_price': target,
                'profit_percent': profit_percent,
                'profit_amount': profit_amount,
                'risk_reward': profit_percent / risk_percent
            })
        
        return {
            'position_type': position_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_percent': risk_percent,
            'risk_amount': risk_amount,
            'position_size': position_size,
            'coins': coins,
            'profits': profits
        }
    
    def calculate_dynamic_profit_points(self, signal: str, current_price: float, atr_value: float, 
                                     method: str = 'atr', profit_levels: int = 3) -> List[Dict[str, float]]:
        """
        محاسبه نقاط سود داینامیک
        
        Args:
            signal (str): سیگنال معاملاتی ('BUY', 'SELL', 'NEUTRAL')
            current_price (float): قیمت فعلی
            atr_value (float): مقدار ATR
            method (str): روش محاسبه ('atr', 'percent', 'fibonacci')
            profit_levels (int): تعداد سطوح سود
            
        Returns:
            list: لیست نقاط سود
        """
        if method == 'atr':
            # محاسبه بر اساس ATR
            profit_points = []
            for i in range(1, profit_levels + 1):
                if signal == 'BUY':
                    profit_price = current_price + (atr_value * i * 2)
                    profit_percent = (profit_price / current_price - 1) * 100
                else:  # SELL
                    profit_price = current_price - (atr_value * i * 2)
                    profit_percent = (current_price / profit_price - 1) * 100
                
                profit_points.append({
                    'level': i,
                    'price': profit_price,
                    'percent': profit_percent,
                    'atr_multiple': i * 2
                })
            
            return profit_points
        
        elif method == 'percent':
            # محاسبه بر اساس درصد
            percent_levels = [1.5, 3.0, 7.0, 15.0, 25.0]
            profit_points = []
            
            for i in range(min(profit_levels, len(percent_levels))):
                percent = percent_levels[i]
                if signal == 'BUY':
                    profit_price = current_price * (1 + percent / 100)
                else:  # SELL
                    profit_price = current_price * (1 - percent / 100)
                
                profit_points.append({
                    'level': i + 1,
                    'price': profit_price,
                    'percent': percent,
                    'atr_multiple': percent / (atr_value / current_price * 100)
                })
            
            return profit_points
        
        elif method == 'fibonacci':
            # محاسبه بر اساس فیبوناچی
            fib_levels = [1.618, 2.0, 2.618, 3.618, 4.618]
            profit_points = []
            
            # محاسبه دامنه ATR
            atr_range = atr_value
            
            for i in range(min(profit_levels, len(fib_levels))):
                fib = fib_levels[i]
                if signal == 'BUY':
                    profit_price = current_price + (atr_range * fib)
                    profit_percent = (profit_price / current_price - 1) * 100
                else:  # SELL
                    profit_price = current_price - (atr_range * fib)
                    profit_percent = (current_price / profit_price - 1) * 100
                
                profit_points.append({
                    'level': i + 1,
                    'price': profit_price,
                    'percent': profit_percent,
                    'fib_level': fib
                })
            
            return profit_points
        
        else:
            logger.warning(f"روش {method} نامعتبر است. استفاده از روش پیش‌فرض 'atr'")
            return self.calculate_dynamic_profit_points(signal, current_price, atr_value, method='atr', profit_levels=profit_levels)
    
    def calculate_vwap_targets(self, vwap: float, current_price: float, signal: str) -> List[Dict[str, float]]:
        """
        محاسبه اهداف قیمتی بر اساس VWAP
        
        Args:
            vwap (float): قیمت VWAP
            current_price (float): قیمت فعلی
            signal (str): سیگنال معاملاتی ('BUY', 'SELL', 'NEUTRAL')
            
        Returns:
            list: لیست اهداف قیمتی
        """
        # محاسبه فاصله بین قیمت فعلی و VWAP
        vwap_distance = abs(current_price - vwap)
        
        # ضرایب VWAP
        vwap_multiples = [1.0, 1.5, 2.0, 3.0]
        
        # محاسبه اهداف
        targets = []
        
        if signal == 'BUY':
            for i, multiple in enumerate(vwap_multiples):
                target_price = vwap + (vwap_distance * multiple)
                percent = (target_price / current_price - 1) * 100
                
                targets.append({
                    'level': i + 1,
                    'price': target_price,
                    'percent': percent,
                    'vwap_multiple': multiple
                })
        
        elif signal == 'SELL':
            for i, multiple in enumerate(vwap_multiples):
                target_price = vwap - (vwap_distance * multiple)
                percent = (current_price / target_price - 1) * 100
                
                targets.append({
                    'level': i + 1,
                    'price': target_price,
                    'percent': percent,
                    'vwap_multiple': multiple
                })
        
        return targets


# ایجاد نمونه سراسری
_target_calculator_instance = None

def get_target_calculator(df: Optional[pd.DataFrame] = None, risk_percent: float = 1.0, reward_ratio: float = 3.0) -> TargetPriceCalculator:
    """
    دریافت نمونه سراسری از کلاس TargetPriceCalculator
    
    Args:
        df (pd.DataFrame, optional): دیتافریم داده‌های OHLCV
        risk_percent (float): درصد ریسک
        reward_ratio (float): نسبت ریوارد به ریسک
        
    Returns:
        TargetPriceCalculator: نمونه سراسری
    """
    global _target_calculator_instance
    
    if _target_calculator_instance is None or df is not None:
        _target_calculator_instance = TargetPriceCalculator(df, risk_percent, reward_ratio)
    
    return _target_calculator_instance