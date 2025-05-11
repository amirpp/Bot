"""
ماژول انتخاب خودکار اندیکاتورهای مناسب برای شرایط فعلی بازار

این ماژول شامل توابع و کلاس‌های مورد نیاز برای انتخاب هوشمند بهترین اندیکاتورهای تکنیکال
برای شرایط فعلی بازار و نوع ارز مورد تحلیل است.
"""

import pandas as pd
import numpy as np
import logging
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from technical_analysis import AVAILABLE_INDICATORS, TOP_INDICATORS

# تنظیم لاگر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndicatorSelector:
    """کلاس انتخاب هوشمند اندیکاتورهای مناسب"""
    
    def __init__(self, df: pd.DataFrame, target_forward_periods: int = 5, symbol: str = "BTC/USDT"):
        """
        مقداردهی اولیه انتخاب‌کننده اندیکاتور
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            target_forward_periods (int): تعداد دوره‌های آینده برای پیش‌بینی
            symbol (str): نماد ارز
        """
        self.df = df
        self.target_forward_periods = target_forward_periods
        self.symbol = symbol
        
        # تعیین وزن‌های اولیه برای انواع مختلف اندیکاتورها بر اساس نوع ارز
        self.indicator_weights = self._initialize_indicator_weights()
        
    def _initialize_indicator_weights(self) -> Dict[str, float]:
        """
        مقداردهی اولیه وزن‌های اندیکاتورها بر اساس نوع ارز
        
        Returns:
            dict: دیکشنری وزن‌های اندیکاتورها
        """
        # وزن‌های پایه
        weights = {
            'momentum': 1.0,  # اندیکاتورهای مومنتوم
            'trend': 1.0,     # اندیکاتورهای روند
            'volatility': 1.0, # اندیکاتورهای نوسان
            'volume': 1.0,     # اندیکاتورهای حجم
            'pattern': 1.0     # اندیکاتورهای الگو
        }
        
        # تنظیم وزن‌ها بر اساس نوع ارز
        if 'BTC' in self.symbol:
            # بیت‌کوین: اندیکاتورهای روند و حجم مهم‌تر هستند
            weights['trend'] = 1.5
            weights['volume'] = 1.3
            
        elif 'ETH' in self.symbol:
            # اتریوم: اندیکاتورهای مومنتوم و نوسان مهم‌تر هستند
            weights['momentum'] = 1.3
            weights['volatility'] = 1.2
            
        elif any(alt in self.symbol for alt in ['XRP', 'ADA', 'SOL']):
            # آلت‌کوین‌های بزرگ: تعادل بین روند و مومنتوم
            weights['trend'] = 1.2
            weights['momentum'] = 1.2
            
        elif any(alt in self.symbol for alt in ['DOGE', 'SHIB']):
            # میم‌کوین‌ها: نوسان و حجم مهم‌تر هستند
            weights['volatility'] = 1.5
            weights['volume'] = 1.4
            
        # بررسی نوسان اخیر
        if len(self.df) > 20:
            recent_volatility = self.df['close'].pct_change().std() * np.sqrt(252)
            
            # برای ارزهای با نوسان بالا، اندیکاتورهای نوسان مهم‌تر هستند
            if recent_volatility > 0.05:  # نوسان بالاتر از 5% روزانه
                weights['volatility'] *= 1.2
                weights['pattern'] *= 1.1
            else:
                weights['trend'] *= 1.1
                
        return weights
        
    def _categorize_indicators(self) -> Dict[str, List[str]]:
        """
        دسته‌بندی اندیکاتورها به گروه‌های مختلف
        
        Returns:
            dict: دیکشنری اندیکاتورها بر اساس دسته‌بندی
        """
        categorized = {
            'momentum': ['RSI', 'Stochastic', 'CCI', 'MFI', 'Momentum', 'ROC', 'Williams %R', 'Awesome Oscillator'],
            'trend': ['Moving Average', 'MACD', 'ADX', 'Ichimoku', 'Supertrend', 'Parabolic SAR', 'MESA', 'TRIX', 'KAMA', 'Hull Moving Average'],
            'volatility': ['Bollinger Bands', 'ATR', 'Keltner Channel', 'Standard Deviation', 'Donchian Channel'],
            'volume': ['Volume', 'OBV', 'Chaikin Money Flow', 'VWAP', 'PVT', 'MFI', 'Volume Profile', 'Elder Force Index'],
            'pattern': ['Fibonacci', 'ZigZag', 'Elliott Wave', 'Harmonic Patterns', 'Chart Patterns', 'Heikin-Ashi', 'Pivot Points']
        }
        
        return categorized
        
    def evaluate_market_condition(self) -> Dict[str, Any]:
        """
        ارزیابی شرایط فعلی بازار
        
        Returns:
            dict: دیکشنری وضعیت بازار
        """
        # ایجاد یک دیتافریم از 50 داده آخر
        recent_df = self.df.tail(50)
        
        # محاسبه پارامترهای مهم بازار
        market_state = {}
        
        # 1. محاسبه روند - با استفاده از شیب میانگین متحرک 20 روزه
        if 'close' in recent_df.columns:
            closes = recent_df['close'].values
            if len(closes) > 20:
                # محاسبه میانگین متحرک 20 روزه
                ma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
                # محاسبه شیب
                trend_slope = (ma20[-1] - ma20[0]) / (len(ma20) - 1)
                # نرمال‌سازی شیب
                normalized_slope = trend_slope / ma20[0]
                
                if normalized_slope > 0.01:
                    market_state['trend'] = 'strong_uptrend'
                elif normalized_slope > 0.001:
                    market_state['trend'] = 'uptrend'
                elif normalized_slope < -0.01:
                    market_state['trend'] = 'strong_downtrend'
                elif normalized_slope < -0.001:
                    market_state['trend'] = 'downtrend'
                else:
                    market_state['trend'] = 'sideways'
            else:
                market_state['trend'] = 'unknown'
        else:
            market_state['trend'] = 'unknown'
            
        # 2. محاسبه نوسان - با استفاده از انحراف معیار تغییرات قیمت
        if 'close' in recent_df.columns and len(recent_df) > 5:
            returns = recent_df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # تبدیل به نوسان سالانه
            
            if volatility > 0.1:
                market_state['volatility'] = 'high'
            elif volatility > 0.05:
                market_state['volatility'] = 'medium'
            else:
                market_state['volatility'] = 'low'
        else:
            market_state['volatility'] = 'unknown'
            
        # 3. محاسبه حجم - مقایسه حجم اخیر با میانگین حجم
        if 'volume' in recent_df.columns and len(recent_df) > 10:
            recent_volume = recent_df['volume'].iloc[-5:].mean()
            avg_volume = recent_df['volume'].mean()
            
            volume_ratio = recent_volume / avg_volume
            
            if volume_ratio > 1.5:
                market_state['volume'] = 'high'
            elif volume_ratio < 0.7:
                market_state['volume'] = 'low'
            else:
                market_state['volume'] = 'normal'
        else:
            market_state['volume'] = 'unknown'
            
        # 4. تشخیص الگوی بازار
        if 'close' in recent_df.columns and len(recent_df) > 20:
            # بررسی تغییرات قیمت برای تشخیص الگوهای بازار
            changes = recent_df['close'].pct_change().dropna()
            pos_changes = (changes > 0).sum()
            neg_changes = (changes < 0).sum()
            
            if pos_changes > neg_changes * 2:
                market_state['pattern'] = 'bull_run'
            elif neg_changes > pos_changes * 2:
                market_state['pattern'] = 'bear_market'
            elif pos_changes > neg_changes * 1.5:
                market_state['pattern'] = 'upward_momentum'
            elif neg_changes > pos_changes * 1.5:
                market_state['pattern'] = 'downward_pressure'
            else:
                # تشخیص الگوی رنج برای بازار
                high_low_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['low'].min()
                if high_low_range < 0.05:  # محدوده کمتر از 5%
                    market_state['pattern'] = 'tight_range'
                else:
                    market_state['pattern'] = 'choppy'
        else:
            market_state['pattern'] = 'unknown'
        
        return market_state
        
    def select_top_indicators(self, top_n: int = 10) -> List[str]:
        """
        انتخاب بهترین اندیکاتورها بر اساس شرایط بازار
        
        Args:
            top_n (int): تعداد اندیکاتورهای انتخابی
            
        Returns:
            list: لیست اندیکاتورهای انتخاب شده
        """
        # بررسی شرایط بازار
        market_state = self.evaluate_market_condition()
        
        # دسته‌بندی اندیکاتورها
        categorized_indicators = self._categorize_indicators()
        
        # نهایی کردن وزن‌ها بر اساس شرایط بازار
        final_weights = self.indicator_weights.copy()
        
        # تنظیم وزن‌ها بر اساس روند
        if market_state['trend'] in ['strong_uptrend', 'strong_downtrend']:
            final_weights['trend'] *= 1.5
            final_weights['momentum'] *= 1.2
        elif market_state['trend'] == 'sideways':
            final_weights['volatility'] *= 1.3
            final_weights['pattern'] *= 1.2
            
        # تنظیم وزن‌ها بر اساس نوسان
        if market_state['volatility'] == 'high':
            final_weights['volatility'] *= 1.4
            final_weights['momentum'] *= 1.2
        elif market_state['volatility'] == 'low':
            final_weights['trend'] *= 1.3
            
        # تنظیم وزن‌ها بر اساس حجم
        if market_state['volume'] == 'high':
            final_weights['volume'] *= 1.5
        elif market_state['volume'] == 'low':
            final_weights['trend'] *= 1.2
            
        # تنظیم وزن‌ها بر اساس الگوی بازار
        if market_state['pattern'] in ['bull_run', 'upward_momentum']:
            final_weights['momentum'] *= 1.3
            final_weights['trend'] *= 1.2
        elif market_state['pattern'] in ['bear_market', 'downward_pressure']:
            final_weights['volume'] *= 1.2
            final_weights['trend'] *= 1.2
        elif market_state['pattern'] in ['tight_range', 'choppy']:
            final_weights['volatility'] *= 1.4
            final_weights['pattern'] *= 1.3
            
        # انتخاب اندیکاتورها از هر دسته با در نظر گرفتن وزن‌ها
        selected_indicators = []
        
        # تعیین تعداد اندیکاتور از هر دسته
        total_weight = sum(final_weights.values())
        category_counts = {}
        
        for category, weight in final_weights.items():
            # تعداد اندیکاتور از هر دسته متناسب با وزن آن
            category_counts[category] = max(1, int((weight / total_weight) * top_n))
            
        # تنظیم مجدد برای حصول اطمینان از تعداد کلی صحیح
        while sum(category_counts.values()) > top_n:
            # کاهش از دسته با کمترین وزن
            min_category = min(final_weights, key=final_weights.get)
            if category_counts[min_category] > 1:
                category_counts[min_category] -= 1
            else:
                # اگر همه دسته‌ها به حداقل رسیده‌اند، از دسته با بیشترین تعداد کم کنیم
                max_count_category = max(category_counts, key=category_counts.get)
                category_counts[max_count_category] -= 1
        
        # انتخاب اندیکاتورها از هر دسته
        for category, count in category_counts.items():
            available_in_category = categorized_indicators[category]
            
            # انتخاب تصادفی اندیکاتورها از هر دسته
            if len(available_in_category) <= count:
                selected_from_category = available_in_category
            else:
                # انتخاب با توجه به اولویت‌های خاص
                if category == 'trend' and market_state['trend'] in ['strong_uptrend', 'strong_downtrend']:
                    # برای روندهای قوی، اولویت با اندیکاتورهای تعقیب‌کننده روند مانند میانگین متحرک و MACD
                    high_priority = ['Moving Average', 'MACD', 'Supertrend']
                    remaining = [ind for ind in available_in_category if ind not in high_priority]
                    
                    selected_from_category = high_priority[:min(len(high_priority), count)]
                    if len(selected_from_category) < count:
                        selected_from_category.extend(random.sample(remaining, count - len(selected_from_category)))
                        
                elif category == 'volatility' and market_state['volatility'] == 'high':
                    # برای نوسان بالا، اولویت با Bollinger Bands و ATR
                    high_priority = ['Bollinger Bands', 'ATR', 'Keltner Channel']
                    remaining = [ind for ind in available_in_category if ind not in high_priority]
                    
                    selected_from_category = high_priority[:min(len(high_priority), count)]
                    if len(selected_from_category) < count:
                        selected_from_category.extend(random.sample(remaining, count - len(selected_from_category)))
                        
                else:
                    # انتخاب تصادفی برای سایر شرایط
                    selected_from_category = random.sample(available_in_category, count)
                
            selected_indicators.extend(selected_from_category)
            
        # حصول اطمینان از اینکه تعداد صحیح اندیکاتور برگردانده می‌شود
        if len(selected_indicators) > top_n:
            selected_indicators = selected_indicators[:top_n]
        elif len(selected_indicators) < top_n:
            # اضافه کردن اندیکاتورهای باقی‌مانده به صورت تصادفی تا رسیدن به تعداد مورد نظر
            all_indicators = [ind for category_inds in categorized_indicators.values() for ind in category_inds]
            remaining = [ind for ind in all_indicators if ind not in selected_indicators]
            
            # انتخاب تصادفی از باقی‌مانده‌ها
            additional = random.sample(remaining, min(top_n - len(selected_indicators), len(remaining)))
            selected_indicators.extend(additional)
        
        # اطمینان از وجود حداقل یک اندیکاتور از هر دسته مهم
        essential_categories = ['trend', 'momentum', 'volatility']
        for category in essential_categories:
            category_present = any(ind in categorized_indicators[category] for ind in selected_indicators)
            if not category_present and len(categorized_indicators[category]) > 0:
                # اضافه کردن یک اندیکاتور از دسته مورد نظر
                selected_ind = random.choice(categorized_indicators[category])
                
                # حذف یک اندیکاتور تصادفی برای حفظ تعداد
                if len(selected_indicators) >= top_n:
                    # حذف از دسته‌ای که بیشترین تعداد را دارد
                    counts = {cat: sum(1 for ind in selected_indicators if ind in categorized_indicators[cat]) 
                             for cat in categorized_indicators.keys()}
                    max_cat = max(counts, key=counts.get)
                    
                    # انتخاب یک اندیکاتور تصادفی از دسته با بیشترین تعداد
                    candidates = [ind for ind in selected_indicators if ind in categorized_indicators[max_cat]]
                    if candidates:
                        selected_indicators.remove(random.choice(candidates))
                        
                selected_indicators.append(selected_ind)
                
        return selected_indicators
        
    def get_indicator_importance(self, indicators: List[str]) -> Dict[str, float]:
        """
        محاسبه اهمیت هر اندیکاتور برای شرایط فعلی
        
        Args:
            indicators (list): لیست اندیکاتورهای انتخاب شده
            
        Returns:
            dict: دیکشنری اهمیت هر اندیکاتور
        """
        # دسته‌بندی اندیکاتورها
        categorized = self._categorize_indicators()
        
        # ایجاد دیکشنری معکوس برای پیدا کردن دسته هر اندیکاتور
        indicator_to_category = {}
        for category, inds in categorized.items():
            for ind in inds:
                indicator_to_category[ind] = category
        
        # محاسبه اهمیت هر اندیکاتور
        importance = {}
        market_state = self.evaluate_market_condition()
        
        for indicator in indicators:
            # یافتن دسته اندیکاتور
            category = None
            for cat_ind, cat in indicator_to_category.items():
                if cat_ind in indicator:
                    category = cat
                    break
            
            if not category:
                category = 'trend'  # دسته پیش‌فرض
                
            # اختصاص امتیاز پایه
            base_score = self.indicator_weights.get(category, 1.0)
            
            # تنظیم امتیاز بر اساس شرایط بازار
            modifier = 1.0
            
            # روند
            if category == 'trend':
                if market_state['trend'] in ['strong_uptrend', 'strong_downtrend']:
                    modifier *= 1.5
                elif market_state['trend'] == 'sideways':
                    modifier *= 0.8
            
            # نوسان
            elif category == 'volatility':
                if market_state['volatility'] == 'high':
                    modifier *= 1.4
                elif market_state['volatility'] == 'low':
                    modifier *= 0.9
            
            # حجم
            elif category == 'volume':
                if market_state['volume'] == 'high':
                    modifier *= 1.3
                elif market_state['volume'] == 'low':
                    modifier *= 0.7
            
            # مومنتوم
            elif category == 'momentum':
                if market_state['trend'] in ['uptrend', 'strong_uptrend'] or \
                   market_state['trend'] in ['downtrend', 'strong_downtrend']:
                    modifier *= 1.2
                    
            # اختصاص امتیازهای ویژه برای برخی اندیکاتورها
            if 'RSI' in indicator:
                if market_state['trend'] == 'sideways':
                    modifier *= 1.3  # RSI در بازارهای رنج بسیار مفید است
            
            elif 'Bollinger' in indicator:
                if market_state['volatility'] == 'high':
                    modifier *= 1.3  # باندهای بولینگر در نوسانات بالا مفید هستند
            
            elif 'MACD' in indicator:
                if market_state['trend'] in ['uptrend', 'downtrend']:
                    modifier *= 1.2  # MACD در تشخیص روندها مفید است
            
            # محاسبه امتیاز نهایی
            importance[indicator] = base_score * modifier
        
        # نرمال‌سازی امتیازها
        if indicators:
            max_score = max(importance.values())
            for ind in importance:
                importance[ind] = (importance[ind] / max_score) * 100
        
        return importance
    
    def dynamic_indicator_adjustment(self, initial_indicators: List[str], performance_data: Optional[Dict[str, float]] = None) -> List[str]:
        """
        تنظیم پویای اندیکاتورها بر اساس عملکرد آنها
        
        Args:
            initial_indicators (list): لیست اولیه اندیکاتورها
            performance_data (dict, optional): داده‌های عملکرد اندیکاتورها
            
        Returns:
            list: لیست تنظیم‌شده اندیکاتورها
        """
        if not performance_data:
            # اگر داده‌های عملکرد موجود نباشد، لیست اولیه را برمی‌گردانیم
            return initial_indicators
            
        # محاسبه اهمیت اندیکاتورها
        importance = self.get_indicator_importance(initial_indicators)
        
        # ترکیب اهمیت با داده‌های عملکرد
        combined_scores = {}
        for indicator in initial_indicators:
            # امتیاز اهمیت (0-100)
            importance_score = importance.get(indicator, 50)
            
            # امتیاز عملکرد (0-100)
            performance_score = performance_data.get(indicator, 50)
            
            # ترکیب امتیازها با وزن بیشتر برای عملکرد
            combined_scores[indicator] = 0.4 * importance_score + 0.6 * performance_score
            
        # مرتب‌سازی اندیکاتورها بر اساس امتیاز ترکیبی
        sorted_indicators = sorted(combined_scores.keys(), key=lambda ind: combined_scores[ind], reverse=True)
        
        # تعیین تعداد اندیکاتورها برای حفظ
        keep_count = int(len(initial_indicators) * 0.7)  # حفظ 70% از اندیکاتورهای اصلی
        keep_indicators = sorted_indicators[:keep_count]
        
        # انتخاب اندیکاتورهای جدید برای جایگزینی
        new_count = len(initial_indicators) - keep_count
        if new_count > 0:
            # دسته‌بندی اندیکاتورها
            categorized = self._categorize_indicators()
            all_indicators = [ind for category_inds in categorized.values() for ind in category_inds]
            
            # حذف اندیکاتورهای موجود در لیست
            available_indicators = [ind for ind in all_indicators if ind not in initial_indicators]
            
            # انتخاب تصادفی اندیکاتورهای جدید
            new_indicators = random.sample(available_indicators, min(new_count, len(available_indicators)))
            
            # ترکیب اندیکاتورهای حفظ شده و جدید
            adjusted_indicators = keep_indicators + new_indicators
            
            return adjusted_indicators
        else:
            return keep_indicators
