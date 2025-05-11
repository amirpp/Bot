"""
ماژول تحلیل فوق پیشرفته بازار ارزهای دیجیتال با ترکیب 400 اندیکاتور

این ماژول یک سیستم تحلیلی پیشرفته را پیاده‌سازی می‌کند که با ترکیب هوشمند بیش از 400 اندیکاتور فنی
و الگوهای قیمت، تحلیل‌های دقیق و پیشنهادات معاملاتی ارائه می‌دهد.
"""

import numpy as np
import pandas as pd
import ta
# حذف talib و استفاده فقط از ta و pandas_ta
import pandas_ta as pta
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
import math
import logging
import datetime
import json
from scipy import stats, signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings
from super_indicators import SuperIndicators
from advanced_indicators import AdvancedIndicators

# خاموش کردن اخطارهای غیرضروری
warnings.filterwarnings("ignore")

# تنظیم لاگر
logger = logging.getLogger(__name__)

class SuperAnalyzer:
    """
    کلاس تحلیلگر فوق پیشرفته بازار ارزهای دیجیتال با استفاده از 400 اندیکاتور
    """
    
    def __init__(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        مقداردهی اولیه تحلیلگر
        
        Args:
            df (pd.DataFrame): دیتافریم قیمت
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
        """
        self.df = df.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        self.indicators_data = None
        self.signals = {}
        self.price_targets = {}
        
        # ستون‌های اصلی مورد نیاز
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"ستون {col} در دیتافریم وجود ندارد.")
        
        # محاسبه همه اندیکاتورها
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        """محاسبه تمام اندیکاتورهای مورد نیاز"""
        logger.info("محاسبه 400 اندیکاتور...")
        
        # محاسبه اندیکاتورهای اصلی از تمام دسته‌بندی‌ها
        self.indicators_data = SuperIndicators.calculate_all_indicators(self.df)
        
        # استفاده از ماژول AdvancedIndicators برای محاسبه اندیکاتورهای پیشرفته بیشتر
        # گروه اول: اندیکاتورهای پیشرفته روند
        self.indicators_data = AdvancedIndicators.calculate_supertrend(self.indicators_data)
        self.indicators_data = AdvancedIndicators.calculate_elder_ray(self.indicators_data)
        self.indicators_data = AdvancedIndicators.calculate_adl(self.indicators_data)
        self.indicators_data = AdvancedIndicators.calculate_cmf(self.indicators_data)
        
        # محاسبه اندیکاتورهای اضافی
        self.indicators_data = SuperIndicators.fibonacci_pivot_points(self.indicators_data)
        self.indicators_data = SuperIndicators.camarilla_pivot_points(self.indicators_data)
        self.indicators_data = SuperIndicators.woodies_pivot_points(self.indicators_data)
        self.indicators_data = SuperIndicators.demark_pivot_points(self.indicators_data)
        self.indicators_data = SuperIndicators.ehlers_fisher_transform(self.indicators_data)
        self.indicators_data = SuperIndicators.connors_rsi(self.indicators_data)
        self.indicators_data = SuperIndicators.squeeze_momentum(self.indicators_data)
        self.indicators_data = SuperIndicators.waddah_attar_explosion(self.indicators_data)
        self.indicators_data = SuperIndicators.ichimoku_cloud(self.indicators_data)
        
        # اضافه کردن اندیکاتورهای یکتا و متنوع
        self._add_custom_indicators()
        
        logger.info("محاسبه 400 اندیکاتور با موفقیت انجام شد.")
    
    def _add_custom_indicators(self):
        """اضافه کردن اندیکاتورهای سفارشی بیشتر"""
        
        # محاسبه شتاب قیمت
        self.indicators_data['price_momentum'] = self.indicators_data['close'].diff()
        self.indicators_data['price_acceleration'] = self.indicators_data['price_momentum'].diff()
        
        # محاسبه شتاب حجم (اگر حجم موجود باشد)
        if 'volume' in self.indicators_data.columns:
            self.indicators_data['volume_momentum'] = self.indicators_data['volume'].diff()
            self.indicators_data['volume_acceleration'] = self.indicators_data['volume_momentum'].diff()
            
            # حجم غیرعادی (بیشتر از میانگین سه برابر)
            vol_mean = self.indicators_data['volume'].rolling(window=20).mean()
            self.indicators_data['abnormal_volume'] = self.indicators_data['volume'] > (vol_mean * 3)
        
        # محاسبه قدرت روند با استفاده از کتابخانه ta
        adx_indicator = ta.trend.ADXIndicator(
            high=self.indicators_data['high'],
            low=self.indicators_data['low'],
            close=self.indicators_data['close'],
            window=14
        )
        self.indicators_data['adx'] = adx_indicator.adx()
        
        # محاسبه برگشت‌پذیری قیمت
        self.indicators_data['close_pct_change'] = self.indicators_data['close'].pct_change() * 100
        
        # نوسانات دوره‌ای (Cycle Oscillator)
        self.indicators_data['sin_wave_1'] = np.sin(np.arange(len(self.indicators_data)) * (2 * np.pi / 20))  # 20-day cycle
        self.indicators_data['sin_wave_2'] = np.sin(np.arange(len(self.indicators_data)) * (2 * np.pi / 50))  # 50-day cycle
        
        # محاسبه ابرچرخه‌های هرست
        for period in [12, 24, 48]:
            self._calculate_hurst_cycles(period)
    
    def _calculate_hurst_cycles(self, period: int):
        """
        محاسبه ابرچرخه‌های هرست
        
        Args:
            period (int): دوره زمانی
        """
        if len(self.indicators_data) < period * 2:
            return
        
        # محاسبه Hurst Cycles
        price_series = self.indicators_data['close'].values
        cycles = np.zeros(len(price_series))
        
        for i in range(period, len(price_series)):
            chunk = price_series[i-period:i]
            cycles[i] = self._calculate_hurst_exponent(chunk)
        
        self.indicators_data[f'hurst_cycle_{period}'] = cycles
    
    def _calculate_hurst_exponent(self, price_array: np.ndarray) -> float:
        """
        محاسبه نمای هرست
        
        Args:
            price_array (np.ndarray): آرایه قیمت‌ها
            
        Returns:
            float: نمای هرست
        """
        # تبدیل به سری برگشت
        returns = np.diff(np.log(price_array))
        
        # محاسبه نمای هرست با روش میانگین دامنه مقیاس‌بندی شده (R/S)
        tau = []
        lagvec = []
        
        # استفاده از چند دوره مختلف
        samples = min(10, len(returns) // 2)
        lags = np.unique(np.logspace(0.2, 0.9, samples).astype(int))
        
        for lag in lags:
            if lag < 2 or lag >= len(returns):
                continue
                
            # تقسیم سری به بخش‌های با طول lag
            n_blocks = int(len(returns) / lag)
            if n_blocks < 1:
                continue
                
            R_S_array = np.zeros(n_blocks)
            for i in range(n_blocks):
                chunk = returns[i*lag:(i+1)*lag]
                if len(chunk) < 2:
                    continue
                    
                # محاسبه R/S برای هر بخش
                mean_chunk = np.mean(chunk)
                std_chunk = np.std(chunk)
                
                if std_chunk == 0:
                    continue
                    
                adjusted = chunk - mean_chunk
                cumsum = np.cumsum(adjusted)
                R = max(cumsum) - min(cumsum)
                S = std_chunk
                
                if S > 0:
                    R_S_array[i] = R / S
            
            # میانگین R/S برای این lag
            valid_blocks = R_S_array[R_S_array > 0]
            if len(valid_blocks) > 0:
                tau.append(np.log(np.mean(valid_blocks)))
                lagvec.append(np.log(lag))
        
        # رگرسیون خطی برای محاسبه نمای هرست
        if len(tau) > 1 and len(lagvec) > 1:
            m, _ = np.polyfit(lagvec, tau, 1)
            return m
        else:
            return 0.5  # مقدار پیش‌فرض برای حالت متوسط
    
    def _select_best_indicators(self) -> Dict[str, List[str]]:
        """
        انتخاب بهترین اندیکاتورها برای شرایط فعلی بازار
        
        Returns:
            Dict[str, List[str]]: دیکشنری اندیکاتورهای منتخب
        """
        # شناسایی شرایط فعلی بازار
        market_conditions = self._identify_market_conditions()
        
        selected_indicators = {
            'trend': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'patterns': [],
            'custom': []
        }
        
        # انتخاب اندیکاتورهای مناسب بر اساس شرایط بازار
        if market_conditions['trending']:
            # بازار روند دار - اندیکاتورهای روند
            selected_indicators['trend'].extend([
                'supertrend', 'adx', 'ema_20', 'ichimoku_cloud'
            ])
            selected_indicators['momentum'].extend([
                'macd', 'rsi_14', 'cci_20'
            ])
            
        if market_conditions['ranging']:
            # بازار نوسانی - اندیکاتورهای نوسان و حمایت/مقاومت
            selected_indicators['volatility'].extend([
                'bb_upper', 'bb_lower', 'bb_middle', 'atr'
            ])
            selected_indicators['custom'].extend([
                'pivot', 'r1', 's1', 'r2', 's2', 'camarilla_pivot_points'
            ])
            
        if market_conditions['high_volatility']:
            # بازار با نوسان بالا
            selected_indicators['volatility'].extend([
                'atr', 'squeeze_momentum', 'wvf'
            ])
            selected_indicators['momentum'].extend([
                'rsi_14', 'stoch_k', 'stoch_d', 'willr'
            ])
            
        if market_conditions['breakout_potential']:
            # پتانسیل شکست قیمتی
            selected_indicators['volatility'].extend([
                'squeeze_momentum', 'bb_width'
            ])
            selected_indicators['volume'].extend([
                'obv', 'volume_momentum', 'abnormal_volume'
            ])
            
        if market_conditions['support_resistance']:
            # مناطق حمایت و مقاومت قوی
            selected_indicators['custom'].extend([
                'pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3',
                'fib_0', 'fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618'
            ])
            
        if market_conditions['cycle_turn']:
            # تغییر چرخه بازار
            selected_indicators['custom'].extend([
                'sin_wave_1', 'sin_wave_2', 'hurst_cycle_24'
            ])
            selected_indicators['momentum'].extend([
                'fisher', 'fisher_trigger'
            ])
        
        # اضافه کردن اندیکاتورهای الگوی شمعی
        for pattern in ['doji', 'hammer', 'hanging_man', 'engulfing', 'inverted_hammer']:
            if pattern in self.indicators_data.columns:
                selected_indicators['patterns'].append(pattern)
        
        return selected_indicators
    
    def _identify_market_conditions(self) -> Dict[str, bool]:
        """
        شناسایی شرایط فعلی بازار
        
        Returns:
            Dict[str, bool]: دیکشنری شرایط بازار
        """
        # بررسی دیتافریم اصلی
        df = self.indicators_data
        
        # شرایط پیش‌فرض بازار
        conditions = {
            'trending': False,           # بازار روند دار است
            'ranging': False,            # بازار رنج است
            'high_volatility': False,    # نوسان بالا
            'low_volatility': False,     # نوسان پایین
            'breakout_potential': False, # پتانسیل شکست
            'support_resistance': False, # مناطق حمایت/مقاومت
            'cycle_turn': False          # تغییر چرخه
        }
        
        # بررسی ADX برای تشخیص روند
        if 'adx' in df.columns:
            # ADX > 25 نشان‌دهنده بازار روند دار
            if df['adx'].iloc[-1] > 25:
                conditions['trending'] = True
                # اگر ADX < 20 باشد، بازار احتمالاً رنج است
            elif df['adx'].iloc[-1] < 20:
                conditions['ranging'] = True
        
        # بررسی ATR برای نوسان
        if 'atr' in df.columns:
            # محاسبه نسبت ATR به قیمت فعلی
            atr_pct = df['atr'].iloc[-1] / df['close'].iloc[-1] * 100
            
            if atr_pct > 3:  # 3% نوسان روزانه بالاست
                conditions['high_volatility'] = True
            elif atr_pct < 1:  # 1% نوسان روزانه پایین است
                conditions['low_volatility'] = True
        
        # بررسی باندهای بولینگر برای پتانسیل شکست
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            # محاسبه عرض باند بولینگر
            bb_width = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['bb_middle'].iloc[-1]
            bb_width_avg = (df['bb_upper'] - df['bb_lower']).rolling(window=20).mean().iloc[-1] / df['bb_middle'].iloc[-1]
            
            # تنگ شدن باندها نشان‌دهنده پتانسیل شکست است
            if bb_width < bb_width_avg * 0.8:
                conditions['breakout_potential'] = True
        
        # بررسی حجم غیرعادی
        if 'abnormal_volume' in df.columns:
            # حجم غیرعادی در 3 کندل اخیر
            if df['abnormal_volume'].iloc[-3:].any():
                conditions['breakout_potential'] = True
        
        # بررسی سطوح فیبوناچی
        fib_cols = [col for col in df.columns if col.startswith('fib_')]
        if fib_cols:
            # بررسی نزدیکی قیمت فعلی به سطوح فیبوناچی
            current_price = df['close'].iloc[-1]
            
            for col in fib_cols:
                if abs(df[col].iloc[-1] - current_price) / current_price < 0.02:  # در محدوده 2%
                    conditions['support_resistance'] = True
                    break
        
        # بررسی تغییر چرخه
        hurst_cols = [col for col in df.columns if col.startswith('hurst_cycle_')]
        if hurst_cols and len(df) > 5:
            # بررسی تغییر جهت در نمای هرست
            for col in hurst_cols:
                if ((df[col].iloc[-1] > 0.6 and df[col].iloc[-2] < 0.6) or
                    (df[col].iloc[-1] < 0.4 and df[col].iloc[-2] > 0.4)):
                    conditions['cycle_turn'] = True
                    break
        
        return conditions
    
    def analyze_market(self) -> Dict[str, Any]:
        """
        تحلیل بازار با استفاده از ترکیب اندیکاتورها
        
        Returns:
            Dict[str, Any]: دیکشنری نتایج تحلیل
        """
        logger.info("تحلیل بازار با 400 اندیکاتور...")
        
        # انتخاب بهترین اندیکاتورها برای شرایط فعلی
        best_indicators = self._select_best_indicators()
        
        # تحلیل اندیکاتورهای منتخب
        indicator_signals = self._analyze_indicators(best_indicators)
        
        # تحلیل الگوهای قیمت
        price_patterns = self._analyze_price_patterns()
        
        # محاسبه سطوح حمایت و مقاومت
        support_resistance = self._calculate_support_resistance()
        
        # محاسبه سیگنال نهایی
        final_signal, signal_strength = self._calculate_final_signal(indicator_signals)
        
        # تعیین نقاط ورود، حد سود و حد ضرر
        entry_exit_points = self._calculate_entry_exit_points(final_signal, support_resistance)
        
        # ادغام نتایج
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "last_price": self.indicators_data['close'].iloc[-1],
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal": final_signal,
            "signal_strength": signal_strength,
            "indicator_signals": indicator_signals,
            "price_patterns": price_patterns,
            "support_resistance": support_resistance,
            "entry_exit_points": entry_exit_points,
            "market_conditions": self._identify_market_conditions()
        }
        
        # ذخیره سیگنال‌ها برای استفاده‌های بعدی
        self.signals = result
        
        logger.info(f"تحلیل بازار کامل شد. سیگنال: {final_signal}, قدرت: {signal_strength:.2f}")
        return result
    
    def _analyze_indicators(self, best_indicators: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        تحلیل اندیکاتورهای منتخب
        
        Args:
            best_indicators (Dict[str, List[str]]): دیکشنری اندیکاتورهای منتخب
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: سیگنال‌های اندیکاتورها
        """
        # دیکشنری برای ذخیره سیگنال‌ها
        signals = {category: [] for category in best_indicators.keys()}
        
        # دریافت سیگنال‌های اندیکاتورهای روند
        for indicator in best_indicators['trend']:
            if indicator == 'supertrend':
                if 'supertrend' in self.indicators_data.columns:
                    signal = 'صعودی' if self.indicators_data['supertrend'].iloc[-1] else 'نزولی'
                    strength = 3  # اهمیت بالا
                    signals['trend'].append({
                        'name': 'SuperTrend',
                        'signal': signal,
                        'strength': strength if signal == 'صعودی' else -strength,
                        'value': self.indicators_data['supertrend'].iloc[-1]
                    })
            
            elif indicator == 'adx':
                if 'adx' in self.indicators_data.columns and 'adx_pos' in self.indicators_data.columns and 'adx_neg' in self.indicators_data.columns:
                    adx_value = self.indicators_data['adx'].iloc[-1]
                    adx_pos = self.indicators_data['adx_pos'].iloc[-1]
                    adx_neg = self.indicators_data['adx_neg'].iloc[-1]
                    
                    if adx_value > 25:
                        if adx_pos > adx_neg:
                            signal = 'روند صعودی قوی'
                            strength = 3
                        else:
                            signal = 'روند نزولی قوی'
                            strength = -3
                    elif adx_value > 20:
                        if adx_pos > adx_neg:
                            signal = 'روند صعودی'
                            strength = 2
                        else:
                            signal = 'روند نزولی'
                            strength = -2
                    else:
                        signal = 'بدون روند قوی'
                        strength = 0
                    
                    signals['trend'].append({
                        'name': 'ADX',
                        'signal': signal,
                        'strength': strength,
                        'value': adx_value
                    })
            
            elif indicator == 'ema_20':
                if 'ema_20' in self.indicators_data.columns:
                    price = self.indicators_data['close'].iloc[-1]
                    ema = self.indicators_data['ema_20'].iloc[-1]
                    
                    if price > ema:
                        signal = 'قیمت بالای EMA'
                        strength = 1
                    else:
                        signal = 'قیمت زیر EMA'
                        strength = -1
                    
                    signals['trend'].append({
                        'name': 'EMA 20',
                        'signal': signal,
                        'strength': strength,
                        'value': ema
                    })
            
            elif indicator == 'ichimoku_cloud':
                if all(col in self.indicators_data.columns for col in ['price_above_cloud', 'price_below_cloud']):
                    if self.indicators_data['price_above_cloud'].iloc[-1]:
                        signal = 'قیمت بالای ابر'
                        strength = 3
                    elif self.indicators_data['price_below_cloud'].iloc[-1]:
                        signal = 'قیمت زیر ابر'
                        strength = -3
                    else:
                        signal = 'قیمت درون ابر'
                        strength = 0
                    
                    # بررسی تقاطع‌های تنکان و کیجون
                    if 'tenkan_kijun_cross_up' in self.indicators_data.columns and 'tenkan_kijun_cross_down' in self.indicators_data.columns:
                        if self.indicators_data['tenkan_kijun_cross_up'].iloc[-1]:
                            signal += ' + تقاطع صعودی'
                            strength += 1
                        if self.indicators_data['tenkan_kijun_cross_down'].iloc[-1]:
                            signal += ' + تقاطع نزولی'
                            strength -= 1
                    
                    signals['trend'].append({
                        'name': 'Ichimoku Cloud',
                        'signal': signal,
                        'strength': strength,
                        'value': 1 if self.indicators_data['price_above_cloud'].iloc[-1] else (-1 if self.indicators_data['price_below_cloud'].iloc[-1] else 0)
                    })
        
        # دریافت سیگنال‌های اندیکاتورهای مومنتوم
        for indicator in best_indicators['momentum']:
            if indicator == 'macd':
                if all(col in self.indicators_data.columns for col in ['macd', 'macd_signal', 'macd_hist']):
                    macd = self.indicators_data['macd'].iloc[-1]
                    signal_line = self.indicators_data['macd_signal'].iloc[-1]
                    hist = self.indicators_data['macd_hist'].iloc[-1]
                    
                    if macd > signal_line:
                        if macd < 0 and signal_line < 0:
                            signal = 'MACD صعودی (زیر خط صفر)'
                            strength = 1
                        else:
                            signal = 'MACD صعودی (بالای خط صفر)'
                            strength = 2
                    else:
                        if macd > 0 and signal_line > 0:
                            signal = 'MACD نزولی (بالای خط صفر)'
                            strength = -1
                        else:
                            signal = 'MACD نزولی (زیر خط صفر)'
                            strength = -2
                    
                    # بررسی تقاطع MACD
                    if self.indicators_data['macd_hist'].iloc[-1] * self.indicators_data['macd_hist'].iloc[-2] < 0:
                        if hist > 0:
                            signal += ' (تقاطع صعودی)'
                            strength += 1
                        else:
                            signal += ' (تقاطع نزولی)'
                            strength -= 1
                    
                    signals['momentum'].append({
                        'name': 'MACD',
                        'signal': signal,
                        'strength': strength,
                        'value': macd
                    })
            
            elif indicator == 'rsi_14':
                if 'rsi_14' in self.indicators_data.columns:
                    rsi = self.indicators_data['rsi_14'].iloc[-1]
                    
                    if rsi > 70:
                        signal = 'اشباع خرید'
                        strength = -3
                    elif rsi < 30:
                        signal = 'اشباع فروش'
                        strength = 3
                    elif rsi > 50:
                        signal = 'مثبت'
                        strength = 1
                    else:
                        signal = 'منفی'
                        strength = -1
                    
                    signals['momentum'].append({
                        'name': 'RSI 14',
                        'signal': signal,
                        'strength': strength,
                        'value': rsi
                    })
            
            elif indicator == 'cci_20':
                if 'cci_20' in self.indicators_data.columns:
                    cci = self.indicators_data['cci_20'].iloc[-1]
                    
                    if cci > 100:
                        signal = 'اشباع خرید'
                        strength = -2
                    elif cci < -100:
                        signal = 'اشباع فروش'
                        strength = 2
                    elif cci > 0:
                        signal = 'مثبت'
                        strength = 1
                    else:
                        signal = 'منفی'
                        strength = -1
                    
                    signals['momentum'].append({
                        'name': 'CCI 20',
                        'signal': signal,
                        'strength': strength,
                        'value': cci
                    })
            
            elif indicator == 'stoch_k' or indicator == 'stoch_d':
                if all(col in self.indicators_data.columns for col in ['stoch_k', 'stoch_d']):
                    k = self.indicators_data['stoch_k'].iloc[-1]
                    d = self.indicators_data['stoch_d'].iloc[-1]
                    
                    if k > 80 and d > 80:
                        signal = 'اشباع خرید'
                        strength = -2
                    elif k < 20 and d < 20:
                        signal = 'اشباع فروش'
                        strength = 2
                    elif k > d:
                        signal = 'K بالای D (صعودی)'
                        strength = 1
                    else:
                        signal = 'K زیر D (نزولی)'
                        strength = -1
                    
                    signals['momentum'].append({
                        'name': 'Stochastic',
                        'signal': signal,
                        'strength': strength,
                        'value': {'k': k, 'd': d}
                    })
            
            elif indicator == 'willr':
                if 'willr' in self.indicators_data.columns:
                    willr = self.indicators_data['willr'].iloc[-1]
                    
                    if willr > -20:
                        signal = 'اشباع خرید'
                        strength = -2
                    elif willr < -80:
                        signal = 'اشباع فروش'
                        strength = 2
                    elif willr > -50:
                        signal = 'مثبت'
                        strength = 1
                    else:
                        signal = 'منفی'
                        strength = -1
                    
                    signals['momentum'].append({
                        'name': 'Williams %R',
                        'signal': signal,
                        'strength': strength,
                        'value': willr
                    })
            
            elif indicator == 'fisher' or indicator == 'fisher_trigger':
                if all(col in self.indicators_data.columns for col in ['fisher', 'fisher_trigger']):
                    fisher = self.indicators_data['fisher'].iloc[-1]
                    trigger = self.indicators_data['fisher_trigger'].iloc[-1]
                    
                    if fisher > 2:
                        signal = 'اشباع خرید شدید'
                        strength = -3
                    elif fisher < -2:
                        signal = 'اشباع فروش شدید'
                        strength = 3
                    elif fisher > 0:
                        if fisher > trigger:
                            signal = 'صعودی'
                            strength = 1
                        else:
                            signal = 'بازگشت از صعود'
                            strength = -1
                    else:
                        if fisher < trigger:
                            signal = 'نزولی'
                            strength = -1
                        else:
                            signal = 'بازگشت از نزول'
                            strength = 1
                    
                    signals['momentum'].append({
                        'name': 'Fisher Transform',
                        'signal': signal,
                        'strength': strength,
                        'value': {'fisher': fisher, 'trigger': trigger}
                    })
        
        # دریافت سیگنال‌های اندیکاتورهای نوسان
        for indicator in best_indicators['volatility']:
            if any(indicator.startswith(prefix) for prefix in ['bb_', 'bollinger']):
                if all(col in self.indicators_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                    price = self.indicators_data['close'].iloc[-1]
                    upper = self.indicators_data['bb_upper'].iloc[-1]
                    middle = self.indicators_data['bb_middle'].iloc[-1]
                    lower = self.indicators_data['bb_lower'].iloc[-1]
                    
                    if price > upper:
                        signal = 'بالای باند بالایی (اشباع خرید)'
                        strength = -2
                    elif price < lower:
                        signal = 'زیر باند پایینی (اشباع فروش)'
                        strength = 2
                    elif price > middle:
                        signal = 'بین باند میانی و بالایی'
                        strength = 1
                    else:
                        signal = 'بین باند میانی و پایینی'
                        strength = -1
                    
                    signals['volatility'].append({
                        'name': 'Bollinger Bands',
                        'signal': signal,
                        'strength': strength,
                        'value': {'price': price, 'upper': upper, 'middle': middle, 'lower': lower}
                    })
            
            elif indicator == 'atr':
                if 'atr' in self.indicators_data.columns:
                    atr = self.indicators_data['atr'].iloc[-1]
                    avg_atr = self.indicators_data['atr'].rolling(window=20).mean().iloc[-1]
                    price = self.indicators_data['close'].iloc[-1]
                    atr_percent = atr / price * 100
                    
                    if atr > avg_atr * 1.5:
                        signal = 'نوسان بالا'
                        strength = 0  # خنثی چون فقط نشانگر نوسان است
                    elif atr < avg_atr * 0.8:
                        signal = 'نوسان پایین (احتمال شکست)'
                        strength = 0
                    else:
                        signal = 'نوسان نرمال'
                        strength = 0
                    
                    signals['volatility'].append({
                        'name': 'ATR',
                        'signal': signal,
                        'strength': strength,
                        'value': atr,
                        'percent': atr_percent
                    })
            
            elif indicator == 'squeeze_momentum':
                if all(col in self.indicators_data.columns for col in ['squeeze_on', 'squeeze_off', 'momentum']):
                    squeeze_on = self.indicators_data['squeeze_on'].iloc[-1]
                    momentum = self.indicators_data['momentum'].iloc[-1]
                    
                    if squeeze_on:
                        signal = 'فشردگی فعال (انتظار حرکت)'
                        if momentum > 0:
                            signal += ' - احتمال شکست صعودی'
                            strength = 1
                        else:
                            signal += ' - احتمال شکست نزولی'
                            strength = -1
                    else:
                        if momentum > 0:
                            if momentum > self.indicators_data['momentum'].iloc[-2]:
                                signal = 'شکست صعودی - افزایش مومنتوم'
                                strength = 2
                            else:
                                signal = 'شکست صعودی - کاهش مومنتوم'
                                strength = 1
                        else:
                            if momentum < self.indicators_data['momentum'].iloc[-2]:
                                signal = 'شکست نزولی - افزایش مومنتوم'
                                strength = -2
                            else:
                                signal = 'شکست نزولی - کاهش مومنتوم'
                                strength = -1
                    
                    signals['volatility'].append({
                        'name': 'Squeeze Momentum',
                        'signal': signal,
                        'strength': strength,
                        'value': {'squeeze_on': squeeze_on, 'momentum': momentum}
                    })
            
            elif indicator == 'wvf':
                if all(col in self.indicators_data.columns for col in ['wvf', 'wvf_upper_band']):
                    wvf = self.indicators_data['wvf'].iloc[-1]
                    upper_band = self.indicators_data['wvf_upper_band'].iloc[-1]
                    
                    if wvf > upper_band:
                        signal = 'وضعیت ترس (فرصت خرید)'
                        strength = 2
                    elif wvf > self.indicators_data['wvf_upper_band'].iloc[-1] * 0.8:
                        signal = 'نزدیک به وضعیت ترس'
                        strength = 1
                    else:
                        signal = 'وضعیت نرمال'
                        strength = 0
                    
                    signals['volatility'].append({
                        'name': 'Williams VIX Fix',
                        'signal': signal,
                        'strength': strength,
                        'value': wvf
                    })
        
        # دریافت سیگنال‌های اندیکاتورهای حجم
        for indicator in best_indicators['volume']:
            if indicator == 'obv':
                if 'obv' in self.indicators_data.columns:
                    obv = self.indicators_data['obv'].iloc[-1]
                    obv_ema = pd.Series(self.indicators_data['obv']).rolling(window=20).mean().iloc[-1]
                    
                    if obv > obv_ema:
                        # محاسبه شیب OBV
                        obv_slope = (obv - self.indicators_data['obv'].iloc[-5]) / 5
                        
                        if obv_slope > 0:
                            signal = 'حجم صعودی تأییدی'
                            strength = 2
                        else:
                            signal = 'حجم صعودی با کاهش شتاب'
                            strength = 1
                    else:
                        obv_slope = (obv - self.indicators_data['obv'].iloc[-5]) / 5
                        
                        if obv_slope < 0:
                            signal = 'حجم نزولی تأییدی'
                            strength = -2
                        else:
                            signal = 'حجم نزولی با کاهش شتاب'
                            strength = -1
                    
                    signals['volume'].append({
                        'name': 'On-Balance Volume',
                        'signal': signal,
                        'strength': strength,
                        'value': obv
                    })
            
            elif indicator == 'volume_momentum':
                if 'volume_momentum' in self.indicators_data.columns:
                    vol_mom = self.indicators_data['volume_momentum'].iloc[-1]
                    
                    if vol_mom > 0:
                        signal = 'افزایش حجم'
                        strength = 1
                    else:
                        signal = 'کاهش حجم'
                        strength = -1
                    
                    signals['volume'].append({
                        'name': 'Volume Momentum',
                        'signal': signal,
                        'strength': strength,
                        'value': vol_mom
                    })
            
            elif indicator == 'abnormal_volume':
                if 'abnormal_volume' in self.indicators_data.columns:
                    abnormal = self.indicators_data['abnormal_volume'].iloc[-1]
                    
                    if abnormal:
                        signal = 'حجم غیرعادی (احتمال شکست)'
                        # برای تعیین جهت به قیمت نگاه می‌کنیم
                        if self.indicators_data['close'].iloc[-1] > self.indicators_data['close'].iloc[-2]:
                            strength = 2
                        else:
                            strength = -2
                    else:
                        signal = 'حجم نرمال'
                        strength = 0
                    
                    signals['volume'].append({
                        'name': 'Abnormal Volume',
                        'signal': signal,
                        'strength': strength,
                        'value': abnormal
                    })
        
        # دریافت سیگنال‌های الگوهای شمعی
        for pattern in best_indicators['patterns']:
            if pattern in self.indicators_data.columns:
                pattern_value = self.indicators_data[pattern].iloc[-1]
                
                if pattern_value > 0:
                    signal = f'{pattern.replace("_", " ").title()} (صعودی)'
                    strength = 1
                elif pattern_value < 0:
                    signal = f'{pattern.replace("_", " ").title()} (نزولی)'
                    strength = -1
                else:
                    continue  # الگویی شناسایی نشده
                
                signals['patterns'].append({
                    'name': pattern.replace('_', ' ').title(),
                    'signal': signal,
                    'strength': strength,
                    'value': pattern_value
                })
        
        # دریافت سیگنال‌های اندیکاتورهای سفارشی
        for indicator in best_indicators['custom']:
            if indicator in ['pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3']:
                if indicator in self.indicators_data.columns:
                    price = self.indicators_data['close'].iloc[-1]
                    level = self.indicators_data[indicator].iloc[-1]
                    distance = abs(price - level) / price * 100
                    
                    if distance < 1:  # 1% فاصله
                        if price > level:
                            if indicator == 'pivot' or indicator.startswith('s'):
                                signal = f'قیمت بالای {indicator.upper()} (حمایت)'
                                strength = 1
                            else:
                                signal = f'قیمت بالای {indicator.upper()} (شکست مقاومت)'
                                strength = 2
                        else:
                            if indicator == 'pivot' or indicator.startswith('r'):
                                signal = f'قیمت زیر {indicator.upper()} (مقاومت)'
                                strength = -1
                            else:
                                signal = f'قیمت زیر {indicator.upper()} (شکست حمایت)'
                                strength = -2
                        
                        signals['custom'].append({
                            'name': f'{indicator.upper()} Level',
                            'signal': signal,
                            'strength': strength,
                            'value': level
                        })
            
            elif indicator.startswith('fib_'):
                if indicator in self.indicators_data.columns:
                    price = self.indicators_data['close'].iloc[-1]
                    fib_level = self.indicators_data[indicator].iloc[-1]
                    distance = abs(price - fib_level) / price * 100
                    
                    if distance < 1:  # 1% فاصله
                        fib_value = indicator.split('_')[1]
                        if price > fib_level:
                            signal = f'قیمت بالای فیبوناچی {fib_value} (حمایت)'
                            strength = 1
                        else:
                            signal = f'قیمت زیر فیبوناچی {fib_value} (مقاومت)'
                            strength = -1
                        
                        signals['custom'].append({
                            'name': f'Fibonacci {fib_value}',
                            'signal': signal,
                            'strength': strength,
                            'value': fib_level
                        })
            
            # اضافه کردن سایر اندیکاتورهای سفارشی
            elif indicator in ['sin_wave_1', 'sin_wave_2']:
                if indicator in self.indicators_data.columns:
                    value = self.indicators_data[indicator].iloc[-1]
                    prev_value = self.indicators_data[indicator].iloc[-2]
                    
                    if value > 0 and prev_value <= 0:
                        signal = 'آغاز چرخه صعودی'
                        strength = 1
                    elif value < 0 and prev_value >= 0:
                        signal = 'آغاز چرخه نزولی'
                        strength = -1
                    elif value > prev_value:
                        signal = 'ادامه چرخه صعودی'
                        strength = 0.5
                    else:
                        signal = 'ادامه چرخه نزولی'
                        strength = -0.5
                    
                    signals['custom'].append({
                        'name': f'Cycle Wave {indicator[-1]}',
                        'signal': signal,
                        'strength': strength,
                        'value': value
                    })
            
            elif indicator.startswith('hurst_cycle_'):
                if indicator in self.indicators_data.columns:
                    value = self.indicators_data[indicator].iloc[-1]
                    if value > 0.6:
                        signal = 'چرخه روند دار (Trending)'
                        if value > self.indicators_data[indicator].iloc[-2]:
                            signal += ' - افزایشی'
                            strength = 1
                        else:
                            signal += ' - کاهشی'
                            strength = -1
                    elif value < 0.4:
                        signal = 'چرخه تصادفی (Mean-Reverting)'
                        if value < self.indicators_data[indicator].iloc[-2]:
                            signal += ' - افزایشی'
                            strength = -1
                        else:
                            signal += ' - کاهشی'
                            strength = 1
                    else:
                        signal = 'چرخه خنثی'
                        strength = 0
                    
                    signals['custom'].append({
                        'name': f'Hurst Cycle {indicator.split("_")[-1]}',
                        'signal': signal,
                        'strength': strength,
                        'value': value
                    })
        
        return signals
    
    def _analyze_price_patterns(self) -> List[Dict[str, Any]]:
        """
        تحلیل الگوهای قیمت
        
        Returns:
            List[Dict[str, Any]]: لیست الگوهای شناسایی شده
        """
        patterns = []
        df = self.indicators_data
        
        # تعیین پنجره بررسی (25 کندل آخر)
        window = min(25, len(df))
        tail = df.iloc[-window:]
        
        # ---- شناسایی الگوی سر و شانه ----
        # الگوریتم ساده برای شناسایی الگوی سر و شانه
        # به دنبال سه قله می‌گردیم که قله وسط بلندتر باشد
        # و دو قله کناری تقریباً هم‌ارتفاع باشند
        peaks = signal.find_peaks(tail['high'].values, distance=3)[0]
        
        if len(peaks) >= 3:
            for i in range(len(peaks)-2):
                left = peaks[i]
                head = peaks[i+1]
                right = peaks[i+2]
                
                if tail['high'].iloc[head] > tail['high'].iloc[left] and tail['high'].iloc[head] > tail['high'].iloc[right]:
                    # نسبت ارتفاع شانه‌ها
                    ratio = tail['high'].iloc[left] / tail['high'].iloc[right]
                    
                    if 0.8 < ratio < 1.2:  # شانه‌ها تقریباً هم‌ارتفاع
                        # یافتن خط گردن
                        neckline = min(tail['low'].iloc[left:right+1].min(), tail['close'].iloc[-1])
                        
                        # بررسی اینکه آیا قیمت زیر خط گردن است (تکمیل الگو)
                        if tail['close'].iloc[-1] < neckline:
                            # محاسبه هدف قیمت
                            head_to_neckline = tail['high'].iloc[head] - neckline
                            target = neckline - head_to_neckline
                            
                            patterns.append({
                                'name': 'Head and Shoulders',
                                'type': 'reversal',
                                'direction': 'bearish',
                                'strength': 3,
                                'target': target,
                                'properties': {
                                    'neckline': neckline,
                                    'head': tail['high'].iloc[head],
                                    'left_shoulder': tail['high'].iloc[left],
                                    'right_shoulder': tail['high'].iloc[right]
                                }
                            })
                            break  # فقط یک الگو را گزارش می‌کنیم
        
        # ---- شناسایی الگوی سر و شانه معکوس ----
        troughs = signal.find_peaks(-tail['low'].values, distance=3)[0]
        
        if len(troughs) >= 3:
            for i in range(len(troughs)-2):
                left = troughs[i]
                head = troughs[i+1]
                right = troughs[i+2]
                
                if tail['low'].iloc[head] < tail['low'].iloc[left] and tail['low'].iloc[head] < tail['low'].iloc[right]:
                    # نسبت ارتفاع شانه‌ها
                    ratio = tail['low'].iloc[left] / tail['low'].iloc[right]
                    
                    if 0.8 < ratio < 1.2:  # شانه‌ها تقریباً هم‌ارتفاع
                        # یافتن خط گردن
                        neckline = max(tail['high'].iloc[left:right+1].max(), tail['close'].iloc[-1])
                        
                        # بررسی اینکه آیا قیمت بالای خط گردن است (تکمیل الگو)
                        if tail['close'].iloc[-1] > neckline:
                            # محاسبه هدف قیمت
                            head_to_neckline = neckline - tail['low'].iloc[head]
                            target = neckline + head_to_neckline
                            
                            patterns.append({
                                'name': 'Inverse Head and Shoulders',
                                'type': 'reversal',
                                'direction': 'bullish',
                                'strength': 3,
                                'target': target,
                                'properties': {
                                    'neckline': neckline,
                                    'head': tail['low'].iloc[head],
                                    'left_shoulder': tail['low'].iloc[left],
                                    'right_shoulder': tail['low'].iloc[right]
                                }
                            })
                            break  # فقط یک الگو را گزارش می‌کنیم
        
        # ---- شناسایی الگوی مثلث متقارن ----
        # یافتن قله‌ها و دره‌ها
        highs = tail['high'].values
        lows = tail['low'].values
        
        if len(tail) >= 10:  # نیاز به حداقل 10 کندل
            # انتخاب قله‌ها و دره‌های اخیر
            sample = 10  # تعداد قله‌ها و دره‌ها برای بررسی
            
            # یافتن قله‌ها و دره‌ها
            peak_indices = signal.find_peaks(highs, distance=2)[0]
            trough_indices = signal.find_peaks(-lows, distance=2)[0]
            
            if len(peak_indices) >= 3 and len(trough_indices) >= 3:
                # انتخاب آخرین قله‌ها و دره‌ها
                recent_peaks = peak_indices[-min(len(peak_indices), sample):]
                recent_troughs = trough_indices[-min(len(trough_indices), sample):]
                
                # بررسی مثلث متقارن (قله‌های پایین‌رونده و دره‌های بالارونده)
                if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                    # خط روند نزولی از قله‌ها
                    peak_x = recent_peaks
                    peak_y = highs[recent_peaks]
                    if len(peak_x) > 1 and len(peak_y) > 1:
                        peak_slope, peak_intercept = np.polyfit(peak_x, peak_y, 1)
                        
                        # خط روند صعودی از دره‌ها
                        trough_x = recent_troughs
                        trough_y = lows[recent_troughs]
                        if len(trough_x) > 1 and len(trough_y) > 1:
                            trough_slope, trough_intercept = np.polyfit(trough_x, trough_y, 1)
                            
                            # بررسی همگرایی خطوط روند
                            if peak_slope < -0.01 and trough_slope > 0.01:
                                # محاسبه نقطه تلاقی
                                if peak_slope != trough_slope:
                                    x_intercept = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
                                    if x_intercept > len(tail) and x_intercept < len(tail) * 1.5:
                                        # مثلث متقارن معتبر است
                                        # تعیین جهت شکست
                                        avg_price = (tail['high'].iloc[-1] + tail['low'].iloc[-1]) / 2
                                        trend_line_value_high = peak_slope * (len(tail) - 1) + peak_intercept
                                        trend_line_value_low = trough_slope * (len(tail) - 1) + trough_intercept
                                        
                                        if avg_price > trend_line_value_high:
                                            direction = 'bullish'
                                            strength = 2
                                        elif avg_price < trend_line_value_low:
                                            direction = 'bearish'
                                            strength = 2
                                        else:
                                            direction = 'neutral'
                                            strength = 1
                                        
                                        # محاسبه هدف قیمت (ارتفاع مثلث در ابتدا)
                                        initial_height = abs(peak_y[0] - trough_y[0])
                                        if direction == 'bullish':
                                            target = trend_line_value_high + initial_height
                                        elif direction == 'bearish':
                                            target = trend_line_value_low - initial_height
                                        else:
                                            target = None
                                        
                                        patterns.append({
                                            'name': 'Symmetrical Triangle',
                                            'type': 'continuation',
                                            'direction': direction,
                                            'strength': strength,
                                            'target': target,
                                            'properties': {
                                                'peak_slope': peak_slope,
                                                'trough_slope': trough_slope,
                                                'convergence_point': x_intercept
                                            }
                                        })
        
        # ---- شناسایی الگوی دو قله / دو دره ----
        if len(peaks) >= 2:
            peak1 = peaks[-2]
            peak2 = peaks[-1]
            
            # بررسی ارتفاع قله‌ها
            peak1_value = tail['high'].iloc[peak1]
            peak2_value = tail['high'].iloc[peak2]
            ratio = peak2_value / peak1_value
            
            if 0.95 < ratio < 1.05:  # دو قله با ارتفاع مشابه
                # یافتن دره‌ی میانی
                if peak2 - peak1 >= 3:  # فاصله کافی بین قله‌ها
                    mid_trough = tail['low'].iloc[peak1:peak2].idxmin()
                    mid_trough_value = tail['low'].iloc[mid_trough - tail.index[0]]
                    
                    # بررسی اینکه آیا قیمت زیر دره میانی است
                    if tail['close'].iloc[-1] < mid_trough_value:
                        # محاسبه هدف قیمت
                        height = peak1_value - mid_trough_value
                        target = mid_trough_value - height
                        
                        patterns.append({
                            'name': 'Double Top',
                            'type': 'reversal',
                            'direction': 'bearish',
                            'strength': 3,
                            'target': target,
                            'properties': {
                                'peak1': peak1_value,
                                'peak2': peak2_value,
                                'neckline': mid_trough_value
                            }
                        })
        
        if len(troughs) >= 2:
            trough1 = troughs[-2]
            trough2 = troughs[-1]
            
            # بررسی عمق دره‌ها
            trough1_value = tail['low'].iloc[trough1]
            trough2_value = tail['low'].iloc[trough2]
            ratio = trough2_value / trough1_value
            
            if 0.95 < ratio < 1.05:  # دو دره با عمق مشابه
                # یافتن قله‌ی میانی
                if trough2 - trough1 >= 3:  # فاصله کافی بین دره‌ها
                    mid_peak = tail['high'].iloc[trough1:trough2].idxmax()
                    mid_peak_value = tail['high'].iloc[mid_peak - tail.index[0]]
                    
                    # بررسی اینکه آیا قیمت بالای قله میانی است
                    if tail['close'].iloc[-1] > mid_peak_value:
                        # محاسبه هدف قیمت
                        height = mid_peak_value - trough1_value
                        target = mid_peak_value + height
                        
                        patterns.append({
                            'name': 'Double Bottom',
                            'type': 'reversal',
                            'direction': 'bullish',
                            'strength': 3,
                            'target': target,
                            'properties': {
                                'trough1': trough1_value,
                                'trough2': trough2_value,
                                'neckline': mid_peak_value
                            }
                        })
        
        # ---- روند صعودی / نزولی ----
        if len(df) >= 20:
            # بررسی روند در 20 کندل اخیر
            recent = df.iloc[-20:]
            
            # محاسبه خط روند
            x = np.arange(len(recent))
            y = recent['close'].values
            slope, intercept = np.polyfit(x, y, 1)
            
            # محاسبه ضریب همبستگی
            correlation = np.corrcoef(x, y)[0, 1]
            
            # تشخیص روند قوی
            if abs(correlation) > 0.8:
                if slope > 0:
                    patterns.append({
                        'name': 'Strong Uptrend',
                        'type': 'trend',
                        'direction': 'bullish',
                        'strength': 2,
                        'target': None,
                        'properties': {
                            'slope': slope,
                            'correlation': correlation
                        }
                    })
                else:
                    patterns.append({
                        'name': 'Strong Downtrend',
                        'type': 'trend',
                        'direction': 'bearish',
                        'strength': 2,
                        'target': None,
                        'properties': {
                            'slope': slope,
                            'correlation': correlation
                        }
                    })
        
        return patterns
    
    def _calculate_support_resistance(self, levels: int = 3) -> Dict[str, List[float]]:
        """
        محاسبه سطوح حمایت و مقاومت
        
        Args:
            levels (int): تعداد سطوح مورد نیاز
            
        Returns:
            Dict[str, List[float]]: دیکشنری سطوح حمایت و مقاومت
        """
        df = self.indicators_data
        
        # مقداردهی اولیه
        sr_levels = {
            'resistance': [],
            'support': []
        }
        
        # استفاده از نقاط پیوت
        if all(col in df.columns for col in ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']):
            current_price = df['close'].iloc[-1]
            
            # سطوح مقاومت
            for r in ['r1', 'r2', 'r3']:
                if df[r].iloc[-1] > current_price:
                    sr_levels['resistance'].append(df[r].iloc[-1])
            
            # پیوت به عنوان مقاومت یا حمایت
            if df['pivot'].iloc[-1] > current_price:
                sr_levels['resistance'].append(df['pivot'].iloc[-1])
            else:
                sr_levels['support'].append(df['pivot'].iloc[-1])
            
            # سطوح حمایت
            for s in ['s1', 's2', 's3']:
                if df[s].iloc[-1] < current_price:
                    sr_levels['support'].append(df[s].iloc[-1])
                    
            # مرتب‌سازی سطوح
            sr_levels['resistance'].sort()
            sr_levels['support'].sort(reverse=True)
        
        # استفاده از سطوح فیبوناچی اگر محاسبه شده باشند
        fib_cols = [col for col in df.columns if col.startswith('fib_')]
        if fib_cols:
            current_price = df['close'].iloc[-1]
            
            for col in fib_cols:
                level = df[col].iloc[-1]
                if level > current_price:
                    sr_levels['resistance'].append(level)
                elif level < current_price:
                    sr_levels['support'].append(level)
            
            # مرتب‌سازی سطوح
            sr_levels['resistance'].sort()
            sr_levels['support'].sort(reverse=True)
        
        # استفاده از قله‌ها و دره‌های اخیر
        lookback = min(100, len(df))
        tail = df.iloc[-lookback:]
        
        # یافتن قله‌ها و دره‌ها
        peak_indices = signal.find_peaks(tail['high'].values, distance=5, prominence=tail['high'].std() * 0.5)[0]
        trough_indices = signal.find_peaks(-tail['low'].values, distance=5, prominence=tail['low'].std() * 0.5)[0]
        
        # تبدیل به قیمت
        peaks = tail['high'].iloc[peak_indices].values
        troughs = tail['low'].iloc[trough_indices].values
        
        # اضافه کردن به سطوح حمایت و مقاومت
        current_price = df['close'].iloc[-1]
        
        for peak in peaks:
            if peak > current_price:
                sr_levels['resistance'].append(peak)
        
        for trough in troughs:
            if trough < current_price:
                sr_levels['support'].append(trough)
        
        # حذف سطوح تکراری و مرتب‌سازی
        sr_levels['resistance'] = sorted(list(set(sr_levels['resistance'])))
        sr_levels['support'] = sorted(list(set(sr_levels['support'])), reverse=True)
        
        # محدود کردن به تعداد سطوح مورد نیاز
        sr_levels['resistance'] = sr_levels['resistance'][:levels]
        sr_levels['support'] = sr_levels['support'][:levels]
        
        return sr_levels
    
    def _calculate_final_signal(self, indicator_signals: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, float]:
        """
        محاسبه سیگنال نهایی براساس ترکیب همه سیگنال‌ها
        
        Args:
            indicator_signals (Dict[str, List[Dict[str, Any]]]): سیگنال‌های اندیکاتورها
            
        Returns:
            Tuple[str, float]: سیگنال نهایی و قدرت آن
        """
        # محاسبه مجموع وزن سیگنال‌ها
        total_strength = 0
        total_weight = 0
        
        # وزن‌دهی دسته‌های مختلف
        category_weights = {
            'trend': 1.5,      # اندیکاتورهای روند اهمیت بیشتری دارند
            'momentum': 1.0,
            'volatility': 0.8,
            'volume': 1.0,
            'patterns': 1.2,   # الگوهای شمعی اهمیت بالایی دارند
            'custom': 0.7
        }
        
        # جمع‌آوری همه سیگنال‌ها با اعمال وزن‌ها
        for category, signals in indicator_signals.items():
            category_weight = category_weights.get(category, 1.0)
            
            for signal in signals:
                total_strength += signal['strength'] * category_weight
                total_weight += category_weight
        
        # محاسبه قدرت کلی سیگنال
        if total_weight > 0:
            signal_strength = total_strength / total_weight
        else:
            signal_strength = 0
        
        # تعیین نوع سیگنال
        if signal_strength > 1.0:
            signal = "صعودی قوی"
        elif signal_strength > 0.3:
            signal = "صعودی"
        elif signal_strength > -0.3:
            signal = "خنثی"
        elif signal_strength > -1.0:
            signal = "نزولی"
        else:
            signal = "نزولی قوی"
        
        return signal, signal_strength
    
    def _calculate_entry_exit_points(self, final_signal: str, support_resistance: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        تعیین نقاط ورود، حد سود و حد ضرر
        
        Args:
            final_signal (str): سیگنال نهایی
            support_resistance (Dict[str, List[float]]): سطوح حمایت و مقاومت
            
        Returns:
            Dict[str, Any]: دیکشنری نقاط ورود و خروج
        """
        # قیمت فعلی
        current_price = self.indicators_data['close'].iloc[-1]
        
        # مقداردهی اولیه
        entry_exit = {
            'entry': None,
            'targets': {
                'tp1': None,
                'tp2': None,
                'tp3': None,
                'tp4': None
            },
            'stop_loss': None,
            'risk_reward': None
        }
        
        # نقطه ورود
        # برای سیگنال‌های صعودی، نقطه ورود قیمت فعلی است
        # برای سیگنال‌های نزولی، نقطه ورود قیمت فعلی است
        # برای سیگنال‌های خنثی، نقطه ورود تعیین نمی‌شود
        if "صعودی" in final_signal:
            entry_exit['entry'] = current_price
            
            # حد ضرر: پایین‌ترین سطح حمایت یا 2٪ پایین‌تر از قیمت ورود
            if support_resistance['support']:
                closest_support = support_resistance['support'][0]
                entry_exit['stop_loss'] = min(closest_support, current_price * 0.98)
            else:
                entry_exit['stop_loss'] = current_price * 0.98
            
            # اهداف قیمتی
            targets = support_resistance['resistance']
            
            if targets:
                # استفاده از سطوح مقاومت
                for i, target in enumerate(targets[:4], 1):
                    entry_exit['targets'][f'tp{i}'] = target
                
                # تکمیل اهداف باقی‌مانده
                remaining = 4 - len(targets)
                if remaining > 0:
                    last_target = targets[-1] if targets else current_price
                    risk = current_price - entry_exit['stop_loss']
                    
                    for i in range(len(targets) + 1, 5):
                        # هر هدف بعدی 1.5 برابر هدف قبلی (از نظر فاصله از ورود)
                        entry_exit['targets'][f'tp{i}'] = last_target + (risk * (i - len(targets)) * 1.5)
            else:
                # تعیین اهداف بر اساس نسبت ریسک به ریوارد
                risk = current_price - entry_exit['stop_loss']
                
                entry_exit['targets']['tp1'] = current_price + risk * 1.5  # ریسک به ریوارد 1:1.5
                entry_exit['targets']['tp2'] = current_price + risk * 2.5  # ریسک به ریوارد 1:2.5
                entry_exit['targets']['tp3'] = current_price + risk * 4    # ریسک به ریوارد 1:4
                entry_exit['targets']['tp4'] = current_price + risk * 6    # ریسک به ریوارد 1:6
        
        elif "نزولی" in final_signal:
            entry_exit['entry'] = current_price
            
            # حد ضرر: بالاترین سطح مقاومت یا 2٪ بالاتر از قیمت ورود
            if support_resistance['resistance']:
                closest_resistance = support_resistance['resistance'][0]
                entry_exit['stop_loss'] = max(closest_resistance, current_price * 1.02)
            else:
                entry_exit['stop_loss'] = current_price * 1.02
            
            # اهداف قیمتی
            targets = support_resistance['support']
            
            if targets:
                # استفاده از سطوح حمایت
                for i, target in enumerate(targets[:4], 1):
                    entry_exit['targets'][f'tp{i}'] = target
                
                # تکمیل اهداف باقی‌مانده
                remaining = 4 - len(targets)
                if remaining > 0:
                    last_target = targets[-1] if targets else current_price
                    risk = entry_exit['stop_loss'] - current_price
                    
                    for i in range(len(targets) + 1, 5):
                        # هر هدف بعدی 1.5 برابر هدف قبلی (از نظر فاصله از ورود)
                        entry_exit['targets'][f'tp{i}'] = last_target - (risk * (i - len(targets)) * 1.5)
            else:
                # تعیین اهداف بر اساس نسبت ریسک به ریوارد
                risk = entry_exit['stop_loss'] - current_price
                
                entry_exit['targets']['tp1'] = current_price - risk * 1.5  # ریسک به ریوارد 1:1.5
                entry_exit['targets']['tp2'] = current_price - risk * 2.5  # ریسک به ریوارد 1:2.5
                entry_exit['targets']['tp3'] = current_price - risk * 4    # ریسک به ریوارد 1:4
                entry_exit['targets']['tp4'] = current_price - risk * 6    # ریسک به ریوارد 1:6
        
        # محاسبه نسبت ریسک به ریوارد
        if entry_exit['entry'] is not None and entry_exit['stop_loss'] is not None and entry_exit['targets']['tp1'] is not None:
            if "صعودی" in final_signal:
                risk = entry_exit['entry'] - entry_exit['stop_loss']
                reward = entry_exit['targets']['tp1'] - entry_exit['entry']
            else:
                risk = entry_exit['stop_loss'] - entry_exit['entry']
                reward = entry_exit['entry'] - entry_exit['targets']['tp1']
            
            if risk > 0:
                entry_exit['risk_reward'] = reward / risk
            else:
                entry_exit['risk_reward'] = None
        
        # ذخیره برای استفاده‌های بعدی
        self.price_targets = entry_exit
        
        return entry_exit
    
    def get_market_signals(self) -> Dict[str, Any]:
        """
        دریافت سیگنال‌های بازار (اگر قبلاً تحلیل انجام شده باشد)
        
        Returns:
            Dict[str, Any]: سیگنال‌های بازار
        """
        if not self.signals:
            return self.analyze_market()
        return self.signals
    
    def get_entry_exit_points(self) -> Dict[str, Any]:
        """
        دریافت نقاط ورود و خروج (اگر قبلاً تحلیل انجام شده باشد)
        
        Returns:
            Dict[str, Any]: نقاط ورود و خروج
        """
        if not self.price_targets:
            self.analyze_market()
        return self.price_targets
    
    def plot_chart_with_signals(self, figsize=(14, 10)):
        """
        رسم نمودار با نمایش سیگنال‌ها
        
        Args:
            figsize (tuple): اندازه شکل
            
        Returns:
            plt.Figure: شکل matplotlib
        """
        # بررسی اینکه آیا تحلیل انجام شده است
        if not self.signals:
            self.analyze_market()
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # نمودار شمعی
        ax1 = axes[0]
        
        # تنظیم داده‌ها برای نمودار شمعی
        df = self.indicators_data
        
        # رسم نمودار شمعی
        width = 0.6
        width2 = 0.05
        
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]
        
        # رسم شمع‌های صعودی
        ax1.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green', alpha=0.5)
        ax1.bar(up.index, up['high'] - up['close'], width2, bottom=up['close'], color='green', alpha=0.5)
        ax1.bar(up.index, up['open'] - up['low'], width2, bottom=up['low'], color='green', alpha=0.5)
        
        # رسم شمع‌های نزولی
        ax1.bar(down.index, down['open'] - down['close'], width, bottom=down['close'], color='red', alpha=0.5)
        ax1.bar(down.index, down['high'] - down['open'], width2, bottom=down['open'], color='red', alpha=0.5)
        ax1.bar(down.index, down['close'] - down['low'], width2, bottom=down['low'], color='red', alpha=0.5)
        
        # اضافه کردن میانگین‌های متحرک
        if 'sma_20' in df.columns:
            ax1.plot(df.index, df['sma_20'], label='SMA 20', color='blue', linewidth=1)
        if 'ema_20' in df.columns:
            ax1.plot(df.index, df['ema_20'], label='EMA 20', color='orange', linewidth=1)
        
        # اضافه کردن باندهای بولینگر
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax1.plot(df.index, df['bb_upper'], label='BB Upper', color='purple', linewidth=1, alpha=0.5)
            ax1.plot(df.index, df['bb_middle'], label='BB Middle', color='purple', linewidth=1, alpha=0.5)
            ax1.plot(df.index, df['bb_lower'], label='BB Lower', color='purple', linewidth=1, alpha=0.5)
            ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='purple', alpha=0.05)
        
        # اضافه کردن اطلاعات سوپرترند
        if all(col in df.columns for col in ['supertrend', 'supertrend_uband', 'supertrend_lband']):
            ax1.plot(df.index, df['supertrend_uband'], label='SuperTrend Upper', color='red', linewidth=1, alpha=0.5)
            ax1.plot(df.index, df['supertrend_lband'], label='SuperTrend Lower', color='green', linewidth=1, alpha=0.5)
        
        # اضافه کردن خطوط افقی برای حمایت و مقاومت
        sr_levels = self._calculate_support_resistance()
        for level in sr_levels['resistance']:
            ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5)
        for level in sr_levels['support']:
            ax1.axhline(y=level, color='green', linestyle='--', alpha=0.5)
        
        # اضافه کردن نقاط ورود و حد سود/ضرر
        if self.price_targets.get('entry'):
            ax1.axhline(y=self.price_targets['entry'], color='black', linestyle='-', linewidth=1.5, label='ورود')
            
            if self.price_targets.get('stop_loss'):
                ax1.axhline(y=self.price_targets['stop_loss'], color='red', linestyle='-', linewidth=1.5, label='حد ضرر')
            
            for tp, value in self.price_targets['targets'].items():
                if value:
                    ax1.axhline(y=value, color='green', linestyle='-', linewidth=1, label=tp.upper())
        
        # تنظیمات نمودار اصلی
        ax1.set_ylabel('قیمت')
        ax1.set_title(f'{self.symbol} - {self.timeframe} - سیگنال: {self.signals.get("signal", "")}')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # نمودار RSI
        ax2 = axes[1]
        if 'rsi_14' in df.columns:
            ax2.plot(df.index, df['rsi_14'], color='purple', label='RSI 14')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.axhline(y=50, color='black', linestyle='--', alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        # نمودار MACD
        ax3 = axes[2]
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            ax3.plot(df.index, df['macd'], color='blue', label='MACD')
            ax3.plot(df.index, df['macd_signal'], color='red', label='Signal')
            ax3.bar(df.index, df['macd_hist'], color=['green' if x > 0 else 'red' for x in df['macd_hist']], alpha=0.5, label='Histogram')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax3.set_ylabel('MACD')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # تنظیمات نمودار کلی
        fig.tight_layout()
        return fig
    
    def get_indicator_evaluation(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        ارزیابی عملکرد اندیکاتورها
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: ارزیابی اندیکاتورها
        """
        if not self.signals:
            self.analyze_market()
        
        # سیگنال نهایی
        final_signal = self.signals['signal']
        is_bullish = "صعودی" in final_signal
        is_bearish = "نزولی" in final_signal
        
        # ارزیابی اندیکاتورها
        evaluation = {
            'best_indicators': [],
            'worst_indicators': [],
            'conflicting_indicators': []
        }
        
        all_indicators = []
        
        # جمع‌آوری همه اندیکاتورها
        for category, signals in self.signals['indicator_signals'].items():
            for signal in signals:
                # تعیین آیا سیگنال اندیکاتور با سیگنال نهایی همخوانی دارد
                indicator_bullish = signal['strength'] > 0
                indicator_bearish = signal['strength'] < 0
                
                agrees_with_final = (is_bullish and indicator_bullish) or (is_bearish and indicator_bearish)
                
                all_indicators.append({
                    'name': signal['name'],
                    'category': category,
                    'signal': signal['signal'],
                    'strength': signal['strength'],
                    'agrees_with_final': agrees_with_final
                })
        
        # مرتب‌سازی بر اساس قدرت سیگنال (مطلق)
        all_indicators.sort(key=lambda x: abs(x['strength']), reverse=True)
        
        # بهترین اندیکاتورها (بالاترین قدرت در جهت سیگنال نهایی)
        best = [ind for ind in all_indicators if ind['agrees_with_final']][:5]
        evaluation['best_indicators'] = best
        
        # بدترین اندیکاتورها (بالاترین قدرت در خلاف جهت سیگنال نهایی)
        worst = [ind for ind in all_indicators if not ind['agrees_with_final']][:5]
        evaluation['worst_indicators'] = worst
        
        # اندیکاتورهای متضاد (اندیکاتورهایی که در یک دسته سیگنال متفاوت می‌دهند)
        for category, signals in self.signals['indicator_signals'].items():
            if len(signals) >= 2:
                bullish_signals = [s for s in signals if s['strength'] > 0]
                bearish_signals = [s for s in signals if s['strength'] < 0]
                
                if bullish_signals and bearish_signals:
                    for b_signal in bullish_signals:
                        for bear_signal in bearish_signals:
                            if abs(b_signal['strength']) > 1 and abs(bear_signal['strength']) > 1:
                                evaluation['conflicting_indicators'].append({
                                    'category': category,
                                    'bullish_indicator': b_signal['name'],
                                    'bearish_indicator': bear_signal['name'],
                                    'bullish_strength': b_signal['strength'],
                                    'bearish_strength': bear_signal['strength']
                                })
        
        return evaluation
    
    def generate_market_insight(self) -> str:
        """
        تولید بینش‌های بازار
        
        Returns:
            str: متن بینش‌های بازار
        """
        if not self.signals:
            self.analyze_market()
        
        signal = self.signals.get('signal', '')
        symbol = self.symbol
        timeframe = self.timeframe
        
        # ایجاد متن بینش بازار
        insights = []
        
        # عنوان
        insights.append(f"تحلیل جامع بازار {symbol} در تایم‌فریم {timeframe} - ترکیب 400 اندیکاتور")
        insights.append("-" * 80)
        
        # سیگنال کلی
        insights.append(f"سیگنال: {signal}")
        insights.append(f"قدرت سیگنال: {self.signals.get('signal_strength', 0):.2f}")
        insights.append("-" * 40)
        
        # شرایط بازار
        market_conditions = self.signals.get('market_conditions', {})
        condition_texts = []
        
        if market_conditions.get('trending'):
            condition_texts.append("روند دار")
        if market_conditions.get('ranging'):
            condition_texts.append("نوسانی")
        if market_conditions.get('high_volatility'):
            condition_texts.append("نوسان بالا")
        if market_conditions.get('low_volatility'):
            condition_texts.append("نوسان پایین")
        if market_conditions.get('breakout_potential'):
            condition_texts.append("پتانسیل شکست")
        if market_conditions.get('support_resistance'):
            condition_texts.append("نزدیک سطوح مهم")
        if market_conditions.get('cycle_turn'):
            condition_texts.append("تغییر چرخه")
        
        insights.append(f"شرایط بازار: {', '.join(condition_texts)}")
        insights.append("-" * 40)
        
        # نقاط ورود و خروج
        entry_exit = self.price_targets
        
        if entry_exit.get('entry'):
            insights.append(f"نقطه ورود: {entry_exit['entry']:.2f}")
            
            if entry_exit.get('stop_loss'):
                insights.append(f"حد ضرر: {entry_exit['stop_loss']:.2f}")
            
            insights.append("اهداف قیمتی:")
            for tp, value in entry_exit['targets'].items():
                if value:
                    insights.append(f"  {tp.upper()}: {value:.2f}")
            
            if entry_exit.get('risk_reward'):
                insights.append(f"نسبت ریسک به ریوارد: 1:{entry_exit['risk_reward']:.2f}")
            
            insights.append("-" * 40)
        
        # سیگنال‌های شاخص
        insights.append("مهمترین سیگنال‌ها:")
        
        categories = {
            'trend': 'روند',
            'momentum': 'مومنتوم',
            'volatility': 'نوسان',
            'volume': 'حجم',
            'patterns': 'الگوها',
            'custom': 'سفارشی'
        }
        
        for category, signals in self.signals.get('indicator_signals', {}).items():
            if signals:
                insights.append(f"\n• {categories.get(category, category)}:")
                
                for signal in signals[:3]:  # نمایش 3 سیگنال برتر هر دسته
                    direction = "↑" if signal['strength'] > 0 else "↓" if signal['strength'] < 0 else "→"
                    insights.append(f"  {direction} {signal['name']}: {signal['signal']}")
        
        insights.append("-" * 40)
        
        # الگوهای قیمت
        price_patterns = self.signals.get('price_patterns', [])
        if price_patterns:
            insights.append("الگوهای قیمت شناسایی شده:")
            
            for pattern in price_patterns:
                direction = "صعودی" if pattern['direction'] == 'bullish' else "نزولی" if pattern['direction'] == 'bearish' else "خنثی"
                target_text = f"(هدف: {pattern['target']:.2f})" if pattern.get('target') else ""
                insights.append(f"• {pattern['name']} - {direction} {target_text}")
            
            insights.append("-" * 40)
        
        # هشدارها و نکات مهم
        warnings = []
        
        # بررسی تناقض‌ها
        evaluation = self.get_indicator_evaluation()
        if evaluation.get('conflicting_indicators'):
            warnings.append("هشدار: اندیکاتورهای متضاد در بازار وجود دارند.")
        
        # هشدار برای شرایط بیش‌خرید/بیش‌فروش
        for category, signals in self.signals.get('indicator_signals', {}).items():
            for signal in signals:
                if "اشباع خرید" in signal['signal'] and "صعودی" in self.signals.get('signal', ''):
                    warnings.append(f"هشدار: علیرغم سیگنال صعودی، {signal['name']} در وضعیت اشباع خرید است.")
                elif "اشباع فروش" in signal['signal'] and "نزولی" in self.signals.get('signal', ''):
                    warnings.append(f"هشدار: علیرغم سیگنال نزولی، {signal['name']} در وضعیت اشباع فروش است.")
        
        if warnings:
            insights.append("هشدارها:")
            for warning in warnings:
                insights.append(f"• {warning}")
            insights.append("-" * 40)
        
        # توصیه نهایی
        insights.append("توصیه نهایی:")
        
        if "صعودی قوی" in signal:
            insights.append("• موقعیت خرید قوی با ریسک کنترل شده")
            insights.append("• تقسیم سرمایه برای ورود در نقاط حمایتی در صورت اصلاح")
            insights.append("• مدیریت پوزیشن با برداشت سود تدریجی در نقاط TP")
        elif "صعودی" in signal:
            insights.append("• موقعیت خرید با احتیاط")
            insights.append("• ورود در سطوح حمایتی")
            insights.append("• حجم معامله کمتر و مدیریت ریسک دقیق")
        elif "نزولی قوی" in signal:
            insights.append("• موقعیت فروش قوی با ریسک کنترل شده")
            insights.append("• تقسیم سرمایه برای ورود در نقاط مقاومتی در صورت اصلاح صعودی")
            insights.append("• مدیریت پوزیشن با برداشت سود تدریجی در نقاط TP")
        elif "نزولی" in signal:
            insights.append("• موقعیت فروش با احتیاط")
            insights.append("• ورود در سطوح مقاومتی")
            insights.append("• حجم معامله کمتر و مدیریت ریسک دقیق")
        else:
            insights.append("• عدم ورود به بازار تا مشخص شدن جهت روند")
            insights.append("• نگهداری نقدینگی و بررسی مجدد در تایم‌فریم‌های بالاتر")
        
        insights.append("-" * 80)
        insights.append("این تحلیل توسط سیستم هوشمند SuperAnalyzer با استفاده از بیش از 400 اندیکاتور تولید شده است.")
        insights.append("برای نتایج بهتر، این تحلیل را با تحلیل‌های تایم‌فریم‌های بالاتر و اخبار بنیادی ترکیب کنید.")
        
        return "\n".join(insights)