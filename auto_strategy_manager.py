"""
ماژول مدیریت خودکار استراتژی‌های معاملاتی

این ماژول شامل توابع و کلاس‌های مورد نیاز برای تولید و مدیریت خودکار استراتژی‌های معاملاتی
با توجه به شرایط بازار و پروفایل ریسک کاربر است.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import math
from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoStrategyManager:
    """کلاس اصلی مدیریت خودکار استراتژی‌های معاملاتی"""
    
    # تعریف انواع استراتژی
    STRATEGY_TYPES = {
        "trend_following": "دنبال کردن روند",
        "mean_reversion": "بازگشت به میانگین",
        "breakout": "شکست",
        "oscillator": "نوسان‌گر",
        "harmonic": "الگوهای هارمونیک",
        "support_resistance": "حمایت و مقاومت",
        "volume_based": "مبتنی بر حجم",
        "volatility_based": "مبتنی بر نوسانات",
        "low_risk": "ریسک پایین"
    }
    
    def __init__(self, 
                risk_profile: str = 'متعادل', 
                capital: float = 1000.0,
                max_trades: int = 5,
                use_stop_loss: bool = True):
        """
        مقداردهی اولیه مدیریت استراتژی
        
        Args:
            risk_profile (str): پروفایل ریسک کاربر (کم‌ریسک، متعادل، پرریسک)
            capital (float): سرمایه کل موجود
            max_trades (int): حداکثر تعداد معاملات همزمان
            use_stop_loss (bool): استفاده از حد ضرر
        """
        self.risk_profile = risk_profile
        self.capital = capital
        self.max_trades = max_trades
        self.use_stop_loss = use_stop_loss
        
        # تنظیم نسبت ریسک به پاداش بر اساس پروفایل ریسک
        self.risk_reward_ratios = {
            'کم‌ریسک': 3.0,    # ریسک کم، برای 1% ریسک، انتظار 3% سود
            'متعادل': 2.0,     # ریسک متعادل، برای 1% ریسک، انتظار 2% سود
            'پرریسک': 1.5      # ریسک بالا، برای 1% ریسک، انتظار 1.5% سود
        }
        
        # محدودیت‌های مربوط به اهرم بر اساس پروفایل ریسک
        self.max_leverage = {
            'کم‌ریسک': 1,      # بدون اهرم
            'متعادل': 3,       # حداکثر اهرم 3x
            'پرریسک': 5        # حداکثر اهرم 5x
        }
        
        # درصد سرمایه در هر معامله بر اساس پروفایل ریسک
        self.position_sizes = {
            'کم‌ریسک': 0.05,    # 5% از سرمایه در هر معامله
            'متعادل': 0.1,     # 10% از سرمایه در هر معامله
            'پرریسک': 0.2      # 20% از سرمایه در هر معامله
        }
        
        # تاریخچه استراتژی‌های تولید شده
        self.strategy_history = []
        
        logger.info(f"مدیریت خودکار استراتژی با پروفایل ریسک '{risk_profile}' و سرمایه {capital} راه‌اندازی شد")
    
    def get_market_condition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        تعیین شرایط بازار با تحلیل داده‌های قیمت
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            
        Returns:
            dict: اطلاعات شرایط بازار
        """
        if df.empty or len(df) < 20:
            logger.warning("داده‌های ناکافی برای تعیین شرایط بازار")
            return {
                'trend': 'نامشخص',
                'volatility': 'متوسط',
                'volume': 'متوسط',
                'momentum': 'خنثی',
                'description': 'اطلاعات کافی برای تحلیل موجود نیست'
            }
        
        try:
            # محاسبه میانگین متحرک‌ها برای تعیین روند
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['MA200'] = df['close'].rolling(window=200).mean()
            
            # محاسبه شاخص نوسانات
            df['daily_return'] = df['close'].pct_change()
            
            # محاسبه میانگین حجم
            if 'volume' in df.columns:
                df['volume_MA20'] = df['volume'].rolling(window=20).mean()
            
            # برش آخرین رکورد
            last_record = df.iloc[-1]
            
            # تعیین روند
            if last_record['MA20'] > last_record['MA50'] > last_record['MA200']:
                trend = 'صعودی قوی'
            elif last_record['MA20'] > last_record['MA50']:
                trend = 'صعودی'
            elif last_record['MA50'] < last_record['MA200']:
                trend = 'نزولی قوی'
            elif last_record['MA20'] < last_record['MA50']:
                trend = 'نزولی'
            else:
                trend = 'نوسانی'
            
            # تعیین نوسانات
            volatility = df['daily_return'].std() * 100  # تبدیل به درصد
            
            if volatility < 1:
                volatility_desc = 'کم'
            elif volatility < 3:
                volatility_desc = 'متوسط'
            else:
                volatility_desc = 'بالا'
            
            # تعیین حجم
            volume_desc = 'نامشخص'
            if 'volume' in df.columns and 'volume_MA20' in df.columns:
                volume_ratio = last_record['volume'] / last_record['volume_MA20']
                
                if volume_ratio < 0.7:
                    volume_desc = 'کم'
                elif volume_ratio < 1.3:
                    volume_desc = 'متوسط'
                else:
                    volume_desc = 'بالا'
            
            # محاسبه مومنتوم
            if len(df) >= 20:
                recent_returns = df['daily_return'].iloc[-20:].mean() * 100
                
                if recent_returns > 1:
                    momentum = 'قوی مثبت'
                elif recent_returns > 0.2:
                    momentum = 'مثبت'
                elif recent_returns < -1:
                    momentum = 'قوی منفی'
                elif recent_returns < -0.2:
                    momentum = 'منفی'
                else:
                    momentum = 'خنثی'
            else:
                momentum = 'نامشخص'
            
            # توصیف شرایط بازار
            description = f"بازار در یک روند {trend} با نوسانات {volatility_desc} و حجم {volume_desc} است. "
            description += f"مومنتوم بازار {momentum} است. "
            
            # تعیین بهترین استراتژی‌ها برای شرایط فعلی
            suitable_strategies = self._get_suitable_strategies(trend, volatility_desc, volume_desc, momentum)
            
            description += f"در این شرایط، استراتژی‌های {', '.join(suitable_strategies[:3])} می‌توانند مناسب باشند."
            
            # نتایج نهایی
            return {
                'trend': trend,
                'volatility': volatility_desc,
                'volatility_value': float(volatility),
                'volume': volume_desc,
                'momentum': momentum,
                'description': description,
                'suitable_strategies': suitable_strategies,
                'last_price': float(last_record['close']),
                'ma20': float(last_record['MA20']),
                'ma50': float(last_record['MA50']),
                'ma200': float(last_record['MA200']),
                'summary': self._generate_market_summary(trend, volatility_desc, volume_desc, momentum)
            }
        
        except Exception as e:
            logger.error(f"خطا در تعیین شرایط بازار: {str(e)}")
            return {
                'trend': 'نامشخص',
                'volatility': 'متوسط',
                'volume': 'متوسط',
                'momentum': 'خنثی',
                'description': 'خطا در تحلیل شرایط بازار',
                'error': str(e)
            }
    
    def _get_suitable_strategies(self, trend: str, volatility: str, volume: str, momentum: str) -> List[str]:
        """
        تعیین استراتژی‌های مناسب برای شرایط فعلی بازار
        
        Args:
            trend (str): روند بازار
            volatility (str): نوسانات بازار
            volume (str): حجم معاملات
            momentum (str): مومنتوم بازار
            
        Returns:
            list: لیست استراتژی‌های مناسب
        """
        strategies = []
        
        # روند صعودی قوی
        if trend == 'صعودی قوی':
            strategies.append(self.STRATEGY_TYPES['trend_following'])
            if volume == 'بالا':
                strategies.append(self.STRATEGY_TYPES['breakout'])
            if momentum in ['قوی مثبت', 'مثبت']:
                strategies.append(self.STRATEGY_TYPES['oscillator'])
        
        # روند صعودی
        elif trend == 'صعودی':
            strategies.append(self.STRATEGY_TYPES['trend_following'])
            strategies.append(self.STRATEGY_TYPES['support_resistance'])
            if momentum == 'خنثی':
                strategies.append(self.STRATEGY_TYPES['mean_reversion'])
        
        # روند نزولی
        elif trend == 'نزولی':
            strategies.append(self.STRATEGY_TYPES['mean_reversion'])
            strategies.append(self.STRATEGY_TYPES['support_resistance'])
            if self.risk_profile != 'کم‌ریسک':
                strategies.append(self.STRATEGY_TYPES['oscillator'])
        
        # روند نزولی قوی
        elif trend == 'نزولی قوی':
            strategies.append(self.STRATEGY_TYPES['low_risk'])
            if momentum in ['قوی منفی', 'منفی'] and self.risk_profile != 'کم‌ریسک':
                strategies.append(self.STRATEGY_TYPES['trend_following'])  # برای پوزیشن شورت
        
        # روند نوسانی
        else:  # نوسانی
            strategies.append(self.STRATEGY_TYPES['mean_reversion'])
            strategies.append(self.STRATEGY_TYPES['support_resistance'])
            if volatility == 'بالا':
                strategies.append(self.STRATEGY_TYPES['volatility_based'])
        
        # استراتژی‌های خاص حجم
        if volume == 'بالا':
            strategies.append(self.STRATEGY_TYPES['volume_based'])
        
        # استراتژی‌های خاص نوسانات
        if volatility == 'بالا':
            if self.risk_profile != 'کم‌ریسک':
                strategies.append(self.STRATEGY_TYPES['volatility_based'])
        
        # استراتژی‌های هارمونیک
        if self.risk_profile in ['متعادل', 'پرریسک'] and volatility != 'کم':
            strategies.append(self.STRATEGY_TYPES['harmonic'])
        
        # اضافه کردن استراتژی کم‌ریسک برای پروفایل کم‌ریسک همیشه
        if self.risk_profile == 'کم‌ریسک':
            strategies.append(self.STRATEGY_TYPES['low_risk'])
        
        # حذف تکراری‌ها و مرتب‌سازی
        return list(dict.fromkeys(strategies))
    
    def _generate_market_summary(self, trend: str, volatility: str, volume: str, momentum: str) -> str:
        """
        تولید خلاصه وضعیت بازار
        
        Args:
            trend (str): روند بازار
            volatility (str): نوسانات بازار
            volume (str): حجم معاملات
            momentum (str): مومنتوم بازار
            
        Returns:
            str: خلاصه وضعیت بازار
        """
        summary = f"بازار در حال حاضر در یک روند {trend} است. "
        
        if trend.startswith('صعودی'):
            if momentum in ['قوی مثبت', 'مثبت']:
                summary += "این روند با قدرت مومنتوم مثبت پشتیبانی می‌شود. "
            elif momentum == 'خنثی':
                summary += "مومنتوم خنثی نشان‌دهنده احتمال تداوم این روند با شتاب فعلی است. "
            else:
                summary += "ضعف در مومنتوم نشان‌دهنده احتمال اصلاح یا کندی روند است. "
        elif trend.startswith('نزولی'):
            if momentum in ['قوی منفی', 'منفی']:
                summary += "این روند با فشار فروش قابل توجه ادامه دارد. "
            elif momentum == 'خنثی':
                summary += "کاهش فشار فروش ممکن است منجر به کف قیمت شود. "
            else:
                summary += "مومنتوم مثبت در روند نزولی می‌تواند نشانه‌ای از واگرایی و تغییر احتمالی روند باشد. "
        else:  # روند نوسانی
            summary += "قیمت‌ها در یک محدوده مشخص نوسان می‌کنند. "
            
        # اضافه کردن اطلاعات نوسانات
        if volatility == 'بالا':
            summary += "نوسانات بالا نشان‌دهنده عدم قطعیت و تغییرات سریع قیمت است. "
        elif volatility == 'کم':
            summary += "نوسانات کم معمولاً قبل از حرکت‌های بزرگ قیمتی رخ می‌دهد. "
        
        # اضافه کردن اطلاعات حجم
        if volume == 'بالا':
            summary += "حجم معاملات بالا اعتبار روند فعلی را تأیید می‌کند. "
        elif volume == 'کم':
            summary += "حجم معاملات پایین نشان‌دهنده کاهش اطمینان به روند فعلی است. "
        
        return summary
    
    def generate_strategy(self, 
                         symbol: str, 
                         df: pd.DataFrame, 
                         market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        تولید استراتژی معاملاتی برای شرایط فعلی بازار
        
        Args:
            symbol (str): نماد ارز
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            market_condition (dict, optional): شرایط بازار (در صورت عدم ارائه محاسبه می‌شود)
            
        Returns:
            dict: استراتژی معاملاتی تولید شده
        """
        try:
            # تعیین شرایط بازار اگر ارائه نشده باشد
            if market_condition is None:
                market_condition = self.get_market_condition(df)
            
            # آخرین قیمت
            last_price = market_condition.get('last_price', df['close'].iloc[-1])
            
            # انتخاب استراتژی مناسب
            suitable_strategies = market_condition.get('suitable_strategies', [])
            if not suitable_strategies:
                suitable_strategies = self._get_suitable_strategies(
                    market_condition.get('trend', 'نوسانی'),
                    market_condition.get('volatility', 'متوسط'),
                    market_condition.get('volume', 'متوسط'),
                    market_condition.get('momentum', 'خنثی')
                )
            
            primary_strategy = suitable_strategies[0] if suitable_strategies else self.STRATEGY_TYPES['low_risk']
            
            # تعیین سیگنال معاملاتی
            signal = self._generate_trading_signal(df, market_condition)
            
            # محاسبه سایز پوزیشن
            position_size = self._calculate_position_size(last_price, signal)
            
            # محاسبه قیمت‌های ورود، خروج و حد ضرر
            entry_exit = self._calculate_entry_exit_points(df, signal, market_condition)
            
            # ساخت توضیحات استراتژی
            description = self._generate_strategy_description(
                primary_strategy, signal, market_condition, entry_exit
            )
            
            # تولید استراتژی نهایی
            strategy = {
                'symbol': symbol,
                'strategy_type': primary_strategy,
                'signal': signal,
                'position_size': position_size,
                'risk_reward': self.risk_reward_ratios.get(self.risk_profile, 2.0),
                'entry_price': entry_exit['entry_price'],
                'stop_loss': entry_exit['stop_loss'],
                'target_1': entry_exit['target_1'],
                'target_2': entry_exit['target_2'],
                'target_3': entry_exit['target_3'],
                'risk_percentage': entry_exit['risk_percentage'],
                'description': description,
                'market_condition': market_condition,
                'timestamp': datetime.now().isoformat(),
                'confidence': entry_exit['confidence']
            }
            
            # ذخیره در تاریخچه
            self.strategy_history.append(strategy)
            
            return strategy
        
        except Exception as e:
            logger.error(f"خطا در تولید استراتژی معاملاتی: {str(e)}")
            return {
                'symbol': symbol,
                'strategy_type': self.STRATEGY_TYPES['low_risk'],
                'signal': 'NEUTRAL',
                'position_size': 0,
                'description': f"خطا در تولید استراتژی: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_trading_signal(self, df: pd.DataFrame, market_condition: Dict[str, Any]) -> str:
        """
        تولید سیگنال معاملاتی بر اساس شرایط بازار
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            market_condition (dict): شرایط بازار
            
        Returns:
            str: سیگنال معاملاتی (BUY, SELL, NEUTRAL)
        """
        trend = market_condition.get('trend', 'نوسانی')
        momentum = market_condition.get('momentum', 'خنثی')
        
        # برای کاربران کم‌ریسک، در شرایط نوسانی یا نزولی، سیگنال خنثی ارسال می‌کنیم
        if self.risk_profile == 'کم‌ریسک' and (trend.startswith('نزولی') or trend == 'نوسانی'):
            return 'NEUTRAL'
        
        # تصمیم‌گیری بر اساس روند و مومنتوم
        if trend.startswith('صعودی'):
            if momentum in ['قوی مثبت', 'مثبت', 'خنثی']:
                return 'BUY'
            else:
                return 'NEUTRAL'
        elif trend.startswith('نزولی'):
            if self.risk_profile == 'پرریسک' and momentum in ['قوی منفی', 'منفی']:
                return 'SELL'
            elif momentum in ['قوی مثبت', 'مثبت'] and trend == 'نزولی':  # نه نزولی قوی
                return 'BUY'  # احتمال برگشت
            else:
                return 'NEUTRAL'
        else:  # نوسانی
            # بررسی پرایس اکشن اخیر
            last_prices = df['close'].tail(5).values
            if last_prices[-1] > last_prices[-2] > last_prices[-3]:
                return 'BUY'
            elif last_prices[-1] < last_prices[-2] < last_prices[-3]:
                return 'SELL' if self.risk_profile == 'پرریسک' else 'NEUTRAL'
            else:
                return 'NEUTRAL'
    
    def _calculate_position_size(self, price: float, signal: str) -> float:
        """
        محاسبه اندازه پوزیشن بر اساس مدیریت ریسک
        
        Args:
            price (float): قیمت فعلی ارز
            signal (str): سیگنال معاملاتی
            
        Returns:
            float: اندازه پوزیشن
        """
        if signal == 'NEUTRAL':
            return 0.0
        
        # دریافت درصد سرمایه قابل استفاده برای هر معامله
        position_percentage = self.position_sizes.get(self.risk_profile, 0.1)
        
        # محاسبه میزان سرمایه قابل استفاده
        capital_per_trade = self.capital * position_percentage
        
        # محاسبه تعداد واحدهای قابل خرید
        position_size = capital_per_trade / price
        
        return position_size
    
    def _calculate_entry_exit_points(self, df: pd.DataFrame, signal: str, market_condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        محاسبه قیمت‌های ورود، خروج و حد ضرر
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            signal (str): سیگنال معاملاتی
            market_condition (dict): شرایط بازار
            
        Returns:
            dict: قیمت‌های ورود، خروج و حد ضرر
        """
        if signal == 'NEUTRAL':
            last_price = float(df['close'].iloc[-1])
            return {
                'entry_price': last_price,
                'stop_loss': last_price * 0.97,
                'target_1': last_price * 1.03,
                'target_2': last_price * 1.05,
                'target_3': last_price * 1.1,
                'risk_percentage': 0,
                'confidence': 0
            }
        
        # آخرین قیمت
        last_price = float(df['close'].iloc[-1])
        
        # نوسانات اخیر برای محاسبه حد ضرر
        recent_volatility = df['close'].pct_change().abs().tail(20).mean() * 100  # درصد
        
        # تعیین درصد حد ضرر بر اساس نوسانات و پروفایل ریسک
        risk_multiplier = {
            'کم‌ریسک': 1.5,
            'متعادل': 2.0,
            'پرریسک': 2.5
        }.get(self.risk_profile, 2.0)
        
        stop_loss_pct = min(max(recent_volatility * risk_multiplier, 1.0), 5.0)  # حداقل 1%، حداکثر 5%
        
        # تعیین نسبت ریسک به پاداش
        risk_reward = self.risk_reward_ratios.get(self.risk_profile, 2.0)
        
        # محاسبه قیمت‌های هدف و حد ضرر
        if signal == 'BUY':
            stop_loss = last_price * (1 - stop_loss_pct / 100)
            target_1 = last_price * (1 + (stop_loss_pct * risk_reward) / 100)
            target_2 = last_price * (1 + (stop_loss_pct * risk_reward * 1.5) / 100)
            target_3 = last_price * (1 + (stop_loss_pct * risk_reward * 2) / 100)
        else:  # SELL
            stop_loss = last_price * (1 + stop_loss_pct / 100)
            target_1 = last_price * (1 - (stop_loss_pct * risk_reward) / 100)
            target_2 = last_price * (1 - (stop_loss_pct * risk_reward * 1.5) / 100)
            target_3 = last_price * (1 - (stop_loss_pct * risk_reward * 2) / 100)
        
        # محاسبه اطمینان سیگنال
        confidence = self._calculate_signal_confidence(df, signal, market_condition)
        
        return {
            'entry_price': last_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            'risk_percentage': stop_loss_pct,
            'confidence': confidence
        }
    
    def _calculate_signal_confidence(self, df: pd.DataFrame, signal: str, market_condition: Dict[str, Any]) -> float:
        """
        محاسبه میزان اطمینان به سیگنال
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            signal (str): سیگنال معاملاتی
            market_condition (dict): شرایط بازار
            
        Returns:
            float: میزان اطمینان (0-100)
        """
        if signal == 'NEUTRAL':
            return 0.0
        
        # فاکتورهای مختلف برای محاسبه اطمینان
        factors = []
        
        # روند بازار
        trend = market_condition.get('trend', 'نوسانی')
        if signal == 'BUY' and trend.startswith('صعودی'):
            trend_factor = 0.8
            if trend == 'صعودی قوی':
                trend_factor = 1.0
        elif signal == 'SELL' and trend.startswith('نزولی'):
            trend_factor = 0.8
            if trend == 'نزولی قوی':
                trend_factor = 1.0
        else:
            trend_factor = 0.3
        factors.append(trend_factor)
        
        # مومنتوم
        momentum = market_condition.get('momentum', 'خنثی')
        if signal == 'BUY' and momentum in ['قوی مثبت', 'مثبت']:
            momentum_factor = 0.9 if momentum == 'قوی مثبت' else 0.7
        elif signal == 'SELL' and momentum in ['قوی منفی', 'منفی']:
            momentum_factor = 0.9 if momentum == 'قوی منفی' else 0.7
        else:
            momentum_factor = 0.4
        factors.append(momentum_factor)
        
        # حجم معاملات
        volume = market_condition.get('volume', 'متوسط')
        volume_factor = 0.5
        if volume == 'بالا':
            volume_factor = 0.8
        factors.append(volume_factor)
        
        # میانگین‌های متحرک (MA20 بالاتر/پایین‌تر از MA50)
        if 'ma20' in market_condition and 'ma50' in market_condition:
            ma20 = market_condition['ma20']
            ma50 = market_condition['ma50']
            if signal == 'BUY' and ma20 > ma50:
                ma_factor = 0.7
            elif signal == 'SELL' and ma20 < ma50:
                ma_factor = 0.7
            else:
                ma_factor = 0.3
            factors.append(ma_factor)
        
        # محاسبه میانگین فاکتورها
        confidence = sum(factors) / len(factors) * 100
        
        return confidence
    
    def _generate_strategy_description(self, strategy_type: str, signal: str, 
                                     market_condition: Dict[str, Any], 
                                     entry_exit: Dict[str, Any]) -> str:
        """
        تولید توضیحات استراتژی
        
        Args:
            strategy_type (str): نوع استراتژی
            signal (str): سیگنال معاملاتی
            market_condition (dict): شرایط بازار
            entry_exit (dict): قیمت‌های ورود و خروج
            
        Returns:
            str: توضیحات استراتژی
        """
        description = ""
        
        # مقدمه بر اساس سیگنال
        if signal == 'BUY':
            description = "استراتژی خرید (لانگ) با "
        elif signal == 'SELL':
            description = "استراتژی فروش (شورت) با "
        else:
            return "در حال حاضر سیگنال معاملاتی مشخصی وجود ندارد. پیشنهاد می‌شود منتظر شرایط بهتر بازار بمانید."
        
        # اضافه کردن نوع استراتژی
        description += f"رویکرد {strategy_type}"
        
        # اضافه کردن اطلاعات شرایط بازار
        description += f" برای شرایط فعلی بازار (روند {market_condition.get('trend', 'نامشخص')}, "
        description += f"با مومنتوم {market_condition.get('momentum', 'نامشخص')}) "
        
        # اضافه کردن اطلاعات قیمت ورود و اهداف
        description += f"توصیه می‌شود. "
        description += f"قیمت ورود {entry_exit['entry_price']:.2f} با حد ضرر {entry_exit['stop_loss']:.2f} "
        description += f"و هدف‌های قیمتی {entry_exit['target_1']:.2f}, {entry_exit['target_2']:.2f} و {entry_exit['target_3']:.2f} تعیین شده است. "
        
        # اضافه کردن میزان ریسک و پاداش
        description += f"این معامله با ریسک {entry_exit['risk_percentage']:.1f}% و نسبت ریسک به پاداش {self.risk_reward_ratios.get(self.risk_profile, 2.0):.1f} می‌باشد."
        
        # اضافه کردن میزان اطمینان
        description += f" اطمینان به این سیگنال {entry_exit['confidence']:.1f}% است."
        
        return description
    
    def compare_strategies(self, 
                          df: pd.DataFrame, 
                          symbol: str, 
                          strategy_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        مقایسه استراتژی‌های مختلف برای شرایط فعلی بازار
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            symbol (str): نماد ارز
            strategy_types (list, optional): لیست انواع استراتژی‌ها برای مقایسه
            
        Returns:
            dict: نتایج مقایسه استراتژی‌ها
        """
        try:
            # تعیین شرایط بازار
            market_condition = self.get_market_condition(df)
            
            # اگر استراتژی‌ها مشخص نشده‌اند، از استراتژی‌های مناسب استفاده می‌کنیم
            if not strategy_types:
                strategy_types = list(self.STRATEGY_TYPES.values())
            
            # باحاصلان ریسک پروفایل اصلی
            original_risk_profile = self.risk_profile
            
            # نتایج مقایسه
            comparison_results = {
                'symbol': symbol,
                'market_condition': market_condition,
                'comparison_time': datetime.now().isoformat(),
                'strategies': []
            }
            
            # مقایسه استراتژی‌ها
            for strategy_type in strategy_types:
                # تغییر موقت پروفایل ریسک برای شبیه‌سازی استراتژی‌ها
                if strategy_type == self.STRATEGY_TYPES['low_risk']:
                    self.risk_profile = 'کم‌ریسک'
                elif strategy_type == self.STRATEGY_TYPES['trend_following']:
                    self.risk_profile = 'متعادل'
                elif strategy_type == self.STRATEGY_TYPES['volatility_based']:
                    self.risk_profile = 'پرریسک'
                
                # تولید استراتژی
                strategy = self.generate_strategy(symbol, df, market_condition)
                
                # اضافه کردن به نتایج مقایسه
                comparison_results['strategies'].append({
                    'strategy_type': strategy_type,
                    'signal': strategy['signal'],
                    'entry_price': strategy['entry_price'],
                    'stop_loss': strategy['stop_loss'],
                    'target_1': strategy['target_1'],
                    'risk_percentage': strategy['risk_percentage'],
                    'risk_reward': strategy['risk_reward'],
                    'confidence': strategy['confidence'],
                    'description': strategy['description']
                })
            
            # بازگرداندن پروفایل ریسک اصلی
            self.risk_profile = original_risk_profile
            
            # مرتب‌سازی بر اساس اطمینان
            comparison_results['strategies'].sort(key=lambda x: x['confidence'], reverse=True)
            
            # تعیین بهترین استراتژی
            if comparison_results['strategies']:
                best_strategy = comparison_results['strategies'][0]
                comparison_results['best_strategy'] = best_strategy
                comparison_results['summary'] = f"با توجه به شرایط فعلی بازار، استراتژی {best_strategy['strategy_type']} با اطمینان {best_strategy['confidence']:.1f}% بهترین گزینه است."
            else:
                comparison_results['best_strategy'] = None
                comparison_results['summary'] = "هیچ استراتژی مناسبی برای شرایط فعلی بازار یافت نشد."
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"خطا در مقایسه استراتژی‌ها: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'comparison_time': datetime.now().isoformat()
            }
    
    def backtest_strategy(self, 
                         df: pd.DataFrame, 
                         strategy_type: str, 
                         risk_profile: Optional[str] = None) -> Dict[str, Any]:
        """
        بک‌تست استراتژی معاملاتی بر روی داده‌های تاریخی
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            strategy_type (str): نوع استراتژی
            risk_profile (str, optional): پروفایل ریسک (اگر با پروفایل فعلی متفاوت است)
            
        Returns:
            dict: نتایج بک‌تست
        """
        # پیاده‌سازی بک‌تست با سیگنال‌های شبیه‌سازی شده
        # (این بخش در آینده پیاده‌سازی خواهد شد)
        return {}


def get_market_condition(df: pd.DataFrame) -> Dict[str, Any]:
    """
    تعیین شرایط بازار با تحلیل داده‌های قیمت
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        
    Returns:
        dict: اطلاعات شرایط بازار
    """
    manager = AutoStrategyManager()
    return manager.get_market_condition(df)


def generate_auto_strategy(df: pd.DataFrame, symbol: str, risk_profile: str = 'متعادل') -> Dict[str, Any]:
    """
    تولید استراتژی خودکار معاملاتی
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
        risk_profile (str): پروفایل ریسک کاربر
        
    Returns:
        dict: استراتژی معاملاتی تولید شده
    """
    manager = AutoStrategyManager(risk_profile=risk_profile)
    return manager.generate_strategy(symbol, df)


def compare_trading_strategies(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    مقایسه استراتژی‌های مختلف معاملاتی
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
        
    Returns:
        dict: نتایج مقایسه استراتژی‌ها
    """
    manager = AutoStrategyManager()
    return manager.compare_strategies(df, symbol)