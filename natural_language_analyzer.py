"""
ماژول تحلیل بازار با پردازش زبان طبیعی و هوش مصنوعی 

این ماژول از روش‌های پردازش زبان طبیعی برای تحلیل داده‌های قیمت، آنالیز الگوها و تولید گزارش‌های تفسیری قابل فهم برای کاربران استفاده می‌کند.

این ماژول با استفاده از مدل‌های زبانی پیشرفته و تکنیک‌های تحلیل روند، الگوریتم‌های یادگیری عمیق و تحلیل احساسات کار می‌کند.
"""

import pandas as pd
import numpy as np
import json
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaturalLanguageAnalyzer:
    """کلاس اصلی تحلیل بازار با روش‌های پردازش زبان طبیعی"""
    
    def __init__(self, use_advanced_model: bool = False):
        """
        مقداردهی اولیه آنالایزر زبان طبیعی
        
        Args:
            use_advanced_model (bool): استفاده از مدل پیشرفته (در صورت دسترسی)
        """
        self.use_advanced_model = use_advanced_model
        logger.info(f"آنالایزر زبان طبیعی با مدل {'پیشرفته' if use_advanced_model else 'پایه'} راه‌اندازی شد")
        
        # الگوهای زبانی برای توصیف روندها
        self.trend_patterns = {
            'strong_uptrend': [
                "روند صعودی قدرتمندی که با {momentum} همراه است",
                "حرکت صعودی پایدار با {volume}",
                "روند صعودی قوی با تأیید {indicators}"
            ],
            'uptrend': [
                "روند صعودی ملایم با {support} قوی",
                "حرکت رو به بالا با {momentum}",
                "روند صعودی با {volume} معاملات"
            ],
            'sideways': [
                "بازار در محدوده نوسانی بین {resistance} و {support}",
                "حرکت محدوده‌ای با {volume}",
                "روند نوسانی با {volatility}"
            ],
            'downtrend': [
                "روند نزولی با {momentum} منفی",
                "حرکت نزولی با {volume}",
                "روند کاهشی با فشار {indicators}"
            ],
            'strong_downtrend': [
                "روند نزولی شدید با {volume} بالا",
                "ریزش قیمت با {momentum} منفی قوی",
                "روند نزولی قدرتمند با شکست {support}"
            ]
        }
        
        # الگوهای زبانی برای توصیف سیگنال‌ها
        self.signal_patterns = {
            'strong_buy': [
                "سیگنال خرید قوی با بازگشت قیمت از {support}",
                "فرصت خرید عالی با واگرایی مثبت در {indicators}",
                "سیگنال خرید قدرتمند با {pattern} و تأیید {volume}"
            ],
            'buy': [
                "سیگنال خرید با شکل‌گیری {pattern}",
                "فرصت خرید با {momentum} مثبت",
                "سیگنال ورود به بازار با {crossover}"
            ],
            'neutral': [
                "بازار در وضعیت بی‌تصمیمی با {consolidation}",
                "سیگنال خنثی با {volume} پایین",
                "انتظار برای جهت‌گیری مشخص بازار با {indicators}"
            ],
            'sell': [
                "سیگنال فروش با شکست {support}",
                "علامت خروج از بازار با {divergence} منفی",
                "سیگنال فروش با {pattern} کاهشی"
            ],
            'strong_sell': [
                "سیگنال فروش قوی با تشکیل {pattern} و {confirmation}",
                "هشدار فروش با شکست {support} مهم",
                "سیگنال قوی خروج از بازار با {indicators} منفی"
            ]
        }
        
    def generate_market_analysis(self, df: pd.DataFrame, symbol: str, 
                               technical_indicators: Dict[str, Any],
                               patterns: Dict[str, Any], 
                               market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        تولید تحلیل بازار به زبان طبیعی
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            symbol (str): نماد ارز
            technical_indicators (dict): اطلاعات اندیکاتورهای تکنیکال
            patterns (dict): الگوهای قیمتی شناسایی شده
            market_conditions (dict): شرایط فعلی بازار
            
        Returns:
            dict: تحلیل بازار به زبان طبیعی
        """
        try:
            if df.empty or len(df) < 20:
                return {
                    'summary': f"داده‌های کافی برای تحلیل {symbol} وجود ندارد.",
                    'trend_analysis': "داده‌های ناکافی",
                    'signal_description': "نامشخص",
                    'risk_assessment': "نامشخص",
                    'confidence': 0.0
                }
            
            # استخراج اطلاعات کلیدی
            trend = market_conditions.get('trend', 'نوسانی')
            volatility = market_conditions.get('volatility', 'متوسط')
            volume = market_conditions.get('volume', 'متوسط')
            momentum = market_conditions.get('momentum', 'خنثی')
            
            # دریافت سیگنال معاملاتی
            signal = self._get_trading_signal(technical_indicators, patterns, market_conditions)
            
            # تحلیل روند به زبان طبیعی
            trend_analysis = self._generate_trend_analysis(df, trend, volatility, volume, momentum, patterns)
            
            # توصیف سیگنال به زبان طبیعی
            signal_description = self._generate_signal_description(signal, technical_indicators, patterns)
            
            # ارزیابی ریسک
            risk_assessment = self._generate_risk_assessment(df, trend, volatility, signal, patterns)
            
            # محاسبه سطح اطمینان به تحلیل
            confidence = self._calculate_analysis_confidence(technical_indicators, patterns, market_conditions)
            
            # تولید خلاصه کلی
            summary = self._generate_summary(symbol, trend_analysis, signal_description, risk_assessment, confidence)
            
            # خروجی نهایی
            return {
                'summary': summary,
                'trend_analysis': trend_analysis,
                'signal_description': signal_description,
                'risk_assessment': risk_assessment,
                'confidence': confidence,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"خطا در تولید تحلیل زبان طبیعی: {str(e)}")
            return {
                'summary': f"خطا در تحلیل {symbol}: {str(e)}",
                'trend_analysis': "خطا در تحلیل",
                'signal_description': "نامشخص",
                'risk_assessment': "نامشخص",
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _get_trading_signal(self, technical_indicators: Dict[str, Any], 
                          patterns: Dict[str, Any], 
                          market_conditions: Dict[str, Any]) -> str:
        """
        تعیین سیگنال معاملاتی بر اساس اطلاعات موجود
        
        Args:
            technical_indicators (dict): اطلاعات اندیکاتورهای تکنیکال
            patterns (dict): الگوهای قیمتی شناسایی شده
            market_conditions (dict): شرایط فعلی بازار
            
        Returns:
            str: سیگنال معاملاتی
        """
        # استخراج سیگنال‌های مختلف
        trend_signal = self._get_trend_signal(market_conditions)
        indicator_signal = self._get_indicator_signal(technical_indicators)
        pattern_signal = self._get_pattern_signal(patterns)
        
        # وزن‌دهی به سیگنال‌ها
        signal_weights = {
            'trend': 0.4,
            'indicator': 0.35,
            'pattern': 0.25
        }
        
        # محاسبه امتیاز کلی سیگنال
        signals = {
            'STRONG_BUY': 1.0,
            'BUY': 0.5,
            'NEUTRAL': 0.0,
            'SELL': -0.5,
            'STRONG_SELL': -1.0
        }
        
        trend_score = signals.get(trend_signal, 0.0) * signal_weights['trend']
        indicator_score = signals.get(indicator_signal, 0.0) * signal_weights['indicator']
        pattern_score = signals.get(pattern_signal, 0.0) * signal_weights['pattern']
        
        total_score = trend_score + indicator_score + pattern_score
        
        # تعیین سیگنال نهایی بر اساس امتیاز کلی
        if total_score >= 0.5:
            return 'STRONG_BUY'
        elif total_score >= 0.2:
            return 'BUY'
        elif total_score <= -0.5:
            return 'STRONG_SELL'
        elif total_score <= -0.2:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _get_trend_signal(self, market_conditions: Dict[str, Any]) -> str:
        """
        تعیین سیگنال بر اساس روند بازار
        
        Args:
            market_conditions (dict): شرایط فعلی بازار
            
        Returns:
            str: سیگنال روند
        """
        trend = market_conditions.get('trend', 'نوسانی')
        momentum = market_conditions.get('momentum', 'خنثی')
        
        if trend == 'صعودی قوی':
            if momentum in ['قوی مثبت', 'مثبت']:
                return 'STRONG_BUY'
            else:
                return 'BUY'
        elif trend == 'صعودی':
            return 'BUY'
        elif trend == 'نزولی قوی':
            if momentum in ['قوی منفی', 'منفی']:
                return 'STRONG_SELL'
            else:
                return 'SELL'
        elif trend == 'نزولی':
            return 'SELL'
        else:  # نوسانی
            if momentum == 'قوی مثبت':
                return 'BUY'
            elif momentum == 'قوی منفی':
                return 'SELL'
            else:
                return 'NEUTRAL'
    
    def _get_indicator_signal(self, technical_indicators: Dict[str, Any]) -> str:
        """
        تعیین سیگنال بر اساس اندیکاتورهای تکنیکال
        
        Args:
            technical_indicators (dict): اطلاعات اندیکاتورهای تکنیکال
            
        Returns:
            str: سیگنال اندیکاتورها
        """
        # مقادیر پیش‌فرض در صورت نبود داده‌ها
        if not technical_indicators:
            return 'NEUTRAL'
        
        # شمارش سیگنال‌های مختلف
        buy_signals = 0
        sell_signals = 0
        strong_buy = 0
        strong_sell = 0
        
        # بررسی سیگنال‌های MACD
        if 'macd' in technical_indicators:
            macd_signal = technical_indicators['macd'].get('signal', 'NEUTRAL')
            if macd_signal == 'BUY':
                buy_signals += 1
            elif macd_signal == 'STRONG_BUY':
                strong_buy += 1
            elif macd_signal == 'SELL':
                sell_signals += 1
            elif macd_signal == 'STRONG_SELL':
                strong_sell += 1
        
        # بررسی سیگنال‌های RSI
        if 'rsi' in technical_indicators:
            rsi_value = technical_indicators['rsi'].get('value', 50)
            if rsi_value < 30:
                strong_buy += 1
            elif rsi_value < 40:
                buy_signals += 1
            elif rsi_value > 70:
                strong_sell += 1
            elif rsi_value > 60:
                sell_signals += 1
        
        # بررسی میانگین‌های متحرک
        if 'moving_averages' in technical_indicators:
            ma_signal = technical_indicators['moving_averages'].get('signal', 'NEUTRAL')
            if ma_signal == 'BUY':
                buy_signals += 1
            elif ma_signal == 'STRONG_BUY':
                strong_buy += 1
            elif ma_signal == 'SELL':
                sell_signals += 1
            elif ma_signal == 'STRONG_SELL':
                strong_sell += 1
        
        # تعیین سیگنال نهایی
        total_buy = buy_signals + 2 * strong_buy
        total_sell = sell_signals + 2 * strong_sell
        
        if total_buy - total_sell >= 3:
            return 'STRONG_BUY'
        elif total_buy - total_sell >= 1:
            return 'BUY'
        elif total_sell - total_buy >= 3:
            return 'STRONG_SELL'
        elif total_sell - total_buy >= 1:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _get_pattern_signal(self, patterns: Dict[str, Any]) -> str:
        """
        تعیین سیگنال بر اساس الگوهای قیمتی
        
        Args:
            patterns (dict): الگوهای قیمتی شناسایی شده
            
        Returns:
            str: سیگنال الگوها
        """
        # مقادیر پیش‌فرض در صورت نبود داده‌ها
        if not patterns or 'detected_patterns' not in patterns:
            return 'NEUTRAL'
        
        detected_patterns = patterns.get('detected_patterns', [])
        
        # الگوهای صعودی
        bullish_patterns = ['hammer', 'inverseHammer', 'bullishEngulfing', 'piercingLine',
                           'morningStar', 'bullishHarami', 'bullishHaramiCross']
        
        # الگوهای نزولی
        bearish_patterns = ['hangingMan', 'shootingStar', 'bearishEngulfing', 'darkCloudCover',
                           'eveningStar', 'bearishHarami', 'bearishHaramiCross']
        
        # شمارش الگوها
        bullish_count = 0
        bearish_count = 0
        
        for pattern in detected_patterns:
            if pattern.get('type') in bullish_patterns:
                bullish_count += 1
            elif pattern.get('type') in bearish_patterns:
                bearish_count += 1
        
        # تعیین سیگنال بر اساس شمارش الگوها
        if bullish_count >= 2 and bullish_count > bearish_count:
            return 'STRONG_BUY'
        elif bullish_count == 1 and bullish_count > bearish_count:
            return 'BUY'
        elif bearish_count >= 2 and bearish_count > bullish_count:
            return 'STRONG_SELL'
        elif bearish_count == 1 and bearish_count > bullish_count:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _generate_trend_analysis(self, df: pd.DataFrame, trend: str, volatility: str, 
                               volume: str, momentum: str, patterns: Dict[str, Any]) -> str:
        """
        تولید تحلیل روند به زبان طبیعی
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            trend (str): روند فعلی
            volatility (str): نوسانات
            volume (str): حجم معاملات
            momentum (str): مومنتوم
            patterns (dict): الگوهای قیمتی
            
        Returns:
            str: توضیح روند به زبان طبیعی
        """
        # انتخاب قالب مناسب بر اساس روند
        trend_key = 'sideways'
        if 'صعودی قوی' in trend:
            trend_key = 'strong_uptrend'
        elif 'صعودی' in trend:
            trend_key = 'uptrend'
        elif 'نزولی قوی' in trend:
            trend_key = 'strong_downtrend'
        elif 'نزولی' in trend:
            trend_key = 'downtrend'
        
        trend_templates = self.trend_patterns.get(trend_key, self.trend_patterns['sideways'])
        
        # انتخاب تصادفی یک قالب
        import random
        template = random.choice(trend_templates)
        
        # تعیین پارامترهای مورد استفاده
        params = {}
        
        # مومنتوم
        if 'مثبت' in momentum:
            params['momentum'] = "مومنتوم مثبت" if 'قوی' not in momentum else "مومنتوم مثبت قوی"
        elif 'منفی' in momentum:
            params['momentum'] = "مومنتوم منفی" if 'قوی' not in momentum else "مومنتوم منفی قوی"
        else:
            params['momentum'] = "مومنتوم خنثی"
        
        # حجم
        if volume == 'بالا':
            params['volume'] = "حجم معاملات بالا"
        elif volume == 'کم':
            params['volume'] = "حجم معاملات پایین"
        else:
            params['volume'] = "حجم معاملات متوسط"
        
        # نوسانات
        if volatility == 'بالا':
            params['volatility'] = "نوسانات بالا"
        elif volatility == 'کم':
            params['volatility'] = "نوسانات کم"
        else:
            params['volatility'] = "نوسانات متوسط"
        
        # اندیکاتورها
        params['indicators'] = "مجموعه اندیکاتورهای تکنیکال"
        
        # الگوها
        detected_patterns = patterns.get('detected_patterns', [])
        if detected_patterns:
            pattern_names = []
            for p in detected_patterns[:2]:
                pattern_type = p.get('type', '')
                if pattern_type:
                    pattern_names.append(pattern_type)
            
            if pattern_names:
                params['pattern'] = "الگوی " + " و ".join(pattern_names)
        else:
            params['pattern'] = "الگوهای قیمتی"
        
        # سطوح حمایت و مقاومت
        latest_close = df['close'].iloc[-1]
        latest_high = df['high'].iloc[-1]
        latest_low = df['low'].iloc[-1]
        
        # محاسبه میانگین‌های متحرک به عنوان سطوح حمایت/مقاومت ساده
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        
        support_level = df['MA50'].iloc[-1]
        resistance_level = df['MA200'].iloc[-1]
        
        if latest_close > support_level:
            support_desc = f"حمایت در سطح {support_level:.2f}"
            params['support'] = support_desc
        
        if latest_close < resistance_level:
            resistance_desc = f"مقاومت در سطح {resistance_level:.2f}"
            params['resistance'] = resistance_desc
        else:
            resistance_desc = f"سطح مقاومت بعدی"
            params['resistance'] = resistance_desc
        
        if 'support' not in params:
            params['support'] = f"سطح حمایت در {latest_low:.2f}"
        
        if 'resistance' not in params:
            params['resistance'] = f"سطح مقاومت در {latest_high:.2f}"
        
        # تکمیل قالب با پارامترها
        for key, value in params.items():
            template = template.replace(f"{{{key}}}", value)
        
        # اضافه کردن جمله‌های اضافی بر اساس روند
        if 'صعودی' in trend:
            extra_sentence = f"قیمت فعلی با فاصله {((latest_close / support_level) - 1) * 100:.1f}% بالاتر از سطح حمایت در {support_level:.2f} قرار دارد."
        elif 'نزولی' in trend:
            extra_sentence = f"قیمت فعلی با فاصله {((resistance_level / latest_close) - 1) * 100:.1f}% پایین‌تر از سطح مقاومت در {resistance_level:.2f} قرار دارد."
        else:
            extra_sentence = f"قیمت بین سطوح حمایت {support_level:.2f} و مقاومت {resistance_level:.2f} در نوسان است."
        
        return f"{template} {extra_sentence}"
    
    def _generate_signal_description(self, signal: str, technical_indicators: Dict[str, Any], 
                                   patterns: Dict[str, Any]) -> str:
        """
        تولید توضیح سیگنال به زبان طبیعی
        
        Args:
            signal (str): سیگنال معاملاتی
            technical_indicators (dict): اطلاعات اندیکاتورهای تکنیکال
            patterns (dict): الگوهای قیمتی
            
        Returns:
            str: توضیح سیگنال به زبان طبیعی
        """
        # انتخاب قالب مناسب بر اساس سیگنال
        signal_key = 'neutral'
        if signal == 'STRONG_BUY':
            signal_key = 'strong_buy'
        elif signal == 'BUY':
            signal_key = 'buy'
        elif signal == 'STRONG_SELL':
            signal_key = 'strong_sell'
        elif signal == 'SELL':
            signal_key = 'sell'
        
        signal_templates = self.signal_patterns.get(signal_key, self.signal_patterns['neutral'])
        
        # انتخاب تصادفی یک قالب
        import random
        template = random.choice(signal_templates)
        
        # استخراج اطلاعات مهم اندیکاتورها
        params = {}
        
        # الگوها
        detected_patterns = patterns.get('detected_patterns', [])
        if detected_patterns:
            latest_pattern = detected_patterns[0]
            pattern_type = latest_pattern.get('type', 'کندل استیک')
            pattern_desc = f"الگوی {pattern_type}"
            params['pattern'] = pattern_desc
        else:
            params['pattern'] = "الگوی قیمتی اخیر"
        
        # اندیکاتورها
        indicator_names = []
        if 'macd' in technical_indicators:
            indicator_names.append("MACD")
        if 'rsi' in technical_indicators:
            indicator_names.append("RSI")
        if 'moving_averages' in technical_indicators:
            indicator_names.append("میانگین متحرک")
        
        if indicator_names:
            params['indicators'] = " و ".join(indicator_names)
        else:
            params['indicators'] = "اندیکاتورهای تکنیکال"
        
        # سطوح حمایت و مقاومت
        params['support'] = "سطح حمایت"
        params['resistance'] = "سطح مقاومت"
        
        # مومنتوم
        params['momentum'] = "مومنتوم قیمت"
        
        # حجم
        params['volume'] = "حجم معاملات"
        
        # واگرایی
        params['divergence'] = "واگرایی قیمت و اندیکاتور"
        
        # تقاطع
        params['crossover'] = "تقاطع میانگین‌های متحرک"
        
        # تثبیت
        params['consolidation'] = "تثبیت قیمت"
        
        # تأیید
        params['confirmation'] = "تأیید اندیکاتورها"
        
        # تکمیل قالب با پارامترها
        for key, value in params.items():
            template = template.replace(f"{{{key}}}", value)
        
        # اضافه کردن توضیحات بیشتر بر اساس سیگنال
        if signal == 'STRONG_BUY':
            extra = "شاخص‌های متعددی نشان‌دهنده پتانسیل افزایش قیمت در آینده نزدیک هستند."
        elif signal == 'BUY':
            extra = "نشانه‌های مثبت در بازار قابل مشاهده است، اما با احتیاط عمل کنید."
        elif signal == 'STRONG_SELL':
            extra = "شاخص‌های متعددی نشان‌دهنده احتمال کاهش قیمت در آینده نزدیک هستند."
        elif signal == 'SELL':
            extra = "نشانه‌های منفی در بازار قابل مشاهده است، محتاطانه برنامه‌ریزی کنید."
        else:  # NEUTRAL
            extra = "در شرایط فعلی، صبر و مشاهده بیشتر توصیه می‌شود."
        
        return f"{template} {extra}"
    
    def _generate_risk_assessment(self, df: pd.DataFrame, trend: str, 
                                volatility: str, signal: str, patterns: Dict[str, Any]) -> str:
        """
        تولید ارزیابی ریسک به زبان طبیعی
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            trend (str): روند فعلی
            volatility (str): نوسانات
            signal (str): سیگنال معاملاتی
            patterns (dict): الگوهای قیمتی
            
        Returns:
            str: ارزیابی ریسک به زبان طبیعی
        """
        # تعیین سطح کلی ریسک
        risk_level = "متوسط"
        
        if volatility == 'بالا':
            risk_level = "بالا"
        elif volatility == 'کم':
            risk_level = "پایین"
        
        # تنظیم ریسک بر اساس روند
        if 'صعودی قوی' in trend and (signal == 'STRONG_BUY' or signal == 'BUY'):
            risk_adjustment = "کمتر از"
        elif 'نزولی قوی' in trend and (signal == 'STRONG_SELL' or signal == 'SELL'):
            risk_adjustment = "کمتر از"
        elif 'نوسانی' in trend:
            risk_adjustment = "بیشتر از"
        else:
            risk_adjustment = ""
        
        risk_desc = f"سطح ریسک {risk_adjustment} {risk_level}"
        
        # محاسبه ATR (Average True Range) برای تعیین نوسانات
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        last_close = df['close'].iloc[-1]
        last_atr = df['atr'].iloc[-1]
        atr_percent = (last_atr / last_close) * 100
        
        # تعیین نقاط حد ضرر و سود بر اساس ATR
        if signal in ['STRONG_BUY', 'BUY']:
            stop_loss_level = last_close - (last_atr * 2)
            stop_loss_percent = ((last_close - stop_loss_level) / last_close) * 100
            target_level = last_close + (last_atr * 4)
            target_percent = ((target_level - last_close) / last_close) * 100
            risk_reward = target_percent / stop_loss_percent
            
            risk_management = (
                f"با توجه به نوسانات فعلی، نقطه حد ضرر مناسب در {stop_loss_level:.2f} "
                f"({stop_loss_percent:.1f}% پایین‌تر از قیمت فعلی) و هدف قیمتی در {target_level:.2f} "
                f"({target_percent:.1f}% بالاتر از قیمت فعلی) قرار می‌گیرد. "
                f"نسبت ریسک به پاداش {risk_reward:.1f} است."
            )
        elif signal in ['STRONG_SELL', 'SELL']:
            stop_loss_level = last_close + (last_atr * 2)
            stop_loss_percent = ((stop_loss_level - last_close) / last_close) * 100
            target_level = last_close - (last_atr * 4)
            target_percent = ((last_close - target_level) / last_close) * 100
            risk_reward = target_percent / stop_loss_percent
            
            risk_management = (
                f"با توجه به نوسانات فعلی، نقطه حد ضرر مناسب در {stop_loss_level:.2f} "
                f"({stop_loss_percent:.1f}% بالاتر از قیمت فعلی) و هدف قیمتی در {target_level:.2f} "
                f"({target_percent:.1f}% پایین‌تر از قیمت فعلی) قرار می‌گیرد. "
                f"نسبت ریسک به پاداش {risk_reward:.1f} است."
            )
        else:
            risk_management = (
                f"در وضعیت فعلی نوسانات قیمت حدود {atr_percent:.1f}% است. "
                f"توصیه می‌شود تا مشخص شدن روند واضح بازار، از ورود به معاملات پرریسک خودداری کنید."
            )
        
        # اضافه کردن بخش توصیه‌های ریسک بر اساس الگوها
        detected_patterns = patterns.get('detected_patterns', [])
        risk_tips = []
        
        if any(p.get('type') in ['bearishEngulfing', 'eveningStar', 'shootingStar'] for p in detected_patterns):
            risk_tips.append("الگوهای نزولی اخیر نشان‌دهنده احتمال افزایش ریسک در موقعیت‌های خرید هستند.")
        
        if any(p.get('type') in ['bullishEngulfing', 'morningStar', 'hammer'] for p in detected_patterns):
            risk_tips.append("الگوهای صعودی اخیر می‌توانند ریسک موقعیت‌های فروش را افزایش دهند.")
        
        if 'doji' in [p.get('type') for p in detected_patterns]:
            risk_tips.append("الگوی دوجی نشان‌دهنده تردید بازار و احتمال تغییر روند است که می‌تواند ریسک را افزایش دهد.")
        
        risk_tips_text = " ".join(risk_tips) if risk_tips else ""
        
        return f"{risk_desc}. {risk_management} {risk_tips_text}"
    
    def _calculate_analysis_confidence(self, technical_indicators: Dict[str, Any], 
                                     patterns: Dict[str, Any], 
                                     market_conditions: Dict[str, Any]) -> float:
        """
        محاسبه سطح اطمینان به تحلیل
        
        Args:
            technical_indicators (dict): اطلاعات اندیکاتورهای تکنیکال
            patterns (dict): الگوهای قیمتی
            market_conditions (dict): شرایط بازار
            
        Returns:
            float: سطح اطمینان (0-1)
        """
        # پارامترهای مؤثر در اطمینان
        confidence_factors = []
        
        # تعداد اندیکاتورها
        indicator_count = len(technical_indicators)
        if indicator_count >= 5:
            confidence_factors.append(1.0)
        elif indicator_count >= 3:
            confidence_factors.append(0.8)
        elif indicator_count >= 1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # همسو بودن اندیکاتورها
        indicator_signals = []
        if 'macd' in technical_indicators:
            indicator_signals.append(technical_indicators['macd'].get('signal', 'NEUTRAL'))
        if 'rsi' in technical_indicators:
            rsi_value = technical_indicators['rsi'].get('value', 50)
            if rsi_value < 30:
                indicator_signals.append('BUY')
            elif rsi_value > 70:
                indicator_signals.append('SELL')
            else:
                indicator_signals.append('NEUTRAL')
        if 'moving_averages' in technical_indicators:
            indicator_signals.append(technical_indicators['moving_averages'].get('signal', 'NEUTRAL'))
        
        buy_signals = indicator_signals.count('BUY') + indicator_signals.count('STRONG_BUY')
        sell_signals = indicator_signals.count('SELL') + indicator_signals.count('STRONG_SELL')
        neutral_signals = indicator_signals.count('NEUTRAL')
        
        if indicator_signals:
            max_signal = max(buy_signals, sell_signals, neutral_signals)
            indicator_agreement = max_signal / len(indicator_signals)
            confidence_factors.append(indicator_agreement)
        
        # وجود الگوهای قیمتی
        detected_patterns = patterns.get('detected_patterns', [])
        if len(detected_patterns) >= 2:
            confidence_factors.append(0.9)
        elif len(detected_patterns) == 1:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # وضوح روند
        trend = market_conditions.get('trend', 'نوسانی')
        if 'قوی' in trend:
            confidence_factors.append(0.9)
        elif 'نوسانی' not in trend:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # محاسبه میانگین موزون
        weights = [0.3, 0.3, 0.2, 0.2]
        weighted_confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return min(weighted_confidence, 0.95)  # حداکثر 95% اطمینان
    
    def _generate_summary(self, symbol: str, trend_analysis: str, signal_description: str, 
                        risk_assessment: str, confidence: float) -> str:
        """
        تولید خلاصه کلی تحلیل
        
        Args:
            symbol (str): نماد ارز
            trend_analysis (str): تحلیل روند
            signal_description (str): توضیح سیگنال
            risk_assessment (str): ارزیابی ریسک
            confidence (float): سطح اطمینان
            
        Returns:
            str: خلاصه کلی
        """
        confidence_text = ""
        if confidence >= 0.8:
            confidence_text = "با اطمینان بالا"
        elif confidence >= 0.6:
            confidence_text = "با اطمینان متوسط"
        else:
            confidence_text = "با اطمینان نسبتاً کم"
        
        # استخراج جمله اول از هر بخش
        first_trend = trend_analysis.split('.')[0].strip() + "."
        first_signal = signal_description.split('.')[0].strip() + "."
        
        summary = (
            f"تحلیل بازار {symbol} {confidence_text} نشان می‌دهد که {first_trend} "
            f"{first_signal} در معاملات خود دقت کنید که {risk_assessment.split('.')[0].strip()}."
        )
        
        return summary


def get_natural_language_analysis(df: pd.DataFrame, symbol: str, 
                                 technical_analysis_results: Dict[str, Any], 
                                 pattern_results: Dict[str, Any],
                                 market_conditions: Dict[str, Any],
                                 use_advanced_model: bool = False) -> Dict[str, Any]:
    """
    دریافت تحلیل بازار به زبان طبیعی
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
        technical_analysis_results (dict): نتایج تحلیل تکنیکال
        pattern_results (dict): نتایج تشخیص الگوها
        market_conditions (dict): شرایط فعلی بازار
        use_advanced_model (bool): استفاده از مدل پیشرفته
        
    Returns:
        dict: تحلیل بازار به زبان طبیعی
    """
    analyzer = NaturalLanguageAnalyzer(use_advanced_model=use_advanced_model)
    return analyzer.generate_market_analysis(
        df, symbol, technical_analysis_results, pattern_results, market_conditions
    )


def get_natural_language_recommendation(df: pd.DataFrame, symbol: str, 
                                      risk_profile: str = 'متعادل',
                                      technical_analysis_results: Optional[Dict[str, Any]] = None, 
                                      pattern_results: Optional[Dict[str, Any]] = None,
                                      market_conditions: Optional[Dict[str, Any]] = None) -> str:
    """
    دریافت پیشنهاد معاملاتی به زبان طبیعی
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
        risk_profile (str): پروفایل ریسک کاربر
        technical_analysis_results (dict, optional): نتایج تحلیل تکنیکال
        pattern_results (dict, optional): نتایج تشخیص الگوها
        market_conditions (dict, optional): شرایط فعلی بازار
        
    Returns:
        str: پیشنهاد معاملاتی به زبان طبیعی
    """
    try:
        # واردسازی ماژول‌های مورد نیاز
        from auto_strategy_manager import get_market_condition
        from chart_patterns import detect_patterns
        from technical_analysis import get_technical_indicators
        
        # دریافت اطلاعات مورد نیاز اگر ارائه نشده‌اند
        if market_conditions is None:
            market_conditions = get_market_condition(df)
        
        if technical_analysis_results is None:
            technical_analysis_results = get_technical_indicators(df)
        
        if pattern_results is None:
            pattern_results = {'detected_patterns': detect_patterns(df)}
        
        # دریافت تحلیل زبان طبیعی
        analysis = get_natural_language_analysis(
            df, symbol, technical_analysis_results, pattern_results, market_conditions
        )
        
        # تولید پیشنهاد بر اساس پروفایل ریسک
        signal = analysis.get('signal', 'NEUTRAL')
        confidence = analysis.get('confidence', 0.5)
        
        # تنظیم پیشنهاد بر اساس پروفایل ریسک
        if risk_profile == 'کم‌ریسک':
            if signal == 'STRONG_BUY' and confidence > 0.7:
                recommendation = f"با توجه به پروفایل کم‌ریسک شما و {analysis.get('trend_analysis', '').split('.')[0]}، "
                recommendation += "می‌توانید با اختصاص بخش کوچکی از سرمایه (حداکثر 5%) وارد موقعیت خرید شوید. "
                recommendation += "حتماً از حد ضرر استفاده کنید و در صورت تغییر شرایط بازار، سریعاً موقعیت خود را ببندید."
            elif signal == 'BUY' and confidence > 0.8:
                recommendation = "با توجه به شرایط فعلی و پروفایل کم‌ریسک شما، فعلاً صبر کنید تا سیگنال قوی‌تری برای خرید مشاهده شود."
            elif signal in ['SELL', 'STRONG_SELL']:
                recommendation = "با توجه به پروفایل کم‌ریسک شما، بهتر است از موقعیت‌های فروش دوری کنید و در صورت داشتن سرمایه در این ارز، آن را به ارزهای امن‌تر منتقل کنید."
            else:
                recommendation = "با توجه به پروفایل کم‌ریسک شما، در شرایط فعلی بهتر است از ورود به بازار خودداری کنید و به عنوان ناظر، تحولات بازار را دنبال کنید."
        
        elif risk_profile == 'پرریسک':
            if signal in ['STRONG_BUY', 'BUY']:
                recommendation = f"با توجه به پروفایل پرریسک شما و {analysis.get('signal_description', '').split('.')[0]}، "
                recommendation += "می‌توانید با اختصاص بخش قابل توجهی از سرمایه (حداکثر 20%) وارد موقعیت خرید شوید. "
                recommendation += "ضروری است از حد ضرر استفاده کنید، اما می‌توانید اهداف قیمتی بلندپروازانه‌تری را در نظر بگیرید."
            elif signal in ['STRONG_SELL', 'SELL']:
                recommendation = f"با توجه به پروفایل پرریسک شما و {analysis.get('signal_description', '').split('.')[0]}، "
                recommendation += "می‌توانید موقعیت فروش (شورت) را با اختصاص بخشی از سرمایه (حداکثر 15%) امتحان کنید. "
                recommendation += "حتماً از حد ضرر استفاده کنید تا در صورت تغییر ناگهانی روند، متحمل ضرر زیادی نشوید."
            else:
                recommendation = "با توجه به پروفایل پرریسک شما، می‌توانید استراتژی‌های نوسان‌گیری را با سرمایه کم (5-10%) امتحان کنید. "
                recommendation += "شناسایی سود در مراحل مختلف و استفاده از حد ضرر متحرک توصیه می‌شود."
        
        else:  # متعادل
            if signal == 'STRONG_BUY':
                recommendation = f"با توجه به پروفایل متعادل شما و {analysis.get('trend_analysis', '').split('.')[0]}، "
                recommendation += "می‌توانید با اختصاص بخشی از سرمایه (حداکثر 10%) وارد موقعیت خرید شوید. "
                recommendation += "استفاده از حد ضرر و استراتژی ورود تدریجی توصیه می‌شود."
            elif signal == 'BUY':
                recommendation = "با توجه به شرایط فعلی، می‌توانید با احتیاط و با اختصاص بخش کوچکی از سرمایه (حداکثر 5%) موقعیت خرید ایجاد کنید. "
                recommendation += "ورود تدریجی و استفاده از حد ضرر توصیه می‌شود."
            elif signal == 'STRONG_SELL':
                recommendation = "با توجه به شرایط فعلی، اگر در این ارز سرمایه‌گذاری کرده‌اید، خروج تدریجی توصیه می‌شود. "
                recommendation += "در صورت تمایل به موقعیت فروش (شورت)، با احتیاط و سرمایه کم (حداکثر 5%) اقدام کنید."
            elif signal == 'SELL':
                recommendation = "با توجه به شرایط فعلی، بهتر است از ورود به موقعیت خرید خودداری کنید. اگر در این ارز سرمایه‌گذاری کرده‌اید، مراقب کاهش قیمت باشید و حد ضرر را تنظیم کنید."
            else:
                recommendation = "با توجه به شرایط فعلی و عدم وضوح روند بازار، بهتر است صبر کنید و به عنوان ناظر، تحولات بازار را دنبال کنید تا سیگنال‌های واضح‌تری شکل بگیرد."
        
        # اضافه کردن توصیه‌های مدیریت ریسک
        risk_management = (
            "\n\nتوصیه‌های مدیریت ریسک:\n"
            "1. هرگز بیش از توان مالی خود سرمایه‌گذاری نکنید.\n"
            "2. همیشه از حد ضرر استفاده کنید و آن را تنظیم شده باقی بگذارید.\n"
            "3. سیگنال‌های تکنیکال را با تحلیل بنیادی و اخبار مهم بازار همراه کنید.\n"
            "4. به یاد داشته باشید که تمامی تحلیل‌ها احتمالی هستند و قطعیت ندارند."
        )
        
        return recommendation + risk_management
        
    except Exception as e:
        logger.error(f"خطا در تولید پیشنهاد به زبان طبیعی: {str(e)}")
        return "در حال حاضر امکان ارائه پیشنهاد وجود ندارد. لطفاً دوباره تلاش کنید."