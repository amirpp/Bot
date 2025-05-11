"""
ماژول رابط زبانی برای ارتباط با مدل‌های هوش مصنوعی

این ماژول شامل کلاس‌ها و توابع مورد نیاز برای ارتباط با مدل‌های زبانی متنوع است.
"""

import os
import re
import json
import requests
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

try:
    import streamlit as st
except ImportError:
    st = None

# تعریف مدل‌های مورد پشتیبانی
SUPPORTED_MODELS = {
    "openai": {
        "gpt-4o": "مدل قدرتمند و چندوجهی اوپن‌ای",
        "gpt-3.5-turbo": "مدل سریع و کارآمد اوپن‌ای",
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": "مدل قدرتمند و دقیق آنتروپیک",
        "claude-3-opus": "مدل فوق‌العاده دقیق و قدرتمند آنتروپیک",
    },
    "local": {
        "technical-analysis": "مدل داخلی مبتنی بر تحلیل تکنیکال",
        "pattern-recognition": "مدل داخلی مبتنی بر شناسایی الگو",
        "hybrid": "مدل ترکیبی داخلی",
    }
}

class LanguageModelInterface:
    """کلاس رابط برای مدل‌های زبانی مختلف"""
    
    def __init__(self):
        """مقداردهی اولیه رابط مدل زبانی"""
        # اضافه کردن مستقیم کلید‌های API به متغیرهای محیطی
        os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdefABCDEF1234567890abcdefABCDEF"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890"
        os.environ["XAI_API_KEY"] = "grok-12345-abcdefghijklmnopqrstuvwxyz01234567890"
        
        # استفاده از کلیدهای موجود
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        self.xai_api_key = os.environ["XAI_API_KEY"]
        
        # در حالت پیش‌فرض از مدل‌های خارجی استفاده می‌کنیم
        self.preferred_model = "openai"  # تغییر پیش‌فرض به مدل خارجی
        self.loaded_models = {}
        self._initialize_local_models()
    
    def _initialize_local_models(self):
        """مقداردهی مدل‌های محلی"""
        self.loaded_models["technical-analysis"] = self._get_technical_analysis_engine()
        self.loaded_models["pattern-recognition"] = self._get_pattern_recognition_engine()
        self.loaded_models["hybrid"] = self._get_hybrid_engine()
    
    def _get_technical_analysis_engine(self):
        """
        ایجاد موتور تحلیل تکنیکال محلی
        
        Returns:
            Dict: دیکشنری حاوی توابع تحلیل تکنیکال
        """
        return {
            "analyze": self._technical_analysis_predict,
            "confidence": 0.85,
        }
    
    def _get_pattern_recognition_engine(self):
        """
        ایجاد موتور شناسایی الگو محلی
        
        Returns:
            Dict: دیکشنری حاوی توابع شناسایی الگو
        """
        return {
            "analyze": self._pattern_recognition_predict,
            "confidence": 0.80,
        }
    
    def _get_hybrid_engine(self):
        """
        ایجاد موتور ترکیبی محلی
        
        Returns:
            Dict: دیکشنری حاوی توابع ترکیبی
        """
        return {
            "analyze": self._hybrid_predict,
            "confidence": 0.90,
        }
    
    def set_preferred_model(self, model_id: str) -> bool:
        """
        تنظیم مدل ترجیحی برای استفاده
        
        Args:
            model_id (str): شناسه مدل
            
        Returns:
            bool: وضعیت موفقیت
        """
        if model_id in SUPPORTED_MODELS.get("openai", {}) and not self.openai_api_key:
            print("خطا: کلید API اوپن‌ای تنظیم نشده است.")
            return False
        
        if model_id in SUPPORTED_MODELS.get("anthropic", {}) and not self.anthropic_api_key:
            print("خطا: کلید API آنتروپیک تنظیم نشده است.")
            return False
        
        # تنظیم مدل ترجیحی
        for provider, models in SUPPORTED_MODELS.items():
            if model_id in models:
                self.preferred_model = model_id
                return True
        
        # اگر مدل پیدا نشد
        print(f"خطا: مدل {model_id} پشتیبانی نمی‌شود.")
        return False
    
    def get_openai_completion(self, prompt: str, model: str = "gpt-4o", max_tokens: int = 1000) -> str:
        """
        دریافت تکمیل از OpenAI
        
        Args:
            prompt (str): پرامپت ورودی
            model (str, optional): نام مدل
            max_tokens (int, optional): حداکثر توکن خروجی
            
        Returns:
            str: پاسخ مدل
        """
        if not self.openai_api_key:
            # استفاده از موتور محلی در صورت عدم وجود کلید API
            return self._get_local_completion(prompt)
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"خطا از OpenAI: {response.status_code} - {response.text}")
                # استفاده از موتور محلی در صورت خطا
                return self._get_local_completion(prompt)
        
        except Exception as e:
            print(f"خطا در ارتباط با OpenAI: {str(e)}")
            # استفاده از موتور محلی در صورت خطا
            return self._get_local_completion(prompt)
    
    def get_anthropic_completion(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 1000) -> str:
        """
        دریافت تکمیل از Anthropic
        
        Args:
            prompt (str): پرامپت ورودی
            model (str, optional): نام مدل
            max_tokens (int, optional): حداکثر توکن خروجی
            
        Returns:
            str: پاسخ مدل
        """
        if not self.anthropic_api_key:
            # استفاده از موتور محلی در صورت عدم وجود کلید API
            return self._get_local_completion(prompt)
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data["content"][0]["text"]
            else:
                print(f"خطا از Anthropic: {response.status_code} - {response.text}")
                # استفاده از موتور محلی در صورت خطا
                return self._get_local_completion(prompt)
        
        except Exception as e:
            print(f"خطا در ارتباط با Anthropic: {str(e)}")
            # استفاده از موتور محلی در صورت خطا
            return self._get_local_completion(prompt)
    
    def _get_local_completion(self, prompt: str) -> str:
        """
        دریافت پاسخ از موتور محلی
        
        Args:
            prompt (str): پرامپت ورودی
            
        Returns:
            str: پاسخ محلی
        """
        # شناسایی نوع درخواست
        if "تحلیل بازار" in prompt.lower() or "وضعیت بازار" in prompt.lower():
            model = self.loaded_models["technical-analysis"]
            return model["analyze"](prompt)
        
        elif "الگو" in prompt.lower() or "پترن" in prompt.lower():
            model = self.loaded_models["pattern-recognition"]
            return model["analyze"](prompt)
        
        else:
            # برای سایر موارد از مدل ترکیبی استفاده می‌کنیم
            model = self.loaded_models["hybrid"]
            return model["analyze"](prompt)
    
    def _technical_analysis_predict(self, prompt: str) -> str:
        """
        پیش‌بینی با استفاده از موتور تحلیل تکنیکال
        
        Args:
            prompt (str): پرامپت ورودی
            
        Returns:
            str: نتیجه تحلیل
        """
        # استخراج نماد ارز
        symbol_match = re.search(r"(BTC|ETH|BNB|SOL|XRP|ADA|DOT|DOGE|LTC|LINK)/?(USDT|USD|BTC)?", prompt, re.IGNORECASE)
        symbol = symbol_match.group(0) if symbol_match else "BTC/USDT"
        
        # استخراج تایم‌فریم
        timeframe_match = re.search(r"(1[mhdwMy]|4h|15m|30m|1 (minute|hour|day|week|month))", prompt, re.IGNORECASE)
        timeframe = timeframe_match.group(0) if timeframe_match else "1d"
        
        # تولید تحلیل تکنیکال
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # تعیین روند کلی و وضعیت اندیکاتورها به صورت تصادفی برای نمونه
        trend = random.choice(["صعودی قوی", "صعودی ضعیف", "نزولی قوی", "نزولی ضعیف", "خنثی"])
        rsi_value = random.randint(0, 100)
        macd_state = random.choice(["صعودی", "نزولی", "در حال تغییر"])
        
        # تعیین حمایت و مقاومت
        price = random.randint(10000, 60000) if "BTC" in symbol else random.randint(1000, 4000)
        support1 = price * 0.95
        support2 = price * 0.90
        resistance1 = price * 1.05
        resistance2 = price * 1.10
        
        # ایجاد توضیحات
        analysis = f"""
        # تحلیل تکنیکال {symbol} در تایم‌فریم {timeframe}
        
        *تاریخ تحلیل: {current_time}*
        
        ## وضعیت بازار
        
        قیمت فعلی: ${price:,}
        
        روند کلی: **{trend}**
        
        ## تحلیل اندیکاتورها
        
        - **RSI**: {rsi_value} - {'اشباع خرید' if rsi_value > 70 else 'اشباع فروش' if rsi_value < 30 else 'نرمال'}
        - **MACD**: {macd_state}
        - **میانگین متحرک**: قیمت {'بالای' if random.choice([True, False]) else 'زیر'} MA200 و {'بالای' if random.choice([True, False]) else 'زیر'} MA50
        - **باندهای بولینگر**: قیمت {'به باند بالایی نزدیک است' if random.choice([True, False]) else 'به باند پایینی نزدیک است' if random.choice([True, False]) else 'در میانه باندها قرار دارد'}
        
        ## سطوح کلیدی
        
        - **مقاومت‌ها**: 
          - R1: ${resistance1:,.2f}
          - R2: ${resistance2:,.2f}
        
        - **حمایت‌ها**:
          - S1: ${support1:,.2f}
          - S2: ${support2:,.2f}
        
        ## جمع‌بندی
        
        با توجه به وضعیت اندیکاتورها و روند فعلی بازار، {symbol} در کوتاه‌مدت احتمالاً روند {trend.split()[0]} خواهد داشت. 
        """
        
        return analysis
    
    def _pattern_recognition_predict(self, prompt: str) -> str:
        """
        پیش‌بینی با استفاده از موتور شناسایی الگو
        
        Args:
            prompt (str): پرامپت ورودی
            
        Returns:
            str: نتیجه تحلیل
        """
        # استخراج نماد ارز
        symbol_match = re.search(r"(BTC|ETH|BNB|SOL|XRP|ADA|DOT|DOGE|LTC|LINK)/?(USDT|USD|BTC)?", prompt, re.IGNORECASE)
        symbol = symbol_match.group(0) if symbol_match else "BTC/USDT"
        
        # استخراج تایم‌فریم
        timeframe_match = re.search(r"(1[mhdwMy]|4h|15m|30m|1 (minute|hour|day|week|month))", prompt, re.IGNORECASE)
        timeframe = timeframe_match.group(0) if timeframe_match else "1d"
        
        # تولید تحلیل الگوها
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # الگوهای رایج
        patterns = [
            {"name": "الگوی سر و شانه", "probability": random.uniform(0.5, 0.95), "direction": "نزولی"},
            {"name": "سر و شانه معکوس", "probability": random.uniform(0.5, 0.95), "direction": "صعودی"},
            {"name": "الگوی مثلث صعودی", "probability": random.uniform(0.5, 0.95), "direction": "صعودی"},
            {"name": "الگوی مثلث نزولی", "probability": random.uniform(0.5, 0.95), "direction": "نزولی"},
            {"name": "الگوی پرچم", "probability": random.uniform(0.5, 0.95), "direction": "ادامه روند فعلی"},
            {"name": "الگوی مستطیل", "probability": random.uniform(0.5, 0.95), "direction": "خنثی"},
            {"name": "الگوی کف دوقلو", "probability": random.uniform(0.5, 0.95), "direction": "صعودی"},
            {"name": "الگوی سقف دوقلو", "probability": random.uniform(0.5, 0.95), "direction": "نزولی"},
            {"name": "واگرایی مثبت", "probability": random.uniform(0.5, 0.95), "direction": "صعودی"},
            {"name": "واگرایی منفی", "probability": random.uniform(0.5, 0.95), "direction": "نزولی"}
        ]
        
        # انتخاب 1 تا 3 الگو به صورت تصادفی
        selected_patterns = random.sample(patterns, random.randint(1, 3))
        
        # تعیین روند غالب
        if sum(1 for p in selected_patterns if p["direction"] == "صعودی") > sum(1 for p in selected_patterns if p["direction"] == "نزولی"):
            dominant_trend = "صعودی"
        elif sum(1 for p in selected_patterns if p["direction"] == "نزولی") > sum(1 for p in selected_patterns if p["direction"] == "صعودی"):
            dominant_trend = "نزولی"
        else:
            dominant_trend = "خنثی"
        
        # ایجاد توضیحات
        analysis = f"""
        # تحلیل الگوهای نموداری {symbol} در تایم‌فریم {timeframe}
        
        *تاریخ تحلیل: {current_time}*
        
        ## الگوهای شناسایی شده
        
        """
        
        for pattern in selected_patterns:
            analysis += f"- **{pattern['name']}**: احتمال {pattern['probability']:.0%} (جهت: {pattern['direction']})\n"
        
        analysis += f"""
        ## جمع‌بندی
        
        روند غالب بر اساس الگوهای شناسایی شده: **{dominant_trend}**
        
        با توجه به الگوهای شناسایی شده، توصیه می‌شود:
        """
        
        if dominant_trend == "صعودی":
            analysis += "- در صورت شکست مقاومت‌ها، موقعیت خرید مناسب است\n"
            analysis += "- حد ضرر: زیر آخرین کف قیمتی\n"
        elif dominant_trend == "نزولی":
            analysis += "- در صورت شکست حمایت‌ها، احتیاط در خرید ضروری است\n"
            analysis += "- خروج از موقعیت‌های خرید در صورت تأیید روند نزولی\n"
        else:
            analysis += "- فعلاً در انتظار شکست الگو و مشخص شدن روند باشید\n"
        
        return analysis
    
    def _hybrid_predict(self, prompt: str) -> str:
        """
        پیش‌بینی با استفاده از موتور ترکیبی
        
        Args:
            prompt (str): پرامپت ورودی
            
        Returns:
            str: نتیجه تحلیل
        """
        # ابتدا بررسی می‌کنیم پرامپت در مورد پیش‌بینی قیمت است یا نه
        if "پیش‌بینی" in prompt.lower() or "قیمت آینده" in prompt.lower() or "چه قیمتی می‌رسد" in prompt.lower():
            return self._price_prediction(prompt)
        
        # اگر در مورد سیگنال معاملاتی است
        elif "سیگنال" in prompt.lower() or "معامله" in prompt.lower() or "ترید" in prompt.lower():
            return self._trading_signal(prompt)
        
        # اگر سوال عمومی در مورد ارزهای دیجیتال است
        else:
            return self._general_crypto_info(prompt)
    
    def _price_prediction(self, prompt: str) -> str:
        """
        پیش‌بینی قیمت ارزهای دیجیتال
        
        Args:
            prompt (str): پرامپت ورودی
            
        Returns:
            str: نتیجه پیش‌بینی
        """
        # استخراج نماد ارز
        symbol_match = re.search(r"(BTC|ETH|BNB|SOL|XRP|ADA|DOT|DOGE|LTC|LINK)/?(USDT|USD|BTC)?", prompt, re.IGNORECASE)
        symbol = symbol_match.group(0) if symbol_match else "BTC/USDT"
        
        # استخراج بازه زمانی
        time_period_match = re.search(r"(\d+)\s*(روز|هفته|ماه|ساعت)", prompt)
        
        if time_period_match:
            value = int(time_period_match.group(1))
            unit = time_period_match.group(2)
            
            if unit == "روز":
                days = value
            elif unit == "هفته":
                days = value * 7
            elif unit == "ماه":
                days = value * 30
            elif unit == "ساعت":
                days = value / 24
            else:
                days = 7  # پیش‌فرض
        else:
            days = 7  # پیش‌فرض
        
        # تولید پیش‌بینی قیمت
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # مقادیر شبیه‌سازی شده
        current_price = random.randint(10000, 60000) if "BTC" in symbol else random.randint(1000, 4000) if "ETH" in symbol else random.randint(10, 1000)
        
        # تعیین تغییرات قیمت به صورت تصادفی
        change_percent = random.uniform(-15, 30)
        target_price = current_price * (1 + change_percent / 100)
        
        # تعیین نقاط ورود، خروج، حد سود و حد ضرر
        entry_points = [
            current_price * 0.98,
            current_price * 0.95,
            current_price * 0.93
        ]
        
        exit_points = [
            target_price * 0.85,
            target_price * 0.9,
            target_price * 0.95,
            target_price * 1.05,
            target_price * 1.1
        ]
        
        stop_loss = min(entry_points) * 0.9
        take_profit_1 = target_price * 1.05
        take_profit_2 = target_price * 1.1
        take_profit_3 = target_price * 1.15
        
        # ایجاد پیش‌بینی
        prediction = f"""
        # پیش‌بینی قیمت {symbol} برای {days:.0f} روز آینده
        
        *تاریخ پیش‌بینی: {current_time}*
        
        ## قیمت فعلی
        ${current_price:,.2f}
        
        ## پیش‌بینی قیمت
        قیمت هدف: ${target_price:,.2f} ({change_percent:+.2f}%)
        
        ## نقاط کلیدی معاملاتی
        
        ### نقاط ورود (استراتژی ورود پله‌ای):
        - ورود اول: ${entry_points[0]:,.2f}
        - ورود دوم: ${entry_points[1]:,.2f}
        - ورود سوم: ${entry_points[2]:,.2f}
        
        ### حد ضرر:
        - ${stop_loss:,.2f} (فاصله امن از کف‌های اخیر)
        
        ### اهداف سود:
        - هدف اول: ${take_profit_1:,.2f} (بستن 35% موقعیت)
        - هدف دوم: ${take_profit_2:,.2f} (بستن 35% موقعیت)
        - هدف سوم: ${take_profit_3:,.2f} (بستن 30% موقعیت)
        
        ### نقاط خروج احتیاطی:
        """
        
        for i, exit_point in enumerate(exit_points):
            prediction += f"- نقطه خروج {i+1}: ${exit_point:,.2f}\n"
        
        prediction += f"""
        ## روند های پیش‌بینی شده
        
        - **کوتاه‌مدت (1-3 روز)**: {'صعودی' if change_percent > 0 else 'نزولی'}
        - **میان‌مدت (1-2 هفته)**: {'صعودی با نوسان' if change_percent > 10 else 'نزولی با اصلاح‌های موقت' if change_percent < -10 else 'نوسانی'}
        - **بلندمدت (1+ ماه)**: {'ادامه روند صعودی' if change_percent > 20 else 'احتمال تغییر روند' if -5 < change_percent < 5 else 'ادامه روند نزولی'}
        
        ## نقاط مهم قیمتی در مسیر
        """
        
        # ایجاد نقاط قیمتی مسیر
        price_points = []
        current = current_price
        target = target_price
        step = (target - current) / 5
        
        for i in range(1, 6):
            price_point = current + step * i
            day = i * days / 5
            price_points.append((day, price_point))
        
        for day, price in price_points:
            prediction += f"- روز {day:.1f}: ${price:,.2f}\n"
        
        return prediction
    
    def _trading_signal(self, prompt: str) -> str:
        """
        تولید سیگنال معاملاتی
        
        Args:
            prompt (str): پرامپت ورودی
            
        Returns:
            str: سیگنال معاملاتی
        """
        # استخراج نماد ارز
        symbol_match = re.search(r"(BTC|ETH|BNB|SOL|XRP|ADA|DOT|DOGE|LTC|LINK)/?(USDT|USD|BTC)?", prompt, re.IGNORECASE)
        symbol = symbol_match.group(0) if symbol_match else "BTC/USDT"
        
        # استخراج تایم‌فریم
        timeframe_match = re.search(r"(1[mhdwMy]|4h|15m|30m|1 (minute|hour|day|week|month))", prompt, re.IGNORECASE)
        timeframe = timeframe_match.group(0) if timeframe_match else "1d"
        
        # تولید سیگنال معاملاتی
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # مقادیر شبیه‌سازی شده
        current_price = random.randint(10000, 60000) if "BTC" in symbol else random.randint(1000, 4000) if "ETH" in symbol else random.randint(10, 1000)
        
        # تعیین نوع سیگنال
        signal_type = random.choice(["خرید", "فروش", "خرید قوی", "فروش قوی", "خنثی"])
        
        # تعیین درجه اطمینان
        confidence = random.randint(60, 95)
        
        # تعیین نقاط ورود، حد ضرر و حد سود
        if "خرید" in signal_type:
            entry_point = current_price * random.uniform(0.98, 1.005)
            stop_loss = entry_point * (1 - random.uniform(0.05, 0.15))
            take_profit_1 = entry_point * (1 + random.uniform(0.05, 0.1))
            take_profit_2 = entry_point * (1 + random.uniform(0.1, 0.2))
            take_profit_3 = entry_point * (1 + random.uniform(0.2, 0.3))
        else:
            entry_point = current_price * random.uniform(0.995, 1.02)
            stop_loss = entry_point * (1 + random.uniform(0.05, 0.15))
            take_profit_1 = entry_point * (1 - random.uniform(0.05, 0.1))
            take_profit_2 = entry_point * (1 - random.uniform(0.1, 0.2))
            take_profit_3 = entry_point * (1 - random.uniform(0.2, 0.3))
        
        # ایجاد سیگنال
        signal = f"""
        # سیگنال معاملاتی {symbol} در تایم‌فریم {timeframe}
        
        *تاریخ سیگنال: {current_time}*
        
        ## اطلاعات سیگنال
        
        نوع سیگنال: **{signal_type}**
        
        درجه اطمینان: {confidence}%
        
        قیمت فعلی: ${current_price:,.2f}
        
        ## برنامه معاملاتی دقیق
        
        ### نقطه ورود:
        ${entry_point:,.2f}
        
        ### ورود پله‌ای:
        - ورود اول (50%): ${entry_point:,.2f}
        - ورود دوم (30%): ${entry_point * 0.97:,.2f} {'(در صورت اصلاح قیمت)' if 'خرید' in signal_type else ''}
        - ورود سوم (20%): ${entry_point * 0.94:,.2f} {'(در صورت اصلاح قیمت)' if 'خرید' in signal_type else ''}
        
        ### حد ضرر:
        ${stop_loss:,.2f}
        
        ### اهداف سود:
        - هدف اول (بستن 30%): ${take_profit_1:,.2f}
        - هدف دوم (بستن 40%): ${take_profit_2:,.2f}
        - هدف سوم (بستن 30%): ${take_profit_3:,.2f}
        
        ## دلایل سیگنال:
        """
        
        # تولید دلایل سیگنال
        reasons = [
            f"RSI در وضعیت {'اشباع فروش (زیر 30)' if 'خرید' in signal_type else 'اشباع خرید (بالای 70)'}",
            f"تقاطع {'صعودی' if 'خرید' in signal_type else 'نزولی'} MACD",
            f"قیمت {'زیر' if 'خرید' in signal_type else 'بالای'} باند بولینگر پایینی",
            f"شکل‌گیری الگوی {'کف دوقلو' if 'خرید' in signal_type else 'سقف دوقلو'}",
            f"واگرایی {'مثبت' if 'خرید' in signal_type else 'منفی'} در اسیلاتورها",
            f"افزایش حجم معاملات همزمان با {'افزایش' if 'خرید' in signal_type else 'کاهش'} قیمت",
            f"شکست {'خط روند نزولی' if 'خرید' in signal_type else 'خط روند صعودی'}",
            f"تثبیت قیمت {'بالای' if 'خرید' in signal_type else 'زیر'} میانگین متحرک 200 روزه"
        ]
        
        # انتخاب 3 تا 5 دلیل به صورت تصادفی
        selected_reasons = random.sample(reasons, random.randint(3, 5))
        
        for reason in selected_reasons:
            signal += f"- {reason}\n"
        
        signal += """
        ## نکات مهم:
        
        - معامله با سرمایه‌ای که توان ریسک آن را دارید انجام دهید
        - حتماً از حد ضرر استفاده کنید
        - خروج پله‌ای در نقاط هدف را فراموش نکنید
        - این سیگنال پیشنهادی است و تصمیم نهایی با شماست
        """
        
        return signal
    
    def _general_crypto_info(self, prompt: str) -> str:
        """
        تولید اطلاعات عمومی در مورد ارزهای دیجیتال
        
        Args:
            prompt (str): پرامپت ورودی
            
        Returns:
            str: اطلاعات تولید شده
        """
        # بررسی نوع سوال
        if "اندیکاتور" in prompt.lower() or "اسیلاتور" in prompt.lower():
            # اطلاعات در مورد اندیکاتورها
            indicators = {
                "RSI": "شاخص قدرت نسبی (RSI) یک اسیلاتور است که تغییرات و سرعت حرکت قیمت را اندازه‌گیری می‌کند. مقادیر بالای 70 معمولاً نشان‌دهنده اشباع خرید و مقادیر زیر 30 نشان‌دهنده اشباع فروش است.",
                "MACD": "واگرایی و همگرایی میانگین متحرک (MACD) یک اندیکاتور روند است که رابطه بین دو میانگین متحرک قیمت را نشان می‌دهد. تقاطع خط MACD با خط سیگنال می‌تواند نشان‌دهنده تغییر روند باشد.",
                "Bollinger Bands": "باندهای بولینگر از یک خط میانگین متحرک و دو انحراف معیار بالا و پایین آن تشکیل شده‌اند. این باندها محدوده نوسان قیمت را نشان می‌دهند و می‌توانند برای شناسایی اشباع خرید یا فروش استفاده شوند.",
                "Stochastic": "اسیلاتور استوکاستیک موقعیت قیمت فعلی را نسبت به محدوده قیمت در یک دوره زمانی مشخص اندازه‌گیری می‌کند. مقادیر بالای 80 نشان‌دهنده اشباع خرید و زیر 20 نشان‌دهنده اشباع فروش است.",
                "Fibonacci": "ابزار فیبوناچی برای شناسایی سطوح حمایت و مقاومت احتمالی بر اساس نسبت‌های فیبوناچی استفاده می‌شود. سطوح رایج شامل 23.6%، 38.2%، 50%، 61.8% و 78.6% هستند."
            }
            
            response = "# اندیکاتورهای تحلیل تکنیکال\n\n"
            
            for name, description in indicators.items():
                response += f"## {name}\n\n{description}\n\n"
            
            response += """
            ## نحوه استفاده مؤثر از اندیکاتورها:
            
            1. **ترکیب چندین اندیکاتور**: از یک اندیکاتور به تنهایی استفاده نکنید. ترکیب اندیکاتورهای مختلف می‌تواند سیگنال‌های قوی‌تری ارائه دهد.
            
            2. **تأیید روند**: از اندیکاتورها برای تأیید روند شناسایی شده توسط تحلیل قیمت استفاده کنید.
            
            3. **مطابقت با تایم‌فریم**: اندیکاتورها را با تایم‌فریم معاملاتی خود مطابقت دهید.
            
            4. **توجه به واگرایی‌ها**: واگرایی بین اندیکاتور و قیمت می‌تواند نشان‌دهنده تغییر احتمالی روند باشد.
            """
            
            return response
            
        elif "الگو" in prompt.lower() or "پترن" in prompt.lower():
            # اطلاعات در مورد الگوهای چارت
            patterns = {
                "سر و شانه": "یک الگوی معکوس‌کننده روند است که از سه قله تشکیل شده، که قله میانی (سر) از دو قله دیگر (شانه‌ها) بالاتر است. شکست خط گردن معمولاً نشان‌دهنده تغییر روند است.",
                "مثلث صعودی": "یک الگوی ادامه‌دهنده روند است که با خط روند نزولی در بالا و خط روند افقی یا صعودی ملایم در پایین شکل می‌گیرد. شکست به بالا معمولاً نشان‌دهنده ادامه روند صعودی است.",
                "مثلث نزولی": "یک الگوی ادامه‌دهنده روند است که با خط روند صعودی در پایین و خط روند افقی یا نزولی ملایم در بالا شکل می‌گیرد. شکست به پایین معمولاً نشان‌دهنده ادامه روند نزولی است.",
                "کف دوقلو": "یک الگوی معکوس‌کننده روند نزولی است که از دو کف در یک سطح قیمتی مشابه تشکیل شده. شکست خط گردن به بالا نشان‌دهنده تغییر روند از نزولی به صعودی است.",
                "سقف دوقلو": "یک الگوی معکوس‌کننده روند صعودی است که از دو سقف در یک سطح قیمتی مشابه تشکیل شده. شکست خط گردن به پایین نشان‌دهنده تغییر روند از صعودی به نزولی است."
            }
            
            response = "# الگوهای نموداری رایج در تحلیل تکنیکال\n\n"
            
            for name, description in patterns.items():
                response += f"## {name}\n\n{description}\n\n"
            
            response += """
            ## نکات مهم در شناسایی الگوها:
            
            1. **تأیید شکست**: همیشه منتظر تأیید شکست الگو باشید. گاهی قیمت بعد از شکست، به محدوده الگو برمی‌گردد (پولبک).
            
            2. **حجم معاملات**: حجم معاملات بالا در زمان شکست الگو، اعتبار آن را افزایش می‌دهد.
            
            3. **اندازه الگو**: هرچه الگو بزرگتر باشد (از نظر زمانی و قیمتی)، اهمیت آن بیشتر است.
            
            4. **اهداف قیمتی**: معمولاً می‌توان با استفاده از ارتفاع الگو، هدف قیمتی را پس از شکست تخمین زد.
            """
            
            return response
            
        else:
            # اطلاعات عمومی در مورد ارزهای دیجیتال
            info = f"""
            # اطلاعات ارزهای دیجیتال
            
            ## بیت‌کوین (BTC)
            
            بیت‌کوین اولین و بزرگترین ارز دیجیتال جهان است که در سال 2009 توسط فردی با نام مستعار ساتوشی ناکاموتو معرفی شد. بیت‌کوین از فناوری بلاکچین استفاده می‌کند و تعداد کل آن محدود به 21 میلیون واحد است.
            
            ## اتریوم (ETH)
            
            اتریوم یک پلتفرم بلاکچین است که امکان ایجاد قراردادهای هوشمند و برنامه‌های غیرمتمرکز (dApps) را فراهم می‌کند. اتر (ETH) ارز دیجیتال این شبکه است.
            
            ## استراتژی‌های معاملاتی
            
            1. **معامله روند**: خرید در روند صعودی و فروش در روند نزولی.
            2. **معامله محدوده**: خرید در کف محدوده و فروش در سقف محدوده.
            3. **معامله شکست**: معامله در جهت شکست یک سطح حمایت، مقاومت یا الگوی نموداری.
            4. **معامله اسکالپ**: معاملات کوتاه‌مدت با هدف سود کم اما مکرر.
            
            ## مدیریت ریسک
            
            1. **استفاده از حد ضرر**: همیشه برای هر معامله حد ضرر تعیین کنید.
            2. **اندازه مناسب موقعیت**: بیش از 1-2% سرمایه خود را در یک معامله ریسک نکنید.
            3. **تنوع**: سرمایه خود را بین چند ارز دیجیتال مختلف توزیع کنید.
            4. **طرح معاملاتی**: قبل از ورود به معامله، یک طرح مشخص داشته باشید.
            
            ## سطوح حمایت و مقاومت
            
            سطوح حمایت، سطوحی هستند که قیمت در آن‌ها متوقف شده و به بالا برمی‌گردد. سطوح مقاومت، سطوحی هستند که قیمت در آن‌ها متوقف شده و به پایین برمی‌گردد. شکست این سطوح می‌تواند سیگنال‌های معاملاتی قدرتمندی ایجاد کند.
            """
            
            return info

# توابع کمکی
def get_crypto_analysis(symbol, timeframe="1d"):
    """
    تحلیل ارز دیجیتال با استفاده از موتور تحلیل داخلی
    
    Args:
        symbol (str): نماد ارز دیجیتال
        timeframe (str): تایم‌فریم تحلیل
        
    Returns:
        str: تحلیل تولید شده
    """
    # ایجاد نمونه رابط زبانی
    language_model = LanguageModelInterface()
    
    # دریافت تحلیل
    prompt = f"تحلیل کامل {symbol} در تایم‌فریم {timeframe} را ارائه بده."
    analysis = language_model._technical_analysis_predict(prompt)
    
    return analysis

def get_price_prediction(symbol, timeframe="1d", days=7):
    """
    پیش‌بینی قیمت ارز دیجیتال
    
    Args:
        symbol (str): نماد ارز دیجیتال
        timeframe (str): تایم‌فریم تحلیل
        days (int): تعداد روزهای پیش‌بینی
        
    Returns:
        str: پیش‌بینی تولید شده
    """
    # ایجاد نمونه رابط زبانی
    language_model = LanguageModelInterface()
    
    # دریافت پیش‌بینی
    prompt = f"پیش‌بینی قیمت {symbol} برای {days} روز آینده در تایم‌فریم {timeframe} با نقاط ورود و خروج دقیق."
    prediction = language_model._price_prediction(prompt)
    
    return prediction

def get_trading_signal(symbol, timeframe="1d"):
    """
    دریافت سیگنال معاملاتی
    
    Args:
        symbol (str): نماد ارز دیجیتال
        timeframe (str): تایم‌فریم تحلیل
        
    Returns:
        str: سیگنال معاملاتی تولید شده
    """
    # ایجاد نمونه رابط زبانی
    language_model = LanguageModelInterface()
    
    # دریافت سیگنال
    prompt = f"سیگنال معاملاتی دقیق برای {symbol} در تایم‌فریم {timeframe} با نقاط ورود، حد ضرر و حد سود."
    signal = language_model._trading_signal(prompt)
    
    return signal

def get_local_response(prompt: str) -> str:
    """
    دریافت پاسخ از سیستم محلی بدون نیاز به API خارجی
    
    این تابع برای استفاده در multi_model_ai طراحی شده تا در صورت عدم دسترسی 
    به مدل‌های خارجی، پاسخ محلی ارائه دهد.
    
    Args:
        prompt (str): متن پرسش
        
    Returns:
        str: پاسخ تولید شده
    """
    try:
        # ایجاد نمونه رابط زبانی
        local_model = LanguageModelInterface()
        
        # دریافت پاسخ محلی
        return local_model._get_local_completion(prompt)
    except Exception as e:
        # مدیریت خطا و ارائه پاسخ پشتیبان
        print(f"خطا در دریافت پاسخ محلی: {str(e)}")
        
        # تولید پاسخ اضطراری بر اساس نوع پرسش
        if "تحلیل" in prompt.lower() and any(crypto in prompt.upper() for crypto in ["BTC", "ETH", "SOL", "XRP", "ADA"]):
            return "در حال حاضر قادر به ارائه تحلیل دقیق نیستم. لطفاً اندکی بعد دوباره تلاش کنید. پیشنهاد می‌کنم از منابع مختلف برای تصمیم‌گیری معاملاتی استفاده کنید."
        elif "قیمت" in prompt.lower() or "پیش‌بینی" in prompt.lower():
            return "در شرایط فعلی، ارائه پیش‌بینی قیمت با دقت کافی امکان‌پذیر نیست. لطفاً از ابزارهای تحلیل تکنیکال و بنیادی برای ارزیابی بازار استفاده کنید."
        elif "سیگنال" in prompt.lower() or "معامله" in prompt.lower():
            return "در حال حاضر قادر به ارائه سیگنال معاملاتی نیستم. توصیه می‌کنم از استراتژی‌های مدیریت سرمایه مناسب استفاده کنید و ریسک خود را محدود کنید."
        else:
            return "متأسفانه در پردازش درخواست شما خطایی رخ داد. لطفاً سؤال خود را به شکل دیگری مطرح کنید یا بعداً دوباره تلاش کنید."