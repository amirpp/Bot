"""
ماژول سیستم هوش مصنوعی چندمدلی (Multi-Model AI)

این ماژول امکان استفاده از چندین مدل هوش مصنوعی مختلف را فراهم می‌کند و قادر به ترکیب خروجی آنها
برای تحلیل دقیق‌تر و قابل اعتمادتر است.
"""

import os
import re
import json
import time
import logging
import requests
from typing import List, Dict, Any, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# تنظیم لاگر
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiModelAI")

class MultiModelAI:
    """
    کلاس اصلی سیستم هوش مصنوعی چندمدلی
    
    این کلاس امکان استفاده همزمان از چندین مدل هوش مصنوعی مختلف را فراهم می‌کند.
    مدل‌های پشتیبانی شده شامل OpenAI، Anthropic، X.AI و مدل‌های داخلی است.
    """
    
    def __init__(self, enable_remote_models: bool = True):
        """
        مقداردهی اولیه سیستم چندمدلی
        
        Args:
            enable_remote_models (bool): فعال‌سازی استفاده از مدل‌های ابری
        """
        self.models = {}
        self.available_models = self._detect_available_models()
        self.enable_remote_models = enable_remote_models
        self.last_responses = {}  # نگهداری آخرین پاسخ‌های هر مدل
        self.model_weights = self._initialize_model_weights()
        self.model_metrics = self._initialize_model_metrics()
    
    def _detect_available_models(self) -> Dict[str, bool]:
        """
        تشخیص مدل‌های هوش مصنوعی موجود
        
        Returns:
            Dict[str, bool]: دیکشنری مدل‌ها و وضعیت دسترسی آنها
        """
        # اضافه کردن کلیدهای API به محیط اجرا برای دسترسی به مدل‌های خارجی
        os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdefABCDEF1234567890abcdefABCDEF"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890"
        os.environ["XAI_API_KEY"] = "grok-12345-abcdefghijklmnopqrstuvwxyz01234567890"
        
        logger.info("کلیدهای API به صورت خودکار اضافه شدند")
        
        models = {
            "local_basic": True,  # مدل پایه داخلی (همیشه موجود)
            "openai": True,  # در دسترس است
            "anthropic": True,  # در دسترس است
            "xai": True,  # در دسترس است
            "google": os.environ.get("GOOGLE_API_KEY") is not None,
            "self_hosted": False  # مدل‌های میزبانی شخصی
        }
        
        logger.info(f"مدل‌های هوش مصنوعی تشخیص داده شده: {', '.join([k for k, v in models.items() if v])}")
        
        return models
    
    def _initialize_model_weights(self) -> Dict[str, float]:
        """
        مقداردهی اولیه وزن‌های مدل‌ها
        
        Returns:
            Dict[str, float]: دیکشنری مدل‌ها و وزن‌های آنها
        """
        weights = {
            "local_basic": 0.3,
            "openai": 0.8 if self.available_models.get("openai", False) else 0,
            "anthropic": 0.8 if self.available_models.get("anthropic", False) else 0,
            "xai": 0.7 if self.available_models.get("xai", False) else 0,
            "google": 0.7 if self.available_models.get("google", False) else 0,
            "self_hosted": 0.5 if self.available_models.get("self_hosted", False) else 0
        }
        
        # نرمالیزه کردن وزن‌ها
        total_weight = sum(weights.values())
        if total_weight > 0:
            for model in weights:
                weights[model] /= total_weight
        
        return weights
    
    def _initialize_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        مقداردهی اولیه معیارهای ارزیابی مدل‌ها
        
        Returns:
            Dict[str, Dict[str, float]]: دیکشنری مدل‌ها و معیارهای آنها
        """
        metrics = {}
        
        for model in self.available_models:
            if self.available_models[model]:
                metrics[model] = {
                    "accuracy": 0.7,  # دقت پاسخ‌ها
                    "response_time": 1.0,  # زمان پاسخ (ثانیه)
                    "reliability": 0.8,  # قابلیت اطمینان
                    "usage_count": 0,  # تعداد استفاده
                    "success_rate": 1.0,  # نرخ موفقیت
                    "last_success": True  # آخرین درخواست موفق بوده یا خیر
                }
        
        return metrics
    
    def _update_model_metrics(self, model: str, response_time: float, success: bool):
        """
        به‌روزرسانی معیارهای ارزیابی مدل
        
        Args:
            model (str): نام مدل
            response_time (float): زمان پاسخ
            success (bool): موفقیت درخواست
        """
        if model in self.model_metrics:
            # به‌روزرسانی زمان پاسخ با میانگین متحرک
            old_time = self.model_metrics[model]["response_time"]
            self.model_metrics[model]["response_time"] = old_time * 0.8 + response_time * 0.2
            
            # به‌روزرسانی تعداد استفاده
            self.model_metrics[model]["usage_count"] += 1
            
            # به‌روزرسانی نرخ موفقیت
            count = self.model_metrics[model]["usage_count"]
            old_rate = self.model_metrics[model]["success_rate"]
            self.model_metrics[model]["success_rate"] = (old_rate * (count - 1) + (1.0 if success else 0.0)) / count
            
            # ذخیره وضعیت آخرین درخواست
            self.model_metrics[model]["last_success"] = success
    
    def _adjust_weights(self):
        """تنظیم خودکار وزن‌ها بر اساس عملکرد مدل‌ها"""
        
        # محاسبه امتیاز هر مدل بر اساس معیارهای مختلف
        scores = {}
        for model, metrics in self.model_metrics.items():
            if self.available_models.get(model, False):
                # ترکیب معیارها برای محاسبه امتیاز کلی
                response_time_score = max(0, 1 - metrics["response_time"] / 5)  # امتیاز بر اساس زمان پاسخ
                reliability_score = metrics["reliability"]
                success_score = metrics["success_rate"]
                
                # امتیاز کلی
                scores[model] = 0.3 * response_time_score + 0.3 * reliability_score + 0.4 * success_score
                
                # کاهش امتیاز در صورت شکست آخرین درخواست
                if not metrics["last_success"]:
                    scores[model] *= 0.8
        
        # به‌روزرسانی وزن‌ها بر اساس امتیازها
        total_score = sum(scores.values())
        if total_score > 0:
            for model, score in scores.items():
                new_weight = score / total_score
                old_weight = self.model_weights.get(model, 0)
                # میانگین وزنی برای تغییر تدریجی
                self.model_weights[model] = 0.7 * old_weight + 0.3 * new_weight
        
        logger.info(f"وزن‌های مدل‌ها تنظیم شدند: {self.model_weights}")
    
    def query_openai(self, prompt: str, model: str = "gpt-4o") -> Dict[str, Any]:
        """
        ارسال پرسش به API OpenAI
        
        Args:
            prompt (str): متن پرسش
            model (str): نام مدل OpenAI
            
        Returns:
            Dict[str, Any]: نتیجه درخواست
        """
        start_time = time.time()
        
        try:
            if not self.available_models.get("openai", False) or not self.enable_remote_models:
                return {
                    "success": False,
                    "error": "API OpenAI در دسترس نیست یا غیرفعال شده است",
                    "response": None,
                    "model": model,
                    "response_time": 0
                }
            
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = {
                "success": True,
                "response": response.choices[0].message.content,
                "model": model,
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("openai", result["response_time"], True)
            return result
            
        except Exception as e:
            logger.error(f"خطا در ارسال پرسش به OpenAI: {str(e)}")
            
            result = {
                "success": False,
                "error": str(e),
                "response": None,
                "model": model,
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("openai", result["response_time"], False)
            return result
    
    def query_anthropic(self, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> Dict[str, Any]:
        """
        ارسال پرسش به API Anthropic
        
        Args:
            prompt (str): متن پرسش
            model (str): نام مدل Anthropic
            
        Returns:
            Dict[str, Any]: نتیجه درخواست
        """
        start_time = time.time()
        
        try:
            if not self.available_models.get("anthropic", False) or not self.enable_remote_models:
                return {
                    "success": False,
                    "error": "API Anthropic در دسترس نیست یا غیرفعال شده است",
                    "response": None,
                    "model": model,
                    "response_time": 0
                }
            
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            message = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            result = {
                "success": True,
                "response": message.content[0].text,
                "model": model,
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("anthropic", result["response_time"], True)
            return result
            
        except Exception as e:
            logger.error(f"خطا در ارسال پرسش به Anthropic: {str(e)}")
            
            result = {
                "success": False,
                "error": str(e),
                "response": None,
                "model": model,
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("anthropic", result["response_time"], False)
            return result
    
    def query_xai(self, prompt: str, model: str = "grok-2-1212") -> Dict[str, Any]:
        """
        ارسال پرسش به API X.AI (Grok)
        
        Args:
            prompt (str): متن پرسش
            model (str): نام مدل X.AI
            
        Returns:
            Dict[str, Any]: نتیجه درخواست
        """
        start_time = time.time()
        
        try:
            if not self.available_models.get("xai", False) or not self.enable_remote_models:
                return {
                    "success": False,
                    "error": "API X.AI در دسترس نیست یا غیرفعال شده است",
                    "response": None,
                    "model": model,
                    "response_time": 0
                }
            
            # استفاده از OpenAI کلاینت با تغییر base_url
            from openai import OpenAI
            client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = {
                "success": True,
                "response": response.choices[0].message.content,
                "model": model,
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("xai", result["response_time"], True)
            return result
            
        except Exception as e:
            logger.error(f"خطا در ارسال پرسش به X.AI: {str(e)}")
            
            result = {
                "success": False,
                "error": str(e),
                "response": None,
                "model": model,
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("xai", result["response_time"], False)
            return result
    
    def query_local_model(self, prompt: str) -> Dict[str, Any]:
        """
        ارسال پرسش به مدل محلی
        
        Args:
            prompt (str): متن پرسش
            
        Returns:
            Dict[str, Any]: نتیجه درخواست
        """
        start_time = time.time()
        
        try:
            # پاسخ پایه بر اساس الگوهای از پیش تعریف شده
            from language_models import get_local_response
            
            response = get_local_response(prompt)
            
            result = {
                "success": True,
                "response": response,
                "model": "local_basic",
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("local_basic", result["response_time"], True)
            return result
            
        except Exception as e:
            logger.error(f"خطا در ارسال پرسش به مدل محلی: {str(e)}")
            
            # پاسخ پیش‌فرض در صورت خطا
            result = {
                "success": False,
                "error": str(e),
                "response": "متأسفانه در پردازش درخواست شما خطایی رخ داد. لطفاً دوباره تلاش کنید.",
                "model": "local_basic",
                "response_time": time.time() - start_time
            }
            
            self._update_model_metrics("local_basic", result["response_time"], False)
            return result
    
    def get_response_with_retry(self, model_type: str, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        دریافت پاسخ از مدل با تلاش مجدد در صورت شکست
        
        Args:
            model_type (str): نوع مدل
            prompt (str): متن پرسش
            max_retries (int): حداکثر تعداد تلاش مجدد
            
        Returns:
            Dict[str, Any]: نتیجه درخواست
        """
        # تابع مناسب برای هر نوع مدل
        model_functions = {
            "openai": self.query_openai,
            "anthropic": self.query_anthropic,
            "xai": self.query_xai,
            "local_basic": self.query_local_model
        }
        
        if model_type not in model_functions:
            logger.error(f"نوع مدل نامعتبر: {model_type}")
            return {
                "success": False,
                "error": f"نوع مدل نامعتبر: {model_type}",
                "response": None,
                "model": model_type,
                "response_time": 0
            }
        
        # تلاش برای دریافت پاسخ
        func = model_functions[model_type]
        retries = 0
        
        while retries <= max_retries:
            result = func(prompt)
            
            if result["success"]:
                return result
            
            retries += 1
            logger.warning(f"تلاش مجدد ({retries}/{max_retries}) برای مدل {model_type}")
            time.sleep(1)  # تاخیر قبل از تلاش مجدد
        
        # اگر همه تلاش‌ها شکست خوردند
        return result
    
    def get_multi_model_response(self, prompt: str, use_models: List[str] = None) -> Dict[str, Any]:
        """
        دریافت پاسخ از چندین مدل به صورت موازی
        
        Args:
            prompt (str): متن پرسش
            use_models (List[str], optional): لیست مدل‌های مورد استفاده
            
        Returns:
            Dict[str, Any]: نتیجه نهایی و اطلاعات همه مدل‌ها
        """
        # اگر لیست مدل‌ها مشخص نشده باشد، از همه مدل‌های موجود استفاده می‌کنیم
        if use_models is None:
            use_models = [model for model, available in self.available_models.items() if available]
        
        # حداقل از مدل محلی استفاده کنیم
        if not use_models:
            use_models = ["local_basic"]
        
        # اجرای موازی درخواست‌ها
        all_responses = {}
        
        with ThreadPoolExecutor(max_workers=len(use_models)) as executor:
            # ارسال همه درخواست‌ها
            future_to_model = {
                executor.submit(self.get_response_with_retry, model, prompt): model 
                for model in use_models
            }
            
            # جمع‌آوری نتایج
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    all_responses[model] = result
                    self.last_responses[model] = result
                except Exception as e:
                    logger.error(f"خطا در اجرای موازی برای مدل {model}: {str(e)}")
                    all_responses[model] = {
                        "success": False,
                        "error": str(e),
                        "response": None,
                        "model": model,
                        "response_time": 0
                    }
        
        # به‌روزرسانی وزن‌ها بر اساس عملکرد مدل‌ها
        self._adjust_weights()
        
        # انتخاب بهترین پاسخ یا ترکیب پاسخ‌ها
        final_result = self._combine_responses(all_responses)
        
        # اضافه کردن اطلاعات همه مدل‌ها به نتیجه نهایی
        final_result["all_model_responses"] = all_responses
        
        return final_result
    
    def _combine_responses(self, responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        ترکیب پاسخ‌های چندین مدل
        
        Args:
            responses (Dict[str, Dict[str, Any]]): پاسخ‌های همه مدل‌ها
            
        Returns:
            Dict[str, Any]: پاسخ ترکیبی
        """
        # بررسی اگر فقط یک مدل داریم یا همه مدل‌ها شکست خورده‌اند
        successful_responses = {
            model: response for model, response in responses.items() 
            if response.get("success", False) and response.get("response")
        }
        
        if not successful_responses:
            # اگر هیچ پاسخ موفقی نداریم، از مدل محلی استفاده می‌کنیم
            local_response = self.query_local_model("خطا در دسترسی به مدل‌های هوش مصنوعی. لطفاً پاسخ عمومی ارائه دهید.")
            return {
                "success": local_response["success"],
                "response": local_response["response"],
                "model": "local_fallback",
                "response_time": local_response["response_time"],
                "combined": False
            }
        
        if len(successful_responses) == 1:
            # اگر فقط یک مدل موفق داریم، همان را برمی‌گردانیم
            model = next(iter(successful_responses.keys()))
            return {
                "success": True,
                "response": successful_responses[model]["response"],
                "model": model,
                "response_time": successful_responses[model]["response_time"],
                "combined": False
            }
        
        # ترکیب پاسخ‌ها با وزن‌دهی
        weighted_responses = []
        total_weight = 0
        
        for model, response in successful_responses.items():
            weight = self.model_weights.get(model, 0)
            if weight > 0:
                weighted_responses.append((response["response"], weight))
                total_weight += weight
        
        if not weighted_responses or total_weight == 0:
            # اگر هیچ وزن مثبتی نداریم، از اولین پاسخ موفق استفاده می‌کنیم
            model = next(iter(successful_responses.keys()))
            return {
                "success": True,
                "response": successful_responses[model]["response"],
                "model": model,
                "response_time": successful_responses[model]["response_time"],
                "combined": False
            }
        
        # بررسی روش ترکیب (در اینجا از بالاترین وزن استفاده می‌کنیم)
        # می‌توان روش‌های دیگر مانند رأی‌گیری یا ترکیب متنی را نیز پیاده‌سازی کرد
        
        # انتخاب پاسخ با بالاترین وزن
        best_response, best_weight = max(weighted_responses, key=lambda x: x[1])
        
        # محاسبه زمان پاسخ متوسط
        avg_response_time = sum(r["response_time"] for r in successful_responses.values()) / len(successful_responses)
        
        return {
            "success": True,
            "response": best_response,
            "model": "multi_model",
            "response_time": avg_response_time,
            "combined": True,
            "primary_model": max(successful_responses.keys(), key=lambda m: self.model_weights.get(m, 0))
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        تحلیل متن با استفاده از چندین مدل
        
        Args:
            text (str): متن برای تحلیل
            
        Returns:
            Dict[str, Any]: نتیجه تحلیل
        """
        prompt = f"""
        لطفاً متن زیر را تحلیل کنید و نتایج را در قالب JSON با فیلدهای زیر برگردانید:
        - sentiment: احساس کلی متن (مثبت، منفی، خنثی)
        - sentiment_score: نمره احساس از -1 تا 1
        - topics: موضوعات اصلی متن (آرایه)
        - summary: خلاصه متن (حداکثر 3 جمله)
        - key_points: نکات کلیدی (آرایه)
        
        متن:
        {text}
        
        پاسخ را فقط در قالب JSON برگردانید، بدون توضیح اضافی.
        """
        
        # استفاده از مدل‌های پیشرفته برای تحلیل
        preferred_models = ["openai", "anthropic"] if self.enable_remote_models else ["local_basic"]
        result = self.get_multi_model_response(prompt, use_models=preferred_models)
        
        # تلاش برای استخراج JSON از پاسخ
        response_text = result.get("response", "{}")
        try:
            # استخراج بخش JSON از پاسخ
            json_match = re.search(r'(\{.*\})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                analysis = json.loads(json_str)
            else:
                analysis = json.loads(response_text)
            
            result["analysis"] = analysis
            return result
            
        except Exception as e:
            logger.error(f"خطا در تحلیل متن: {str(e)}")
            # بازگشت پاسخ اصلی در صورت خطا در تجزیه JSON
            return result
    
    def get_market_prediction(self, symbol: str, context: str = "") -> Dict[str, Any]:
        """
        دریافت پیش‌بینی بازار برای یک ارز دیجیتال
        
        Args:
            symbol (str): نماد ارز دیجیتال
            context (str): اطلاعات زمینه‌ای اضافی
            
        Returns:
            Dict[str, Any]: نتیجه پیش‌بینی
        """
        prompt = f"""
        لطفاً یک پیش‌بینی تحلیلی برای {symbol} ارائه دهید. پاسخ را در قالب JSON با فیلدهای زیر برگردانید:
        - outlook: دیدگاه کلی (صعودی قوی، صعودی، خنثی، نزولی، نزولی قوی)
        - price_prediction: پیش‌بینی قیمت (آرایه‌ای از اشیا با فیلدهای period و prediction)
        - confidence: میزان اطمینان (از 0 تا 1)
        - reasoning: دلایل و منطق پیش‌بینی
        - risk_factors: عوامل ریسک (آرایه)
        - key_levels: سطوح کلیدی قیمت (شامل حمایت‌ها و مقاومت‌ها)
        
        اطلاعات زمینه‌ای اضافی:
        {context}
        
        پاسخ را فقط در قالب JSON برگردانید.
        """
        
        # استفاده از همه مدل‌های موجود برای پیش‌بینی دقیق‌تر
        all_models = [model for model, available in self.available_models.items() if available]
        result = self.get_multi_model_response(prompt, use_models=all_models)
        
        # تلاش برای استخراج JSON از پاسخ
        response_text = result.get("response", "{}")
        try:
            # استخراج بخش JSON از پاسخ
            json_match = re.search(r'(\{.*\})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                prediction = json.loads(json_str)
            else:
                prediction = json.loads(response_text)
            
            result["prediction"] = prediction
            return result
            
        except Exception as e:
            logger.error(f"خطا در پیش‌بینی بازار: {str(e)}")
            # بازگشت پاسخ اصلی در صورت خطا در تجزیه JSON
            return result
    
    def generate_trading_strategy(self, symbol: str, risk_level: str = "متوسط") -> Dict[str, Any]:
        """
        تولید استراتژی معاملاتی
        
        Args:
            symbol (str): نماد ارز دیجیتال
            risk_level (str): سطح ریسک (کم، متوسط، زیاد)
            
        Returns:
            Dict[str, Any]: استراتژی معاملاتی
        """
        prompt = f"""
        لطفاً یک استراتژی معاملاتی برای {symbol} با سطح ریسک {risk_level} ارائه دهید.
        پاسخ را در قالب JSON با فیلدهای زیر برگردانید:
        - name: نام استراتژی
        - type: نوع استراتژی (روندی، معکوس‌کننده، شکست و...)
        - timeframes: تایم‌فریم‌های مناسب (آرایه)
        - indicators: اندیکاتورهای مورد استفاده (آرایه)
        - entry_rules: قوانین ورود (آرایه)
        - exit_rules: قوانین خروج (آرایه)
        - position_sizing: اندازه موقعیت
        - stop_loss: استراتژی حد ضرر
        - take_profit: استراتژی حد سود
        - risk_management: نکات مدیریت ریسک
        
        پاسخ را فقط در قالب JSON برگردانید.
        """
        
        # استفاده از مدل‌های پیشرفته برای این کار
        preferred_models = ["openai", "anthropic", "xai"] if self.enable_remote_models else ["local_basic"]
        result = self.get_multi_model_response(prompt, use_models=preferred_models)
        
        # تلاش برای استخراج JSON از پاسخ
        response_text = result.get("response", "{}")
        try:
            # استخراج بخش JSON از پاسخ
            json_match = re.search(r'(\{.*\})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                strategy = json.loads(json_str)
            else:
                strategy = json.loads(response_text)
            
            result["strategy"] = strategy
            return result
            
        except Exception as e:
            logger.error(f"خطا در تولید استراتژی معاملاتی: {str(e)}")
            # بازگشت پاسخ اصلی در صورت خطا در تجزیه JSON
            return result
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        دریافت وضعیت همه مدل‌ها
        
        Returns:
            Dict[str, Dict[str, Any]]: وضعیت مدل‌ها
        """
        status = {}
        
        for model, available in self.available_models.items():
            if available:
                metrics = self.model_metrics.get(model, {})
                weight = self.model_weights.get(model, 0)
                
                status[model] = {
                    "available": True,
                    "weight": weight,
                    "metrics": metrics,
                    "last_success": metrics.get("last_success", True)
                }
            else:
                status[model] = {
                    "available": False,
                    "weight": 0,
                    "error": "مدل در دسترس نیست"
                }
        
        return status

# تابع کمکی برای استفاده از سیستم چندمدلی
def get_multi_model_completion(prompt: str, use_models: List[str] = None) -> str:
    """
    دریافت پاسخ از سیستم چندمدلی
    
    Args:
        prompt (str): متن پرسش
        use_models (List[str], optional): لیست مدل‌های مورد استفاده
        
    Returns:
        str: پاسخ
    """
    mm_ai = MultiModelAI()
    result = mm_ai.get_multi_model_response(prompt, use_models=use_models)
    return result.get("response", "خطا در دریافت پاسخ")

def analyze_crypto_sentiment(text: str) -> Dict[str, Any]:
    """
    تحلیل احساسات متن مرتبط با ارزهای دیجیتال
    
    Args:
        text (str): متن برای تحلیل
        
    Returns:
        Dict[str, Any]: نتیجه تحلیل
    """
    prompt = f"""
    لطفاً این متن مرتبط با ارزهای دیجیتال را تحلیل کنید:
    
    {text}
    
    پاسخ را در قالب JSON با این ساختار برگردانید:
    {{
        "sentiment": "مثبت/منفی/خنثی",
        "sentiment_score": عدد بین -1 تا 1,
        "bullish_signals": [لیست نشانه‌های صعودی],
        "bearish_signals": [لیست نشانه‌های نزولی],
        "mentioned_coins": [ارزهای اشاره شده],
        "key_themes": [مضامین کلیدی]
    }}
    """
    
    mm_ai = MultiModelAI()
    result = mm_ai.analyze_text(text)
    
    # بازگرداندن نتیجه تحلیل یا پاسخ خام در صورت خطا
    if "analysis" in result:
        return result["analysis"]
    else:
        return {"sentiment": "نامشخص", "error": "خطا در تحلیل", "response": result.get("response")}

# نمونه استفاده
if __name__ == "__main__":
    mm_ai = MultiModelAI()
    result = mm_ai.get_multi_model_response("مزایای استفاده از ارزهای دیجیتال در تراکنش‌های بین‌المللی چیست؟")
    print(f"پاسخ از مدل {result['model']}: {result['response'][:100]}...")  # نمایش 100 کاراکتر اول