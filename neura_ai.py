"""
ماژول نیورا - سیستم هوش مصنوعی خودترمیم، خودآموز و خودنویس

این ماژول شامل توابع و کلاس‌های مورد نیاز برای سیستم هوش مصنوعی پیشرفته نیورا است.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
import threading
import logging
import re
import requests
from typing import Dict, List, Any, Union, Optional

try:
    import streamlit as st
except ImportError:
    st = None

# واردسازی ماژول‌های داخلی
from language_models import LanguageModelInterface
from technical_analysis import perform_technical_analysis
from chart_patterns import analyze_chart_patterns
from advanced_ai_engine import AdvancedAIEngine

# تنظیم لاگر
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("neura_ai.log")
    ]
)
logger = logging.getLogger("Neura")

class NeuraAI:
    """کلاس اصلی نیورا - سیستم هوش مصنوعی خودترمیم، خودآموز و خودنویس"""
    
    def __init__(self, system_name="نیورا"):
        """
        مقداردهی اولیه سیستم نیورا
        
        Args:
            system_name (str): نام سیستم
        """
        self.name = system_name
        self.version = "1.5.0"  # ارتقا به نسخه جدید
        self.language_model = LanguageModelInterface()
        self.ai_engine = AdvancedAIEngine()  # موتور هوش مصنوعی پیشرفته
        self.memory = self._initialize_memory()
        self.skills = self._initialize_skills()
        self.learning_data = self._initialize_learning_data()
        self.last_query = None
        self.last_response = None
        self.active = True
        self.threads = []
        
        # راه‌اندازی سیستم‌های پایش
        self._start_monitoring_systems()
        
        logger.info(f"نیورا نسخه {self.version} راه‌اندازی شد.")
    
    def _initialize_memory(self) -> Dict:
        """
        مقداردهی حافظه سیستم
        
        Returns:
            Dict: دیکشنری حاوی اطلاعات حافظه
        """
        try:
            # تلاش برای بارگذاری از فایل
            if os.path.exists("neura_memory.json"):
                with open("neura_memory.json", "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"خطا در بارگذاری حافظه: {str(e)}")
        
        # ایجاد حافظه پیش‌فرض
        return {
            "conversations": [],
            "analysis_history": [],
            "learned_patterns": {},
            "last_updated": str(datetime.now())
        }
    
    def _save_memory(self):
        """ذخیره حافظه در فایل"""
        try:
            self.memory["last_updated"] = str(datetime.now())
            with open("neura_memory.json", "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"خطا در ذخیره‌سازی حافظه: {str(e)}")
    
    def _initialize_skills(self) -> Dict[str, Dict]:
        """
        مقداردهی مهارت‌های سیستم
        
        Returns:
            Dict: دیکشنری حاوی اطلاعات مهارت‌ها
        """
        return {
            "market_analysis": {
                "name": "تحلیل بازار",
                "description": "تحلیل وضعیت فعلی بازار ارزهای دیجیتال",
                "confidence": 0.9,
                "function": self.analyze_market
            },
            "price_prediction": {
                "name": "پیش‌بینی قیمت",
                "description": "پیش‌بینی قیمت آینده ارزهای دیجیتال",
                "confidence": 0.8,
                "function": self.predict_price
            },
            "pattern_recognition": {
                "name": "تشخیص الگو",
                "description": "تشخیص الگوهای نموداری در قیمت ارزها",
                "confidence": 0.85,
                "function": self.recognize_patterns
            },
            "sentiment_analysis": {
                "name": "تحلیل احساسات",
                "description": "تحلیل احساسات بازار و اخبار",
                "confidence": 0.75,
                "function": self.analyze_sentiment
            },
            "code_generation": {
                "name": "تولید کد",
                "description": "تولید کد بر اساس درخواست کاربر",
                "confidence": 0.7,
                "function": self.generate_code
            },
            "self_improvement": {
                "name": "خودبهسازی",
                "description": "بهبود عملکرد خود بر اساس بازخورد",
                "confidence": 0.65,
                "function": self.improve_self
            }
        }
    
    def _initialize_learning_data(self) -> Dict:
        """
        مقداردهی داده‌های یادگیری سیستم
        
        Returns:
            Dict: دیکشنری حاوی داده‌های یادگیری
        """
        return {
            "training_examples": [],
            "performance_metrics": {
                "accuracy": 0.0,
                "response_time": 0.0,
                "user_satisfaction": 0.0
            },
            "improvement_targets": {
                "accuracy": 0.95,
                "response_time": 1.0,
                "user_satisfaction": 0.9
            }
        }
    
    def _start_monitoring_systems(self):
        """راه‌اندازی سیستم‌های پایش"""
        # سیستم پایش سلامت
        health_monitor = threading.Thread(
            target=self._health_monitoring_task,
            daemon=True
        )
        self.threads.append(health_monitor)
        health_monitor.start()
        
        # سیستم یادگیری مداوم
        continuous_learning = threading.Thread(
            target=self._continuous_learning_task,
            daemon=True
        )
        self.threads.append(continuous_learning)
        continuous_learning.start()
        
        logger.info("سیستم‌های پایش نیورا راه‌اندازی شدند.")
    
    def _health_monitoring_task(self):
        """وظیفه پایش سلامت سیستم"""
        while self.active:
            try:
                # بررسی وضعیت سیستم
                system_status = self._check_system_status()
                
                # ترمیم مشکلات احتمالی
                if system_status.get("needs_repair", False):
                    self._self_repair(system_status.get("issues", []))
                
                # ذخیره وضعیت در حافظه
                self.memory["system_status"] = system_status
                self._save_memory()
                
                # انتظار تا بررسی بعدی
                time.sleep(300)  # هر 5 دقیقه
            except Exception as e:
                logger.error(f"خطا در پایش سلامت: {str(e)}")
                time.sleep(60)  # انتظار کوتاه‌تر در صورت خطا
    
    def _continuous_learning_task(self):
        """وظیفه یادگیری مداوم"""
        while self.active:
            try:
                # بررسی تاریخچه تعاملات برای یادگیری
                if len(self.memory["conversations"]) > 0:
                    recent_conversations = self.memory["conversations"][-10:]
                    self._learn_from_conversations(recent_conversations)
                
                # بهینه‌سازی مهارت‌ها
                self._optimize_skills()
                
                # ذخیره داده‌های یادگیری
                self._save_memory()
                
                # انتظار تا یادگیری بعدی
                time.sleep(600)  # هر 10 دقیقه
            except Exception as e:
                logger.error(f"خطا در یادگیری مداوم: {str(e)}")
                time.sleep(60)  # انتظار کوتاه‌تر در صورت خطا
    
    def _check_system_status(self) -> Dict:
        """
        بررسی وضعیت سیستم
        
        Returns:
            Dict: اطلاعات وضعیت سیستم
        """
        status = {
            "timestamp": str(datetime.now()),
            "health": "سالم",
            "needs_repair": False,
            "issues": [],
            "memory_usage": len(json.dumps(self.memory)),
            "active_threads": sum(1 for t in self.threads if t.is_alive())
        }
        
        # بررسی حافظه
        if status["memory_usage"] > 10000000:  # 10MB
            status["health"] = "نیاز به بهینه‌سازی"
            status["needs_repair"] = True
            status["issues"].append("حجم حافظه بالا")
        
        # بررسی ترد‌ها
        expected_threads = 2  # تعداد ترد‌های مورد انتظار
        if status["active_threads"] < expected_threads:
            status["health"] = "نیاز به ترمیم"
            status["needs_repair"] = True
            status["issues"].append("ترد‌های غیرفعال")
        
        return status
    
    def _self_repair(self, issues: List[str]):
        """
        ترمیم خودکار مشکلات
        
        Args:
            issues (List[str]): لیست مشکلات شناسایی شده
        """
        logger.info(f"شروع ترمیم خودکار برای مشکلات: {issues}")
        
        for issue in issues:
            if issue == "حجم حافظه بالا":
                # پاک‌سازی حافظه قدیمی
                if len(self.memory["conversations"]) > 100:
                    self.memory["conversations"] = self.memory["conversations"][-50:]
                if len(self.memory["analysis_history"]) > 100:
                    self.memory["analysis_history"] = self.memory["analysis_history"][-50:]
                logger.info("حافظه بهینه‌سازی شد.")
            
            elif issue == "ترد‌های غیرفعال":
                # راه‌اندازی مجدد ترد‌ها
                self._restart_dead_threads()
                logger.info("ترد‌های غیرفعال مجددا راه‌اندازی شدند.")
        
        # ذخیره تغییرات
        self._save_memory()
    
    def _restart_dead_threads(self):
        """راه‌اندازی مجدد ترد‌های غیرفعال"""
        # حذف ترد‌های غیرفعال
        self.threads = [t for t in self.threads if t.is_alive()]
        
        # بررسی و اضافه کردن ترد‌های مورد نیاز
        has_health_monitor = False
        has_learning = False
        
        for t in self.threads:
            if t._target == self._health_monitoring_task:
                has_health_monitor = True
            elif t._target == self._continuous_learning_task:
                has_learning = True
        
        if not has_health_monitor:
            health_monitor = threading.Thread(
                target=self._health_monitoring_task,
                daemon=True
            )
            self.threads.append(health_monitor)
            health_monitor.start()
            logger.info("ترد پایش سلامت مجددا راه‌اندازی شد.")
        
        if not has_learning:
            continuous_learning = threading.Thread(
                target=self._continuous_learning_task,
                daemon=True
            )
            self.threads.append(continuous_learning)
            continuous_learning.start()
            logger.info("ترد یادگیری مداوم مجددا راه‌اندازی شد.")
    
    def _learn_from_conversations(self, conversations: List[Dict]):
        """
        یادگیری از تعاملات گذشته
        
        Args:
            conversations (List[Dict]): لیست تعاملات
        """
        if not conversations:
            return
        
        # استخراج الگوهای پرسش و پاسخ
        for conv in conversations:
            if "query" in conv and "response" in conv:
                query_type = self._categorize_query(conv["query"])
                if query_type:
                    # ذخیره نمونه یادگیری
                    self.learning_data["training_examples"].append({
                        "query": conv["query"],
                        "query_type": query_type,
                        "response": conv["response"],
                        "timestamp": conv.get("timestamp", str(datetime.now()))
                    })
                    
                    # بروزرسانی الگوهای آموخته شده
                    if query_type not in self.memory["learned_patterns"]:
                        self.memory["learned_patterns"][query_type] = []
                    
                    self.memory["learned_patterns"][query_type].append({
                        "pattern": self._extract_query_pattern(conv["query"]),
                        "example": conv["query"]
                    })
    
    def _categorize_query(self, query: str) -> Optional[str]:
        """
        دسته‌بندی نوع پرسش
        
        Args:
            query (str): متن پرسش
            
        Returns:
            Optional[str]: نوع پرسش یا None
        """
        query_lower = query.lower()
        
        # الگوهای تشخیص نوع پرسش
        patterns = {
            "market_analysis": ["تحلیل بازار", "وضعیت بازار", "شرایط بازار", "روند بازار"],
            "price_prediction": ["پیش‌بینی قیمت", "قیمت آینده", "چه قیمتی میرسه", "قیمت میره"],
            "technical_indicators": ["اندیکاتور", "rsi", "ماکد", "macd", "میانگین متحرک", "بولینگر"],
            "trading_strategy": ["استراتژی", "روش معامله", "معامله کنم", "ترید کنم", "فروش", "خرید"],
            "general_question": ["چیست", "چطور", "چگونه", "کدام", "چرا", "چه"],
            "command": ["اجرا کن", "نشان بده", "محاسبه کن", "ایجاد کن", "بساز"]
        }
        
        for category, keywords in patterns.items():
            if any(kw in query_lower for kw in keywords):
                return category
        
        return None
    
    def _extract_query_pattern(self, query: str) -> str:
        """
        استخراج الگوی پرسش
        
        Args:
            query (str): متن پرسش
            
        Returns:
            str: الگوی پرسش
        """
        # حذف کلمات خاص و اعداد
        pattern = re.sub(r'\b(در|به|از|با|برای|که|چه|چرا|چگونه)\b', '*', query)
        pattern = re.sub(r'\d+', '#', pattern)
        return pattern
    
    def _optimize_skills(self):
        """بهینه‌سازی مهارت‌ها بر اساس یادگیری"""
        # بهبود اطمینان در مهارت‌ها بر اساس عملکرد
        for skill_name in self.skills:
            if skill_name in self.memory["learned_patterns"]:
                patterns_count = len(self.memory["learned_patterns"][skill_name])
                if patterns_count > 5:
                    # افزایش اطمینان با افزایش تعداد الگوها
                    current_confidence = self.skills[skill_name]["confidence"]
                    improved_confidence = min(0.95, current_confidence + 0.01)
                    self.skills[skill_name]["confidence"] = improved_confidence
    
    def analyze_market(self, symbol: str, timeframe: str = "1h", data = None) -> str:
        """
        تحلیل بازار ارز دیجیتال
        
        Args:
            symbol (str): نماد ارز دیجیتال
            timeframe (str): تایم‌فریم تحلیل
            data: داده‌های قیمت (اختیاری)
            
        Returns:
            str: تحلیل بازار
        """
        start_time = time.time()
        
        try:
            # دریافت داده‌های بازار اگر ارسال نشده باشند
            if data is None:
                from crypto_data import get_market_data
                data = get_market_data(symbol, timeframe, limit=100)
            
            if data is None or len(data) < 20:
                # اگر داده‌ای وجود نداشت، از هوش مصنوعی زبانی استفاده کن
                prompt = f"تحلیل جامع بازار {symbol} در تایم‌فریم {timeframe}"
                response = self.language_model.get_openai_completion(prompt)
            else:
                # انجام تحلیل تکنیکال پایه
                basic_analysis = perform_technical_analysis(data)
                
                # افزودن تحلیل الگوهای نموداری
                patterns = analyze_chart_patterns(data)
                if patterns:
                    basic_analysis["chart_patterns"] = patterns
                
                # استفاده از موتور هوش مصنوعی پیشرفته
                # 1. تحلیل سیکل بازار
                market_cycle = self.ai_engine.analyze_market_cycle(data)
                
                # 2. تولید سیگنال‌های معاملاتی
                trading_signals = self.ai_engine.generate_trading_signals(data)
                
                # ترکیب نتایج برای تحلیل نهایی
                advanced_analysis = {
                    "basic_analysis": basic_analysis,
                    "market_cycle": market_cycle,
                    "trading_signals": trading_signals
                }
                
                # تبدیل نتایج تحلیل به متن قابل فهم برای انسان
                # 1. تنظیم فرمت خروجی بر اساس سیکل بازار
                cycle_text = self.ai_engine.format_analysis_for_humans(market_cycle, "detailed")
                
                # 2. تنظیم فرمت خروجی بر اساس سیگنال‌های معاملاتی
                signal_text = self.ai_engine.format_analysis_for_humans(trading_signals, "detailed")
                
                # 3. ترکیب پاسخ نهایی با اطلاعات پایه
                response = f"# تحلیل پیشرفته بازار {symbol} در تایم‌فریم {timeframe}\n\n"
                
                # اطلاعات پایه
                response += "## اطلاعات اصلی\n"
                
                if "trend" in basic_analysis:
                    response += f"**روند کلی**: {basic_analysis['trend']}\n\n"
                
                if "indicators" in basic_analysis:
                    response += "### شاخص‌های تکنیکال\n"
                    for indicator, value in basic_analysis["indicators"].items():
                        response += f"- **{indicator}**: {value}\n"
                    response += "\n"
                
                if "support_resistance" in basic_analysis:
                    response += "### سطوح حمایت و مقاومت\n"
                    response += f"- **حمایت‌ها**: {', '.join([str(s) for s in basic_analysis['support_resistance']['supports']])}\n"
                    response += f"- **مقاومت‌ها**: {', '.join([str(r) for r in basic_analysis['support_resistance']['resistances']])}\n\n"
                
                if "chart_patterns" in basic_analysis:
                    response += "### الگوهای نموداری شناسایی شده\n"
                    for pattern, confidence in basic_analysis["chart_patterns"].items():
                        response += f"- **{pattern}**: {confidence:.0%} اطمینان\n"
                    response += "\n"
                
                # اضافه کردن تحلیل پیشرفته
                response += f"\n## تحلیل سیکل بازار\n{cycle_text}\n\n"
                response += f"\n## سیگنال معاملاتی\n{signal_text}\n"
            
            # ذخیره تحلیل در تاریخچه
            self.memory["analysis_history"].append({
                "type": "market_analysis",
                "symbol": symbol,
                "timeframe": timeframe,
                "result": response,
                "timestamp": str(datetime.now())
            })
            
            # بروزرسانی معیارهای عملکرد
            self.learning_data["performance_metrics"]["response_time"] = time.time() - start_time
            self._save_memory()
            
            return response
            
        except Exception as e:
            logger.error(f"خطا در تحلیل بازار: {str(e)}")
            logger.error(traceback.format_exc())
            
            # در صورت خطا، استفاده از هوش مصنوعی زبانی
            prompt = f"تحلیل جامع بازار {symbol} در تایم‌فریم {timeframe}"
            response = self.language_model.get_openai_completion(prompt)
            
            return f"[تحلیل از سیستم بکاپ]\n\n{response}\n\n[خطای سیستم اصلی: {str(e)}]"
    
    def predict_price(self, symbol: str, timeframe: str = "1d", days_ahead: int = 7, data = None) -> str:
        """
        پیش‌بینی قیمت ارز دیجیتال
        
        Args:
            symbol (str): نماد ارز دیجیتال
            timeframe (str): تایم‌فریم تحلیل
            days_ahead (int): تعداد روزهای آینده
            data: داده‌های قیمت (اختیاری)
            
        Returns:
            str: پیش‌بینی قیمت
        """
        start_time = time.time()
        
        try:
            # دریافت داده‌های بازار اگر ارسال نشده باشند
            if data is None:
                from crypto_data import get_market_data
                data = get_market_data(symbol, timeframe, limit=100)
            
            if data is None or len(data) < 30:
                # اگر داده‌ای وجود نداشت، از هوش مصنوعی زبانی استفاده کن
                prompt = f"پیش‌بینی قیمت {symbol} برای {days_ahead} روز آینده در تایم‌فریم {timeframe}"
                response = self.language_model.get_openai_completion(prompt)
            else:
                # استفاده از موتور هوش مصنوعی پیشرفته برای پیش‌بینی قیمت
                prediction_result = self.ai_engine.predict_price_ml(data, days_ahead=days_ahead)
                
                # تبدیل نتایج پیش‌بینی به متن قابل فهم برای انسان
                response = self.ai_engine.format_analysis_for_humans(prediction_result, "detailed")
                
                # افزودن اطلاعات اضافی
                response = f"# پیش‌بینی قیمت {symbol} برای {days_ahead} روز آینده\n\n{response}\n\n"
                response += "## نکات ضروری\n"
                response += "* این پیش‌بینی با استفاده از مدل‌های یادگیری ماشین انجام شده و قطعیت ندارد.\n"
                response += "* عوامل بنیادی، اخبار و رویدادهای جهانی می‌توانند تأثیر عمده‌ای بر قیمت داشته باشند.\n"
                response += "* همیشه از مدیریت سرمایه مناسب استفاده کنید و بر اساس پیش‌بینی‌های قیمتی تصمیمات عجولانه نگیرید.\n"
            
            # ذخیره پیش‌بینی در تاریخچه
            self.memory["analysis_history"].append({
                "type": "price_prediction",
                "symbol": symbol,
                "timeframe": timeframe,
                "days_ahead": days_ahead,
                "result": response,
                "timestamp": str(datetime.now())
            })
            
            # بروزرسانی معیارهای عملکرد
            self.learning_data["performance_metrics"]["response_time"] = time.time() - start_time
            self._save_memory()
            
            return response
            
        except Exception as e:
            logger.error(f"خطا در پیش‌بینی قیمت: {str(e)}")
            logger.error(traceback.format_exc())
            
            # در صورت خطا، استفاده از هوش مصنوعی زبانی
            prompt = f"پیش‌بینی قیمت {symbol} برای {days_ahead} روز آینده در تایم‌فریم {timeframe}"
            response = self.language_model.get_openai_completion(prompt)
            
            return f"[پیش‌بینی از سیستم بکاپ]\n\n{response}\n\n[خطای سیستم اصلی: {str(e)}]"
    
    def recognize_patterns(self, data, timeframe: str = "1d") -> Dict:
        """
        تشخیص الگوهای نموداری
        
        Args:
            data: داده‌های قیمت
            timeframe (str): تایم‌فریم تحلیل
            
        Returns:
            Dict: الگوهای شناسایی شده
        """
        if not isinstance(data, pd.DataFrame):
            return {"patterns": [], "error": "داده‌های نامعتبر"}
        
        try:
            # انجام تحلیل تکنیکال
            indicators = [
                'RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 
                'ADX', 'ATR', 'EMA', 'SMA'
            ]
            df = perform_technical_analysis(data, indicators)
            
            # تشخیص الگوها با سیستم قدیمی
            basic_patterns = analyze_chart_patterns(df)
            
            # استفاده از موتور هوش مصنوعی پیشرفته برای تحلیل عمیق‌تر
            # 1. بررسی الگوهای پیشرفته
            pattern_templates = self.ai_engine._load_pattern_templates()
            advanced_patterns = {}
            
            # 2. تحلیل و ترکیب الگوها
            for pattern_name, pattern_info in pattern_templates.items():
                if pattern_name in basic_patterns:
                    # الگو شناسایی شده است، امتیاز بالاتری به آن بدهیم
                    advanced_patterns[pattern_name] = {
                        "confidence": basic_patterns[pattern_name],
                        "type": pattern_info["type"],
                        "confirmation": pattern_info["confirmation"],
                        "target": pattern_info["target"],
                        "reliability": pattern_info["reliability"]
                    }
            
            # 3. امتیازدهی به الگوها و مرتب‌سازی بر اساس اطمینان
            sorted_patterns = sorted(
                [(name, info["confidence"] * info["reliability"]) 
                 for name, info in advanced_patterns.items()],
                key=lambda x: x[1], reverse=True
            )
            
            # 4. افزودن اطلاعات زمانی و مکانی بهتر
            result = {
                "basic_patterns": basic_patterns,
                "advanced_patterns": advanced_patterns,
                "recommended_patterns": [name for name, _ in sorted_patterns[:3]],
                "pattern_count": len(advanced_patterns),
                "timeframe": timeframe,
                "timestamp": str(datetime.now()),
                "data_points": len(data),
                "primary_trend": "صعودی" if data['close'].iloc[-1] > data['close'].iloc[0] else "نزولی",
                "volatility": float(data['high'].iloc[-20:].max() / data['low'].iloc[-20:].min() - 1) * 100
            }
            
            # تعیین فاز بازار
            if "rsi" in df.columns:
                last_rsi = df["rsi"].iloc[-1]
                if last_rsi > 70:
                    result["market_phase"] = "اشباع خرید"
                elif last_rsi < 30:
                    result["market_phase"] = "اشباع فروش"
                else:
                    result["market_phase"] = "میانی"
            
            return result
            
        except Exception as e:
            logger.error(f"خطا در تشخیص الگو: {str(e)}")
            logger.error(traceback.format_exc())
            return {"patterns": [], "error": str(e)}
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        تحلیل احساسات متن
        
        Args:
            text (str): متن برای تحلیل
            
        Returns:
            Dict: نتیجه تحلیل احساسات
        """
        try:
            # استفاده از لغت‌نامه احساسات پیشرفته از موتور هوش مصنوعی
            sentiment_lexicon = self.ai_engine._load_sentiment_lexicon()
            
            # کلمات کلیدی به همراه وزن‌های آنها
            positive_words = sentiment_lexicon["positive"]
            negative_words = sentiment_lexicon["negative"]
            neutral_words = sentiment_lexicon["neural"]
            word_weights = sentiment_lexicon.get("weights", {})
            
            # پیش‌پردازش متن
            text_lower = text.lower()
            
            # میزان مثبت و منفی بودن متن
            positive_score = 0
            negative_score = 0
            neutral_score = 0
            
            # شمارش کلمات کلیدی با وزن‌دهی
            for word in positive_words:
                if word in text_lower:
                    weight = word_weights.get(word, 0.5)
                    positive_score += weight
            
            for word in negative_words:
                if word in text_lower:
                    weight = abs(word_weights.get(word, -0.5))  # تبدیل به عدد مثبت
                    negative_score += weight
            
            for word in neutral_words:
                if word in text_lower:
                    neutral_score += 0.2
            
            # محاسبه نمره نهایی
            total_weight = positive_score + negative_score + neutral_score
            if total_weight == 0:
                # اگر هیچ کلمه کلیدی یافت نشد، از مدل زبانی استفاده کنیم
                prompt = f"لطفاً احساسات این متن را تحلیل کنید و نتیجه را به صورت یک عدد بین -1 تا 1 برگردانید:\n\n{text}"
                response = self.language_model.get_openai_completion(prompt)
                try:
                    sentiment_score = float(response.strip())
                    if sentiment_score > 0.2:
                        sentiment = "مثبت"
                    elif sentiment_score < -0.2:
                        sentiment = "منفی"
                    else:
                        sentiment = "خنثی"
                    
                    normalized_score = (sentiment_score + 1) / 2  # تبدیل به بازه 0 تا 1
                    return {
                        "sentiment": sentiment,
                        "score": normalized_score,
                        "model": "زبانی",
                        "confidence": 0.7,
                        "sentiment_words": []
                    }
                except:
                    # اگر نتوانستیم مقدار را به عدد تبدیل کنیم، مقدار پیش‌فرض برگردانیم
                    return {
                        "sentiment": "خنثی",
                        "score": 0.5,
                        "model": "پیش‌فرض",
                        "confidence": 0.5,
                        "sentiment_words": []
                    }
            
            # محاسبه تعادل بین مثبت و منفی
            net_score = positive_score - negative_score
            normalized_score = (net_score / total_weight + 1) / 2  # تبدیل به بازه 0 تا 1
            
            # تعیین احساس کلی
            if normalized_score > 0.6:
                sentiment = "مثبت"
                confidence = min(0.95, 0.6 + (normalized_score - 0.6) * 2)
            elif normalized_score < 0.4:
                sentiment = "منفی"
                confidence = min(0.95, 0.6 + (0.4 - normalized_score) * 2)
            else:
                sentiment = "خنثی"
                confidence = 0.5 - abs(normalized_score - 0.5)
            
            # استخراج کلمات کلیدی یافت شده
            found_words = []
            for word in positive_words:
                if word in text_lower:
                    found_words.append({"word": word, "type": "مثبت", "weight": word_weights.get(word, 0.5)})
            
            for word in negative_words:
                if word in text_lower:
                    found_words.append({"word": word, "type": "منفی", "weight": word_weights.get(word, -0.5)})
            
            # مرتب‌سازی براساس وزن (بیشترین تأثیر)
            found_words = sorted(found_words, key=lambda x: abs(x["weight"]), reverse=True)
            
            # نتیجه نهایی
            return {
                "sentiment": sentiment,
                "score": normalized_score,
                "model": "پیشرفته",
                "confidence": confidence,
                "positive_score": positive_score,
                "negative_score": negative_score,
                "neutral_score": neutral_score,
                "sentiment_words": found_words[:5]  # 5 کلمه کلیدی با بیشترین تأثیر
            }
            
        except Exception as e:
            logger.error(f"خطا در تحلیل احساسات: {str(e)}")
            logger.error(traceback.format_exc())
            
            # برگرداندن نتیجه ساده‌تر در صورت خطا
            positive_words = ["صعودی", "افزایش", "بهبود", "مثبت", "سود", "موفقیت", "رشد", "قوی"]
            negative_words = ["نزولی", "کاهش", "ضعف", "منفی", "ضرر", "شکست", "سقوط", "ضعیف"]
            
            text_lower = text.lower()
            
            # شمارش کلمات کلیدی
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total = positive_count + negative_count
            if total == 0:
                return {"sentiment": "خنثی", "score": 0.5, "model": "ساده"}
            
            # محاسبه امتیاز
            score = positive_count / total
            
            # تعیین احساس
            if score > 0.6:
                sentiment = "مثبت"
            elif score < 0.4:
                sentiment = "منفی"
            else:
                sentiment = "خنثی"
            
            return {
                "sentiment": sentiment,
                "score": score,
                "model": "ساده",
                "positive_words": positive_count,
                "negative_words": negative_count
            }
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """
        تولید کد بر اساس توضیحات با استفاده از هوش مصنوعی پیشرفته
        
        Args:
            description (str): توضیحات کد مورد نیاز
            language (str): زبان برنامه‌نویسی
            
        Returns:
            str: کد تولید شده
        """
        try:
            # آماده‌سازی درخواست جامع‌تر با جزئیات بیشتر
            if language.lower() == "python":
                prompt = f"""لطفاً کد پایتون برای مورد زیر بنویسید:
                
                {description}
                
                ملاحظات مهم:
                * کد باید بهینه، خوانا و با توضیحات کامل فارسی باشد
                * استفاده از بهترین شیوه‌های برنامه‌نویسی پایتون (PEP 8)
                * به‌کارگیری تایپ‌هینت‌ها برای پارامترها و مقادیر برگشتی
                * پیاده‌سازی مدیریت خطا با try/except
                * نوشتن docstring برای توابع و کلاس‌ها
                * استفاده از ساختارهای داده مناسب
                * تقسیم‌بندی کد به توابع کوچک و قابل مدیریت
                * بهینه‌سازی کارایی در صورت امکان
                
                کد را درون بلوک ```python قرار دهید.
                """
            
            elif language.lower() in ["js", "javascript"]:
                prompt = f"""لطفاً کد جاوااسکریپت برای مورد زیر بنویسید:
                
                {description}
                
                ملاحظات مهم:
                * کد باید با استانداردهای مدرن ES6+ باشد
                * استفاده از async/await به جای Promise.then() در صورت امکان
                * استفاده از ساختارهای مدرن مثل destructuring، arrow functions و spread syntax
                * افزودن توضیحات کامل برای توابع و بخش‌های پیچیده
                * استفاده از let و const به جای var
                * مدیریت خطا با try/catch
                * تقسیم‌بندی کد به توابع کوچک و قابل مدیریت
                
                کد را درون بلوک ```javascript قرار دهید.
                """
            
            elif language.lower() == "java":
                prompt = f"""لطفاً کد جاوا برای مورد زیر بنویسید:
                
                {description}
                
                ملاحظات مهم:
                * استفاده از اصول شیء‌گرایی مناسب
                * رعایت قراردادهای نام‌گذاری جاوا
                * استفاده از Generics در صورت نیاز
                * افزودن JavaDoc برای کلاس‌ها و متدها
                * استفاده از مدیریت خطا و استثناها
                * طراحی مناسب کلاس‌ها و اینترفیس‌ها
                * بهینه‌سازی کارایی در صورت امکان
                
                کد را درون بلوک ```java قرار دهید.
                """
            
            else:
                # برای سایر زبان‌ها
                prompt = f"""لطفاً کد {language} برای مورد زیر بنویسید:
                
                {description}
                
                ملاحظات مهم:
                * کد باید بهینه، خوانا و با توضیحات کامل باشد
                * استفاده از بهترین شیوه‌های برنامه‌نویسی {language}
                * پیاده‌سازی مدیریت خطا
                * افزودن توضیحات مناسب
                * تقسیم‌بندی کد به بخش‌های قابل مدیریت
                * بهینه‌سازی کارایی در صورت امکان
                
                کد را درون بلوک ```{language} قرار دهید.
                """
            
            # درخواست با استفاده از مدل زبانی
            response = self.language_model.get_openai_completion(prompt)
            
            # پردازش نتیجه
            if "```" in response:
                code_pattern = r"```(?:\w+)?\s*([\s\S]*?)\s*```"
                matches = re.findall(code_pattern, response)
                if matches:
                    code = matches[0].strip()
                    
                    # در صورت نیاز، اضافه کردن توضیحات اضافی
                    header = f"""#!/usr/bin/env {language.lower()}
# -*- coding: utf-8 -*-
\"\"\"
تولید شده توسط سیستم هوش مصنوعی نیورا
تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
توضیحات: {description[:80]}{'...' if len(description) > 80 else ''}
\"\"\"

"""
                    
                    # اضافه کردن header فقط برای پایتون
                    if language.lower() == "python" and not code.startswith("#!"):
                        code = header + code
                    
                    return code
            
            # اگر کد در بلوک کد نباشد، کل پاسخ را برگردانیم
            return response
        
        except Exception as e:
            logger.error(f"خطا در تولید کد: {str(e)}")
            logger.error(traceback.format_exc())
            
            # در صورت خطا، استفاده از روش ساده‌تر
            simple_prompt = f"لطفاً کد {language} برای {description} بنویسید. کد باید بهینه و خوانا باشد."
            try:
                return self.language_model.get_openai_completion(simple_prompt)
            except:
                return f"خطا در تولید کد: {str(e)}"
    
    def improve_self(self) -> Dict:
        """
        خودبهسازی و بهبود عملکرد سیستم با یادگیری پیشرفته و خودترمیمی
        
        Returns:
            Dict: نتایج بهبود
        """
        improvements = {
            "applied_improvements": [],
            "new_skills": [],
            "optimized_systems": [],
            "refined_models": [],
            "optimized_memory": False
        }
        
        try:
            # 1. بهینه‌سازی حافظه
            memory_improvements = self._optimize_memory()
            if memory_improvements:
                improvements["optimized_memory"] = True
                improvements["applied_improvements"].extend(memory_improvements)
            
            # 2. بهبود مهارت‌ها بر اساس تاریخچه تحلیل
            skill_improvements = self._optimize_skills_advanced()
            if skill_improvements:
                improvements["applied_improvements"].extend(skill_improvements)
            
            # 3. بررسی و ترمیم سلامت سیستم
            system_status = self._check_system_status()
            if system_status.get("needs_repair", False):
                self._self_repair(system_status.get("issues", []))
                improvements["applied_improvements"].append("ترمیم مشکلات سیستم")
                improvements["optimized_systems"].extend([f"ترمیم: {issue}" for issue in system_status.get("issues", [])])
            
            # 4. بهینه‌سازی مدل‌های یادگیری ماشین
            if hasattr(self, 'ai_engine'):
                # بررسی و بهبود مدل‌های یادگیری ماشین موجود
                for model_name, model in self.ai_engine.models.get("price_prediction", {}).items():
                    if model is not None:
                        # بهبود مدل با تنظیم پارامترها بر اساس عملکرد
                        current_metrics = self.ai_engine.model_metrics.get(model_name, {})
                        rmse = current_metrics.get("rmse", 0)
                        r2 = current_metrics.get("r2", 0)
                        
                        if hasattr(model, "n_estimators") and rmse > 0:
                            # افزایش تعداد درخت‌ها برای کاهش خطا
                            if rmse > 10 and model.n_estimators < 150:
                                model.n_estimators = min(200, model.n_estimators + 25)
                                improvements["refined_models"].append(f"بهبود مدل {model_name} با افزایش تعداد درخت‌ها")
                        
                        if hasattr(model, "max_depth") and r2 < 0.7:
                            # افزایش عمق درخت برای بهبود قدرت پیش‌بینی
                            if model.max_depth < 15:
                                model.max_depth += 1
                                improvements["refined_models"].append(f"بهبود مدل {model_name} با افزایش عمق")
                
                improvements["applied_improvements"].append("بهینه‌سازی مدل‌های یادگیری ماشین")
            
            # 5. تحلیل و یادگیری از تعاملات اخیر
            if len(self.memory["conversations"]) > 5:
                recent_conversations = self.memory["conversations"][-10:]
                self._learn_from_conversations(recent_conversations)
                improvements["applied_improvements"].append("یادگیری از تعاملات اخیر")
            
            # 6. بهینه‌سازی الگوهای تشخیص پرسش
            if "learned_patterns" in self.memory and len(self.memory["learned_patterns"]) > 0:
                # اطلاعات الگوهای یادگرفته شده
                pattern_count = sum(len(patterns) for patterns in self.memory["learned_patterns"].values())
                improvements["applied_improvements"].append(f"بهینه‌سازی {pattern_count} الگوی شناسایی پرسش")
            
            # 7. بررسی و راه‌اندازی مجدد ترد‌های غیرفعال
            active_threads = sum(1 for t in self.threads if t.is_alive())
            if active_threads < 2:  # تعداد مورد انتظار
                self._restart_dead_threads()
                improvements["applied_improvements"].append("راه‌اندازی مجدد ترد‌های غیرفعال")
            
            # ذخیره تغییرات نهایی
            self._save_memory()
            
            # 8. گزارش نهایی بهبودها
            if not improvements["applied_improvements"]:
                improvements["applied_improvements"].append("سیستم در وضعیت بهینه قرار دارد")
            
            return improvements
        
        except Exception as e:
            logger.error(f"خطا در بهبود خود: {str(e)}")
            logger.error(traceback.format_exc())
            
            # در صورت خطا، حداقل بهینه‌سازی‌های اساسی را انجام دهیم
            if len(self.memory["conversations"]) > 200:
                self.memory["conversations"] = self.memory["conversations"][-100:]
                improvements["optimized_memory"] = True
                improvements["applied_improvements"].append("بهینه‌سازی حافظه (حالت اضطراری)")
            
            self._save_memory()
            return improvements
    
    def _optimize_memory(self) -> List[str]:
        """
        بهینه‌سازی پیشرفته حافظه سیستم
        
        Returns:
            List[str]: لیست بهبودهای انجام شده
        """
        improvements = []
        
        # بهینه‌سازی تاریخچه تعاملات
        if len(self.memory["conversations"]) > 200:
            # نگهداری 100 مورد آخر
            old_size = len(self.memory["conversations"])
            self.memory["conversations"] = self.memory["conversations"][-100:]
            improvements.append(f"بهینه‌سازی حافظه تعاملات (کاهش از {old_size} به {len(self.memory['conversations'])} مورد)")
        
        # بهینه‌سازی تاریخچه تحلیل‌ها
        if len(self.memory["analysis_history"]) > 150:
            old_size = len(self.memory["analysis_history"])
            # نگهداری تحلیل‌های مهم و جدید
            self.memory["analysis_history"] = self.memory["analysis_history"][-75:]
            improvements.append(f"بهینه‌سازی تاریخچه تحلیل‌ها (کاهش از {old_size} به {len(self.memory['analysis_history'])} مورد)")
        
        # حذف تکرارها در الگوهای یادگرفته شده
        if "learned_patterns" in self.memory:
            for category, patterns in self.memory["learned_patterns"].items():
                if len(patterns) > 20:
                    old_count = len(patterns)
                    # حذف دوتایی‌های مشابه
                    unique_patterns = list(set(patterns))
                    self.memory["learned_patterns"][category] = unique_patterns
                    if len(unique_patterns) < old_count:
                        improvements.append(f"حذف الگوهای تکراری در دسته {category} (کاهش از {old_count} به {len(unique_patterns)})")
        
        return improvements
    
    def _optimize_skills_advanced(self) -> List[str]:
        """
        بهینه‌سازی پیشرفته مهارت‌ها
        
        Returns:
            List[str]: لیست بهبودهای انجام شده
        """
        improvements = []
        
        # 1. بررسی عملکرد مهارت‌ها بر اساس تاریخچه
        skill_usage = {}
        skill_success = {}
        
        # شمارش استفاده از هر مهارت و موارد موفق
        for interaction in self.memory["conversations"]:
            query_type = interaction.get("query_type")
            if query_type in self.skills:
                # افزایش شمارنده استفاده
                skill_usage[query_type] = skill_usage.get(query_type, 0) + 1
                
                # بررسی خطاها در پاسخ
                if "خطا" not in interaction.get("response", "").lower()[:50]:
                    skill_success[query_type] = skill_success.get(query_type, 0) + 1
        
        # 2. بهبود مهارت‌ها بر اساس میزان استفاده و موفقیت
        for skill_name, usage_count in skill_usage.items():
            if skill_name in self.skills:
                # محاسبه نرخ موفقیت
                success_rate = skill_success.get(skill_name, 0) / max(1, usage_count)
                current_confidence = self.skills[skill_name]["confidence"]
                
                if usage_count > 5:
                    # تنظیم اطمینان بر اساس نرخ موفقیت
                    if success_rate > 0.8 and current_confidence < 0.95:
                        new_confidence = min(0.95, current_confidence + 0.03)
                        self.skills[skill_name]["confidence"] = new_confidence
                        improvements.append(f"افزایش اطمینان در مهارت {skill_name} به {new_confidence:.2f} (نرخ موفقیت: {success_rate:.2f})")
                    elif success_rate < 0.5 and current_confidence > 0.6:
                        new_confidence = max(0.6, current_confidence - 0.02)
                        self.skills[skill_name]["confidence"] = new_confidence
                        improvements.append(f"کاهش اطمینان در مهارت {skill_name} به {new_confidence:.2f} (نرخ موفقیت: {success_rate:.2f})")
        
        # 3. بررسی تاریخچه تحلیل برای بهبود مهارت‌های خاص
        analysis_types = {}
        for analysis in self.memory["analysis_history"]:
            analysis_type = analysis.get("type")
            if analysis_type:
                analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
        
        # اگر تحلیل خاصی بیشتر استفاده شده، اطمینان آن را افزایش دهیم
        for analysis_type, count in analysis_types.items():
            if analysis_type in self.skills and count > 10:
                current_confidence = self.skills[analysis_type]["confidence"]
                if current_confidence < 0.9:
                    new_confidence = min(0.9, current_confidence + 0.02)
                    self.skills[analysis_type]["confidence"] = new_confidence
                    improvements.append(f"افزایش اطمینان در تحلیل {analysis_type} به {new_confidence:.2f} (استفاده: {count} بار)")
        
        return improvements
    
    def _check_system_status(self) -> Dict[str, Any]:
        """
        بررسی وضعیت سلامت سیستم
        
        Returns:
            Dict[str, Any]: وضعیت سیستم
        """
        status = {
            "status": "healthy",
            "needs_repair": False,
            "issues": [],
            "memory_usage": 0,
            "active_threads": 0,
            "model_status": {}
        }
        
        try:
            # بررسی حافظه
            memory_size = 0
            if hasattr(self, 'memory'):
                memory_size = len(json.dumps(self.memory))
                status["memory_usage"] = memory_size
                
                if memory_size > 5_000_000:  # 5MB
                    status["issues"].append("حافظه بیش از حد بزرگ")
                    status["needs_repair"] = True
            
            # بررسی ترد‌ها
            active_threads = 0
            if hasattr(self, 'threads'):
                active_threads = sum(1 for t in self.threads if t.is_alive())
                status["active_threads"] = active_threads
                
                if len(self.threads) > 0 and active_threads < len(self.threads):
                    status["issues"].append("ترد‌های غیرفعال")
                    status["needs_repair"] = True
            
            # بررسی اتصال به مدل‌های زبانی
            language_model_status = {}
            if hasattr(self, 'language_model'):
                # بررسی دسترسی به OpenAI
                language_model_status["openai"] = len(self.language_model.openai_api_key) > 0
                
                # بررسی دسترسی به Anthropic
                language_model_status["anthropic"] = len(self.language_model.anthropic_api_key) > 0
            
            status["model_status"] = language_model_status
            
            # بررسی ساختار حافظه
            if hasattr(self, 'memory'):
                required_keys = ["conversations", "analysis_history", "patterns", "config"]
                for key in required_keys:
                    if key not in self.memory:
                        status["issues"].append(f"کلید {key} در حافظه وجود ندارد")
                        status["needs_repair"] = True
            
            # بررسی خطاهای تکراری اخیر
            if hasattr(self, 'memory') and "conversations" in self.memory:
                recent_conversations = self.memory["conversations"][-10:] if len(self.memory["conversations"]) > 10 else self.memory["conversations"]
                error_count = sum(1 for c in recent_conversations if "خطا" in c.get("response", "").lower()[:50])
                
                if error_count > 5:
                    status["issues"].append("خطاهای تکراری")
                    status["needs_repair"] = True
            
            # بررسی کلی سلامت
            if status["needs_repair"]:
                status["status"] = "needs_repair"
            
            return status
            
        except Exception as e:
            logger.error(f"خطا در بررسی وضعیت سیستم: {str(e)}")
            logger.error(traceback.format_exc())
            
            # وضعیت پیش‌فرض در صورت خطا
            return {
                "status": "error",
                "needs_repair": True,
                "issues": ["خطا در بررسی وضعیت"],
                "error": str(e)
            }
    
    def _self_repair(self, issues: List[str]) -> bool:
        """
        ترمیم خودکار مشکلات سیستم
        
        Args:
            issues (List[str]): لیست مشکلات شناسایی شده
            
        Returns:
            bool: وضعیت موفقیت ترمیم
        """
        success = True
        
        try:
            logger.info(f"شروع ترمیم خودکار برای مشکلات: {issues}")
            
            for issue in issues:
                if issue == "حافظه بیش از حد بزرگ":
                    # بهینه‌سازی حافظه
                    self._optimize_memory()
                    logger.info("حافظه بهینه‌سازی شد.")
                
                elif issue == "ترد‌های غیرفعال":
                    # راه‌اندازی مجدد ترد‌ها
                    self._restart_dead_threads()
                    logger.info("ترد‌های غیرفعال مجددا راه‌اندازی شدند.")
                
                elif "کلید" in issue and "در حافظه وجود ندارد":
                    # اضافه کردن کلیدهای مفقود به حافظه
                    missing_key = issue.split(" ")[1]
                    if missing_key == "conversations" and missing_key not in self.memory:
                        self.memory["conversations"] = []
                    elif missing_key == "analysis_history" and missing_key not in self.memory:
                        self.memory["analysis_history"] = []
                    elif missing_key == "patterns" and missing_key not in self.memory:
                        self.memory["patterns"] = {}
                    elif missing_key == "config" and missing_key not in self.memory:
                        self.memory["config"] = {"version": "1.5.0", "last_update": str(datetime.now())}
                    
                    logger.info(f"کلید {missing_key} به حافظه اضافه شد.")
                
                elif issue == "خطاهای تکراری":
                    # بازنشانی وضعیت سیستم
                    self._reset_error_state()
                    logger.info("وضعیت خطا بازنشانی شد.")
                
                else:
                    logger.warning(f"روش ترمیم برای مشکل '{issue}' پیدا نشد.")
                    success = False
            
            # ذخیره تغییرات
            self._save_memory()
            
            return success
            
        except Exception as e:
            logger.error(f"خطا در ترمیم خودکار: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _restart_dead_threads(self) -> None:
        """راه‌اندازی مجدد ترد‌های غیرفعال"""
        if not hasattr(self, 'threads'):
            self.threads = []
        
        # بررسی و راه‌اندازی مجدد ترد یادگیری مداوم
        continuous_learning_is_alive = False
        for thread in self.threads:
            if thread.name == "continuous_learning" and thread.is_alive():
                continuous_learning_is_alive = True
                break
        
        if not continuous_learning_is_alive:
            # راه‌اندازی ترد یادگیری مداوم
            import threading
            continuous_learning_thread = threading.Thread(
                target=self._continuous_learning_loop,
                name="continuous_learning",
                daemon=True
            )
            continuous_learning_thread.start()
            self.threads.append(continuous_learning_thread)
            logger.info("ترد یادگیری مداوم مجددا راه‌اندازی شد.")
    
    def _continuous_learning_loop(self) -> None:
        """حلقه یادگیری مداوم برای بهبود خودکار سیستم"""
        while True:
            try:
                # بررسی وضعیت سیستم
                status = self._check_system_status()
                
                # ترمیم مشکلات در صورت نیاز
                if status.get("needs_repair", False):
                    self._self_repair(status.get("issues", []))
                
                # بهبود خودکار در بازه‌های زمانی مشخص
                self.improve_self()
                
                # استراحت
                time.sleep(3600)  # هر یک ساعت
                
            except Exception as e:
                logger.error(f"خطا در حلقه یادگیری مداوم: {str(e)}")
                time.sleep(300)  # استراحت کوتاه در صورت خطا
    
    def _reset_error_state(self) -> None:
        """بازنشانی وضعیت خطا"""
        # پاکسازی خطاهای ذخیره شده
        if hasattr(self, 'error_count'):
            self.error_count = 0
        
        # بازنشانی پرچم‌های خطا
        if hasattr(self, 'last_error'):
            self.last_error = None
        
        # بازنشانی زمان آخرین خطا
        if hasattr(self, 'last_error_time'):
            self.last_error_time = None
    
    def _learn_from_conversations(self, conversations: List[Dict]) -> None:
        """
        یادگیری از تعاملات با کاربر
        
        Args:
            conversations (List[Dict]): لیست تعاملات اخیر
        """
        # ایجاد مکان ذخیره‌سازی الگوهای یادگرفته شده در صورت نیاز
        if "learned_patterns" not in self.memory:
            self.memory["learned_patterns"] = {
                "market_analysis": [],
                "price_prediction": [],
                "technical_indicators": [],
                "trading_strategy": [],
                "command": [],
                "general": []
            }
        
        # بررسی تعاملات
        for conv in conversations:
            query = conv.get("query", "")
            response = conv.get("response", "")
            query_type = conv.get("query_type", "general")
            
            # یادگیری الگوهای پرسش
            if len(query) > 10 and query_type in self.memory["learned_patterns"]:
                # اضافه کردن الگو به حافظه
                patterns = self.memory["learned_patterns"][query_type]
                if query not in patterns:
                    patterns.append(query)
                    if len(patterns) > 50:  # محدودیت تعداد الگوها
                        patterns.pop(0)  # حذف قدیمی‌ترین الگو
        
        # ذخیره تغییرات
        self._save_memory()
    
    def _reset_error_state(self) -> None:
        """بازنشانی وضعیت خطا"""
        # پاکسازی خطاهای ذخیره شده
        if hasattr(self, 'error_count'):
            self.error_count = 0
        
        # بازنشانی پرچم‌های خطا
        if hasattr(self, 'last_error'):
            self.last_error = None
        
        # بازنشانی زمان آخرین خطا
        if hasattr(self, 'last_error_time'):
            self.last_error_time = None
    
    def _learn_from_conversations(self, conversations: List[Dict]) -> None:
        """
        یادگیری از تعاملات با کاربر
        
        Args:
            conversations (List[Dict]): لیست تعاملات اخیر
        """
        # ایجاد مکان ذخیره‌سازی الگوهای یادگرفته شده در صورت نیاز
        if "learned_patterns" not in self.memory:
            self.memory["learned_patterns"] = {
                "market_analysis": [],
                "price_prediction": [],
                "technical_indicators": [],
                "trading_strategy": [],
                "command": [],
                "general": []
            }
        
        # بررسی تعاملات
        for conv in conversations:
            query = conv.get("query", "")
            response = conv.get("response", "")
            query_type = conv.get("query_type", "general")
            
            # یادگیری الگوهای پرسش
            if len(query) > 10 and query_type in self.memory["learned_patterns"]:
                # اضافه کردن الگو به حافظه
                patterns = self.memory["learned_patterns"][query_type]
                if query not in patterns:
                    patterns.append(query)
                    if len(patterns) > 50:  # محدودیت تعداد الگوها
                        patterns.pop(0)  # حذف قدیمی‌ترین الگو
        
        # ذخیره تغییرات
        self._save_memory()
    
    def process_query(self, query: str) -> str:
        """
        پردازش پرسش کاربر
        
        Args:
            query (str): پرسش کاربر
            
        Returns:
            str: پاسخ به پرسش
        """
        self.last_query = query
        start_time = time.time()
        
        try:
            # شناسایی نوع پرسش
            query_type = self._categorize_query(query)
            
            # پردازش بر اساس نوع پرسش
            if query_type == "market_analysis":
                # استخراج نماد ارز و تایم‌فریم
                symbol = self._extract_symbol_from_query(query) or "BTC/USDT"
                timeframe = self._extract_timeframe_from_query(query) or "1d"
                response = self.analyze_market(symbol, timeframe)
            
            elif query_type == "price_prediction":
                # استخراج نماد ارز و تایم‌فریم
                symbol = self._extract_symbol_from_query(query) or "BTC/USDT"
                timeframe = self._extract_timeframe_from_query(query) or "1d"
                days = self._extract_days_from_query(query) or 7
                response = self.predict_price(symbol, timeframe, days)
            
            elif query_type == "technical_indicators":
                # پاسخ درباره اندیکاتورها
                indicator = self._extract_indicator_from_query(query)
                if indicator:
                    response = self.language_model.get_openai_completion(f"توضیح اندیکاتور {indicator}")
                else:
                    response = self.language_model.get_openai_completion(query)
            
            elif query_type == "trading_strategy":
                # پاسخ درباره استراتژی معاملاتی
                symbol = self._extract_symbol_from_query(query) or "BTC/USDT"
                timeframe = self._extract_timeframe_from_query(query) or "1d"
                response = self.language_model.get_openai_completion(f"استراتژی معاملاتی برای {symbol} در تایم‌فریم {timeframe}")
            
            elif query_type == "command":
                # پردازش دستورات
                if "بهبود" in query or "ارتقا" in query:
                    improvements = self.improve_self()
                    response = f"بهبود سیستم انجام شد:\n" + "\n".join([f"- {imp}" for imp in improvements["applied_improvements"]])
                elif "کد" in query:
                    language = "python"
                    if "جاوا" in query:
                        language = "java"
                    elif "جاوااسکریپت" in query:
                        language = "javascript"
                    response = self.generate_code(query, language)
                else:
                    response = self.language_model.get_openai_completion(query)
            
            else:
                # پاسخ عمومی
                response = self.language_model.get_openai_completion(query)
            
            # ذخیره تعامل
            self.memory["conversations"].append({
                "query": query,
                "response": response,
                "query_type": query_type,
                "timestamp": str(datetime.now()),
                "response_time": time.time() - start_time
            })
            
            # بروزرسانی حافظه
            self._save_memory()
            
            self.last_response = response
            return response
            
        except Exception as e:
            error_msg = f"خطا در پردازش پرسش: {str(e)}"
            logger.error(error_msg)
            self.last_response = error_msg
            
            # تلاش برای ترمیم
            try:
                return f"با عرض پوزش، خطایی رخ داد. در حال ترمیم...\n\n{self.language_model.get_openai_completion(query)}"
            except:
                return "متأسفانه در پاسخ‌دهی خطایی رخ داد. لطفاً دوباره تلاش کنید."
    
    def _extract_symbol_from_query(self, query: str) -> Optional[str]:
        """
        استخراج نماد ارز از پرسش
        
        Args:
            query (str): متن پرسش
            
        Returns:
            Optional[str]: نماد ارز یا None
        """
        # لیست نمادهای رایج
        common_symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "AVAX", "LINK", "DOGE", "SHIB", "MATIC", "LTC"]
        
        query_upper = query.upper()
        
        # بررسی نمادهای رایج در متن
        for symbol in common_symbols:
            if symbol in query_upper:
                return f"{symbol}/USDT"
        
        # تطبیق الگو برای نماد/ارز
        symbol_pattern = r"\b([A-Za-z]{2,10})/([A-Za-z]{2,10})\b"
        matches = re.findall(symbol_pattern, query_upper)
        if matches:
            return f"{matches[0][0]}/{matches[0][1]}"
        
        # چک کردن نام‌های فارسی ارزها
        persian_names = {
            "بیتکوین": "BTC/USDT",
            "بیت کوین": "BTC/USDT",
            "اتریوم": "ETH/USDT",
            "سولانا": "SOL/USDT",
            "ریپل": "XRP/USDT",
            "کاردانو": "ADA/USDT",
            "دوج کوین": "DOGE/USDT",
            "شیبا": "SHIB/USDT"
        }
        
        for name, symbol in persian_names.items():
            if name in query.lower():
                return symbol
        
        return None
    
    def _extract_timeframe_from_query(self, query: str) -> Optional[str]:
        """
        استخراج تایم‌فریم از پرسش
        
        Args:
            query (str): متن پرسش
            
        Returns:
            Optional[str]: تایم‌فریم یا None
        """
        query_lower = query.lower()
        
        # تایم‌فریم‌های استاندارد
        timeframes = {
            "1m": ["یک دقیقه", "1 دقیقه", "دقیقه ای", "1m"],
            "5m": ["پنج دقیقه", "5 دقیقه", "5m"],
            "15m": ["پانزده دقیقه", "15 دقیقه", "ربع ساعت", "15m"],
            "30m": ["سی دقیقه", "30 دقیقه", "نیم ساعت", "30m"],
            "1h": ["یک ساعت", "1 ساعت", "ساعتی", "1h"],
            "4h": ["چهار ساعت", "4 ساعت", "4h"],
            "1d": ["روزانه", "یک روز", "1 روز", "1d", "daily"],
            "1w": ["هفتگی", "یک هفته", "1 هفته", "1w", "weekly"],
            "1M": ["ماهانه", "یک ماه", "1 ماه", "1M", "monthly"]
        }
        
        for tf, keywords in timeframes.items():
            if any(kw in query_lower for kw in keywords):
                return tf
        
        # دسته‌بندی‌های کلی
        if any(kw in query_lower for kw in ["کوتاه مدت", "کوتاه‌مدت", "امروز", "فردا"]):
            return "1d"
        elif any(kw in query_lower for kw in ["میان مدت", "میان‌مدت", "هفته", "هفتگی"]):
            return "1w"
        elif any(kw in query_lower for kw in ["بلند مدت", "بلندمدت", "ماه", "ماهانه"]):
            return "1M"
        
        return None
    
    def _extract_days_from_query(self, query: str) -> Optional[int]:
        """
        استخراج تعداد روز از پرسش
        
        Args:
            query (str): متن پرسش
            
        Returns:
            Optional[int]: تعداد روز یا None
        """
        # تطبیق الگو برای عدد + روز
        day_pattern = r"(\d+)(?:\s*)(روز|هفته|ماه)"
        matches = re.findall(day_pattern, query)
        
        if matches:
            number, unit = matches[0]
            number = int(number)
            
            if unit == "هفته":
                return number * 7
            elif unit == "ماه":
                return number * 30
            else:  # روز
                return number
        
        return None
    
    def _extract_indicator_from_query(self, query: str) -> Optional[str]:
        """
        استخراج نام اندیکاتور از پرسش
        
        Args:
            query (str): متن پرسش
            
        Returns:
            Optional[str]: نام اندیکاتور یا None
        """
        query_lower = query.lower()
        
        # اندیکاتورهای رایج
        indicators = {
            "RSI": ["rsi", "آر اس آی", "قدرت نسبی"],
            "MACD": ["ماکد", "macd", "همگرایی واگرایی"],
            "Bollinger Bands": ["بولینگر", "bollinger", "باند بولینگر"],
            "Moving Average": ["میانگین متحرک", "moving average", "ma", "ema", "sma"],
            "Stochastic": ["استوکاستیک", "stochastic"],
            "Ichimoku": ["ایچیموکو", "ichimoku"],
            "Fibonacci": ["فیبوناچی", "fibonacci"]
        }
        
        for indicator, keywords in indicators.items():
            if any(kw in query_lower for kw in keywords):
                return indicator
        
        return None
    
    def shutdown(self):
        """خاموش کردن سیستم"""
        self.active = False
        self._save_memory()
        logger.info("نیورا با موفقیت خاموش شد.")


# کلاس واسط تلگرام برای نیورا
class NeuraTelegramInterface:
    """واسط تلگرام برای سیستم هوش مصنوعی نیورا"""
    
    def __init__(self, token, neura_instance=None):
        """
        مقداردهی اولیه واسط تلگرام
        
        Args:
            token (str): توکن ربات تلگرام
            neura_instance (NeuraAI, optional): نمونه نیورا
        """
        self.token = token
        self.neura = neura_instance or NeuraAI()
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id = 0
        self.active = True
        self.authorized_users = self._load_authorized_users()
        self.command_handlers = self._initialize_command_handlers()
        
        # راه‌اندازی ترد پردازش پیام‌ها
        self.message_thread = threading.Thread(
            target=self._process_messages_loop,
            daemon=True
        )
        self.message_thread.start()
        
        logger.info("واسط تلگرام نیورا راه‌اندازی شد.")
    
    def _load_authorized_users(self) -> List[int]:
        """
        بارگذاری لیست کاربران مجاز
        
        Returns:
            List[int]: لیست شناسه‌های کاربران مجاز
        """
        try:
            if os.path.exists("authorized_users.json"):
                with open("authorized_users.json", "r") as f:
                    users = json.load(f)
                    return users.get("users", [])
        except:
            pass
        
        # در صورت نبود فایل، بازگشت لیست خالی
        return []
    
    def _save_authorized_users(self):
        """ذخیره لیست کاربران مجاز"""
        try:
            with open("authorized_users.json", "w") as f:
                json.dump({"users": self.authorized_users}, f)
        except Exception as e:
            logger.error(f"خطا در ذخیره کاربران مجاز: {str(e)}")
    
    def _initialize_command_handlers(self) -> Dict[str, callable]:
        """
        مقداردهی پردازنده‌های دستورات
        
        Returns:
            Dict[str, callable]: دیکشنری دستورات و توابع پردازنده
        """
        return {
            "/start": self._handle_start_command,
            "/help": self._handle_help_command,
            "/status": self._handle_status_command,
            "/analyze": self._handle_analyze_command,
            "/predict": self._handle_predict_command,
            "/authorize": self._handle_authorize_command
        }
    
    def _process_messages_loop(self):
        """حلقه پردازش پیام‌های دریافتی"""
        while self.active:
            try:
                self._process_new_messages()
                time.sleep(1)  # تأخیر برای کاهش فشار بر API
            except Exception as e:
                logger.error(f"خطا در حلقه پردازش پیام‌ها: {str(e)}")
                time.sleep(5)  # تأخیر بیشتر در صورت خطا
    
    def _process_new_messages(self):
        """پردازش پیام‌های جدید"""
        # دریافت به‌روزرسانی‌ها
        updates = self._get_updates()
        
        for update in updates:
            # به‌روزرسانی آخرین شناسه
            if update["update_id"] > self.last_update_id:
                self.last_update_id = update["update_id"]
            
            # پردازش پیام
            if "message" in update:
                self._process_message(update["message"])
    
    def _get_updates(self) -> List[Dict]:
        """
        دریافت به‌روزرسانی‌های جدید از API تلگرام
        
        Returns:
            List[Dict]: لیست به‌روزرسانی‌ها
        """
        try:
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30
            }
            response = requests.get(f"{self.api_url}/getUpdates", params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok", False):
                    return data.get("result", [])
        except Exception as e:
            logger.error(f"خطا در دریافت به‌روزرسانی‌ها: {str(e)}")
        
        return []
    
    def _process_message(self, message: Dict):
        """
        پردازش پیام دریافتی
        
        Args:
            message (Dict): پیام دریافتی
        """
        try:
            # دریافت اطلاعات پیام
            chat_id = message.get("chat", {}).get("id")
            user_id = message.get("from", {}).get("id")
            text = message.get("text", "")
            
            if not text or not chat_id:
                return
            
            # بررسی مجاز بودن کاربر
            if not self._is_user_authorized(user_id) and not text.startswith("/start"):
                self._send_message(chat_id, "شما مجاز به استفاده از این ربات نیستید.")
                return
            
            # پردازش دستورات
            if text.startswith("/"):
                self._process_command(chat_id, user_id, text)
            else:
                # پردازش پرسش عادی
                response = self.neura.process_query(text)
                self._send_message(chat_id, response)
        except Exception as e:
            logger.error(f"خطا در پردازش پیام: {str(e)}")
    
    def _process_command(self, chat_id: int, user_id: int, text: str):
        """
        پردازش دستور
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            text (str): متن دستور
        """
        # جداسازی دستور و پارامترها
        parts = text.split()
        command = parts[0].lower()
        params = parts[1:] if len(parts) > 1 else []
        
        # یافتن پردازنده مناسب
        handler = self.command_handlers.get(command)
        if handler:
            handler(chat_id, user_id, params)
        else:
            self._send_message(chat_id, "دستور نامعتبر است. برای دیدن لیست دستورات، /help را بفرستید.")
    
    def _handle_start_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور شروع
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        # اضافه کردن کاربر به لیست مجاز در صورتی که لیست خالی باشد
        if not self.authorized_users:
            self.authorized_users.append(user_id)
            self._save_authorized_users()
        
        welcome_message = f"""
        به ربات هوش مصنوعی نیورا خوش آمدید!
        
        نیورا یک سیستم هوش مصنوعی پیشرفته برای تحلیل بازار ارزهای دیجیتال است.
        
        برای دیدن لیست دستورات، /help را بفرستید.
        """
        
        self._send_message(chat_id, welcome_message)
    
    def _handle_help_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور راهنما
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        help_message = """
        راهنمای دستورات ربات نیورا:
        
        /start - شروع کار با ربات
        /help - نمایش این راهنما
        /status - نمایش وضعیت سیستم
        /analyze (symbol) (timeframe) - تحلیل بازار ارز مشخص شده
        /predict (symbol) (timeframe) (days) - پیش‌بینی قیمت
        
        مثال‌ها:
        /analyze BTC/USDT 1d
        /predict ETH/USDT 4h 7
        
        همچنین می‌توانید سؤالات خود را به صورت مستقیم بپرسید.
        """
        
        self._send_message(chat_id, help_message)
    
    def _handle_status_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور وضعیت
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        status = self.neura._check_system_status()
        
        status_message = f"""
        وضعیت سیستم نیورا:
        
        نام: {self.neura.name}
        نسخه: {self.neura.version}
        وضعیت سلامت: {status['health']}
        تعداد مکالمات ذخیره شده: {len(self.neura.memory['conversations'])}
        تعداد تحلیل‌های ذخیره شده: {len(self.neura.memory['analysis_history'])}
        ترد‌های فعال: {status['active_threads']}
        
        زمان بررسی: {status['timestamp']}
        """
        
        self._send_message(chat_id, status_message)
    
    def _handle_analyze_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور تحلیل
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        # بررسی پارامترها
        if len(params) < 1:
            self._send_message(chat_id, "لطفاً نماد ارز را مشخص کنید. مثال: /analyze BTC/USDT 1d")
            return
        
        # استخراج پارامترها
        symbol = params[0]
        timeframe = params[1] if len(params) > 1 else "1d"
        
        # ارسال پیام در حال پردازش
        self._send_message(chat_id, f"در حال تحلیل {symbol} در تایم‌فریم {timeframe}...")
        
        # انجام تحلیل
        analysis = self.neura.analyze_market(symbol, timeframe)
        
        # ارسال نتیجه
        self._send_message(chat_id, analysis)
    
    def _handle_predict_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور پیش‌بینی
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        # بررسی پارامترها
        if len(params) < 1:
            self._send_message(chat_id, "لطفاً نماد ارز را مشخص کنید. مثال: /predict BTC/USDT 1d 7")
            return
        
        # استخراج پارامترها
        symbol = params[0]
        timeframe = params[1] if len(params) > 1 else "1d"
        days = int(params[2]) if len(params) > 2 else 7
        
        # ارسال پیام در حال پردازش
        self._send_message(chat_id, f"در حال پیش‌بینی قیمت {symbol} برای {days} روز آینده...")
        
        # انجام پیش‌بینی
        prediction = self.neura.predict_price(symbol, timeframe, days)
        
        # ارسال نتیجه
        self._send_message(chat_id, prediction)
    
    def _handle_authorize_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور مجوز
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        # بررسی اجازه ادمین
        if not self._is_admin(user_id):
            self._send_message(chat_id, "شما مجاز به استفاده از این دستور نیستید.")
            return
        
        # بررسی پارامترها
        if len(params) < 1:
            self._send_message(chat_id, "لطفاً شناسه کاربر را مشخص کنید. مثال: /authorize 123456789")
            return
        
        try:
            # اضافه کردن کاربر به لیست مجاز
            new_user_id = int(params[0])
            if new_user_id not in self.authorized_users:
                self.authorized_users.append(new_user_id)
                self._save_authorized_users()
                self._send_message(chat_id, f"کاربر با شناسه {new_user_id} به لیست کاربران مجاز اضافه شد.")
            else:
                self._send_message(chat_id, "این کاربر قبلاً در لیست مجاز قرار دارد.")
        except ValueError:
            self._send_message(chat_id, "شناسه کاربر باید یک عدد باشد.")
    
    def _send_message(self, chat_id: int, text: str):
        """
        ارسال پیام به کاربر
        
        Args:
            chat_id (int): شناسه چت
            text (str): متن پیام
        """
        try:
            # تقسیم پیام‌های طولانی
            if len(text) > 4000:
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                for chunk in chunks:
                    params = {
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "Markdown"
                    }
                    requests.post(f"{self.api_url}/sendMessage", json=params)
            else:
                params = {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown"
                }
                requests.post(f"{self.api_url}/sendMessage", json=params)
        except Exception as e:
            logger.error(f"خطا در ارسال پیام: {str(e)}")
    
    def _is_user_authorized(self, user_id: int) -> bool:
        """
        بررسی مجاز بودن کاربر
        
        Args:
            user_id (int): شناسه کاربر
            
        Returns:
            bool: وضعیت مجاز بودن
        """
        # اگر لیست مجاز خالی باشد، همه کاربران مجاز هستند
        if not self.authorized_users:
            return True
        
        return user_id in self.authorized_users
    
    def _is_admin(self, user_id: int) -> bool:
        """
        بررسی ادمین بودن کاربر
        
        Args:
            user_id (int): شناسه کاربر
            
        Returns:
            bool: وضعیت ادمین بودن
        """
        # اولین کاربر در لیست، ادمین است
        return len(self.authorized_users) > 0 and self.authorized_users[0] == user_id
    
    def shutdown(self):
        """خاموش کردن واسط تلگرام"""
        self.active = False
        self.neura.shutdown()
        logger.info("واسط تلگرام نیورا با موفقیت خاموش شد.")


def start_neura_telegram_bot(token: str):
    """
    راه‌اندازی ربات تلگرام نیورا
    
    Args:
        token (str): توکن ربات تلگرام
    """
    try:
        # ایجاد نمونه نیورا
        neura = NeuraAI()
        
        # ایجاد واسط تلگرام
        telegram_interface = NeuraTelegramInterface(token, neura)
        
        logger.info("ربات تلگرام نیورا با موفقیت راه‌اندازی شد.")
        return telegram_interface
    except Exception as e:
        logger.error(f"خطا در راه‌اندازی ربات تلگرام: {str(e)}")
        return None