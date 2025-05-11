"""
ماژول معماری میکروسرویس برای سیستم تحلیل ارزهای دیجیتال

این ماژول ساختار میکروسرویس‌ها را برای بخش‌های مختلف سیستم مثل تحلیل داده‌ها، پیش‌بینی و اعلان پیاده‌سازی می‌کند.
هر میکروسرویس به صورت مستقل عمل می‌کند و از طریق رابط‌های API با سایر بخش‌ها ارتباط برقرار می‌کند.
"""

import os
import json
import logging
import threading
import time
import queue
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Callable, Union, Optional, Tuple

# تنظیم لاگ‌ها
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("microservices")

# ------------------- کلاس پایه میکروسرویس -------------------

class MicroService:
    """کلاس پایه برای همه میکروسرویس‌ها"""
    
    def __init__(self, name: str, input_queue: queue.Queue = None, output_queue: queue.Queue = None):
        """
        مقداردهی اولیه میکروسرویس
        
        Args:
            name (str): نام میکروسرویس
            input_queue (queue.Queue, optional): صف ورودی برای دریافت داده‌ها
            output_queue (queue.Queue, optional): صف خروجی برای ارسال نتایج
        """
        self.name = name
        self.input_queue = input_queue or queue.Queue()
        self.output_queue = output_queue or queue.Queue()
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(f"microservices.{name}")
        self.logger.info(f"میکروسرویس {name} ایجاد شد")
        
    def start(self):
        """راه‌اندازی میکروسرویس در یک ترد جداگانه"""
        if self.running:
            self.logger.warning(f"میکروسرویس {self.name} قبلاً در حال اجرا است")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info(f"میکروسرویس {self.name} راه‌اندازی شد")
        
    def stop(self):
        """توقف میکروسرویس"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.logger.info(f"میکروسرویس {self.name} متوقف شد")
        
    def _run(self):
        """متد اصلی اجرای میکروسرویس که باید توسط کلاس‌های فرزند پیاده‌سازی شود"""
        raise NotImplementedError("این متد باید توسط کلاس‌های فرزند پیاده‌سازی شود")
        
    def process(self, data: Any) -> Any:
        """
        پردازش داده‌های ورودی
        
        Args:
            data (Any): داده‌های ورودی
            
        Returns:
            Any: نتیجه پردازش
        """
        raise NotImplementedError("این متد باید توسط کلاس‌های فرزند پیاده‌سازی شود")
        
    def send(self, data: Any):
        """
        ارسال داده به صف خروجی
        
        Args:
            data (Any): داده‌های خروجی
        """
        if self.output_queue:
            self.output_queue.put(data)
            
    def receive(self, timeout: float = 0.1) -> Optional[Any]:
        """
        دریافت داده از صف ورودی
        
        Args:
            timeout (float, optional): زمان انتظار
            
        Returns:
            Optional[Any]: داده‌های دریافتی یا None در صورت خالی بودن صف
        """
        try:
            return self.input_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

# ------------------- میکروسرویس‌های اصلی -------------------

class DataCollectionService(MicroService):
    """میکروسرویس جمع‌آوری داده‌ها از منابع مختلف"""
    
    def __init__(self, name: str = "data_collection", output_queue: queue.Queue = None):
        """
        مقداردهی اولیه میکروسرویس جمع‌آوری داده‌ها
        
        Args:
            name (str, optional): نام میکروسرویس
            output_queue (queue.Queue, optional): صف خروجی
        """
        super().__init__(name=name, output_queue=output_queue)
        self.data_sources = {}
        self.collection_interval = 60  # بازه جمع‌آوری داده‌ها به ثانیه
        
    def add_data_source(self, name: str, source_function: Callable, params: Dict[str, Any] = None):
        """
        افزودن منبع داده جدید
        
        Args:
            name (str): نام منبع داده
            source_function (Callable): تابع دریافت داده
            params (Dict[str, Any], optional): پارامترهای منبع داده
        """
        self.data_sources[name] = {
            "function": source_function,
            "params": params or {}
        }
        self.logger.info(f"منبع داده {name} اضافه شد")
        
    def set_collection_interval(self, interval: int):
        """
        تنظیم بازه زمانی جمع‌آوری داده‌ها
        
        Args:
            interval (int): بازه زمانی به ثانیه
        """
        self.collection_interval = interval
        
    def _run(self):
        """حلقه اصلی جمع‌آوری داده‌ها"""
        while self.running:
            try:
                for source_name, source_config in self.data_sources.items():
                    try:
                        function = source_config["function"]
                        params = source_config["params"]
                        
                        # جمع‌آوری داده از منبع
                        data = function(**params)
                        
                        # ارسال داده‌ها به صف خروجی
                        if data is not None:
                            self.send({
                                "source": source_name,
                                "timestamp": datetime.now().isoformat(),
                                "data": data
                            })
                            self.logger.info(f"داده‌های {source_name} جمع‌آوری و ارسال شد")
                    except Exception as e:
                        self.logger.error(f"خطا در جمع‌آوری داده از {source_name}: {str(e)}")
                
                # انتظار تا جمع‌آوری بعدی
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"خطا در حلقه اصلی جمع‌آوری داده‌ها: {str(e)}")
                time.sleep(5)  # انتظار کوتاه قبل از تلاش مجدد
        
    def process(self, data: Any) -> Any:
        """
        پردازش دستی داده‌ها
        
        Args:
            data (Any): داده‌های ورودی
            
        Returns:
            Any: داده‌های پردازش شده
        """
        self.send(data)
        return data


class AnalysisService(MicroService):
    """میکروسرویس تحلیل داده‌های ارزهای دیجیتال"""
    
    def __init__(self, name: str = "analysis", input_queue: queue.Queue = None, output_queue: queue.Queue = None):
        """
        مقداردهی اولیه میکروسرویس تحلیل
        
        Args:
            name (str, optional): نام میکروسرویس
            input_queue (queue.Queue, optional): صف ورودی
            output_queue (queue.Queue, optional): صف خروجی
        """
        super().__init__(name=name, input_queue=input_queue, output_queue=output_queue)
        self.analyzers = {}
        
    def add_analyzer(self, name: str, analyzer_function: Callable, params: Dict[str, Any] = None):
        """
        افزودن تحلیلگر جدید
        
        Args:
            name (str): نام تحلیلگر
            analyzer_function (Callable): تابع تحلیل
            params (Dict[str, Any], optional): پارامترهای تحلیلگر
        """
        self.analyzers[name] = {
            "function": analyzer_function,
            "params": params or {}
        }
        self.logger.info(f"تحلیلگر {name} اضافه شد")
        
    def _run(self):
        """حلقه اصلی تحلیل داده‌ها"""
        while self.running:
            try:
                # دریافت داده از صف ورودی
                data = self.receive()
                
                if data is not None:
                    # پردازش داده‌ها
                    results = self.process(data)
                    
                    # ارسال نتایج به صف خروجی
                    self.send(results)
            except Exception as e:
                self.logger.error(f"خطا در حلقه اصلی تحلیل: {str(e)}")
                time.sleep(1)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تحلیل داده‌ها با استفاده از تحلیلگرهای موجود
        
        Args:
            data (Dict[str, Any]): داده‌های ورودی
            
        Returns:
            Dict[str, Any]: نتایج تحلیل
        """
        source = data.get("source", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        input_data = data.get("data")
        
        analysis_results = {
            "source": source,
            "timestamp": timestamp,
            "input_timestamp": data.get("timestamp"),
            "results": {}
        }
        
        try:
            for analyzer_name, analyzer_config in self.analyzers.items():
                try:
                    function = analyzer_config["function"]
                    params = analyzer_config["params"]
                    
                    # انجام تحلیل
                    result = function(input_data, **params)
                    
                    # افزودن نتیجه به نتایج کلی
                    analysis_results["results"][analyzer_name] = result
                    
                except Exception as e:
                    self.logger.error(f"خطا در تحلیلگر {analyzer_name}: {str(e)}")
                    analysis_results["results"][analyzer_name] = {"error": str(e)}
            
            self.logger.info(f"تحلیل داده‌های {source} انجام شد")
            return analysis_results
        except Exception as e:
            self.logger.error(f"خطا در پردازش داده‌ها: {str(e)}")
            return {
                "source": source,
                "timestamp": timestamp,
                "error": str(e)
            }


class PredictionService(MicroService):
    """میکروسرویس پیش‌بینی قیمت ارزهای دیجیتال"""
    
    def __init__(self, name: str = "prediction", input_queue: queue.Queue = None, output_queue: queue.Queue = None):
        """
        مقداردهی اولیه میکروسرویس پیش‌بینی
        
        Args:
            name (str, optional): نام میکروسرویس
            input_queue (queue.Queue, optional): صف ورودی
            output_queue (queue.Queue, optional): صف خروجی
        """
        super().__init__(name=name, input_queue=input_queue, output_queue=output_queue)
        self.prediction_models = {}
        
    def add_model(self, name: str, model_function: Callable, params: Dict[str, Any] = None):
        """
        افزودن مدل پیش‌بینی جدید
        
        Args:
            name (str): نام مدل
            model_function (Callable): تابع پیش‌بینی
            params (Dict[str, Any], optional): پارامترهای مدل
        """
        self.prediction_models[name] = {
            "function": model_function,
            "params": params or {}
        }
        self.logger.info(f"مدل پیش‌بینی {name} اضافه شد")
        
    def _run(self):
        """حلقه اصلی پیش‌بینی"""
        while self.running:
            try:
                # دریافت داده از صف ورودی
                data = self.receive()
                
                if data is not None:
                    # پردازش داده‌ها
                    results = self.process(data)
                    
                    # ارسال نتایج به صف خروجی
                    self.send(results)
            except Exception as e:
                self.logger.error(f"خطا در حلقه اصلی پیش‌بینی: {str(e)}")
                time.sleep(1)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        پیش‌بینی با استفاده از مدل‌های موجود
        
        Args:
            data (Dict[str, Any]): داده‌های ورودی
            
        Returns:
            Dict[str, Any]: نتایج پیش‌بینی
        """
        source = data.get("source", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        input_data = data.get("data")
        analysis_results = data.get("results", {})
        
        prediction_results = {
            "source": source,
            "timestamp": timestamp,
            "input_timestamp": data.get("timestamp"),
            "predictions": {}
        }
        
        try:
            for model_name, model_config in self.prediction_models.items():
                try:
                    function = model_config["function"]
                    params = model_config["params"]
                    
                    # انجام پیش‌بینی
                    prediction = function(input_data, analysis_results, **params)
                    
                    # افزودن نتیجه به نتایج کلی
                    prediction_results["predictions"][model_name] = prediction
                    
                except Exception as e:
                    self.logger.error(f"خطا در مدل {model_name}: {str(e)}")
                    prediction_results["predictions"][model_name] = {"error": str(e)}
            
            self.logger.info(f"پیش‌بینی داده‌های {source} انجام شد")
            return prediction_results
        except Exception as e:
            self.logger.error(f"خطا در پردازش داده‌ها: {str(e)}")
            return {
                "source": source,
                "timestamp": timestamp,
                "error": str(e)
            }


class NotificationService(MicroService):
    """میکروسرویس اعلان و ارسال نتایج به کاربران"""
    
    def __init__(self, name: str = "notification", input_queue: queue.Queue = None):
        """
        مقداردهی اولیه میکروسرویس اعلان
        
        Args:
            name (str, optional): نام میکروسرویس
            input_queue (queue.Queue, optional): صف ورودی
        """
        super().__init__(name=name, input_queue=input_queue)
        self.notification_channels = {}
        
    def add_channel(self, name: str, channel_function: Callable, params: Dict[str, Any] = None):
        """
        افزودن کانال اعلان جدید
        
        Args:
            name (str): نام کانال
            channel_function (Callable): تابع ارسال اعلان
            params (Dict[str, Any], optional): پارامترهای کانال
        """
        self.notification_channels[name] = {
            "function": channel_function,
            "params": params or {}
        }
        self.logger.info(f"کانال اعلان {name} اضافه شد")
        
    def _run(self):
        """حلقه اصلی اعلان"""
        while self.running:
            try:
                # دریافت داده از صف ورودی
                data = self.receive()
                
                if data is not None:
                    # پردازش و ارسال اعلان
                    self.process(data)
            except Exception as e:
                self.logger.error(f"خطا در حلقه اصلی اعلان: {str(e)}")
                time.sleep(1)
        
    def process(self, data: Dict[str, Any]) -> None:
        """
        پردازش داده‌ها و ارسال اعلان
        
        Args:
            data (Dict[str, Any]): داده‌های ورودی
        """
        source = data.get("source", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        predictions = data.get("predictions", {})
        
        try:
            for channel_name, channel_config in self.notification_channels.items():
                try:
                    function = channel_config["function"]
                    params = channel_config["params"]
                    
                    # ارسال اعلان
                    function(source, timestamp, predictions, **params)
                    
                    self.logger.info(f"اعلان {source} از طریق کانال {channel_name} ارسال شد")
                except Exception as e:
                    self.logger.error(f"خطا در ارسال اعلان از طریق کانال {channel_name}: {str(e)}")
        except Exception as e:
            self.logger.error(f"خطا در پردازش اعلان: {str(e)}")


class BlackSwanDetectionService(MicroService):
    """میکروسرویس تشخیص رویدادهای مهم و غیرمنتظره در بازار (Black Swan Events)"""
    
    def __init__(self, name: str = "black_swan_detection", input_queue: queue.Queue = None, output_queue: queue.Queue = None):
        """
        مقداردهی اولیه میکروسرویس تشخیص رویدادهای مهم
        
        Args:
            name (str, optional): نام میکروسرویس
            input_queue (queue.Queue, optional): صف ورودی
            output_queue (queue.Queue, optional): صف خروجی
        """
        super().__init__(name=name, input_queue=input_queue, output_queue=output_queue)
        self.detection_algorithms = {}
        self.threshold = 0.8  # حد آستانه تشخیص رویداد مهم (0 تا 1)
        
    def add_algorithm(self, name: str, algorithm_function: Callable, params: Dict[str, Any] = None):
        """
        افزودن الگوریتم تشخیص جدید
        
        Args:
            name (str): نام الگوریتم
            algorithm_function (Callable): تابع تشخیص
            params (Dict[str, Any], optional): پارامترهای الگوریتم
        """
        self.detection_algorithms[name] = {
            "function": algorithm_function,
            "params": params or {}
        }
        self.logger.info(f"الگوریتم تشخیص {name} اضافه شد")
        
    def set_threshold(self, threshold: float):
        """
        تنظیم حد آستانه تشخیص
        
        Args:
            threshold (float): حد آستانه (0 تا 1)
        """
        self.threshold = max(0.0, min(1.0, threshold))
        
    def _run(self):
        """حلقه اصلی تشخیص رویدادهای مهم"""
        while self.running:
            try:
                # دریافت داده از صف ورودی
                data = self.receive()
                
                if data is not None:
                    # پردازش داده‌ها
                    results = self.process(data)
                    
                    # ارسال نتایج به صف خروجی اگر رویداد مهمی تشخیص داده شده باشد
                    if results.get("is_black_swan", False):
                        self.send(results)
            except Exception as e:
                self.logger.error(f"خطا در حلقه اصلی تشخیص رویدادهای مهم: {str(e)}")
                time.sleep(1)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        بررسی داده‌ها برای تشخیص رویدادهای مهم
        
        Args:
            data (Dict[str, Any]): داده‌های ورودی
            
        Returns:
            Dict[str, Any]: نتایج تشخیص
        """
        source = data.get("source", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        input_data = data.get("data")
        
        detection_results = {
            "source": source,
            "timestamp": timestamp,
            "input_timestamp": data.get("timestamp"),
            "is_black_swan": False,
            "detection_score": 0.0,
            "algorithm_results": {}
        }
        
        try:
            total_score = 0.0
            algorithm_count = 0
            
            for algorithm_name, algorithm_config in self.detection_algorithms.items():
                try:
                    function = algorithm_config["function"]
                    params = algorithm_config["params"]
                    
                    # اجرای الگوریتم تشخیص
                    result = function(input_data, **params)
                    
                    # بررسی نتیجه
                    if isinstance(result, dict) and "score" in result:
                        score = result["score"]
                        total_score += score
                        algorithm_count += 1
                    else:
                        score = 0.0
                    
                    # افزودن نتیجه به نتایج کلی
                    detection_results["algorithm_results"][algorithm_name] = result
                    
                except Exception as e:
                    self.logger.error(f"خطا در الگوریتم {algorithm_name}: {str(e)}")
                    detection_results["algorithm_results"][algorithm_name] = {"error": str(e)}
            
            # محاسبه امتیاز نهایی
            if algorithm_count > 0:
                detection_results["detection_score"] = total_score / algorithm_count
                
                # تعیین وضعیت رویداد مهم
                if detection_results["detection_score"] >= self.threshold:
                    detection_results["is_black_swan"] = True
                    self.logger.warning(f"رویداد مهم در {source} با امتیاز {detection_results['detection_score']:.2f} تشخیص داده شد")
            
            return detection_results
        except Exception as e:
            self.logger.error(f"خطا در پردازش داده‌ها: {str(e)}")
            return {
                "source": source,
                "timestamp": timestamp,
                "is_black_swan": False,
                "error": str(e)
            }

# ------------------- مدیریت میکروسرویس‌ها -------------------

class MicroserviceOrchestrator:
    """مدیریت و هماهنگ‌سازی میکروسرویس‌ها"""
    
    def __init__(self):
        """مقداردهی اولیه هماهنگ‌کننده میکروسرویس‌ها"""
        self.services = {}
        self.queues = {}
        self.logger = logging.getLogger("microservices.orchestrator")
        self.logger.info("هماهنگ‌کننده میکروسرویس‌ها ایجاد شد")
        
    def create_queue(self, name: str) -> queue.Queue:
        """
        ایجاد یک صف جدید برای ارتباط بین میکروسرویس‌ها
        
        Args:
            name (str): نام صف
            
        Returns:
            queue.Queue: صف ایجاد شده
        """
        if name in self.queues:
            return self.queues[name]
            
        new_queue = queue.Queue()
        self.queues[name] = new_queue
        self.logger.info(f"صف {name} ایجاد شد")
        return new_queue
        
    def add_service(self, service: MicroService):
        """
        افزودن میکروسرویس جدید
        
        Args:
            service (MicroService): میکروسرویس
        """
        self.services[service.name] = service
        self.logger.info(f"میکروسرویس {service.name} به هماهنگ‌کننده اضافه شد")
        
    def connect_services(self, source_service_name: str, target_service_name: str, queue_name: str = None):
        """
        ایجاد ارتباط بین دو میکروسرویس
        
        Args:
            source_service_name (str): نام میکروسرویس منبع
            target_service_name (str): نام میکروسرویس مقصد
            queue_name (str, optional): نام صف ارتباطی
        """
        if source_service_name not in self.services:
            raise ValueError(f"میکروسرویس منبع {source_service_name} یافت نشد")
            
        if target_service_name not in self.services:
            raise ValueError(f"میکروسرویس مقصد {target_service_name} یافت نشد")
            
        # ایجاد صف ارتباطی
        queue_name = queue_name or f"{source_service_name}_to_{target_service_name}"
        connection_queue = self.create_queue(queue_name)
        
        # تنظیم صف‌های ورودی و خروجی میکروسرویس‌ها
        source_service = self.services[source_service_name]
        target_service = self.services[target_service_name]
        
        source_service.output_queue = connection_queue
        target_service.input_queue = connection_queue
        
        self.logger.info(f"ارتباط از {source_service_name} به {target_service_name} برقرار شد")
        
    def start_all(self):
        """راه‌اندازی تمام میکروسرویس‌ها"""
        for name, service in self.services.items():
            service.start()
            
    def stop_all(self):
        """توقف تمام میکروسرویس‌ها"""
        for name, service in self.services.items():
            service.stop()
            
    def get_service(self, name: str) -> Optional[MicroService]:
        """
        دریافت میکروسرویس با نام مشخص
        
        Args:
            name (str): نام میکروسرویس
            
        Returns:
            Optional[MicroService]: میکروسرویس یا None در صورت عدم وجود
        """
        return self.services.get(name)
        
    def get_service_status(self) -> Dict[str, bool]:
        """
        دریافت وضعیت تمام میکروسرویس‌ها
        
        Returns:
            Dict[str, bool]: دیکشنری نام میکروسرویس و وضعیت آن
        """
        return {name: service.running for name, service in self.services.items()}

# ------------------- توابع آماده برای استفاده در میکروسرویس‌ها -------------------

def get_market_data(symbol: str, timeframe: str, lookback_days: int = 30, exchange: str = "binance") -> Optional[pd.DataFrame]:
    """
    دریافت داده‌های بازار برای یک ارز دیجیتال
    
    Args:
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        lookback_days (int, optional): تعداد روزهای تاریخچه
        exchange (str, optional): نام صرافی
        
    Returns:
        Optional[pd.DataFrame]: دیتافریم داده‌های بازار یا None در صورت خطا
    """
    from crypto_data import get_crypto_data
    try:
        return get_crypto_data(symbol, timeframe, lookback_days, exchange)
    except Exception as e:
        logger.error(f"خطا در دریافت داده‌های بازار: {str(e)}")
        return None

def technical_analysis(df: pd.DataFrame, indicators: List[str] = None) -> Dict[str, Any]:
    """
    انجام تحلیل تکنیکال روی داده‌های بازار
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های بازار
        indicators (List[str], optional): لیست اندیکاتورهای موردنظر
        
    Returns:
        Dict[str, Any]: نتایج تحلیل تکنیکال
    """
    from technical_analysis import perform_technical_analysis
    try:
        analyzed_df = perform_technical_analysis(df, indicators)
        
        # استخراج سیگنال‌ها و نتایج تحلیل
        last_row = analyzed_df.iloc[-1].to_dict()
        
        signals = {}
        indicators_data = {}
        
        for column in analyzed_df.columns:
            if column in ['open', 'high', 'low', 'close', 'volume']:
                continue
                
            indicators_data[column] = last_row.get(column)
            
            # بررسی سیگنال‌های خرید و فروش
            if 'signal' in column.lower():
                signal_value = last_row.get(column)
                if isinstance(signal_value, str):
                    signals[column] = signal_value
        
        return {
            "dataframe": analyzed_df,
            "signals": signals,
            "indicators": indicators_data
        }
    except Exception as e:
        logger.error(f"خطا در تحلیل تکنیکال: {str(e)}")
        return {"error": str(e)}

def predict_price(df: pd.DataFrame, analysis_results: Dict[str, Any], days_ahead: int = 7) -> Dict[str, Any]:
    """
    پیش‌بینی قیمت ارز دیجیتال
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های بازار
        analysis_results (Dict[str, Any]): نتایج تحلیل تکنیکال
        days_ahead (int, optional): تعداد روزهای پیش‌بینی
        
    Returns:
        Dict[str, Any]: نتایج پیش‌بینی
    """
    # پیاده‌سازی یک پیش‌بینی ساده
    try:
        last_prices = df['close'].tail(30).values
        last_price = df['close'].iloc[-1]
        
        # محاسبه روند کلی
        overall_trend = "neutral"
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            if sma_20 > sma_50:
                overall_trend = "bullish"
            elif sma_20 < sma_50:
                overall_trend = "bearish"
        
        # محاسبه تغییرات روزانه اخیر
        daily_changes = []
        for i in range(1, len(last_prices)):
            daily_changes.append((last_prices[i] / last_prices[i-1]) - 1)
        
        avg_change = sum(daily_changes) / len(daily_changes) if daily_changes else 0
        
        # تعدیل تغییرات بر اساس روند
        if overall_trend == "bullish":
            avg_change = max(0.001, avg_change)
        elif overall_trend == "bearish":
            avg_change = min(-0.001, avg_change)
        
        # پیش‌بینی قیمت‌های آینده
        predicted_prices = []
        current_price = last_price
        
        for _ in range(days_ahead):
            change = avg_change * (1 + 0.2 * (2 * np.random.random() - 1))  # تغییر با کمی نویز
            current_price = current_price * (1 + change)
            predicted_prices.append(current_price)
        
        return {
            "days_ahead": days_ahead,
            "predicted_prices": predicted_prices,
            "overall_trend": overall_trend,
            "confidence": 0.7
        }
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی قیمت: {str(e)}")
        return {"error": str(e)}

def send_telegram_notification(source: str, timestamp: str, predictions: Dict[str, Any], token: str = None, chat_id: str = None) -> bool:
    """
    ارسال اعلان از طریق تلگرام
    
    Args:
        source (str): منبع داده
        timestamp (str): زمان داده
        predictions (Dict[str, Any]): نتایج پیش‌بینی
        token (str, optional): توکن ربات تلگرام
        chat_id (str, optional): شناسه چت
        
    Returns:
        bool: موفقیت‌آمیز بودن ارسال
    """
    from telegram_bot import send_telegram_message
    
    try:
        # ساخت متن پیام
        message = f"🔔 اعلان تحلیل ارز دیجیتال\n\n"
        message += f"📊 منبع: {source}\n"
        message += f"⏰ زمان: {timestamp}\n\n"
        
        # افزودن پیش‌بینی‌ها
        if predictions:
            for model_name, prediction in predictions.items():
                if isinstance(prediction, dict) and "error" not in prediction:
                    message += f"📈 مدل {model_name}:\n"
                    
                    if "predicted_prices" in prediction:
                        prices = prediction["predicted_prices"]
                        days_ahead = prediction.get("days_ahead", len(prices))
                        
                        message += f"    پیش‌بینی {days_ahead} روز آینده:\n"
                        
                        for i, price in enumerate(prices):
                            message += f"    روز {i+1}: {price:.2f} USDT\n"
                    
                    if "overall_trend" in prediction:
                        trend = prediction["overall_trend"]
                        trend_text = "صعودی 📈" if trend == "bullish" else "نزولی 📉" if trend == "bearish" else "خنثی ↔️"
                        message += f"    روند کلی: {trend_text}\n"
                    
                    if "confidence" in prediction:
                        confidence = prediction["confidence"] * 100
                        message += f"    اطمینان: {confidence:.1f}%\n"
                    
                    message += "\n"
        
        # ارسال پیام
        if token and chat_id:
            return send_telegram_message(chat_id, message, token)
        else:
            logger.warning("توکن یا شناسه چت تلگرام مشخص نشده است")
            return False
    except Exception as e:
        logger.error(f"خطا در ارسال اعلان تلگرام: {str(e)}")
        return False

def detect_black_swan_events(df: pd.DataFrame, window_size: int = 20, threshold: float = 3.0) -> Dict[str, Any]:
    """
    تشخیص رویدادهای مهم و غیرمنتظره (Black Swan Events)
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های بازار
        window_size (int, optional): اندازه پنجره برای محاسبه میانگین و انحراف معیار
        threshold (float, optional): حد آستانه انحراف معیار برای تشخیص رویداد مهم
        
    Returns:
        Dict[str, Any]: نتایج تشخیص
    """
    try:
        # محاسبه تغییرات قیمت
        price_changes = df['close'].pct_change().dropna()
        
        # محاسبه میانگین و انحراف معیار تغییرات قیمت در پنجره مشخص شده
        rolling_mean = price_changes.rolling(window=window_size).mean()
        rolling_std = price_changes.rolling(window=window_size).std()
        
        # تشخیص رویدادهای خارج از محدوده نرمال
        z_scores = (price_changes - rolling_mean) / rolling_std
        extreme_events = z_scores[z_scores.abs() > threshold]
        
        # بررسی آخرین نقطه داده
        latest_z_score = z_scores.iloc[-1] if not z_scores.empty else 0
        latest_price_change = price_changes.iloc[-1] if not price_changes.empty else 0
        
        is_black_swan = abs(latest_z_score) > threshold
        severity = min(1.0, abs(latest_z_score) / (threshold * 2))
        
        # تعیین نوع رویداد (مثبت یا منفی)
        event_type = None
        if is_black_swan:
            event_type = "positive" if latest_price_change > 0 else "negative"
        
        return {
            "score": severity,
            "is_black_swan": is_black_swan,
            "z_score": latest_z_score,
            "price_change": latest_price_change,
            "event_type": event_type,
            "extreme_events_count": len(extreme_events),
            "threshold": threshold
        }
    except Exception as e:
        logger.error(f"خطا در تشخیص رویدادهای مهم: {str(e)}")
        return {"score": 0.0, "error": str(e)}

# ------------------- ایجاد و پیکربندی میکروسرویس‌ها -------------------

def create_crypto_microservices(symbols: List[str] = None, timeframes: List[str] = None) -> MicroserviceOrchestrator:
    """
    ایجاد و پیکربندی میکروسرویس‌های تحلیل ارزهای دیجیتال
    
    Args:
        symbols (List[str], optional): لیست نمادهای ارز
        timeframes (List[str], optional): لیست تایم‌فریم‌ها
        
    Returns:
        MicroserviceOrchestrator: هماهنگ‌کننده میکروسرویس‌ها
    """
    # مقادیر پیش‌فرض
    symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    timeframes = timeframes or ["1h", "4h", "1d"]
    
    # ایجاد هماهنگ‌کننده
    orchestrator = MicroserviceOrchestrator()
    
    # ایجاد میکروسرویس جمع‌آوری داده‌ها
    data_collection = DataCollectionService(name="data_collection")
    
    # افزودن منابع داده برای هر ترکیب نماد و تایم‌فریم
    for symbol in symbols:
        for timeframe in timeframes:
            source_name = f"{symbol}_{timeframe}"
            data_collection.add_data_source(
                name=source_name,
                source_function=get_market_data,
                params={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "lookback_days": 30,
                    "exchange": "binance"
                }
            )
    
    # تنظیم بازه جمع‌آوری داده‌ها (هر 60 ثانیه)
    data_collection.set_collection_interval(60)
    
    # ایجاد میکروسرویس تحلیل
    analysis = AnalysisService(name="analysis")
    
    # افزودن تحلیلگرها
    analysis.add_analyzer(
        name="technical",
        analyzer_function=technical_analysis,
        params={
            "indicators": None  # استفاده از تمام اندیکاتورهای پیش‌فرض
        }
    )
    
    # ایجاد میکروسرویس پیش‌بینی
    prediction = PredictionService(name="prediction")
    
    # افزودن مدل‌های پیش‌بینی
    prediction.add_model(
        name="price_prediction",
        model_function=predict_price,
        params={
            "days_ahead": 7
        }
    )
    
    # ایجاد میکروسرویس تشخیص رویدادهای مهم
    black_swan = BlackSwanDetectionService(name="black_swan")
    
    # افزودن الگوریتم‌های تشخیص
    black_swan.add_algorithm(
        name="price_volatility",
        algorithm_function=detect_black_swan_events,
        params={
            "window_size": 20,
            "threshold": 3.0
        }
    )
    
    # ایجاد میکروسرویس اعلان
    notification = NotificationService(name="notification")
    
    # افزودن کانال‌های اعلان
    notification.add_channel(
        name="telegram",
        channel_function=send_telegram_notification,
        params={
            "token": None,  # باید توسط کاربر تنظیم شود
            "chat_id": None  # باید توسط کاربر تنظیم شود
        }
    )
    
    # افزودن میکروسرویس‌ها به هماهنگ‌کننده
    orchestrator.add_service(data_collection)
    orchestrator.add_service(analysis)
    orchestrator.add_service(prediction)
    orchestrator.add_service(black_swan)
    orchestrator.add_service(notification)
    
    # ایجاد ارتباط بین میکروسرویس‌ها
    orchestrator.connect_services("data_collection", "analysis")
    orchestrator.connect_services("analysis", "prediction")
    orchestrator.connect_services("prediction", "notification")
    orchestrator.connect_services("analysis", "black_swan")
    orchestrator.connect_services("black_swan", "notification")
    
    return orchestrator

# ------------------- تنظیم و راه‌اندازی میکروسرویس‌ها -------------------

def setup_microservices(symbols: List[str] = None, timeframes: List[str] = None, telegram_token: str = None, telegram_chat_id: str = None) -> MicroserviceOrchestrator:
    """
    تنظیم و راه‌اندازی میکروسرویس‌ها
    
    Args:
        symbols (List[str], optional): لیست نمادهای ارز
        timeframes (List[str], optional): لیست تایم‌فریم‌ها
        telegram_token (str, optional): توکن ربات تلگرام
        telegram_chat_id (str, optional): شناسه چت تلگرام
        
    Returns:
        MicroserviceOrchestrator: هماهنگ‌کننده میکروسرویس‌ها
    """
    # ایجاد میکروسرویس‌ها
    orchestrator = create_crypto_microservices(symbols, timeframes)
    
    # تنظیم پارامترهای تلگرام
    if telegram_token and telegram_chat_id:
        notification_service = orchestrator.get_service("notification")
        if notification_service:
            for channel_name, channel_config in notification_service.notification_channels.items():
                if channel_name == "telegram":
                    channel_config["params"]["token"] = telegram_token
                    channel_config["params"]["chat_id"] = telegram_chat_id
                    logger.info("پارامترهای تلگرام تنظیم شدند")
    
    # راه‌اندازی میکروسرویس‌ها
    orchestrator.start_all()
    logger.info("تمام میکروسرویس‌ها راه‌اندازی شدند")
    
    return orchestrator

# ------------------- استفاده از میکروسرویس‌ها در برنامه اصلی -------------------

def get_microservices_orchestrator(symbols: List[str] = None, timeframes: List[str] = None, telegram_token: str = None, telegram_chat_id: str = None) -> MicroserviceOrchestrator:
    """
    دریافت هماهنگ‌کننده میکروسرویس‌ها (Singleton)
    
    Args:
        symbols (List[str], optional): لیست نمادهای ارز
        timeframes (List[str], optional): لیست تایم‌فریم‌ها
        telegram_token (str, optional): توکن ربات تلگرام
        telegram_chat_id (str, optional): شناسه چت تلگرام
        
    Returns:
        MicroserviceOrchestrator: هماهنگ‌کننده میکروسرویس‌ها
    """
    # بررسی وجود نمونه قبلی
    if not hasattr(get_microservices_orchestrator, "instance") or get_microservices_orchestrator.instance is None:
        # ایجاد نمونه جدید
        get_microservices_orchestrator.instance = setup_microservices(symbols, timeframes, telegram_token, telegram_chat_id)
    
    return get_microservices_orchestrator.instance

def stop_microservices():
    """توقف میکروسرویس‌ها"""
    if hasattr(get_microservices_orchestrator, "instance") and get_microservices_orchestrator.instance is not None:
        get_microservices_orchestrator.instance.stop_all()
        logger.info("تمام میکروسرویس‌ها متوقف شدند")
        get_microservices_orchestrator.instance = None