"""
ماژول مدیریت میکروسرویس‌ها برای سیستم تحلیل ارزهای دیجیتال

این ماژول وظیفه راه‌اندازی، مدیریت و هماهنگی میکروسرویس‌های مختلف سیستم را بر عهده دارد.
"""

import os
import logging
import threading
import time
from typing import Dict, List, Any, Optional

# ماژول‌های میکروسرویس
from microservices_architecture import (
    get_microservices_orchestrator,
    stop_microservices,
    DataCollectionService,
    AnalysisService,
    PredictionService,
    BlackSwanDetectionService,
    NotificationService
)

# تنظیم لاگ‌ها
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='microservices.log',
    filemode='a'
)

logger = logging.getLogger("microservices_manager")

# ------------------- مدیریت میکروسرویس‌ها -------------------

class MicroservicesManager:
    """کلاس مدیریت میکروسرویس‌های سیستم"""
    
    def __init__(self):
        """مقداردهی اولیه"""
        self.orchestrator = None
        self.status = {}
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.timeframes = ["1h", "4h", "1d"]
        self.telegram_token = None
        self.telegram_chat_id = None
        self.is_running = False
        self.monitor_thread = None
        
    def initialize(self, symbols: List[str] = None, timeframes: List[str] = None) -> None:
        """
        مقداردهی میکروسرویس‌ها
        
        Args:
            symbols (List[str], optional): لیست نمادهای ارز
            timeframes (List[str], optional): لیست تایم‌فریم‌ها
        """
        # تنظیم پارامترها
        self.symbols = symbols or self.symbols
        self.timeframes = timeframes or self.timeframes
        
        try:
            # بارگیری تنظیمات تلگرام
            self._load_telegram_settings()
            
            # ایجاد هماهنگ‌کننده میکروسرویس‌ها
            self.orchestrator = get_microservices_orchestrator(
                symbols=self.symbols,
                timeframes=self.timeframes,
                telegram_token=self.telegram_token,
                telegram_chat_id=self.telegram_chat_id
            )
            
            # راه‌اندازی میکروسرویس‌ها
            self.is_running = True
            
            # شروع مانیتورینگ
            self._start_monitoring()
            
            logger.info("مدیریت میکروسرویس‌ها با موفقیت مقداردهی شد")
            return True
        except Exception as e:
            logger.error(f"خطا در مقداردهی میکروسرویس‌ها: {str(e)}")
            return False
        
    def _load_telegram_settings(self) -> None:
        """بارگیری تنظیمات تلگرام"""
        try:
            # تلاش برای خواندن توکن تلگرام از فایل
            token_path = "./telegram_token.txt"
            chat_id_path = "./telegram_chat_id.txt"
            
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    self.telegram_token = f.read().strip()
                    
            if os.path.exists(chat_id_path):
                with open(chat_id_path, 'r') as f:
                    self.telegram_chat_id = f.read().strip()
                    
            logger.info("تنظیمات تلگرام بارگیری شدند")
        except Exception as e:
            logger.error(f"خطا در بارگیری تنظیمات تلگرام: {str(e)}")
            
    def set_telegram_config(self, token: str, chat_id: str) -> bool:
        """
        تنظیم پیکربندی تلگرام
        
        Args:
            token (str): توکن ربات تلگرام
            chat_id (str): شناسه چت
            
        Returns:
            bool: موفقیت عملیات
        """
        try:
            # ذخیره تنظیمات
            self.telegram_token = token
            self.telegram_chat_id = chat_id
            
            # ذخیره در فایل
            with open("./telegram_token.txt", 'w') as f:
                f.write(token)
                
            with open("./telegram_chat_id.txt", 'w') as f:
                f.write(chat_id)
            
            # اعمال تنظیمات در میکروسرویس‌ها
            if self.orchestrator:
                notification_service = self.orchestrator.get_service("notification")
                if notification_service:
                    for channel_name, channel_config in notification_service.notification_channels.items():
                        if channel_name == "telegram":
                            channel_config["params"]["token"] = token
                            channel_config["params"]["chat_id"] = chat_id
                            
            logger.info("تنظیمات تلگرام با موفقیت ذخیره شدند")
            return True
        except Exception as e:
            logger.error(f"خطا در تنظیم پیکربندی تلگرام: {str(e)}")
            return False
            
    def start(self) -> bool:
        """
        شروع میکروسرویس‌ها
        
        Returns:
            bool: موفقیت عملیات
        """
        try:
            if not self.orchestrator:
                # مقداردهی در صورت نیاز
                self.initialize()
                
            if self.orchestrator and not self.is_running:
                # راه‌اندازی میکروسرویس‌ها
                self.orchestrator.start_all()
                self.is_running = True
                
                # شروع مانیتورینگ
                self._start_monitoring()
                
                logger.info("میکروسرویس‌ها با موفقیت شروع شدند")
                return True
                
            elif self.is_running:
                logger.info("میکروسرویس‌ها قبلاً در حال اجرا هستند")
                return True
                
            return False
        except Exception as e:
            logger.error(f"خطا در شروع میکروسرویس‌ها: {str(e)}")
            return False
            
    def stop(self) -> bool:
        """
        توقف میکروسرویس‌ها
        
        Returns:
            bool: موفقیت عملیات
        """
        try:
            if self.orchestrator and self.is_running:
                # توقف میکروسرویس‌ها
                self.orchestrator.stop_all()
                stop_microservices()
                self.is_running = False
                
                # توقف مانیتورینگ
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=1.0)
                    
                logger.info("میکروسرویس‌ها با موفقیت متوقف شدند")
                return True
                
            elif not self.is_running:
                logger.info("میکروسرویس‌ها قبلاً متوقف شده‌اند")
                return True
                
            return False
        except Exception as e:
            logger.error(f"خطا در توقف میکروسرویس‌ها: {str(e)}")
            return False
            
    def restart(self) -> bool:
        """
        راه‌اندازی مجدد میکروسرویس‌ها
        
        Returns:
            bool: موفقیت عملیات
        """
        try:
            self.stop()
            time.sleep(1)  # انتظار برای اطمینان از توقف کامل
            return self.start()
        except Exception as e:
            logger.error(f"خطا در راه‌اندازی مجدد میکروسرویس‌ها: {str(e)}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت میکروسرویس‌ها
        
        Returns:
            Dict[str, Any]: وضعیت میکروسرویس‌ها
        """
        if not self.orchestrator:
            return {"error": "میکروسرویس‌ها مقداردهی نشده‌اند"}
            
        try:
            # وضعیت اجرا
            running_status = {name: service.running for name, service in self.orchestrator.services.items()}
            
            # تنظیمات
            settings = {
                "symbols": self.symbols,
                "timeframes": self.timeframes,
                "telegram_configured": bool(self.telegram_token and self.telegram_chat_id),
            }
            
            # وضعیت صف‌ها
            queues_status = {}
            for name, queue in self.orchestrator.queues.items():
                queues_status[name] = {
                    "size": queue.qsize(),
                    "empty": queue.empty()
                }
                
            return {
                "running": self.is_running,
                "services": running_status,
                "settings": settings,
                "queues": queues_status
            }
        except Exception as e:
            logger.error(f"خطا در دریافت وضعیت میکروسرویس‌ها: {str(e)}")
            return {"error": str(e)}
            
    def _start_monitoring(self) -> None:
        """شروع مانیتورینگ میکروسرویس‌ها"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
            
        self.monitor_thread = threading.Thread(target=self._monitor_services)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_services(self) -> None:
        """مانیتورینگ مستمر میکروسرویس‌ها"""
        while self.is_running:
            try:
                # بررسی وضعیت میکروسرویس‌ها
                if self.orchestrator:
                    for name, service in self.orchestrator.services.items():
                        if not service.running and self.is_running:
                            logger.warning(f"میکروسرویس {name} متوقف شده است، تلاش برای راه‌اندازی مجدد...")
                            service.start()
                
                # ارسال پینگ به سیستم برای جلوگیری از خاموش شدن
                self._ping_system()
                
                # انتظار قبل از بررسی مجدد
                time.sleep(10)
            except Exception as e:
                logger.error(f"خطا در مانیتورینگ میکروسرویس‌ها: {str(e)}")
                time.sleep(30)  # انتظار بیشتر در صورت بروز خطا
                
    def _ping_system(self) -> None:
        """ارسال پینگ به سیستم برای جلوگیری از خاموش شدن"""
        try:
            # لاگ کردن یک پیام برای نگه داشتن سیستم در حالت فعال
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.debug(f"پینگ سیستم در {current_time} - سیستم فعال است")
            
            # اگر تلگرام تنظیم شده، یک پیام اعلان وضعیت هر ساعت ارسال می‌کنیم
            if self.telegram_token and self.telegram_chat_id:
                current_hour = time.localtime().tm_hour
                # فقط هر ساعت یک بار پیام ارسال می‌کنیم
                if current_hour != getattr(self, '_last_ping_hour', None):
                    self._last_ping_hour = current_hour
                    
                    # هر 8 ساعت یک گزارش کامل ارسال می‌کنیم
                    if current_hour % 8 == 0:
                        # لاگ کردن ارسال گزارش
                        logger.info(f"ارسال گزارش وضعیت سیستم به تلگرام در ساعت {current_hour}")
                        
                        # ارسال گزارش وضعیت سیستم
                        status = self._get_system_status_report()
                        from telegram_bot import send_telegram_message
                        send_telegram_message(self.telegram_chat_id, status)
        except Exception as e:
            logger.error(f"خطا در پینگ سیستم: {str(e)}")
    
    def _get_system_status_report(self) -> str:
        """
        تهیه گزارش وضعیت سیستم
        
        Returns:
            str: گزارش وضعیت سیستم به فرمت HTML
        """
        try:
            # دریافت وضعیت میکروسرویس‌ها
            status = self.get_status()
            
            # تولید پیام
            message = f"""
<b>🔄 گزارش وضعیت سیستم</b>

<b>زمان گزارش:</b> {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
<b>وضعیت کلی:</b> {'فعال ✅' if self.is_running else 'غیرفعال ❌'}

<b>میکروسرویس‌های فعال:</b>
"""
            # افزودن وضعیت میکروسرویس‌ها
            if status.get('services'):
                for name, running in status['services'].items():
                    message += f"• {name}: {'فعال ✅' if running else 'غیرفعال ❌'}\n"
            
            message += "\n<b>ارزهای تحت نظر:</b>\n"
            for symbol in self.symbols:
                message += f"• {symbol}\n"
                
            message += "\n<b>تایم‌فریم‌های تحت نظر:</b>\n"
            for timeframe in self.timeframes:
                message += f"• {timeframe}\n"
                
            message += "\n<i>سیستم در حال کار است و به بررسی بازار ادامه می‌دهد.</i>"
            
            return message
        except Exception as e:
            logger.error(f"خطا در تهیه گزارش وضعیت سیستم: {str(e)}")
            return f"<b>خطا در تهیه گزارش وضعیت سیستم:</b> {str(e)}"
                
    def get_service(self, name: str) -> Optional[Any]:
        """
        دریافت یک میکروسرویس با نام مشخص
        
        Args:
            name (str): نام میکروسرویس
            
        Returns:
            Optional[Any]: میکروسرویس یا None در صورت عدم وجود
        """
        if self.orchestrator:
            return self.orchestrator.get_service(name)
        return None
        
    def manual_send_to_service(self, service_name: str, data: Any) -> bool:
        """
        ارسال دستی داده به یک میکروسرویس
        
        Args:
            service_name (str): نام میکروسرویس
            data (Any): داده‌ای که باید ارسال شود
            
        Returns:
            bool: موفقیت عملیات
        """
        if not self.orchestrator:
            return False
            
        try:
            service = self.orchestrator.get_service(service_name)
            if service:
                service.input_queue.put(data)
                logger.info(f"داده با موفقیت به میکروسرویس {service_name} ارسال شد")
                return True
                
            logger.error(f"میکروسرویس {service_name} یافت نشد")
            return False
        except Exception as e:
            logger.error(f"خطا در ارسال داده به میکروسرویس {service_name}: {str(e)}")
            return False
            
    def manual_get_from_service(self, service_name: str, timeout: float = 0.1) -> Optional[Any]:
        """
        دریافت دستی داده از یک میکروسرویس
        
        Args:
            service_name (str): نام میکروسرویس
            timeout (float): زمان انتظار
            
        Returns:
            Optional[Any]: داده دریافتی یا None در صورت خالی بودن صف
        """
        if not self.orchestrator:
            return None
            
        try:
            service = self.orchestrator.get_service(service_name)
            if service and service.output_queue:
                try:
                    data = service.output_queue.get(block=True, timeout=timeout)
                    return data
                except:
                    return None
                    
            logger.error(f"میکروسرویس {service_name} یافت نشد یا صف خروجی ندارد")
            return None
        except Exception as e:
            logger.error(f"خطا در دریافت داده از میکروسرویس {service_name}: {str(e)}")
            return None
    
    def send_test_notification(self, message: str = None) -> bool:
        """
        ارسال اعلان تست
        
        Args:
            message (str, optional): پیام اعلان
            
        Returns:
            bool: موفقیت عملیات
        """
        if not message:
            message = f"این یک پیام تست از سیستم تحلیل ارزهای دیجیتال است.\nزمان: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
        if not self.orchestrator:
            return False
            
        try:
            notification_service = self.orchestrator.get_service("notification")
            if notification_service:
                test_data = {
                    "source": "test",
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "predictions": {
                        "test": {
                            "message": message
                        }
                    }
                }
                
                notification_service.process(test_data)
                logger.info("اعلان تست با موفقیت ارسال شد")
                return True
                
            logger.error("میکروسرویس اعلان یافت نشد")
            return False
        except Exception as e:
            logger.error(f"خطا در ارسال اعلان تست: {str(e)}")
            return False
            
    def add_symbol(self, symbol: str) -> bool:
        """
        افزودن نماد جدید
        
        Args:
            symbol (str): نماد ارز
            
        Returns:
            bool: موفقیت عملیات
        """
        if symbol in self.symbols:
            logger.info(f"نماد {symbol} قبلاً در سیستم موجود است")
            return False
            
        try:
            self.symbols.append(symbol)
            
            # افزودن منابع داده جدید برای نماد
            if self.orchestrator:
                data_collection = self.orchestrator.get_service("data_collection")
                if data_collection:
                    for timeframe in self.timeframes:
                        source_name = f"{symbol}_{timeframe}"
                        from microservices_architecture import get_market_data
                        
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
                        
            logger.info(f"نماد {symbol} با موفقیت اضافه شد")
            return True
        except Exception as e:
            logger.error(f"خطا در افزودن نماد {symbol}: {str(e)}")
            return False

# ------------------- نمونه سینگلتون -------------------

def get_microservices_manager() -> MicroservicesManager:
    """
    دریافت نمونه مدیریت میکروسرویس‌ها (Singleton)
    
    Returns:
        MicroservicesManager: مدیریت میکروسرویس‌ها
    """
    if not hasattr(get_microservices_manager, "instance") or get_microservices_manager.instance is None:
        get_microservices_manager.instance = MicroservicesManager()
    
    return get_microservices_manager.instance

# ------------------- خارج کردن از حافظه -------------------

def shutdown_microservices_manager() -> None:
    """خاموش کردن میکروسرویس‌ها و خارج کردن از حافظه"""
    if hasattr(get_microservices_manager, "instance") and get_microservices_manager.instance is not None:
        get_microservices_manager.instance.stop()
        get_microservices_manager.instance = None
        logger.info("مدیریت میکروسرویس‌ها با موفقیت خارج شد")

# ------------------- توابع کمکی -------------------

def initialize_microservices(symbols: List[str] = None, timeframes: List[str] = None) -> bool:
    """
    مقداردهی و شروع میکروسرویس‌ها
    
    Args:
        symbols (List[str], optional): لیست نمادهای ارز
        timeframes (List[str], optional): لیست تایم‌فریم‌ها
        
    Returns:
        bool: موفقیت عملیات
    """
    manager = get_microservices_manager()
    return manager.initialize(symbols, timeframes) and manager.start()

def get_microservices_status() -> Dict[str, Any]:
    """
    دریافت وضعیت میکروسرویس‌ها
    
    Returns:
        Dict[str, Any]: وضعیت میکروسرویس‌ها
    """
    manager = get_microservices_manager()
    return manager.get_status()

def restart_microservices() -> bool:
    """
    راه‌اندازی مجدد میکروسرویس‌ها
    
    Returns:
        bool: موفقیت عملیات
    """
    manager = get_microservices_manager()
    return manager.restart()

def set_telegram_configuration(token: str, chat_id: str) -> bool:
    """
    تنظیم پیکربندی تلگرام
    
    Args:
        token (str): توکن ربات تلگرام
        chat_id (str): شناسه چت
        
    Returns:
        bool: موفقیت عملیات
    """
    manager = get_microservices_manager()
    return manager.set_telegram_config(token, chat_id)

def send_test_telegram_message(message: str = None) -> bool:
    """
    ارسال پیام تست به تلگرام
    
    Args:
        message (str, optional): پیام تست
        
    Returns:
        bool: موفقیت عملیات
    """
    manager = get_microservices_manager()
    return manager.send_test_notification(message)

def is_microservices_running() -> bool:
    """
    بررسی در حال اجرا بودن میکروسرویس‌ها
    
    Returns:
        bool: وضعیت اجرا
    """
    manager = get_microservices_manager()
    return manager.is_running