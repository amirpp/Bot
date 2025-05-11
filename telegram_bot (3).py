"""
ماژول ربات تلگرام برای دسترسی به تحلیل‌های ارز دیجیتال

این ماژول شامل کلاس‌ها و توابع مورد نیاز برای راه‌اندازی ربات تلگرام است.
"""

import os
import json
import time
import logging
import threading
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# تنظیم لاگر
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("telegram_bot.log")
    ]
)
logger = logging.getLogger("TelegramBot")

# واردسازی ماژول‌های پروژه
from neura_ai import NeuraAI
from api_services import get_top_cryptocurrencies, get_ohlcv_data_multi_source
from technical_analysis import perform_technical_analysis
from chart_patterns import analyze_chart_patterns
from high_potential_crypto import find_high_potential_cryptocurrencies

class TelegramBot:
    """کلاس اصلی ربات تلگرام"""
    
    def __init__(self, token: str):
        """
        مقداردهی اولیه ربات تلگرام
        
        Args:
            token (str): توکن ربات تلگرام
        """
        self.token = token
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id = 0
        self.active = True
        self.authorized_users = self._load_authorized_users()
        self.neura = NeuraAI("نیورا")
        self.command_handlers = self._initialize_command_handlers()
        
        # راه‌اندازی ترد پردازش پیام‌ها
        self.message_thread = threading.Thread(
            target=self._process_messages_loop,
            daemon=True
        )
        self.message_thread.start()
        
        logger.info("ربات تلگرام راه‌اندازی شد.")
    
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
            "/top": self._handle_top_command,
            "/potential": self._handle_potential_command,
            "/patterns": self._handle_patterns_command,
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
        به ربات تحلیل ارزهای دیجیتال خوش آمدید!
        
        این ربات با استفاده از هوش مصنوعی نیورا و تحلیل تکنیکال پیشرفته، به شما در تحلیل بازار و پیش‌بینی قیمت کمک می‌کند.
        
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
        راهنمای دستورات ربات تحلیل ارزهای دیجیتال:
        
        /start - شروع کار با ربات
        /help - نمایش این راهنما
        /status - نمایش وضعیت سیستم
        /analyze (symbol) (timeframe) - تحلیل بازار ارز مشخص شده
        /predict (symbol) (timeframe) (days) - پیش‌بینی قیمت
        /top (limit) - نمایش ارزهای برتر
        /potential (method) (limit) - یافتن ارزهای با پتانسیل بالا
        /patterns (symbol) (timeframe) - شناسایی الگوهای نموداری
        
        مثال‌ها:
        /analyze BTC/USDT 1d
        /predict ETH/USDT 4h 7
        /top 5
        /potential ترکیبی 5
        /patterns BTC/USDT 1d
        
        همچنین می‌توانید سؤالات خود را به صورت مستقیم بپرسید و از هوش مصنوعی نیورا کمک بگیرید.
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
        وضعیت سیستم:
        
        نام: {self.neura.name}
        نسخه: {self.neura.version}
        وضعیت سلامت: {status['health']}
        تعداد مکالمات ذخیره شده: {len(self.neura.memory['conversations'])}
        تعداد تحلیل‌های ذخیره شده: {len(self.neura.memory['analysis_history'])}
        ترد‌های فعال: {status['active_threads']}
        
        زمان بررسی: {status['timestamp']}
        
        کاربران مجاز: {len(self.authorized_users)}
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
    
    def _handle_top_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور ارزهای برتر
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        # استخراج پارامترها
        limit = int(params[0]) if params and params[0].isdigit() else 5
        limit = min(limit, 20)  # محدود کردن تعداد
        
        # ارسال پیام در حال پردازش
        self._send_message(chat_id, f"در حال دریافت لیست {limit} ارز برتر...")
        
        try:
            # دریافت لیست ارزهای برتر
            top_coins = get_top_cryptocurrencies(limit=limit)
            
            # آماده‌سازی پیام پاسخ
            if top_coins:
                message = f"📊 {limit} ارز برتر بر اساس ارزش بازار:\n\n"
                
                for i, coin in enumerate(top_coins[:limit], 1):
                    symbol = coin.get("symbol", "نامشخص")
                    name = coin.get("name", "نامشخص")
                    price = coin.get("price", 0)
                    price_change = coin.get("price_change_24h", 0)
                    
                    # نماد تغییر قیمت
                    change_symbol = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "⚪️"
                    
                    # افزودن به پیام
                    message += f"{i}. {symbol} ({name})\n"
                    message += f"   قیمت: ${price:.2f}\n"
                    message += f"   تغییر 24h: {change_symbol} {price_change:.2f}%\n\n"
            else:
                message = "متأسفانه در دریافت اطلاعات ارزهای برتر خطایی رخ داده است."
            
            # ارسال پیام
            self._send_message(chat_id, message)
        except Exception as e:
            self._send_message(chat_id, f"خطا در دریافت ارزهای برتر: {str(e)}")
    
    def _handle_potential_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور ارزهای با پتانسیل بالا
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        # استخراج پارامترها
        method = params[0] if params and params[0] in ["ترکیبی", "حجم بالا", "رشد سریع", "الگوی نموداری", "هوش مصنوعی"] else "ترکیبی"
        limit = int(params[1]) if len(params) > 1 and params[1].isdigit() else 5
        limit = min(limit, 10)  # محدود کردن تعداد
        
        # ارسال پیام در حال پردازش
        self._send_message(chat_id, f"در حال یافتن ارزهای با پتانسیل بالا با روش {method}...\nاین فرآیند ممکن است چند دقیقه طول بکشد.")
        
        try:
            # دریافت لیست ارزهای برتر
            top_coins = get_top_cryptocurrencies(limit=50)
            
            if not top_coins:
                self._send_message(chat_id, "متأسفانه در دریافت اطلاعات ارزهای برتر خطایی رخ داده است.")
                return
            
            # یافتن ارزهای با پتانسیل بالا
            potential_coins = find_high_potential_cryptocurrencies(top_coins, method=method, limit=limit)
            
            # آماده‌سازی پیام پاسخ
            if potential_coins:
                message = f"🔍 ارزهای با پتانسیل بالا (روش: {method}):\n\n"
                
                for i, coin in enumerate(potential_coins, 1):
                    symbol = coin.get("symbol", "نامشخص")
                    name = coin.get("name", "نامشخص")
                    score = coin.get("potential_score", 0)
                    direction = coin.get("potential_direction", "نامشخص")
                    current_price = coin.get("current_price", 0)
                    
                    # نماد جهت پتانسیل
                    dir_symbol = "📈" if direction == "صعودی" else "📉" if direction == "نزولی" else "⚖️"
                    
                    # افزودن به پیام
                    message += f"{i}. {symbol} ({name})\n"
                    message += f"   امتیاز پتانسیل: {score:.1f}/100\n"
                    message += f"   جهت: {dir_symbol} {direction}\n"
                    message += f"   قیمت فعلی: ${current_price:.6f}\n"
                    
                    # افزودن جزئیات پتانسیل
                    if "potential_details" in coin and coin["potential_details"]:
                        message += f"   دلایل: {', '.join(coin['potential_details'])}\n"
                    
                    message += "\n"
            else:
                message = "هیچ ارزی با پتانسیل بالا یافت نشد."
            
            # ارسال پیام
            self._send_message(chat_id, message)
        except Exception as e:
            self._send_message(chat_id, f"خطا در یافتن ارزهای با پتانسیل بالا: {str(e)}")
    
    def _handle_patterns_command(self, chat_id: int, user_id: int, params: List[str]):
        """
        پردازش دستور شناسایی الگوهای نموداری
        
        Args:
            chat_id (int): شناسه چت
            user_id (int): شناسه کاربر
            params (List[str]): پارامترها
        """
        # بررسی پارامترها
        if len(params) < 1:
            self._send_message(chat_id, "لطفاً نماد ارز را مشخص کنید. مثال: /patterns BTC/USDT 1d")
            return
        
        # استخراج پارامترها
        symbol = params[0]
        timeframe = params[1] if len(params) > 1 else "1d"
        
        # ارسال پیام در حال پردازش
        self._send_message(chat_id, f"در حال شناسایی الگوهای نموداری برای {symbol} در تایم‌فریم {timeframe}...")
        
        try:
            # دریافت داده‌های قیمت
            df = get_ohlcv_data_multi_source(symbol, timeframe=timeframe, lookback_days=30)
            
            if df is None or df.empty:
                self._send_message(chat_id, f"داده‌های قیمت برای {symbol} در دسترس نیست.")
                return
            
            # انجام تحلیل تکنیکال
            indicators = [
                'RSI', 'MACD', 'Bollinger Bands', 'Stochastic',
                'ADX', 'ATR', 'EMA', 'SMA'
            ]
            df = perform_technical_analysis(df, indicators)
            
            # شناسایی الگوها
            patterns = analyze_chart_patterns(df)
            
            # آماده‌سازی پیام پاسخ
            if patterns:
                message = f"🔍 الگوهای نموداری شناسایی شده برای {symbol} در تایم‌فریم {timeframe}:\n\n"
                
                # دسته‌بندی الگوها بر اساس جهت
                bullish_patterns = [p for p in patterns if p.get("direction") == "bullish"]
                bearish_patterns = [p for p in patterns if p.get("direction") == "bearish"]
                neutral_patterns = [p for p in patterns if p.get("direction") == "neutral"]
                
                # افزودن الگوهای صعودی
                if bullish_patterns:
                    message += "📈 الگوهای صعودی:\n"
                    for p in bullish_patterns:
                        message += f"   - {p.get('type')}"
                        if "strength" in p:
                            message += f" (قدرت: {p.get('strength')}%)"
                        message += "\n"
                    message += "\n"
                
                # افزودن الگوهای نزولی
                if bearish_patterns:
                    message += "📉 الگوهای نزولی:\n"
                    for p in bearish_patterns:
                        message += f"   - {p.get('type')}"
                        if "strength" in p:
                            message += f" (قدرت: {p.get('strength')}%)"
                        message += "\n"
                    message += "\n"
                
                # افزودن الگوهای خنثی
                if neutral_patterns:
                    message += "⚖️ الگوهای خنثی:\n"
                    for p in neutral_patterns:
                        message += f"   - {p.get('type')}"
                        if "strength" in p:
                            message += f" (قدرت: {p.get('strength')}%)"
                        message += "\n"
                    message += "\n"
                
                # توصیه‌های کلی
                if len(bullish_patterns) > len(bearish_patterns):
                    message += "🔄 نتیجه‌گیری: غلبه الگوهای صعودی - احتمال روند صعودی"
                elif len(bearish_patterns) > len(bullish_patterns):
                    message += "🔄 نتیجه‌گیری: غلبه الگوهای نزولی - احتمال روند نزولی"
                else:
                    message += "🔄 نتیجه‌گیری: توازن الگوهای صعودی و نزولی - شرایط خنثی"
            else:
                message = f"هیچ الگوی نموداری برای {symbol} در تایم‌فریم {timeframe} شناسایی نشد."
            
            # ارسال پیام
            self._send_message(chat_id, message)
        except Exception as e:
            self._send_message(chat_id, f"خطا در شناسایی الگوهای نموداری: {str(e)}")
    
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
        """خاموش کردن ربات"""
        self.active = False
        self.neura.shutdown()
        logger.info("ربات تلگرام با موفقیت خاموش شد.")


def start_telegram_bot(token: str = None):
    """
    راه‌اندازی ربات تلگرام
    
    Args:
        token (str, optional): توکن ربات تلگرام
    
    Returns:
        TelegramBot: نمونه ربات تلگرام یا None در صورت خطا
    """
    try:
        # تلاش برای خواندن توکن از فایل در صورت عدم ارائه
        if not token:
            try:
                if os.path.exists("telegram_token.txt"):
                    with open("telegram_token.txt", "r") as f:
                        token = f.read().strip()
            except:
                pass
        
        # بررسی توکن
        if not token:
            logger.error("توکن ربات تلگرام در دسترس نیست.")
            return None
        
        # ایجاد ربات
        bot = TelegramBot(token)
        logger.info("ربات تلگرام با موفقیت راه‌اندازی شد.")
        return bot
    except Exception as e:
        logger.error(f"خطا در راه‌اندازی ربات تلگرام: {str(e)}")
        return None


if __name__ == "__main__":
    # دریافت توکن از محیط یا فایل
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    # راه‌اندازی ربات
    bot = start_telegram_bot(token)
    
    if bot:
        try:
            # ادامه اجرا تا وقتی که با Ctrl+C متوقف شود
            logger.info("ربات تلگرام در حال اجراست. برای توقف Ctrl+C را فشار دهید.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("دریافت سیگنال توقف...")
            bot.shutdown()
            logger.info("ربات تلگرام متوقف شد.")
    else:
        logger.error("راه‌اندازی ربات تلگرام ناموفق بود.")