"""
ماژول ربات تلگرام برای ارسال سیگنال‌ها و تحلیل‌های ارز دیجیتال

این ماژول امکان ارسال سیگنال‌ها، نمودارها و تحلیل‌های بازار ارزهای دیجیتال را
به کانال یا گروه تلگرام فراهم می‌کند.

قابلیت‌های اصلی:
- ارسال سیگنال‌های خرید و فروش لحظه‌ای
- ارسال تحلیل‌های کامل با نمودار
- پاسخگویی به دستورات کاربران
- دریافت قیمت ارزهای مختلف
- ارسال هشدارهای قیمتی و اخبار مهم
- دریافت تحلیل‌های هوشمند از ابزارهای هوش مصنوعی
- ارسال خودکار سیگنال‌ها در زمان‌های مشخص
"""

import os
import time
import logging
import threading
import requests
import io
import traceback
import json
from typing import Dict, List, Any, Optional, Union, Tuple, cast
import base64
from datetime import datetime, timedelta

# تنظیم لاگر
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# متغیرهای گلوبال
telegram_bot_token = None
bot_running = False
bot_thread = None
price_alerts = []  # لیست هشدارهای قیمت فعال
active_alerts_check = False  # وضعیت فعال بودن بررسی هشدارها
alerts_thread = None  # ترد بررسی هشدارها

def initialize_telegram_bot(token: str) -> bool:
    """
    راه‌اندازی اولیه ربات تلگرام
    
    Args:
        token (str): توکن ربات تلگرام
        
    Returns:
        bool: موفقیت یا عدم موفقیت
    """
    global telegram_bot_token
    
    try:
        # تنظیم توکن ربات
        telegram_bot_token = token
        
        # بررسی اعتبار توکن با دریافت اطلاعات ربات
        response = requests.get(
            f"https://api.telegram.org/bot{token}/getMe"
        )
        
        if response.status_code == 200:
            bot_info = response.json()
            logger.info(f"ربات تلگرام با موفقیت راه‌اندازی شد: {bot_info['result']['username']}")
            return True
        else:
            logger.error(f"خطا در راه‌اندازی ربات تلگرام: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"خطا در راه‌اندازی ربات تلگرام: {str(e)}")
        return False

def send_telegram_message(chat_id: str, message: str) -> bool:
    """
    ارسال پیام متنی به تلگرام
    
    Args:
        chat_id (str): شناسه چت
        message (str): متن پیام
        
    Returns:
        bool: موفقیت یا عدم موفقیت
    """
    global telegram_bot_token
    
    if not telegram_bot_token:
        logger.error("توکن ربات تلگرام تنظیم نشده است.")
        return False
        
    try:
        # ارسال پیام با استفاده از API تلگرام
        response = requests.post(
            f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
        )
        
        if response.status_code == 200:
            logger.info(f"پیام با موفقیت به چت {chat_id} ارسال شد")
            return True
        else:
            logger.error(f"خطا در ارسال پیام به تلگرام: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"خطا در ارسال پیام به تلگرام: {str(e)}")
        return False

def send_telegram_photo(chat_id: str, photo_buffer: io.BytesIO, caption: Optional[str] = None) -> bool:
    """
    ارسال تصویر به تلگرام
    
    Args:
        chat_id (str): شناسه چت
        photo_buffer (io.BytesIO): بافر تصویر
        caption (str, optional): توضیحات تصویر
        
    Returns:
        bool: موفقیت یا عدم موفقیت
    """
    global telegram_bot_token
    
    if not telegram_bot_token:
        logger.error("توکن ربات تلگرام تنظیم نشده است.")
        return False
        
    try:
        # آماده‌سازی تصویر
        photo_buffer.seek(0)
        
        # ارسال تصویر با استفاده از API تلگرام
        files = {
            'photo': ('chart.jpg', photo_buffer, 'image/jpeg')
        }
        
        data = {
            'chat_id': chat_id
        }
        
        if caption:
            data['caption'] = caption
            data['parse_mode'] = 'HTML'
        
        response = requests.post(
            f"https://api.telegram.org/bot{telegram_bot_token}/sendPhoto",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            logger.info(f"تصویر با موفقیت به چت {chat_id} ارسال شد")
            return True
        else:
            logger.error(f"خطا در ارسال تصویر به تلگرام: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"خطا در ارسال تصویر به تلگرام: {str(e)}")
        return False

def send_signal_message(
    chat_id: str, 
    symbol: str, 
    timeframe: str, 
    signal: str, 
    price: float, 
    signals_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    targets: Optional[Dict[str, Any]] = None,
    reasons: Optional[List[str]] = None
) -> bool:
    """
    ارسال پیام سیگنال به تلگرام با اهداف قیمتی و دلایل
    
    Args:
        chat_id (str): شناسه چت
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        signal (str): سیگنال (BUY, SELL, NEUTRAL)
        price (float): قیمت فعلی
        signals_dict (dict, optional): دیکشنری سیگنال‌های اندیکاتورهای مختلف
        targets (dict, optional): دیکشنری اهداف قیمتی
        reasons (list, optional): لیست دلایل سیگنال
        
    Returns:
        bool: موفقیت یا عدم موفقیت
    """
    try:
        # تعیین ایموجی و رنگ سیگنال
        signal_emoji = "🔴"
        if signal == "BUY":
            signal_emoji = "🟢"
        elif signal == "NEUTRAL":
            signal_emoji = "🟡"
            
        # ساخت پیام
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"""
<b>{signal_emoji} سیگنال {signal} برای {symbol}</b>

<b>تایم‌فریم:</b> {timeframe}
<b>قیمت فعلی:</b> {price:.2f} USDT
<b>تاریخ و زمان:</b> {current_time}
"""

        # اضافه کردن اطلاعات اهداف در صورت وجود
        if targets and signal != "NEUTRAL":
            confidence = targets.get('confidence', 0)
            
            message += f"""
<b>📊 اهداف قیمتی ({confidence}% اطمینان):</b>
"""
            
            if signal == "BUY":
                message += f"""
• ورود: <b>{price:.2f}</b> USDT
• حد ضرر: <b>{targets.get('stop_loss', price * 0.97):.2f}</b> USDT ({((targets.get('stop_loss', price * 0.97) / price) - 1) * 100:.2f}%)

• هدف اول: <b>{targets.get('target_1', price * 1.05):.2f}</b> USDT ({((targets.get('target_1', price * 1.05) / price) - 1) * 100:.2f}%) - {targets.get('timeline_1', 'کوتاه مدت')}
• هدف دوم: <b>{targets.get('target_2', price * 1.10):.2f}</b> USDT ({((targets.get('target_2', price * 1.10) / price) - 1) * 100:.2f}%) - {targets.get('timeline_2', 'میان مدت')}
• هدف سوم: <b>{targets.get('target_3', price * 1.15):.2f}</b> USDT ({((targets.get('target_3', price * 1.15) / price) - 1) * 100:.2f}%) - {targets.get('timeline_3', 'بلند مدت')}

• نسبت سود به ضرر: <b>{targets.get('risk_reward_1', 2.0):.2f}</b>
"""
            elif signal == "SELL":
                message += f"""
• ورود: <b>{price:.2f}</b> USDT
• حد ضرر: <b>{targets.get('stop_loss', price * 1.03):.2f}</b> USDT ({((targets.get('stop_loss', price * 1.03) / price) - 1) * 100:.2f}%)

• هدف اول: <b>{targets.get('target_1', price * 0.95):.2f}</b> USDT ({((targets.get('target_1', price * 0.95) / price) - 1) * 100:.2f}%) - {targets.get('timeline_1', 'کوتاه مدت')}
• هدف دوم: <b>{targets.get('target_2', price * 0.90):.2f}</b> USDT ({((targets.get('target_2', price * 0.90) / price) - 1) * 100:.2f}%) - {targets.get('timeline_2', 'میان مدت')}
• هدف سوم: <b>{targets.get('target_3', price * 0.85):.2f}</b> USDT ({((targets.get('target_3', price * 0.85) / price) - 1) * 100:.2f}%) - {targets.get('timeline_3', 'بلند مدت')}

• نسبت سود به ضرر: <b>{targets.get('risk_reward_1', 2.0):.2f}</b>
"""
        
        # اضافه کردن دلایل در صورت وجود
        if reasons and len(reasons) > 0:
            message += "\n<b>📝 دلایل سیگنال:</b>\n"
            for idx, reason in enumerate(reasons[:3], 1):
                message += f"{idx}. {reason}\n"
                
        # اضافه کردن اطلاعات اندیکاتورها
        message += "\n<b>🔍 تحلیل اندیکاتورها:</b>\n"
        if signals_dict:
            for indicator, signal_data in signals_dict.items():
                if isinstance(signal_data, dict) and 'signal' in signal_data:
                    ind_signal = signal_data['signal']
                    ind_desc = signal_data.get('description', '-')
                    
                    ind_emoji = "🔴"
                    if ind_signal == "BUY":
                        ind_emoji = "🟢"
                    elif ind_signal == "NEUTRAL":
                        ind_emoji = "🟡"
                        
                    message += f"{ind_emoji} <b>{indicator}:</b> {ind_desc}\n"
        
        message += f"""
<i>📊 برای تحلیل بیشتر، از دستورات زیر استفاده کنید:
/signal {symbol.split('/')[0]} - دریافت سیگنال معاملاتی
/price {symbol.split('/')[0]} - دریافت قیمت و تغییرات
/analysis {symbol.split('/')[0]} - دریافت تحلیل کامل
/indicators {symbol.split('/')[0]} - مشاهده وضعیت اندیکاتورها
/help - راهنمای دستورات</i>
"""
        
        # ارسال پیام
        return send_telegram_message(chat_id, message)
        
    except Exception as e:
        logger.error(f"خطا در ارسال پیام سیگنال به تلگرام: {str(e)}")
        return False

def start_telegram_bot(
    token: str,
    chat_id: str,
    symbols: List[str],
    interval: str,
    signal_type: str,
    selected_indicators: List[str],
    send_chart: bool
) -> None:
    """
    شروع سرویس خودکار ربات تلگرام
    
    Args:
        token (str): توکن ربات تلگرام
        chat_id (str): شناسه چت
        symbols (list): لیست نمادهای مورد نظر
        interval (str): تناوب ارسال
        signal_type (str): نوع سیگنال
        selected_indicators (list): لیست اندیکاتورهای انتخاب شده
        send_chart (bool): ارسال نمودار
    """
    global telegram_bot_token, bot_running
    
    try:
        # تنظیم توکن ربات
        telegram_bot_token = token
        bot_running = True
        
        # تبدیل تناوب ارسال به ثانیه
        interval_seconds = 3600  # پیش‌فرض: هر 1 ساعت
        if interval == "هر 4 ساعت":
            interval_seconds = 4 * 3600
        elif interval == "هر 12 ساعت":
            interval_seconds = 12 * 3600
        elif interval == "هر 24 ساعت":
            interval_seconds = 24 * 3600
        elif interval == "فقط سیگنال‌های مهم":
            interval_seconds = None  # حالت خاص
        
        logger.info(f"سرویس ربات تلگرام با موفقیت شروع شد (تناوب: {interval}, سیگنال: {signal_type})")
        
        # ارسال پیام شروع سرویس
        start_message = f"""
<b>🤖 سرویس سیگنال خودکار فعال شد</b>

<b>ارزهای تحت نظر:</b> {', '.join(symbols)}
<b>تناوب ارسال:</b> {interval}
<b>نوع سیگنال:</b> {signal_type}
<b>ارسال نمودار:</b> {'فعال' if send_chart else 'غیرفعال'}

<i>سیگنال‌ها به صورت خودکار ارسال خواهند شد.</i>
"""
        send_telegram_message(chat_id, start_message)
        
        # حلقه اصلی سرویس
        while bot_running:
            if interval_seconds is None:
                # حالت فقط سیگنال‌های مهم
                # در این حالت، سرویس فقط در انتظار می‌ماند و سیگنال‌ها از جای دیگر ارسال می‌شوند
                time.sleep(60)
                continue
                
            # بررسی هر نماد و ارسال سیگنال
            for symbol in symbols:
                try:
                    # در اینجا به جای فراخوانی مستقیم توابع تحلیل، 
                    # فقط یک پیام اطلاع‌رسانی می‌فرستیم
                    # در یک سیستم واقعی، اینجا باید توابع تحلیل و دریافت داده‌ها فراخوانی شوند
                    
                    # دریافت داده‌های بازار و تولید سیگنال واقعی
                    from crypto_data import get_crypto_data, get_current_price
                    from technical_analysis import generate_signals, perform_technical_analysis
                    
                    # دریافت داده‌های ارز
                    df = get_crypto_data(symbol, '1h', lookback_days=3)
                    if df is None or df.empty:
                        logger.error(f"خطا در دریافت داده‌های {symbol}")
                        continue
                        
                    # انجام تحلیل تکنیکال
                    df = perform_technical_analysis(df, selected_indicators)
                    
                    # دریافت قیمت فعلی
                    current_price = get_current_price(symbol)
                    
                    # تولید سیگنال‌ها
                    signals = generate_signals(df)
                    
                    # تولید پیام برای ارسال
                    message = f"""
<b>🔔 سیگنال معاملاتی {symbol}</b>

<b>قیمت فعلی:</b> {current_price:.2f} USDT
<b>سیگنال:</b> {signals['combined_signal']} (اطمینان: {signals.get('confidence', 75)}%)
<b>تایم‌فریم:</b> 1h
<b>زمان:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>توصیه‌های معاملاتی:</b>
"""
                    # اضافه کردن توصیه‌های معاملاتی بر اساس سیگنال
                    if signals['combined_signal'] == "BUY":
                        message += """
• ورود: قیمت فعلی
• حد ضرر: 2-3% زیر قیمت ورود
• هدف اول: 5-7% بالاتر از قیمت ورود
• هدف دوم: 10-15% بالاتر از قیمت ورود
"""
                    elif signals['combined_signal'] == "SELL":
                        message += """
• خروج: قیمت فعلی
• شورت: در صورت شکست حمایت‌های نزدیک
• هدف اول: 5-7% پایین‌تر از قیمت فعلی
• هدف دوم: 10-15% پایین‌تر از قیمت فعلی
"""
                    else:
                        message += """
• حالت انتظار
• صبر کنید تا سیگنال واضح‌تری تشکیل شود
• در این شرایط از معاملات با اهرم بالا خودداری کنید
"""

                    message += """
<i>برای اطلاعات بیشتر، از دستورات /signal, /price و /analysis استفاده کنید</i>
"""
                    send_telegram_message(chat_id, message)
                    
                except Exception as e:
                    logger.error(f"خطا در پردازش نماد {symbol}: {str(e)}")
            
            # انتظار تا زمان بعدی
            time.sleep(interval_seconds)
            
    except Exception as e:
        logger.error(f"خطا در اجرای سرویس ربات تلگرام: {str(e)}")
        bot_running = False

def check_price_alerts():
    """
    بررسی مداوم هشدارهای قیمتی تنظیم شده
    """
    global active_alerts_check, price_alerts
    
    logger.info("سرویس بررسی هشدارهای قیمت شروع شد")
    
    while active_alerts_check and len(price_alerts) > 0:
        try:
            from crypto_data import get_current_price
            
            # بررسی هر هشدار
            triggered_alerts = []
            
            for idx, alert in enumerate(price_alerts):
                try:
                    symbol = alert['symbol']
                    target_price = alert['price']
                    condition = alert['condition']  # "above" یا "below"
                    chat_id = alert['chat_id']
                    
                    # دریافت قیمت فعلی
                    current_price = get_current_price(symbol)
                    
                    # بررسی شرط هشدار
                    if (condition == "above" and current_price >= target_price) or \
                       (condition == "below" and current_price <= target_price):
                        # ارسال پیام هشدار
                        alert_message = f"""
<b>⚠️ هشدار قیمت فعال شد!</b>

<b>ارز:</b> {symbol}
<b>قیمت فعلی:</b> {current_price:.2f} USDT
<b>قیمت هدف:</b> {target_price:.2f} USDT
<b>شرط:</b> {'بالاتر از' if condition == 'above' else 'پایین‌تر از'}
<b>زمان:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>این هشدار از لیست هشدارهای فعال حذف شد.</i>
"""
                        send_telegram_message(chat_id, alert_message)
                        # اضافه کردن به لیست هشدارهای فعال شده
                        triggered_alerts.append(idx)
                        
                except Exception as e:
                    logger.error(f"خطا در بررسی هشدار {idx}: {str(e)}")
            
            # حذف هشدارهای فعال شده
            for idx in sorted(triggered_alerts, reverse=True):
                price_alerts.pop(idx)
            
            # انتظار برای بررسی بعدی
            time.sleep(60)  # بررسی هر 60 ثانیه
            
        except Exception as e:
            logger.error(f"خطا در بررسی هشدارهای قیمت: {str(e)}")
            time.sleep(300)  # در صورت خطا، 5 دقیقه صبر کن و دوباره تلاش کن

def start_price_alert_service():
    """
    شروع سرویس بررسی هشدارهای قیمت
    """
    global active_alerts_check, alerts_thread
    
    if not active_alerts_check:
        active_alerts_check = True
        alerts_thread = threading.Thread(target=check_price_alerts)
        alerts_thread.daemon = True
        alerts_thread.start()
        logger.info("سرویس بررسی هشدارهای قیمت شروع شد")
        return True
    
    return False

def stop_price_alert_service():
    """
    توقف سرویس بررسی هشدارهای قیمت
    """
    global active_alerts_check, alerts_thread
    
    if active_alerts_check:
        active_alerts_check = False
        
        if alerts_thread and alerts_thread.is_alive():
            alerts_thread.join(timeout=5)
            
        logger.info("سرویس بررسی هشدارهای قیمت متوقف شد")
        return True
    
    return False

def handle_telegram_command(chat_id: str, command: str) -> bool:
    """
    پردازش دستورات تلگرام
    
    Args:
        chat_id (str): شناسه چت
        command (str): دستور دریافت شده
        
    Returns:
        bool: موفقیت یا عدم موفقیت
    """
    try:
        logger.info(f"دریافت دستور تلگرام: {command}")
        
        # دستور /signal برای دریافت سیگنال‌های فعلی
        if command.startswith('/signal') or command.startswith('/سیگنال'):
            parts = command.split()
            symbol = "BTC/USDT"  # مقدار پیش‌فرض
            
            if len(parts) > 1:
                symbol = parts[1].upper()
                if not '/' in symbol:
                    symbol = f"{symbol}/USDT"
            
            from crypto_data import get_crypto_data, get_current_price
            from technical_analysis import perform_technical_analysis, generate_signals
            
            # دریافت داده‌های ارز
            df = get_crypto_data(symbol, '1h', lookback_days=3)
            if df is None or df.empty:
                message = f"خطا در دریافت داده‌های {symbol}"
                send_telegram_message(chat_id, message)
                return False
            
            # انجام تحلیل تکنیکال
            df = perform_technical_analysis(df)
            
            # دریافت قیمت فعلی
            current_price = get_current_price(symbol)
            
            # تولید سیگنال‌ها
            signal_result = generate_signals(df)
            
            combined_signal = signal_result.get('combined_signal', 'NEUTRAL')
            signals_dict = signal_result.get('signals_dict', {})
            targets = signal_result.get('targets', {})
            reasons = signal_result.get('reasons', [])
            
            # ارسال پیام سیگنال با اطلاعات کامل‌تر
            return send_signal_message(
                chat_id=chat_id,
                symbol=symbol,
                timeframe='1h',
                signal=combined_signal,
                price=current_price,
                signals_dict=signals_dict,
                targets=targets,
                reasons=reasons
            )
            
        # دستور /price برای دریافت قیمت فعلی
        elif command.startswith('/price') or command.startswith('/قیمت'):
            parts = command.split()
            symbol = "BTC/USDT"  # مقدار پیش‌فرض
            
            if len(parts) > 1:
                symbol = parts[1].upper()
                if not '/' in symbol:
                    symbol = f"{symbol}/USDT"
            
            from crypto_data import get_current_price, get_crypto_data
            
            # دریافت قیمت فعلی
            current_price = get_current_price(symbol)
            
            # دریافت تغییرات قیمت
            df = get_crypto_data(symbol, '1d', lookback_days=7)
            change_24h = 0
            change_7d = 0
            
            if df is not None and not df.empty:
                if len(df) > 1:
                    change_24h = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
                if len(df) > 7:
                    change_7d = ((df['close'].iloc[-1] / df['close'].iloc[-7]) - 1) * 100
            
            # ارسال پیام قیمت
            message = f"""
<b>💰 قیمت فعلی {symbol}</b>

<b>قیمت:</b> {current_price:.2f} USDT
<b>تغییر 24 ساعته:</b> {change_24h:.2f}%
<b>تغییر 7 روزه:</b> {change_7d:.2f}%
<b>زمان:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            return send_telegram_message(chat_id, message)
        
        # دستور /help برای نمایش راهنما
        elif command.startswith('/help') or command.startswith('/راهنما'):
            message = """
<b>🤖 راهنمای ربات تحلیل ارزهای دیجیتال</b>

<b>دستورات اصلی:</b>
/signal [symbol] - دریافت سیگنال معاملاتی با اهداف دقیق
/سیگنال [symbol] - دریافت سیگنال معاملاتی با اهداف دقیق
/price [symbol] - دریافت قیمت فعلی و تغییرات
/قیمت [symbol] - دریافت قیمت فعلی و تغییرات
/analysis [symbol] - دریافت تحلیل کامل با نمودار
/تحلیل [symbol] - دریافت تحلیل کامل با نمودار

<b>دستورات بیشتر:</b>
/indicators [symbol] - مشاهده وضعیت اندیکاتورها
/شاخص‌ها [symbol] - مشاهده وضعیت اندیکاتورها
/compare [symbol1] [symbol2] - مقایسه دو ارز
/مقایسه [symbol1] [symbol2] - مقایسه دو ارز
/alert [symbol] [price] - تنظیم هشدار قیمت (بالاتر رفتن)
/alert_down [symbol] [price] - تنظیم هشدار کاهش قیمت
/هشدار [symbol] [price] - تنظیم هشدار قیمت
/alerts - مشاهده همه هشدارهای فعال
/هشدارها - مشاهده همه هشدارهای فعال
/delete_alert [شماره] - حذف هشدار با شماره مشخص
/status - وضعیت سیستم و بازار
/وضعیت - وضعیت سیستم و بازار
/help - نمایش این راهنما
/راهنما - نمایش این راهنما

<b>مثال‌ها:</b> 
/signal BTC - دریافت سیگنال بیت‌کوین
/price ETH - مشاهده قیمت اتریوم
/analysis SOL - تحلیل کامل سولانا
/compare BTC ETH - مقایسه بیت‌کوین و اتریوم
/alert BTC 65000 - هشدار وقتی بیت‌کوین به 65000 برسد
/alert_down ETH 3000 - هشدار وقتی اتریوم زیر 3000 برود
/alerts - مشاهده همه هشدارهای فعال
/delete_alert 2 - حذف هشدار شماره 2
"""
            return send_telegram_message(chat_id, message)
            
        # دستور /analysis برای دریافت تحلیل کامل
        elif command.startswith('/analysis') or command.startswith('/تحلیل'):
            parts = command.split()
            symbol = "BTC/USDT"  # مقدار پیش‌فرض
            
            if len(parts) > 1:
                symbol = parts[1].upper()
                if not '/' in symbol:
                    symbol = f"{symbol}/USDT"
            
            # ارسال پیام در حال تحلیل
            send_telegram_message(chat_id, f"در حال تحلیل {symbol}، لطفاً منتظر بمانید...")
            
            # انجام تحلیل
            from auto_analysis_service import perform_one_time_analysis
            
            # دریافت توکن و چت آیدی
            global telegram_bot_token
            if telegram_bot_token:
                perform_one_time_analysis(symbol, '1h', telegram_bot_token, chat_id)
                return True
            else:
                send_telegram_message(chat_id, "خطا: توکن تلگرام تنظیم نشده است.")
                return False
        
        # دستورات مربوط به هشدارهای قیمت
        elif command.startswith('/alert') and not command.startswith('/alerts') and not command.startswith('/alert_down'):
            # تنظیم هشدار قیمت (بالاتر رفتن)
            
            parts = command.split()
            if len(parts) < 3:
                return send_telegram_message(chat_id, "فرمت صحیح: /alert BTC 65000")
            
            symbol = parts[1].upper()
            if not '/' in symbol:
                symbol = f"{symbol}/USDT"
                
            try:
                target_price = float(parts[2])
            except ValueError:
                return send_telegram_message(chat_id, "قیمت باید یک عدد باشد")
            
            # دریافت قیمت فعلی
            from crypto_data import get_current_price
            current_price = get_current_price(symbol)
            
            if current_price >= target_price:
                return send_telegram_message(chat_id, f"قیمت فعلی {current_price:.2f} از قیمت هدف {target_price:.2f} بالاتر است!")
            
            # ایجاد هشدار جدید
            alert = {
                'symbol': symbol,
                'price': target_price,
                'condition': 'above',  # بالاتر رفتن
                'chat_id': chat_id,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            price_alerts.append(alert)
            
            # شروع سرویس بررسی هشدارها اگر فعال نیست
            if not active_alerts_check:
                start_price_alert_service()
                
            message = f"""
<b>⚙️ هشدار قیمت تنظیم شد</b>

<b>ارز:</b> {symbol}
<b>قیمت فعلی:</b> {current_price:.2f} USDT
<b>قیمت هدف:</b> {target_price:.2f} USDT
<b>شرط:</b> بالاتر رفتن از قیمت هدف
<b>شماره هشدار:</b> {len(price_alerts)}

<i>هشدار پس از رسیدن به قیمت هدف، ارسال و حذف خواهد شد.</i>
"""
            return send_telegram_message(chat_id, message)
            
        elif command.startswith('/alert_down'):
            # تنظیم هشدار قیمت (پایین رفتن)
            
            parts = command.split()
            if len(parts) < 3:
                return send_telegram_message(chat_id, "فرمت صحیح: /alert_down BTC 55000")
            
            symbol = parts[1].upper()
            if not '/' in symbol:
                symbol = f"{symbol}/USDT"
                
            try:
                target_price = float(parts[2])
            except ValueError:
                return send_telegram_message(chat_id, "قیمت باید یک عدد باشد")
            
            # دریافت قیمت فعلی
            from crypto_data import get_current_price
            current_price = get_current_price(symbol)
            
            if current_price <= target_price:
                return send_telegram_message(chat_id, f"قیمت فعلی {current_price:.2f} از قیمت هدف {target_price:.2f} پایین‌تر است!")
            
            # ایجاد هشدار جدید
            alert = {
                'symbol': symbol,
                'price': target_price,
                'condition': 'below',  # پایین رفتن
                'chat_id': chat_id,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            price_alerts.append(alert)
            
            # شروع سرویس بررسی هشدارها اگر فعال نیست
            if not active_alerts_check:
                start_price_alert_service()
                
            message = f"""
<b>⚙️ هشدار قیمت تنظیم شد</b>

<b>ارز:</b> {symbol}
<b>قیمت فعلی:</b> {current_price:.2f} USDT
<b>قیمت هدف:</b> {target_price:.2f} USDT
<b>شرط:</b> پایین رفتن از قیمت هدف
<b>شماره هشدار:</b> {len(price_alerts)}

<i>هشدار پس از رسیدن به قیمت هدف، ارسال و حذف خواهد شد.</i>
"""
            return send_telegram_message(chat_id, message)
            
        elif command.startswith('/alerts') or command.startswith('/هشدارها'):
            # مشاهده لیست هشدارهای فعال
            if not price_alerts:
                return send_telegram_message(chat_id, "هیچ هشدار فعالی وجود ندارد.")
            
            message = "<b>📋 لیست هشدارهای فعال:</b>\n\n"
            
            for idx, alert in enumerate(price_alerts, 1):
                if alert['chat_id'] == chat_id:  # فقط هشدارهای این چت
                    symbol = alert['symbol']
                    price = alert['price']
                    condition = 'بالاتر رفتن از' if alert['condition'] == 'above' else 'پایین رفتن از'
                    created_at = alert.get('created_at', 'نامشخص')
                    
                    message += f"<b>{idx}.</b> {symbol} - {condition} {price:.2f} USDT (تنظیم شده در: {created_at})\n"
            
            message += "\n<i>برای حذف هشدار از دستور /delete_alert [شماره] استفاده کنید</i>"
            
            return send_telegram_message(chat_id, message)
            
        elif command.startswith('/delete_alert') or command.startswith('/حذف_هشدار'):
            # حذف هشدار با شماره مشخص
            parts = command.split()
            if len(parts) < 2:
                return send_telegram_message(chat_id, "فرمت صحیح: /delete_alert 1")
            
            try:
                alert_idx = int(parts[1]) - 1
            except ValueError:
                return send_telegram_message(chat_id, "شماره هشدار باید یک عدد باشد")
            
            if alert_idx < 0 or alert_idx >= len(price_alerts):
                return send_telegram_message(chat_id, "شماره هشدار نامعتبر است")
            
            if price_alerts[alert_idx]['chat_id'] != chat_id:
                return send_telegram_message(chat_id, "دسترسی به این هشدار امکان‌پذیر نیست")
            
            # حذف هشدار
            deleted_alert = price_alerts.pop(alert_idx)
            
            message = f"""
<b>✅ هشدار با موفقیت حذف شد</b>

<b>ارز:</b> {deleted_alert['symbol']}
<b>قیمت هدف:</b> {deleted_alert['price']:.2f} USDT
<b>شرط:</b> {'بالاتر رفتن از' if deleted_alert['condition'] == 'above' else 'پایین رفتن از'} قیمت هدف
"""
            return send_telegram_message(chat_id, message)
            
        elif command.startswith('/compare') or command.startswith('/مقایسه'):
            # مقایسه دو ارز
            parts = command.split()
            if len(parts) < 3:
                return send_telegram_message(chat_id, "فرمت صحیح: /compare BTC ETH")
            
            symbol1 = parts[1].upper()
            if not '/' in symbol1:
                symbol1 = f"{symbol1}/USDT"
                
            symbol2 = parts[2].upper()
            if not '/' in symbol2:
                symbol2 = f"{symbol2}/USDT"
            
            # دریافت داده‌های ارزها
            from crypto_data import get_crypto_data, get_current_price
            
            # ارسال پیام در حال پردازش
            send_telegram_message(chat_id, f"در حال مقایسه {symbol1} و {symbol2}، لطفاً منتظر بمانید...")
            
            # دریافت داده‌ها
            df1 = get_crypto_data(symbol1, '1d', lookback_days=30)
            df2 = get_crypto_data(symbol2, '1d', lookback_days=30)
            
            if df1 is None or df1.empty or df2 is None or df2.empty:
                return send_telegram_message(chat_id, f"خطا در دریافت داده‌های {symbol1} یا {symbol2}")
            
            # محاسبه تغییرات
            price1 = get_current_price(symbol1)
            price2 = get_current_price(symbol2)
            
            change1_1d = ((df1['close'].iloc[-1] / df1['close'].iloc[-2]) - 1) * 100
            change1_7d = ((df1['close'].iloc[-1] / df1['close'].iloc[-7]) - 1) * 100
            change1_30d = ((df1['close'].iloc[-1] / df1['close'].iloc[0]) - 1) * 100
            
            change2_1d = ((df2['close'].iloc[-1] / df2['close'].iloc[-2]) - 1) * 100
            change2_7d = ((df2['close'].iloc[-1] / df2['close'].iloc[-7]) - 1) * 100
            change2_30d = ((df2['close'].iloc[-1] / df2['close'].iloc[0]) - 1) * 100
            
            # محاسبه نسبت قیمت‌ها
            ratio_current = price1 / price2
            ratio_1d_ago = df1['close'].iloc[-2] / df2['close'].iloc[-2]
            ratio_7d_ago = df1['close'].iloc[-7] / df2['close'].iloc[-7]
            ratio_30d_ago = df1['close'].iloc[0] / df2['close'].iloc[0]
            
            # تغییر نسبت
            ratio_change_1d = ((ratio_current / ratio_1d_ago) - 1) * 100
            ratio_change_7d = ((ratio_current / ratio_7d_ago) - 1) * 100
            ratio_change_30d = ((ratio_current / ratio_30d_ago) - 1) * 100
            
            # ساخت پیام
            message = f"""
<b>📊 مقایسه {symbol1.split('/')[0]} و {symbol2.split('/')[0]}</b>

<b>قیمت‌های فعلی:</b>
• {symbol1.split('/')[0]}: {price1:.2f} USDT
• {symbol2.split('/')[0]}: {price2:.2f} USDT
• نسبت: {ratio_current:.6f} {symbol1.split('/')[0]}/{symbol2.split('/')[0]}

<b>تغییرات 24 ساعته:</b>
• {symbol1.split('/')[0]}: {change1_1d:.2f}%
• {symbol2.split('/')[0]}: {change2_1d:.2f}%
• تفاوت: {change1_1d - change2_1d:.2f}%
• تغییر نسبت: {ratio_change_1d:.2f}%

<b>تغییرات 7 روزه:</b>
• {symbol1.split('/')[0]}: {change1_7d:.2f}%
• {symbol2.split('/')[0]}: {change2_7d:.2f}%
• تفاوت: {change1_7d - change2_7d:.2f}%
• تغییر نسبت: {ratio_change_7d:.2f}%

<b>تغییرات 30 روزه:</b>
• {symbol1.split('/')[0]}: {change1_30d:.2f}%
• {symbol2.split('/')[0]}: {change2_30d:.2f}%
• تفاوت: {change1_30d - change2_30d:.2f}%
• تغییر نسبت: {ratio_change_30d:.2f}%

<b>نتیجه مقایسه:</b>
• {'🟢 ' + symbol1.split('/')[0] if change1_30d > change2_30d else '🔴 ' + symbol2.split('/')[0]} در 30 روز گذشته عملکرد بهتری داشته است.
• {'🟢 ' + symbol1.split('/')[0] if change1_7d > change2_7d else '🔴 ' + symbol2.split('/')[0]} در 7 روز گذشته عملکرد بهتری داشته است.
• {'🟢 ' + symbol1.split('/')[0] if change1_1d > change2_1d else '🔴 ' + symbol2.split('/')[0]} در 24 ساعت گذشته عملکرد بهتری داشته است.
"""
            
            return send_telegram_message(chat_id, message)
            
        elif command.startswith('/status') or command.startswith('/وضعیت'):
            # نمایش وضعیت سیستم و بازار
            from crypto_data import get_current_price, get_crypto_data
            
            # بررسی وضعیت بازار
            btc_price = get_current_price("BTC/USDT")
            eth_price = get_current_price("ETH/USDT")
            
            # شاخص ترس و طمع
            fear_greed_index = "نامشخص"
            fear_greed_value = 0
            try:
                from sentiment_analysis_fixed import get_fear_greed_index
                fear_greed_result = get_fear_greed_index()
                if fear_greed_result:
                    fear_greed_value = fear_greed_result.get('current_value', 0)
                    fear_greed_index = fear_greed_result.get('status_fa', 'نامشخص')
            except Exception as e:
                logger.error(f"خطا در دریافت شاخص ترس و طمع: {str(e)}")
                fear_greed_value = 50
                fear_greed_index = "خنثی"
            
            # وضعیت سرویس‌ها
            bot_status = "فعال" if bot_running else "غیرفعال"
            alerts_status = "فعال" if active_alerts_check else "غیرفعال"
            alerts_count = len(price_alerts)
            
            # ارسال پیام وضعیت
            message = f"""
<b>📊 وضعیت سیستم و بازار</b>

<b>سیستم:</b>
• سرویس ربات: {bot_status}
• سرویس هشدارها: {alerts_status} ({alerts_count} هشدار فعال)

<b>بازار:</b>
• قیمت بیت‌کوین: {btc_price:.2f} USDT
• قیمت اتریوم: {eth_price:.2f} USDT
• شاخص ترس و طمع: {fear_greed_value} ({fear_greed_index})

<b>زمان:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return send_telegram_message(chat_id, message)
        
        else:
            # دستور ناشناخته
            message = "دستور نامعتبر است. برای مشاهده دستورات قابل استفاده، از /help یا /راهنما استفاده کنید."
            return send_telegram_message(chat_id, message)
            
    except Exception as e:
        logger.error(f"خطا در پردازش دستور تلگرام: {str(e)}")
        send_telegram_message(chat_id, f"خطا در پردازش دستور: {str(e)}")
        return False

def stop_telegram_bot() -> bool:
    """
    توقف سرویس ربات تلگرام
    
    Returns:
        bool: موفقیت یا عدم موفقیت
    """
    global bot_running, bot_thread
    
    try:
        bot_running = False
        
        if bot_thread and bot_thread.is_alive():
            # انتظار برای پایان thread
            bot_thread.join(timeout=5)
            
        logger.info("سرویس ربات تلگرام با موفقیت متوقف شد")
        return True
        
    except Exception as e:
        logger.error(f"خطا در توقف سرویس ربات تلگرام: {str(e)}")
        return False
