"""
ماژول رابط مدل‌های زبانی برای تحلیل و توضیح نتایج

این ماژول شامل کلاس و توابع مورد نیاز برای تعامل با مدل‌های زبانی برای تحلیل و توضیح نتایج تحلیل تکنیکال است.
"""

import os
import json
import time
import streamlit as st
import random
from typing import Dict, List, Any

class LanguageModelInterface:
    """کلاس رابط مدل‌های زبانی"""
    
    def __init__(self, model_name="local-model"):
        """
        مقداردهی اولیه رابط مدل زبانی
        
        Args:
            model_name (str): نام مدل زبانی
        """
        self.model_name = model_name
        self.cache = {}
        self.cache_ttl = 3600  # زمان اعتبار کش (ثانیه)
        
        # دیکشنری پاسخ‌های از پیش آماده شده برای سوالات متداول
        self.predefined_responses = self._load_predefined_responses()
    
    def _load_predefined_responses(self):
        """
        بارگذاری پاسخ‌های از پیش آماده شده
        
        Returns:
            dict: دیکشنری پاسخ‌های آماده
        """
        # این بخش می‌تواند از یک فایل JSON خارجی بارگذاری شود
        return {
            "RSI": """
            ### شاخص قدرت نسبی (RSI)
            
            **توضیح مختصر:**
            شاخص قدرت نسبی (RSI) یک اندیکاتور مومنتوم است که تغییرات قیمت را اندازه‌گیری می‌کند تا سرعت و تغییر حرکت‌های قیمتی را ارزیابی کند.
            
            **نحوه محاسبه:**
            RSI با اندازه‌گیری میانگین افزایش‌ها و کاهش‌های قیمت در یک دوره زمانی مشخص (معمولاً 14 روز) محاسبه می‌شود:
            RSI = 100 - (100 / (1 + RS))
            که در آن RS نسبت میانگین حرکت‌های صعودی به میانگین حرکت‌های نزولی است.
            
            **تفسیر مقادیر:**
            - RSI بالای 70: شرایط اشباع خرید (احتمال برگشت قیمت به سمت پایین)
            - RSI بین 30 و 70: منطقه خنثی
            - RSI زیر 30: شرایط اشباع فروش (احتمال برگشت قیمت به سمت بالا)
            
            **سیگنال‌های معاملاتی:**
            - خرید: RSI از زیر 30 به بالا حرکت کند
            - فروش: RSI از بالای 70 به پایین حرکت کند
            - واگرایی: ناهمخوانی بین حرکت قیمت و شاخص RSI
            """,
            
            "MACD": """
            ### واگرایی میانگین متحرک همگرا (MACD)
            
            **توضیح مختصر:**
            MACD یک اندیکاتور روند و مومنتوم است که رابطه بین دو میانگین متحرک قیمت را نشان می‌دهد.
            
            **نحوه محاسبه:**
            MACD = EMA(12) - EMA(26)
            خط سیگنال = EMA(9) از MACD
            هیستوگرام = MACD - خط سیگنال
            
            EMA = میانگین متحرک نمایی
            
            **تفسیر مقادیر:**
            - MACD بالای صفر: روند صعودی
            - MACD زیر صفر: روند نزولی
            - هیستوگرام مثبت: قدرت صعودی
            - هیستوگرام منفی: قدرت نزولی
            """,
            
            "Bollinger Bands": """
            ### باندهای بولینگر
            
            **توضیح مختصر:**
            باندهای بولینگر اندیکاتوری است که نوسان‌پذیری قیمت را نشان می‌دهد و از یک میانگین متحرک و دو باند بالایی و پایینی تشکیل شده است.
            
            **نحوه محاسبه:**
            میانه باند = SMA(20)
            باند بالایی = SMA(20) + (انحراف معیار قیمت × 2)
            باند پایینی = SMA(20) - (انحراف معیار قیمت × 2)
            
            SMA = میانگین متحرک ساده
            """,
            
            "Head and Shoulders": """
            ### الگوی سر و شانه‌ها
            
            **توضیح مختصر:**
            الگوی سر و شانه‌ها یک الگوی برگشتی است که معمولاً در پایان یک روند صعودی شکل می‌گیرد و نشان‌دهنده تغییر روند از صعودی به نزولی است. نسخه معکوس آن (سر و شانه‌های معکوس) در پایان روند نزولی ظاهر می‌شود.
            
            **نحوه تشخیص:**
            1. شانه چپ: یک اوج یا فرود نسبی
            2. سر: اوج یا فرود بزرگتر از شانه‌ها
            3. شانه راست: اوج یا فرود مشابه با شانه چپ
            4. خط گردن: خط روندی که شانه‌ها را به هم متصل می‌کند
            """,
            
            "Double Top": """
            ### الگوی دابل تاپ (دو قله)
            
            **توضیح مختصر:**
            الگوی دابل تاپ یک الگوی برگشتی است که در پایان یک روند صعودی شکل می‌گیرد و شامل دو قله تقریباً هم‌ارتفاع است که نشان‌دهنده تغییر روند از صعودی به نزولی است.
            
            **نحوه تشخیص:**
            1. دو قله تقریباً هم‌ارتفاع
            2. دره میانی بین دو قله
            3. خط حمایت (خط گردن) که از دره میانی عبور می‌کند
            """,
        }
    
    def _save_to_cache(self, key, value):
        """
        ذخیره مقدار در کش
        
        Args:
            key (str): کلید
            value (Any): مقدار
        """
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
    
    def _get_from_cache(self, key):
        """
        دریافت مقدار از کش
        
        Args:
            key (str): کلید
            
        Returns:
            Any: مقدار یا None اگر موجود نباشد یا منقضی شده باشد
        """
        if key in self.cache:
            cache_entry = self.cache[key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["value"]
        return None
    
    def get_openai_completion(self, prompt):
        """
        شبیه‌سازی دریافت پاسخ از مدل زبانی هوشمند
        
        Args:
            prompt (str): متن درخواست
            
        Returns:
            str: پاسخ شبیه‌سازی شده
        """
        # بررسی کش
        cache_key = "local_" + prompt[:100]
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # بررسی کلمات کلیدی در پرامپت برای انتخاب پاسخ مناسب
        response = self._generate_response_based_on_prompt(prompt)
        
        # ذخیره در کش
        self._save_to_cache(cache_key, response)
        return response
    
    def _generate_response_based_on_prompt(self, prompt):
        """
        تولید پاسخ بر اساس محتوای پرامپت
        
        Args:
            prompt (str): متن درخواست
            
        Returns:
            str: پاسخ تولید شده
        """
        # بررسی کلمات کلیدی برای اندیکاتورها
        indicator_keywords = {
            "RSI": ["RSI", "شاخص قدرت نسبی", "اندیکاتور RSI", "اوسیلاتور RSI"],
            "MACD": ["MACD", "واگرایی میانگین متحرک", "مکدی"],
            "Bollinger Bands": ["Bollinger", "باند بولینگر", "باندهای بولینگر"],
        }
        
        # بررسی کلمات کلیدی برای الگوهای نموداری
        pattern_keywords = {
            "Head and Shoulders": ["سر و شانه", "الگوی سر و شانه", "Head and Shoulders"],
            "Double Top": ["دو قله", "دابل تاپ", "Double Top"],
        }
        
        # بررسی درخواست توضیح اندیکاتور
        for indicator, keywords in indicator_keywords.items():
            if any(keyword in prompt for keyword in keywords) and "توضیح" in prompt:
                if indicator in self.predefined_responses:
                    return self.predefined_responses[indicator]
        
        # بررسی درخواست توضیح الگوی نموداری
        for pattern, keywords in pattern_keywords.items():
            if any(keyword in prompt for keyword in keywords) and "توضیح" in prompt:
                if pattern in self.predefined_responses:
                    return self.predefined_responses[pattern]
        
        # پاسخ پیش‌فرض
        return """
        با توجه به بررسی داده‌ها و تحلیل‌های انجام شده، می‌توان گفت که بازار در حال حاضر در یک روند نسبتاً خنثی قرار دارد. 
        
        اندیکاتورهای فنی نشان می‌دهند که مومنتوم بازار متعادل است و نیاز به مشاهده دقیق‌تر برای شناسایی سیگنال‌های معتبر وجود دارد. 
        
        سطوح حمایت و مقاومت کلیدی باید به دقت بررسی شوند و بهتر است از استراتژی‌های مبتنی بر شکست این سطوح استفاده شود.
        """
    
    def analyze_market_condition(self, symbol, timeframe, price_data=None):
        """
        تحلیل شرایط بازار
        
        Args:
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            price_data (pd.DataFrame, optional): داده‌های قیمت
            
        Returns:
            str: تحلیل بازار
        """
        # تعیین روند تصادفی
        trends = ["صعودی قوی", "صعودی ضعیف", "نزولی قوی", "نزولی ضعیف", "خنثی"]
        trend = random.choice(trends)
        
        # تعیین سطوح حمایت و مقاومت تصادفی
        base_price = 30000 if "BTC" in symbol else (2000 if "ETH" in symbol else 0.5)
        resistance1 = base_price * (1 + random.uniform(0.02, 0.05))
        resistance2 = resistance1 * (1 + random.uniform(0.01, 0.03))
        support1 = base_price * (1 - random.uniform(0.02, 0.05))
        support2 = support1 * (1 - random.uniform(0.01, 0.03))
        
        # تعیین اندیکاتورها
        rsi_value = random.randint(30, 70)
        macd_signal = "صعودی" if random.random() > 0.5 else "نزولی"
        bb_position = random.choice(["نزدیک به باند بالایی", "نزدیک به باند میانی", "نزدیک به باند پایینی"])
        
        # تولید تحلیل
        analysis = f"""
        ## تحلیل {symbol} در تایم‌فریم {timeframe}
        
        ### تحلیل روند
        روند فعلی {symbol} **{trend}** است. حجم معاملات نشان‌دهنده {random.choice(['افزایش', 'کاهش', 'ثبات'])} علاقه معامله‌گران است.
        
        ### شاخص‌های فنی
        - RSI: {rsi_value} - {("اشباع خرید" if rsi_value > 70 else "اشباع فروش" if rsi_value < 30 else "خنثی")}
        - MACD: سیگنال {macd_signal}
        - باندهای بولینگر: قیمت {bb_position}
        
        ### سطوح کلیدی
        سطوح مقاومت:
        - R1: {resistance1:.2f}
        - R2: {resistance2:.2f}
        
        سطوح حمایت:
        - S1: {support1:.2f}
        - S2: {support2:.2f}
        
        ### پیش‌بینی کوتاه‌مدت
        با توجه به شرایط فعلی، روند کوتاه‌مدت {random.choice(['صعودی', 'نزولی', 'خنثی'])} به نظر می‌رسد. 
        {random.choice([
            'معامله‌گران باید منتظر شکست سطوح کلیدی باشند', 
            'استراتژی خرید در کف و فروش در سقف توصیه می‌شود', 
            'رعایت احتیاط و مدیریت سرمایه اهمیت بالایی دارد'
        ])}.
        """
        
        return analysis
    
    def suggest_trading_strategy(self, symbol, timeframe, indicators=None):
        """
        پیشنهاد استراتژی معاملاتی
        
        Args:
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            indicators (dict, optional): دیکشنری اندیکاتورها و مقادیر آنها
            
        Returns:
            str: استراتژی معاملاتی پیشنهادی
        """
        if indicators is None:
            indicators = {}
        
        indicators_text = ""
        for indicator, value in indicators.items():
            indicators_text += f"- {indicator}: {value}\n"
        
        strategies = [
            "معامله روند (Trend Following)",
            "معامله نوسانی (Swing Trading)",
            "معامله محدوده (Range Trading)",
            "استراتژی شکست (Breakout Strategy)",
            "استراتژی واگرایی (Divergence Strategy)"
        ]
        
        strategy = random.choice(strategies)
        
        response = f"""
        ## استراتژی معاملاتی {strategy}
        
        ### توضیح کلی
        {strategy} یک روش معاملاتی است که بر اساس {random.choice([
            'شناسایی و پیروی از روندهای اصلی بازار',
            'استفاده از نوسانات کوتاه‌مدت و میان‌مدت',
            'معامله در محدوده‌های مشخص قیمتی',
            'شناسایی و استفاده از شکست سطوح کلیدی',
            'تشخیص واگرایی بین قیمت و اندیکاتورها'
        ])} طراحی شده است.
        
        ### نقاط ورود
        {random.choice([
            'ورود پس از تأیید شکست سطح مقاومت/حمایت',
            'ورود در هنگام بازگشت قیمت به میانگین متحرک',
            'ورود در سطوح اشباع خرید/فروش اندیکاتورهای نوسانگر',
            'ورود پس از تشکیل الگوهای کندل استیک تأییدی',
            'ورود در سطوح فیبوناچی کلیدی'
        ])}
        
        ### مدیریت ریسک
        - حد ضرر: {random.choice(['زیر/بالای سطح حمایت/مقاومت اخیر', 'در فاصله 2-3% از نقطه ورود', 'زیر/بالای الگوی نموداری'])}
        - نسبت ریسک به سود: {random.choice(['1:2', '1:3', '1:1.5'])}
        - حداکثر ریسک هر معامله: {random.choice(['1%', '2%', '0.5%'])} از کل سرمایه
        
        ### زمان‌بندی خروج
        {random.choice([
            'خروج در رسیدن به اهداف قیمتی از پیش تعیین شده',
            'خروج تدریجی در سطوح مختلف سود',
            'خروج بر اساس سیگنال‌های واگرایی',
            'خروج در شکست خط روند اصلی',
            'استفاده از حد ضرر متحرک برای قفل سود'
        ])}
        """
        
        return response
    
    def explain_indicator(self, indicator_name):
        """
        توضیح یک اندیکاتور
        
        Args:
            indicator_name (str): نام اندیکاتور
            
        Returns:
            str: توضیح اندیکاتور
        """
        # بررسی اگر توضیح آماده وجود دارد
        for key, keywords in {
            "RSI": ["RSI", "rsi", "قدرت نسبی", "اندیکاتور RSI"],
            "MACD": ["MACD", "macd", "مکدی", "واگرایی میانگین"],
            "Bollinger Bands": ["Bollinger", "bollinger", "بولینگر", "باند بولینگر"],
        }.items():
            if any(keyword in indicator_name for keyword in keywords):
                if key in self.predefined_responses:
                    return self.predefined_responses[key]
        
        # پاسخ پیش‌فرض برای اندیکاتورهایی که توضیح آماده ندارند
        return f"""
        ### اندیکاتور {indicator_name}
        
        **توضیح مختصر:**
        {indicator_name} یک اندیکاتور تکنیکال است که برای تحلیل روند قیمت و پیش‌بینی حرکات آینده بازار استفاده می‌شود.
        
        **کاربردهای اصلی:**
        - شناسایی روند بازار
        - تشخیص نقاط ورود و خروج
        - اندازه‌گیری قدرت روند
        
        **بهترین شرایط استفاده:**
        این اندیکاتور در ترکیب با سایر ابزارهای تحلیل تکنیکال بهترین نتیجه را می‌دهد و نباید به تنهایی مبنای تصمیم‌گیری قرار گیرد.
        """
    
    def explain_pattern(self, pattern_name):
        """
        توضیح یک الگوی نموداری
        
        Args:
            pattern_name (str): نام الگو
            
        Returns:
            str: توضیح الگو
        """
        # بررسی اگر توضیح آماده وجود دارد
        for key, keywords in {
            "Head and Shoulders": ["سر و شانه", "Head and Shoulders", "head and shoulders", "سروشانه"],
            "Double Top": ["دو قله", "Double Top", "double top", "دابل تاپ"],
        }.items():
            if any(keyword in pattern_name for keyword in keywords):
                if key in self.predefined_responses:
                    return self.predefined_responses[key]
        
        # پاسخ پیش‌فرض برای الگوهایی که توضیح آماده ندارند
        return f"""
        ### الگوی {pattern_name}
        
        **توضیح مختصر:**
        {pattern_name} یک الگوی نموداری است که معمولاً نشان‌دهنده تغییر روند یا ادامه روند فعلی است.
        
        **اهمیت روانشناسی:**
        این الگو نشان‌دهنده تغییر در رفتار معامله‌گران و توازن بین قدرت خریداران و فروشندگان است.
        
        **دقت پیش‌بینی:**
        دقت این الگو بستگی به تایم‌فریم، حجم معاملات و تأییدات دیگر دارد. در تایم‌فریم‌های بزرگتر معمولاً اعتبار بیشتری دارد.
        """
    
    def generate_market_insight(self, top_coins=None):
        """
        تولید بینش بازار بر اساس وضعیت ارزهای دیجیتال برتر
        
        Args:
            top_coins (list, optional): لیستی از دیکشنری‌های اطلاعات ارزهای برتر
            
        Returns:
            str: بینش بازار
        """
        if top_coins is None or len(top_coins) == 0:
            top_coins = [
                {"name": "Bitcoin", "symbol": "BTC", "price": 29500 + random.uniform(-500, 500), "price_change_24h": random.uniform(-5, 5)},
                {"name": "Ethereum", "symbol": "ETH", "price": 1800 + random.uniform(-50, 50), "price_change_24h": random.uniform(-5, 5)},
                {"name": "Binance Coin", "symbol": "BNB", "price": 300 + random.uniform(-10, 10), "price_change_24h": random.uniform(-5, 5)},
                {"name": "XRP", "symbol": "XRP", "price": 0.5 + random.uniform(-0.05, 0.05), "price_change_24h": random.uniform(-5, 5)},
                {"name": "Cardano", "symbol": "ADA", "price": 0.3 + random.uniform(-0.03, 0.03), "price_change_24h": random.uniform(-5, 5)},
            ]
        
        # تعیین روند کلی بازار
        avg_change = sum(coin.get("price_change_24h", 0) for coin in top_coins) / len(top_coins)
        market_sentiment = "مثبت" if avg_change > 1 else "منفی" if avg_change < -1 else "خنثی"
        
        coins_info = []
        for coin in top_coins:
            coin_info = f"{coin['name']} ({coin['symbol']}): قیمت {coin.get('price', 'N/A'):.2f} دلار، تغییر 24 ساعته {coin.get('price_change_24h', 0):.2f}%"
            coins_info.append(coin_info)
        
        insight = f"""
        ## تحلیل کلی بازار ارزهای دیجیتال
        
        ### وضعیت فعلی بازار
        بازار ارزهای دیجیتال در 24 ساعت گذشته روند {market_sentiment} داشته است. میانگین تغییرات قیمت {avg_change:.2f}% بوده است.
        
        ### ارزهای برتر
        {chr(10).join(coins_info)}
        
        ### عوامل تأثیرگذار
        {random.choice([
            'تحولات تنظیم‌کننده و اخبار مرتبط با آن',
            'نوسانات بازارهای سنتی و تأثیر آن بر ارزهای دیجیتال',
            'ورود یا خروج سرمایه‌های نهادی به بازار',
            'پیشرفت‌های فنی و به‌روزرسانی‌های مهم در پروژه‌های اصلی',
            'نقدینگی و حجم معاملات در صرافی‌های اصلی'
        ])} از مهمترین عوامل تأثیرگذار بر بازار در دوره فعلی است.
        
        ### چشم‌انداز کوتاه‌مدت
        با توجه به شرایط فعلی، انتظار می‌رود در کوتاه‌مدت بازار روند {random.choice(['صعودی', 'نزولی', 'نوسانی'])} را تجربه کند.
        معامله‌گران باید {random.choice([
            'به مدیریت ریسک توجه ویژه داشته باشند',
            'از استراتژی‌های محافظه‌کارانه استفاده کنند',
            'به دنبال فرصت‌های خرید در اصلاحات قیمتی باشند',
            'منتظر شکست سطوح کلیدی برای ورود به معاملات جدید باشند'
        ])}.
        """
        
        return insight

    def analyze_price_action(self, symbol, timeframe, df=None):
        """
        تحلیل اکشن قیمت
        
        Args:
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            df (pd.DataFrame, optional): دیتافریم داده‌های قیمت
            
        Returns:
            str: تحلیل اکشن قیمت
        """
        try:
            # اگر داده‌های قیمت موجود نباشد، پیام مناسب برگردانده می‌شود
            if df is None or df.empty:
                return f"داده‌های کافی برای تحلیل اکشن قیمت {symbol} در تایم‌فریم {timeframe} موجود نیست."
            
            # اطلاعات اولیه قیمت
            current_price = df['close'].iloc[-1]
            
            # لیست اندیکاتورها
            indicators_info = []
            
            # جمع‌آوری اطلاعات موجود
            if 'rsi' in df.columns:
                rsi_value = df['rsi'].iloc[-1]
                indicators_info.append(f"RSI: {rsi_value:.2f}")
            
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_value = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                indicators_info.append(f"MACD: {macd_value:.2f}, Signal: {macd_signal:.2f}")
            
            # ایجاد متن درخواست برای مدل زبانی
            prompt = f"""
            لطفاً تحلیل کاملی از شرایط فعلی ارز {symbol} در تایم‌فریم {timeframe} ارائه دهید.
            
            اطلاعات قیمت:
            - قیمت فعلی: {current_price:.2f}
            
            اندیکاتورها:
            {chr(10).join(indicators_info)}
            """
            
            # استفاده از تحلیل تصادفی
            return self.analyze_market_condition(symbol, timeframe)
            
        except Exception as e:
            return f"خطا در تحلیل اکشن قیمت: {str(e)}"