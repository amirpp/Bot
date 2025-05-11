"""
ماژول تحلیل احساسات بازار ارزهای دیجیتال

این ماژول شامل توابع جمع‌آوری و تحلیل داده‌های احساسی از منابع مختلف مانند توییتر، ردیت، 
تلگرام و سایر منابع خبری است که می‌تواند به عنوان شاخص‌های مکمل در کنار تحلیل تکنیکال استفاده شود.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# برای حالت آفلاین، داده‌های شبیه‌سازی شده با نوسانات واقع‌گرایانه ایجاد می‌کنیم
# در حالت آنلاین، می‌توان از API‌های واقعی مانند Twitter API، Reddit API یا Crypto Fear & Greed Index استفاده کرد

class SentimentAnalyzer:
    """کلاس تحلیل احساسات بازار"""
    
    def __init__(self, symbol, timeframe="1d", lookback_days=30):
        """
        مقداردهی اولیه تحلیلگر احساسات
        
        Args:
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم تحلیل
            lookback_days (int): تعداد روزهای مورد بررسی
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.currency = symbol.split('/')[0] if '/' in symbol else symbol
        
        # تنظیم منابع برای تحلیل احساسات
        self.sources = ['twitter', 'reddit', 'telegram', 'news', 'trading_platforms']
        
        # تنظیم وزن منابع مختلف
        self.source_weights = {
            'twitter': 0.35,
            'reddit': 0.25,
            'telegram': 0.20,
            'news': 0.15,
            'trading_platforms': 0.05
        }
    
    def fetch_sentiment_data(self):
        """
        دریافت داده‌های احساسی از منابع مختلف
        
        Returns:
            pd.DataFrame: دیتافریم با داده‌های احساسی
        """
        # در حالت واقعی، اینجا از API های مختلف استفاده می‌شود
        # برای مثال: Twitter API, Reddit API, تلگرام و...
        # در حالت آفلاین، داده‌های شبیه‌سازی شده ایجاد می‌کنیم
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # ایجاد تاریخ‌ها بر اساس تایم‌فریم
        if self.timeframe == '1h':
            dates = [start_date + timedelta(hours=i) for i in range(self.lookback_days * 24)]
        elif self.timeframe == '4h':
            dates = [start_date + timedelta(hours=i*4) for i in range(self.lookback_days * 6)]
        else:  # روزانه
            dates = [start_date + timedelta(days=i) for i in range(self.lookback_days)]
        
        # ساخت دیتاست احساسات پایه با تاریخ
        sentiment_data = pd.DataFrame({'timestamp': dates})
        sentiment_data.set_index('timestamp', inplace=True)
        
        # ایجاد داده‌های احساسی واقع‌گرایانه برای هر منبع
        for source in self.sources:
            # سازگار کردن روند احساسات با ارز مورد نظر
            if self.currency.upper() == 'BTC':
                base_sentiment = np.random.normal(0.65, 0.2, len(dates))  # روند صعودی با نوسان
            elif self.currency.upper() == 'ETH':
                base_sentiment = np.random.normal(0.60, 0.25, len(dates))  # روند صعودی با نوسان بیشتر
            elif self.currency.upper() in ['SOL', 'MATIC', 'AVAX']:
                base_sentiment = np.random.normal(0.55, 0.3, len(dates))  # ارزهای آلت‌کوین با نوسان بیشتر
            else:
                base_sentiment = np.random.normal(0.5, 0.3, len(dates))  # ارزهای کوچکتر با نوسان زیاد
            
            # اضافه کردن روند‌ها و الگوهای بازار به داده‌ها
            trend = np.sin(np.linspace(0, 5, len(dates))) * 0.1  # اضافه کردن الگوی سینوسی
            
            # اضافه کردن نویز متفاوت برای هر منبع
            if source == 'twitter':
                noise = np.random.normal(0, 0.15, len(dates))
            elif source == 'reddit':
                noise = np.random.normal(0, 0.2, len(dates))
            elif source == 'telegram':
                noise = np.random.normal(0, 0.25, len(dates))  # تلگرام نوسان بیشتری دارد
            else:
                noise = np.random.normal(0, 0.1, len(dates))
            
            # ترکیب داده‌های پایه، روند و نویز
            sentiment_scores = base_sentiment + trend + noise
            
            # نرمالایز کردن بین 0 و 1
            sentiment_scores = np.clip(sentiment_scores, 0, 1)
            
            # افزودن به دیتاست
            sentiment_data[f'{source}_sentiment'] = sentiment_scores
            sentiment_data[f'{source}_volume'] = np.random.randint(1000, 50000, len(dates)) * \
                                               (1 + abs(sentiment_scores - 0.5))  # حجم بیشتر برای احساسات شدیدتر
        
        return sentiment_data
    
    def calculate_combined_sentiment(self, sentiment_data=None):
        """
        محاسبه احساسات ترکیبی از تمام منابع
        
        Args:
            sentiment_data (pd.DataFrame): دیتافریم داده‌های احساسی
            
        Returns:
            pd.DataFrame: دیتافریم با احساسات ترکیبی
        """
        if sentiment_data is None:
            sentiment_data = self.fetch_sentiment_data()
        
        # محاسبه احساسات ترکیبی با استفاده از وزن‌های منابع
        weighted_sentiments = []
        
        for source in self.sources:
            weighted_sentiments.append(sentiment_data[f'{source}_sentiment'] * self.source_weights[source])
        
        sentiment_data['combined_sentiment'] = sum(weighted_sentiments)
        
        # محاسبه حجم ترکیبی (وزن‌دار)
        weighted_volumes = []
        
        for source in self.sources:
            weighted_volumes.append(sentiment_data[f'{source}_volume'] * self.source_weights[source])
        
        sentiment_data['combined_volume'] = sum(weighted_volumes)
        
        # محاسبه شاخص‌های احساسی
        sentiment_data['bullish_ratio'] = self._calculate_bullish_ratio(sentiment_data)
        sentiment_data['fear_greed_index'] = self._calculate_fear_greed_index(sentiment_data)
        sentiment_data['social_dominance'] = self._calculate_social_dominance(sentiment_data)
        
        return sentiment_data
    
    def _calculate_bullish_ratio(self, sentiment_data):
        """محاسبه نسبت دیدگاه‌های صعودی به کل دیدگاه‌ها"""
        # نسبت‌های بالاتر از 0.5 به عنوان دیدگاه صعودی در نظر گرفته می‌شوند
        return np.where(sentiment_data['combined_sentiment'] > 0.5, 
                        (sentiment_data['combined_sentiment'] - 0.5) * 2, 0)
    
    def _calculate_fear_greed_index(self, sentiment_data):
        """محاسبه شاخص ترس و طمع"""
        # تبدیل احساسات به شاخص 0-100
        return sentiment_data['combined_sentiment'] * 100
    
    def _calculate_social_dominance(self, sentiment_data):
        """محاسبه غالبیت اجتماعی ارز"""
        # شبیه‌سازی غالبیت اجتماعی بر اساس احساسات و حجم
        dominance = sentiment_data['combined_sentiment'] * np.log1p(sentiment_data['combined_volume'] / 1000)
        
        # نرمالایز بین 0 و 1
        return (dominance - dominance.min()) / (dominance.max() - dominance.min() + 1e-8)
    
    def get_recent_sentiment_metrics(self):
        """
        استخراج معیارهای احساسی اخیر
        
        Returns:
            dict: دیکشنری شاخص‌های احساسی
        """
        sentiment_data = self.calculate_combined_sentiment()
        
        # استخراج داده‌های اخیر
        recent_data = sentiment_data.iloc[-1]
        
        # محاسبه روند احساسات (مثبت یا منفی بودن شیب)
        sentiment_trend = sentiment_data['combined_sentiment'].diff().iloc[-3:].mean()
        sentiment_trend_str = "صعودی" if sentiment_trend > 0.01 else "نزولی" if sentiment_trend < -0.01 else "خنثی"
        
        # برچسب‌گذاری احساسات
        sentiment_score = recent_data['combined_sentiment']
        if sentiment_score > 0.75:
            sentiment_label = "خیلی صعودی"
        elif sentiment_score > 0.6:
            sentiment_label = "صعودی"
        elif sentiment_score > 0.4:
            sentiment_label = "خنثی"
        elif sentiment_score > 0.25:
            sentiment_label = "نزولی"
        else:
            sentiment_label = "خیلی نزولی"
        
        # برچسب‌گذاری ترس و طمع
        fear_greed = recent_data['fear_greed_index']
        if fear_greed > 75:
            fear_greed_label = "طمع شدید"
        elif fear_greed > 60:
            fear_greed_label = "طمع"
        elif fear_greed > 40:
            fear_greed_label = "خنثی"
        elif fear_greed > 25:
            fear_greed_label = "ترس"
        else:
            fear_greed_label = "ترس شدید"
        
        # دیکشنری نتایج
        results = {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "sentiment_trend": sentiment_trend_str,
            "fear_greed_index": fear_greed,
            "fear_greed_label": fear_greed_label,
            "bullish_ratio": recent_data['bullish_ratio'],
            "social_dominance": recent_data['social_dominance'],
            "twitter_sentiment": recent_data['twitter_sentiment'],
            "reddit_sentiment": recent_data['reddit_sentiment'],
            "telegram_sentiment": recent_data['telegram_sentiment'],
            "news_sentiment": recent_data['news_sentiment'],
            "platforms_sentiment": recent_data['trading_platforms_sentiment'],
            "combined_volume": recent_data['combined_volume']
        }
        
        return results
    
    def generate_sentiment_insights(self):
        """
        تولید بینش‌های احساسی بازار
        
        Returns:
            str: متن تحلیل احساسات
        """
        metrics = self.get_recent_sentiment_metrics()
        sentiment_data = self.calculate_combined_sentiment()
        
        insights = f"**تحلیل احساسات بازار برای {self.currency}:**\n\n"
        
        # بخش اصلی بینش‌ها
        insights += f"**وضعیت کلی بازار:** {metrics['sentiment_label']}\n"
        insights += f"**شاخص ترس و طمع:** {metrics['fear_greed_label']} ({metrics['fear_greed_index']:.1f}/100)\n"
        insights += f"**روند احساسات:** {metrics['sentiment_trend']}\n"
        insights += f"**نسبت دیدگاه‌های صعودی:** {metrics['bullish_ratio']:.2f}\n\n"
        
        # بینش‌های پیشرفته
        insights += "**تحلیل منابع:**\n"
        
        twitter_sentiment = metrics['twitter_sentiment']
        reddit_sentiment = metrics['reddit_sentiment']
        telegram_sentiment = metrics['telegram_sentiment']
        
        insights += f"- توییتر: {self._sentiment_to_text(twitter_sentiment)}\n"
        insights += f"- ردیت: {self._sentiment_to_text(reddit_sentiment)}\n"
        insights += f"- تلگرام: {self._sentiment_to_text(telegram_sentiment)}\n"
        insights += f"- اخبار: {self._sentiment_to_text(metrics['news_sentiment'])}\n"
        
        # بررسی خاص تناقض بین منابع
        if max(twitter_sentiment, reddit_sentiment, telegram_sentiment) - min(twitter_sentiment, reddit_sentiment, telegram_sentiment) > 0.3:
            insights += "\n**هشدار:** تناقض قابل‌توجهی بین منابع مختلف وجود دارد! این می‌تواند نشانه عدم قطعیت در بازار باشد.\n"
        
        # بررسی رابطه با روند قیمت
        if metrics['sentiment_label'] in ["خیلی صعودی", "صعودی"] and metrics['sentiment_trend'] == "نزولی":
            insights += "\n**توجه:** احساسات بازار صعودی است اما در حال کاهش می‌باشد. این می‌تواند نشانه‌ای از تغییر روند باشد.\n"
        elif metrics['sentiment_label'] in ["خیلی نزولی", "نزولی"] and metrics['sentiment_trend'] == "صعودی":
            insights += "\n**توجه:** احساسات بازار نزولی است اما در حال بهبود می‌باشد. این می‌تواند نشانه‌ای از تغییر روند باشد.\n"
        
        # توصیه‌های عملی بر اساس تحلیل احساسات
        insights += "\n**توصیه‌های معاملاتی بر اساس تحلیل احساسات:**\n"
        
        if metrics['sentiment_label'] == "خیلی صعودی" and metrics['fear_greed_label'] == "طمع شدید":
            insights += "- بازار در وضعیت طمع شدید قرار دارد. احتیاط کنید و از ورود به معاملات پرریسک خودداری نمایید.\n"
            insights += "- روند صعودی قوی است اما ممکن است به زودی تصحیح قیمت رخ دهد.\n"
            insights += "- بخشی از سود خود را برداشت کنید و حد ضرر را به دقت تنظیم نمایید.\n"
        elif metrics['sentiment_label'] == "خیلی نزولی" and metrics['fear_greed_label'] == "ترس شدید":
            insights += "- بازار در وضعیت ترس شدید قرار دارد. این می‌تواند فرصت خرید باشد، اما با احتیاط.\n"
            insights += "- ورود تدریجی به بازار را در نظر بگیرید.\n"
            insights += "- منتظر نشانه‌های تثبیت قیمت باشید.\n"
        elif metrics['sentiment_label'] in ["صعودی", "خیلی صعودی"]:
            insights += "- روند احساسات صعودی است. معاملات خرید با حد ضرر مناسب توصیه می‌شود.\n"
            insights += "- به شاخص ترس و طمع توجه کنید و در صورت ورود به منطقه طمع، احتیاط بیشتری داشته باشید.\n"
        elif metrics['sentiment_label'] in ["نزولی", "خیلی نزولی"]:
            insights += "- روند احساسات نزولی است. معاملات فروش با حد ضرر مناسب توصیه می‌شود.\n"
            insights += "- به شاخص ترس و طمع توجه کنید و در صورت ورود به منطقه ترس، احتیاط بیشتری داشته باشید.\n"
        else:
            insights += "- بازار در وضعیت خنثی قرار دارد. صبر کنید تا سیگنال واضح‌تری ایجاد شود.\n"
            insights += "- معاملات کوتاه‌مدت در محدوده فعلی قیمت می‌تواند سودآور باشد.\n"
        
        return insights
    
    def _sentiment_to_text(self, sentiment_score):
        """تبدیل نمره احساسات به متن توصیفی"""
        if sentiment_score > 0.75:
            return "بسیار مثبت"
        elif sentiment_score > 0.6:
            return "مثبت"
        elif sentiment_score > 0.4:
            return "خنثی"
        elif sentiment_score > 0.25:
            return "منفی"
        else:
            return "بسیار منفی"
    
    def plot_sentiment_data(self):
        """
        رسم نمودار داده‌های احساسی
        
        Returns:
            plt.Figure: نمودار احساسات
        """
        sentiment_data = self.calculate_combined_sentiment()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # نمودار احساسات ترکیبی
        ax1.plot(sentiment_data.index, sentiment_data['combined_sentiment'], 'b-', linewidth=2)
        ax1.set_title(f'تحلیل احساسات {self.currency}')
        ax1.set_ylabel('احساسات ترکیبی')
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        ax1.fill_between(sentiment_data.index, sentiment_data['combined_sentiment'], 0.5, 
                        where=(sentiment_data['combined_sentiment'] >= 0.5), 
                        color='green', alpha=0.3, interpolate=True)
        ax1.fill_between(sentiment_data.index, sentiment_data['combined_sentiment'], 0.5, 
                        where=(sentiment_data['combined_sentiment'] <= 0.5), 
                        color='red', alpha=0.3, interpolate=True)
        ax1.grid(True, alpha=0.3)
        
        # نمودار شاخص ترس و طمع
        ax2.plot(sentiment_data.index, sentiment_data['fear_greed_index'], 'g-', linewidth=2)
        ax2.set_ylabel('شاخص ترس و طمع')
        ax2.axhline(y=50, color='r', linestyle='--', alpha=0.3)
        ax2.axhline(y=75, color='r', linestyle='--', alpha=0.3)
        ax2.axhline(y=25, color='r', linestyle='--', alpha=0.3)
        ax2.fill_between(sentiment_data.index, sentiment_data['fear_greed_index'], 75, 
                        where=(sentiment_data['fear_greed_index'] >= 75), 
                        color='red', alpha=0.3, interpolate=True, label='طمع شدید')
        ax2.fill_between(sentiment_data.index, sentiment_data['fear_greed_index'], 25, 
                        where=(sentiment_data['fear_greed_index'] <= 25), 
                        color='red', alpha=0.3, interpolate=True, label='ترس شدید')
        ax2.grid(True, alpha=0.3)
        
        # نمودار احساسات منابع مختلف
        for source in self.sources:
            ax3.plot(sentiment_data.index, sentiment_data[f'{source}_sentiment'], 
                    label=source, alpha=0.7)
        ax3.set_ylabel('احساسات منابع')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def get_market_sentiment(symbol, timeframe="1d", lookback_days=30):
    """
    دریافت تحلیل احساسات بازار برای یک ارز دیجیتال
    
    Args:
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم تحلیل
        lookback_days (int): تعداد روزهای مورد بررسی
        
    Returns:
        tuple: (معیارهای احساسی، بینش‌ها، نمودار)
    """
    analyzer = SentimentAnalyzer(symbol, timeframe, lookback_days)
    metrics = analyzer.get_recent_sentiment_metrics()
    insights = analyzer.generate_sentiment_insights()
    chart = analyzer.plot_sentiment_data()
    
    return metrics, insights, chart


def get_sentiment_signal(sentiment_metrics, price_action=None):
    """
    تولید سیگنال معاملاتی بر اساس تحلیل احساسات
    
    Args:
        sentiment_metrics (dict): معیارهای احساسی
        price_action (dict): داده‌های اکشن قیمت (اختیاری)
        
    Returns:
        tuple: (نوع سیگنال، قدرت سیگنال، توضیحات)
    """
    sentiment_score = sentiment_metrics['sentiment_score']
    fear_greed = sentiment_metrics['fear_greed_index']
    
    signal_type = "NEUTRAL"
    signal_strength = 0
    reasons = []
    
    # تعیین سیگنال بر اساس احساسات
    if sentiment_score > 0.7:
        signal_type = "BUY"
        signal_strength = min(100, sentiment_score * 100)
        reasons.append(f"احساسات بازار بسیار مثبت است ({sentiment_score:.2f}/1)")
    elif sentiment_score > 0.6:
        signal_type = "BUY"
        signal_strength = min(80, sentiment_score * 80)
        reasons.append(f"احساسات بازار مثبت است ({sentiment_score:.2f}/1)")
    elif sentiment_score < 0.3:
        signal_type = "SELL"
        signal_strength = min(100, (1 - sentiment_score) * 100)
        reasons.append(f"احساسات بازار بسیار منفی است ({sentiment_score:.2f}/1)")
    elif sentiment_score < 0.4:
        signal_type = "SELL"
        signal_strength = min(80, (1 - sentiment_score) * 80)
        reasons.append(f"احساسات بازار منفی است ({sentiment_score:.2f}/1)")
    
    # تنظیم قدرت سیگنال بر اساس شاخص ترس و طمع
    if fear_greed > 80 and signal_type == "BUY":
        signal_strength = max(50, signal_strength - 20)  # کاهش قدرت سیگنال خرید در طمع شدید
        reasons.append(f"هشدار: شاخص طمع بسیار بالاست ({fear_greed:.1f}/100)")
    elif fear_greed < 20 and signal_type == "SELL":
        signal_strength = max(50, signal_strength - 20)  # کاهش قدرت سیگنال فروش در ترس شدید
        reasons.append(f"هشدار: شاخص ترس بسیار بالاست ({fear_greed:.1f}/100)")
    
    # بررسی تناقض بین منابع
    twitter_sentiment = sentiment_metrics['twitter_sentiment']
    reddit_sentiment = sentiment_metrics['reddit_sentiment']
    telegram_sentiment = sentiment_metrics['telegram_sentiment']
    
    sentiment_variance = max(twitter_sentiment, reddit_sentiment, telegram_sentiment) - min(twitter_sentiment, reddit_sentiment, telegram_sentiment)
    if sentiment_variance > 0.4:
        signal_strength = max(40, signal_strength - 30)  # کاهش قدرت سیگنال در صورت تناقض
        reasons.append(f"تناقض زیادی بین منابع مختلف وجود دارد (واریانس: {sentiment_variance:.2f})")
    
    # ترکیب با داده‌های اکشن قیمت (در صورت وجود)
    if price_action:
        price_trend = price_action.get('trend', 'neutral')
        
        if price_trend == "bullish" and signal_type == "BUY":
            signal_strength = min(100, signal_strength + 10)  # تقویت سیگنال خرید در روند صعودی قیمت
            reasons.append("روند قیمت صعودی است که با سیگنال خرید همسو می‌باشد")
        elif price_trend == "bearish" and signal_type == "SELL":
            signal_strength = min(100, signal_strength + 10)  # تقویت سیگنال فروش در روند نزولی قیمت
            reasons.append("روند قیمت نزولی است که با سیگنال فروش همسو می‌باشد")
        elif price_trend == "bullish" and signal_type == "SELL":
            signal_strength = max(30, signal_strength - 40)  # کاهش قدرت سیگنال فروش در روند صعودی قیمت
            reasons.append("هشدار: روند قیمت صعودی است اما احساسات بازار منفی می‌باشد")
        elif price_trend == "bearish" and signal_type == "BUY":
            signal_strength = max(30, signal_strength - 40)  # کاهش قدرت سیگنال خرید در روند نزولی قیمت
            reasons.append("هشدار: روند قیمت نزولی است اما احساسات بازار مثبت می‌باشد")
    
    # توضیحات سیگنال
    signal_description = f"**تحلیل احساسات بازار:**\n\n"
    
    for reason in reasons:
        signal_description += f"- {reason}\n"
    
    signal_description += f"\n**منابع احساسی:**\n"
    signal_description += f"- توییتر: {sentiment_metrics['twitter_sentiment']:.2f}\n"
    signal_description += f"- ردیت: {sentiment_metrics['reddit_sentiment']:.2f}\n"
    signal_description += f"- تلگرام: {sentiment_metrics['telegram_sentiment']:.2f}\n"
    signal_description += f"- اخبار: {sentiment_metrics['news_sentiment']:.2f}\n"
    
    if sentiment_metrics['fear_greed_label']:
        signal_description += f"\n**شاخص ترس و طمع:** {sentiment_metrics['fear_greed_label']} ({sentiment_metrics['fear_greed_index']:.1f}/100)\n"
    
    return signal_type, round(signal_strength), signal_description