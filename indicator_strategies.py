"""
ماژول مدیریت استراتژی‌های اندیکاتور

این ماژول شامل کلاس و توابع مورد نیاز برای مدیریت استراتژی‌های مختلف اندیکاتور تکنیکال است.
"""

import os
import json
import time
import streamlit as st
from datetime import datetime

class IndicatorStrategyManager:
    """کلاس مدیریت استراتژی‌های اندیکاتور"""
    
    def __init__(self, storage_dir="data"):
        """
        مقداردهی اولیه مدیر استراتژی‌های اندیکاتور
        
        Args:
            storage_dir (str): مسیر دایرکتوری ذخیره‌سازی
        """
        self.storage_dir = storage_dir
        self.strategies_file = os.path.join(storage_dir, "indicator_strategies.json")
        
        # ایجاد دایرکتوری ذخیره‌سازی اگر وجود ندارد
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        
        # بارگذاری استراتژی‌های موجود
        self.strategies = self._load_strategies()
    
    def _load_strategies(self):
        """
        بارگذاری استراتژی‌ها از فایل
        
        Returns:
            list: لیست استراتژی‌ها
        """
        if os.path.exists(self.strategies_file):
            try:
                with open(self.strategies_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _save_strategies(self):
        """
        ذخیره استراتژی‌ها در فایل
        
        Returns:
            bool: آیا ذخیره‌سازی موفقیت‌آمیز بوده است؟
        """
        try:
            with open(self.strategies_file, 'w') as f:
                json.dump(self.strategies, f, indent=2)
            return True
        except Exception as e:
            print(f"خطا در ذخیره استراتژی‌ها: {str(e)}")
            return False
    
    def save_strategy(self, name, indicators, description="", overwrite=False):
        """
        ذخیره یک استراتژی جدید
        
        Args:
            name (str): نام استراتژی
            indicators (list): لیست اندیکاتورها
            description (str, optional): توضیحات استراتژی
            overwrite (bool): آیا استراتژی موجود بازنویسی شود؟
            
        Returns:
            bool: آیا ذخیره‌سازی موفقیت‌آمیز بوده است؟
        """
        # بررسی پارامترهای ورودی
        if not name or not indicators:
            return False
        
        # بررسی تکراری بودن نام
        existing_strategy = self.get_strategy(name)
        if existing_strategy and not overwrite:
            st.warning(f"استراتژی با نام '{name}' قبلاً وجود دارد. لطفاً نام دیگری انتخاب کنید یا آن را بازنویسی کنید.")
            return False
        
        # ایجاد استراتژی جدید
        new_strategy = {
            "name": name,
            "indicators": indicators,
            "description": description,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # حذف استراتژی قبلی با همین نام (در صورت وجود)
        if existing_strategy:
            self.strategies = [s for s in self.strategies if s["name"] != name]
        
        # افزودن استراتژی جدید
        self.strategies.append(new_strategy)
        
        # ذخیره تغییرات
        return self._save_strategies()
    
    def get_strategy(self, name):
        """
        دریافت یک استراتژی با نام مشخص
        
        Args:
            name (str): نام استراتژی
            
        Returns:
            dict: استراتژی یا None در صورت عدم وجود
        """
        # بارگذاری مجدد استراتژی‌ها برای اطمینان از به‌روز بودن
        self.strategies = self._load_strategies()
        
        # جستجوی استراتژی
        for strategy in self.strategies:
            if strategy["name"] == name:
                return strategy
        
        return None
    
    def get_all_strategies(self):
        """
        دریافت تمام استراتژی‌های ذخیره شده
        
        Returns:
            list: لیست تمام استراتژی‌ها
        """
        # بارگذاری مجدد استراتژی‌ها برای اطمینان از به‌روز بودن
        self.strategies = self._load_strategies()
        
        # مرتب‌سازی بر اساس زمان به‌روزرسانی (جدیدترین اول)
        self.strategies.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return self.strategies
    
    def delete_strategy(self, name):
        """
        حذف یک استراتژی
        
        Args:
            name (str): نام استراتژی
            
        Returns:
            bool: آیا حذف موفقیت‌آمیز بوده است؟
        """
        # بررسی وجود استراتژی
        existing_strategy = self.get_strategy(name)
        if not existing_strategy:
            return False
        
        # حذف استراتژی
        self.strategies = [s for s in self.strategies if s["name"] != name]
        
        # ذخیره تغییرات
        return self._save_strategies()
    
    def update_strategy(self, name, new_name=None, indicators=None, description=None):
        """
        به‌روزرسانی یک استراتژی موجود
        
        Args:
            name (str): نام استراتژی
            new_name (str, optional): نام جدید استراتژی
            indicators (list, optional): لیست جدید اندیکاتورها
            description (str, optional): توضیحات جدید
            
        Returns:
            bool: آیا به‌روزرسانی موفقیت‌آمیز بوده است؟
        """
        # بررسی وجود استراتژی
        existing_strategy = self.get_strategy(name)
        if not existing_strategy:
            return False
        
        # به‌روزرسانی استراتژی
        updated_strategy = existing_strategy.copy()
        
        if new_name:
            updated_strategy["name"] = new_name
        
        if indicators is not None:
            updated_strategy["indicators"] = indicators
        
        if description is not None:
            updated_strategy["description"] = description
        
        updated_strategy["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # حذف استراتژی قدیمی
        self.strategies = [s for s in self.strategies if s["name"] != name]
        
        # افزودن استراتژی به‌روز شده
        self.strategies.append(updated_strategy)
        
        # ذخیره تغییرات
        return self._save_strategies()
    
    def export_strategies(self):
        """
        صدور تمام استراتژی‌ها به صورت JSON
        
        Returns:
            str: استراتژی‌ها به فرمت JSON
        """
        return json.dumps(self.strategies, indent=2)
    
    def import_strategies(self, json_data, overwrite=False):
        """
        ورود استراتژی‌ها از JSON
        
        Args:
            json_data (str): داده‌های JSON
            overwrite (bool): آیا استراتژی‌های موجود بازنویسی شوند؟
            
        Returns:
            bool: آیا ورود موفقیت‌آمیز بوده است؟
        """
        try:
            # تبدیل JSON به لیست
            imported_strategies = json.loads(json_data)
            
            if not isinstance(imported_strategies, list):
                return False
            
            # اعمال استراتژی‌ها
            for strategy in imported_strategies:
                # بررسی وجود فیلدهای ضروری
                if "name" not in strategy or "indicators" not in strategy:
                    continue
                
                # افزودن یا به‌روزرسانی استراتژی
                self.save_strategy(
                    name=strategy["name"],
                    indicators=strategy["indicators"],
                    description=strategy.get("description", ""),
                    overwrite=overwrite
                )
            
            return True
        
        except Exception as e:
            print(f"خطا در ورود استراتژی‌ها: {str(e)}")
            return False
    
    def get_recommended_strategies(self, market_condition=None):
        """
        پیشنهاد استراتژی‌های مناسب بر اساس شرایط بازار
        
        Args:
            market_condition (str, optional): شرایط بازار ('صعودی'، 'نزولی'، 'نوسانی' یا None)
            
        Returns:
            list: لیست استراتژی‌های پیشنهادی
        """
        # استراتژی‌های پیش‌فرض برای شرایط مختلف بازار
        default_strategies = {
            "صعودی": {
                "name": "استراتژی روند صعودی",
                "indicators": ["EMA", "MACD", "Supertrend", "OBV", "Volume"],
                "description": "استراتژی مناسب برای بازارهای صعودی با تمرکز بر روند و حجم"
            },
            "نزولی": {
                "name": "استراتژی روند نزولی",
                "indicators": ["RSI", "MACD", "Bollinger Bands", "ATR", "Volume"],
                "description": "استراتژی مناسب برای بازارهای نزولی با تمرکز بر اشباع فروش و نوسان"
            },
            "نوسانی": {
                "name": "استراتژی نوسانی",
                "indicators": ["RSI", "Stochastic", "Bollinger Bands", "Williams %R", "CCI"],
                "description": "استراتژی مناسب برای بازارهای نوسانی با تمرکز بر اسیلاتورها"
            },
            "ترکیبی": {
                "name": "استراتژی ترکیبی",
                "indicators": ["RSI", "MACD", "Bollinger Bands", "EMA", "Volume"],
                "description": "استراتژی عمومی با ترکیبی از اندیکاتورهای روند و نوسان"
            }
        }
        
        # دریافت تمام استراتژی‌های ذخیره شده
        all_strategies = self.get_all_strategies()
        
        # اگر شرایط بازار مشخص شده باشد
        if market_condition and market_condition in default_strategies:
            # بررسی آیا استراتژی پیش‌فرض مربوطه در استراتژی‌های ذخیره شده وجود دارد
            for strategy in all_strategies:
                if strategy["name"] == default_strategies[market_condition]["name"]:
                    return [strategy]
            
            # اگر وجود نداشت، استراتژی پیش‌فرض را برگردان
            return [default_strategies[market_condition]]
        
        # اگر شرایط بازار مشخص نشده باشد، تمام استراتژی‌ها را برگردان
        if not all_strategies:
            # اگر هیچ استراتژی ذخیره شده‌ای وجود ندارد، استراتژی‌های پیش‌فرض را برگردان
            return list(default_strategies.values())
        
        return all_strategies

def create_default_strategies():
    """
    ایجاد استراتژی‌های پیش‌فرض در صورت نبود استراتژی
    
    Returns:
        IndicatorStrategyManager: مدیر استراتژی با استراتژی‌های پیش‌فرض
    """
    # ایجاد مدیر استراتژی
    manager = IndicatorStrategyManager()
    
    # بررسی وجود استراتژی‌های قبلی
    if not manager.get_all_strategies():
        # ایجاد استراتژی‌های پیش‌فرض
        manager.save_strategy(
            name="استراتژی روند صعودی",
            indicators=["EMA", "MACD", "Supertrend", "OBV", "Volume"],
            description="استراتژی مناسب برای بازارهای صعودی با تمرکز بر روند و حجم"
        )
        
        manager.save_strategy(
            name="استراتژی روند نزولی",
            indicators=["RSI", "MACD", "Bollinger Bands", "ATR", "Volume"],
            description="استراتژی مناسب برای بازارهای نزولی با تمرکز بر اشباع فروش و نوسان"
        )
        
        manager.save_strategy(
            name="استراتژی نوسانی",
            indicators=["RSI", "Stochastic", "Bollinger Bands", "Williams %R", "CCI"],
            description="استراتژی مناسب برای بازارهای نوسانی با تمرکز بر اسیلاتورها"
        )
        
        manager.save_strategy(
            name="استراتژی ترکیبی",
            indicators=["RSI", "MACD", "Bollinger Bands", "EMA", "Volume"],
            description="استراتژی عمومی با ترکیبی از اندیکاتورهای روند و نوسان"
        )
    
    return manager
