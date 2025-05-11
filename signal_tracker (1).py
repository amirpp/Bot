"""
ماژول ردیابی و ذخیره‌سازی سیگنال‌های معاملاتی

این ماژول شامل کلاس و توابع مورد نیاز برای ردیابی، ذخیره و بازیابی سیگنال‌های معاملاتی است.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

class SignalTracker:
    """کلاس ردیابی سیگنال‌های معاملاتی"""
    
    def __init__(self, file_path="signals.json"):
        """
        مقداردهی اولیه ردیاب سیگنال
        
        Args:
            file_path (str): مسیر فایل برای ذخیره‌سازی سیگنال‌ها
        """
        self.file_path = file_path
        self.signals = self._load_signals()
    
    def _load_signals(self):
        """
        بارگذاری سیگنال‌های ذخیره شده از فایل
        
        Returns:
            list: لیست سیگنال‌های بارگذاری شده
        """
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    signals = json.load(f)
                return signals
            else:
                return []
        except Exception as e:
            print(f"خطا در بارگذاری سیگنال‌ها: {str(e)}")
            return []
    
    def _save_signals(self):
        """ذخیره‌سازی سیگنال‌ها در فایل"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.signals, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"خطا در ذخیره‌سازی سیگنال‌ها: {str(e)}")
    
    def add_signal(self, symbol, timeframe, signal_type, confidence, description="", entry_price=None, price_targets=None):
        """
        افزودن سیگنال جدید
        
        Args:
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            signal_type (str): نوع سیگنال (BUY یا SELL)
            confidence (float): میزان اطمینان (0-100)
            description (str): توضیحات سیگنال
            entry_price (float): قیمت ورود
            price_targets (dict): اهداف قیمتی و حد ضرر {'tp1': float, 'tp2': float, 'tp3': float, 'tp4': float, 'sl': float}
            
        Returns:
            bool: آیا سیگنال با موفقیت اضافه شد؟
        """
        try:
            # ایجاد سیگنال جدید
            new_signal = {
                "id": len(self.signals) + 1,
                "symbol": symbol,
                "timeframe": timeframe,
                "signal_type": signal_type,
                "confidence": float(confidence),
                "description": description,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": entry_price
            }
            
            # اضافه کردن اهداف قیمتی اگر موجود باشند
            if price_targets:
                # فقط مقادیر مهم را ذخیره می‌کنیم
                targets = {}
                for key in ['tp1', 'tp2', 'tp3', 'tp4', 'sl']:
                    if key in price_targets:
                        targets[key] = price_targets[key]
                
                new_signal["price_targets"] = targets
            
            # افزودن به لیست سیگنال‌ها
            self.signals.append(new_signal)
            
            # ذخیره‌سازی
            self._save_signals()
            
            return True
        except Exception as e:
            print(f"خطا در افزودن سیگنال: {str(e)}")
            return False
    
    def update_signal(self, signal_id, result=None, exit_price=None, profit_loss=None, notes=None):
        """
        به‌روزرسانی سیگنال موجود
        
        Args:
            signal_id (int): شناسه سیگنال
            result (str): نتیجه سیگنال (profit, loss, neutral)
            exit_price (float): قیمت خروج
            profit_loss (float): میزان سود/ضرر (درصد)
            notes (str): یادداشت‌های اضافی
            
        Returns:
            bool: آیا سیگنال با موفقیت به‌روزرسانی شد؟
        """
        try:
            # یافتن سیگنال
            for i, signal in enumerate(self.signals):
                if signal["id"] == signal_id:
                    # به‌روزرسانی فیلدها
                    if result is not None:
                        self.signals[i]["result"] = result
                    
                    if exit_price is not None:
                        self.signals[i]["exit_price"] = exit_price
                    
                    if profit_loss is not None:
                        self.signals[i]["profit_loss"] = profit_loss
                    
                    if notes is not None:
                        self.signals[i]["notes"] = notes
                    
                    # ذخیره‌سازی
                    self._save_signals()
                    
                    return True
            
            # سیگنال یافت نشد
            return False
        except Exception as e:
            print(f"خطا در به‌روزرسانی سیگنال: {str(e)}")
            return False
    
    def get_signals(self, symbol=None, timeframe=None, signal_type=None, days=None):
        """
        دریافت سیگنال‌ها با امکان فیلتر کردن
        
        Args:
            symbol (str): فیلتر بر اساس نماد
            timeframe (str): فیلتر بر اساس تایم‌فریم
            signal_type (str): فیلتر بر اساس نوع سیگنال
            days (int): تعداد روزهای اخیر
            
        Returns:
            list: لیست سیگنال‌های فیلتر شده
        """
        try:
            filtered_signals = self.signals.copy()
            
            # فیلتر بر اساس نماد
            if symbol:
                filtered_signals = [s for s in filtered_signals if s.get("symbol") == symbol]
            
            # فیلتر بر اساس تایم‌فریم
            if timeframe:
                filtered_signals = [s for s in filtered_signals if s.get("timeframe") == timeframe]
            
            # فیلتر بر اساس نوع سیگنال
            if signal_type:
                filtered_signals = [s for s in filtered_signals if s.get("signal_type") == signal_type]
            
            # فیلتر بر اساس تاریخ
            if days:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                filtered_signals = [s for s in filtered_signals if s.get("timestamp", "").split()[0] >= cutoff_date]
            
            return filtered_signals
        except Exception as e:
            print(f"خطا در دریافت سیگنال‌ها: {str(e)}")
            return []
    
    def get_recent_signals(self, limit=10):
        """
        دریافت سیگنال‌های اخیر
        
        Args:
            limit (int): حداکثر تعداد سیگنال‌های برگشتی
            
        Returns:
            list: لیست سیگنال‌های اخیر
        """
        try:
            # مرتب‌سازی بر اساس زمان (جدیدترین اول)
            sorted_signals = sorted(
                self.signals, 
                key=lambda x: datetime.strptime(x.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )
            
            # محدود کردن به تعداد درخواست شده
            return sorted_signals[:limit]
        except Exception as e:
            print(f"خطا در دریافت سیگنال‌های اخیر: {str(e)}")
            return []
    
    def get_performance_stats(self, days=30):
        """
        دریافت آمار عملکرد سیگنال‌ها
        
        Args:
            days (int): تعداد روزهای اخیر برای محاسبه آمار
            
        Returns:
            dict: دیکشنری حاوی آمار عملکرد
        """
        try:
            # دریافت سیگنال‌های دوره زمانی مشخص
            signals = self.get_signals(days=days)
            
            # آمار پایه
            total_signals = len(signals)
            
            if total_signals == 0:
                return {
                    "total_signals": 0,
                    "profit_count": 0,
                    "loss_count": 0,
                    "neutral_count": 0,
                    "profit_rate": 0,
                    "avg_profit": 0,
                    "avg_loss": 0,
                    "buy_signals": 0,
                    "sell_signals": 0
                }
            
            # شمارش سیگنال‌ها بر اساس نتیجه
            profit_signals = [s for s in signals if s.get("result") == "profit"]
            loss_signals = [s for s in signals if s.get("result") == "loss"]
            neutral_signals = [s for s in signals if s.get("result") == "neutral"]
            
            # شمارش سیگنال‌ها بر اساس نوع
            buy_signals = [s for s in signals if s.get("signal_type") == "BUY"]
            sell_signals = [s for s in signals if s.get("signal_type") == "SELL"]
            
            # محاسبه نرخ سود
            profit_rate = len(profit_signals) / total_signals * 100 if total_signals > 0 else 0
            
            # میانگین سود و ضرر
            avg_profit = np.mean([s.get("profit_loss", 0) for s in profit_signals]) if profit_signals else 0
            avg_loss = np.mean([s.get("profit_loss", 0) for s in loss_signals]) if loss_signals else 0
            
            # بازگشت آمار
            return {
                "total_signals": total_signals,
                "profit_count": len(profit_signals),
                "loss_count": len(loss_signals),
                "neutral_count": len(neutral_signals),
                "profit_rate": profit_rate,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "buy_signals": len(buy_signals),
                "sell_signals": len(sell_signals)
            }
        except Exception as e:
            print(f"خطا در محاسبه آمار عملکرد: {str(e)}")
            return {}
    
    def clear_old_signals(self, days=90):
        """
        حذف سیگنال‌های قدیمی
        
        Args:
            days (int): سیگنال‌های قدیمی‌تر از این تعداد روز حذف می‌شوند
            
        Returns:
            int: تعداد سیگنال‌های حذف شده
        """
        try:
            # محاسبه تاریخ برش
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # شمارش سیگنال‌های قبلی
            old_count = len(self.signals)
            
            # فیلتر کردن سیگنال‌های جدیدتر از تاریخ برش
            self.signals = [s for s in self.signals if s.get("timestamp", "").split()[0] >= cutoff_date]
            
            # محاسبه تعداد سیگنال‌های حذف شده
            removed_count = old_count - len(self.signals)
            
            # ذخیره‌سازی
            if removed_count > 0:
                self._save_signals()
            
            return removed_count
        except Exception as e:
            print(f"خطا در حذف سیگنال‌های قدیمی: {str(e)}")
            return 0
    
    def export_to_csv(self, file_path="signals_export.csv"):
        """
        خروجی گرفتن سیگنال‌ها به فرمت CSV
        
        Args:
            file_path (str): مسیر فایل خروجی
            
        Returns:
            bool: آیا خروجی با موفقیت ایجاد شد؟
        """
        try:
            if not self.signals:
                return False
            
            # تبدیل به دیتافریم
            df = pd.DataFrame(self.signals)
            
            # ذخیره به فرمت CSV
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            return True
        except Exception as e:
            print(f"خطا در خروجی گرفتن سیگنال‌ها: {str(e)}")
            return False
    
    def get_signal_by_id(self, signal_id):
        """
        یافتن سیگنال با شناسه
        
        Args:
            signal_id (int): شناسه سیگنال
            
        Returns:
            dict: سیگنال یافت شده یا None
        """
        try:
            for signal in self.signals:
                if signal.get("id") == signal_id:
                    return signal
            return None
        except Exception as e:
            print(f"خطا در یافتن سیگنال: {str(e)}")
            return None
