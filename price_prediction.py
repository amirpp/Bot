"""
ماژول پیش‌بینی قیمت ارزهای دیجیتال به صورت آفلاین

این ماژول شامل توابع و کلاس‌های مورد نیاز برای پیش‌بینی قیمت ارزها با استفاده از مدل‌های مختلف یادگیری ماشین است.
"""

import pandas as pd
import numpy as np
import datetime
import streamlit as st
import pickle
import base64
import io
from datetime import timedelta
import time
import random

# برای مدل‌های پیش‌بینی
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PricePredictor:
    """کلاس پیش‌بینی‌کننده قیمت ارزهای دیجیتال"""
    
    def __init__(self, symbol=None):
        """
        مقداردهی اولیه کلاس پیش‌بینی‌کننده
        
        Args:
            symbol (str, optional): نماد ارز
        """
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = 30  # تعداد روزهای گذشته برای پیش‌بینی
        self.forecast_days = 7  # تعداد روزهای پیش‌بینی
        self.features = []  # ویژگی‌های مورد استفاده
        self.last_training_time = None
        
        # تنظیمات مدل
        self.model_settings = {
            'linear_regression': {
                'name': 'رگرسیون خطی',
                'params': {}
            },
            'random_forest': {
                'name': 'جنگل تصادفی',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 20,
                    'random_state': 42
                }
            }
        }
    
    def prepare_data(self, df, target_column='close', add_indicators=True):
        """
        آماده‌سازی داده‌ها برای آموزش مدل
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            target_column (str): ستون هدف برای پیش‌بینی
            add_indicators (bool): اضافه کردن اندیکاتورها به ویژگی‌ها
            
        Returns:
            tuple: داده‌های آموزش و آزمون (X_train, X_test, y_train, y_test)
        """
        if df is None or df.empty or len(df) < self.lookback + 10:
            st.error("داده‌های ناکافی برای آموزش مدل پیش‌بینی")
            return None, None, None, None
        
        # کپی دیتافریم برای جلوگیری از تغییر در داده‌های اصلی
        data = df.copy()
        
        # حذف ستون‌های تاریخ و نامرتبط
        if isinstance(data.index, pd.DatetimeIndex):
            # اضافه کردن ویژگی‌های زمانی
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['month'] = data.index.month
        
        # اضافه کردن ویژگی‌های تکنیکال
        available_features = ['open', 'high', 'low', 'close', 'volume']
        
        # اضافه کردن اندیکاتورها به ویژگی‌ها
        if add_indicators:
            # بررسی اندیکاتورهای موجود
            indicator_cols = ['rsi', 'macd', 'ema', 'sma', 'bb_width', 'atr', 'obv', 'adx']
            
            for col in indicator_cols:
                if col in data.columns:
                    available_features.append(col)
        
        # حذف ستون‌های با مقادیر NaN
        for col in available_features:
            if col in data.columns and data[col].isnull().any():
                data[col] = data[col].fillna(method='ffill')
        
        # ساخت ویژگی‌های تاخیری
        for feature in available_features:
            if feature in data.columns:
                # افزودن تاخیرهای مختلف برای هر ویژگی
                for lag in [1, 3, 7, 14]:
                    if len(data) > lag:
                        data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
        
        # افزودن تغییرات درصدی
        for feature in ['close', 'volume']:
            if feature in data.columns:
                data[f'{feature}_pct_change'] = data[feature].pct_change() * 100
                data[f'{feature}_pct_change_3d'] = data[feature].pct_change(3) * 100
                data[f'{feature}_pct_change_7d'] = data[feature].pct_change(7) * 100
        
        # حذف ردیف‌های با مقادیر NaN
        data = data.dropna()
        
        if len(data) < self.lookback + 10:
            st.error("پس از پردازش، داده‌های کافی برای آموزش باقی نمانده است")
            return None, None, None, None
        
        # ذخیره لیست ویژگی‌ها
        self.features = [col for col in data.columns if col != target_column]
        
        # تقسیم به داده‌های آموزش و آزمون
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # نرمال‌سازی داده‌ها
        X_train = self.scaler.fit_transform(train_data[self.features])
        y_train = train_data[target_column].values
        
        X_test = self.scaler.transform(test_data[self.features])
        y_test = test_data[target_column].values
        
        return X_train, X_test, y_train, y_test, data
    
    def train_model(self, df, model_type='random_forest', target_column='close'):
        """
        آموزش مدل پیش‌بینی قیمت
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            model_type (str): نوع مدل ('linear_regression' یا 'random_forest')
            target_column (str): ستون هدف برای پیش‌بینی
            
        Returns:
            bool: آیا آموزش موفقیت‌آمیز بوده است؟
        """
        # آماده‌سازی داده‌ها
        X_train, X_test, y_train, y_test, processed_data = self.prepare_data(df, target_column)
        
        if X_train is None or len(X_train) == 0:
            return False
        
        try:
            # ساخت مدل
            if model_type == 'linear_regression':
                self.model = LinearRegression(**self.model_settings['linear_regression']['params'])
            elif model_type == 'random_forest':
                self.model = RandomForestRegressor(**self.model_settings['random_forest']['params'])
            else:
                st.error(f"نوع مدل {model_type} نامعتبر است")
                return False
            
            # آموزش مدل
            self.model.fit(X_train, y_train)
            
            # ارزیابی مدل
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # نمایش متریک‌های ارزیابی
            st.write(f"### نتایج ارزیابی مدل ({self.model_settings[model_type]['name']})")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("MSE", f"{mse:.2f}")
            metrics_col2.metric("MAE", f"{mae:.2f}")
            metrics_col3.metric("R²", f"{r2:.4f}")
            
            # ثبت زمان آموزش
            self.last_training_time = datetime.datetime.now()
            
            return True
        
        except Exception as e:
            st.error(f"خطا در آموزش مدل: {str(e)}")
            return False
    
    def predict(self, df, days=7, with_confidence=True):
        """
        پیش‌بینی قیمت آینده
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            days (int): تعداد روزهای پیش‌بینی
            with_confidence (bool): محاسبه حدود اطمینان
            
        Returns:
            dict: نتایج پیش‌بینی
        """
        if self.model is None:
            # اگر مدل آموزش ندیده است، از یک روش ساده استفاده می‌کنیم
            return self._simple_forecast(df, days)
        
        try:
            # کپی دیتافریم برای جلوگیری از تغییر در داده‌های اصلی
            data = df.copy()
            
            # آماده‌سازی داده‌ها برای پیش‌بینی
            # اضافه کردن ویژگی‌های زمانی
            if isinstance(data.index, pd.DatetimeIndex):
                data['day_of_week'] = data.index.dayofweek
                data['day_of_month'] = data.index.day
                data['month'] = data.index.month
            
            # بررسی ویژگی‌های موردنیاز
            required_features = self.features
            
            # پر کردن ویژگی‌های موردنیاز اگر وجود ندارند
            for feature in required_features:
                if feature not in data.columns:
                    # تنظیم مقدار پیش‌فرض برای ویژگی‌های غیرموجود
                    if 'lag' in feature:
                        # برای ویژگی‌های تاخیری، استفاده از ستون اصلی
                        base_col = feature.split('_lag_')[0]
                        lag = int(feature.split('_lag_')[1])
                        
                        if base_col in data.columns and len(data) > lag:
                            data[feature] = data[base_col].shift(lag)
                        else:
                            data[feature] = 0
                    elif 'pct_change' in feature:
                        # برای تغییرات درصدی
                        data[feature] = 0
                    else:
                        data[feature] = 0
            
            # حذف ردیف‌های با مقادیر NaN
            data = data.fillna(method='ffill')
            
            # گرفتن آخرین داده‌ها برای پیش‌بینی
            last_data = data.iloc[-1:][required_features]
            
            # نرمال‌سازی داده‌ها
            X_pred = self.scaler.transform(last_data)
            
            # پیش‌بینی روزانه
            forecasted_values = []
            upper_bound = []
            lower_bound = []
            
            # قیمت آخرین روز
            last_price = data['close'].iloc[-1]
            
            for day in range(days):
                # پیش‌بینی برای روز فعلی
                pred = self.model.predict(X_pred)[0]
                forecasted_values.append(pred)
                
                # محاسبه حدود اطمینان
                if with_confidence:
                    if hasattr(self.model, 'estimators_'):
                        # برای جنگل تصادفی، محاسبه انحراف معیار پیش‌بینی‌های مختلف
                        predictions = [estimator.predict(X_pred)[0] for estimator in self.model.estimators_]
                        std = np.std(predictions)
                        upper = pred + 1.96 * std
                        lower = pred - 1.96 * std
                    else:
                        # برای سایر مدل‌ها، استفاده از انحراف استاندارد ثابت
                        std = last_price * 0.02 * (day + 1) / days  # افزایش عدم قطعیت با گذشت زمان
                        upper = pred + std
                        lower = pred - std
                        
                    upper_bound.append(upper)
                    lower_bound.append(lower)
                    
                # به‌روزرسانی داده‌ها برای پیش‌بینی روز بعد
                # این قسمت نیاز به بهبود دارد
                # فعلاً فقط قیمت پیش‌بینی شده را جایگزین می‌کنیم
                if day < days - 1:
                    X_pred_updated = X_pred.copy()
                    
                    # یافتن ستون close در ویژگی‌ها
                    if 'close' in required_features:
                        close_idx = required_features.index('close')
                        X_pred_updated[0, close_idx] = pred / last_price  # نرمال‌سازی شده
                        
                    # به‌روزرسانی لگ‌ها
                    for i, feature in enumerate(required_features):
                        if 'lag_1' in feature:
                            base_feature = feature.replace('_lag_1', '')
                            if base_feature in required_features:
                                base_idx = required_features.index(base_feature)
                                X_pred_updated[0, i] = X_pred[0, base_idx]
                    
                    X_pred = X_pred_updated
            
            result = {'forecasted_values': forecasted_values}
            
            if with_confidence:
                result['upper_bound'] = upper_bound
                result['lower_bound'] = lower_bound
            
            return result
        
        except Exception as e:
            st.error(f"خطا در پیش‌بینی قیمت: {str(e)}")
            return self._simple_forecast(df, days)
    
    def _simple_forecast(self, df, days=7):
        """
        پیش‌بینی ساده قیمت بدون استفاده از مدل یادگیری ماشین
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            days (int): تعداد روزهای پیش‌بینی
            
        Returns:
            dict: نتایج پیش‌بینی
        """
        if df is None or df.empty or len(df) < 10:
            # مقادیر پیش‌فرض
            last_price = 30000 if 'BTC' in self.symbol else 2000 if 'ETH' in self.symbol else 100
            
            forecasted_values = [last_price * (1 + random.uniform(-0.02, 0.03)) for _ in range(days)]
            upper_bound = [val * 1.05 for val in forecasted_values]
            lower_bound = [val * 0.95 for val in forecasted_values]
            
            return {
                'forecasted_values': forecasted_values,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound
            }
        
        # گرفتن داده‌های قیمت
        prices = df['close'].values
        
        # محاسبه میانگین تغییرات اخیر
        window = min(30, len(prices) - 1)
        pct_changes = []
        
        for i in range(1, window + 1):
            pct_change = (prices[-i] / prices[-i-1]) - 1
            pct_changes.append(pct_change)
        
        avg_change = np.mean(pct_changes)
        std_change = np.std(pct_changes)
        
        # پیش‌بینی با استفاده از میانگین تغییرات
        last_price = prices[-1]
        forecasted_values = []
        upper_bound = []
        lower_bound = []
        
        current_price = last_price
        for _ in range(days):
            # افزایش تصادفی عدم قطعیت با گذشت زمان
            this_change = avg_change + np.random.normal(0, std_change)
            current_price = current_price * (1 + this_change)
            
            forecasted_values.append(current_price)
            
            # محاسبه حدود اطمینان
            confidence_interval = current_price * std_change * 1.96  # فاصله 95% اطمینان
            upper_bound.append(current_price + confidence_interval)
            lower_bound.append(current_price - confidence_interval)
        
        return {
            'forecasted_values': forecasted_values,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        }
    
    def save_model(self):
        """
        ذخیره مدل آموزش دیده برای استفاده بعدی
        
        Returns:
            str: لینک دانلود مدل
        """
        if self.model is None:
            st.error("مدلی برای ذخیره وجود ندارد!")
            return None
        
        try:
            # ذخیره مدل و تنظیمات آن
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'symbol': self.symbol,
                'training_time': self.last_training_time,
                'lookback': self.lookback,
                'forecast_days': self.forecast_days
            }
            
            # ذخیره به صورت بایت‌ها
            output = io.BytesIO()
            pickle.dump(model_data, output)
            output.seek(0)
            
            # تبدیل به base64 برای دانلود
            b64 = base64.b64encode(output.getvalue()).decode()
            
            # ایجاد لینک دانلود
            download_link = f'<a href="data:application/octet-stream;base64,{b64}" download="{self.symbol}_prediction_model.pkl">دانلود مدل آموزش‌دیده</a>'
            
            return download_link
        
        except Exception as e:
            st.error(f"خطا در ذخیره مدل: {str(e)}")
            return None
    
    def load_model(self, model_file):
        """
        بارگذاری مدل از فایل
        
        Args:
            model_file: فایل مدل آپلود شده
            
        Returns:
            bool: آیا بارگذاری موفقیت‌آمیز بوده است؟
        """
        try:
            # بارگذاری مدل
            model_data = pickle.load(model_file)
            
            # بازیابی اطلاعات مدل
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.symbol = model_data['symbol']
            self.last_training_time = model_data['training_time']
            self.lookback = model_data['lookback']
            self.forecast_days = model_data['forecast_days']
            
            st.success(f"مدل پیش‌بینی {self.symbol} با موفقیت بارگذاری شد.")
            return True
        
        except Exception as e:
            st.error(f"خطا در بارگذاری مدل: {str(e)}")
            return False
