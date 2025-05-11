"""
ماژول پیش‌بینی نرخ ارز (نسخه سبک)

این ماژول شامل توابع مورد نیاز برای پیش‌بینی نرخ ارز با روش‌های آماری ساده است
و نیازی به کتابخانه‌های سنگین یادگیری عمیق مانند TensorFlow ندارد.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json
import pickle
import time

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExchangeRatePredictor:
    """کلاس اصلی پیش‌بینی نرخ ارز با روش‌های آماری ساده"""
    
    def __init__(self, 
                prediction_periods: int = 7,
                feature_columns: Optional[List[str]] = None,
                cache_dir: str = './prediction_cache'):
        """
        مقداردهی اولیه پیش‌بینی‌کننده نرخ ارز
        
        Args:
            prediction_periods (int): تعداد دوره‌های پیش‌بینی
            feature_columns (list): لیست ستون‌های ویژگی
            cache_dir (str): مسیر دایرکتوری کش
        """
        self.prediction_periods = prediction_periods
        
        if feature_columns is None:
            self.feature_columns = ['close', 'volume', 'high', 'low']
        else:
            self.feature_columns = feature_columns
            
        self.cache_dir = cache_dir
        self.models = {}
        self.scalers = {}
        
        # ایجاد دایرکتوری کش اگر وجود ندارد
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
                logger.info(f"دایرکتوری کش ایجاد شد: {cache_dir}")
            except Exception as e:
                logger.warning(f"خطا در ایجاد دایرکتوری کش: {str(e)}")
        
        logger.info(f"پیش‌بینی‌کننده نرخ ارز با {prediction_periods} دوره پیش‌بینی راه‌اندازی شد")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close', 
                   window_size: int = 10) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        آماده‌سازی داده‌ها برای پیش‌بینی
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            target_column (str): ستون هدف برای پیش‌بینی
            window_size (int): اندازه پنجره برای ویژگی‌های تاریخی
            
        Returns:
            tuple: (ویژگی‌ها، برچسب‌ها، اسکیلر)
        """
        if df.empty:
            raise ValueError("دیتافریم خالی است")
            
        # کپی دیتافریم
        data = df.copy()
        
        # استفاده فقط از ستون‌های مورد نظر
        selected_columns = [col for col in self.feature_columns if col in data.columns]
        
        if not selected_columns:
            raise ValueError(f"هیچ یک از ستون‌های {self.feature_columns} در دیتافریم یافت نشد")
            
        # اطمینان از وجود ستون هدف
        if target_column not in data.columns:
            raise ValueError(f"ستون هدف {target_column} در دیتافریم یافت نشد")
            
        # مقیاس‌بندی داده‌ها
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[selected_columns])
        
        # ایجاد ویژگی‌ها و برچسب‌ها
        X, y = [], []
        
        for i in range(window_size, len(scaled_data)):
            # ویژگی‌ها: پنجره sliding از داده‌های گذشته
            X.append(scaled_data[i - window_size:i].flatten())
            # برچسب: قیمت بعدی
            y.append(scaled_data[i, selected_columns.index(target_column)])
        
        return np.array(X), np.array(y), scaler
    
    def train_linear_regression(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        آموزش مدل رگرسیون خطی
        
        Args:
            X (np.ndarray): داده‌های ویژگی
            y (np.ndarray): داده‌های هدف
            
        Returns:
            LinearRegression: مدل آموزش دیده
        """
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """
        آموزش مدل جنگل تصادفی
        
        Args:
            X (np.ndarray): داده‌های ویژگی
            y (np.ndarray): داده‌های هدف
            
        Returns:
            RandomForestRegressor: مدل آموزش دیده
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    def train_models(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        آموزش مدل‌های مختلف پیش‌بینی
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            
        Returns:
            dict: نتایج آموزش
        """
        try:
            # آماده‌سازی داده‌ها
            X, y, scaler = self.prepare_data(df)
            
            # تقسیم داده‌ها به مجموعه‌های آموزش و آزمایش
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # آموزش مدل‌ها
            linear_model = self.train_linear_regression(X_train, y_train)
            rf_model = self.train_random_forest(X_train, y_train)
            
            # ارزیابی مدل‌ها
            linear_score = linear_model.score(X_test, y_test)
            rf_score = rf_model.score(X_test, y_test)
            
            # ذخیره مدل‌ها و اسکیلر
            model_key = f"{symbol}_{timeframe}"
            self.models[model_key] = {
                'linear': linear_model,
                'random_forest': rf_model
            }
            self.scalers[model_key] = scaler
            
            # ذخیره مدل‌ها در کش
            try:
                self.save_models(symbol, timeframe)
            except Exception as e:
                logger.warning(f"خطا در ذخیره مدل‌ها: {str(e)}")
            
            return {
                'linear_score': linear_score,
                'rf_score': rf_score,
                'trained': True,
                'window_size': 10,
                'feature_columns': self.feature_columns
            }
            
        except Exception as e:
            logger.error(f"خطا در آموزش مدل‌ها: {str(e)}")
            return {
                'trained': False,
                'error': str(e)
            }
    
    def predict_next_periods(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        پیش‌بینی دوره‌های آینده
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌های قیمت
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            
        Returns:
            dict: نتایج پیش‌بینی
        """
        try:
            model_key = f"{symbol}_{timeframe}"
            
            # بررسی وجود مدل‌ها
            if model_key not in self.models:
                # تلاش برای بارگیری مدل‌ها از کش
                loaded = self.load_models(symbol, timeframe)
                
                if not loaded:
                    # آموزش مدل‌ها اگر در کش نیستند
                    training_result = self.train_models(df, symbol, timeframe)
                    
                    if not training_result['trained']:
                        return {
                            'success': False,
                            'error': 'خطا در آموزش مدل‌ها'
                        }
            
            # آماده‌سازی داده‌های آخر برای پیش‌بینی
            window_size = 10
            feature_cols = [col for col in self.feature_columns if col in df.columns]
            
            scaler = self.scalers[model_key]
            last_data = df[feature_cols].iloc[-window_size:].values
            last_scaled = scaler.transform(last_data)
            last_features = last_scaled.flatten().reshape(1, -1)
            
            # پیش‌بینی با هر دو مدل
            models = self.models[model_key]
            linear_model = models['linear']
            rf_model = models['random_forest']
            
            # پیش‌بینی‌های اولیه
            linear_pred = linear_model.predict(last_features)[0]
            rf_pred = rf_model.predict(last_features)[0]
            
            # نتایج پیش‌بینی برای دوره‌های آینده
            predictions = []
            current_features = last_features.copy()
            
            last_close = df['close'].iloc[-1]
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            last_volume = df['volume'].iloc[-1]
            
            for i in range(self.prediction_periods):
                # پیش‌بینی با هر دو مدل
                linear_pred = linear_model.predict(current_features)[0]
                rf_pred = rf_model.predict(current_features)[0]
                
                # میانگین وزنی پیش‌بینی‌ها
                ensemble_pred = 0.4 * linear_pred + 0.6 * rf_pred
                
                # تبدیل به قیمت واقعی
                feature_idx = feature_cols.index('close')
                inverse_data = np.zeros((1, len(feature_cols)))
                inverse_data[0, feature_idx] = ensemble_pred
                
                # مقداردهی سایر ویژگی‌ها برای inverse_transform
                if 'high' in feature_cols:
                    high_idx = feature_cols.index('high')
                    inverse_data[0, high_idx] = ensemble_pred * 1.01  # تقریب بالاتر
                
                if 'low' in feature_cols:
                    low_idx = feature_cols.index('low')
                    inverse_data[0, low_idx] = ensemble_pred * 0.99  # تقریب پایین‌تر
                
                if 'volume' in feature_cols:
                    volume_idx = feature_cols.index('volume')
                    inverse_data[0, volume_idx] = last_volume  # استفاده از آخرین حجم
                
                # برگرداندن مقیاس
                pred_value = scaler.inverse_transform(inverse_data)[0, feature_idx]
                
                # محاسبه فاصله زمانی برای تایم‌فریم
                current_date = df.index[-1]
                
                if 'm' in timeframe:
                    minutes = int(timeframe.replace('m', ''))
                    next_date = current_date + timedelta(minutes=minutes * (i + 1))
                elif 'h' in timeframe:
                    hours = int(timeframe.replace('h', ''))
                    next_date = current_date + timedelta(hours=hours * (i + 1))
                elif 'd' in timeframe:
                    days = int(timeframe.replace('d', ''))
                    next_date = current_date + timedelta(days=days * (i + 1))
                elif 'w' in timeframe:
                    weeks = int(timeframe.replace('w', ''))
                    next_date = current_date + timedelta(weeks=weeks * (i + 1))
                else:
                    next_date = current_date + timedelta(days=i + 1)
                
                # افزودن پیش‌بینی به لیست
                predictions.append({
                    'date': next_date,
                    'close': float(pred_value),
                    'estimate_high': float(pred_value * 1.01),
                    'estimate_low': float(pred_value * 0.99),
                    'linear_pred': float(linear_pred),
                    'rf_pred': float(rf_pred),
                    'ensemble_pred': float(ensemble_pred),
                    'period': i + 1
                })
                
                # بروزرسانی ویژگی‌ها برای پیش‌بینی بعدی
                # حذف اولین window_size/len(feature_cols) مقدار و افزودن پیش‌بینی جدید
                feature_size = window_size * len(feature_cols)
                new_features = current_features.flatten()[len(feature_cols):].copy()
                new_features = np.append(new_features, inverse_data.flatten())
                current_features = new_features.reshape(1, feature_size)
            
            # ایجاد دیتافریم پیش‌بینی
            dates = [p['date'] for p in predictions]
            closes = [p['close'] for p in predictions]
            highs = [p['estimate_high'] for p in predictions]
            lows = [p['estimate_low'] for p in predictions]
            
            pred_df = pd.DataFrame({
                'close': closes,
                'high': highs,
                'low': lows
            }, index=dates)
            
            return {
                'success': True,
                'predictions': predictions,
                'prediction_df': pred_df
            }
            
        except Exception as e:
            logger.error(f"خطا در پیش‌بینی دوره‌های آینده: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_models(self, symbol: str, timeframe: str) -> bool:
        """
        ذخیره مدل‌ها و اسکیلر در فایل
        
        Args:
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            
        Returns:
            bool: نتیجه ذخیره‌سازی
        """
        try:
            model_key = f"{symbol}_{timeframe}"
            
            if model_key not in self.models:
                return False
            
            models = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # ایجاد مسیر فایل‌ها
            linear_path = os.path.join(self.cache_dir, f"{model_key}_linear.pkl")
            rf_path = os.path.join(self.cache_dir, f"{model_key}_rf.pkl")
            scaler_path = os.path.join(self.cache_dir, f"{model_key}_scaler.pkl")
            
            # ذخیره مدل‌ها و اسکیلر
            joblib.dump(models['linear'], linear_path)
            joblib.dump(models['random_forest'], rf_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"مدل‌ها و اسکیلر برای {model_key} با موفقیت ذخیره شدند")
            return True
            
        except Exception as e:
            logger.error(f"خطا در ذخیره مدل‌ها: {str(e)}")
            return False
    
    def load_models(self, symbol: str, timeframe: str) -> bool:
        """
        بارگیری مدل‌ها و اسکیلر از فایل
        
        Args:
            symbol (str): نماد ارز
            timeframe (str): تایم‌فریم
            
        Returns:
            bool: نتیجه بارگیری
        """
        try:
            model_key = f"{symbol}_{timeframe}"
            
            # ایجاد مسیر فایل‌ها
            linear_path = os.path.join(self.cache_dir, f"{model_key}_linear.pkl")
            rf_path = os.path.join(self.cache_dir, f"{model_key}_rf.pkl")
            scaler_path = os.path.join(self.cache_dir, f"{model_key}_scaler.pkl")
            
            # بررسی وجود همه فایل‌ها
            if not all(os.path.exists(path) for path in [linear_path, rf_path, scaler_path]):
                return False
            
            # بارگیری مدل‌ها و اسکیلر
            linear_model = joblib.load(linear_path)
            rf_model = joblib.load(rf_path)
            scaler = joblib.load(scaler_path)
            
            # ذخیره در حافظه
            self.models[model_key] = {
                'linear': linear_model,
                'random_forest': rf_model
            }
            self.scalers[model_key] = scaler
            
            logger.info(f"مدل‌ها و اسکیلر برای {model_key} با موفقیت بارگیری شدند")
            return True
            
        except Exception as e:
            logger.error(f"خطا در بارگیری مدل‌ها: {str(e)}")
            return False


def get_exchange_rate_prediction(df: pd.DataFrame, symbol: str, timeframe: str, model_type: str = "statistical") -> Dict[str, Any]:
    """
    دریافت پیش‌بینی نرخ ارز
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        model_type (str): نوع مدل پیش‌بینی (statistical, random_forest)
        
    Returns:
        dict: نتایج پیش‌بینی
    """
    try:
        # حداقل داده‌های مورد نیاز
        if df.empty or len(df) < 50:
            return {
                'success': False,
                'error': 'داده‌های ناکافی برای پیش‌بینی'
            }
        
        # ایجاد پیش‌بینی‌کننده
        predictor = ExchangeRatePredictor(prediction_periods=14)
        
        # پیش‌بینی دوره‌های آینده
        prediction_result = predictor.predict_next_periods(df, symbol, timeframe)
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی نرخ ارز: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def get_multi_model_prediction(df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    دریافت پیش‌بینی‌های چند مدلی نرخ ارز
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        
    Returns:
        dict: نتایج پیش‌بینی چند مدلی
    """
    try:
        # حداقل داده‌های مورد نیاز
        if df.empty or len(df) < 50:
            return {
                'success': False,
                'error': 'داده‌های ناکافی برای پیش‌بینی'
            }
        
        # دوره‌های پیش‌بینی براساس تایم‌فریم
        periods = 14
        if 'h' in timeframe:
            periods = 24  # برای تایم‌فریم ساعتی، 24 دوره (1 روز)
        elif 'd' in timeframe:
            periods = 14  # برای تایم‌فریم روزانه، 14 دوره (2 هفته)
        elif 'w' in timeframe:
            periods = 8   # برای تایم‌فریم هفتگی، 8 دوره (2 ماه)
            
        # ایجاد پیش‌بینی‌کننده
        predictor = ExchangeRatePredictor(prediction_periods=periods)
        
        # آموزش مدل‌ها
        training_result = predictor.train_models(df, symbol, timeframe)
        
        if not training_result.get('trained', False):
            return {
                'success': False,
                'error': 'خطا در آموزش مدل‌ها'
            }
        
        # پیش‌بینی دوره‌های آینده
        prediction_result = predictor.predict_next_periods(df, symbol, timeframe)
        
        if not prediction_result['success']:
            return prediction_result
        
        # افزودن اطلاعات مدل‌ها
        prediction_result['model_info'] = {
            'linear_score': training_result['linear_score'],
            'rf_score': training_result['rf_score'],
            'feature_columns': training_result['feature_columns']
        }
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی چند مدلی نرخ ارز: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def evaluate_prediction_accuracy(df: pd.DataFrame, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    ارزیابی دقت پیش‌بینی‌ها
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های واقعی (برای مقایسه با پیش‌بینی‌های قبلی)
        prediction_result (dict): نتایج پیش‌بینی
        
    Returns:
        dict: نتایج ارزیابی
    """
    try:
        if not prediction_result.get('success', False):
            return {
                'success': False,
                'error': 'پیش‌بینی نامعتبر'
            }
        
        # دیتافریم پیش‌بینی
        pred_df = prediction_result.get('prediction_df')
        
        if pred_df is None or pred_df.empty:
            return {
                'success': False,
                'error': 'دیتافریم پیش‌بینی خالی است'
            }
        
        # تاریخ آخرین داده واقعی
        last_date = df.index[-1]
        
        # فیلتر کردن داده‌های قبل از تاریخ آخرین داده واقعی
        historical_preds = pred_df[pred_df.index <= last_date]
        
        if historical_preds.empty:
            return {
                'success': False,
                'error': 'داده‌های پیش‌بینی برای مقایسه با داده‌های واقعی وجود ندارد'
            }
        
        # داده‌های واقعی مربوطه
        relevant_dates = historical_preds.index
        actual_data = df[df.index.isin(relevant_dates)]
        
        if actual_data.empty:
            return {
                'success': False,
                'error': 'داده‌های واقعی برای مقایسه یافت نشد'
            }
        
        # محاسبه خطاها
        merged_df = pd.merge(
            actual_data['close'], 
            historical_preds['close'], 
            left_index=True, 
            right_index=True,
            suffixes=('_actual', '_pred')
        )
        
        # محاسبه RMSE و MAE
        errors = merged_df['close_actual'] - merged_df['close_pred']
        rmse = math.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        
        # محاسبه درصد خطا
        percent_errors = np.abs(errors / merged_df['close_actual']) * 100
        mape = np.mean(percent_errors)
        
        # محاسبه همبستگی
        correlation = merged_df['close_actual'].corr(merged_df['close_pred'])
        
        # محاسبه دقت جهت
        actual_diffs = merged_df['close_actual'].diff().dropna()
        pred_diffs = merged_df['close_pred'].diff().dropna()
        
        direction_accuracy = np.mean((actual_diffs > 0) == (pred_diffs > 0)) * 100
        
        return {
            'success': True,
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'correlation': float(correlation),
            'direction_accuracy': float(direction_accuracy),
            'num_samples': len(merged_df)
        }
        
    except Exception as e:
        logger.error(f"خطا در ارزیابی دقت پیش‌بینی: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }