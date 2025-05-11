"""
ماژول پیش‌بینی نرخ ارز با استفاده از یادگیری عمیق

این ماژول شامل کلاس‌ها و توابع مورد نیاز برای پیش‌بینی نرخ ارز با استفاده از مدل‌های یادگیری عمیق است.
این مدل‌ها می‌توانند روند آینده قیمت ارزهای دیجیتال را پیش‌بینی کنند.
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# بررسی وجود TensorFlow و نصب آن در صورت نیاز
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Bidirectional, GRU, Attention, Layer
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError:
    logger.warning("مشکل در واردسازی TensorFlow")

# مسیر ذخیره مدل‌ها
MODEL_CACHE_DIR = './dl_models_cache'

# ایجاد دایرکتوری کش مدل‌ها اگر وجود ندارد
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class ExchangeRatePredictor:
    """کلاس اصلی پیش‌بینی نرخ ارز با یادگیری عمیق"""
    
    def __init__(self, symbol: str, 
                forecast_days: int = 7, 
                lookback_days: int = 60,
                model_type: str = 'lstm',
                retrain: bool = False):
        """
        مقداردهی اولیه پیش‌بینی‌کننده نرخ ارز
        
        Args:
            symbol (str): نماد ارز
            forecast_days (int): تعداد روزهای آینده برای پیش‌بینی
            lookback_days (int): تعداد روزهای گذشته برای استفاده در آموزش
            model_type (str): نوع مدل ('lstm', 'gru', 'bilstm', 'cnn_lstm')
            retrain (bool): آموزش مجدد مدل حتی اگر از قبل وجود داشته باشد
        """
        self.symbol = symbol
        self.forecast_days = forecast_days
        self.lookback_days = lookback_days
        self.model_type = model_type
        
        # تنظیمات پیش‌پردازش داده‌ها
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 15  # تعداد روزهای متوالی برای استفاده در پیش‌بینی
        
        # تنظیمات مدل
        self.epochs = 100
        self.batch_size = 32
        self.early_stopping_patience = 10
        
        # مدل
        self.model = None
        
        # بررسی وجود مدل از قبل
        self.model_path = os.path.join(MODEL_CACHE_DIR, f'{symbol}_{model_type}_model.h5')
        self.scaler_path = os.path.join(MODEL_CACHE_DIR, f'{symbol}_{model_type}_scaler.pkl')
        self.last_training_date_path = os.path.join(MODEL_CACHE_DIR, f'{symbol}_{model_type}_last_training.json')
        
        if not retrain and os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self._load_model()
        
        logger.info(f"پیش‌بینی‌کننده نرخ ارز برای {symbol} با مدل {model_type} و پیش‌بینی {forecast_days} روز آینده راه‌اندازی شد")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        پیش‌بینی نرخ ارز برای روزهای آینده
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            
        Returns:
            dict: نتایج پیش‌بینی
        """
        try:
            if df.empty or len(df) < self.sequence_length:
                return {
                    'error': 'داده‌های ناکافی برای پیش‌بینی',
                    'forecast': []
                }
            
            # بررسی نیاز به آموزش مدل
            if self.model is None or self._is_retraining_needed(df):
                logger.info(f"آموزش مدل جدید برای {self.symbol}...")
                training_success = self.train(df)
                if not training_success:
                    return {
                        'error': 'خطا در آموزش مدل',
                        'forecast': []
                    }
            
            # آماده‌سازی داده‌های ورودی
            preprocessed_data = self._preprocess_data(df)
            prediction_data = self._prepare_input_data(preprocessed_data, is_training=False)
            
            # انجام پیش‌بینی
            input_sequences = prediction_data
            predicted_scaled = self.model.predict(input_sequences)
            
            # تبدیل به مقیاس اصلی
            predicted_prices = self.scaler.inverse_transform(predicted_scaled)
            
            # تهیه خروجی
            last_date = df.index[-1]
            forecast = []
            
            for i in range(self.forecast_days):
                next_date = last_date + timedelta(days=i+1)
                forecast.append({
                    'date': next_date.strftime('%Y-%m-%d'),
                    'price': float(predicted_prices[0, i]),
                    'confidence': self._calculate_confidence(i)
                })
            
            # تهیه خروجی نهایی
            prediction_result = {
                'symbol': self.symbol,
                'last_price': float(df['close'].iloc[-1]),
                'forecast': forecast,
                'model_type': self.model_type,
                'prediction_date': datetime.now().isoformat(),
                'confidence_score': self._calculate_overall_confidence()
            }
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"خطا در پیش‌بینی نرخ ارز: {str(e)}")
            return {
                'error': f'خطا در پیش‌بینی: {str(e)}',
                'forecast': []
            }
    
    def train(self, df: pd.DataFrame) -> bool:
        """
        آموزش مدل پیش‌بینی نرخ ارز
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            
        Returns:
            bool: موفقیت در آموزش مدل
        """
        try:
            if df.empty or len(df) < self.lookback_days:
                logger.warning(f"داده‌های ناکافی برای آموزش مدل {self.symbol}")
                return False
            
            logger.info(f"شروع آموزش مدل برای {self.symbol}...")
            
            # آماده‌سازی داده‌ها
            data = self._preprocess_data(df)
            X, y = self._prepare_training_data(data)
            
            # تقسیم داده‌ها به آموزش و اعتبارسنجی
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # ایجاد مدل
            if self.model is None:
                self.model = self._create_model(X.shape[1:])
            
            # کالبک‌های آموزش
            callbacks = [
                EarlyStopping(patience=self.early_stopping_patience, monitor='val_loss', verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
                ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # آموزش مدل
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # ذخیره مدل و اسکیلر
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # ذخیره تاریخ آخرین آموزش
            training_info = {
                'last_training_date': datetime.now().isoformat(),
                'data_end_date': df.index[-1].isoformat(),
                'data_rows': len(df),
                'epochs_trained': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
            
            with open(self.last_training_date_path, 'w') as f:
                json.dump(training_info, f)
            
            logger.info(f"آموزش مدل {self.symbol} با موفقیت به پایان رسید")
            return True
            
        except Exception as e:
            logger.error(f"خطا در آموزش مدل {self.symbol}: {str(e)}")
            return False
    
    def _load_model(self) -> None:
        """بارگذاری مدل از فایل ذخیره شده"""
        try:
            logger.info(f"بارگذاری مدل موجود برای {self.symbol}...")
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"مدل {self.symbol} با موفقیت بارگذاری شد")
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل {self.symbol}: {str(e)}")
            self.model = None
    
    def _is_retraining_needed(self, df: pd.DataFrame) -> bool:
        """
        بررسی نیاز به آموزش مجدد مدل
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات جدید
            
        Returns:
            bool: نیاز به آموزش مجدد
        """
        if not os.path.exists(self.last_training_date_path):
            return True
        
        try:
            with open(self.last_training_date_path, 'r') as f:
                training_info = json.load(f)
            
            last_training_date = datetime.fromisoformat(training_info['last_training_date'])
            data_end_date = datetime.fromisoformat(training_info['data_end_date'])
            
            # آموزش مجدد اگر بیش از 7 روز از آخرین آموزش گذشته باشد
            days_since_training = (datetime.now() - last_training_date).days
            if days_since_training > 7:
                logger.info(f"آموزش مجدد نیاز است (آخرین آموزش {days_since_training} روز پیش بوده است)")
                return True
            
            # آموزش مجدد اگر داده‌های جدید قابل توجهی اضافه شده باشند
            if df.index[-1] > pd.to_datetime(data_end_date):
                new_data_days = (df.index[-1] - pd.to_datetime(data_end_date)).days
                if new_data_days > 5:  # اگر بیش از 5 روز داده جدید اضافه شده باشد
                    logger.info(f"آموزش مجدد نیاز است ({new_data_days} روز داده جدید اضافه شده است)")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"خطا در بررسی نیاز به آموزش مجدد: {str(e)}")
            return True
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        پیش‌پردازش داده‌ها برای آموزش یا پیش‌بینی
        
        Args:
            df (pd.DataFrame): دیتافریم اطلاعات قیمت
            
        Returns:
            pd.DataFrame: دیتافریم پیش‌پردازش شده
        """
        # استفاده از ستون‌های مورد نیاز
        used_columns = ['close', 'volume']
        processed_df = df[used_columns].copy()
        
        # افزودن ویژگی‌های جدید
        processed_df['close_pct_change'] = processed_df['close'].pct_change()
        processed_df['volume_pct_change'] = processed_df['volume'].pct_change()
        processed_df['close_rolling_mean_5'] = processed_df['close'].rolling(window=5).mean()
        processed_df['close_rolling_mean_10'] = processed_df['close'].rolling(window=10).mean()
        processed_df['close_rolling_std_5'] = processed_df['close'].rolling(window=5).std()
        
        # جایگزینی مقادیر NaN با صفر
        processed_df.fillna(0, inplace=True)
        
        # مقیاس‌بندی داده‌ها
        close_price = processed_df[['close']].values
        scaled_features = self.scaler.fit_transform(close_price)
        
        # ایجاد دیتافریم مقیاس‌بندی شده
        scaled_df = pd.DataFrame(scaled_features, index=processed_df.index, columns=['close_scaled'])
        
        return scaled_df
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        آماده‌سازی داده‌ها برای آموزش
        
        Args:
            data (pd.DataFrame): دیتافریم پیش‌پردازش شده
            
        Returns:
            tuple: (ورودی آموزش، خروجی آموزش)
        """
        X = []
        y = []
        
        # ایجاد توالی‌های ورودی و خروجی
        for i in range(len(data) - self.sequence_length - self.forecast_days):
            # توالی ورودی
            X.append(data.iloc[i:i+self.sequence_length].values)
            
            # توالی خروجی
            output_seq = data.iloc[i+self.sequence_length:i+self.sequence_length+self.forecast_days].values
            y.append(output_seq)
        
        return np.array(X), np.array(y)
    
    def _prepare_input_data(self, data: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """
        آماده‌سازی داده‌های ورودی برای پیش‌بینی
        
        Args:
            data (pd.DataFrame): دیتافریم پیش‌پردازش شده
            is_training (bool): آیا برای آموزش است
            
        Returns:
            np.ndarray: داده‌های ورودی آماده
        """
        if is_training:
            # برای آموزش، کل داده‌ها آماده می‌شوند
            X = []
            for i in range(len(data) - self.sequence_length - self.forecast_days):
                X.append(data.iloc[i:i+self.sequence_length].values)
            return np.array(X)
        else:
            # برای پیش‌بینی، فقط آخرین توالی نیاز است
            latest_data = data.iloc[-self.sequence_length:].values
            return np.array([latest_data])
    
    def _create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        ایجاد مدل یادگیری عمیق
        
        Args:
            input_shape (tuple): شکل ورودی
            
        Returns:
            Sequential: مدل ایجاد شده
        """
        model = Sequential()
        
        if self.model_type == 'lstm':
            # مدل LSTM ساده
            model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(self.forecast_days))
            
        elif self.model_type == 'gru':
            # مدل GRU
            model.add(GRU(units=100, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(GRU(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(self.forecast_days))
            
        elif self.model_type == 'bilstm':
            # مدل LSTM دوطرفه
            model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(units=50)))
            model.add(Dropout(0.2))
            model.add(Dense(self.forecast_days))
            
        elif self.model_type == 'cnn_lstm':
            # مدل ترکیبی CNN-LSTM
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(units=100, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(self.forecast_days))
            
        else:
            # مدل پیش‌فرض LSTM
            model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(self.forecast_days))
        
        # کامپایل مدل
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def _calculate_confidence(self, day_ahead: int) -> float:
        """
        محاسبه میزان اطمینان به پیش‌بینی بر اساس فاصله زمانی
        
        Args:
            day_ahead (int): تعداد روزهای آینده
            
        Returns:
            float: میزان اطمینان (0-100)
        """
        # هر چه فاصله زمانی بیشتر باشد، اطمینان کمتر است
        base_confidence = 85.0  # اطمینان پایه برای روز اول
        decay_rate = 5.0  # نرخ کاهش اطمینان به ازای هر روز
        
        confidence = max(40.0, base_confidence - (day_ahead * decay_rate))
        return confidence
    
    def _calculate_overall_confidence(self) -> float:
        """
        محاسبه میزان اطمینان کلی به مدل
        
        Returns:
            float: میزان اطمینان کلی (0-100)
        """
        # بررسی وجود اطلاعات آموزش
        if not os.path.exists(self.last_training_date_path):
            return 50.0
        
        try:
            with open(self.last_training_date_path, 'r') as f:
                training_info = json.load(f)
            
            # فاکتورهای مؤثر در اطمینان
            last_training_date = datetime.fromisoformat(training_info['last_training_date'])
            days_since_training = (datetime.now() - last_training_date).days
            
            final_loss = training_info.get('final_val_loss', 0.05)
            data_rows = training_info.get('data_rows', 100)
            
            # محاسبه اطمینان
            freshness_score = max(0, 100 - (days_since_training * 2))  # تازگی مدل
            loss_score = 100 - min(100, final_loss * 1000)  # بر اساس خطای نهایی
            data_score = min(100, (data_rows / 200) * 100)  # بر اساس تعداد داده‌ها
            
            # میانگین وزن‌دهی شده
            confidence = (freshness_score * 0.4) + (loss_score * 0.4) + (data_score * 0.2)
            
            return max(40, min(95, confidence))  # محدود کردن به بازه 40-95
            
        except Exception as e:
            logger.error(f"خطا در محاسبه اطمینان کلی: {str(e)}")
            return 50.0


def get_exchange_rate_prediction(df: pd.DataFrame, symbol: str, forecast_days: int = 7, model_type: str = 'lstm') -> Dict[str, Any]:
    """
    دریافت پیش‌بینی نرخ ارز
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
        forecast_days (int): تعداد روزهای آینده برای پیش‌بینی
        model_type (str): نوع مدل ('lstm', 'gru', 'bilstm', 'cnn_lstm')
        
    Returns:
        dict: نتایج پیش‌بینی
    """
    try:
        predictor = ExchangeRatePredictor(
            symbol=symbol,
            forecast_days=forecast_days,
            model_type=model_type
        )
        
        predictions = predictor.predict(df)
        return predictions
    
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی نرخ ارز: {str(e)}")
        return {
            'error': f'خطا در پیش‌بینی: {str(e)}',
            'forecast': []
        }


def get_multi_model_prediction(df: pd.DataFrame, symbol: str, forecast_days: int = 7) -> Dict[str, Any]:
    """
    ترکیب پیش‌بینی‌های چندین مدل مختلف برای افزایش دقت
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
        forecast_days (int): تعداد روزهای آینده برای پیش‌بینی
        
    Returns:
        dict: نتایج پیش‌بینی ترکیبی
    """
    try:
        # پیش‌بینی با مدل‌های مختلف
        lstm_prediction = get_exchange_rate_prediction(df, symbol, forecast_days, 'lstm')
        gru_prediction = get_exchange_rate_prediction(df, symbol, forecast_days, 'gru')
        
        # بررسی خطا در پیش‌بینی‌ها
        if 'error' in lstm_prediction or 'error' in gru_prediction:
            return lstm_prediction if 'error' in gru_prediction else gru_prediction
        
        # ترکیب پیش‌بینی‌ها
        combined_forecast = []
        
        for i in range(forecast_days):
            lstm_price = lstm_prediction['forecast'][i]['price'] if i < len(lstm_prediction['forecast']) else 0
            lstm_confidence = lstm_prediction['forecast'][i]['confidence'] if i < len(lstm_prediction['forecast']) else 0
            
            gru_price = gru_prediction['forecast'][i]['price'] if i < len(gru_prediction['forecast']) else 0
            gru_confidence = gru_prediction['forecast'][i]['confidence'] if i < len(gru_prediction['forecast']) else 0
            
            # محاسبه میانگین وزنی بر اساس اطمینان
            total_confidence = lstm_confidence + gru_confidence
            if total_confidence > 0:
                combined_price = ((lstm_price * lstm_confidence) + (gru_price * gru_confidence)) / total_confidence
            else:
                combined_price = (lstm_price + gru_price) / 2
            
            # محاسبه اطمینان ترکیبی
            combined_confidence = max(lstm_confidence, gru_confidence) * 0.9  # کمی محافظه‌کارانه‌تر
            
            combined_forecast.append({
                'date': lstm_prediction['forecast'][i]['date'],
                'price': float(combined_price),
                'confidence': float(combined_confidence),
                'lstm_price': float(lstm_price),
                'gru_price': float(gru_price)
            })
        
        # نتیجه نهایی
        return {
            'symbol': symbol,
            'last_price': lstm_prediction['last_price'],
            'forecast': combined_forecast,
            'model_type': 'combined',
            'prediction_date': datetime.now().isoformat(),
            'confidence_score': min(lstm_prediction.get('confidence_score', 50), gru_prediction.get('confidence_score', 50))
        }
    
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی ترکیبی: {str(e)}")
        return {
            'error': f'خطا در پیش‌بینی ترکیبی: {str(e)}',
            'forecast': []
        }