"""
ماژول مدل‌های یادگیری عمیق برای پیش‌بینی قیمت ارزهای دیجیتال

این ماژول مدل‌های مختلف یادگیری عمیق شامل LSTM، GRU و Transformer را برای پیش‌بینی 
قیمت ارزهای دیجیتال پیاده‌سازی می‌کند.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Tuple, Union, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# تنظیم لاگ‌ها
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("deep_learning_models")

# فایل کش برای ذخیره مدل‌های آموزش دیده
MODELS_CACHE_DIR = "./dl_models_cache"
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

# ------------------- توابع پیش‌پردازش داده‌ها -------------------

def prepare_data(df: pd.DataFrame, target_column: str = 'close', sequence_length: int = 60, prediction_horizon: int = 7, test_size: float = 0.2) -> Tuple:
    """
    آماده‌سازی داده‌ها برای آموزش مدل‌های یادگیری عمیق
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های اصلی
        target_column (str): نام ستون هدف
        sequence_length (int): طول دنباله برای ویژگی‌های ورودی (تعداد داده‌های گذشته)
        prediction_horizon (int): افق پیش‌بینی (تعداد داده‌های آینده)
        test_size (float): نسبت داده‌های تست
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # کپی از داده‌ها
    data = df.copy()
    
    # استخراج ویژگی‌های مهم
    features = ['open', 'high', 'low', 'close', 'volume']
    
    # افزودن سایر ویژگی‌های مفید از دیتافریم
    for col in data.columns:
        if 'sma' in col or 'ema' in col or 'rsi' in col or 'macd' in col or 'bb_' in col:
            features.append(col)
    
    # فیلتر کردن ستون‌های موجود
    features = [f for f in features if f in data.columns]
    
    # پر کردن مقادیر خالی
    data = data[features].fillna(method='ffill').fillna(method='bfill')
    
    # نرمال‌سازی داده‌ها
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # ایجاد دنباله‌های ورودی و خروجی
    X, y = [], []
    target_idx = features.index(target_column)
    
    for i in range(len(scaled_data) - sequence_length - prediction_horizon + 1):
        X.append(scaled_data[i:i+sequence_length])
        
        # برای پیش‌بینی چند گام به جلو
        future_values = [scaled_data[i+sequence_length+j][target_idx] for j in range(prediction_horizon)]
        y.append(future_values)
    
    X, y = np.array(X), np.array(y)
    
    # تقسیم داده‌ها به مجموعه‌های آموزش و تست
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler, features, target_idx

def inverse_transform_predictions(predictions: np.ndarray, scaler: MinMaxScaler, features: List[str], target_idx: int) -> np.ndarray:
    """
    تبدیل معکوس پیش‌بینی‌ها به مقیاس اصلی
    
    Args:
        predictions (np.ndarray): پیش‌بینی‌های نرمال شده
        scaler (MinMaxScaler): مقیاس‌کننده
        features (List[str]): لیست ویژگی‌ها
        target_idx (int): شاخص ویژگی هدف
        
    Returns:
        np.ndarray: پیش‌بینی‌ها در مقیاس اصلی
    """
    # ایجاد ماتریس با همان تعداد ویژگی
    dummy = np.zeros((len(predictions), len(features)))
    
    # قرار دادن مقادیر پیش‌بینی شده در ستون هدف
    for i in range(len(predictions)):
        dummy[i, target_idx] = predictions[i]
    
    # تبدیل معکوس
    dummy_inverse = scaler.inverse_transform(dummy)
    
    # بازگرداندن فقط ستون هدف
    return dummy_inverse[:, target_idx]

# ------------------- کلاس پایه مدل‌های یادگیری عمیق -------------------

class BaseDLModel:
    """کلاس پایه برای مدل‌های یادگیری عمیق"""
    
    def __init__(self, model_name: str, sequence_length: int = 60, prediction_horizon: int = 7):
        """
        مقداردهی اولیه
        
        Args:
            model_name (str): نام مدل
            sequence_length (int): طول دنباله ورودی
            prediction_horizon (int): افق پیش‌بینی
        """
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.features = None
        self.target_idx = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        ساخت مدل (باید در کلاس‌های فرزند پیاده‌سازی شود)
        
        Args:
            input_shape (Tuple[int, int]): شکل ورودی (sequence_length, n_features)
        """
        raise NotImplementedError("باید در کلاس‌های فرزند پیاده‌سازی شود.")
        
    def train(self, df: pd.DataFrame, target_column: str = 'close', epochs: int = 50, batch_size: int = 32, validation_split: float = 0.1) -> Dict[str, Any]:
        """
        آموزش مدل
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌ها
            target_column (str): نام ستون هدف
            epochs (int): تعداد دوره‌های آموزش
            batch_size (int): اندازه بسته
            validation_split (float): نسبت داده‌های اعتبارسنجی
            
        Returns:
            Dict[str, Any]: تاریخچه آموزش
        """
        logger.info(f"آماده‌سازی داده‌ها برای آموزش مدل {self.model_name}")
        
        # آماده‌سازی داده‌ها
        X_train, X_test, y_train, y_test, self.scaler, self.features, self.target_idx = prepare_data(
            df, 
            target_column=target_column, 
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon
        )
        
        # ساخت مدل
        if self.model is None:
            logger.info(f"ساخت مدل {self.model_name}")
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        logger.info(f"آموزش مدل {self.model_name} با {epochs} دوره و اندازه بسته {batch_size}")
        
        # آموزش مدل
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # ارزیابی مدل
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        
        # پیش‌بینی و محاسبه معیارهای ارزیابی
        y_pred = self.model.predict(X_test)
        
        # تبدیل معکوس به مقیاس اصلی
        if y_pred.shape[1] == self.prediction_horizon:
            y_actual_list = []
            y_pred_list = []
            
            for day in range(self.prediction_horizon):
                y_actual = inverse_transform_predictions(y_test[:, day], self.scaler, self.features, self.target_idx)
                y_pred_day = inverse_transform_predictions(y_pred[:, day], self.scaler, self.features, self.target_idx)
                
                y_actual_list.append(y_actual)
                y_pred_list.append(y_pred_day)
                
            # محاسبه معیارهای ارزیابی برای روز اول پیش‌بینی
            mse = mean_squared_error(y_actual_list[0], y_pred_list[0])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_actual_list[0], y_pred_list[0])
            
            logger.info(f"ارزیابی مدل {self.model_name} - Loss: {loss}, RMSE: {rmse}, MAE: {mae}")
            
            # ذخیره مدل
            self.save_model()
            
            return {
                "model_name": self.model_name,
                "loss": loss,
                "rmse": rmse,
                "mae": mae,
                "history": {
                    "loss": self.history.history['loss'],
                    "val_loss": self.history.history['val_loss']
                }
            }
        else:
            logger.error(f"خطا در شکل خروجی مدل: {y_pred.shape}")
            return {"error": "خطا در شکل خروجی مدل"}
            
    def predict(self, df: pd.DataFrame, days_ahead: int = None) -> np.ndarray:
        """
        پیش‌بینی قیمت
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌ها
            days_ahead (int, optional): تعداد روزهای پیش‌بینی
            
        Returns:
            np.ndarray: پیش‌بینی‌های قیمت
        """
        if self.model is None:
            logger.error(f"مدل {self.model_name} آموزش ندیده است")
            return np.array([])
        
        # تعیین تعداد روزهای پیش‌بینی
        days_ahead = days_ahead or self.prediction_horizon
        
        # استفاده از sequence_length داده آخر برای پیش‌بینی
        data = df.copy().iloc[-self.sequence_length:]
        
        # استخراج ویژگی‌های مشابه با مجموعه آموزش
        if self.features:
            # استفاده از ویژگی‌های ذخیره شده
            features = [f for f in self.features if f in data.columns]
            data = data[features].fillna(method='ffill').fillna(method='bfill')
            
            # نرمال‌سازی داده‌ها
            scaled_data = self.scaler.transform(data)
            
            # شکل‌دهی داده برای ورود به مدل
            X = np.array([scaled_data])
            
            # پیش‌بینی
            predictions = self.model.predict(X)[0]
            
            # تبدیل معکوس به مقیاس اصلی
            if len(predictions) >= days_ahead:
                result = []
                for day in range(min(days_ahead, len(predictions))):
                    # تبدیل پیش‌بینی روز به مقیاس اصلی
                    pred_value = inverse_transform_predictions(
                        np.array([predictions[day]]), 
                        self.scaler, 
                        self.features, 
                        self.target_idx
                    )[0]
                    result.append(pred_value)
                    
                return np.array(result)
            else:
                logger.error(f"تعداد پیش‌بینی‌ها ({len(predictions)}) کمتر از روزهای درخواست شده ({days_ahead}) است")
                return np.array([])
        else:
            logger.error("ویژگی‌های مدل تنظیم نشده‌اند")
            return np.array([])
            
    def save_model(self) -> None:
        """ذخیره مدل آموزش دیده"""
        if self.model:
            try:
                # ذخیره مدل
                model_path = os.path.join(MODELS_CACHE_DIR, f"{self.model_name}.h5")
                self.model.save(model_path)
                
                # ذخیره تنظیمات
                settings = {
                    "model_name": self.model_name,
                    "sequence_length": self.sequence_length,
                    "prediction_horizon": self.prediction_horizon,
                    "features": self.features,
                    "target_idx": self.target_idx,
                    "history": {
                        "loss": self.history.history['loss'] if self.history else [],
                        "val_loss": self.history.history['val_loss'] if self.history else []
                    }
                }
                
                settings_path = os.path.join(MODELS_CACHE_DIR, f"{self.model_name}_settings.json")
                with open(settings_path, 'w') as f:
                    json.dump(settings, f)
                
                # ذخیره scaler
                if self.scaler:
                    import joblib
                    scaler_path = os.path.join(MODELS_CACHE_DIR, f"{self.model_name}_scaler.pkl")
                    joblib.dump(self.scaler, scaler_path)
                    
                logger.info(f"مدل {self.model_name} با موفقیت ذخیره شد")
            except Exception as e:
                logger.error(f"خطا در ذخیره مدل {self.model_name}: {str(e)}")
                
    def load_model(self) -> bool:
        """
        بارگیری مدل از فایل
        
        Returns:
            bool: موفقیت بارگیری
        """
        try:
            # بارگیری مدل
            from tensorflow.keras.models import load_model
            import joblib
            
            model_path = os.path.join(MODELS_CACHE_DIR, f"{self.model_name}.h5")
            if not os.path.exists(model_path):
                logger.warning(f"فایل مدل {self.model_name} یافت نشد")
                return False
                
            self.model = load_model(model_path)
            
            # بارگیری تنظیمات
            settings_path = os.path.join(MODELS_CACHE_DIR, f"{self.model_name}_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    
                self.sequence_length = settings.get("sequence_length", self.sequence_length)
                self.prediction_horizon = settings.get("prediction_horizon", self.prediction_horizon)
                self.features = settings.get("features")
                self.target_idx = settings.get("target_idx")
                
            # بارگیری scaler
            scaler_path = os.path.join(MODELS_CACHE_DIR, f"{self.model_name}_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            logger.info(f"مدل {self.model_name} با موفقیت بارگیری شد")
            return True
        except Exception as e:
            logger.error(f"خطا در بارگیری مدل {self.model_name}: {str(e)}")
            return False
            
    def plot_training_history(self) -> plt.Figure:
        """
        نمایش نمودار تاریخچه آموزش
        
        Returns:
            plt.Figure: نمودار تاریخچه
        """
        if self.history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.history.history['loss'], label='Training Loss')
            ax.plot(self.history.history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training History - {self.model_name}')
            ax.legend()
            ax.grid(True)
            return fig
        else:
            logger.warning("تاریخچه آموزش موجود نیست")
            return None

# ------------------- مدل‌های خاص یادگیری عمیق -------------------

class LSTMModel(BaseDLModel):
    """مدل LSTM برای پیش‌بینی قیمت"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 7, units: int = 50, dropout: float = 0.2):
        """
        مقداردهی اولیه
        
        Args:
            sequence_length (int): طول دنباله ورودی
            prediction_horizon (int): افق پیش‌بینی
            units (int): تعداد واحدهای LSTM
            dropout (float): نرخ dropout
        """
        super().__init__("LSTM", sequence_length, prediction_horizon)
        self.units = units
        self.dropout = dropout
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        ساخت مدل LSTM
        
        Args:
            input_shape (Tuple[int, int]): شکل ورودی (sequence_length, n_features)
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units),
            Dropout(self.dropout),
            Dense(self.prediction_horizon)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        logger.info(f"مدل {self.model_name} ساخته شد")


class GRUModel(BaseDLModel):
    """مدل GRU برای پیش‌بینی قیمت"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 7, units: int = 50, dropout: float = 0.2):
        """
        مقداردهی اولیه
        
        Args:
            sequence_length (int): طول دنباله ورودی
            prediction_horizon (int): افق پیش‌بینی
            units (int): تعداد واحدهای GRU
            dropout (float): نرخ dropout
        """
        super().__init__("GRU", sequence_length, prediction_horizon)
        self.units = units
        self.dropout = dropout
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        ساخت مدل GRU
        
        Args:
            input_shape (Tuple[int, int]): شکل ورودی (sequence_length, n_features)
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        self.model = Sequential([
            GRU(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            GRU(self.units),
            Dropout(self.dropout),
            Dense(self.prediction_horizon)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        logger.info(f"مدل {self.model_name} ساخته شد")


class TransformerModel(BaseDLModel):
    """مدل Transformer برای پیش‌بینی قیمت"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 7, d_model: int = 64, num_heads: int = 4, ff_dim: int = 128, dropout: float = 0.2):
        """
        مقداردهی اولیه
        
        Args:
            sequence_length (int): طول دنباله ورودی
            prediction_horizon (int): افق پیش‌بینی
            d_model (int): ابعاد مدل
            num_heads (int): تعداد سرهای توجه
            ff_dim (int): ابعاد لایه feed-forward
            dropout (float): نرخ dropout
        """
        super().__init__("Transformer", sequence_length, prediction_horizon)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        ساخت مدل Transformer
        
        Args:
            input_shape (Tuple[int, int]): شکل ورودی (sequence_length, n_features)
        """
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
        from tensorflow.keras.optimizers import Adam
        
        def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
            # توجه چندسره
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model
            )(inputs, inputs)
            attention_output = Dropout(dropout)(attention_output)
            attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
            
            # لایه feed-forward
            outputs = Dense(ff_dim, activation='relu')(attention_output)
            outputs = Dense(d_model)(outputs)
            outputs = Dropout(dropout)(outputs)
            outputs = LayerNormalization(epsilon=1e-6)(attention_output + outputs)
            
            return outputs
        
        inputs = Input(shape=input_shape)
        x = Dense(self.d_model)(inputs)
        
        # اعمال transformer encoder
        x = transformer_encoder(x, self.d_model, self.num_heads, self.ff_dim, self.dropout)
        x = transformer_encoder(x, self.d_model, self.num_heads, self.ff_dim, self.dropout)
        
        # جمع‌آوری اطلاعات از توالی
        x = GlobalAveragePooling1D()(x)
        
        # لایه خروجی
        outputs = Dense(self.prediction_horizon)(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        logger.info(f"مدل {self.model_name} ساخته شد")

# ------------------- کلاس مدیریت مدل‌های یادگیری عمیق -------------------

class DeepLearningManager:
    """مدیریت مدل‌های یادگیری عمیق"""
    
    def __init__(self):
        """مقداردهی اولیه"""
        self.models = {}
        
    def add_model(self, model: BaseDLModel) -> None:
        """
        افزودن مدل جدید
        
        Args:
            model (BaseDLModel): مدل یادگیری عمیق
        """
        self.models[model.model_name] = model
        logger.info(f"مدل {model.model_name} به مدیریت مدل‌ها اضافه شد")
        
    def get_model(self, model_name: str) -> Optional[BaseDLModel]:
        """
        دریافت مدل با نام مشخص
        
        Args:
            model_name (str): نام مدل
            
        Returns:
            Optional[BaseDLModel]: مدل یا None در صورت عدم وجود
        """
        return self.models.get(model_name)
        
    def train_model(self, model_name: str, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        آموزش مدل مشخص
        
        Args:
            model_name (str): نام مدل
            df (pd.DataFrame): دیتافریم داده‌ها
            **kwargs: پارامترهای اضافی برای آموزش
            
        Returns:
            Dict[str, Any]: نتایج آموزش
        """
        model = self.get_model(model_name)
        if model:
            return model.train(df, **kwargs)
        else:
            logger.error(f"مدل {model_name} یافت نشد")
            return {"error": f"مدل {model_name} یافت نشد"}
            
    def predict_with_model(self, model_name: str, df: pd.DataFrame, days_ahead: int = None) -> np.ndarray:
        """
        پیش‌بینی با مدل مشخص
        
        Args:
            model_name (str): نام مدل
            df (pd.DataFrame): دیتافریم داده‌ها
            days_ahead (int, optional): تعداد روزهای پیش‌بینی
            
        Returns:
            np.ndarray: پیش‌بینی‌های قیمت
        """
        model = self.get_model(model_name)
        if model:
            return model.predict(df, days_ahead)
        else:
            logger.error(f"مدل {model_name} یافت نشد")
            return np.array([])
            
    def ensemble_predict(self, df: pd.DataFrame, days_ahead: int = 7, model_weights: Dict[str, float] = None) -> np.ndarray:
        """
        پیش‌بینی ترکیبی با استفاده از چندین مدل
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌ها
            days_ahead (int, optional): تعداد روزهای پیش‌بینی
            model_weights (Dict[str, float], optional): وزن‌های مدل‌ها
            
        Returns:
            np.ndarray: پیش‌بینی‌های ترکیبی
        """
        # تنظیم وزن‌های پیش‌فرض (برابر)
        if model_weights is None:
            model_weights = {name: 1.0 / len(self.models) for name in self.models}
        
        predictions = []
        total_weight = 0
        
        for model_name, model in self.models.items():
            if model_name in model_weights:
                weight = model_weights[model_name]
                pred = model.predict(df, days_ahead)
                
                if len(pred) > 0:
                    predictions.append((pred, weight))
                    total_weight += weight
        
        if predictions and total_weight > 0:
            # نرمال‌سازی وزن‌ها
            normalized_weights = [weight / total_weight for _, weight in predictions]
            
            # ترکیب پیش‌بینی‌ها با وزن‌ها
            ensemble_pred = np.zeros(days_ahead)
            for i, (pred, _) in enumerate(predictions):
                ensemble_pred += pred[:days_ahead] * normalized_weights[i]
                
            return ensemble_pred
        else:
            logger.warning("هیچ پیش‌بینی معتبری برای ترکیب وجود ندارد")
            return np.array([])
            
    def load_all_models(self) -> int:
        """
        بارگیری تمام مدل‌های موجود
        
        Returns:
            int: تعداد مدل‌های بارگیری شده
        """
        loaded_count = 0
        for model_name, model in self.models.items():
            if model.load_model():
                loaded_count += 1
                
        return loaded_count
        
    def save_all_models(self) -> int:
        """
        ذخیره تمام مدل‌ها
        
        Returns:
            int: تعداد مدل‌های ذخیره شده
        """
        saved_count = 0
        for model_name, model in self.models.items():
            try:
                model.save_model()
                saved_count += 1
            except Exception as e:
                logger.error(f"خطا در ذخیره مدل {model_name}: {str(e)}")
                
        return saved_count
        
    def evaluate_all_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        ارزیابی تمام مدل‌ها
        
        Args:
            df (pd.DataFrame): دیتافریم داده‌ها
            
        Returns:
            Dict[str, Dict[str, Any]]: نتایج ارزیابی مدل‌ها
        """
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"ارزیابی مدل {model_name}")
            try:
                # پیش‌پردازش داده‌ها
                X_train, X_test, y_train, y_test, scaler, features, target_idx = prepare_data(
                    df, 
                    sequence_length=model.sequence_length,
                    prediction_horizon=model.prediction_horizon
                )
                
                # پیش‌بینی
                y_pred = model.model.predict(X_test)
                
                # ارزیابی
                evaluation = {}
                
                for day in range(model.prediction_horizon):
                    y_actual = inverse_transform_predictions(y_test[:, day], scaler, features, target_idx)
                    y_pred_day = inverse_transform_predictions(y_pred[:, day], scaler, features, target_idx)
                    
                    mse = mean_squared_error(y_actual, y_pred_day)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_actual, y_pred_day)
                    r2 = r2_score(y_actual, y_pred_day)
                    
                    evaluation[f"day_{day+1}"] = {
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2
                    }
                
                results[model_name] = evaluation
                logger.info(f"ارزیابی مدل {model_name} انجام شد")
            except Exception as e:
                logger.error(f"خطا در ارزیابی مدل {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
                
        return results

# ------------------- توابع کمکی -------------------

def create_default_dl_manager() -> DeepLearningManager:
    """
    ایجاد مدیریت مدل‌های یادگیری عمیق با مدل‌های پیش‌فرض
    
    Returns:
        DeepLearningManager: مدیریت مدل‌های یادگیری عمیق
    """
    manager = DeepLearningManager()
    
    # افزودن مدل LSTM
    lstm_model = LSTMModel(sequence_length=60, prediction_horizon=7, units=50)
    manager.add_model(lstm_model)
    
    # افزودن مدل GRU
    gru_model = GRUModel(sequence_length=60, prediction_horizon=7, units=50)
    manager.add_model(gru_model)
    
    # افزودن مدل Transformer
    transformer_model = TransformerModel(sequence_length=60, prediction_horizon=7)
    manager.add_model(transformer_model)
    
    # تلاش برای بارگیری مدل‌های از قبل آموزش دیده
    loaded_count = manager.load_all_models()
    logger.info(f"{loaded_count} مدل از قبل آموزش دیده بارگیری شد")
    
    return manager

def get_dl_manager() -> DeepLearningManager:
    """
    دریافت نمونه از مدیریت مدل‌های یادگیری عمیق (Singleton)
    
    Returns:
        DeepLearningManager: مدیریت مدل‌های یادگیری عمیق
    """
    if not hasattr(get_dl_manager, "instance") or get_dl_manager.instance is None:
        get_dl_manager.instance = create_default_dl_manager()
    
    return get_dl_manager.instance

def train_model_for_symbol(symbol: str, timeframe: str, model_name: str, lookback_days: int = 365) -> Dict[str, Any]:
    """
    آموزش مدل برای نماد مشخص
    
    Args:
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        model_name (str): نام مدل
        lookback_days (int): تعداد روزهای تاریخچه
        
    Returns:
        Dict[str, Any]: نتایج آموزش
    """
    from crypto_data import get_crypto_data
    from technical_analysis import perform_technical_analysis
    
    logger.info(f"آموزش مدل {model_name} برای {symbol} در تایم‌فریم {timeframe}")
    
    try:
        # دریافت داده‌ها
        df = get_crypto_data(symbol, timeframe, lookback_days, "binance")
        
        if df is None or df.empty:
            logger.error(f"خطا در دریافت داده‌های {symbol}")
            return {"error": f"خطا در دریافت داده‌های {symbol}"}
            
        # انجام تحلیل تکنیکال
        df = perform_technical_analysis(df)
        
        # دریافت مدیریت مدل‌ها
        manager = get_dl_manager()
        
        # آموزش مدل
        results = manager.train_model(model_name, df, epochs=50)
        
        return results
    except Exception as e:
        logger.error(f"خطا در آموزش مدل: {str(e)}")
        return {"error": str(e)}

def predict_future_prices(symbol: str, timeframe: str, days_ahead: int = 7, use_ensemble: bool = True) -> Dict[str, Any]:
    """
    پیش‌بینی قیمت‌های آینده
    
    Args:
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        days_ahead (int): تعداد روزهای پیش‌بینی
        use_ensemble (bool): استفاده از ترکیب مدل‌ها
        
    Returns:
        Dict[str, Any]: نتایج پیش‌بینی
    """
    from crypto_data import get_crypto_data
    from technical_analysis import perform_technical_analysis
    
    logger.info(f"پیش‌بینی قیمت {symbol} در تایم‌فریم {timeframe} برای {days_ahead} روز آینده")
    
    try:
        # دریافت داده‌ها
        df = get_crypto_data(symbol, timeframe, 60, "binance")
        
        if df is None or df.empty:
            logger.error(f"خطا در دریافت داده‌های {symbol}")
            return {"error": f"خطا در دریافت داده‌های {symbol}"}
            
        # انجام تحلیل تکنیکال
        df = perform_technical_analysis(df)
        
        # دریافت مدیریت مدل‌ها
        manager = get_dl_manager()
        
        # پیش‌بینی
        if use_ensemble:
            # پیش‌بینی ترکیبی
            predictions = manager.ensemble_predict(df, days_ahead)
            method = "ensemble"
        else:
            # استفاده از بهترین مدل (فرض LSTM)
            predictions = manager.predict_with_model("LSTM", df, days_ahead)
            method = "LSTM"
            
        if len(predictions) == 0:
            logger.warning("هیچ پیش‌بینی معتبری دریافت نشد")
            return {"error": "هیچ پیش‌بینی معتبری دریافت نشد"}
            
        # ایجاد دیتافریم نتایج
        current_price = df['close'].iloc[-1]
        date_range = [datetime.now() + timedelta(days=i+1) for i in range(len(predictions))]
        
        results_df = pd.DataFrame({
            'date': date_range,
            'price': predictions,
            'change': [(p / current_price - 1) * 100 for p in predictions]
        })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "predictions": predictions.tolist(),
            "dates": [d.strftime('%Y-%m-%d') for d in date_range],
            "changes": [(p / current_price - 1) * 100 for p in predictions],
            "method": method
        }
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی قیمت: {str(e)}")
        return {"error": str(e)}

# ------------------- فانکشن اصلی برای استفاده در برنامه -------------------

def get_price_predictions(df: pd.DataFrame, symbol: str, days_ahead: int = 7, use_deep_learning: bool = True) -> Dict[str, Any]:
    """
    دریافت پیش‌بینی‌های قیمت برای استفاده در برنامه اصلی
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        symbol (str): نماد ارز
        days_ahead (int): تعداد روزهای پیش‌بینی
        use_deep_learning (bool): استفاده از یادگیری عمیق
        
    Returns:
        Dict[str, Any]: نتایج پیش‌بینی
    """
    logger.info(f"دریافت پیش‌بینی‌های قیمت برای {symbol}")
    
    try:
        if use_deep_learning:
            # تلاش برای استفاده از مدل‌های یادگیری عمیق
            manager = get_dl_manager()
            loaded_count = manager.load_all_models()
            
            if loaded_count > 0:
                # استفاده از ترکیب مدل‌های یادگیری عمیق
                predictions = manager.ensemble_predict(df, days_ahead)
                method = "deep_learning_ensemble"
            else:
                # استفاده از روش ساده‌تر
                logger.warning("هیچ مدل یادگیری عمیقی در دسترس نیست، استفاده از روش ساده")
                use_deep_learning = False
        
        if not use_deep_learning:
            # روش ساده پیش‌بینی (مثلاً میانگین متحرک یا تخمین خطی)
            
            # استفاده از میانگین متحرک نمایی (EMA)
            if 'close' in df.columns:
                ema = df['close'].ewm(span=10).mean()
                last_ema = ema.iloc[-1]
                
                # محاسبه میانگین تغییرات روزانه
                daily_changes = df['close'].pct_change().dropna().iloc[-30:].mean()
                
                # پیش‌بینی با استفاده از تغییرات میانگین
                predictions = []
                current = last_ema
                
                for _ in range(days_ahead):
                    current = current * (1 + daily_changes)
                    predictions.append(current)
                
                predictions = np.array(predictions)
                method = "ema_forecast"
            else:
                logger.error("ستون 'close' در دیتافریم یافت نشد")
                return {"error": "ستون 'close' در دیتافریم یافت نشد"}
        
        # ایجاد نتایج
        current_price = df['close'].iloc[-1]
        date_range = [datetime.now() + timedelta(days=i+1) for i in range(len(predictions))]
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "predictions": predictions.tolist(),
            "dates": [d.strftime('%Y-%m-%d') for d in date_range],
            "changes": [(p / current_price - 1) * 100 for p in predictions],
            "method": method
        }
    except Exception as e:
        logger.error(f"خطا در دریافت پیش‌بینی‌های قیمت: {str(e)}")
        return {"error": str(e), "symbol": symbol}