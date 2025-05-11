"""ماژول تشخیص رویدادهای Black Swan

این ماژول الگوریتم‌های پیشرفته برای تشخیص رویدادهای غیرمنتظره و استثنایی بازار را فراهم می‌کند.
امکان پیش‌بینی و تشخیص تغییرات شدید قیمت، الگوهای غیرعادی و ریسک‌های غیرمعمول.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# تنظیم لاگر
logger = logging.getLogger(__name__)

class BlackSwanDetector:
    """کلاس تشخیص رویدادهای Black Swan"""
    
    def __init__(self):
        """راه‌اندازی تشخیص‌دهنده رویدادهای Black Swan"""
        self.models = {}
        self.volatility_history = defaultdict(list)
        self.correlation_history = defaultdict(list)
        self.event_history = []
        self.market_regime = "normal"  # می‌تواند normal, high_volatility, crisis, recovery باشد
        self.anomaly_thresholds = {
            "z_score": 3.0,
            "isolation_forest": -0.5,
            "volume_volatility": 3.0,
            "price_gap": 0.05,  # 5% تغییر قیمت یکجا
            "volume_spike": 5.0,  # 5x میانگین حجم
            "liquidity_drop": 0.5,  # 50% کاهش نقدینگی
            "volatility_spike": 2.5,  # 2.5x افزایش نوسان
            "correlation_breakdown": 0.4,  # تغییر همبستگی بیش از 0.4
            "smart_money_flow": 0.7  # جریان شدید پول هوشمند
        }
        
        # مدل‌های تشخیص ناهنجاری
        self.anomaly_detectors = {
            "isolation_forest": lambda: IsolationForest(n_estimators=100, contamination=0.01, random_state=42),
            "local_outlier_factor": lambda: LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True),
            "one_class_svm": lambda: OneClassSVM(kernel="rbf", gamma=0.1, nu=0.01)
        }
        
        # بارگذاری مدل‌های قبلی در صورت وجود
        try:
            self._load_models()
            logger.info("مدل‌های تشخیص Black Swan با موفقیت بارگذاری شدند")
        except Exception as e:
            logger.warning(f"خطا در بارگذاری مدل‌های قبلی: {str(e)}. مدل‌های جدید ایجاد خواهند شد.")
            
    def _load_models(self):
        """بارگذاری مدل‌های آموزش دیده قبلی"""
        try:
            self.models = joblib.load('black_swan_models.joblib')
        except FileNotFoundError:
            logger.info("مدل‌های ذخیره شده یافت نشدند. مدل‌های جدید ایجاد خواهند شد.")
            self.models = {}
            
    def _save_models(self):
        """ذخیره مدل‌های آموزش دیده"""
        joblib.dump(self.models, 'black_swan_models.joblib')
        logger.info("مدل‌های تشخیص Black Swan با موفقیت ذخیره شدند")
            
    def fit_models(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """آموزش مدل‌های تشخیص ناهنجاری
        
        Args:
            data (pd.DataFrame): دیتافریم داده‌های قیمت
            symbol (str): نماد ارز دیجیتال
            
        Returns:
            Dict[str, Any]: نتایج آموزش مدل‌ها
        """
        if len(data) < 100:
            logger.warning(f"داده‌های ناکافی برای آموزش مدل‌ها. حداقل 100 داده مورد نیاز است. تعداد داده‌های فعلی: {len(data)}")
            return {"success": False, "error": "insufficient_data"}
        
        try:
            # استخراج ویژگی‌ها
            features = self._extract_features(data)
            
            if features.shape[0] < 50 or np.any(np.isnan(features)):
                logger.warning("داده‌های ناکافی یا نامعتبر برای آموزش مدل‌ها")
                return {"success": False, "error": "invalid_features"}
            
            # آموزش مدل‌ها
            self.models[symbol] = {}
            
            for model_name, model_constructor in self.anomaly_detectors.items():
                model = model_constructor()
                model.fit(features)
                self.models[symbol][model_name] = model
                logger.info(f"مدل {model_name} برای {symbol} با موفقیت آموزش داده شد")
            
            # محاسبه مرزهای آماری
            self.models[symbol]["price_stats"] = {
                "returns_mean": np.mean(data['close'].pct_change().dropna()),
                "returns_std": np.std(data['close'].pct_change().dropna()),
                "volume_mean": np.mean(data['volume']),
                "volume_std": np.std(data['volume']),
                "volatility_mean": self._calculate_volatility(data).mean(),
                "volatility_std": self._calculate_volatility(data).std()
            }
            
            # ذخیره مدل‌ها
            self._save_models()
            
            return {"success": True, "models": list(self.models[symbol].keys())}
            
        except Exception as e:
            logger.error(f"خطا در آموزش مدل‌های تشخیص Black Swan: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def detect_anomalies(self, data: pd.DataFrame, symbol: str, sensitivity: float = 1.0) -> Dict[str, Any]:
        """تشخیص ناهنجاری‌ها و رویدادهای Black Swan
        
        Args:
            data (pd.DataFrame): دیتافریم داده‌های قیمت
            symbol (str): نماد ارز دیجیتال
            sensitivity (float): حساسیت تشخیص (1.0 = عادی، <1 = کمتر حساس، >1 = حساستر)
            
        Returns:
            Dict[str, Any]: نتایج تشخیص ناهنجاری‌ها
        """
        if len(data) < 10:
            logger.warning(f"داده‌های ناکافی برای تشخیص. تعداد داده‌های فعلی: {len(data)}")
            return {"success": False, "error": "insufficient_data"}
        
        # بررسی وجود مدل‌ها
        if symbol not in self.models:
            logger.info(f"مدل برای {symbol} یافت نشد. آموزش مدل‌ها...")
            fit_result = self.fit_models(data, symbol)
            if not fit_result["success"]:
                return {"success": False, "error": "model_training_failed"}
        
        try:
            # استخراج ویژگی‌ها
            current_features = self._extract_features(data.iloc[-50:])  # آخرین 50 داده برای تشخیص
            latest_features = current_features[-1:]
            
            if np.any(np.isnan(latest_features)):
                logger.warning("داده‌های نامعتبر برای تشخیص ناهنجاری")
                return {"success": False, "error": "invalid_features"}
            
            # تشخیص ناهنجاری با مدل‌های مختلف
            anomaly_scores = {}
            
            # استفاده از مدل‌های ماشین لرنینگ
            for model_name in ["isolation_forest", "local_outlier_factor", "one_class_svm"]:
                if model_name in self.models[symbol]:
                    model = self.models[symbol][model_name]
                    score = model.decision_function(latest_features)[0] if hasattr(model, "decision_function") else \
                            model.score_samples(latest_features)[0] if hasattr(model, "score_samples") else 0
                    anomaly_scores[model_name] = score
            
            # تشخیص ناهنجاری با روش‌های آماری و قوانین تخصصی
            anomaly_stats = self._statistical_anomaly_detection(data, symbol, sensitivity)
            
            # ترکیب نتایج
            anomaly_combined = self._combine_anomaly_scores(anomaly_scores, anomaly_stats, sensitivity)
            
            # تعیین وضعیت بازار
            self.market_regime = self._determine_market_regime(data, anomaly_combined)
            
            # افزودن به تاریخچه رویدادها اگر ناهنجاری شدید است
            if anomaly_combined["severe_anomaly"]:
                event = {
                    "timestamp": data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                    "symbol": symbol,
                    "anomaly_type": anomaly_combined["type"],
                    "severity": anomaly_combined["severity"],
                    "price": data.iloc[-1]["close"],
                    "description": anomaly_combined["description"]
                }
                self.event_history.append(event)
                logger.warning(f"Black Swan شناسایی شد! نوع: {event['anomaly_type']}, شدت: {event['severity']:.2f}")
            
            return {
                "success": True,
                "anomaly_detected": anomaly_combined["anomaly_detected"],
                "severe_anomaly": anomaly_combined["severe_anomaly"],
                "anomaly_type": anomaly_combined["type"],
                "severity": anomaly_combined["severity"],
                "description": anomaly_combined["description"],
                "market_regime": self.market_regime,
                "scores": {
                    "ml_models": anomaly_scores,
                    "statistical": anomaly_stats
                },
                "risk_level": self._calculate_risk_level(anomaly_combined, self.market_regime)
            }
            
        except Exception as e:
            logger.error(f"خطا در تشخیص ناهنجاری‌ها: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """استخراج ویژگی‌ها از داده‌های قیمت
        
        Args:
            data (pd.DataFrame): دیتافریم داده‌های قیمت
            
        Returns:
            np.ndarray: ماتریس ویژگی‌ها
        """
        # محاسبه ویژگی‌های مختلف
        features_list = []
        
        # محاسبه بازدهی‌ها
        returns = data['close'].pct_change().fillna(0).values
        features_list.append(returns)
        
        # محاسبه log returns
        log_returns = np.diff(np.log(data['close'].values))
        log_returns = np.append(0, log_returns)  # اضافه کردن صفر برای ردیف اول
        features_list.append(log_returns)
        
        # محاسبه نوسان (از کندل‌های قیمت)
        volatility = ((data['high'] - data['low']) / data['close']).values
        features_list.append(volatility)
        
        # محاسبه حجم نرمالایز شده
        volume_ma = data['volume'].rolling(window=20).mean().fillna(method='bfill').values
        normalized_volume = (data['volume'] / volume_ma).fillna(1).values
        features_list.append(normalized_volume)
        
        # محاسبه شکاف قیمتی (gap)
        price_gaps = np.zeros_like(returns)
        for i in range(1, len(data)):
            price_gaps[i] = (data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        features_list.append(price_gaps)
        
        # محاسبه تغییرات حجم
        volume_change = data['volume'].pct_change().fillna(0).values
        features_list.append(volume_change)
        
        # محاسبه نسبت بالا-پایین (با میانگین متحرک 20 روزه)
        high_low_ratio = ((data['high'] - data['close']) / (data['close'] - data['low'])).replace([np.inf, -np.inf], np.nan).fillna(1).values
        features_list.append(high_low_ratio)
        
        # جریان قدرت (تعادل روزانه قیمت-حجم)
        money_flow = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['volume']
        money_flow = money_flow.replace([np.inf, -np.inf], np.nan).fillna(0).values
        features_list.append(money_flow)
        
        # تبدیل به ماتریس ویژگی‌ها
        features = np.column_stack(features_list)
        
        # حذف ردیف‌های با داده‌های نامعتبر
        valid_rows = ~np.any(np.isnan(features), axis=1) & ~np.any(np.isinf(features), axis=1)
        features = features[valid_rows]
        
        return features
    
    def _statistical_anomaly_detection(self, data: pd.DataFrame, symbol: str, sensitivity: float) -> Dict[str, float]:
        """تشخیص ناهنجاری با روش‌های آماری
        
        Args:
            data (pd.DataFrame): دیتافریم داده‌های قیمت
            symbol (str): نماد ارز دیجیتال
            sensitivity (float): حساسیت تشخیص
            
        Returns:
            Dict[str, float]: نتایج تشخیص آماری
        """
        if symbol not in self.models or "price_stats" not in self.models[symbol]:
            # استفاده از کل دیتاست برای محاسبه آمارها
            price_stats = {
                "returns_mean": np.mean(data['close'].pct_change().dropna()),
                "returns_std": np.std(data['close'].pct_change().dropna()),
                "volume_mean": np.mean(data['volume']),
                "volume_std": np.std(data['volume']),
                "volatility_mean": self._calculate_volatility(data).mean(),
                "volatility_std": self._calculate_volatility(data).std()
            }
        else:
            price_stats = self.models[symbol]["price_stats"]
        
        # محاسبه مقادیر فعلی
        latest_return = data['close'].pct_change().iloc[-1]
        latest_volume = data['volume'].iloc[-1]
        latest_volatility = self._calculate_volatility(data).iloc[-1]
        
        # محاسبه Z-score برای مقادیر فعلی (با اعمال حساسیت)
        return_z = abs(latest_return - price_stats["returns_mean"]) / price_stats["returns_std"] * sensitivity
        volume_z = abs(latest_volume - price_stats["volume_mean"]) / price_stats["volume_std"] * sensitivity
        volatility_z = abs(latest_volatility - price_stats["volatility_mean"]) / price_stats["volatility_std"] * sensitivity
        
        # محاسبه شاخص‌های دیگر
        price_gap = 0
        if len(data) > 1:
            price_gap = abs((data['open'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]) * sensitivity
        
        volume_spike = (latest_volume / price_stats["volume_mean"]) * sensitivity
        
        # محاسبه شاخص نقدینگی با استفاده از شاخص شکاف قیمت-حجم
        liquidity_index = 0
        if len(data) >= 5:
            recent_price_range = (data['high'].iloc[-5:] - data['low'].iloc[-5:]).mean()
            recent_volume = data['volume'].iloc[-5:].mean()
            liquidity_index = (recent_price_range / data['close'].iloc[-1]) / (recent_volume / price_stats["volume_mean"]) * sensitivity
        
        # محاسبه شاخص هوشمند جریان پول
        smart_money_flow = 0
        if len(data) >= 5:
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
            volume_change = (data['volume'].iloc[-1] - data['volume'].iloc[-5]) / data['volume'].iloc[-5]
            smart_money_flow = abs(price_change * volume_change) * sensitivity
        
        return {
            "return_z_score": return_z,
            "volume_z_score": volume_z,
            "volatility_z_score": volatility_z,
            "price_gap": price_gap,
            "volume_spike": volume_spike,
            "liquidity_index": liquidity_index,
            "smart_money_flow": smart_money_flow
        }
    
    def _combine_anomaly_scores(self, ml_scores: Dict[str, float], stat_scores: Dict[str, float], sensitivity: float) -> Dict[str, Any]:
        """ترکیب نتایج تشخیص ناهنجاری از روش‌های مختلف
        
        Args:
            ml_scores (Dict[str, float]): امتیازات مدل‌های ماشین لرنینگ
            stat_scores (Dict[str, float]): امتیازات آماری
            sensitivity (float): حساسیت تشخیص
            
        Returns:
            Dict[str, Any]: نتایج ترکیبی
        """
        # تعیین آستانه‌های تطبیقی با توجه به حساسیت
        thresholds = {}
        for key, value in self.anomaly_thresholds.items():
            thresholds[key] = value / sensitivity  # مقدار کمتر = حساسیت بیشتر
        
        # بررسی نتایج مدل‌های ماشین لرنینگ
        ml_anomaly = False
        if "isolation_forest" in ml_scores and ml_scores["isolation_forest"] < thresholds["isolation_forest"]:
            ml_anomaly = True
        
        # بررسی نتایج آماری
        stat_anomalies = []
        anomaly_types = []
        max_severity = 0
        
        # بررسی Z-score بازدهی
        if stat_scores["return_z_score"] > thresholds["z_score"]:
            stat_anomalies.append("return_z_score")
            anomaly_types.append("price_shock")
            max_severity = max(max_severity, stat_scores["return_z_score"] / thresholds["z_score"])
        
        # بررسی Z-score حجم
        if stat_scores["volume_z_score"] > thresholds["z_score"]:
            stat_anomalies.append("volume_z_score")
            anomaly_types.append("volume_anomaly")
            max_severity = max(max_severity, stat_scores["volume_z_score"] / thresholds["z_score"])
        
        # بررسی Z-score نوسان
        if stat_scores["volatility_z_score"] > thresholds["volatility_spike"]:
            stat_anomalies.append("volatility_z_score")
            anomaly_types.append("volatility_spike")
            max_severity = max(max_severity, stat_scores["volatility_z_score"] / thresholds["volatility_spike"])
        
        # بررسی شکاف قیمتی
        if stat_scores["price_gap"] > thresholds["price_gap"]:
            stat_anomalies.append("price_gap")
            anomaly_types.append("price_gap")
            max_severity = max(max_severity, stat_scores["price_gap"] / thresholds["price_gap"])
        
        # بررسی جهش حجم
        if stat_scores["volume_spike"] > thresholds["volume_spike"]:
            stat_anomalies.append("volume_spike")
            anomaly_types.append("volume_spike")
            max_severity = max(max_severity, stat_scores["volume_spike"] / thresholds["volume_spike"])
        
        # بررسی شاخص نقدینگی
        if stat_scores["liquidity_index"] > thresholds["liquidity_drop"]:
            stat_anomalies.append("liquidity_index")
            anomaly_types.append("liquidity_crisis")
            max_severity = max(max_severity, stat_scores["liquidity_index"] / thresholds["liquidity_drop"])
        
        # بررسی جریان پول هوشمند
        if stat_scores["smart_money_flow"] > thresholds["smart_money_flow"]:
            stat_anomalies.append("smart_money_flow")
            anomaly_types.append("smart_money_movement")
            max_severity = max(max_severity, stat_scores["smart_money_flow"] / thresholds["smart_money_flow"])
        
        # تعیین نتیجه نهایی
        anomaly_detected = ml_anomaly or len(stat_anomalies) >= 2  # ناهنجاری مدل یا حداقل 2 ناهنجاری آماری
        severe_anomaly = ml_anomaly and len(stat_anomalies) >= 2  # ناهنجاری شدید: هر دو نوع تشخیص داده شده
        
        # تعیین نوع ناهنجاری غالب
        anomaly_type = "unknown"
        if anomaly_types:
            # انتخاب نوع ناهنجاری با بیشترین تکرار
            from collections import Counter
            anomaly_type = Counter(anomaly_types).most_common(1)[0][0]
        
        # تولید توضیحات
        description = self._generate_anomaly_description(anomaly_type, stat_anomalies, max_severity)
        
        return {
            "anomaly_detected": anomaly_detected,
            "severe_anomaly": severe_anomaly,
            "type": anomaly_type,
            "severity": max_severity,
            "description": description
        }
    
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """محاسبه نوسان قیمت با استفاده از انحراف معیار بازدهی‌ها
        
        Args:
            data (pd.DataFrame): دیتافریم داده‌های قیمت
            window (int): پنجره زمانی محاسبه نوسان
            
        Returns:
            pd.Series: سری زمانی نوسان
        """
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(window)  # تبدیل به نوسان سالانه
        return volatility
    
    def _determine_market_regime(self, data: pd.DataFrame, anomaly_result: Dict[str, Any]) -> str:
        """تعیین رژیم فعلی بازار
        
        Args:
            data (pd.DataFrame): دیتافریم داده‌های قیمت
            anomaly_result (Dict[str, Any]): نتایج تشخیص ناهنجاری
            
        Returns:
            str: وضعیت بازار (normal, high_volatility, crisis, recovery)
        """
        # محاسبه نوسان فعلی
        recent_volatility = self._calculate_volatility(data.iloc[-20:]).iloc[-1] if len(data) >= 20 else 0
        
        # محاسبه روند قیمت (20 روزه)
        if len(data) >= 20:
            price_trend = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        else:
            price_trend = 0
        
        # تعیین وضعیت بازار
        if anomaly_result["severe_anomaly"] and anomaly_result["type"] in ["price_shock", "liquidity_crisis"]:
            return "crisis"  # حالت بحران
        elif recent_volatility > 0.03:  # نوسان بالا (بیش از 3% روزانه)
            return "high_volatility"
        elif self.market_regime == "crisis" and price_trend > 0.1:  # بهبودی از وضعیت بحران
            return "recovery"
        elif self.market_regime == "high_volatility" and recent_volatility < 0.02:
            return "normal"
        else:
            return self.market_regime  # حفظ وضعیت فعلی
    
    def _calculate_risk_level(self, anomaly_result: Dict[str, Any], market_regime: str) -> Dict[str, Any]:
        """محاسبه سطح ریسک بر اساس نتایج تشخیص ناهنجاری و وضعیت بازار
        
        Args:
            anomaly_result (Dict[str, Any]): نتایج تشخیص ناهنجاری
            market_regime (str): وضعیت بازار
            
        Returns:
            Dict[str, Any]: اطلاعات سطح ریسک
        """
        # تعیین سطح ریسک پایه بر اساس وضعیت بازار
        base_risk = {
            "normal": 2,
            "high_volatility": 3,
            "recovery": 4,
            "crisis": 5
        }.get(market_regime, 2)  # عدد بزرگتر = ریسک بیشتر
        
        # تعدیل ریسک بر اساس شدت ناهنجاری
        if anomaly_result["severe_anomaly"]:
            risk_adjustment = min(3, anomaly_result["severity"] / 2)  # حداکثر 3 واحد افزایش
        elif anomaly_result["anomaly_detected"]:
            risk_adjustment = min(1.5, anomaly_result["severity"] / 3)  # حداکثر 1.5 واحد افزایش
        else:
            risk_adjustment = 0
        
        # محاسبه ریسک نهایی (1 تا 10)
        final_risk = min(10, base_risk + risk_adjustment)
        
        # تعیین سطح ریسک به صورت اسمی
        risk_level = "low"
        if final_risk >= 8:
            risk_level = "extreme"
        elif final_risk >= 6:
            risk_level = "high"
        elif final_risk >= 4:
            risk_level = "moderate"
        elif final_risk >= 2:
            risk_level = "low"
        
        # پیشنهادات مدیریت ریسک
        risk_recommendations = self._generate_risk_recommendations(risk_level, anomaly_result["type"], market_regime)
        
        return {
            "numeric": final_risk,
            "level": risk_level,
            "recommendations": risk_recommendations
        }
    
    def _generate_anomaly_description(self, anomaly_type: str, stat_anomalies: List[str], severity: float) -> str:
        """تولید توضیحات توصیفی برای ناهنجاری تشخیص داده شده
        
        Args:
            anomaly_type (str): نوع ناهنجاری
            stat_anomalies (List[str]): لیست ناهنجاری‌های آماری تشخیص داده شده
            severity (float): شدت ناهنجاری
            
        Returns:
            str: توضیحات توصیفی
        """
        severity_desc = "متوسط"
        if severity > 3:
            severity_desc = "بسیار شدید"
        elif severity > 2:
            severity_desc = "شدید"
        elif severity < 1:
            severity_desc = "خفیف"
        
        descriptions = {
            "price_shock": f"شوک قیمتی {severity_desc} تشخیص داده شد. تغییرات غیرمعمول و ناگهانی در قیمت.",
            "volume_anomaly": f"ناهنجاری {severity_desc} در حجم معاملات شناسایی شد. تغییرات غیرمعمول در الگوی حجم معاملات.",
            "volatility_spike": f"افزایش {severity_desc} نوسانات تشخیص داده شد. افزایش شدید در نوسانات قیمت.",
            "price_gap": f"شکاف قیمتی {severity_desc} تشخیص داده شد. تغییر ناگهانی در قیمت بین دو دوره معاملاتی.",
            "volume_spike": f"جهش {severity_desc} در حجم معاملات تشخیص داده شد. افزایش ناگهانی در حجم معاملات.",
            "liquidity_crisis": f"بحران نقدینگی {severity_desc} تشخیص داده شد. کاهش شدید در نقدینگی بازار.",
            "smart_money_movement": f"حرکت {severity_desc} سرمایه هوشمند شناسایی شد. تغییرات غیرمعمول در الگوی جریان سرمایه.",
            "unknown": f"ناهنجاری {severity_desc} با نوع نامشخص تشخیص داده شد."
        }
        
        base_description = descriptions.get(anomaly_type, descriptions["unknown"])
        
        # افزودن جزئیات بیشتر در صورت لزوم
        if len(stat_anomalies) > 1:
            base_description += f" شاخص‌های آماری متعدد ({len(stat_anomalies)}) این ناهنجاری را تأیید می‌کنند."
        
        return base_description
    
    def _generate_risk_recommendations(self, risk_level: str, anomaly_type: str, market_regime: str) -> List[str]:
        """تولید توصیه‌های مدیریت ریسک
        
        Args:
            risk_level (str): سطح ریسک (low, moderate, high, extreme)
            anomaly_type (str): نوع ناهنجاری
            market_regime (str): وضعیت بازار
            
        Returns:
            List[str]: لیست توصیه‌ها
        """
        recommendations = []
        
        # توصیه‌های عمومی بر اساس سطح ریسک
        if risk_level == "extreme":
            recommendations.append("خروج سریع از موقعیت‌های معاملاتی با لوریج بالا")
            recommendations.append("کاهش اندازه معاملات به حداقل ممکن")
            recommendations.append("استفاده از معاملات محافظتی مانند استرادل برای پوشش ریسک")
            
        elif risk_level == "high":
            recommendations.append("کاهش لوریج به 50% حد معمول")
            recommendations.append("تنظیم حد ضرر نزدیک‌تر برای همه معاملات")
            recommendations.append("اجتناب از باز کردن موقعیت‌های جدید تا بازگشت به ثبات")
            
        elif risk_level == "moderate":
            recommendations.append("کاهش اندازه معاملات به 75% حد معمول")
            recommendations.append("بررسی دقیق‌تر سیگنال‌ها قبل از ورود به معامله")
            
        else:  # "low"
            recommendations.append("شرایط معمولی بازار: پیروی از استراتژی‌های معمول معاملاتی")
        
        # توصیه‌های اختصاصی بر اساس نوع ناهنجاری
        if anomaly_type == "price_shock":
            recommendations.append("صبر کنید تا وضعیت شوک قیمتی تثبیت شود (حداقل 24 ساعت)")
        elif anomaly_type == "liquidity_crisis":
            recommendations.append("اجتناب از قرار دادن سفارشات بزرگ که ممکن است با لغزش قیمت مواجه شوند")
        elif anomaly_type == "volatility_spike":
            recommendations.append("حد ضررها را نزدیک‌تر تنظیم کنید تا ریسک نوسانات شدید را مدیریت کنید")
        elif anomaly_type == "smart_money_movement":
            recommendations.append("به جهت جریان سرمایه هوشمند توجه کنید، اما با احتیاط بیشتر معامله کنید")
        
        # توصیه‌های اختصاصی بر اساس وضعیت بازار
        if market_regime == "crisis":
            recommendations.append("قرار دادن استراتژی‌های دفاعی: تبدیل دارایی‌های پرریسک به دارایی‌های امن مانند استیبل کوین")
        elif market_regime == "high_volatility":
            recommendations.append("تنظیم پارامترهای معاملاتی با توجه به نوسان بالا: افزایش فاصله حد سود و حد ضرر")
        elif market_regime == "recovery":
            recommendations.append("شناسایی فرصت‌های خرید در مرحله بهبودی، اما با اندازه معامله کمتر")
        
        return recommendations

    def get_most_recent_events(self, count: int = 5) -> List[Dict[str, Any]]:
        """دریافت آخرین رویدادهای بلک سوان تشخیص داده شده
        
        Args:
            count (int): تعداد رویدادهای بازگردانده شده
            
        Returns:
            List[Dict[str, Any]]: لیست رویدادها
        """
        return sorted(self.event_history, key=lambda x: x["timestamp"], reverse=True)[:count]
    
    def get_market_state_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه وضعیت فعلی بازار
        
        Returns:
            Dict[str, Any]: خلاصه وضعیت بازار
        """
        return {
            "regime": self.market_regime,
            "recent_events_count": len([e for e in self.event_history if (datetime.now() - e["timestamp"]).days < 7]),
            "current_status": {
                "normal": "وضعیت عادی بازار",
                "high_volatility": "وضعیت نوسان شدید",
                "crisis": "وضعیت بحرانی",
                "recovery": "وضعیت بهبودی از بحران"
            }.get(self.market_regime, "وضعیت نامشخص")
        }

# نمونه استفاده از کلاس
if __name__ == "__main__":
    # راه‌اندازی تشخیص‌دهنده Black Swan
    detector = BlackSwanDetector()
    
    # مثالی از روش استفاده:
    # 1. آموزش مدل‌ها با داده‌های تاریخی
    # df_historical = ... # دریافت داده‌های تاریخی
    # detector.fit_models(df_historical, "BTC/USDT")
    
    # 2. تشخیص ناهنجاری‌ها در داده‌های جدید
    # df_recent = ... # دریافت داده‌های اخیر
    # anomaly_results = detector.detect_anomalies(df_recent, "BTC/USDT", sensitivity=1.2)
    
    # 3. بررسی نتایج
    # if anomaly_results["anomaly_detected"]:
    #     print(f"هشدار: ناهنجاری تشخیص داده شد! نوع: {anomaly_results['anomaly_type']}")
    #     print(f"توضیحات: {anomaly_results['description']}")
    #     print(f"سطح ریسک: {anomaly_results['risk_level']['level']} ({anomaly_results['risk_level']['numeric']}/10)")
    #     print(f"توصیه‌های مدیریت ریسک:")
    #     for rec in anomaly_results['risk_level']['recommendations']:
    #         print(f"- {rec}")
