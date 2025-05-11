"""
ماژول تحلیل ریسک و بازده معاملاتی

این ماژول شامل توابع و کلاس‌های مورد نیاز برای تحلیل ریسک و بازده معاملات،
محاسبه پارامترهای مدیریت ریسک و بهینه‌سازی سایز پوزیشن است.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast
from datetime import datetime, timedelta
import math
import json

# تنظیم لاگر
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskRewardAnalyzer:
    """کلاس اصلی تحلیل ریسک و بازده معاملاتی"""
    
    def __init__(self, 
                initial_capital: float = 1000.0,
                max_risk_per_trade: float = 0.02,
                target_risk_reward_ratio: float = 2.0,
                confidence_level: float = 0.95):
        """
        مقداردهی اولیه تحلیل‌گر ریسک و بازده
        
        Args:
            initial_capital (float): سرمایه اولیه
            max_risk_per_trade (float): حداکثر ریسک در هر معامله (به درصد)
            target_risk_reward_ratio (float): نسبت هدف ریسک به بازده
            confidence_level (float): سطح اطمینان برای محاسبات آماری
        """
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.target_risk_reward_ratio = target_risk_reward_ratio
        self.confidence_level = confidence_level
        self.current_capital = initial_capital
        self.trade_history = []
        
        logger.info(f"تحلیل‌گر ریسک و بازده با سرمایه {initial_capital} و حداکثر ریسک "
                   f"{max_risk_per_trade * 100}% راه‌اندازی شد")
    
    def calculate_optimal_position_size(self, 
                                      entry_price: float, 
                                      stop_loss_price: float, 
                                      market_volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        محاسبه سایز بهینه پوزیشن بر اساس مدیریت ریسک
        
        Args:
            entry_price (float): قیمت ورود
            stop_loss_price (float): قیمت حد ضرر
            market_volatility (float, optional): نوسانات بازار
            
        Returns:
            dict: اطلاعات سایز پوزیشن
        """
        # محاسبه ریسک هر واحد
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return {
                'position_size': 0.0,
                'units': 0.0,
                'risk_amount': 0.0,
                'risk_percent': 0.0,
                'max_loss': 0.0
            }
        
        # محاسبه تعداد واحدها بر اساس حداکثر ریسک مجاز
        max_risk_amount = self.current_capital * self.max_risk_per_trade
        
        # کاهش ریسک در صورت نوسانات بالا
        if market_volatility is not None:
            # تعدیل ریسک بر اساس نوسانات
            # نوسانات بالاتر = ریسک کمتر
            volatility_multiplier = 1.0
            if market_volatility > 0.03:  # نوسانات بالا (3%+)
                volatility_multiplier = 0.7
            elif market_volatility > 0.02:  # نوسانات متوسط (2-3%)
                volatility_multiplier = 0.85
            
            max_risk_amount *= volatility_multiplier
        
        units = max_risk_amount / risk_per_unit
        position_size = units * entry_price
        
        # محاسبه درصد ریسک واقعی
        actual_risk_percent = (max_risk_amount / self.current_capital) * 100
        
        return {
            'position_size': position_size,
            'units': units,
            'risk_amount': max_risk_amount,
            'risk_percent': actual_risk_percent,
            'max_loss': max_risk_amount
        }
    
    def calculate_risk_reward_ratio(self, 
                                  entry_price: float, 
                                  stop_loss_price: float, 
                                  target_price: float) -> Dict[str, Any]:
        """
        محاسبه نسبت ریسک به بازده
        
        Args:
            entry_price (float): قیمت ورود
            stop_loss_price (float): قیمت حد ضرر
            target_price (float): قیمت هدف
            
        Returns:
            dict: اطلاعات نسبت ریسک به بازده
        """
        # محاسبه ریسک و بازده
        risk = abs(entry_price - stop_loss_price)
        reward = abs(entry_price - target_price)
        
        if risk <= 0:
            return {
                'risk_reward_ratio': 0.0,
                'is_favorable': False,
                'risk_amount': 0.0,
                'reward_amount': reward,
                'recommendation': "نسبت ریسک به بازده قابل محاسبه نیست - حد ضرر نامناسب"
            }
        
        # محاسبه نسبت ریسک به بازده
        risk_reward_ratio = reward / risk
        
        # ارزیابی مطلوب بودن معامله
        is_favorable = risk_reward_ratio >= self.target_risk_reward_ratio
        
        # توصیه‌ها
        if risk_reward_ratio >= 3.0:
            recommendation = "عالی - نسبت ریسک به بازده بسیار مطلوب"
        elif risk_reward_ratio >= 2.0:
            recommendation = "خوب - نسبت ریسک به بازده مطلوب"
        elif risk_reward_ratio >= 1.5:
            recommendation = "قابل قبول - نسبت ریسک به بازده نسبتاً مناسب"
        elif risk_reward_ratio >= 1.0:
            recommendation = "مرزی - نسبت ریسک به بازده حداقل"
        else:
            recommendation = "نامطلوب - نسبت ریسک به بازده پایین، معامله توصیه نمی‌شود"
        
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'is_favorable': is_favorable,
            'risk_amount': risk,
            'reward_amount': reward,
            'recommendation': recommendation
        }
    
    def calculate_multi_targets_strategy(self, 
                                       entry_price: float, 
                                       stop_loss_price: float, 
                                       target_prices: List[float],
                                       position_portions: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        محاسبه استراتژی چند هدفه با خروج تدریجی
        
        Args:
            entry_price (float): قیمت ورود
            stop_loss_price (float): قیمت حد ضرر
            target_prices (list): لیست قیمت‌های هدف
            position_portions (list, optional): لیست درصد پوزیشن برای هر هدف
            
        Returns:
            dict: اطلاعات استراتژی
        """
        if not target_prices:
            return {
                'weighted_risk_reward': 0.0,
                'targets': [],
                'is_favorable': False,
                'max_risk_reward': 0.0,
                'recommendation': "بدون هدف قیمتی"
            }
        
        # تنظیم سهم پوزیشن‌ها
        if position_portions is None:
            # توزیع یکنواخت
            portion_per_target = 1.0 / len(target_prices)
            position_portions = [portion_per_target] * len(target_prices)
        
        # اطمینان از جمع شدن به 1
        if abs(sum(position_portions) - 1.0) > 0.001:
            # نرمال‌سازی
            total = sum(position_portions)
            position_portions = [p / total for p in position_portions]
        
        # محاسبه نسبت ریسک به بازده برای هر هدف
        targets_info = []
        weighted_risk_reward = 0.0
        max_risk_reward = 0.0
        
        for target_price, portion in zip(target_prices, position_portions):
            risk = abs(entry_price - stop_loss_price)
            reward = abs(entry_price - target_price)
            
            if risk > 0:
                risk_reward = reward / risk
                max_risk_reward = max(max_risk_reward, risk_reward)
            else:
                risk_reward = 0.0
            
            weighted_risk_reward += risk_reward * portion
            
            targets_info.append({
                'target_price': target_price,
                'portion': portion,
                'risk_reward': risk_reward,
                'profit_percent': ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            })
        
        # ارزیابی مطلوب بودن استراتژی
        is_favorable = weighted_risk_reward >= self.target_risk_reward_ratio
        
        # توصیه‌ها
        if weighted_risk_reward >= 3.0:
            recommendation = "عالی - استراتژی چند هدفه با ریسک به بازده بسیار مطلوب"
        elif weighted_risk_reward >= 2.0:
            recommendation = "خوب - استراتژی چند هدفه با ریسک به بازده مطلوب"
        elif weighted_risk_reward >= 1.5:
            recommendation = "قابل قبول - استراتژی چند هدفه با ریسک به بازده نسبتاً مناسب"
        elif weighted_risk_reward >= 1.0:
            recommendation = "مرزی - استراتژی چند هدفه با ریسک به بازده حداقل"
        else:
            recommendation = "نامطلوب - استراتژی چند هدفه با ریسک به بازده پایین، معامله توصیه نمی‌شود"
        
        return {
            'weighted_risk_reward': weighted_risk_reward,
            'targets': targets_info,
            'is_favorable': is_favorable,
            'max_risk_reward': max_risk_reward,
            'recommendation': recommendation
        }
    
    def calculate_break_even_points(self, 
                                  entry_price: float, 
                                  stop_loss_price: float, 
                                  trade_costs: float = 0.001) -> Dict[str, Any]:
        """
        محاسبه نقاط سر به سر برای بهینه‌سازی مدیریت پوزیشن
        
        Args:
            entry_price (float): قیمت ورود
            stop_loss_price (float): قیمت حد ضرر
            trade_costs (float): هزینه‌های معامله (کارمزد و اسپرد)
            
        Returns:
            dict: اطلاعات نقاط سر به سر
        """
        # محاسبه ریسک و هزینه‌ها
        risk = abs(entry_price - stop_loss_price)
        
        # هزینه‌های ورود و خروج
        entry_cost = entry_price * trade_costs
        exit_cost = entry_price * trade_costs  # تخمین
        total_costs = entry_cost + exit_cost
        
        # محاسبه نقطه سر به سر (جایی که ضرر = 0)
        if entry_price > stop_loss_price:  # پوزیشن خرید
            break_even_price = entry_price + total_costs
            move_to_break_even_price = entry_price + risk * 0.5
        else:  # پوزیشن فروش
            break_even_price = entry_price - total_costs
            move_to_break_even_price = entry_price - risk * 0.5
        
        # محاسبه درصد فاصله
        break_even_percent = abs((break_even_price / entry_price) - 1) * 100
        move_to_break_even_percent = abs((move_to_break_even_price / entry_price) - 1) * 100
        
        return {
            'break_even_price': break_even_price,
            'break_even_percent': break_even_percent,
            'move_to_break_even_price': move_to_break_even_price,
            'move_to_break_even_percent': move_to_break_even_percent,
            'total_costs_percent': (total_costs / entry_price) * 100
        }
    
    def calculate_stop_loss_types(self, 
                                entry_price: float, 
                                initial_stop_loss: float, 
                                high_price: float,
                                low_price: float,
                                atr: Optional[float] = None) -> Dict[str, Any]:
        """
        محاسبه انواع مختلف حد ضرر
        
        Args:
            entry_price (float): قیمت ورود
            initial_stop_loss (float): حد ضرر اولیه
            high_price (float): قیمت بالا
            low_price (float): قیمت پایین
            atr (float, optional): شاخص ATR
            
        Returns:
            dict: انواع حد ضرر
        """
        result = {
            'initial_stop_loss': initial_stop_loss,
            'initial_risk_percent': abs((initial_stop_loss / entry_price) - 1) * 100
        }
        
        # تعیین نوع پوزیشن
        is_long = entry_price > initial_stop_loss
        
        # محاسبه حد ضرر ATR
        if atr is not None:
            if is_long:
                atr_stop_loss = entry_price - (atr * 2)
            else:
                atr_stop_loss = entry_price + (atr * 2)
            
            result['atr_stop_loss'] = atr_stop_loss
            result['atr_risk_percent'] = abs((atr_stop_loss / entry_price) - 1) * 100
        
        # محاسبه حد ضرر نوسانی (Swing)
        if is_long:
            swing_stop_loss = low_price * 0.99  # 1% زیر کمترین قیمت
        else:
            swing_stop_loss = high_price * 1.01  # 1% بالای بالاترین قیمت
        
        result['swing_stop_loss'] = swing_stop_loss
        result['swing_risk_percent'] = abs((swing_stop_loss / entry_price) - 1) * 100
        
        # محاسبه حد ضرر درصدی
        percent_risk = 0.02  # ریسک 2%
        if is_long:
            percent_stop_loss = entry_price * (1 - percent_risk)
        else:
            percent_stop_loss = entry_price * (1 + percent_risk)
        
        result['percent_stop_loss'] = percent_stop_loss
        result['percent_risk_percent'] = percent_risk * 100
        
        return result
    
    def calculate_expectancy(self, win_rate: float, avg_win: float, avg_loss: float) -> Dict[str, Any]:
        """
        محاسبه امید ریاضی سیستم معاملاتی
        
        Args:
            win_rate (float): نرخ برد (0-1)
            avg_win (float): میانگین سود هر معامله موفق
            avg_loss (float): میانگین ضرر هر معامله ناموفق
            
        Returns:
            dict: اطلاعات امید ریاضی
        """
        if avg_loss == 0:
            return {
                'expectancy': 0.0,
                'expectancy_per_dollar': 0.0,
                'profit_factor': 0.0,
                'expectancy_rating': "غیرقابل محاسبه - میانگین ضرر صفر است"
            }
        
        # محاسبه امید ریاضی
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # محاسبه امید ریاضی به ازای هر دلار در ریسک
        expectancy_per_dollar = expectancy / avg_loss
        
        # محاسبه ضریب سود
        profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss)
        
        # رتبه‌بندی امید ریاضی
        if expectancy_per_dollar >= 0.5:
            expectancy_rating = "عالی - سیستم معاملاتی بسیار سودآور"
        elif expectancy_per_dollar >= 0.3:
            expectancy_rating = "خوب - سیستم معاملاتی سودآور"
        elif expectancy_per_dollar >= 0.1:
            expectancy_rating = "قابل قبول - سیستم معاملاتی با سوددهی مثبت"
        elif expectancy_per_dollar >= 0:
            expectancy_rating = "مرزی - سیستم معاملاتی در آستانه سوددهی"
        else:
            expectancy_rating = "نامطلوب - سیستم معاملاتی زیان‌ده"
        
        return {
            'expectancy': expectancy,
            'expectancy_per_dollar': expectancy_per_dollar,
            'profit_factor': profit_factor,
            'expectancy_rating': expectancy_rating
        }
    
    def simulate_capital_growth(self, 
                              win_rate: float, 
                              avg_win: float, 
                              avg_loss: float, 
                              num_trades: int, 
                              risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """
        شبیه‌سازی رشد سرمایه بر اساس پارامترهای سیستم معاملاتی
        
        Args:
            win_rate (float): نرخ برد (0-1)
            avg_win (float): میانگین سود هر معامله موفق (به درصد)
            avg_loss (float): میانگین ضرر هر معامله ناموفق (به درصد)
            num_trades (int): تعداد معاملات
            risk_per_trade (float): ریسک هر معامله (به درصد)
            
        Returns:
            dict: نتایج شبیه‌سازی
        """
        if win_rate < 0 or win_rate > 1:
            return {
                'error': 'نرخ برد باید بین 0 و 1 باشد',
                'capital_history': []
            }
        
        # تبدیل درصدها به ضرایب
        avg_win_factor = avg_win / 100
        avg_loss_factor = avg_loss / 100
        risk_factor = risk_per_trade
        
        # شبیه‌سازی رشد سرمایه
        capital = self.initial_capital
        capital_history = [capital]
        profit_history = []
        drawdowns = []
        max_capital = capital
        
        for _ in range(num_trades):
            # تعیین نتیجه معامله
            is_win = np.random.random() < win_rate
            
            # محاسبه تغییر سرمایه
            if is_win:
                # سود
                trade_amount = capital * risk_factor * avg_win_factor
                capital += trade_amount
                profit_history.append(trade_amount)
            else:
                # ضرر
                trade_amount = capital * risk_factor * avg_loss_factor
                capital -= trade_amount
                profit_history.append(-trade_amount)
            
            # ثبت تاریخچه
            capital_history.append(capital)
            
            # محاسبه افت سرمایه
            if capital > max_capital:
                max_capital = capital
            
            current_drawdown = (max_capital - capital) / max_capital * 100
            drawdowns.append(current_drawdown)
        
        # محاسبه آمار
        final_capital = capital_history[-1]
        total_return = ((final_capital / self.initial_capital) - 1) * 100
        avg_return_per_trade = total_return / num_trades
        
        # محاسبه حداکثر افت سرمایه
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # محاسبه نسبت شارپ ساده
        profit_array = np.array(profit_history)
        sharpe_ratio = 0.0
        
        if len(profit_array) > 0 and np.std(profit_array) > 0:
            sharpe_ratio = np.mean(profit_array) / np.std(profit_array) * np.sqrt(252 / num_trades)  # فرض 252 روز معاملاتی در سال
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'avg_return_per_trade': avg_return_per_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'capital_history': capital_history,
            'num_trades': num_trades
        }
    
    def calculate_martingale_risks(self, 
                                 initial_position: float, 
                                 multiplier: float, 
                                 max_iterations: int) -> Dict[str, Any]:
        """
        محاسبه ریسک‌های استراتژی مارتینگل (افزایش سایز با هر ضرر)
        
        Args:
            initial_position (float): سایز اولیه پوزیشن
            multiplier (float): ضریب افزایش سایز
            max_iterations (int): حداکثر تعداد تکرار
            
        Returns:
            dict: اطلاعات ریسک‌های مارتینگل
        """
        if max_iterations <= 0 or multiplier <= 1:
            return {
                'error': 'پارامترهای نامعتبر',
                'iterations': []
            }
        
        iterations = []
        total_capital_used = initial_position
        current_position = initial_position
        
        for i in range(max_iterations):
            iterations.append({
                'iteration': i + 1,
                'position_size': current_position,
                'cumulative_capital': total_capital_used,
                'percent_of_capital': (total_capital_used / self.current_capital) * 100
            })
            
            # محاسبه سایز بعدی
            current_position *= multiplier
            total_capital_used += current_position
        
        # ارزیابی ریسک
        capital_depletion = (total_capital_used / self.current_capital) * 100
        
        if capital_depletion > 100:
            risk_assessment = "بسیار خطرناک - استراتژی مارتینگل به سرمایه کامل نیاز دارد"
        elif capital_depletion > 75:
            risk_assessment = "خطرناک - استراتژی مارتینگل بیش از 75% سرمایه را به خطر می‌اندازد"
        elif capital_depletion > 50:
            risk_assessment = "پرریسک - استراتژی مارتینگل بیش از 50% سرمایه را به خطر می‌اندازد"
        elif capital_depletion > 25:
            risk_assessment = "ریسک متوسط - استراتژی مارتینگل 25-50% سرمایه را به خطر می‌اندازد"
        else:
            risk_assessment = "ریسک پایین - استراتژی مارتینگل کمتر از 25% سرمایه را به خطر می‌اندازد"
        
        return {
            'iterations': iterations,
            'total_capital_required': total_capital_used,
            'capital_depletion_percent': capital_depletion,
            'risk_assessment': risk_assessment
        }
    
    def calculate_trailing_stop_scenarios(self, 
                                        entry_price: float, 
                                        initial_stop_loss: float, 
                                        atr: Optional[float] = None) -> Dict[str, Any]:
        """
        محاسبه سناریوهای مختلف حد ضرر متحرک
        
        Args:
            entry_price (float): قیمت ورود
            initial_stop_loss (float): حد ضرر اولیه
            atr (float, optional): شاخص ATR
            
        Returns:
            dict: سناریوهای حد ضرر متحرک
        """
        # تعیین نوع پوزیشن
        is_long = entry_price > initial_stop_loss
        
        # محاسبه فاصله حد ضرر اولیه
        initial_stop_distance = abs(entry_price - initial_stop_loss)
        initial_stop_percent = (initial_stop_distance / entry_price) * 100
        
        # سناریوهای حد ضرر متحرک
        scenarios = []
        
        # سناریو 1: حد ضرر متحرک درصدی
        percent_factors = [0.5, 1.0, 1.5]
        for factor in percent_factors:
            trail_percent = initial_stop_percent * factor
            
            if is_long:
                activation_price = entry_price * (1 + trail_percent / 100)
            else:
                activation_price = entry_price * (1 - trail_percent / 100)
            
            scenarios.append({
                'type': 'percent',
                'trail_percent': trail_percent,
                'activation_price': activation_price,
                'activation_percent': abs((activation_price / entry_price) - 1) * 100,
                'description': f"حد ضرر متحرک درصدی با فاصله {trail_percent:.2f}% از قیمت جاری"
            })
        
        # سناریو 2: حد ضرر متحرک ATR
        if atr is not None:
            atr_factors = [1.0, 2.0, 3.0]
            for factor in atr_factors:
                trail_distance = atr * factor
                
                if is_long:
                    activation_price = entry_price + trail_distance
                else:
                    activation_price = entry_price - trail_distance
                
                scenarios.append({
                    'type': 'atr',
                    'trail_atr_factor': factor,
                    'trail_atr_value': trail_distance,
                    'activation_price': activation_price,
                    'activation_percent': abs((activation_price / entry_price) - 1) * 100,
                    'description': f"حد ضرر متحرک ATR با فاصله {factor} برابر ATR از قیمت جاری"
                })
        
        # سناریو 3: حد ضرر متحرک نقطه‌ای (به نقطه سر به سر)
        if is_long:
            activation_price = entry_price * 1.03  # 3% بالاتر از نقطه ورود
        else:
            activation_price = entry_price * 0.97  # 3% پایین‌تر از نقطه ورود
        
        scenarios.append({
            'type': 'breakeven',
            'activation_price': activation_price,
            'activation_percent': abs((activation_price / entry_price) - 1) * 100,
            'new_stop_price': entry_price,
            'description': "حد ضرر متحرک به نقطه سر به سر پس از 3% حرکت به نفع معامله"
        })
        
        return {
            'is_long': is_long,
            'initial_stop_loss': initial_stop_loss,
            'initial_stop_percent': initial_stop_percent,
            'scenarios': scenarios
        }
    
    def evaluate_multiple_time_frame_alignment(self, 
                                             higher_tf_bullish: bool, 
                                             current_tf_bullish: bool, 
                                             lower_tf_bullish: bool) -> Dict[str, Any]:
        """
        ارزیابی همسویی تایم‌فریم‌های مختلف
        
        Args:
            higher_tf_bullish (bool): آیا تایم‌فریم بالاتر صعودی است
            current_tf_bullish (bool): آیا تایم‌فریم فعلی صعودی است
            lower_tf_bullish (bool): آیا تایم‌فریم پایین‌تر صعودی است
            
        Returns:
            dict: ارزیابی همسویی تایم‌فریم‌ها
        """
        # محاسبه میزان همسویی
        alignment_count = (higher_tf_bullish + current_tf_bullish + lower_tf_bullish)
        
        if alignment_count == 3:
            # همه تایم‌فریم‌ها صعودی
            alignment_score = 1.0
            recommendation = "بسیار خوب - همه تایم‌فریم‌ها همسو هستند (صعودی)"
            trade_direction = "BUY"
        elif alignment_count == 0:
            # همه تایم‌فریم‌ها نزولی
            alignment_score = 1.0
            recommendation = "بسیار خوب - همه تایم‌فریم‌ها همسو هستند (نزولی)"
            trade_direction = "SELL"
        elif alignment_count == 2:
            # دو تایم‌فریم همسو هستند
            alignment_score = 0.7
            
            if higher_tf_bullish and current_tf_bullish:
                recommendation = "خوب - تایم‌فریم‌های بالاتر و فعلی همسو هستند (صعودی)"
                trade_direction = "BUY"
            elif current_tf_bullish and lower_tf_bullish:
                recommendation = "متوسط - تایم‌فریم‌های فعلی و پایین‌تر همسو هستند (صعودی)"
                trade_direction = "BUY"
            elif higher_tf_bullish and lower_tf_bullish:
                recommendation = "ضعیف - تایم‌فریم‌های بالاتر و پایین‌تر همسو هستند اما تایم‌فریم فعلی ناهمسو است"
                trade_direction = "NEUTRAL"
            elif not higher_tf_bullish and not current_tf_bullish:
                recommendation = "خوب - تایم‌فریم‌های بالاتر و فعلی همسو هستند (نزولی)"
                trade_direction = "SELL"
            elif not current_tf_bullish and not lower_tf_bullish:
                recommendation = "متوسط - تایم‌فریم‌های فعلی و پایین‌تر همسو هستند (نزولی)"
                trade_direction = "SELL"
            else:
                recommendation = "ضعیف - تایم‌فریم‌های بالاتر و پایین‌تر همسو هستند (نزولی) اما تایم‌فریم فعلی ناهمسو است"
                trade_direction = "NEUTRAL"
        else:  # alignment_count == 1
            # فقط یک تایم‌فریم متفاوت است
            alignment_score = 0.3
            
            if current_tf_bullish:
                recommendation = "ضعیف - فقط تایم‌فریم فعلی صعودی است و تایم‌فریم‌های دیگر ناهمسو هستند"
                trade_direction = "NEUTRAL"
            elif higher_tf_bullish:
                recommendation = "ضعیف - فقط تایم‌فریم بالاتر صعودی است"
                trade_direction = "NEUTRAL"
            elif lower_tf_bullish:
                recommendation = "ضعیف - فقط تایم‌فریم پایین‌تر صعودی است"
                trade_direction = "NEUTRAL"
            else:
                # این حالت نباید اتفاق بیفتد (alignment_count باید 2 باشد)
                recommendation = "نامشخص"
                trade_direction = "NEUTRAL"
        
        return {
            'alignment_score': alignment_score,
            'recommendation': recommendation,
            'trade_direction': trade_direction,
            'higher_tf_bullish': higher_tf_bullish,
            'current_tf_bullish': current_tf_bullish,
            'lower_tf_bullish': lower_tf_bullish
        }


def calculate_position_size(capital: float, entry_price: float, stop_loss_price: float, risk_percent: float = 2.0) -> Dict[str, Any]:
    """
    محاسبه سایز پوزیشن بر اساس مدیریت ریسک
    
    Args:
        capital (float): سرمایه کل
        entry_price (float): قیمت ورود
        stop_loss_price (float): قیمت حد ضرر
        risk_percent (float): درصد ریسک سرمایه (پیش‌فرض 2%)
        
    Returns:
        dict: اطلاعات سایز پوزیشن
    """
    # تبدیل درصد به اعشار
    risk_ratio = risk_percent / 100
    
    # محاسبه ریسک هر واحد
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit <= 0:
        return {
            'position_size': 0.0,
            'units': 0.0,
            'max_risk_amount': 0.0
        }
    
    # محاسبه حداکثر مقدار ریسک
    max_risk_amount = capital * risk_ratio
    
    # محاسبه تعداد واحدها
    units = max_risk_amount / risk_per_unit
    position_size = units * entry_price
    
    return {
        'position_size': position_size,
        'units': units,
        'max_risk_amount': max_risk_amount,
        'risk_percent': risk_percent,
        'risk_per_unit': risk_per_unit
    }


def calculate_risk_reward_scenarios(entry_price: float, stop_loss_price: float, targets: List[float]) -> Dict[str, Any]:
    """
    محاسبه سناریوهای ریسک و بازده برای چندین هدف قیمتی
    
    Args:
        entry_price (float): قیمت ورود
        stop_loss_price (float): قیمت حد ضرر
        targets (list): لیست اهداف قیمتی
        
    Returns:
        dict: سناریوهای ریسک و بازده
    """
    if not targets:
        return {
            'error': 'لیست اهداف قیمتی خالی است',
            'scenarios': []
        }
    
    # محاسبه ریسک
    risk = abs(entry_price - stop_loss_price)
    risk_percent = abs((entry_price - stop_loss_price) / entry_price) * 100
    
    if risk <= 0:
        return {
            'error': 'ریسک صفر یا منفی',
            'scenarios': []
        }
    
    # محاسبه سناریوها
    scenarios = []
    
    # تعیین نوع پوزیشن
    is_long = entry_price > stop_loss_price
    
    for i, target in enumerate(targets):
        # محاسبه بازده
        reward = abs(entry_price - target)
        reward_percent = abs((entry_price - target) / entry_price) * 100
        
        # محاسبه نسبت ریسک به بازده
        risk_reward_ratio = reward / risk
        
        # بررسی جهت هدف
        if (is_long and target > entry_price) or (not is_long and target < entry_price):
            target_aligned = True
        else:
            target_aligned = False
        
        # ارزیابی
        if risk_reward_ratio >= 3.0:
            assessment = "عالی"
        elif risk_reward_ratio >= 2.0:
            assessment = "خوب"
        elif risk_reward_ratio >= 1.0:
            assessment = "قابل قبول"
        else:
            assessment = "ضعیف"
        
        scenarios.append({
            'target_number': i + 1,
            'target_price': target,
            'reward': reward,
            'reward_percent': reward_percent,
            'risk_reward_ratio': risk_reward_ratio,
            'assessment': assessment,
            'target_aligned': target_aligned
        })
    
    # محاسبه سناریوی تلفیقی
    if len(targets) > 1:
        # سناریوی خروج تدریجی از پوزیشن
        position_portions = []
        for i in range(len(targets)):
            if i == 0:
                # اولین هدف - 50% پوزیشن
                position_portions.append(0.5)
            elif i == len(targets) - 1:
                # آخرین هدف - باقیمانده پوزیشن
                position_portions.append(1.0 - sum(position_portions))
            else:
                # اهداف میانی - تقسیم مساوی باقیمانده
                portion = (1.0 - 0.5) / (len(targets) - 1)
                position_portions.append(portion)
        
        # محاسبه میانگین وزنی RR
        weighted_rr = 0
        
        for scenario, portion in zip(scenarios, position_portions):
            weighted_rr += scenario['risk_reward_ratio'] * portion
        
        combined_scenario = {
            'weighted_risk_reward': weighted_rr,
            'position_portions': position_portions,
            'is_favorable': weighted_rr >= 2.0
        }
    else:
        combined_scenario = {
            'weighted_risk_reward': scenarios[0]['risk_reward_ratio'],
            'position_portions': [1.0],
            'is_favorable': scenarios[0]['risk_reward_ratio'] >= 2.0
        }
    
    return {
        'risk': risk,
        'risk_percent': risk_percent,
        'scenarios': scenarios,
        'combined_scenario': combined_scenario
    }