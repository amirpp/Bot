"""
ماژول رابط کاربری برای سیستم تحلیل ریسک و بازده

این ماژول شامل توابع مورد نیاز برای نمایش رابط کاربری سیستم تحلیل ریسک و بازده است.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from risk_reward_analyzer import (
    RiskRewardAnalyzer, 
    calculate_position_size, 
    calculate_risk_reward_scenarios
)

def render_risk_reward_ui(df: pd.DataFrame, symbol: str, timeframe: str):
    """
    رندر صفحه تحلیل ریسک و بازده
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
    """
    st.header("📊 تحلیل ریسک و بازده معاملاتی", divider="rainbow")
    
    st.markdown("""
    این بخش به شما کمک می‌کند تا ریسک و بازده معاملات را به صورت دقیق و هوشمند تحلیل کنید و بهترین تصمیمات را برای مدیریت سرمایه بگیرید.
    
    با استفاده از این ابزارها می‌توانید سایز پوزیشن بهینه، نقاط ورود و خروج، حد ضرر و حد سود مناسب را محاسبه کنید.
    """)
    
    # تب‌های صفحه ریسک و بازده
    tab1, tab2, tab3, tab4 = st.tabs([
        "محاسبه‌گر سایز پوزیشن", 
        "تحلیل ریسک/بازده", 
        "شبیه‌ساز رشد سرمایه",
        "استراتژی‌های حد ضرر"
    ])
    
    with tab1:
        render_position_size_calculator(df, symbol)
    
    with tab2:
        render_risk_reward_calculator(df, symbol)
    
    with tab3:
        render_capital_growth_simulator(df, symbol)
    
    with tab4:
        render_stop_loss_strategies(df, symbol)

def render_position_size_calculator(df: pd.DataFrame, symbol: str):
    """
    رندر محاسبه‌گر سایز پوزیشن
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
    """
    st.subheader("محاسبه‌گر سایز پوزیشن", anchor=False)
    
    st.markdown("""
    با استفاده از این ابزار، سایز پوزیشن بهینه را بر اساس سرمایه و میزان ریسک مورد نظر خود محاسبه کنید.
    """)
    
    # دریافت اطلاعات از کاربر
    col1, col2 = st.columns(2)
    
    with col1:
        capital = st.number_input(
            "سرمایه کل (USDT)", 
            min_value=100.0, 
            max_value=1000000.0, 
            value=1000.0, 
            step=100.0,
            key="position_size_capital"
        )
        
        risk_percent = st.slider(
            "میزان ریسک (درصد از سرمایه)", 
            min_value=0.5, 
            max_value=5.0, 
            value=2.0, 
            step=0.5,
            key="position_size_risk"
        )
    
    with col2:
        # دریافت قیمت فعلی از دیتافریم
        if not df.empty:
            current_price = df['close'].iloc[-1]
        else:
            current_price = 30000.0
        
        entry_price = st.number_input(
            "قیمت ورود", 
            min_value=0.0001, 
            max_value=float(current_price * 10), 
            value=float(current_price),
            format="%.2f",
            key="position_size_entry"
        )
        
        stop_loss_price = st.number_input(
            "قیمت حد ضرر", 
            min_value=0.0001, 
            max_value=float(entry_price * 2), 
            value=float(entry_price * 0.95),
            format="%.2f",
            key="position_size_stop"
        )
    
    st.markdown("---")
    
    # محاسبه سایز پوزیشن
    position_result = calculate_position_size(capital, entry_price, stop_loss_price, risk_percent)
    
    # نمایش نتایج محاسبات
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="سایز پوزیشن",
            value=f"{position_result['position_size']:.2f} USDT"
        )
    
    with col2:
        st.metric(
            label="تعداد واحد",
            value=f"{position_result['units']:.4f} {symbol.split('/')[0]}"
        )
    
    with col3:
        st.metric(
            label="حداکثر ریسک",
            value=f"{position_result['max_risk_amount']:.2f} USDT"
        )
    
    with col4:
        risk_per_unit = position_result.get('risk_per_unit', 0)
        st.metric(
            label="ریسک هر واحد",
            value=f"{risk_per_unit:.2f} USDT"
        )
    
    # نمایش گراف توزیع سرمایه
    risk_amount = position_result['max_risk_amount']
    position_amount = position_result['position_size']
    remainder = capital - position_amount
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=['سایز پوزیشن', 'باقی‌مانده سرمایه'],
        values=[position_amount, remainder],
        marker=dict(colors=['#3366CC', '#DDDDDD']),
        textinfo='percent+label',
        hole=0.5
    ))
    
    fig.update_layout(
        title="توزیع سرمایه",
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # توصیه‌ها
    st.markdown("### توصیه‌های مدیریت ریسک")
    
    position_percent = (position_amount / capital) * 100
    
    if position_percent > 50:
        st.warning(f"سایز پوزیشن شما {position_percent:.1f}% از کل سرمایه است که می‌تواند ریسک بالایی داشته باشد. توصیه می‌شود این مقدار را کاهش دهید.")
    elif position_percent > 20:
        st.info(f"سایز پوزیشن شما {position_percent:.1f}% از کل سرمایه است که برای معاملات با اطمینان بالا مناسب است.")
    else:
        st.success(f"سایز پوزیشن شما {position_percent:.1f}% از کل سرمایه است که مدیریت ریسک محتاطانه‌ای را نشان می‌دهد.")
    
    # محاسبه نسبت ریسک به بازده برای چند هدف
    st.markdown("### نقاط هدف و نسبت ریسک به بازده")
    
    default_targets = []
    if not df.empty:
        price_range = df['high'].max() - df['low'].min()
        current_price = df['close'].iloc[-1]
        
        if entry_price > stop_loss_price:  # پوزیشن خرید
            default_targets = [
                entry_price * 1.02,  # هدف اول: 2% بالاتر از قیمت ورود
                entry_price * 1.05,  # هدف دوم: 5% بالاتر از قیمت ورود
                entry_price * 1.10   # هدف سوم: 10% بالاتر از قیمت ورود
            ]
        else:  # پوزیشن فروش
            default_targets = [
                entry_price * 0.98,  # هدف اول: 2% پایین‌تر از قیمت ورود
                entry_price * 0.95,  # هدف دوم: 5% پایین‌تر از قیمت ورود
                entry_price * 0.90   # هدف سوم: 10% پایین‌تر از قیمت ورود
            ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target1 = st.number_input(
            "هدف اول", 
            min_value=0.0001, 
            max_value=float(entry_price * 2),
            value=float(default_targets[0]) if default_targets else float(entry_price * 1.02),
            format="%.2f",
            key="position_size_target1"
        )
    
    with col2:
        target2 = st.number_input(
            "هدف دوم", 
            min_value=0.0001, 
            max_value=float(entry_price * 2),
            value=float(default_targets[1]) if default_targets else float(entry_price * 1.05),
            format="%.2f",
            key="position_size_target2"
        )
    
    with col3:
        target3 = st.number_input(
            "هدف سوم", 
            min_value=0.0001, 
            max_value=float(entry_price * 2),
            value=float(default_targets[2]) if default_targets else float(entry_price * 1.10),
            format="%.2f",
            key="position_size_target3"
        )
    
    targets = [target1, target2, target3]
    
    # محاسبه سناریوهای ریسک و بازده
    risk_reward_result = calculate_risk_reward_scenarios(entry_price, stop_loss_price, targets)
    
    # نمایش جدول سناریوها
    if 'error' not in risk_reward_result:
        scenarios = risk_reward_result['scenarios']
        
        scenario_rows = []
        for scenario in scenarios:
            alignment_icon = "✅" if scenario['target_aligned'] else "❌"
            
            scenario_rows.append({
                'هدف': f"هدف {scenario['target_number']}",
                'قیمت': f"{scenario['target_price']:.2f}",
                'سود (درصد)': f"{scenario['reward_percent']:.2f}%",
                'نسبت ریسک/بازده': f"{scenario['risk_reward_ratio']:.2f}",
                'ارزیابی': f"{scenario['assessment']} {alignment_icon}"
            })
        
        scenario_df = pd.DataFrame(scenario_rows)
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        
        # نمایش استراتژی ترکیبی
        combined = risk_reward_result['combined_scenario']
        
        st.markdown(f"""
        **استراتژی ترکیبی خروج تدریجی:**
        - هدف اول: 50% از پوزیشن
        - هدف دوم: 25% از پوزیشن
        - هدف سوم: 25% از پوزیشن
        
        **نسبت ریسک به بازده ترکیبی:** {combined['weighted_risk_reward']:.2f}
        """)
        
        if combined['is_favorable']:
            st.success("استراتژی ترکیبی مطلوب ارزیابی می‌شود.")
        else:
            st.warning("استراتژی ترکیبی چندان مطلوب نیست. توصیه می‌شود در هدف‌گذاری قیمتی تجدید نظر کنید.")
        
        # نمودار ویژوال ورود/خروج
        fig = go.Figure()
        
        # تعیین نوع پوزیشن
        is_long = entry_price > stop_loss_price
        
        # مرتب‌سازی نقاط بر اساس نوع پوزیشن
        points = [stop_loss_price, entry_price] + targets
        if is_long:
            points.sort()
        else:
            points.sort(reverse=True)
        
        # تعیین رنگ‌ها
        colors = ['red', 'blue', 'green', 'green', 'green']
        
        # ایجاد نمودار
        fig.add_trace(go.Scatter(
            x=list(range(len(points))),
            y=points,
            mode='lines+markers+text',
            marker=dict(
                size=12,
                color=colors,
                symbol=['triangle-down', 'circle', 'triangle-up', 'triangle-up', 'triangle-up']
            ),
            text=['حد ضرر', 'ورود', 'هدف 1', 'هدف 2', 'هدف 3'],
            textposition="top center"
        ))
        
        # تنظیمات نمودار
        fig.update_layout(
            title="نقاط ورود، حد ضرر و اهداف قیمتی",
            xaxis=dict(
                title="موقعیت",
                showticklabels=False
            ),
            yaxis=dict(title="قیمت"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error(risk_reward_result['error'])

def render_risk_reward_calculator(df: pd.DataFrame, symbol: str):
    """
    رندر محاسبه‌گر نسبت ریسک به بازده
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
    """
    st.subheader("تحلیل جامع ریسک/بازده", anchor=False)
    
    st.markdown("""
    با استفاده از این بخش، می‌توانید تحلیل جامعی از ریسک و بازده معاملات خود داشته باشید و استراتژی‌های پیشرفته را بررسی کنید.
    """)
    
    # ایجاد تحلیل‌گر ریسک و بازده
    analyzer = RiskRewardAnalyzer(
        initial_capital=1000.0,
        max_risk_per_trade=0.02,
        target_risk_reward_ratio=2.0
    )
    
    # دریافت اطلاعات ورودی از کاربر
    col1, col2 = st.columns(2)
    
    with col1:
        capital = st.number_input(
            "سرمایه کل (USDT)", 
            min_value=100.0, 
            max_value=1000000.0, 
            value=1000.0, 
            step=100.0,
            key="rr_capital"
        )
        
        risk_percent = st.slider(
            "میزان ریسک (درصد از سرمایه)", 
            min_value=0.1, 
            max_value=5.0, 
            value=2.0, 
            step=0.1,
            key="rr_risk"
        )
        
        target_rr = st.slider(
            "نسبت ریسک به بازده هدف", 
            min_value=1.0, 
            max_value=5.0, 
            value=2.0, 
            step=0.5,
            key="rr_target"
        )
    
    with col2:
        # دریافت قیمت فعلی از دیتافریم
        if not df.empty:
            current_price = df['close'].iloc[-1]
            atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            atr = atr.mean() / current_price
        else:
            current_price = 30000.0
            atr = current_price * 0.02
        
        st.markdown(f"### تنظیمات پیشرفته")
        
        trade_type = st.selectbox(
            "نوع معامله",
            options=["خرید (Long)", "فروش (Short)"],
            index=0,
            key="rr_trade_type"
        )
        
        confidence_level = st.slider(
            "سطح اطمینان به معامله (٪)", 
            min_value=50, 
            max_value=95, 
            value=80, 
            step=5,
            key="rr_confidence"
        )
        
        use_multiple_targets = st.checkbox(
            "استفاده از چندین هدف قیمتی", 
            value=True,
            key="rr_multiple_targets"
        )
    
    st.markdown("---")
    
    # محاسبه پارامترهای معامله
    is_long = trade_type == "خرید (Long)"
    
    entry_price = current_price
    
    if is_long:
        stop_loss_price = entry_price * (1 - (atr * 2))
    else:
        stop_loss_price = entry_price * (1 + (atr * 2))
    
    # تنظیم آنالایزر
    analyzer.initial_capital = capital
    analyzer.max_risk_per_trade = risk_percent / 100
    analyzer.target_risk_reward_ratio = target_rr
    analyzer.confidence_level = confidence_level / 100
    
    # محاسبه سایز پوزیشن
    position_size_result = analyzer.calculate_optimal_position_size(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        market_volatility=atr
    )
    
    # محاسبه اهداف قیمتی بر اساس نسبت ریسک به بازده هدف
    risk_amount = abs(entry_price - stop_loss_price)
    
    if is_long:
        target1 = entry_price + (risk_amount * 1.0)  # نسبت 1:1
        target2 = entry_price + (risk_amount * 2.0)  # نسبت 1:2
        target3 = entry_price + (risk_amount * 3.0)  # نسبت 1:3
    else:
        target1 = entry_price - (risk_amount * 1.0)  # نسبت 1:1
        target2 = entry_price - (risk_amount * 2.0)  # نسبت 1:2
        target3 = entry_price - (risk_amount * 3.0)  # نسبت 1:3
    
    # محاسبه نقاط سر به سر
    break_even_result = analyzer.calculate_break_even_points(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        trade_costs=0.001  # کارمزد 0.1%
    )
    
    # محاسبه استراتژی چند هدفه
    multi_target_result = analyzer.calculate_multi_targets_strategy(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        target_prices=[target1, target2, target3],
        position_portions=[0.5, 0.25, 0.25]  # خروج تدریجی
    )
    
    # نمایش نتایج در چهار بخش مجزا
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### پارامترهای معامله")
        
        # نمایش پارامترهای معامله
        params_df = pd.DataFrame([
            {"پارامتر": "نوع معامله", "مقدار": "خرید (Long)" if is_long else "فروش (Short)"},
            {"پارامتر": "قیمت ورود", "مقدار": f"{entry_price:.2f}"},
            {"پارامتر": "قیمت حد ضرر", "مقدار": f"{stop_loss_price:.2f}"},
            {"پارامتر": "فاصله حد ضرر", "مقدار": f"{abs((stop_loss_price / entry_price - 1) * 100):.2f}%"},
            {"پارامتر": "سایز پوزیشن", "مقدار": f"{position_size_result['position_size']:.2f} USDT"},
            {"پارامتر": "ریسک واقعی", "مقدار": f"{position_size_result['risk_percent']:.2f}%"}
        ])
        
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### نقاط سر به سر")
        
        # نمایش نقاط سر به سر
        break_even_df = pd.DataFrame([
            {"پارامتر": "نقطه سر به سر", "مقدار": f"{break_even_result['break_even_price']:.2f}"},
            {"پارامتر": "فاصله تا نقطه سر به سر", "مقدار": f"{break_even_result['break_even_percent']:.2f}%"},
            {"پارامتر": "نقطه انتقال حد ضرر", "مقدار": f"{break_even_result['move_to_break_even_price']:.2f}"},
            {"پارامتر": "فاصله تا نقطه انتقال", "مقدار": f"{break_even_result['move_to_break_even_percent']:.2f}%"},
            {"پارامتر": "کل هزینه‌های معامله", "مقدار": f"{break_even_result['total_costs_percent']:.2f}%"}
        ])
        
        st.dataframe(break_even_df, use_container_width=True, hide_index=True)
    
    st.markdown("### استراتژی چند هدفه با خروج تدریجی")
    
    # نمایش اطلاعات استراتژی چند هدفه
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="هدف اول (50% پوزیشن)",
            value=f"{target1:.2f}",
            delta=f"{((target1 / entry_price) - 1) * 100:.1f}%" if is_long else f"{((entry_price / target1) - 1) * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="هدف دوم (25% پوزیشن)",
            value=f"{target2:.2f}",
            delta=f"{((target2 / entry_price) - 1) * 100:.1f}%" if is_long else f"{((entry_price / target2) - 1) * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="هدف سوم (25% پوزیشن)",
            value=f"{target3:.2f}",
            delta=f"{((target3 / entry_price) - 1) * 100:.1f}%" if is_long else f"{((entry_price / target3) - 1) * 100:.1f}%"
        )
    
    # نمایش ارزیابی استراتژی
    st.markdown(f"""
    **نسبت ریسک به بازده موزون:** {multi_target_result['weighted_risk_reward']:.2f}
    
    **ارزیابی:** {multi_target_result['recommendation']}
    """)
    
    # نمودار ویژوال ورود/خروج با حد ضرر متحرک
    st.markdown("### نمودار استراتژی معاملاتی")
    
    fig = go.Figure()
    
    # مرتب‌سازی نقاط بر اساس نوع پوزیشن
    points = [stop_loss_price, entry_price, break_even_result['move_to_break_even_price'], target1, target2, target3]
    if is_long:
        points.sort()
    else:
        points.sort(reverse=True)
    
    # تعیین رنگ‌ها و نمادها
    colors = ['red', 'blue', 'orange', 'green', 'green', 'green']
    symbols = ['triangle-down', 'circle', 'diamond', 'triangle-up', 'triangle-up', 'triangle-up']
    labels = ['حد ضرر', 'ورود', 'حد ضرر متحرک', 'هدف 1', 'هدف 2', 'هدف 3']
    
    # ایجاد نمودار
    fig.add_trace(go.Scatter(
        x=list(range(len(points))),
        y=points,
        mode='lines+markers+text',
        marker=dict(
            size=12,
            color=colors,
            symbol=symbols
        ),
        text=labels,
        textposition="top center"
    ))
    
    # تنظیمات نمودار
    fig.update_layout(
        title=f"استراتژی معاملاتی {'خرید' if is_long else 'فروش'} با خروج تدریجی و حد ضرر متحرک",
        xaxis=dict(
            title="مراحل معامله",
            showticklabels=False
        ),
        yaxis=dict(title="قیمت"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # توصیه‌های پیشرفته
    st.markdown("### توصیه‌های پیشرفته مدیریت ریسک")
    
    with st.expander("مشاهده توصیه‌های پیشرفته"):
        st.markdown(f"""
        **1. مدیریت سایز پوزیشن:**
        - سایز پوزیشن محاسبه شده: {position_size_result['position_size']:.2f} USDT (ریسک {position_size_result['risk_percent']:.2f}%)
        - برای محافظه‌کاری بیشتر می‌توانید سایز پوزیشن را {position_size_result['position_size'] * 0.8:.2f} USDT در نظر بگیرید.
        
        **2. مدیریت حد ضرر متحرک:**
        - پس از رسیدن قیمت به {break_even_result['move_to_break_even_price']:.2f} (فاصله {break_even_result['move_to_break_even_percent']:.2f}% از ورود)، حد ضرر را به نقطه ورود منتقل کنید.
        - پس از رسیدن به هدف اول و بستن 50% پوزیشن، حد ضرر را به نقطه سر به سر ({break_even_result['break_even_price']:.2f}) منتقل کنید.
        
        **3. استراتژی خروج بهینه:**
        - هدف اول: بستن 50% پوزیشن در قیمت {target1:.2f} (نسبت ریسک به بازده 1:1)
        - هدف دوم: بستن 25% پوزیشن در قیمت {target2:.2f} (نسبت ریسک به بازده 1:2)
        - هدف سوم: بستن 25% پوزیشن در قیمت {target3:.2f} (نسبت ریسک به بازده 1:3)
        
        **4. اصلاح استراتژی در صورت تغییر شرایط بازار:**
        - اگر شرایط بازار تغییر کرد، هدف سوم را تعدیل کنید یا پوزیشن را زودتر ببندید.
        - در صورت افزایش نوسانات، سایز پوزیشن را کاهش دهید.
        
        **5. کنترل احساسات:**
        - به استراتژی از پیش تعیین شده پایبند بمانید و از تصمیمات احساسی خودداری کنید.
        - در صورت بی‌اطمینانی، از همان ابتدا سایز پوزیشن کوچکتری انتخاب کنید.
        """)

def render_capital_growth_simulator(df: pd.DataFrame, symbol: str):
    """
    رندر شبیه‌ساز رشد سرمایه
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
    """
    st.subheader("شبیه‌ساز رشد سرمایه", anchor=False)
    
    st.markdown("""
    با استفاده از این ابزار، می‌توانید رشد سرمایه خود را بر اساس پارامترهای سیستم معاملاتی شبیه‌سازی کنید و نتایج بلندمدت را مشاهده نمایید.
    """)
    
    # ایجاد آنالایزر
    analyzer = RiskRewardAnalyzer()
    
    # دریافت پارامترهای ورودی
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input(
            "سرمایه اولیه (USDT)",
            min_value=100.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
            key="sim_capital"
        )
        
        win_rate = st.slider(
            "نرخ برد (درصد)",
            min_value=30,
            max_value=80,
            value=55,
            step=5,
            key="sim_win_rate"
        ) / 100
    
    with col2:
        avg_win = st.slider(
            "میانگین سود هر معامله (درصد)",
            min_value=1.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
            key="sim_avg_win"
        )
        
        avg_loss = st.slider(
            "میانگین ضرر هر معامله (درصد)",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            key="sim_avg_loss"
        )
    
    with col3:
        num_trades = st.slider(
            "تعداد معاملات",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="sim_num_trades"
        )
        
        risk_per_trade = st.slider(
            "درصد ریسک هر معامله",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="sim_risk"
        ) / 100
    
    # محاسبه امید ریاضی
    expectancy_result = analyzer.calculate_expectancy(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss
    )
    
    # شبیه‌سازی رشد سرمایه
    analyzer.initial_capital = initial_capital
    
    simulation_result = analyzer.simulate_capital_growth(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        num_trades=num_trades,
        risk_per_trade=risk_per_trade
    )
    
    st.markdown("---")
    
    # نمایش نتایج امید ریاضی
    st.markdown("### امید ریاضی سیستم معاملاتی")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="امید ریاضی",
            value=f"{expectancy_result['expectancy']:.2f}",
            delta="مثبت" if expectancy_result['expectancy'] > 0 else "منفی"
        )
    
    with col2:
        st.metric(
            label="امید به ازای هر دلار ریسک",
            value=f"{expectancy_result['expectancy_per_dollar']:.2f}",
            delta="مثبت" if expectancy_result['expectancy_per_dollar'] > 0 else "منفی"
        )
    
    with col3:
        st.metric(
            label="ضریب سود",
            value=f"{expectancy_result['profit_factor']:.2f}",
            delta="مطلوب" if expectancy_result['profit_factor'] > 1 else "نامطلوب"
        )
    
    with col4:
        st.info(f"**ارزیابی سیستم:**\n{expectancy_result['expectancy_rating']}")
    
    # نمایش نتایج شبیه‌سازی
    st.markdown("### نتایج شبیه‌سازی رشد سرمایه")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="سرمایه نهایی",
            value=f"{simulation_result['final_capital']:.2f} USDT",
            delta=f"{simulation_result['total_return']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="بازده متوسط هر معامله",
            value=f"{simulation_result['avg_return_per_trade']:.2f}%"
        )
    
    with col3:
        st.metric(
            label="حداکثر افت سرمایه",
            value=f"{simulation_result['max_drawdown']:.2f}%",
            delta="پایین" if simulation_result['max_drawdown'] < 20 else "بالا",
            delta_color="normal" if simulation_result['max_drawdown'] < 20 else "inverse"
        )
    
    with col4:
        st.metric(
            label="نسبت شارپ",
            value=f"{simulation_result['sharpe_ratio']:.2f}",
            delta="خوب" if simulation_result['sharpe_ratio'] > 1 else "ضعیف",
            delta_color="normal" if simulation_result['sharpe_ratio'] > 1 else "inverse"
        )
    
    # نمودار رشد سرمایه
    st.markdown("### نمودار رشد سرمایه")
    
    capital_history = simulation_result['capital_history']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(capital_history))),
        y=capital_history,
        mode='lines',
        name='سرمایه',
        line=dict(color='blue', width=2)
    ))
    
    # اضافه کردن خط سرمایه اولیه
    fig.add_trace(go.Scatter(
        x=[0, len(capital_history) - 1],
        y=[initial_capital, initial_capital],
        mode='lines',
        name='سرمایه اولیه',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title="نمودار رشد سرمایه در طول معاملات",
        xaxis_title="تعداد معاملات",
        yaxis_title="سرمایه (USDT)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # توصیه‌های بهبود سیستم معاملاتی
    st.markdown("### توصیه‌های بهبود سیستم معاملاتی")
    
    if expectancy_result['expectancy_per_dollar'] < 0.1:
        st.warning("""
        **سیستم معاملاتی فعلی امید ریاضی پایینی دارد. توصیه‌های زیر به بهبود آن کمک می‌کند:**
        
        1. **افزایش نرخ برد**:
            - بهبود شرایط ورود با استفاده از فیلترهای بیشتر
            - اجتناب از معامله در شرایط بی‌ثبات بازار
            
        2. **افزایش میانگین سود**:
            - استفاده از اهداف چندگانه برای معاملات سودده
            - اجازه دادن به ادامه روند در معاملات سودده
            
        3. **کاهش میانگین ضرر**:
            - تنظیم دقیق‌تر حد ضرر
            - استفاده از حد ضرر متحرک
            
        4. **کاهش ریسک هر معامله**:
            - کاهش درصد ریسک به 1-1.5% در هر معامله
            - افزایش تدریجی ریسک با افزایش سرمایه
        """)
    elif simulation_result['max_drawdown'] > 30:
        st.warning("""
        **افت سرمایه حداکثری بسیار بالاست. توصیه‌های زیر به بهبود آن کمک می‌کند:**
        
        1. **کاهش ریسک هر معامله**:
            - کاهش درصد ریسک به 1-1.5% در هر معامله
            
        2. **تنوع‌بخشی به معاملات**:
            - معامله روی ارزهای متنوع با همبستگی کمتر
            - توزیع ریسک بین انواع مختلف استراتژی‌ها
            
        3. **بهبود مدیریت ریسک**:
            - استفاده از حد ضرر متحرک
            - خروج تدریجی از پوزیشن‌ها
            
        4. **کاهش تعداد معاملات همزمان**:
            - محدود کردن تعداد معاملات همزمان به 3-5 معامله
        """)
    else:
        st.success("""
        **سیستم معاملاتی فعلی از پارامترهای مطلوبی برخوردار است. توصیه‌های زیر به حفظ و بهبود آن کمک می‌کند:**
        
        1. **حفظ نظم و انضباط معاملاتی**:
            - پایبندی به قوانین سیستم
            - اجتناب از تصمیمات احساسی
            
        2. **بهینه‌سازی تدریجی**:
            - ثبت دقیق معاملات و تحلیل عملکرد
            - بهبود تدریجی پارامترها بر اساس داده‌های واقعی
            
        3. **بهبود نسبت شارپ**:
            - کاهش نوسانات سرمایه با تنوع‌بخشی
            - افزایش تدریجی اندازه پوزیشن با افزایش سرمایه
            
        4. **آزمایش در مقیاس کوچک**:
            - آزمایش تغییرات سیستم با ریسک محدود
            - گسترش تدریجی استراتژی‌های موفق
        """)

def render_stop_loss_strategies(df: pd.DataFrame, symbol: str):
    """
    رندر استراتژی‌های حد ضرر
    
    Args:
        df (pd.DataFrame): دیتافریم اطلاعات قیمت
        symbol (str): نماد ارز
    """
    st.subheader("استراتژی‌های پیشرفته حد ضرر", anchor=False)
    
    st.markdown("""
    این بخش به شما کمک می‌کند تا استراتژی‌های پیشرفته حد ضرر را بررسی کنید و بهترین روش را برای محافظت از سرمایه خود انتخاب نمایید.
    """)
    
    # ایجاد آنالایزر
    analyzer = RiskRewardAnalyzer()
    
    # دریافت قیمت فعلی
    if not df.empty:
        current_price = df['close'].iloc[-1]
        # محاسبه ATR
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
    else:
        current_price = 30000.0
        atr = current_price * 0.02
    
    # دریافت اطلاعات ورودی
    col1, col2 = st.columns(2)
    
    with col1:
        trade_type = st.selectbox(
            "نوع معامله",
            options=["خرید (Long)", "فروش (Short)"],
            index=0,
            key="sl_trade_type"
        )
        
        entry_price = st.number_input(
            "قیمت ورود",
            min_value=0.0001,
            max_value=float(current_price * 2),
            value=float(current_price),
            format="%.2f",
            key="sl_entry_price"
        )
    
    with col2:
        stop_loss_type = st.selectbox(
            "نوع حد ضرر",
            options=["ثابت", "ATR", "درصدی", "حمایت/مقاومت"],
            index=1,
            key="sl_type"
        )
        
        if stop_loss_type == "ثابت":
            stop_loss_value = st.number_input(
                "مقدار حد ضرر",
                min_value=0.0001,
                max_value=float(entry_price * 2),
                value=float(entry_price * 0.95) if trade_type == "خرید (Long)" else float(entry_price * 1.05),
                format="%.2f",
                key="sl_value"
            )
            
        elif stop_loss_type == "ATR":
            atr_multiplier = st.slider(
                "ضریب ATR",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="sl_atr_mult"
            )
            stop_loss_value = entry_price - (atr * atr_multiplier) if trade_type == "خرید (Long)" else entry_price + (atr * atr_multiplier)
            
        elif stop_loss_type == "درصدی":
            percent_value = st.slider(
                "درصد فاصله",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="sl_percent"
            )
            stop_loss_value = entry_price * (1 - percent_value/100) if trade_type == "خرید (Long)" else entry_price * (1 + percent_value/100)
            
        else:  # حمایت/مقاومت
            if not df.empty:
                # محاسبه ساده سطوح حمایت/مقاومت
                if trade_type == "خرید (Long)":
                    support_levels = df['low'].rolling(window=20).min().iloc[-10:].unique()
                    support_levels = support_levels[support_levels < entry_price]
                    support_levels = np.sort(support_levels)
                    
                    if len(support_levels) > 0:
                        closest_support = support_levels[-1]
                    else:
                        closest_support = entry_price * 0.95
                    
                    stop_loss_value = closest_support * 0.99  # کمی پایین‌تر از سطح حمایت
                else:
                    resistance_levels = df['high'].rolling(window=20).max().iloc[-10:].unique()
                    resistance_levels = resistance_levels[resistance_levels > entry_price]
                    resistance_levels = np.sort(resistance_levels)
                    
                    if len(resistance_levels) > 0:
                        closest_resistance = resistance_levels[0]
                    else:
                        closest_resistance = entry_price * 1.05
                    
                    stop_loss_value = closest_resistance * 1.01  # کمی بالاتر از سطح مقاومت
            else:
                stop_loss_value = entry_price * 0.95 if trade_type == "خرید (Long)" else entry_price * 1.05
    
    st.markdown("---")
    
    # محاسبه انواع حد ضرر
    stop_loss_result = analyzer.calculate_stop_loss_types(
        entry_price=entry_price,
        initial_stop_loss=stop_loss_value,
        high_price=df['high'].max() if not df.empty else entry_price * 1.1,
        low_price=df['low'].min() if not df.empty else entry_price * 0.9,
        atr=atr
    )
    
    # محاسبه استراتژی‌های حد ضرر متحرک
    trailing_stop_result = analyzer.calculate_trailing_stop_scenarios(
        entry_price=entry_price,
        initial_stop_loss=stop_loss_value,
        atr=atr
    )
    
    # نمایش انواع حد ضرر
    st.markdown("### مقایسه انواع حد ضرر")
    
    stop_loss_df = pd.DataFrame([
        {"نوع": "حد ضرر اولیه", "قیمت": f"{stop_loss_result['initial_stop_loss']:.2f}", "ریسک": f"{stop_loss_result['initial_risk_percent']:.2f}%"},
        {"نوع": "حد ضرر ATR", "قیمت": f"{stop_loss_result.get('atr_stop_loss', 0):.2f}", "ریسک": f"{stop_loss_result.get('atr_risk_percent', 0):.2f}%"},
        {"نوع": "حد ضرر نوسانی", "قیمت": f"{stop_loss_result.get('swing_stop_loss', 0):.2f}", "ریسک": f"{stop_loss_result.get('swing_risk_percent', 0):.2f}%"},
        {"نوع": "حد ضرر درصدی 2%", "قیمت": f"{stop_loss_result.get('percent_stop_loss', 0):.2f}", "ریسک": f"{stop_loss_result.get('percent_risk_percent', 0):.2f}%"}
    ])
    
    st.dataframe(stop_loss_df, use_container_width=True, hide_index=True)
    
    # نمایش استراتژی‌های حد ضرر متحرک
    st.markdown("### استراتژی‌های حد ضرر متحرک")
    
    trailing_scenarios = trailing_stop_result.get('scenarios', [])
    
    if trailing_scenarios:
        scenario_data = []
        for scenario in trailing_scenarios:
            scenario_data.append({
                "نوع استراتژی": scenario['description'],
                "قیمت فعال‌سازی": f"{scenario['activation_price']:.2f}",
                "فاصله فعال‌سازی": f"{scenario['activation_percent']:.2f}%"
            })
        
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    
    # نمودار حد ضرر متحرک
    st.markdown("### نمودار استراتژی حد ضرر متحرک")
    
    fig = go.Figure()
    
    # تعیین نوع پوزیشن
    is_long = trailing_stop_result.get('is_long', True)
    
    # ایجاد نمودار مسیر قیمت فرضی
    x_range = np.arange(10)
    
    if is_long:
        # مسیر قیمت صعودی
        price_path = entry_price * (1 + np.array([0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]))
        
        # خط حد ضرر اولیه
        initial_sl_line = [stop_loss_value] * len(x_range)
        
        # خط حد ضرر متحرک (فرضی)
        trailing_sl_values = []
        
        for i, price in enumerate(price_path):
            # حد ضرر متحرک با فاصله 2%
            if i < 3:
                trailing_sl_values.append(stop_loss_value)
            else:
                trailing_sl_values.append(min(price * 0.98, price_path[i-1] * 0.98))
    else:
        # مسیر قیمت نزولی
        price_path = entry_price * (1 - np.array([0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]))
        
        # خط حد ضرر اولیه
        initial_sl_line = [stop_loss_value] * len(x_range)
        
        # خط حد ضرر متحرک (فرضی)
        trailing_sl_values = []
        
        for i, price in enumerate(price_path):
            # حد ضرر متحرک با فاصله 2%
            if i < 3:
                trailing_sl_values.append(stop_loss_value)
            else:
                trailing_sl_values.append(max(price * 1.02, price_path[i-1] * 1.02))
    
    # اضافه کردن مسیر قیمت
    fig.add_trace(go.Scatter(
        x=x_range,
        y=price_path,
        mode='lines+markers',
        name='مسیر قیمت',
        line=dict(color='blue', width=3)
    ))
    
    # اضافه کردن خط حد ضرر اولیه
    fig.add_trace(go.Scatter(
        x=x_range,
        y=initial_sl_line,
        mode='lines',
        name='حد ضرر اولیه',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # اضافه کردن خط حد ضرر متحرک
    fig.add_trace(go.Scatter(
        x=x_range,
        y=trailing_sl_values,
        mode='lines',
        name='حد ضرر متحرک',
        line=dict(color='green', width=2)
    ))
    
    # نمایش قیمت ورود
    fig.add_trace(go.Scatter(
        x=[0],
        y=[entry_price],
        mode='markers',
        name='قیمت ورود',
        marker=dict(
            color='purple',
            size=12,
            symbol='circle'
        )
    ))
    
    # تنظیمات نمودار
    fig.update_layout(
        title=f"شبیه‌سازی استراتژی حد ضرر متحرک در معامله {'خرید' if is_long else 'فروش'}",
        xaxis_title="روند قیمت",
        yaxis_title="قیمت",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # توصیه‌های حد ضرر
    st.markdown("### توصیه‌های استراتژی حد ضرر")
    
    with st.expander("مشاهده توصیه‌های استراتژی حد ضرر"):
        # تنظیم حالت
        position_type = "خرید (Long)" if is_long else "فروش (Short)"
        risk_percent = abs((stop_loss_value / entry_price - 1) * 100)
        
        st.markdown(f"""
        **1. انتخاب حد ضرر برای معامله {position_type}:**
        
        بهترین حد ضرر برای این معامله بر اساس پارامترهای مختلف:
        
        {'- **حد ضرر ATR:** این روش با تنظیم فاصله حد ضرر بر اساس نوسانات واقعی بازار، مناسب‌ترین گزینه است.' if stop_loss_type == 'ATR' else ''}
        {'- **حد ضرر نوسانی:** با قرار دادن حد ضرر در زیر سطح حمایت اخیر، امکان نوسانات طبیعی بازار را می‌دهد.' if stop_loss_type == 'حمایت/مقاومت' else ''}
        {'- **حد ضرر درصدی:** با ریسک ثابت {risk_percent:.1f}% ساده‌ترین روش است اما ممکن است با نوسانات بازار همخوانی نداشته باشد.' if stop_loss_type == 'درصدی' else ''}
        {'- **حد ضرر ثابت:** در قیمت {stop_loss_value:.2f} با ریسک {risk_percent:.1f}% تعیین شده است.' if stop_loss_type == 'ثابت' else ''}
        
        **2. استراتژی حد ضرر متحرک برای افزایش سود:**
        
        بهترین استراتژی حد ضرر متحرک برای این معامله:
        
        - **انتقال به نقطه سر به سر پس از 3% سود:** این روش ریسک را به صفر می‌رساند و امکان ادامه سود را فراهم می‌کند.
        - **حد ضرر متحرک با فاصله نسبی 2%:** این روش به شما امکان می‌دهد از روند سودده حداکثر استفاده را ببرید.
        
        **3. توصیه‌های کاربردی:**
        
        - از حد ضرر تلفیقی استفاده کنید: شروع با حد ضرر ATR، انتقال به نقطه سر به سر و سپس حد ضرر متحرک.
        - در محیط‌های پرنوسان، فاصله حد ضرر را افزایش دهید اما سایز پوزیشن را کاهش دهید.
        - همیشه حد ضرر را خارج از نواحی تراکم معاملات قرار دهید تا از شکار نقدینگی جلوگیری شود.
        """)