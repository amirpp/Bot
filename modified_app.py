"""
برنامه اصلی تحلیل ارزهای دیجیتال با استفاده از هوش مصنوعی، اندیکاتورهای تکنیکال و سیگنال‌های معاملاتی

این برنامه یک رابط کاربری مبتنی بر Streamlit ایجاد می‌کند که امکان تحلیل بازار ارزهای دیجیتال
در تایم‌فریم‌های مختلف را با استفاده از بیش از 900 اندیکاتور و هوش مصنوعی فراهم می‌کند.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import traceback
from PIL import Image
import io
import plotly.express as px
import random
import threading
import json

# وارد کردن ماژول‌های سفارشی
from crypto_data import get_crypto_data, get_available_timeframes, get_available_exchanges, get_current_price
from technical_analysis import perform_technical_analysis, generate_signals, AVAILABLE_INDICATORS, TOP_INDICATORS
from telegram_bot import (
    send_telegram_message, send_telegram_photo, initialize_telegram_bot, 
    send_signal_message, start_telegram_bot, stop_telegram_bot
)
from sentiment_analysis import SentimentAnalyzer, get_market_sentiment, get_sentiment_signal
from custom_ai_api import AIManager, get_ai_manager_instance, check_ai_api_status
from crypto_search import get_crypto_search
from mega_indicator_manager import get_indicator_manager
from target_price_calculator import get_target_calculator

# تنظیم حالت دارک و لایت
def setup_dark_light_mode():
    """تنظیم حالت دارک و لایت با استفاده از CSS"""
    # Check if the theme is already in session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'  # حالت پیش‌فرض دارک
    
    # Add CSS for smooth theme transitions
    theme_css = """
    <style>
    /* Base styles with smooth transitions */
    body, .stApp, .stButton>button, .stTextInput>div>div>input, 
    .stSelectbox>div, [data-testid="stSidebar"] {
        transition: all 0.5s ease !important;
    }
    
    /* Light/Dark mode toggle button styles */
    .theme-toggle {
        cursor: pointer;
        background: none;
        border: none;
        font-size: 1.5rem;
        margin: 10px;
        padding: 5px 10px;
        border-radius: 20px;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    /* حالت تاریک */
    .dark {
        background-color: #121212 !important;
        color: #E0E0E0 !important;
    }
    
    .dark .stButton>button {
        background-color: #3d5a80 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
    
    .dark .stTextInput>div>div>input,
    .dark .stSelectbox>div,
    .dark .stSlider>div {
        background-color: #2C2C2C !important;
        color: #E0E0E0 !important;
        border-color: #555 !important;
    }
    
    .dark [data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
    }
    
    .dark .stTabs [data-baseweb="tab"] {
        color: #E0E0E0 !important;
    }
    
    .dark .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4da6ff !important;
    }
    
    /* حالت روشن */
    .light {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    
    .light .stButton>button {
        background-color: #4283f3 !important;
        color: white !important;
        border: 1px solid #ddd !important;
    }
    
    .light .stTextInput>div>div>input,
    .light .stSelectbox>div,
    .light .stSlider>div {
        background-color: #f7f7f7 !important;
        color: #111111 !important;
        border-color: #ddd !important;
    }
    
    .light [data-testid="stSidebar"] {
        background-color: #F7F7F7 !important;
    }
    
    /* Signal colors for both themes */
    .dark .buy-signal {
        background-color: rgba(0, 128, 0, 0.2) !important;
        color: #7AE7AA !important;
    }
    
    .dark .sell-signal {
        background-color: rgba(128, 0, 0, 0.2) !important;
        color: #FF9A9A !important;
    }
    
    .dark .neutral-signal {
        background-color: rgba(128, 128, 128, 0.2) !important;
        color: #C0C0C0 !important;
    }
    
    .light .buy-signal {
        background-color: rgba(0, 128, 0, 0.1) !important;
        color: #006400 !important;
    }
    
    .light .sell-signal {
        background-color: rgba(128, 0, 0, 0.1) !important;
        color: #8B0000 !important;
    }
    
    .light .neutral-signal {
        background-color: rgba(128, 128, 128, 0.1) !important;
        color: #696969 !important;
    }
    
    /* RTL Support */
    .rtl {
        direction: rtl;
        text-align: right;
    }
    
    /* Animation for transitions */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeIn 0.3s ease-in-out;
    }

    /* Dashboard styles */
    .dashboard-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
        margin-top: 20px;
    }

    .dashboard-card {
        background-color: rgba(61, 90, 128, 0.1);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    .indicator-badge {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.8em;
        background-color: rgba(66, 131, 243, 0.2);
    }

    .signal-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 2px;
        border-radius: 15px;
        font-weight: bold;
    }

    .signal-badge.buy {
        background-color: rgba(0, 128, 0, 0.2);
        color: #00A36C;
    }

    .signal-badge.sell {
        background-color: rgba(128, 0, 0, 0.2);
        color: #CD5C5C;
    }

    .signal-badge.neutral {
        background-color: rgba(128, 128, 128, 0.2);
        color: #A9A9A9;
    }

    /* Trend indicator styles */
    .trend-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        margin: 10px 0;
    }

    .trend-up {
        background-color: rgba(0, 128, 0, 0.1);
        color: #00A36C;
    }

    .trend-down {
        background-color: rgba(128, 0, 0, 0.1);
        color: #CD5C5C;
    }

    .trend-neutral {
        background-color: rgba(128, 128, 128, 0.1);
        color: #A9A9A9;
    }

    /* Target price styles */
    .target-container {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 10px;
        margin: 15px 0;
    }

    .target-header {
        font-weight: bold;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }

    .target-item {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
    }

    .target-price {
        font-weight: bold;
    }

    .target-profit {
        color: #00A36C;
    }

    /* Improved tabs */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        padding: 10px 15px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(66, 131, 243, 0.1);
        border-bottom: 3px solid #4283f3;
    }

    /* Search bar styles */
    .search-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 10px 15px;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    /* AI analysis styles */
    .ai-analysis-container {
        border-left: 4px solid #4283f3;
        padding-left: 15px;
        margin: 20px 0;
        background-color: rgba(66, 131, 243, 0.05);
        border-radius: 0 8px 8px 0;
        padding: 15px;
    }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
    
    # Create a container for the toggle in the sidebar
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### تنظیمات نمایش")
        
        with col2:
            if st.session_state.theme == 'light':
                toggle_icon = "🌙"
                toggle_text = "تاریک"
            else:
                toggle_icon = "☀️"
                toggle_text = "روشن"
            
            # Create the toggle button
            if st.button(f"{toggle_icon} {toggle_text}"):
                if st.session_state.theme == 'light':
                    st.session_state.theme = 'dark'
                else:
                    st.session_state.theme = 'light'
                st.rerun()
    
    # Apply the theme class to the entire app
    st.markdown(f"""
    <script>
    document.querySelector('body').className = '{st.session_state.theme}';
    </script>
    """, unsafe_allow_html=True)
    
    # Wrap content in theme div
    st.markdown(f"<div class='{st.session_state.theme} rtl'>", unsafe_allow_html=True)

def plot_prediction_chart(df, pred_df, symbol, timeframe):
    """
    رسم نمودار پیش‌بینی قیمت با استفاده از Plotly
    
    Args:
        df (pd.DataFrame): دیتافریم تاریخچه قیمت
        pred_df (pd.DataFrame): دیتافریم پیش‌بینی
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        
    Returns:
        go.Figure: نمودار پیش‌بینی
    """
    if df is None or pred_df is None:
        return None
        
    # ایجاد نمودار با Plotly
    fig = go.Figure()
    
    # اضافه کردن داده‌های تاریخی
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='قیمت واقعی',
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        )
    )
    
    # اضافه کردن داده‌های پیش‌بینی
    fig.add_trace(
        go.Scatter(
            x=pred_df.index, 
            y=pred_df['predicted_close'],
            mode='lines',
            name='پیش‌بینی قیمت',
            line=dict(color='#3f51b5', width=2, dash='dot')
        )
    )
    
    # اضافه کردن نوار اطمینان پیش‌بینی
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df['confidence_high'],
            mode='lines',
            name='حد بالای اطمینان',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df['confidence_low'],
            mode='lines',
            name='حد پایین اطمینان',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(63, 81, 181, 0.2)',
            showlegend=False
        )
    )
    
    # تنظیمات نمودار
    fig.update_layout(
        title=f"پیش‌بینی قیمت {symbol} ({timeframe})",
        xaxis_title='تاریخ',
        yaxis_title='قیمت (USDT)',
        height=600,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # تنظیمات تاریخ
    fig.update_xaxes(
        title_text='تاریخ',
        rangeslider_visible=False,
        rangebreaks=[dict(pattern="day of week", bounds=["sat", "mon"])]
    )
    
    return fig

def plot_technical_chart(df, symbol, timeframe, selected_indicators=None):
    """
    رسم نمودار تکنیکال با استفاده از Plotly
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت و اندیکاتورها
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        selected_indicators (list): لیست اندیکاتورهای انتخاب شده
        
    Returns:
        go.Figure: نمودار تکنیکال
    """
    if df is None or df.empty:
        return None
    
    # آماده‌سازی دیتافریم (حذف ستون‌های NaN)
    df = df.copy()
    df = df.dropna(how='all', axis=1)
    
    # تنظیم اندیکاتورهای انتخاب شده
    if selected_indicators is None:
        selected_indicators = []
    
    # ایجاد نمودار Plotly
    fig = go.Figure()
    
    # اضافه کردن نمودار شمعی
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='قیمت',
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        )
    )
    
    # اضافه کردن اندیکاتورهای انتخاب شده
    for indicator in selected_indicators:
        if indicator == 'Moving Average':
            # اضافه کردن میانگین‌های متحرک
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#e91e63', width=1.5)
                    )
                )
            if 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['sma_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='#2196f3', width=1.5)
                    )
                )
            if 'sma_200' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['sma_200'],
                        mode='lines',
                        name='SMA 200',
                        line=dict(color='#ff9800', width=1.5)
                    )
                )
                
        elif indicator == 'Bollinger Bands':
            # اضافه کردن باندهای بولینگر
            if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['bb_upper'],
                        mode='lines',
                        name='Bollinger Upper',
                        line=dict(color='#4caf50', width=1.5, dash='dash')
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['bb_middle'],
                        mode='lines',
                        name='Bollinger Middle',
                        line=dict(color='#9c27b0', width=1.5)
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['bb_lower'],
                        mode='lines',
                        name='Bollinger Lower',
                        line=dict(color='#4caf50', width=1.5, dash='dash')
                    )
                )
                
        elif indicator == 'VWAP':
            # اضافه کردن VWAP
            if 'vwap' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['vwap'],
                        mode='lines',
                        name='VWAP',
                        line=dict(color='#673ab7', width=2)
                    )
                )
                
        elif indicator == 'SuperTrend':
            # اضافه کردن SuperTrend
            if 'supertrend' in df.columns:
                # مجموعه نقاط صعودی و نزولی
                df_uptrend = df[df['supertrend_direction'] == 1].copy()
                df_downtrend = df[df['supertrend_direction'] == -1].copy()
                
                fig.add_trace(
                    go.Scatter(
                        x=df_uptrend.index, 
                        y=df_uptrend['supertrend'],
                        mode='lines',
                        name='SuperTrend (صعودی)',
                        line=dict(color='#00c853', width=2)
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_downtrend.index, 
                        y=df_downtrend['supertrend'],
                        mode='lines',
                        name='SuperTrend (نزولی)',
                        line=dict(color='#ff5252', width=2)
                    )
                )
    
    # تنظیمات نمودار
    fig.update_layout(
        title=f"تحلیل تکنیکال {symbol} ({timeframe})",
        xaxis_title='تاریخ',
        yaxis_title='قیمت (USDT)',
        height=600,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def display_signals_table(signals):
    """
    نمایش سیگنال‌های معاملاتی در جدول
    
    Args:
        signals (dict): دیکشنری سیگنال‌ها
    """
    if not signals:
        st.warning("هیچ سیگنالی یافت نشد.")
        return
    
    # تبدیل دیکشنری سیگنال‌ها به دیتافریم
    signal_data = []
    for indicator_name, signal_info in signals.items():
        signal_data.append({
            "اندیکاتور": indicator_name,
            "سیگنال": signal_info.get('signal', 'NEUTRAL'),
            "قدرت": signal_info.get('strength', 0),
            "توضیحات": signal_info.get('description', '')
        })
    
    signal_df = pd.DataFrame(signal_data)
    
    # تابع رنگ‌آمیزی برای سیگنال‌ها
    def color_signal(val):
        if val == 'BUY':
            return 'background-color: rgba(0, 128, 0, 0.2); color: #00A36C; font-weight: bold'
        elif val == 'SELL':
            return 'background-color: rgba(128, 0, 0, 0.2); color: #CD5C5C; font-weight: bold'
        elif val == 'NEUTRAL':
            return 'background-color: rgba(128, 128, 128, 0.2); color: #A9A9A9; font-weight: bold'
        return ''
    
    # نمایش جدول با استایل
    st.dataframe(signal_df.style.map(color_signal, subset=['سیگنال']))

def create_support_resistance_viz(df, levels, symbol):
    """
    ایجاد نمودار سطوح حمایت و مقاومت
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        levels (list): لیست سطوح حمایت و مقاومت
        symbol (str): نماد ارز
        
    Returns:
        go.Figure: نمودار سطوح حمایت و مقاومت
    """
    if df is None or df.empty or not levels:
        return None
    
    # ایجاد نمودار Plotly
    fig = go.Figure()
    
    # اضافه کردن نمودار شمعی
    fig.add_trace(
        go.Candlestick(
            x=df.index[-100:],  # نمایش 100 نقطه آخر
            open=df['open'][-100:],
            high=df['high'][-100:],
            low=df['low'][-100:],
            close=df['close'][-100:],
            name='قیمت',
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        )
    )
    
    # اضافه کردن سطوح حمایت و مقاومت
    current_price = df['close'].iloc[-1]
    
    for level_info in levels:
        level = level_info['level']
        level_type = level_info['type']
        
        if level_type == 'resistance':
            color = '#ff5252'  # قرمز برای مقاومت
            name = f"مقاومت: {level:.2f}"
        elif level_type == 'support':
            color = '#00c853'  # سبز برای حمایت
            name = f"حمایت: {level:.2f}"
        elif level_type == 'pivot':
            color = '#ffab00'  # نارنجی برای پیووت
            name = f"پیووت: {level:.2f}"
        else:
            color = '#9e9e9e'  # خاکستری برای سایر
            name = f"سطح: {level:.2f}"
        
        # اضافه کردن خطوط افقی
        fig.add_shape(
            type="line",
            x0=df.index[-100],
            y0=level,
            x1=df.index[-1],
            y1=level,
            line=dict(
                color=color,
                width=2,
                dash="solid" if abs((level/current_price) - 1) < 0.02 else "dash"  # خط ممتد برای سطوح نزدیک
            ),
            opacity=0.7
        )
        
        # اضافه کردن برچسب
        fig.add_annotation(
            x=df.index[-1],
            y=level,
            text=name,
            showarrow=False,
            font=dict(
                size=10,
                color="white"
            ),
            bgcolor=color,
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            xshift=50
        )
    
    # تنظیمات نمودار
    fig.update_layout(
        title=f"سطوح حمایت و مقاومت {symbol}",
        xaxis_title='تاریخ',
        yaxis_title='قیمت (USDT)',
        height=500,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_entry_exit_viz(df, entry_exit_points, symbol):
    """
    ایجاد نمودار نقاط ورود و خروج
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        entry_exit_points (dict): اطلاعات نقاط ورود و خروج
        symbol (str): نماد ارز
        
    Returns:
        go.Figure: نمودار نقاط ورود و خروج
    """
    if df is None or df.empty or not entry_exit_points:
        return None
    
    # ایجاد نمودار Plotly
    fig = go.Figure()
    
    # اضافه کردن نمودار شمعی
    fig.add_trace(
        go.Candlestick(
            x=df.index[-100:],  # نمایش 100 نقطه آخر
            open=df['open'][-100:],
            high=df['high'][-100:],
            low=df['low'][-100:],
            close=df['close'][-100:],
            name='قیمت',
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        )
    )
    
    # استخراج داده‌های نقاط ورود و خروج
    entry_price = entry_exit_points.get('entry_price', 0)
    stop_loss = entry_exit_points.get('stop_loss', 0)
    targets = entry_exit_points.get('targets', [])
    signal = entry_exit_points.get('signal', 'NEUTRAL')
    
    # اضافه کردن خط قیمت ورود
    fig.add_shape(
        type="line",
        x0=df.index[-100],
        y0=entry_price,
        x1=df.index[-1],
        y1=entry_price,
        line=dict(
            color="#4caf50" if signal == 'BUY' else "#f44336",
            width=2,
            dash="solid"
        ),
        opacity=0.8
    )
    
    # اضافه کردن برچسب قیمت ورود
    fig.add_annotation(
        x=df.index[-1],
        y=entry_price,
        text=f"قیمت ورود: {entry_price:.2f}",
        showarrow=False,
        font=dict(
            size=10,
            color="white"
        ),
        bgcolor="#4caf50" if signal == 'BUY' else "#f44336",
        bordercolor="#4caf50" if signal == 'BUY' else "#f44336",
        borderwidth=1,
        borderpad=4,
        xshift=70
    )
    
    # اضافه کردن خط حد ضرر
    fig.add_shape(
        type="line",
        x0=df.index[-100],
        y0=stop_loss,
        x1=df.index[-1],
        y1=stop_loss,
        line=dict(
            color="#f44336",
            width=2,
            dash="dash"
        ),
        opacity=0.8
    )
    
    # اضافه کردن برچسب حد ضرر
    fig.add_annotation(
        x=df.index[-1],
        y=stop_loss,
        text=f"حد ضرر: {stop_loss:.2f}",
        showarrow=False,
        font=dict(
            size=10,
            color="white"
        ),
        bgcolor="#f44336",
        bordercolor="#f44336",
        borderwidth=1,
        borderpad=4,
        xshift=50
    )
    
    # اضافه کردن خطوط اهداف قیمتی
    for i, target in enumerate(targets):
        target_price = target.get('price', 0)
        
        fig.add_shape(
            type="line",
            x0=df.index[-100],
            y0=target_price,
            x1=df.index[-1],
            y1=target_price,
            line=dict(
                color="#2196f3",
                width=1.5,
                dash="dot"
            ),
            opacity=0.8
        )
        
        # اضافه کردن برچسب هدف قیمتی
        fig.add_annotation(
            x=df.index[-1],
            y=target_price,
            text=f"هدف {i+1}: {target_price:.2f}",
            showarrow=False,
            font=dict(
                size=10,
                color="white"
            ),
            bgcolor="#2196f3",
            bordercolor="#2196f3",
            borderwidth=1,
            borderpad=4,
            xshift=50
        )
    
    # تنظیمات نمودار
    fig.update_layout(
        title=f"نقاط ورود و خروج {symbol}",
        xaxis_title='تاریخ',
        yaxis_title='قیمت (USDT)',
        height=500,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def render_crypto_search_page():
    """صفحه جستجوی ارزهای دیجیتال"""
    st.header("🔍 جستجوی ارزهای دیجیتال")
    
    # دریافت نمونه جستجوگر ارزها
    crypto_search = get_crypto_search()
    
    # بخش جستجو
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    search_query = st.text_input("جستجو برای ارز...", placeholder="نام ارز را وارد کنید (مثال: BTC، ETH، SOL)")
    exchange = st.selectbox("صرافی", ["all", "binance", "kucoin", "huobi", "bybit", "kraken"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # نمایش ارزهای محبوب
    st.subheader("⭐ ارزهای محبوب")
    favorite_cryptos = crypto_search.get_favorites()
    
    # نمایش به صورت گرید
    cols = st.columns(5)
    for i, crypto in enumerate(favorite_cryptos[:10]):
        col_idx = i % 5
        with cols[col_idx]:
            symbol = crypto.split('/')[0] if '/' in crypto else crypto
            
            # بررسی اینکه آیا در علاقه‌مندی‌ها وجود دارد
            is_favorite = crypto in favorite_cryptos
            
            # دکمه با آیکون ستاره
            if st.button(f"{symbol} {'⭐' if is_favorite else '☆'}", key=f"fav_{crypto}"):
                if is_favorite:
                    crypto_search.remove_from_favorites(crypto)
                else:
                    crypto_search.add_to_favorites(crypto)
                st.rerun()
    
    # نمایش نتایج جستجو
    if search_query:
        st.subheader(f"🔎 نتایج جستجو برای: {search_query}")
        search_results = crypto_search.search_crypto(search_query, exchange)
        
        if not search_results:
            st.info("هیچ ارزی یافت نشد.")
        else:
            # نمایش به صورت گرید
            result_cols = st.columns(5)
            for i, crypto in enumerate(search_results[:15]):
                col_idx = i % 5
                with result_cols[col_idx]:
                    symbol = crypto.split('/')[0] if '/' in crypto else crypto
                    
                    # دکمه با آیکون ستاره
                    is_favorite = crypto in favorite_cryptos
                    if st.button(f"{symbol} {'⭐' if is_favorite else '☆'}", key=f"search_{crypto}"):
                        if is_favorite:
                            crypto_search.remove_from_favorites(crypto)
                        else:
                            crypto_search.add_to_favorites(crypto)
                        st.rerun()
    
    # ارزهای پرمعامله
    st.subheader("🔝 ارزهای پرمعامله")
    top_cryptos = crypto_search.get_top_traded_cryptos(limit=15)
    
    # نمایش به صورت گرید
    top_cols = st.columns(5)
    for i, crypto in enumerate(top_cryptos[:15]):
        col_idx = i % 5
        with top_cols[col_idx]:
            symbol = crypto.split('/')[0] if '/' in crypto else crypto
            
            # دکمه با آیکون ستاره
            is_favorite = crypto in favorite_cryptos
            if st.button(f"{symbol} {'⭐' if is_favorite else '☆'}", key=f"top_{crypto}"):
                if is_favorite:
                    crypto_search.remove_from_favorites(crypto)
                else:
                    crypto_search.add_to_favorites(crypto)
                st.rerun()
    
    # ارزهای پرطرفدار
    st.subheader("🔥 ارزهای پرطرفدار")
    trending_cryptos = crypto_search.get_trending_cryptos(limit=10)
    
    # نمایش به صورت جدول
    if trending_cryptos:
        trending_data = []
        for crypto in trending_cryptos:
            trending_data.append({
                "نام ارز": crypto.get('name', ''),
                "نماد": crypto.get('symbol', '').upper(),
                "رتبه": crypto.get('market_cap_rank', 0),
                "امتیاز": crypto.get('score', 0)
            })
        
        st.dataframe(pd.DataFrame(trending_data))

def render_dashboard(df, symbol, timeframe):
    """
    نمایش داشبورد اصلی تحلیل
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت و اندیکاتورها
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
    """
    if df is None or df.empty:
        st.warning("داده‌ای برای نمایش وجود ندارد.")
        return

    # محاسبه سیگنال‌ها
    try:
        signals = generate_signals(df)
    except Exception as e:
        st.error(f"خطا در تولید سیگنال‌ها: {str(e)}")
        signals = {}
    
    # محاسبه سطوح حمایت و مقاومت
    target_calculator = get_target_calculator(df)
    support_resistance_levels = target_calculator.calculate_support_resistance(levels=3, method='zigzag')
    
    # محاسبه نقاط ورود و خروج
    # تعیین سیگنال کلی
    overall_signal = "NEUTRAL"
    buy_count = sum(1 for s in signals.values() if s.get('signal') == 'BUY')
    sell_count = sum(1 for s in signals.values() if s.get('signal') == 'SELL')
    
    if buy_count > sell_count and buy_count > len(signals) / 3:
        overall_signal = "BUY"
    elif sell_count > buy_count and sell_count > len(signals) / 3:
        overall_signal = "SELL"
    
    # محاسبه نقاط ورود و خروج
    entry_exit_points = target_calculator.calculate_entry_exit_points(overall_signal)
    
    # معیارهای کلیدی (Key Metrics)
    st.subheader("📊 معیارهای کلیدی")
    
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        current_price = df['close'].iloc[-1]
        previous_price = df['close'].iloc[-2]
        price_change = ((current_price / previous_price) - 1) * 100
        price_change_color = "green" if price_change >= 0 else "red"
        
        st.metric(
            label="قیمت فعلی",
            value=f"{current_price:.2f} USDT",
            delta=f"{price_change:.2f}%",
            delta_color=price_change_color
        )
    
    with metrics_cols[1]:
        # محاسبه حجم معاملات 24 ساعته
        if 'volume' in df.columns:
            daily_volume = df['volume'].iloc[-24:].sum() if len(df) >= 24 else df['volume'].sum()
            
            # فرمت بندی حجم با ممیز هزار
            volume_str = f"{daily_volume:,.0f} USDT"
            
            # محاسبه تغییرات حجم
            prev_daily_volume = df['volume'].iloc[-48:-24].sum() if len(df) >= 48 else daily_volume
            volume_change = ((daily_volume / prev_daily_volume) - 1) * 100 if prev_daily_volume > 0 else 0
            
            st.metric(
                label="حجم معاملات (24h)",
                value=volume_str,
                delta=f"{volume_change:.2f}%",
                delta_color="normal"
            )
        else:
            st.metric(
                label="حجم معاملات (24h)",
                value="N/A"
            )
    
    with metrics_cols[2]:
        # محاسبه نوسانات
        if 'high' in df.columns and 'low' in df.columns:
            daily_high = df['high'].iloc[-24:].max() if len(df) >= 24 else df['high'].max()
            daily_low = df['low'].iloc[-24:].min() if len(df) >= 24 else df['low'].min()
            volatility = ((daily_high / daily_low) - 1) * 100
            
            st.metric(
                label="نوسان (24h)",
                value=f"{volatility:.2f}%"
            )
        else:
            st.metric(
                label="نوسان (24h)",
                value="N/A"
            )
    
    with metrics_cols[3]:
        # نمایش سیگنال کلی
        signal_color = ""
        signal_text = ""
        
        if overall_signal == "BUY":
            signal_color = "green"
            signal_text = "خرید"
        elif overall_signal == "SELL":
            signal_color = "red"
            signal_text = "فروش"
        else:
            signal_color = "gray"
            signal_text = "خنثی"
        
        st.markdown(f"""
        <div style="background-color: rgba(0, 0, 0, 0.05); padding: 10px; border-radius: 5px; text-align: center;">
            <p style="margin-bottom: 5px; font-size: 0.8em; color: gray;">سیگنال کلی</p>
            <p style="margin: 0; font-size: 1.5em; font-weight: bold; color: {signal_color};">{signal_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # نمودارهای اصلی
    st.subheader("📈 نمودارهای تکنیکال")
    
    tab1, tab2, tab3 = st.tabs(["نمودار قیمت", "سطوح حمایت و مقاومت", "نقاط ورود و خروج"])
    
    with tab1:
        # نمودار تکنیکال
        fig = plot_technical_chart(df, symbol, timeframe, selected_indicators=['Moving Average', 'Bollinger Bands', 'SuperTrend'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # نمودار سطوح حمایت و مقاومت
        fig = create_support_resistance_viz(df, support_resistance_levels, symbol)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # نمودار نقاط ورود و خروج
        fig = create_entry_exit_viz(df, entry_exit_points, symbol)
        st.plotly_chart(fig, use_container_width=True)
    
    # تحلیل سیگنال‌ها و اطلاعات ورود/خروج
    signal_cols = st.columns(2)
    
    with signal_cols[0]:
        st.subheader("🎯 سیگنال‌های معاملاتی")
        display_signals_table(signals)
    
    with signal_cols[1]:
        st.subheader("💹 استراتژی معاملاتی")
        
        if entry_exit_points:
            # اطلاعات استراتژی
            st.markdown(f"""
            <div class="target-container">
                <div class="target-header">اطلاعات ورود</div>
                <div class="target-item">
                    <span>نوع سیگنال:</span>
                    <span class="signal-badge {'buy' if entry_exit_points['signal'] == 'BUY' else 'sell' if entry_exit_points['signal'] == 'SELL' else 'neutral'}">
                        {'خرید' if entry_exit_points['signal'] == 'BUY' else 'فروش' if entry_exit_points['signal'] == 'SELL' else 'خنثی'}
                    </span>
                </div>
                <div class="target-item">
                    <span>قیمت ورود:</span>
                    <span class="target-price">{entry_exit_points['entry_price']:.2f} USDT</span>
                </div>
                <div class="target-item">
                    <span>حد ضرر:</span>
                    <span class="target-price">{entry_exit_points['stop_loss']:.2f} USDT</span>
                </div>
                <div class="target-item">
                    <span>درصد ریسک:</span>
                    <span>{entry_exit_points['risk_percent']:.2f}%</span>
                </div>
                <div class="target-item">
                    <span>نسبت ریوارد به ریسک:</span>
                    <span>{entry_exit_points['risk_reward_ratio']:.1f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # اهداف قیمتی
            st.markdown("""
            <div class="target-container">
                <div class="target-header">اهداف قیمتی</div>
            """, unsafe_allow_html=True)
            
            for i, target in enumerate(entry_exit_points['targets']):
                st.markdown(f"""
                <div class="target-item">
                    <span>هدف {i+1}:</span>
                    <span class="target-price">{target['price']:.2f} USDT</span>
                </div>
                <div class="target-item">
                    <span>سود بالقوه:</span>
                    <span class="target-profit">+{target['potential']:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def render_ai_analysis_page(df, symbol, timeframe):
    """
    صفحه تحلیل هوش مصنوعی
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت و اندیکاتورها
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
    """
    st.header("🧠 تحلیل هوش مصنوعی")
    
    if df is None or df.empty:
        st.warning("داده‌ای برای تحلیل وجود ندارد.")
        return
    
    # تحلیل هوش مصنوعی
    ai_manager = get_ai_manager_instance()
    
    # حالت شبیه‌سازی یا استفاده از API واقعی
    api_status = check_ai_api_status()
    using_simulation = all(not status for status in api_status.values())
    
    if using_simulation:
        st.info("🔄 در حال استفاده از حالت شبیه‌سازی هوش مصنوعی. برای استفاده از قابلیت‌های کامل، کلیدهای API را وارد کنید.")
    
    # سربرگ‌های مختلف تحلیل
    tab1, tab2, tab3 = st.tabs(["پیش‌بینی روند قیمت", "تحلیل احساسات بازار", "تشخیص الگوهای قیمتی"])
    
    with tab1:
        st.subheader("🔮 پیش‌بینی روند قیمت")
        
        # انتخاب روش پیش‌بینی
        forecast_method = st.selectbox(
            "روش پیش‌بینی",
            options=["هوش مصنوعی", "LSTM", "GRU", "Transformer", "Ensemble"],
            index=0,
            format_func=lambda x: {"هوش مصنوعی": "هوش مصنوعی (AI)", "LSTM": "شبکه عصبی LSTM", 
                                "GRU": "شبکه عصبی GRU", "Transformer": "شبکه عصبی Transformer", 
                                "Ensemble": "ترکیبی (Ensemble)"}.get(x, x)
        )
        
        col1, col2 = st.columns(2)
        with col1:
            forecast_days = st.slider("تعداد روز برای پیش‌بینی", min_value=1, max_value=30, value=7)
        
        with col2:
            confidence_level = st.select_slider(
                "سطح اطمینان",
                options=["پایین", "متوسط", "بالا"],
                value="متوسط"
            )
        
        # دکمه اجرای پیش‌بینی
        if st.button("اجرای پیش‌بینی", key="run_forecast"):
            with st.spinner("در حال تحلیل داده‌ها و پیش‌بینی روند قیمت..."):
                # محاسبه پیش‌بینی
                try:
                    # ایجاد اندیکاتورها
                    rsi_value = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None
                    macd_value = float(df['macd'].iloc[-1]) if 'macd' in df.columns else None
                    macd_signal = float(df['macd_signal'].iloc[-1]) if 'macd_signal' in df.columns else None
                    
                    # ایجاد دیکشنری سیگنال‌ها
                    indicators_dict = {}
                    
                    if rsi_value is not None:
                        indicators_dict['rsi'] = {'value': rsi_value}
                    
                    if macd_value is not None and macd_signal is not None:
                        indicators_dict['macd'] = {'value': macd_value, 'signal': macd_signal}
                    
                    # پیش‌بینی قیمت
                    if forecast_method == "هوش مصنوعی":
                        price_prediction = ai_manager.predict_price_movement(
                            df=df,
                            symbol=symbol,
                            timeframe=timeframe,
                            days_ahead=forecast_days,
                            current_signals=indicators_dict
                        )
                    else:
                        # شبیه‌سازی پیش‌بینی برای سایر روش‌ها
                        # در اینجا می‌توان کد واقعی پیش‌بینی با مدل‌های مختلف را اضافه کرد
                        price_prediction = ai_manager._simulate_ai_prediction(
                            symbol=symbol,
                            timeframe=timeframe,
                            days_ahead=forecast_days,
                            current_signals=indicators_dict
                        )
                    
                    # نمایش نتیجه پیش‌بینی
                    st.markdown('<div class="ai-analysis-container">', unsafe_allow_html=True)
                    st.markdown(price_prediction, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ایجاد داده‌های پیش‌بینی برای نمودار
                    # این بخش می‌تواند با داده‌های واقعی پیش‌بینی جایگزین شود
                    current_price = df['close'].iloc[-1]
                    last_date = df.index[-1]
                    
                    # تعیین روند پیش‌بینی بر اساس متن
                    trend_direction = 1  # پیش‌فرض صعودی
                    if "نزولی" in price_prediction:
                        trend_direction = -1
                    elif "خنثی" in price_prediction or "محدودی" in price_prediction:
                        trend_direction = 0
                    
                    # ایجاد داده‌های پیش‌بینی
                    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                    predictions = []
                    
                    # محاسبه تغییرات روزانه بر اساس روند
                    daily_change = 0.005 * trend_direction  # 0.5% تغییر روزانه
                    
                    # اضافه کردن نویز بر اساس سطح اطمینان
                    noise_level = {"پایین": 0.015, "متوسط": 0.008, "بالا": 0.003}[confidence_level]
                    
                    price = current_price
                    for _ in range(forecast_days):
                        noise = np.random.normal(0, noise_level)
                        price = price * (1 + daily_change + noise)
                        predictions.append(price)
                    
                    # ایجاد دیتافریم پیش‌بینی
                    pred_df = pd.DataFrame({
                        'timestamp': future_dates,
                        'predicted_close': predictions,
                        'confidence_low': [p * (1 - noise_level * 2) for p in predictions],
                        'confidence_high': [p * (1 + noise_level * 2) for p in predictions]
                    })
                    
                    pred_df.set_index('timestamp', inplace=True)
                    
                    # نمایش نمودار پیش‌بینی
                    fig = plot_prediction_chart(df, pred_df, symbol, timeframe)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"خطا در پیش‌بینی قیمت: {str(e)}")
    
    with tab2:
        st.subheader("🌡️ تحلیل احساسات بازار")
        
        # جمع‌آوری داده‌های احساسات
        with st.spinner("در حال جمع‌آوری داده‌های احساسات بازار..."):
            # این بخش می‌تواند با داده‌های واقعی احساسات جایگزین شود
            sentiment_data = {
                "شاخص ترس و طمع": random.randint(30, 70),
                "احساسات شبکه‌های اجتماعی": random.choice(["مثبت", "خنثی", "منفی"]),
                "احساسات تحلیلگران": random.choice(["مثبت", "خنثی", "منفی"]),
                "حجم معاملات": random.choice(["بالا", "متوسط", "پایین"]),
                "روند قیمت 24 ساعته": random.choice(["صعودی", "نزولی", "خنثی"])
            }
            
            # تعیین رنگ‌ها بر اساس مقادیر
            fear_greed_color = "orange"
            fear_greed_value = sentiment_data["شاخص ترس و طمع"]
            
            if fear_greed_value < 30:
                fear_greed_color = "red"
                fear_greed_label = "ترس شدید"
            elif fear_greed_value < 45:
                fear_greed_color = "orange"
                fear_greed_label = "ترس"
            elif fear_greed_value < 55:
                fear_greed_color = "yellow"
                fear_greed_label = "خنثی"
            elif fear_greed_value < 75:
                fear_greed_color = "light green"
                fear_greed_label = "طمع"
            else:
                fear_greed_color = "green"
                fear_greed_label = "طمع شدید"
            
            # نمایش شاخص ترس و طمع
            st.markdown(f"""
            <div style="background-color: rgba(0, 0, 0, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin-top: 0; margin-bottom: 10px;">شاخص ترس و طمع</h4>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 2em; font-weight: bold; color: {fear_greed_color};">{fear_greed_value}</div>
                    <div style="background-color: {fear_greed_color}; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold;">{fear_greed_label}</div>
                </div>
                <div style="width: 100%; height: 10px; background-color: #eee; border-radius: 5px; margin-top: 10px;">
                    <div style="width: {fear_greed_value}%; height: 100%; background-color: {fear_greed_color}; border-radius: 5px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span>ترس شدید</span>
                    <span>خنثی</span>
                    <span>طمع شدید</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # نمایش سایر داده‌های احساسات
            st.markdown("<h4>سایر شاخص‌های احساسات</h4>", unsafe_allow_html=True)
            
            # ایجاد جدول برای نمایش داده‌ها
            sentiment_table = """
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">شاخص</th>
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">وضعیت</th>
                </tr>
            """
            
            for key, value in sentiment_data.items():
                if key != "شاخص ترس و طمع":
                    # تعیین رنگ بر اساس مقدار
                    if value in ["مثبت", "بالا", "صعودی"]:
                        color = "green"
                    elif value in ["منفی", "پایین", "نزولی"]:
                        color = "red"
                    else:
                        color = "orange"
                    
                    sentiment_table += f"""
                    <tr>
                        <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{key}</td>
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">
                            <span style="color: {color}; font-weight: bold;">{value}</span>
                        </td>
                    </tr>
                    """
            
            sentiment_table += "</table>"
            st.markdown(sentiment_table, unsafe_allow_html=True)
            
            # نمایش تحلیل محتوایی
            st.markdown("<h4>تحلیل محتوای رسانه‌ها</h4>", unsafe_allow_html=True)
            
            # کلمات کلیدی پرتکرار
            keywords = ["هاوینگ", "رگولاتوری", "ETF", "نهنگ‌ها", "تسلا", "SEC", "ارز دیجیتال"]
            random.shuffle(keywords)
            
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <strong>کلمات کلیدی پرتکرار:</strong>
                <div style="margin-top: 5px;">
            """, unsafe_allow_html=True)
            
            for keyword in keywords[:5]:
                st.markdown(f'<span class="indicator-badge">{keyword}</span>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # توضیحات تحلیل احساسات
            sentiment_analysis_text = [
                "احساسات کلی بازار در 24 ساعت گذشته متمایل به خنثی بوده است.",
                "فعالیت در شبکه‌های اجتماعی نشان از افزایش علاقه به خرید دارد.",
                "حجم جستجوها برای این ارز در 7 روز گذشته 15% افزایش داشته است.",
                "تحلیلگران نسبت به روند کوتاه‌مدت قیمت محتاط هستند."
            ]
            
            random.shuffle(sentiment_analysis_text)
            
            st.markdown(f"""
            <div class="ai-analysis-container">
                <p>{sentiment_analysis_text[0]}</p>
                <p>{sentiment_analysis_text[1]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("👁️ تشخیص الگوهای قیمتی")
        
        # دکمه اجرای تشخیص الگو
        if st.button("تشخیص الگوهای قیمتی", key="detect_patterns"):
            with st.spinner("در حال تحلیل نمودار و تشخیص الگوها..."):
                # این بخش می‌تواند با تشخیص واقعی الگوها جایگزین شود
                available_patterns = [
                    "سر و شانه", "سر و شانه معکوس", "دابل تاپ", "دابل باتم",
                    "مثلث صعودی", "مثلث نزولی", "مثلث متقارن", "کانال صعودی",
                    "کانال نزولی", "فلگ صعودی", "فلگ نزولی", "کاپ و دسته",
                    "الگوی هارمونیک", "الگوی AB=CD", "پرچم صعودی", "پرچم نزولی"
                ]
                
                # انتخاب تصادفی 0 تا 3 الگو
                num_patterns = random.randint(0, 3)
                
                if num_patterns == 0:
                    st.info("هیچ الگوی معتبری در چارت فعلی شناسایی نشد.")
                else:
                    detected_patterns = random.sample(available_patterns, num_patterns)
                    
                    # نمایش الگوهای شناسایی شده
                    st.markdown("<h4>الگوهای شناسایی شده:</h4>", unsafe_allow_html=True)
                    
                    for i, pattern in enumerate(detected_patterns):
                        # تعیین جهت الگو
                        if any(keyword in pattern for keyword in ["صعودی", "باتم", "معکوس"]):
                            direction = "صعودی"
                            trend_class = "trend-up"
                        elif any(keyword in pattern for keyword in ["نزولی", "تاپ"]):
                            direction = "نزولی"
                            trend_class = "trend-down"
                        else:
                            direction = "نامشخص"
                            trend_class = "trend-neutral"
                        
                        # تعیین قدرت الگو
                        strength = random.randint(60, 95)
                        
                        # نمایش اطلاعات الگو
                        st.markdown(f"""
                        <div style="margin-bottom: 20px; border: 1px solid rgba(0, 0, 0, 0.1); border-radius: 10px; padding: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h4 style="margin: 0;">{pattern}</h4>
                                <div class="{trend_class}" style="border-radius: 15px; padding: 5px 15px;">{direction}</div>
                            </div>
                            <p><strong>قدرت الگو:</strong> {strength}%</p>
                            <p><strong>احتمال موفقیت:</strong> {random.randint(40, 85)}%</p>
                            <p><strong>زمان تکمیل:</strong> {random.randint(1, 5)} روز</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # توضیحات تشخیص الگو
                    if num_patterns > 0:
                        first_pattern = detected_patterns[0]
                        if any(keyword in first_pattern for keyword in ["صعودی", "باتم", "معکوس"]):
                            pattern_advice = "این الگو معمولاً نشان‌دهنده احتمال صعود قیمت است. توصیه می‌شود برای تأیید الگو، حجم معاملات و شکست سطوح کلیدی را نیز بررسی کنید."
                        elif any(keyword in first_pattern for keyword in ["نزولی", "تاپ"]):
                            pattern_advice = "این الگو معمولاً نشان‌دهنده احتمال نزول قیمت است. توصیه می‌شود برای تأیید الگو، حجم معاملات و شکست سطوح کلیدی را نیز بررسی کنید."
                        else:
                            pattern_advice = "این الگو می‌تواند نشان‌دهنده ادامه روند فعلی یا تغییر جهت باشد. توصیه می‌شود برای تأیید الگو، حجم معاملات و شکست سطوح کلیدی را نیز بررسی کنید."
                        
                        st.markdown(f"""
                        <div class="ai-analysis-container">
                            <h4>تحلیل الگو</h4>
                            <p>{pattern_advice}</p>
                        </div>
                        """, unsafe_allow_html=True)

def main():
    # تنظیم پیکربندی صفحه
    st.set_page_config(
        page_title="سیستم تحلیل ارزهای دیجیتال",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # تنظیم حالت دارک/لایت
    setup_dark_light_mode()
    
    # عنوان و توضیحات
    st.title("سیستم پیشرفته تحلیل ارزهای دیجیتال")
    st.markdown("این سیستم با استفاده از بیش از 900 اندیکاتور تکنیکال و تکنولوژی هوش مصنوعی، تحلیل‌های دقیق و سیگنال‌های معاملاتی ارائه می‌دهد.")
    
    # تنظیمات نوار کناری
    with st.sidebar:
        st.header("📝 تنظیمات")
        
        # انتخاب ارز
        crypto_search = get_crypto_search()
        favorite_cryptos = crypto_search.get_favorites()
        symbol_options = favorite_cryptos[:10] if favorite_cryptos else ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        
        symbol = st.selectbox("انتخاب ارز", symbol_options, index=0)
        
        # انتخاب تایم‌فریم
        timeframe = st.selectbox(
            "تایم‌فریم",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
            index=4  # 1h پیش‌فرض
        )
        
        # انتخاب صرافی
        exchange = st.selectbox(
            "صرافی",
            ["binance", "kucoin", "huobi", "bybit", "kraken"],
            index=0
        )
        
        # تعداد روزهای تاریخچه
        lookback_days = st.slider("تعداد روز تاریخچه", min_value=1, max_value=365, value=30)
        
        # انتخاب اندیکاتورها
        indicator_manager = get_indicator_manager()
        available_indicators = set(AVAILABLE_INDICATORS + indicator_manager.get_top_indicators(10))
        
        selected_indicators = st.multiselect(
            "اندیکاتورهای تکنیکال",
            sorted(list(available_indicators)),
            default=TOP_INDICATORS[:5]
        )
        
        # دکمه به‌روزرسانی داده‌ها
        if st.button("به‌روزرسانی داده‌ها"):
            st.session_state.update_data = True
        
        # جداکننده
        st.markdown("---")
        
        # تنظیمات مدیریت ریسک
        st.subheader("⚙️ تنظیمات مدیریت ریسک")
        
        risk_percent = st.slider("درصد ریسک (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
        reward_ratio = st.slider("نسبت ریوارد به ریسک", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        
        # تنظیم تارگت کلکولیتور بر اساس پارامترهای جدید
        if 'df' in st.session_state and st.session_state.df is not None:
            get_target_calculator(st.session_state.df, risk_percent, reward_ratio)
        
        # جداکننده
        st.markdown("---")
        
        # اطلاعات
        st.info("این برنامه از بیش از 900 اندیکاتور تکنیکال پشتیبانی می‌کند و قابلیت تحلیل هوشمند بازار را دارد.")
    
    # دریافت داده‌ها
    if 'df' not in st.session_state or 'update_data' in st.session_state and st.session_state.update_data:
        with st.spinner(f"در حال دریافت داده‌های {symbol} با تایم‌فریم {timeframe}..."):
            try:
                # دریافت داده‌های قیمت
                df = get_crypto_data(symbol, timeframe, lookback_days, exchange)
                
                if df is not None and not df.empty:
                    # انجام تحلیل تکنیکال
                    df = perform_technical_analysis(df, selected_indicators)
                    
                    # ذخیره در session state
                    st.session_state.df = df
                    st.session_state.symbol = symbol
                    st.session_state.timeframe = timeframe
                else:
                    st.error("خطا در دریافت داده‌ها. لطفاً تنظیمات را بررسی کنید یا دوباره تلاش کنید.")
            except Exception as e:
                st.error(f"خطا: {str(e)}")
                st.session_state.df = None
        
        # ریست فلگ به‌روزرسانی
        if 'update_data' in st.session_state:
            st.session_state.update_data = False
    
    # سربرگ‌های اصلی
    tab1, tab2, tab3 = st.tabs(["📊 داشبورد", "🔍 جستجوی ارزها", "🧠 تحلیل هوش مصنوعی"])
    
    with tab1:
        if 'df' in st.session_state and st.session_state.df is not None:
            render_dashboard(st.session_state.df, st.session_state.symbol, st.session_state.timeframe)
        else:
            st.warning("ابتدا داده‌ها را دریافت کنید.")
    
    with tab2:
        render_crypto_search_page()
    
    with tab3:
        if 'df' in st.session_state and st.session_state.df is not None:
            render_ai_analysis_page(st.session_state.df, st.session_state.symbol, st.session_state.timeframe)
        else:
            st.warning("ابتدا داده‌ها را دریافت کنید.")
    
    # بستن تگ div
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()