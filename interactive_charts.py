"""
ماژول نمودارهای تعاملی برای تحلیل ارزهای دیجیتال

این ماژول شامل توابع نمایش نمودارهای تعاملی با استفاده از Plotly است.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def create_candlestick_chart(df, indicators=None, chart_title=None, show_volume=True, patterns=None, 
                             support_resistance=None, predictions=None):
    """
    ایجاد نمودار شمعی تعاملی با استفاده از Plotly
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        indicators (list): لیست اندیکاتورهای مورد نظر
        chart_title (str): عنوان نمودار
        show_volume (bool): نمایش حجم معاملات
        patterns (list): لیست الگوهای شناسایی شده
        support_resistance (dict): دیکشنری سطوح حمایت و مقاومت
        predictions (dict): دیکشنری پیش‌بینی‌های قیمت
    
    Returns:
        go.Figure: نمودار plotly
    """
    # بررسی داده‌های ورودی
    if df is None or df.empty:
        return None
    
    # تنظیم لیست اندیکاتورها اگر هیچ اندیکاتوری مشخص نشده باشد
    if indicators is None:
        indicators = []
    
    # تشخیص اندیکاتورهای موجود در دیتافریم
    available_indicators = {}
    indicator_rows = 1  # تعداد ردیف‌های مورد نیاز برای اندیکاتورها
    
    # اندیکاتورهایی که در نمودار اصلی نمایش داده می‌شوند
    main_indicators = []
    # اندیکاتورهایی که در نمودارهای جداگانه نمایش داده می‌شوند
    separate_indicators = []
    
    # بررسی اندیکاتورهای فعال در دیتافریم
    if 'rsi' in df.columns and 'RSI' in indicators:
        separate_indicators.append('RSI')
        indicator_rows += 1
    
    if 'macd' in df.columns and 'macd_signal' in df.columns and 'MACD' in indicators:
        separate_indicators.append('MACD')
        indicator_rows += 1
    
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']) and 'Bollinger Bands' in indicators:
        main_indicators.append('Bollinger Bands')
    
    if 'ema' in df.columns and 'EMA' in indicators:
        main_indicators.append('EMA')
    
    if 'sma' in df.columns and 'SMA' in indicators:
        main_indicators.append('SMA')
    
    if all(col in df.columns for col in ['stoch_k', 'stoch_d']) and 'Stochastic' in indicators:
        separate_indicators.append('Stochastic')
        indicator_rows += 1
    
    if 'atr' in df.columns and 'ATR' in indicators:
        separate_indicators.append('ATR')
        indicator_rows += 1
    
    if 'obv' in df.columns and 'OBV' in indicators:
        separate_indicators.append('OBV')
        indicator_rows += 1
    
    if 'cci' in df.columns and 'CCI' in indicators:
        separate_indicators.append('CCI')
        indicator_rows += 1
    
    if 'mfi' in df.columns and 'MFI' in indicators:
        separate_indicators.append('MFI')
        indicator_rows += 1
    
    if all(col in df.columns for col in ['adx', 'dmp', 'dmn']) and 'ADX' in indicators:
        separate_indicators.append('ADX')
        indicator_rows += 1
    
    if 'willr' in df.columns and 'Williams %R' in indicators:
        separate_indicators.append('Williams %R')
        indicator_rows += 1
    
    if 'supertrend' in df.columns and 'Supertrend' in indicators:
        main_indicators.append('Supertrend')
    
    if 'vwap' in df.columns and 'VWAP' in indicators:
        main_indicators.append('VWAP')
    
    # نمایش حجم معاملات
    if show_volume and 'volume' in df.columns:
        indicator_rows += 1
    
    # ایجاد subplot با سایز مناسب
    row_heights = [0.5]
    
    # اضافه کردن ارتفاع برای حجم و اندیکاتورها
    if show_volume and 'volume' in df.columns:
        row_heights.append(0.1)  # حجم 10% از کل نمودار
    
    # اضافه کردن ارتفاع برای سایر اندیکاتورها
    for _ in separate_indicators:
        row_heights.append(0.15)  # هر اندیکاتور 15% از کل نمودار
    
    # ایجاد subplots با ارتفاع‌های مناسب
    total_rows = 1 + (1 if show_volume and 'volume' in df.columns else 0) + len(separate_indicators)
    specs = [[{"secondary_y": True}] for _ in range(total_rows)]
    
    # ایجاد نمودار
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, 
                        row_heights=row_heights, specs=specs, 
                        vertical_spacing=0.03)
    
    # تنظیم نام نمودار
    if chart_title:
        fig.update_layout(title=chart_title)
    
    # ایجاد نمودار شمعی
    candlestick = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='قیمت',
        increasing_line_color='#26a69a', 
        decreasing_line_color='#ef5350'
    )
    fig.add_trace(candlestick, row=1, col=1)
    
    # افزودن نشانگرهای خرید و فروش به نمودار اصلی
    # سیگنال‌های خرید
    buy_signals = {col: df[col] for col in df.columns if col.endswith('_buy_signal')}
    for signal_name, signal_values in buy_signals.items():
        if signal_values.sum() > 0:
            # ایجاد نقاط برای سیگنال‌های خرید (فقط نقاط با مقدار 1)
            signal_points = df[signal_values == 1].index
            signal_prices = df.loc[signal_points, 'low'] * 0.995  # کمی پایین‌تر از کندل
            
            indicator_name = signal_name.replace('_buy_signal', '')
            
            fig.add_trace(go.Scatter(
                x=signal_points,
                y=signal_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green',
                    line=dict(width=1, color='darkgreen')
                ),
                name=f'خرید {indicator_name}',
                hoverinfo='text',
                hovertext=[f"سیگنال خرید {indicator_name}<br>قیمت: {df.loc[point, 'close']:.2f}" for point in signal_points]
            ), row=1, col=1)
    
    # سیگنال‌های فروش
    sell_signals = {col: df[col] for col in df.columns if col.endswith('_sell_signal')}
    for signal_name, signal_values in sell_signals.items():
        if signal_values.sum() > 0:
            # ایجاد نقاط برای سیگنال‌های فروش (فقط نقاط با مقدار 1)
            signal_points = df[signal_values == 1].index
            signal_prices = df.loc[signal_points, 'high'] * 1.005  # کمی بالاتر از کندل
            
            indicator_name = signal_name.replace('_sell_signal', '')
            
            fig.add_trace(go.Scatter(
                x=signal_points,
                y=signal_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red',
                    line=dict(width=1, color='darkred')
                ),
                name=f'فروش {indicator_name}',
                hoverinfo='text',
                hovertext=[f"سیگنال فروش {indicator_name}<br>قیمت: {df.loc[point, 'close']:.2f}" for point in signal_points]
            ), row=1, col=1)
    
    # افزودن اندیکاتورهای نمودار اصلی
    for indicator in main_indicators:
        if indicator == 'Bollinger Bands':
            # باند بالایی
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                hoverinfo='text',
                hovertext=[f"BB Upper: {val:.2f}" for val in df['bb_upper']]
            ), row=1, col=1)
            
            # باند میانی
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['bb_middle'],
                name='BB Middle',
                line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                hoverinfo='text',
                hovertext=[f"BB Middle: {val:.2f}" for val in df['bb_middle']]
            ), row=1, col=1)
            
            # باند پایینی
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 204, 255, 0.2)',
                hoverinfo='text',
                hovertext=[f"BB Lower: {val:.2f}" for val in df['bb_lower']]
            ), row=1, col=1)
        
        elif indicator == 'EMA':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['ema'],
                name='EMA',
                line=dict(color='orange', width=1.5),
                hoverinfo='text',
                hovertext=[f"EMA: {val:.2f}" for val in df['ema']]
            ), row=1, col=1)
        
        elif indicator == 'SMA':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['sma'],
                name='SMA',
                line=dict(color='purple', width=1.5),
                hoverinfo='text',
                hovertext=[f"SMA: {val:.2f}" for val in df['sma']]
            ), row=1, col=1)
            
        elif indicator == 'Supertrend':
            # فیلتر کردن سوپرترند به بخش‌های صعودی و نزولی
            bullish = df[df['supertrend_direction'] == 1]
            bearish = df[df['supertrend_direction'] == -1]
            
            # سوپرترند صعودی (سبز)
            if not bullish.empty:
                fig.add_trace(go.Scatter(
                    x=bullish.index,
                    y=bullish['supertrend'],
                    name='Supertrend (Bullish)',
                    line=dict(color='green', width=1.5),
                    hoverinfo='text',
                    hovertext=[f"Supertrend (Bullish): {val:.2f}" for val in bullish['supertrend']]
                ), row=1, col=1)
            
            # سوپرترند نزولی (قرمز)
            if not bearish.empty:
                fig.add_trace(go.Scatter(
                    x=bearish.index,
                    y=bearish['supertrend'],
                    name='Supertrend (Bearish)',
                    line=dict(color='red', width=1.5),
                    hoverinfo='text',
                    hovertext=[f"Supertrend (Bearish): {val:.2f}" for val in bearish['supertrend']]
                ), row=1, col=1)
        
        elif indicator == 'VWAP':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['vwap'],
                name='VWAP',
                line=dict(color='purple', width=1.5, dash='dash'),
                hoverinfo='text',
                hovertext=[f"VWAP: {val:.2f}" for val in df['vwap']]
            ), row=1, col=1)
    
    # اضافه کردن خطوط حمایت و مقاومت
    if support_resistance and 'support' in support_resistance and 'resistance' in support_resistance:
        # سطوح مقاومت (قرمز)
        for level in support_resistance['resistance']:
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=level,
                x1=df.index[-1],
                y1=level,
                line=dict(color="red", width=1, dash="dash"),
                row=1,
                col=1
            )
            # متن توضیحی
            fig.add_annotation(
                x=df.index[-1],
                y=level,
                text=f"R: {level:.2f}",
                showarrow=False,
                xshift=50,
                font=dict(size=10, color="red"),
                row=1,
                col=1
            )
        
        # سطوح حمایت (سبز)
        for level in support_resistance['support']:
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=level,
                x1=df.index[-1],
                y1=level,
                line=dict(color="green", width=1, dash="dash"),
                row=1,
                col=1
            )
            # متن توضیحی
            fig.add_annotation(
                x=df.index[-1],
                y=level,
                text=f"S: {level:.2f}",
                showarrow=False,
                xshift=50,
                font=dict(size=10, color="green"),
                row=1,
                col=1
            )
    
    # اضافه کردن پیش‌بینی‌ها
    if predictions and 'forecasted_values' in predictions:
        # تنظیم ایندکس برای پیش‌بینی
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=len(predictions['forecasted_values']) + 1, freq='D')[1:]
        
        # اضافه کردن خط پیش‌بینی (میانگین)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=predictions['forecasted_values'],
            name='پیش‌بینی قیمت',
            line=dict(color='blue', width=2),
            mode='lines+markers',
            hoverinfo='text',
            hovertext=[f"پیش‌بینی روز {i+1}: {val:.2f}" for i, val in enumerate(predictions['forecasted_values'])]
        ), row=1, col=1)
        
        # اگر حدود اطمینان نیز محاسبه شده‌اند
        if 'upper_bound' in predictions and 'lower_bound' in predictions:
            # حد بالا
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=predictions['upper_bound'],
                name='حد بالای پیش‌بینی',
                line=dict(color='rgba(0, 0, 255, 0.3)', width=0),
                hoverinfo='text',
                hovertext=[f"حد بالا روز {i+1}: {val:.2f}" for i, val in enumerate(predictions['upper_bound'])]
            ), row=1, col=1)
            
            # حد پایین
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=predictions['lower_bound'],
                name='حد پایین پیش‌بینی',
                line=dict(color='rgba(0, 0, 255, 0.3)', width=0),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.1)',
                hoverinfo='text',
                hovertext=[f"حد پایین روز {i+1}: {val:.2f}" for i, val in enumerate(predictions['lower_bound'])]
            ), row=1, col=1)
    
    # نمایش حجم معاملات
    current_row = 2  # شماره ردیف اندیکاتور بعدی
    
    if show_volume and 'volume' in df.columns:
        # استفاده از رنگ‌بندی متفاوت برای افزایش و کاهش حجم
        colors = ['green' if df['close'].iloc[i] > df['close'].iloc[i-1] else 'red' for i in range(1, len(df))]
        colors.insert(0, 'green')  # رنگ برای اولین روز
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='حجم',
            marker=dict(color=colors, line=dict(color=colors, width=1)),
            hoverinfo='text',
            hovertext=[f"تاریخ: {idx}<br>حجم: {vol}" for idx, vol in zip(df.index, df['volume'])]
        ), row=current_row, col=1)
        
        # میانگین متحرک حجم
        if 'volume_sma' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['volume_sma'],
                name='میانگین حجم',
                line=dict(color='blue', width=1.5),
                hoverinfo='text',
                hovertext=[f"میانگین حجم: {val:.2f}" for val in df['volume_sma']]
            ), row=current_row, col=1)
        
        current_row += 1
    
    # اضافه کردن اندیکاتورهای جداگانه
    for indicator in separate_indicators:
        if indicator == 'RSI' and 'rsi' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple', width=1.5),
                hoverinfo='text',
                hovertext=[f"RSI: {val:.2f}" for val in df['rsi']]
            ), row=current_row, col=1)
            
            # خطوط اشباع خرید و فروش
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[70] * len(df),
                name='اشباع خرید',
                line=dict(color='red', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[30] * len(df),
                name='اشباع فروش',
                line=dict(color='green', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            # خط میانی
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[50] * len(df),
                name='خط میانی',
                line=dict(color='gray', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'MACD' and all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            # خط MACD
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=1.5),
                hoverinfo='text',
                hovertext=[f"MACD: {val:.2f}" for val in df['macd']]
            ), row=current_row, col=1)
            
            # خط سیگنال
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                name='سیگنال MACD',
                line=dict(color='red', width=1.5),
                hoverinfo='text',
                hovertext=[f"سیگنال MACD: {val:.2f}" for val in df['macd_signal']]
            ), row=current_row, col=1)
            
            # هیستوگرام
            colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['macd_histogram'],
                name='هیستوگرام MACD',
                marker=dict(color=colors),
                hoverinfo='text',
                hovertext=[f"هیستوگرام MACD: {val:.2f}" for val in df['macd_histogram']]
            ), row=current_row, col=1)
            
            # خط صفر
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[0] * len(df),
                name='خط صفر',
                line=dict(color='gray', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'Stochastic' and all(col in df.columns for col in ['stoch_k', 'stoch_d']):
            # خط K
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['stoch_k'],
                name='%K',
                line=dict(color='blue', width=1.5),
                hoverinfo='text',
                hovertext=[f"%K: {val:.2f}" for val in df['stoch_k']]
            ), row=current_row, col=1)
            
            # خط D
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['stoch_d'],
                name='%D',
                line=dict(color='red', width=1.5),
                hoverinfo='text',
                hovertext=[f"%D: {val:.2f}" for val in df['stoch_d']]
            ), row=current_row, col=1)
            
            # خطوط اشباع خرید و فروش
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[80] * len(df),
                name='اشباع خرید',
                line=dict(color='red', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[20] * len(df),
                name='اشباع فروش',
                line=dict(color='green', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'ATR' and 'atr' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['atr'],
                name='ATR',
                line=dict(color='orange', width=1.5),
                hoverinfo='text',
                hovertext=[f"ATR: {val:.2f}" for val in df['atr']]
            ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'OBV' and 'obv' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['obv'],
                name='OBV',
                line=dict(color='purple', width=1.5),
                hoverinfo='text',
                hovertext=[f"OBV: {val}" for val in df['obv']]
            ), row=current_row, col=1)
            
            # میانگین متحرک OBV
            if 'obv_ema' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['obv_ema'],
                    name='OBV EMA',
                    line=dict(color='blue', width=1, dash='dot'),
                    hoverinfo='text',
                    hovertext=[f"OBV EMA: {val}" for val in df['obv_ema']]
                ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'CCI' and 'cci' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['cci'],
                name='CCI',
                line=dict(color='green', width=1.5),
                hoverinfo='text',
                hovertext=[f"CCI: {val:.2f}" for val in df['cci']]
            ), row=current_row, col=1)
            
            # خطوط اشباع خرید و فروش
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[100] * len(df),
                name='اشباع خرید',
                line=dict(color='red', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[-100] * len(df),
                name='اشباع فروش',
                line=dict(color='green', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            # خط صفر
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[0] * len(df),
                name='خط صفر',
                line=dict(color='gray', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'MFI' and 'mfi' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['mfi'],
                name='MFI',
                line=dict(color='purple', width=1.5),
                hoverinfo='text',
                hovertext=[f"MFI: {val:.2f}" for val in df['mfi']]
            ), row=current_row, col=1)
            
            # خطوط اشباع خرید و فروش
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[80] * len(df),
                name='اشباع خرید',
                line=dict(color='red', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[20] * len(df),
                name='اشباع فروش',
                line=dict(color='green', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            # خط میانی
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[50] * len(df),
                name='خط میانی',
                line=dict(color='gray', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'ADX' and all(col in df.columns for col in ['adx', 'dmp', 'dmn']):
            # خط ADX
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['adx'],
                name='ADX',
                line=dict(color='black', width=1.5),
                hoverinfo='text',
                hovertext=[f"ADX: {val:.2f}" for val in df['adx']]
            ), row=current_row, col=1)
            
            # خط +DI
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['dmp'],
                name='+DI',
                line=dict(color='green', width=1.5),
                hoverinfo='text',
                hovertext=[f"+DI: {val:.2f}" for val in df['dmp']]
            ), row=current_row, col=1)
            
            # خط -DI
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['dmn'],
                name='-DI',
                line=dict(color='red', width=1.5),
                hoverinfo='text',
                hovertext=[f"-DI: {val:.2f}" for val in df['dmn']]
            ), row=current_row, col=1)
            
            # خط آستانه روند (25)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[25] * len(df),
                name='آستانه روند',
                line=dict(color='gray', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            current_row += 1
        
        elif indicator == 'Williams %R' and 'willr' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['willr'],
                name='Williams %R',
                line=dict(color='orange', width=1.5),
                hoverinfo='text',
                hovertext=[f"Williams %R: {val:.2f}" for val in df['willr']]
            ), row=current_row, col=1)
            
            # خطوط اشباع خرید و فروش
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[-20] * len(df),
                name='اشباع خرید',
                line=dict(color='red', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[-80] * len(df),
                name='اشباع فروش',
                line=dict(color='green', width=1, dash='dot'),
                hoverinfo='none'
            ), row=current_row, col=1)
            
            current_row += 1
    
    # تنظیمات کلی نمودار
    fig.update_layout(
        height=150 * total_rows + 100,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
    )
    
    # تنظیم رنگ پس‌زمینه
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
    )
    
    # تنظیم حاشیه‌ها برای هر subplot
    for i in range(1, total_rows + 1):
        fig.update_yaxes(title_text="", row=i, col=1)
    
    return fig

def create_comparison_chart(df_dict, metric='close', title=None):
    """
    ایجاد نمودار مقایسه‌ای برای ارزهای مختلف
    
    Args:
        df_dict (dict): دیکشنری از دیتافریم‌ها (کلید: نام ارز)
        metric (str): مشخصه مورد مقایسه (قیمت، حجم و غیره)
        title (str): عنوان نمودار
        
    Returns:
        go.Figure: نمودار plotly
    """
    if not df_dict or len(df_dict) == 0:
        return None
    
    fig = go.Figure()
    
    # اضافه کردن هر ارز به نمودار
    for symbol, df in df_dict.items():
        if metric in df.columns:
            # نرمال‌سازی مقادیر برای مقایسه بهتر
            first_value = df[metric].iloc[0] if not df.empty else 1
            normalized_values = (df[metric] / first_value) * 100
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_values,
                name=symbol,
                hoverinfo='text',
                hovertext=[f"{symbol}<br>تاریخ: {idx}<br>{metric}: {val:.2f}<br>تغییر: {(val/first_value-1)*100:.2f}%" for idx, val in zip(df.index, df[metric])]
            ))
    
    # تنظیمات نمودار
    fig.update_layout(
        title=title if title else "مقایسه عملکرد ارزها",
        xaxis_title="تاریخ",
        yaxis_title="درصد تغییر (نسبت به شروع: 100)",
        height=500,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # تنظیم رنگ پس‌زمینه
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
    )
    
    return fig

def create_correlation_heatmap(df_dict, days=30):
    """
    ایجاد نمودار حرارتی همبستگی بین ارزهای مختلف
    
    Args:
        df_dict (dict): دیکشنری از دیتافریم‌ها (کلید: نام ارز)
        days (int): تعداد روزهای اخیر برای محاسبه همبستگی
    
    Returns:
        go.Figure: نمودار plotly
    """
    if not df_dict or len(df_dict) == 0:
        return None
    
    # استخراج داده‌های قیمت روزانه
    price_data = {}
    
    for symbol, df in df_dict.items():
        if not df.empty and 'close' in df.columns:
            # گرفتن داده‌های n روز اخیر
            recent_data = df.tail(days)['close']
            price_data[symbol] = recent_data
    
    # ایجاد دیتافریم همبستگی
    price_df = pd.DataFrame(price_data)
    correlation = price_df.corr().round(2)
    
    # ایجاد نمودار حرارتی
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.index,
        colorscale='RdBu',
        text=correlation.values,
        hoverinfo='text',
        hovertext=[[f"{row} و {col}<br>همبستگی: {value:.2f}" for col, value in zip(correlation.columns, r)] for row, r in zip(correlation.index, correlation.values)],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="همبستگی")
    ))
    
    # تنظیمات نمودار
    fig.update_layout(
        title=f"همبستگی قیمت ارزها در {days} روز اخیر",
        height=600,
        width=700,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # تنظیم رنگ پس‌زمینه
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_multiple_indicators_chart(df, indicator_pairs=None):
    """
    ایجاد نمودار برای مقایسه اندیکاتورهای مختلف
    
    Args:
        df (pd.DataFrame): دیتافریم با اندیکاتورهای محاسبه شده
        indicator_pairs (list): لیست زوج‌های اندیکاتورها برای مقایسه
    
    Returns:
        go.Figure: نمودار plotly
    """
    if df is None or df.empty:
        return None
    
    # اندیکاتورهای پیش‌فرض برای مقایسه اگر مشخص نشده باشند
    if indicator_pairs is None:
        indicator_pairs = [
            ('rsi', 'cci'),
            ('macd', 'macd_signal'),
            ('ema', 'sma'),
            ('obv', 'volume')
        ]
    
    # فیلتر کردن زوج‌های موجود در دیتافریم
    available_pairs = []
    for pair in indicator_pairs:
        indicator1, indicator2 = pair
        if indicator1 in df.columns and indicator2 in df.columns:
            available_pairs.append(pair)
    
    if len(available_pairs) == 0:
        return None
    
    # تعیین تعداد ردیف‌ها و ستون‌ها
    n_rows = 2
    n_cols = 2
    
    # ایجاد subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True)
    
    # اضافه کردن اندیکاتورها به نمودار
    for i, pair in enumerate(available_pairs[:4]):  # حداکثر 4 زوج
        indicator1, indicator2 = pair
        
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # اندیکاتور اول
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[indicator1],
            name=indicator1,
            line=dict(color='blue'),
        ), row=row, col=col)
        
        # اندیکاتور دوم
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[indicator2],
            name=indicator2,
            line=dict(color='red'),
        ), row=row, col=col)
        
        # عنوان فرعی
        fig.update_yaxes(title_text=f"{indicator1} و {indicator2}", row=row, col=col)
    
    # تنظیمات کلی نمودار
    fig.update_layout(
        height=800,
        title="مقایسه اندیکاتورهای مختلف",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # تنظیم رنگ پس‌زمینه
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
    )
    
    return fig

def create_pie_chart(data, labels, title=None):
    """
    ایجاد نمودار دایره‌ای
    
    Args:
        data (list): لیست مقادیر
        labels (list): لیست برچسب‌ها
        title (str): عنوان نمودار
    
    Returns:
        go.Figure: نمودار plotly
    """
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=data,
        textinfo='percent',
        insidetextorientation='radial',
        hoverinfo='label+percent+value',
        marker=dict(colors=px.colors.sequential.Plasma_r)
    )])
    
    # تنظیمات نمودار
    fig.update_layout(
        title=title,
        height=400,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # تنظیم رنگ پس‌زمینه
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_radar_chart(data_dict, title=None):
    """
    ایجاد نمودار راداری
    
    Args:
        data_dict (dict): دیکشنری داده‌ها (کلید: نام، مقدار: دیکشنری از مشخصه‌ها)
        title (str): عنوان نمودار
    
    Returns:
        go.Figure: نمودار plotly
    """
    if not data_dict:
        return None
    
    fig = go.Figure()
    
    # استخراج همه برچسب‌ها
    all_labels = set()
    for data in data_dict.values():
        all_labels.update(data.keys())
    
    all_labels = list(all_labels)
    
    # اضافه کردن هر مجموعه داده
    for name, data in data_dict.items():
        # استخراج مقادیر برای همه برچسب‌ها
        values = [data.get(label, 0) for label in all_labels]
        
        # اضافه کردن مقدار اول به انتها برای بستن شکل
        values.append(values[0])
        labels = all_labels + [all_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=name
        ))
    
    # تنظیمات نمودار
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=title,
        height=500,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # تنظیم رنگ پس‌زمینه
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig
