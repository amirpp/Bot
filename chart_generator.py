"""
ماژول تولید نمودارهای تحلیل تکنیکال

این ماژول توابع مورد نیاز برای ایجاد نمودارهای تحلیل تکنیکال با استفاده از Plotly را ارائه می‌دهد.
"""

import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

def create_chart(
    df: pd.DataFrame, 
    symbol: str, 
    timeframe: str, 
    indicators: List[str],
    chart_type: str = "شمعی"
) -> go.Figure:
    """
    ایجاد نمودار قیمت با اندیکاتورهای انتخابی
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت و اندیکاتورها
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        indicators (List[str]): لیست اندیکاتورهای انتخابی
        chart_type (str): نوع نمودار (شمعی، خطی، OHLC، Heikin Ashi)
        
    Returns:
        go.Figure: نمودار ایجاد شده
    """
    if df.empty:
        # ایجاد یک نمودار خالی با پیام خطا
        fig = go.Figure()
        fig.add_annotation(
            text="داده‌ای برای نمایش وجود ندارد",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # تعیین تعداد ردیف‌های زیرنمودار
    num_rows = determine_subplot_rows(indicators)
    subplot_titles = determine_subplot_titles(indicators, symbol, timeframe)
    
    # ایجاد شکل با زیرنمودارها
    fig = sp.make_subplots(
        rows=num_rows, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=[0.6] + [0.4 / (num_rows - 1)] * (num_rows - 1) if num_rows > 1 else [1]
    )
    
    # افزودن نمودار اصلی قیمت
    add_main_price_chart(fig, df, chart_type)
    
    # افزودن اندیکاتورهای روی نمودار اصلی
    row_count = 1
    
    # افزودن میانگین‌های متحرک به نمودار اصلی
    if any(ind in indicators for ind in ['Moving Average', 'SMA', 'EMA']):
        add_moving_averages(fig, df, row=1)
    
    # افزودن باندهای بولینگر به نمودار اصلی
    if 'Bollinger Bands' in indicators:
        add_bollinger_bands(fig, df, row=1)
    
    # افزودن سوپرترند به نمودار اصلی
    if 'Supertrend' in indicators and 'supertrend' in df.columns:
        add_supertrend(fig, df, row=1)
    
    # افزودن اندیکاتورهای جداگانه به زیرنمودارها
    if 'RSI' in indicators and 'rsi' in df.columns:
        row_count += 1
        add_rsi(fig, df, row=row_count)
    
    if 'MACD' in indicators and 'macd' in df.columns:
        row_count += 1
        add_macd(fig, df, row=row_count)
    
    if 'Stochastic' in indicators and 'stoch_k' in df.columns:
        row_count += 1
        add_stochastic(fig, df, row=row_count)
    
    if 'ADX' in indicators and 'adx' in df.columns:
        row_count += 1
        add_adx(fig, df, row=row_count)
    
    if 'OBV' in indicators and 'obv' in df.columns:
        row_count += 1
        add_obv(fig, df, row=row_count)
    
    if 'Volume' in indicators:
        row_count += 1
        add_volume(fig, df, row=row_count)
    
    # تنظیم عنوان و سبک نمودار
    chart_title = f"{symbol} - {timeframe}"
    fig.update_layout(
        title=chart_title,
        template="plotly_white",
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # بهینه‌سازی نمایش
    fig.update_layout(hovermode="x unified")
    
    # به‌روزرسانی محورهای Y
    for i in range(1, num_rows + 1):
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.3)',
            row=i, col=1
        )
    
    # به‌روزرسانی محور X
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(230, 230, 230, 0.3)',
        row=num_rows, col=1
    )
    
    return fig

def determine_subplot_rows(indicators: List[str]) -> int:
    """
    تعیین تعداد ردیف‌های زیرنمودار
    
    Args:
        indicators (List[str]): لیست اندیکاتورهای انتخابی
        
    Returns:
        int: تعداد ردیف‌های زیرنمودار
    """
    # تعداد اندیکاتورهایی که نیاز به زیرنمودار جداگانه دارند
    separate_indicators = [
        'RSI', 'MACD', 'Stochastic', 'ADX', 'OBV', 'Volume'
    ]
    
    count = 1  # نمودار اصلی
    
    for indicator in separate_indicators:
        if indicator in indicators:
            count += 1
    
    return count

def determine_subplot_titles(indicators: List[str], symbol: str, timeframe: str) -> List[str]:
    """
    تعیین عناوین زیرنمودارها
    
    Args:
        indicators (List[str]): لیست اندیکاتورهای انتخابی
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
        
    Returns:
        List[str]: لیست عناوین زیرنمودارها
    """
    titles = [f"{symbol} - {timeframe}"]
    
    # اندیکاتورهایی که نیاز به زیرنمودار جداگانه دارند
    indicator_map = {
        'RSI': 'RSI - شاخص قدرت نسبی',
        'MACD': 'MACD - واگرایی و همگرایی میانگین متحرک',
        'Stochastic': 'Stochastic - اسیلاتور استوکاستیک',
        'ADX': 'ADX - شاخص روند متوسط جهت‌دار',
        'OBV': 'OBV - حجم در تعادل',
        'Volume': 'Volume - حجم معاملات'
    }
    
    for indicator in indicator_map:
        if indicator in indicators:
            titles.append(indicator_map[indicator])
    
    return titles

def add_main_price_chart(fig: go.Figure, df: pd.DataFrame, chart_type: str, row: int = 1, col: int = 1) -> None:
    """
    افزودن نمودار اصلی قیمت
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        chart_type (str): نوع نمودار
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    if chart_type == "شمعی":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="قیمت",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=row, col=col
        )
    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="قیمت",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=row, col=col
        )
    elif chart_type == "Heikin Ashi":
        # محاسبه کندل‌های Heikin Ashi
        ha_open = np.zeros(len(df))
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_high = np.zeros(len(df))
        ha_low = np.zeros(len(df))
        
        ha_open[0] = df['open'].iloc[0]
        
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high[i] = max(df['high'].iloc[i], ha_open[i], ha_close[i])
            ha_low[i] = min(df['low'].iloc[i], ha_open[i], ha_close[i])
        
        # اصلاح مقدار اولین عنصر در آرایه‌ها
        ha_high[0] = df['high'].iloc[0]
        ha_low[0] = df['low'].iloc[0]
        
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=ha_open,
                high=ha_high,
                low=ha_low,
                close=ha_close,
                name="Heikin Ashi",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=row, col=col
        )
    else:  # خطی
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name="قیمت",
                line=dict(color='#2196f3', width=2)
            ),
            row=row, col=col
        )

def add_moving_averages(fig: go.Figure, df: pd.DataFrame, row: int = 1, col: int = 1) -> None:
    """
    افزودن میانگین‌های متحرک به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    # میانگین متحرک ساده
    if 'sma20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['sma20'],
                name="SMA 20",
                line=dict(color='#FF9800', width=1)
            ),
            row=row, col=col
        )
    
    if 'sma50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['sma50'],
                name="SMA 50",
                line=dict(color='#2196F3', width=1)
            ),
            row=row, col=col
        )
    
    if 'sma200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['sma200'],
                name="SMA 200",
                line=dict(color='#F44336', width=1.5)
            ),
            row=row, col=col
        )
    
    # میانگین متحرک نمایی
    if 'ema20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema20'],
                name="EMA 20",
                line=dict(color='#FF9800', width=1, dash='dash')
            ),
            row=row, col=col
        )
    
    if 'ema50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema50'],
                name="EMA 50",
                line=dict(color='#2196F3', width=1, dash='dash')
            ),
            row=row, col=col
        )
    
    if 'ema200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema200'],
                name="EMA 200",
                line=dict(color='#F44336', width=1.5, dash='dash')
            ),
            row=row, col=col
        )

def add_bollinger_bands(fig: go.Figure, df: pd.DataFrame, row: int = 1, col: int = 1) -> None:
    """
    افزودن باندهای بولینگر به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
        # باند بالایی
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name="BB Upper",
                line=dict(color='rgba(255, 152, 0, 0.7)', width=1),
                legendgroup="bollinger"
            ),
            row=row, col=col
        )
        
        # باند میانی
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_middle'],
                name="BB Middle",
                line=dict(color='rgba(33, 150, 243, 0.7)', width=1),
                legendgroup="bollinger"
            ),
            row=row, col=col
        )
        
        # باند پایینی
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name="BB Lower",
                line=dict(color='rgba(255, 152, 0, 0.7)', width=1),
                legendgroup="bollinger",
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.1)'
            ),
            row=row, col=col
        )

def add_supertrend(fig: go.Figure, df: pd.DataFrame, row: int = 1, col: int = 1) -> None:
    """
    افزودن سوپرترند به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    if 'supertrend' in df.columns and 'supertrend_direction' in df.columns:
        # ایجاد آرایه‌ای با مقادیر جهت سوپرترند
        supertrend_value = df['supertrend'].copy()
        
        # تغییر رنگ سوپرترند بر اساس جهت
        buy_idx = df.index[df['supertrend_direction'] == 1]
        sell_idx = df.index[df['supertrend_direction'] == -1]
        
        # اضافه کردن سوپرترند صعودی
        if len(buy_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_idx,
                    y=supertrend_value.loc[buy_idx],
                    name="Supertrend Buy",
                    mode='lines',
                    line=dict(color='rgba(38, 166, 154, 0.8)', width=2),
                    legendgroup="supertrend"
                ),
                row=row, col=col
            )
        
        # اضافه کردن سوپرترند نزولی
        if len(sell_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_idx,
                    y=supertrend_value.loc[sell_idx],
                    name="Supertrend Sell",
                    mode='lines',
                    line=dict(color='rgba(239, 83, 80, 0.8)', width=2),
                    legendgroup="supertrend"
                ),
                row=row, col=col
            )

def add_rsi(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    """
    افزودن RSI به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    # خط RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            name="RSI",
            line=dict(color='#2196F3', width=1.5)
        ),
        row=row, col=col
    )
    
    # خطوط اشباع خرید و فروش
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[70, 70],
            name="Overbought",
            line=dict(color='rgba(239, 83, 80, 0.7)', width=1, dash='dash'),
            legendgroup="rsi_lines"
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[30, 30],
            name="Oversold",
            line=dict(color='rgba(38, 166, 154, 0.7)', width=1, dash='dash'),
            legendgroup="rsi_lines"
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[50, 50],
            name="RSI Mid",
            line=dict(color='rgba(158, 158, 158, 0.5)', width=1, dash='dot'),
            legendgroup="rsi_lines"
        ),
        row=row, col=col
    )
    
    # تنظیم محور Y
    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        row=row, col=col
    )

def add_macd(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    """
    افزودن MACD به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    # خط MACD
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd'],
            name="MACD",
            line=dict(color='#2196F3', width=1.5)
        ),
        row=row, col=col
    )
    
    # خط سیگنال
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd_signal'],
            name="Signal",
            line=dict(color='#FF9800', width=1.5)
        ),
        row=row, col=col
    )
    
    # هیستوگرام
    if 'macd_diff' in df.columns:
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['macd_diff']]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['macd_diff'],
                name="Histogram",
                marker_color=colors,
                opacity=0.7
            ),
            row=row, col=col
        )
    
    # خط صفر
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[0, 0],
            name="Zero Line",
            line=dict(color='rgba(158, 158, 158, 0.5)', width=1, dash='dot'),
            legendgroup="macd_lines"
        ),
        row=row, col=col
    )
    
    # تنظیم محور Y
    fig.update_yaxes(
        title_text="MACD",
        row=row, col=col
    )

def add_stochastic(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    """
    افزودن Stochastic به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    # خط K
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['stoch_k'],
            name="%K",
            line=dict(color='#2196F3', width=1.5)
        ),
        row=row, col=col
    )
    
    # خط D
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['stoch_d'],
            name="%D",
            line=dict(color='#FF9800', width=1.5)
        ),
        row=row, col=col
    )
    
    # خطوط اشباع خرید و فروش
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[80, 80],
            name="Overbought",
            line=dict(color='rgba(239, 83, 80, 0.7)', width=1, dash='dash'),
            legendgroup="stoch_lines"
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[20, 20],
            name="Oversold",
            line=dict(color='rgba(38, 166, 154, 0.7)', width=1, dash='dash'),
            legendgroup="stoch_lines"
        ),
        row=row, col=col
    )
    
    # تنظیم محور Y
    fig.update_yaxes(
        title_text="Stochastic",
        range=[0, 100],
        row=row, col=col
    )

def add_adx(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    """
    افزودن ADX به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    # خط ADX
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['adx'],
            name="ADX",
            line=dict(color='#673AB7', width=2)
        ),
        row=row, col=col
    )
    
    # خطوط +DI و -DI
    if 'adx_pos' in df.columns and 'adx_neg' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx_pos'],
                name="+DI",
                line=dict(color='#26a69a', width=1.5)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx_neg'],
                name="-DI",
                line=dict(color='#ef5350', width=1.5)
            ),
            row=row, col=col
        )
    
    # خط قدرت روند
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[25, 25],
            name="Trend Strength",
            line=dict(color='rgba(158, 158, 158, 0.5)', width=1, dash='dash'),
            legendgroup="adx_lines"
        ),
        row=row, col=col
    )
    
    # تنظیم محور Y
    fig.update_yaxes(
        title_text="ADX",
        range=[0, 100],
        row=row, col=col
    )

def add_obv(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    """
    افزودن OBV به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    # محاسبه میانگین متحرک OBV
    obv_ma = df['obv'].rolling(window=20).mean()
    
    # خط OBV
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['obv'],
            name="OBV",
            line=dict(color='#2196F3', width=1.5)
        ),
        row=row, col=col
    )
    
    # میانگین متحرک OBV
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=obv_ma,
            name="OBV MA(20)",
            line=dict(color='#FF9800', width=1.5, dash='dash')
        ),
        row=row, col=col
    )
    
    # تنظیم محور Y
    fig.update_yaxes(
        title_text="OBV",
        row=row, col=col
    )

def add_volume(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    """
    افزودن حجم معاملات به نمودار
    
    Args:
        fig (go.Figure): شکل نمودار
        df (pd.DataFrame): دیتافریم داده‌ها
        row (int): ردیف در زیرنمودارها
        col (int): ستون در زیرنمودارها
    """
    # تعیین رنگ بر اساس تغییر قیمت
    colors = []
    for i in range(len(df)):
        if i > 0:
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                colors.append('#26a69a')  # سبز
            else:
                colors.append('#ef5350')  # قرمز
        else:
            colors.append('#26a69a')  # سبز برای اولین عنصر
    
    # محاسبه میانگین متحرک حجم
    volume_ma = df['volume'].rolling(window=20).mean()
    
    # نمودار حجم
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=row, col=col
    )
    
    # میانگین متحرک حجم
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=volume_ma,
            name="Volume MA(20)",
            line=dict(color='#673AB7', width=1.5)
        ),
        row=row, col=col
    )
    
    # تنظیم محور Y
    fig.update_yaxes(
        title_text="Volume",
        row=row, col=col
    )
