def render_black_swan_page(df, symbol, timeframe):
    """
    رندر صفحه تشخیص رویدادهای مهم بازار (Black Swan Events)
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
    """
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    from black_swan_detector import detect_black_swan_events, get_black_swan_report
    
    st.header("⚠️ سیستم تشخیص و هشدار رویدادهای مهم بازار (Black Swan Events)")
    st.markdown("""
    سیستم هوشمند تشخیص و هشدار رویدادهای مهم بازار قادر است تا تغییرات غیرعادی قیمت، حجم معاملات و الگوهای قیمتی 
    که نشان‌دهنده رویدادهای مهم و تأثیرگذار بر بازار هستند را شناسایی کند.
    """)
    
    # ایجاد سه تب برای نمایش اطلاعات مختلف
    tab1, tab2, tab3 = st.tabs(["تشخیص رویدادهای مهم", "تاریخچه رویدادها", "تنظیمات هشدار"])
    
    with tab1:
        # دکمه برای اسکن دستی رویدادهای مهم
        if st.button("🔍 اسکن رویدادهای مهم", key="scan_black_swan"):
            with st.spinner("در حال تحلیل داده‌ها برای تشخیص رویدادهای مهم..."):
                try:
                    # فراخوانی تابع تشخیص رویدادهای مهم
                    black_swan_events = detect_black_swan_events(df, symbol)
                    
                    # دریافت گزارش جامع
                    black_swan_report = get_black_swan_report(df, symbol)
                    
                    # ذخیره در session state
                    st.session_state.black_swan_events = black_swan_events
                    st.session_state.black_swan_report = black_swan_report
                    
                    st.success(f"اسکن با موفقیت انجام شد! {len(black_swan_events)} رویداد مهم شناسایی شد.")
                except Exception as e:
                    st.error(f"خطا در اسکن رویدادهای مهم: {str(e)}")
        
        # نمایش نتایج اسکن
        if 'black_swan_events' in st.session_state and 'black_swan_report' in st.session_state:
            events = st.session_state.black_swan_events
            report = st.session_state.black_swan_report
            
            if len(events) > 0:
                # نمایش خلاصه گزارش
                risk_color = {
                    'بسیار بالا': '#ff3300',
                    'بالا': '#ff6600',
                    'متوسط': '#ff9900',
                    'کم': '#009933'
                }
                
                risk_level = report.get('risk_level', 'کم')
                risk_level_color = risk_color.get(risk_level, '#009933')
                
                st.markdown(f"""
                <div style="background-color: rgba(0, 0, 0, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0;">خلاصه گزارش</h3>
                    <p>{report.get('summary', 'اطلاعات در دسترس نیست.')}</p>
                    <p>
                        <strong>سطح ریسک:</strong> 
                        <span style="color: {risk_level_color}; font-weight: bold;">{risk_level}</span>
                    </p>
                    <p><strong>تعداد رویدادها:</strong> {report.get('event_count', 0)}</p>
                    <p><strong>امتیاز ریسک:</strong> {report.get('risk_score', 0)}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                # نمایش جزئیات رویدادها
                st.subheader("جزئیات رویدادهای شناسایی شده")
                
                for i, event in enumerate(events[:5]):  # نمایش 5 رویداد اول
                    event_type_fa = {
                        'volatility_anomaly': 'ناهنجاری نوسانات',
                        'volume_anomaly': 'ناهنجاری حجم',
                        'price_gap': 'شکاف قیمتی'
                    }
                    
                    event_type = event_type_fa.get(event['type'], 'نامشخص')
                    direction = event['direction']
                    severity = event['severity']
                    confidence = event['confidence'] * 100
                    timestamp = event['timestamp']
                    description = event['description']
                    
                    # تعیین رنگ کارت بر اساس شدت
                    if severity > 7:
                        card_color = "rgba(255, 0, 0, 0.1)"
                        border_color = "#ff0000"
                    elif severity > 5:
                        card_color = "rgba(255, 153, 0, 0.1)"
                        border_color = "#ff9900"
                    else:
                        card_color = "rgba(0, 153, 204, 0.1)"
                        border_color = "#0099cc"
                    
                    st.markdown(f"""
                    <div style="background-color: {card_color}; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid {border_color};">
                        <div style="display: flex; justify-content: space-between;">
                            <h4 style="margin-top: 0;">{event_type} {direction}</h4>
                            <span style="font-weight: bold;">{timestamp}</span>
                        </div>
                        <p>{description}</p>
                        <div style="display: flex; justify-content: space-between;">
                            <span><strong>شدت:</strong> {severity:.1f}/10</span>
                            <span><strong>اطمینان:</strong> {confidence:.1f}%</span>
                        </div>
                        <div style="margin-top: 10px;">
                            <strong>اقدامات پیشنهادی:</strong>
                            <ul style="margin-top: 5px;">
                    """, unsafe_allow_html=True)
                    
                    # نمایش اقدامات پیشنهادی
                    for action in event.get('action_items', [])[:3]:
                        st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div></div>", unsafe_allow_html=True)
                
                # نمایش نمودار با مشخص کردن زمان‌های رویدادهای مهم
                st.subheader("نمودار قیمت با رویدادهای مهم")
                
                fig = go.Figure()
                
                # اضافه کردن نمودار شمعی
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="قیمت",
                    increasing_line_color='#26a69a', 
                    decreasing_line_color='#ef5350'
                ))
                
                # اضافه کردن علامت‌های رویدادهای مهم
                event_dates = [e['timestamp'] for e in events]
                event_types = [e['type'] for e in events]
                event_severities = [e['severity'] for e in events]
                
                marker_colors = []
                marker_symbols = []
                
                for e_type, severity in zip(event_types, event_severities):
                    # انتخاب رنگ بر اساس شدت
                    if severity > 7:
                        color = 'rgba(255, 0, 0, 0.8)'
                    elif severity > 5:
                        color = 'rgba(255, 153, 0, 0.8)'
                    else:
                        color = 'rgba(0, 153, 204, 0.8)'
                    
                    # انتخاب نماد بر اساس نوع
                    if e_type == 'volatility_anomaly':
                        symbol = 'triangle-down'
                    elif e_type == 'volume_anomaly':
                        symbol = 'circle'
                    else:  # price_gap
                        symbol = 'x'
                    
                    marker_colors.append(color)
                    marker_symbols.append(symbol)
                
                # اضافه کردن نشانگرهای رویدادها در نمودار
                if event_dates:
                    event_prices = []
                    for date in event_dates:
                        # استخراج قیمت بالا برای نشانگر
                        try:
                            event_date_str = str(date)
                            if event_date_str in df.index:
                                event_prices.append(df.loc[event_date_str, 'high'] * 1.01)
                            else:
                                # اگر تاریخ دقیقاً مطابقت نداشت، نزدیکترین تاریخ را پیدا می‌کنیم
                                closest_date = min(df.index, key=lambda x: abs((pd.to_datetime(x) - pd.to_datetime(date)).total_seconds()))
                                event_prices.append(df.loc[closest_date, 'high'] * 1.01)
                        except Exception as e:
                            # اگر نتوانستیم قیمت را بیابیم، از میانگین قیمت‌ها استفاده می‌کنیم
                            event_prices.append(df['high'].mean())
                    
                    fig.add_trace(go.Scatter(
                        x=event_dates,
                        y=event_prices,
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbols,
                            size=12,
                            color=marker_colors,
                            line=dict(width=1, color='black')
                        ),
                        name="رویدادهای مهم"
                    ))
                
                fig.update_layout(
                    title=f"نمودار قیمت {symbol} با مشخص کردن رویدادهای مهم",
                    xaxis_title="تاریخ",
                    yaxis_title="قیمت",
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("هیچ رویداد مهمی در داده‌های فعلی شناسایی نشد.")
        else:
            st.info("برای شناسایی رویدادهای مهم، دکمه 'اسکن رویدادهای مهم' را کلیک کنید.")
    
    with tab2:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #888;">
            <h3>تاریخچه رویدادهای مهم</h3>
            <p>تاریخچه رویدادهای مهم شناسایی شده در این بخش نمایش داده خواهد شد.</p>
            <p>برای ثبت اولین رویداد، به تب 'تشخیص رویدادهای مهم' بروید و دکمه 'اسکن رویدادهای مهم' را کلیک کنید.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### تنظیمات هشدار رویدادهای مهم")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("فعال‌سازی هشدارهای خودکار", value=True, key="enable_black_swan_alerts")
            st.slider("حساسیت تشخیص", min_value=1, max_value=10, value=5, key="black_swan_sensitivity")
            st.selectbox("کانال ارسال هشدار", ["تلگرام", "ایمیل", "پیامک", "همه"], key="black_swan_alert_channel")
        
        with col2:
            st.multiselect("انواع رویدادهای قابل هشدار", 
                ["ناهنجاری نوسانات", "ناهنجاری حجم", "شکاف قیمتی", "همبستگی غیرعادی"], 
                default=["ناهنجاری نوسانات", "ناهنجاری حجم", "شکاف قیمتی"],
                key="black_swan_alert_types")
            st.number_input("حداقل شدت برای هشدار", min_value=1.0, max_value=10.0, value=6.0, step=0.5, key="black_swan_min_severity")
            st.checkbox("نمایش اعلان در تلفن همراه", value=True, key="black_swan_mobile_notification")
        
        if st.button("💾 ذخیره تنظیمات", key="save_black_swan_settings"):
            st.success("تنظیمات هشدار با موفقیت ذخیره شد!")
            

def render_strategy_comparison_advanced(df, symbol, timeframe):
    """
    رندر صفحه مقایسه استراتژی‌های پیشرفته و مدیریت خودکار
    
    Args:
        df (pd.DataFrame): دیتافریم داده‌های قیمت
        symbol (str): نماد ارز
        timeframe (str): تایم‌فریم
    """
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    from auto_strategy_manager import get_market_condition, generate_auto_strategy, compare_trading_strategies
    
    st.header("🧠 سیستم مدیریت خودکار استراتژی معاملاتی")
    
    # ایجاد سه تب برای بخش‌های مختلف
    tab1, tab2, tab3 = st.tabs(["تولید استراتژی", "مقایسه استراتژی‌ها", "تنظیمات"])
    
    with tab1:
        st.subheader("تولید استراتژی معاملاتی هوشمند")
        
        # تنظیمات پروفایل ریسک
        col1, col2 = st.columns(2)
        
        with col1:
            risk_profile = st.radio("پروفایل ریسک", ["کم‌ریسک", "متعادل", "پرریسک"], index=1)
            
        with col2:
            capital = st.number_input("سرمایه (USDT)", min_value=100.0, max_value=1000000.0, value=1000.0, step=100.0)
        
        # دکمه تولید استراتژی
        if st.button("🧠 تولید استراتژی هوشمند", key="generate_strategy"):
            with st.spinner("در حال تحلیل بازار و تولید استراتژی بهینه..."):
                try:
                    # دریافت شرایط بازار
                    market_condition = get_market_condition(df)
                    
                    # تولید استراتژی
                    strategy = generate_auto_strategy(df, symbol, risk_profile)
                    
                    # ذخیره در session state
                    st.session_state.market_condition = market_condition
                    st.session_state.auto_strategy = strategy
                    
                    st.success("استراتژی با موفقیت تولید شد!")
                except Exception as e:
                    st.error(f"خطا در تولید استراتژی: {str(e)}")
        
        # نمایش نتایج استراتژی
        if 'auto_strategy' in st.session_state and 'market_condition' in st.session_state:
            strategy = st.session_state.auto_strategy
            market_condition = st.session_state.market_condition
            
            # قالب‌بندی رنگ بر اساس سیگنال
            signal = strategy.get('signal', 'NEUTRAL')
            if signal == 'BUY':
                signal_color = '#00cc66'
                signal_fa = 'خرید (لانگ)'
                signal_bg = 'rgba(0, 204, 102, 0.1)'
                signal_border = '#00cc66'
            elif signal == 'SELL':
                signal_color = '#ff3300'
                signal_fa = 'فروش (شورت)'
                signal_bg = 'rgba(255, 51, 0, 0.1)'
                signal_border = '#ff3300'
            else:
                signal_color = '#999999'
                signal_fa = 'خنثی'
                signal_bg = 'rgba(153, 153, 153, 0.1)'
                signal_border = '#999999'
            
            # نمایش شرایط بازار
            st.markdown("### شرایط فعلی بازار")
            
            # قالب‌بندی وضعیت روند
            trend = market_condition.get('trend', 'نامشخص')
            if 'صعودی' in trend:
                trend_color = '#00cc66'
            elif 'نزولی' in trend:
                trend_color = '#ff3300'
            else:
                trend_color = '#999999'
            
            # نمایش شرایط بازار با استایل CSS
            st.markdown(f"""
            <div style="background-color: rgba(0, 0, 0, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                    <div>
                        <h4 style="margin-top: 0;">روند</h4>
                        <p style="color: {trend_color}; font-weight: bold; font-size: 1.2em;">{trend}</p>
                    </div>
                    <div>
                        <h4 style="margin-top: 0;">نوسانات</h4>
                        <p style="font-weight: bold; font-size: 1.2em;">{market_condition.get('volatility', 'متوسط')}</p>
                    </div>
                    <div>
                        <h4 style="margin-top: 0;">مومنتوم</h4>
                        <p style="font-weight: bold; font-size: 1.2em;">{market_condition.get('momentum', 'خنثی')}</p>
                    </div>
                </div>
                <p style="margin-top: 15px;">{market_condition.get('summary', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # نمایش استراتژی تولید شده
            st.markdown("### استراتژی پیشنهادی")
            
            # نمایش سیگنال معاملاتی
            st.markdown(f"""
            <div style="background-color: {signal_bg}; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid {signal_border};">
                <h3 style="margin-top: 0; color: {signal_color};">{signal_fa}</h3>
                <p>{strategy.get('description', 'توضیحات در دسترس نیست.')}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div>
                        <h4 style="margin-top: 0;">قیمت ورود</h4>
                        <p style="font-weight: bold; font-size: 1.2em;">{strategy.get('entry_price', 0):.2f} USDT</p>
                    </div>
                    <div>
                        <h4 style="margin-top: 0;">حد ضرر</h4>
                        <p style="font-weight: bold; font-size: 1.2em; color: #ff3300;">{strategy.get('stop_loss', 0):.2f} USDT</p>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <h4 style="margin-top: 0;">اهداف قیمتی</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                        <div>
                            <p>هدف اول:</p>
                            <p style="font-weight: bold; color: #00cc66;">{strategy.get('target_1', 0):.2f} USDT</p>
                        </div>
                        <div>
                            <p>هدف دوم:</p>
                            <p style="font-weight: bold; color: #00cc66;">{strategy.get('target_2', 0):.2f} USDT</p>
                        </div>
                        <div>
                            <p>هدف سوم:</p>
                            <p style="font-weight: bold; color: #00cc66;">{strategy.get('target_3', 0):.2f} USDT</p>
                        </div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div>
                        <h4 style="margin-top: 0;">نوع استراتژی</h4>
                        <p>{strategy.get('strategy_type', 'نامشخص')}</p>
                    </div>
                    <div>
                        <h4 style="margin-top: 0;">ریسک</h4>
                        <p>{strategy.get('risk_percentage', 0):.1f}%</p>
                    </div>
                    <div>
                        <h4 style="margin-top: 0;">اطمینان</h4>
                        <p>{strategy.get('confidence', 0):.1f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # نمایش نمودار با نقاط ورود و خروج
            st.subheader("نمودار با نقاط ورود و خروج")
            
            # ایجاد نمودار با نقاط ورود و خروج
            fig = go.Figure()
            
            # اضافه کردن نمودار شمعی
            fig.add_trace(go.Candlestick(
                x=df.index[-100:],  # 100 شمع آخر
                open=df['open'][-100:],
                high=df['high'][-100:],
                low=df['low'][-100:],
                close=df['close'][-100:],
                name="قیمت",
                increasing_line_color='#26a69a', 
                decreasing_line_color='#ef5350'
            ))
            
            # اضافه کردن خط قیمت ورود
            fig.add_shape(
                type="line",
                x0=df.index[-100],
                y0=strategy.get('entry_price', 0),
                x1=df.index[-1],
                y1=strategy.get('entry_price', 0),
                line=dict(color="#4287f5", width=1, dash="dash"),
                name="قیمت ورود"
            )
            
            # اضافه کردن خط حد ضرر
            fig.add_shape(
                type="line",
                x0=df.index[-100],
                y0=strategy.get('stop_loss', 0),
                x1=df.index[-1],
                y1=strategy.get('stop_loss', 0),
                line=dict(color="#ff3300", width=1, dash="dash"),
                name="حد ضرر"
            )
            
            # اضافه کردن خطوط اهداف قیمتی
            target_colors = ["#00cc66", "#00aa44", "#008833"]
            targets = [
                strategy.get('target_1', 0),
                strategy.get('target_2', 0),
                strategy.get('target_3', 0)
            ]
            
            for i, (target, color) in enumerate(zip(targets, target_colors)):
                fig.add_shape(
                    type="line",
                    x0=df.index[-100],
                    y0=target,
                    x1=df.index[-1],
                    y1=target,
                    line=dict(color=color, width=1, dash="dash"),
                    name=f"هدف {i+1}"
                )
            
            # تنظیمات نمودار
            fig.update_layout(
                title=f"نمودار استراتژی معاملاتی برای {symbol}",
                xaxis_title="تاریخ",
                yaxis_title="قیمت",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            # اضافه کردن مقادیر اهداف و حد ضرر به حاشیه نمودار
            fig.add_annotation(
                x=df.index[-1],
                y=strategy.get('entry_price', 0),
                text="ورود",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=0
            )
            
            fig.add_annotation(
                x=df.index[-1],
                y=strategy.get('stop_loss', 0),
                text="حد ضرر",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=0
            )
            
            for i, target in enumerate(targets):
                fig.add_annotation(
                    x=df.index[-1],
                    y=target,
                    text=f"هدف {i+1}",
                    showarrow=True,
                    arrowhead=1,
                    ax=50,
                    ay=0
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # نمایش جزئیات بیشتر استراتژی
            with st.expander("جزئیات بیشتر استراتژی"):
                st.markdown("#### محاسبات ریسک و بازده")
                
                # محاسبه درصد سود/زیان برای هر هدف
                entry_price = strategy.get('entry_price', 0)
                stop_loss = strategy.get('stop_loss', 0)
                risk_percentage = abs((entry_price - stop_loss) / entry_price * 100)
                
                target_percentages = []
                for target in targets:
                    if signal == 'BUY':
                        target_percentage = (target - entry_price) / entry_price * 100
                    else:  # SELL
                        target_percentage = (entry_price - target) / entry_price * 100
                    target_percentages.append(target_percentage)
                
                # محاسبه نسبت‌های ریسک به پاداش
                risk_reward_ratios = [tp / risk_percentage for tp in target_percentages]
                
                # نمایش اطلاعات در جدول
                data = {
                    "معیار": ["هدف 1", "هدف 2", "هدف 3"],
                    "قیمت": [f"{targets[0]:.2f}", f"{targets[1]:.2f}", f"{targets[2]:.2f}"],
                    "درصد سود": [f"{target_percentages[0]:.2f}%", f"{target_percentages[1]:.2f}%", f"{target_percentages[2]:.2f}%"],
                    "نسبت ریسک به پاداش": [f"{risk_reward_ratios[0]:.2f}", f"{risk_reward_ratios[1]:.2f}", f"{risk_reward_ratios[2]:.2f}"]
                }
                
                # تبدیل به دیتافریم و نمایش
                risk_reward_df = pd.DataFrame(data)
                st.table(risk_reward_df)
                
                # محاسبات سایز پوزیشن
                st.markdown("#### محاسبات سایز پوزیشن")
                
                position_size = strategy.get('position_size', 0)
                position_value = position_size * entry_price
                risk_amount = position_value * (risk_percentage / 100)
                
                st.markdown(f"""
                * **سرمایه کل**: {capital:.2f} USDT
                * **سایز پوزیشن**: {position_size:.6f} {symbol.split('/')[0]}
                * **ارزش پوزیشن**: {position_value:.2f} USDT
                * **ریسک معامله**: {risk_amount:.2f} USDT ({risk_percentage:.2f}% سرمایه)
                """)
                
                # راهنمایی برای مدیریت موقعیت
                st.markdown("#### راهنمای مدیریت موقعیت")
                
                if signal == 'BUY':
                    st.markdown("""
                    1. **استراتژی خروج تدریجی**: توصیه می‌شود در صورت رسیدن به هر هدف، بخشی از پوزیشن را ببندید (مثلاً 30% در هدف اول، 30% در هدف دوم و 40% در هدف سوم).
                    2. **حد ضرر متحرک**: پس از رسیدن به هدف اول، حد ضرر را به نقطه سربه‌سر منتقل کنید تا از سود خود محافظت کنید.
                    3. **زمان نگهداری**: برای استراتژی‌های روندی، تا زمانی که روند ادامه دارد در پوزیشن بمانید، برای استراتژی‌های نوسانی، به محض رسیدن به هدف خارج شوید.
                    """)
                elif signal == 'SELL':
                    st.markdown("""
                    1. **استراتژی پوشش تدریجی**: توصیه می‌شود در صورت رسیدن به هر هدف، بخشی از پوزیشن را ببندید (مثلاً 30% در هدف اول، 30% در هدف دوم و 40% در هدف سوم).
                    2. **حد ضرر متحرک**: پس از رسیدن به هدف اول، حد ضرر را به نقطه سربه‌سر منتقل کنید تا از سود خود محافظت کنید.
                    3. **توجه به فشار خرید**: در پوزیشن‌های شورت، مراقب افزایش ناگهانی فشار خرید باشید و آماده خروج سریع باشید.
                    """)
        else:
            st.info("برای دریافت استراتژی معاملاتی هوشمند، دکمه 'تولید استراتژی هوشمند' را کلیک کنید.")
    
    with tab2:
        st.subheader("مقایسه استراتژی‌های معاملاتی")
        
        # دکمه مقایسه استراتژی‌ها
        if st.button("🔄 مقایسه استراتژی‌ها", key="compare_strategies"):
            with st.spinner("در حال مقایسه استراتژی‌های مختلف..."):
                try:
                    # تولید مقایسه استراتژی‌ها
                    comparison = compare_trading_strategies(df, symbol)
                    
                    # ذخیره در session state
                    st.session_state.strategy_comparison = comparison
                    
                    st.success("مقایسه استراتژی‌ها با موفقیت انجام شد!")
                except Exception as e:
                    st.error(f"خطا در مقایسه استراتژی‌ها: {str(e)}")
        
        # نمایش نتایج مقایسه
        if 'strategy_comparison' in st.session_state:
            comparison = st.session_state.strategy_comparison
            
            # نمایش خلاصه مقایسه
            st.markdown(f"""
            <div style="background-color: rgba(0, 0, 0, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">خلاصه مقایسه استراتژی‌ها</h3>
                <p>{comparison.get('summary', 'اطلاعات در دسترس نیست.')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # نمایش لیست استراتژی‌ها
            st.markdown("### مقایسه انواع استراتژی‌ها")
            
            strategies = comparison.get('strategies', [])
            
            if strategies:
                # ایجاد دیتافریم برای نمایش مقایسه
                data = []
                for strategy in strategies:
                    data.append({
                        "نوع استراتژی": strategy.get('strategy_type', 'نامشخص'),
                        "سیگنال": "خرید" if strategy.get('signal') == 'BUY' else "فروش" if strategy.get('signal') == 'SELL' else "خنثی",
                        "قیمت ورود": f"{strategy.get('entry_price', 0):.2f}",
                        "حد ضرر": f"{strategy.get('stop_loss', 0):.2f}",
                        "هدف اول": f"{strategy.get('target_1', 0):.2f}",
                        "درصد ریسک": f"{strategy.get('risk_percentage', 0):.1f}%",
                        "اطمینان": f"{strategy.get('confidence', 0):.1f}%"
                    })
                
                # تبدیل به دیتافریم و نمایش
                strategy_df = pd.DataFrame(data)
                
                # تابع استایل‌دهی برای سلول‌های جدول
                def highlight_confidence(val):
                    try:
                        confidence = float(val.replace('%', ''))
                        if confidence >= 80:
                            return 'background-color: rgba(0, 204, 102, 0.2)'
                        elif confidence >= 60:
                            return 'background-color: rgba(255, 204, 0, 0.2)'
                        else:
                            return 'background-color: rgba(255, 51, 0, 0.1)'
                    except:
                        return ''
                
                # تابع استایل‌دهی برای سلول‌های سیگنال
                def highlight_signal(val):
                    if val == 'خرید':
                        return 'background-color: rgba(0, 204, 102, 0.2)'
                    elif val == 'فروش':
                        return 'background-color: rgba(255, 51, 0, 0.1)'
                    else:
                        return 'background-color: rgba(153, 153, 153, 0.1)'
                
                # اعمال استایل‌ها به جدول
                styled_df = strategy_df.style.applymap(highlight_confidence, subset=['اطمینان']).applymap(highlight_signal, subset=['سیگنال'])
                
                st.table(styled_df)
                
                # نمایش جزئیات بهترین استراتژی
                best_strategy = comparison.get('best_strategy')
                if best_strategy:
                    st.markdown("### جزئیات بهترین استراتژی")
                    
                    st.markdown(f"""
                    <div style="background-color: rgba(0, 204, 102, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #00cc66;">
                        <h3 style="margin-top: 0;">{best_strategy.get('strategy_type', 'نامشخص')}</h3>
                        <p>{best_strategy.get('description', 'توضیحات در دسترس نیست.')}</p>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px;">
                            <div>
                                <h4 style="margin-top: 0;">سیگنال</h4>
                                <p style="font-weight: bold; font-size: 1.2em;">{"خرید" if best_strategy.get('signal') == 'BUY' else "فروش" if best_strategy.get('signal') == 'SELL' else "خنثی"}</p>
                            </div>
                            <div>
                                <h4 style="margin-top: 0;">قیمت ورود</h4>
                                <p style="font-weight: bold; font-size: 1.2em;">{best_strategy.get('entry_price', 0):.2f} USDT</p>
                            </div>
                            <div>
                                <h4 style="margin-top: 0;">اطمینان</h4>
                                <p style="font-weight: bold; font-size: 1.2em;">{best_strategy.get('confidence', 0):.1f}%</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("اطلاعات مقایسه استراتژی‌ها در دسترس نیست.")
        else:
            st.info("برای مقایسه انواع استراتژی‌های معاملاتی، دکمه 'مقایسه استراتژی‌ها' را کلیک کنید.")
    
    with tab3:
        st.subheader("تنظیمات استراتژی خودکار")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("استراتژی پیش‌فرض", ["دنبال کردن روند", "بازگشت به میانگین", "شکست", "نوسان‌گر", "الگوهای هارمونیک", "حمایت و مقاومت", "مبتنی بر حجم", "مبتنی بر نوسانات", "ریسک پایین"], index=0, key="default_strategy")
            st.slider("حداکثر درصد ریسک در هر معامله", min_value=0.5, max_value=5.0, value=1.0, step=0.1, key="max_risk_per_trade")
            st.checkbox("فعال‌سازی حد ضرر خودکار", value=True, key="enable_stop_loss")
        
        with col2:
            st.number_input("نسبت ریسک به پاداش", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="risk_reward_ratio")
            st.number_input("حداکثر تعداد معاملات همزمان", min_value=1, max_value=10, value=3, key="max_concurrent_trades")
            st.checkbox("خروج تدریجی از پوزیشن", value=True, key="enable_partial_exit")
        
        if st.button("💾 ذخیره تنظیمات استراتژی", key="save_strategy_settings"):
            st.success("تنظیمات استراتژی با موفقیت ذخیره شد!")