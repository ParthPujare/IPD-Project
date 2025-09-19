import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import tensorflow as tf
from datetime import datetime, timedelta
import os
import subprocess
from database_utils import get_latest_available_date, get_data_for_prediction, get_historical_data, get_raw_historical_prices
from lstm_model import load_model as load_custom_model
import mplfinance as mpf

# Disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Page configuration
st.set_page_config(
    page_title="Stock Price Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Price Analysis")
st.markdown("Analyze stock prices and market sentiment to make informed investment decisions.")

# Custom model loading function with TensorFlow compatibility
def load_model(model_path):
    """Load model with compatibility for different TensorFlow versions"""
    try:
        # First try loading model directly
        return tf.keras.models.load_model(model_path)
    except Exception:
        try:
            # Get TensorFlow version
            tf_version = tf.__version__
            
            # For TensorFlow 2.6+, use a custom loading approach
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
            
            # Define model architecture manually
            model = Sequential()
            
            # First LSTM layer with return sequences
            model.add(InputLayer(input_shape=(10, 9)))  # Use InputLayer instead of batch_shape
            model.add(LSTM(50, return_sequences=True))
            model.add(Dropout(0.2))
            
            # Second LSTM layer
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            
            # Dense layers
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Try to load weights only (not the architecture)
            try:
                model.load_weights(model_path)
            except:
                pass
            
            return model
        except Exception as e2:
            raise e2

# Function to run a subprocess and handle errors
def run_subprocess(cmd, message):
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Error running {cmd[1]}: {result.stderr}")
            return False
        return True
    except Exception as e:
        st.error(f"Exception running {cmd[1]}: {str(e)}")
        return False

# Sidebar
st.sidebar.title("Options")

# Individual update buttons for more control
update_options = st.sidebar.expander("Update Data Options", expanded=False)
with update_options:
    if st.button("Update Price Data"):
        with st.spinner("Updating price data..."):
            success = run_subprocess(["python", "update_price.py"], "Updating price data")
            if success:
                st.success("Price data updated successfully!")

    if st.button("Fetch News"):
        with st.spinner("Fetching news..."):
            success = run_subprocess(["python", "fetch_news.py"], "Fetching news")
            if success:
                st.success("News fetched successfully!")

    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            success = run_subprocess(["python", "sentiment_analysis.py"], "Analyzing sentiment")
            if success:
                st.success("Sentiment analysis completed successfully!")

    if st.button("Prepare Data"):
        with st.spinner("Preparing data..."):
            success = run_subprocess(["python", "getdataready.py"], "Preparing data")
            if success:
                st.success("Data preparation completed successfully!")

# Main update button (runs each step sequentially)
if st.sidebar.button("Run Full Update Pipeline"):
    steps = [
        (["python", "update_price.py"], "Updating price data"),
        (["python", "fetch_news.py"], "Fetching news"),
        (["python", "sentiment_analysis.py"], "Analyzing sentiment"),
        (["python", "getdataready.py"], "Preparing data")
    ]
    
    all_success = True
    for cmd, message in steps:
        with st.spinner(message):
            if not run_subprocess(cmd, message):
                all_success = False
                break
    
    if all_success:
        st.sidebar.success("All data updated successfully!")
        st.experimental_rerun()  # Refresh the app to show new data

# Currency symbol
CURRENCY_SYMBOL = "₹"

# Try to load both historical data sets
try:
    # Get processed data for model predictions
    historical_data = get_historical_data()
    
    # Get raw historical prices for charting (from historical_prices table)
    raw_prices = get_raw_historical_prices()
    
    if historical_data and raw_prices:
        # Process the raw prices for display
        df_raw = pd.DataFrame(raw_prices)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # Process the model data for predictions
        df_historical = pd.DataFrame(historical_data)
        df_historical['date'] = pd.to_datetime(df_historical['date'])
        
        # Display tabs for different views
        tab1, tab2 = st.tabs(["Price History", "Price & Sentiment"])
        
        with tab1:
            st.subheader("Historical Price Data")
            
            # Add price range selector at the top
            date_range = st.select_slider(
                "Select Date Range",
                options=sorted(df_raw['date'].dt.date.unique()),
                value=(df_raw['date'].min().date(), df_raw['date'].max().date())
            )
            
            # Filter data based on selected date range
            mask = (df_raw['date'].dt.date >= date_range[0]) & (df_raw['date'].dt.date <= date_range[1])
            filtered_df = df_raw[mask]
            
            # Create columns for chart and stats
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create a more detailed price chart with candlesticks
                fig = plt.figure(figsize=(12, 6))
                
                # Plot OHLC as a line chart (simpler version)
                plt.plot(filtered_df['date'], filtered_df['close'], label='Close Price')
                
                # Add volume as bars at the bottom
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                # Normalize volume to fit better on the chart
                max_price = filtered_df['close'].max()
                volume_norm = filtered_df['volume'] * (max_price * 0.3) / filtered_df['volume'].max()
                
                ax2.fill_between(filtered_df['date'], 0, volume_norm, alpha=0.3, color='gray', label='Volume')
                ax2.set_ylabel('Volume')
                
                plt.title('Historical Stock Prices with Volume')
                plt.xlabel('Date')
                ax1.set_ylabel('Price (₹)')
                ax1.grid(True, alpha=0.3)
                
                # Add legends with better placement
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
                
                st.pyplot(fig)
                
                # Create an improved candlestick chart for the selected range
                if not filtered_df.empty and len(filtered_df) <= 90:  # Only show candlesticks for reasonable timeframes
                    st.subheader("Technical Analysis Chart")
                    
                    # Calculate technical indicators
                    # 1. Moving Averages
                    filtered_df['MA20'] = filtered_df['close'].rolling(window=20).mean()
                    filtered_df['MA50'] = filtered_df['close'].rolling(window=50).mean() 
                    
                    # 2. RSI (Relative Strength Index)
                    delta = filtered_df['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    filtered_df['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Create subplots: 3 rows (price, volume, RSI)
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.05, 
                                      subplot_titles=('Price', 'Volume', 'RSI'),
                                      row_heights=[0.6, 0.2, 0.2])
                    
                    # Row 1: Enhanced Candlestick with Moving Averages
                    fig.add_trace(go.Candlestick(
                        x=filtered_df['date'],
                        open=filtered_df['open'],
                        high=filtered_df['high'],
                        low=filtered_df['low'],
                        close=filtered_df['close'],
                        name='Price',
                        increasing=dict(line=dict(color='#26a69a', width=1), fillcolor='#26a69a'),
                        decreasing=dict(line=dict(color='#ef5350', width=1), fillcolor='#ef5350'),
                        whiskerwidth=0.5,
                        opacity=1,
                        text=[f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br>" +
                              f"<b>Open:</b> {CURRENCY_SYMBOL}{o:.2f}<br>" +
                              f"<b>High:</b> {CURRENCY_SYMBOL}{h:.2f}<br>" +
                              f"<b>Low:</b> {CURRENCY_SYMBOL}{l:.2f}<br>" +
                              f"<b>Close:</b> {CURRENCY_SYMBOL}{c:.2f}" 
                              for d, o, h, l, c in zip(
                                  filtered_df['date'],
                                  filtered_df['open'],
                                  filtered_df['high'],
                                  filtered_df['low'],
                                  filtered_df['close']
                              )],
                        hoverinfo="text"
                    ), row=1, col=1)
                    
                    # Add Moving Averages
                    if len(filtered_df) >= 20:
                        fig.add_trace(go.Scatter(
                            x=filtered_df['date'],
                            y=filtered_df['MA20'],
                            line=dict(color='#ffb74d', width=1.5),
                            name="20-Day MA"
                        ), row=1, col=1)
                    
                    if len(filtered_df) >= 50:
                        fig.add_trace(go.Scatter(
                            x=filtered_df['date'],
                            y=filtered_df['MA50'],
                            line=dict(color='#ab47bc', width=1.5),
                            name="50-Day MA"
                        ), row=1, col=1)
                    
                    # Row 2: Volume bars with color coding
                    colors = []
                    for i in range(len(filtered_df)):
                        if i == 0:
                            colors.append('rgba(100, 100, 100, 0.7)')
                        elif filtered_df['close'].iloc[i] > filtered_df['close'].iloc[i-1]:
                            colors.append('rgba(38, 166, 154, 0.7)')  # Green for increasing
                        else:
                            colors.append('rgba(239, 83, 80, 0.7)')  # Red for decreasing
                    
                    fig.add_trace(go.Bar(
                        x=filtered_df['date'],
                        y=filtered_df['volume'],
                        name='Volume',
                        marker=dict(
                            color=colors,
                            line=dict(width=0)
                        ),
                        opacity=0.8
                    ), row=2, col=1)
                    
                    # Row 3: RSI (Relative Strength Index)
                    fig.add_trace(go.Scatter(
                        x=filtered_df['date'],
                        y=filtered_df['RSI'],
                        line=dict(color='#64b5f6', width=1.5),
                        name="RSI"
                    ), row=3, col=1)
                    
                    # Add RSI overbought/oversold levels
                    fig.add_shape(
                        type="line", line_dash="dash",
                        x0=filtered_df['date'].iloc[0], y0=70, x1=filtered_df['date'].iloc[-1], y1=70,
                        line=dict(color="#ef5350", width=1), row=3, col=1
                    )
                    fig.add_shape(
                        type="line", line_dash="dash",
                        x0=filtered_df['date'].iloc[0], y0=30, x1=filtered_df['date'].iloc[-1], y1=30,
                        line=dict(color="#26a69a", width=1), row=3, col=1
                    )
                    
                    # Add centerline for RSI
                    fig.add_shape(
                        type="line", line_dash="dot",
                        x0=filtered_df['date'].iloc[0], y0=50, x1=filtered_df['date'].iloc[-1], y1=50,
                        line=dict(color="rgba(255, 255, 255, 0.3)", width=1), row=3, col=1
                    )
                    
                    # Update layout with better styling and readability
                    fig.update_layout(
                        # Dark theme
                        template="plotly_dark",
                        paper_bgcolor='rgba(30, 30, 46, 1)',
                        plot_bgcolor='rgba(30, 30, 46, 1)',
                        
                        # Title
                        title=dict(
                            text=f'<b>Technical Analysis ({date_range[0]} to {date_range[1]})</b>',
                            y=0.98,
                            x=0.5,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=22, color='white', family="Arial, sans-serif")
                        ),
                        
                        # Layout
                        height=900,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=12, color="white"),
                            bgcolor='rgba(30, 30, 46, 0.7)',
                            bordercolor='rgba(255, 255, 255, 0.2)',
                            borderwidth=1
                        ),
                        
                        # No rangeslider
                        xaxis_rangeslider_visible=False,
                        
                        # Margins
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    
                    # Update axes formatting
                    fig.update_yaxes(
                        title_text=f"Price ({CURRENCY_SYMBOL})", 
                        row=1, col=1,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        zerolinecolor='rgba(255, 255, 255, 0.2)'
                    )
                    fig.update_yaxes(
                        title_text="Volume", 
                        row=2, col=1,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        zerolinecolor='rgba(255, 255, 255, 0.2)'
                    )
                    fig.update_yaxes(
                        title_text="RSI", 
                        row=3, col=1, 
                        range=[0, 100],
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        zerolinecolor='rgba(255, 255, 255, 0.2)'
                    )
                    
                    fig.update_xaxes(
                        title_text="Date", 
                        row=3, col=1,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        showgrid=True
                    )
                    
                    # Add grid to all subplots
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.05)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.05)')
                    
                    # Update y-axes to show proper formatting
                    fig.update_yaxes(
                        tickprefix=f"{CURRENCY_SYMBOL} ",
                        row=1, col=1
                    )
                    
                    # Make subplot titles more prominent
                    fig.update_annotations(font=dict(size=14, color='white'))
                    
                    # Better hover information for scatter traces only
                    fig.update_traces(
                        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y:.2f}<extra></extra>",
                        selector=dict(type='scatter')
                    )
                    
                    # Add informational annotations for RSI
                    fig.add_annotation(
                        text="Overbought", 
                        x=filtered_df['date'].iloc[0], 
                        y=75, 
                        showarrow=False, 
                        font=dict(color="#ef5350", size=12),
                        row=3, col=1
                    )
                    fig.add_annotation(
                        text="Oversold", 
                        x=filtered_df['date'].iloc[0], 
                        y=25, 
                        showarrow=False, 
                        font=dict(color="#26a69a", size=12),
                        row=3, col=1
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add technical analysis explanation
                    with st.expander("Understand this Technical Analysis Chart"):
                        st.markdown("""
                        ### How to Interpret This Chart
                        
                        **Candlestick Chart (Top):**
                        - Green candles indicate price increased during that period
                        - Red candles indicate price decreased during that period
                        - The "wicks" show the high and low prices
                        
                        **Moving Averages:**
                        - **20-Day MA (Orange)**: Shows short-term trend
                        - **50-Day MA (Purple)**: Shows medium-term trend
                        - When the shorter MA crosses above the longer MA, it's considered a bullish signal
                        - When the shorter MA crosses below the longer MA, it's considered a bearish signal
                        
                        **Volume (Middle):**
                        - Higher volume often confirms the strength of a price trend
                        - Green bars show volume on days when price increased
                        - Red bars show volume on days when price decreased
                        
                        **RSI (Bottom):**
                        - Measures the speed and change of price movements
                        - Values over 70 suggest the stock may be overbought (potential reversal down)
                        - Values under 30 suggest the stock may be oversold (potential reversal up)
                        - The 50 line often represents a bull/bear dividing line
                        """)
                elif len(filtered_df) > 90:
                    st.info("Technical analysis chart is only shown for periods of 90 days or less. Please select a shorter time range for detailed analysis.")
            
            with col2:
                # Display summary statistics for the filtered range
                st.write("### Summary Statistics")
                st.dataframe(filtered_df['close'].describe().round(2))
                
                # Show min/max values with dates
                max_price_row = filtered_df.loc[filtered_df['close'].idxmax()]
                min_price_row = filtered_df.loc[filtered_df['close'].idxmin()]
                
                st.write("### Price Range")
                st.metric("Highest Price", f"{CURRENCY_SYMBOL}{max_price_row['close']:.2f}", 
                         f"on {max_price_row['date'].strftime('%Y-%m-%d')}")
                st.metric("Lowest Price", f"{CURRENCY_SYMBOL}{min_price_row['close']:.2f}", 
                         f"on {min_price_row['date'].strftime('%Y-%m-%d')}")
                
                # Calculate and show price change
                first_price = filtered_df['close'].iloc[0]
                last_price = filtered_df['close'].iloc[-1]
                price_change = last_price - first_price
                percent_change = (price_change / first_price) * 100
                
                st.metric("Price Change", f"{CURRENCY_SYMBOL}{price_change:.2f}", 
                         f"{percent_change:.2f}%")
                
                # Additional statistics
                st.write("### Volatility Metrics")
                daily_returns = filtered_df['close'].pct_change().dropna()
                
                if len(daily_returns) > 0:
                    volatility = daily_returns.std() * 100
                    st.metric("Daily Volatility", f"{volatility:.2f}%")
                    
                    # Show max daily gain/loss
                    max_gain = daily_returns.max() * 100
                    max_loss = daily_returns.min() * 100
                    
                    st.metric("Max Daily Gain", f"{max_gain:.2f}%")
                    st.metric("Max Daily Loss", f"{max_loss:.2f}%")
        
        with tab2:
            st.subheader("Price & Sentiment Analysis")
            
            # Add date range selector for sentiment analysis
            sentiment_date_range = st.select_slider(
                "Select Date Range for Sentiment Analysis",
                options=sorted(df_raw['date'].dt.date.unique()),
                value=(df_raw['date'].min().date(), df_raw['date'].max().date())
            )
            
            # Filter data based on selected date range
            mask = (df_raw['date'].dt.date >= sentiment_date_range[0]) & (df_raw['date'].dt.date <= sentiment_date_range[1])
            filtered_raw = df_raw[mask]
            
            # Show correlation between price and sentiment
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a merged dataframe with raw prices and sentiment data
                df_merged = pd.merge(
                    filtered_raw[['date', 'close']], 
                    df_historical[['date', 'sentiment_score']], 
                    on='date', 
                    how='inner'
                )
                
                if not df_merged.empty:
                    # Create an area chart showing price movements with sentiment as color intensity
                    st.subheader("Price Movements with Sentiment Overlay")
                    
                    # Normalize sentiment for color intensity (0-1 range)
                    df_merged['normalized_sentiment'] = (df_merged['sentiment_score'] - df_merged['sentiment_score'].min()) / \
                                                       (df_merged['sentiment_score'].max() - df_merged['sentiment_score'].min()) \
                                                       if df_merged['sentiment_score'].max() != df_merged['sentiment_score'].min() else 0.5
                    
                    # Create color list based on sentiment
                    colors = df_merged['normalized_sentiment'].apply(
                        lambda x: f'rgba(31, 119, 180, {0.3 + 0.7*x})' 
                        if not pd.isna(x) else 'rgba(31, 119, 180, 0.5)'
                    ).tolist()
                    
                    # Calculate daily returns
                    df_merged['daily_return'] = df_merged['close'].pct_change() * 100
                    
                    # Create figure with two subplots stacked vertically
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.1,
                                       subplot_titles=("Price with Sentiment Intensity", "Price-Sentiment Relationship"),
                                       row_heights=[0.7, 0.3])
                    
                    # Add price area chart to top subplot
                    fig.add_trace(
                        go.Scatter(
                            x=df_merged['date'],
                            y=df_merged['close'],
                            fill='tozeroy',
                            name='Stock Price',
                            line=dict(width=1, color='#1f77b4'),
                            fillcolor='rgba(31, 119, 180, 0.3)',
                            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                                         '<b>Price:</b> ₹%{y:.2f}<br>' +
                                         '<b>Sentiment:</b> %{text:.2f}<extra></extra>',
                            text=df_merged['sentiment_score']
                        ),
                        row=1, col=1
                    )
                    
                    # Create a scatter plot of daily returns vs sentiment
                    fig.add_trace(
                        go.Scatter(
                            x=df_merged['sentiment_score'],
                            y=df_merged['daily_return'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=df_merged['daily_return'],
                                colorscale='RdBu',
                                cmin=-3,
                                cmax=3,
                                line=dict(width=1),
                                opacity=0.7
                            ),
                            name='Returns vs Sentiment',
                            hovertemplate='<b>Sentiment:</b> %{x:.2f}<br>' +
                                         '<b>Daily Return:</b> %{y:.2f}%<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Add trend line for scatter plot
                    if len(df_merged) > 2:
                        # Remove NaN values before polyfit
                        mask = ~df_merged['daily_return'].isna() & ~df_merged['sentiment_score'].isna()
                        if sum(mask) > 2:
                            x = df_merged.loc[mask, 'sentiment_score']
                            y = df_merged.loc[mask, 'daily_return']
                            
                            # Calculate trend line
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            
                            # Add trend line to scatter plot
                            fig.add_trace(
                                go.Scatter(
                                    x=[min(x), max(x)],
                                    y=p([min(x), max(x)]),
                                    mode='lines',
                                    line=dict(color='rgba(0,0,0,0.5)', width=2, dash='dash'),
                                    name='Trend Line',
                                    hoverinfo='skip'
                                ),
                                row=2, col=1
                            )
                    
                    # Update layout
                    fig.update_layout(
                        height=700,
                        title_text="Stock Price and Sentiment Analysis",
                        template="plotly_white",
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Update axes
                    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                    fig.update_xaxes(title_text="Sentiment Score", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                    fig.update_yaxes(title_text=f"Price ({CURRENCY_SYMBOL})", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.info("""
                    **Chart Explanation:**
                    - **Top Chart**: Shows stock price with sentiment intensity represented by color depth (darker = more positive)
                    - **Bottom Chart**: Scatter plot showing the relationship between daily price returns and sentiment score
                    - The trend line indicates if there's a correlation between sentiment and price changes
                    """)
                else:
                    st.warning("No matching data available for the selected date range. Please select a different range.")
                
                # Add correlation analysis with improved visualization
                if len(df_merged) > 0:
                    # Create correlation matrix for advanced insights
                    st.subheader("Price-Sentiment Correlation Analysis")
                    
                    # Prepare data for correlation
                    corr_data = pd.DataFrame({
                        'Price': df_merged['close'],
                        'Sentiment': df_merged['sentiment_score']
                    })
                    
                    # Add lagged sentiment features for more insightful analysis
                    for lag in range(1, min(6, len(df_merged) - 5)):
                        corr_data[f'Sentiment (t-{lag})'] = df_merged['sentiment_score'].shift(lag)
                    
                    # Drop NaN rows
                    corr_data = corr_data.dropna()
                    
                    if not corr_data.empty and len(corr_data) > 5:
                        # Calculate correlation matrix
                        corr_matrix = corr_data.corr()
                        
                        # Create correlation heatmap using Plotly
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            colorscale='RdBu_r',
                            zmin=-1, zmax=1,
                            text=[[f'{val:.2f}' for val in row] for row in corr_matrix.values],
                            texttemplate="%{text}",
                            textfont={"size":12}
                        ))
                        
                        fig.update_layout(
                            title="Correlation Heatmap",
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_showgrid=False,
                            yaxis_showgrid=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data for detailed correlation analysis. Select a wider date range.")
            
            with col2:
                # Show sentiment statistics and insights
                if len(df_merged) > 0:
                    # Calculate correlation with improved display
                    corr = df_merged['close'].corr(df_merged['sentiment_score'])
                    st.markdown("""
                    <style>
                    .big-font {
                        font-size:24px !important;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    st.markdown("### Sentiment Metrics")
                    
                    # Use more engaging metrics display
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Price-Sentiment Correlation", f"{corr:.2f}", 
                                 help="Correlation between stock price and sentiment score. Values closer to 1 indicate stronger positive correlation.")
                    
                    # Calculate and show average sentiment
                    avg_sentiment = df_merged['sentiment_score'].mean()
                    with col_b:
                        st.metric("Average Sentiment", f"{avg_sentiment:.2f}", 
                                 help="Average sentiment score (0-1). Higher values indicate more positive sentiment.")
                    
                    # Show recent sentiment trend - fix delta_color
                    if len(df_merged) >= 7:
                        recent_df = df_merged.tail(7)
                        recent_sentiment = recent_df['sentiment_score'].mean()
                        sentiment_change = recent_sentiment - avg_sentiment
                        delta_color = "normal" if sentiment_change > 0 else "inverse"
                        
                        st.metric("Recent 7-Day Sentiment", 
                                 f"{recent_sentiment:.2f}", 
                                 f"{sentiment_change:+.2f}",
                                 delta_color=delta_color,
                                 help="Average sentiment over the last 7 days with change from overall average.")
                    
                    # Add sentiment distribution chart with modern styling
                    st.markdown("### Sentiment Distribution")
                    
                    # Create histogram using Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df_merged['sentiment_score'],
                        nbinsx=20,
                        marker_color='rgba(31, 119, 180, 0.6)',
                        marker_line_color='rgba(31, 119, 180, 1)',
                        marker_line_width=1
                    ))
                    
                    # Add vertical line for mean
                    fig.add_vline(
                        x=avg_sentiment, 
                        line_width=2, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Mean: {avg_sentiment:.2f}",
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(
                        title="Sentiment Score Distribution",
                        xaxis_title="Sentiment Score",
                        yaxis_title="Frequency",
                        bargap=0.05,
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add sentiment trends over time sections
                    if len(df_merged) > 14:
                        st.markdown("### Sentiment Trends")
                        
                        try:
                            # Weekly average sentiment with safer date handling
                            # Create week number as string directly instead of using formatting
                            df_merged['yearweek'] = df_merged['date'].dt.strftime('%Y-W%V')  # ISO week format
                            weekly_sentiment = df_merged.groupby('yearweek')['sentiment_score'].mean().reset_index()
                            weekly_sentiment.rename(columns={'yearweek': 'period'}, inplace=True)
                            
                            # Plot weekly sentiment
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=weekly_sentiment['period'],
                                y=weekly_sentiment['sentiment_score'],
                                marker_color='rgba(31, 119, 180, 0.6)',
                                marker_line_color='rgba(31, 119, 180, 1)',
                                marker_line_width=1
                            ))
                            
                            fig.update_layout(
                                title="Weekly Average Sentiment",
                                xaxis_title="Week",
                                yaxis_title="Sentiment Score",
                                height=250,
                                margin=dict(l=20, r=20, t=40, b=40),
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            
                            # Show only the last 10 weeks if there are many weeks
                            if len(weekly_sentiment) > 10:
                                fig.update_layout(xaxis_range=[len(weekly_sentiment)-10, len(weekly_sentiment)])
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error processing weekly sentiment: {e}")
                            st.info("Some date data might be problematic. Try selecting a different date range.")
        
        # Generate prediction
        st.subheader("Price Prediction")
        
        # Get the latest available data
        latest_date_str = get_latest_available_date()
        if latest_date_str:
            try:
                # Convert latest_date from string to datetime
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
                
                # Load model with TensorFlow compatibility
                try:
                    # Use the original load_custom_model from lstm_model.py
                    try:
                        model = load_custom_model("stock_price_model.h5")
                    except:
                        # Fall back to our custom loader if original fails
                        model = load_model("stock_price_model.h5")
                    
                    # Custom prediction function
                    def predict_next_day_price(model, data):
                        prediction = model.predict(data, verbose=0)  # Add verbose=0 to suppress prediction output
                        return prediction[0][0]
                
                    # Load the price scaler
                    with open("price_scaler.pkl", "rb") as price_file:
                        price_scaler = pickle.load(price_file)
                    
                    # Fetch relevant data for prediction
                    data = get_data_for_prediction(latest_date)
                    
                    if data is not None:
                        # Make prediction
                        predicted_scaled_price = predict_next_day_price(model, data)
                        
                        # Inverse transform the predicted scaled price
                        try:
                            # First attempt with 7 features
                            dummy_features = np.zeros((1, 7))
                            dummy_features[0, -1] = predicted_scaled_price
                            predicted_price = float(price_scaler.inverse_transform(dummy_features)[0][-1])
                        except:
                            try:
                                # Second attempt with whatever dimensions the scaler expects
                                dummy_features = np.zeros((1, len(price_scaler.data_min_)))
                                dummy_features[0, -1] = predicted_scaled_price
                                predicted_price = float(price_scaler.inverse_transform(dummy_features)[0][-1])
                            except Exception as e:
                                st.error(f"Error transforming prediction: {str(e)}")
                                # Last resort: just show the scaled value
                                predicted_price = float(predicted_scaled_price)
                        
                        # Next day
                        next_day = latest_date + timedelta(days=1)
                        
                        # Display prediction with improved styling
                        st.write("### Next Day Prediction")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create metrics with clean styling
                            st.markdown(f"""
                            <div style="padding:20px; margin-bottom:20px;">
                                <h4>Predicted Price for {next_day.strftime('%Y-%m-%d')}</h4>
                                <h2 style="margin:0; font-size:48px;">{CURRENCY_SYMBOL}{predicted_price:.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Get last actual price from raw data (more accurate)
                            last_price = df_raw['close'].iloc[-1] if not df_raw.empty else df_historical['price'].iloc[-1]
                            price_change = predicted_price - last_price
                            percent_change = (price_change / last_price) * 100
                            
                            # Change color based on positive or negative change
                            color = "green" if price_change >= 0 else "red"
                            arrow = "↑" if price_change >= 0 else "↓"
                            
                            st.markdown(f"""
                            <div style="padding:20px;">
                                <h4>Expected Change</h4>
                                <h2 style="margin:0; color:{color}; font-size:32px;">{arrow} {CURRENCY_SYMBOL}{abs(price_change):.2f} ({percent_change:.2f}%)</h2>
                                <p style="margin-top:10px; color:gray;">From last close: {CURRENCY_SYMBOL}{last_price:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Plot recent prices and prediction with improved styling
                            recent_df = df_raw.tail(30).copy() if not df_raw.empty else df_historical.tail(30).copy()
                            
                            # Create a new row for the prediction
                            prediction_row = pd.DataFrame({
                                'date': [next_day],
                                'close': [predicted_price]
                            })
                            
                            # Ensure we're using the correct column name for price data
                            price_col = 'close' if 'close' in recent_df.columns else 'price'
                            
                            # Create plot with improved styling
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Plot historical data with faded blue line
                            ax.plot(recent_df['date'], recent_df[price_col], 'b-', linewidth=2, label='Historical Prices', alpha=0.7)
                            
                            # Add light blue fill below the line
                            ax.fill_between(recent_df['date'], 0, recent_df[price_col], alpha=0.1, color='blue')
                            
                            # Make sure we have at least one data point
                            if len(recent_df) > 0:
                                # Plot prediction with dashed red line
                                ax.plot([recent_df['date'].iloc[-1], next_day], 
                                        [recent_df[price_col].iloc[-1], predicted_price], 
                                        'r--', linewidth=2, label='Prediction')
                                
                                # Add red dot for prediction point
                                ax.scatter([next_day], [predicted_price], color='red', s=100, zorder=5)
                                
                                # Add text annotation for predicted value
                                ax.annotate(f'{CURRENCY_SYMBOL}{predicted_price:.2f}', 
                                          xy=(next_day, predicted_price),
                                          xytext=(10, 10),
                                          textcoords='offset points',
                                          color='darkred',
                                          fontweight='bold')
                            
                            ax.set_xlabel('Date')
                            ax.set_ylabel(f'Price ({CURRENCY_SYMBOL})')
                            ax.set_title('Recent Prices and Prediction', fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.legend(loc='upper left')
                            
                            # Format y-axis with proper currency
                            from matplotlib.ticker import FuncFormatter
                            
                            def rupee_formatter(x, pos):
                                return f'{CURRENCY_SYMBOL}{x:.0f}'
                                
                            ax.yaxis.set_major_formatter(FuncFormatter(rupee_formatter))
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.warning("Insufficient data for prediction.")
                except Exception as model_error:
                    st.error(f"Model loading or prediction error: {str(model_error)}")
                    st.info("Unable to generate prediction due to model compatibility issues.")
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
        else:
            st.warning("No date information available. Please update the data.")
    else:
        st.warning("No historical data available. Please update the data first.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("You might need to update the data or check if the database exists.")

# Display model information
st.subheader("Model Information")
col1, col2 = st.columns(2)

with col1:
    st.write("### LSTM Model Architecture")
    st.code("""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(N_PAST, num_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    """)

with col2:
    st.write("### Features Used")
    st.markdown("""
    - Open, High, Low, Close prices
    - Trading Volume
    - News Sentiment Score
    - Previous Close
    - Moving Averages (3-day and 7-day)
    """)

st.write("### Disclaimer")
st.info("This predictive model is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions.") 