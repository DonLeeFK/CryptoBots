import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Trading parameters
ticker = "BTC-USD"
period = "1mo"
interval = "1h"
macd_fast = 13
macd_slow = 34
macd_signal = 9
lookback_period = 240
atr_period = 14
divergence_threshold = 0  
rsi_period = 13
rsi_oversold = 30
rsi_overbought = 80

def calculate_macd(data, fast, slow, signal):
    """Calculates MACD, signal line, and histogram."""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_atr(data, period):
    """Calculates the Average True Range (ATR)."""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_rsi(data, period):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_divergence(data, macd, signal_line, histogram, lookback, threshold):
    """Detects regular bullish and bearish divergences with a threshold."""
    price = data['Close']

    divergence = 'none'

    # Identify recent swing highs/lows in price
    price_lows_idx = price.iloc[-lookback:].nsmallest(2).index
    price_highs_idx = price.iloc[-lookback:].nlargest(2).index

    # Sort index to maintain chronological order
    price_lows_idx = price_lows_idx.sort_values()
    price_highs_idx = price_highs_idx.sort_values()
    histogram_highs = histogram.loc[price_highs_idx]
    histogram_lows = histogram.loc[price_lows_idx]

    # Check price trend and corresponding histogram
    if price_lows_idx[1] > price_lows_idx[0]:
        # Regular Bullish Divergence
        if price[price_lows_idx[1]] < price[price_lows_idx[0]]:  # Lower low in price
            #histogram_lows = histogram.loc[price_lows_idx]
            if histogram_lows.iloc[1] > histogram_lows.iloc[0]:  # Higher low in histogram
                # Check if divergence meets the threshold
                price_diff = (price[price_lows_idx[0]] - price[price_lows_idx[1]]) / price[price_lows_idx[1]]
                histogram_diff = (histogram_lows.iloc[1] - histogram_lows.iloc[0]) / histogram_lows.iloc[0]
                if price_diff > threshold and histogram_diff > threshold:
                    divergence = 'regular_bullish'
            if histogram_highs.iloc[1] > histogram_highs.iloc[0]:  # Higher high in histogram
                # Check if divergence meets the threshold
                price_diff = (price[price_lows_idx[0]] - price[price_lows_idx[1]]) / price[price_lows_idx[1]]
                histogram_diff = (histogram_highs.iloc[0] - histogram_highs.iloc[1]) / histogram_highs.iloc[1]
                if price_diff > threshold and histogram_diff > threshold:
                    divergence = 'hidden_bullish'   

    if price_highs_idx[1] > price_highs_idx[0]:
        # Regular Bearish Divergence
        if price[price_highs_idx[1]] > price[price_highs_idx[0]]:  # Higher high in price
            #histogram_highs = histogram.loc[price_highs_idx]
            if histogram_highs.iloc[1] < histogram_highs.iloc[0]:  # Lower high in histogram
                # Check if divergence meets the threshold
                price_diff = (price[price_highs_idx[1]] - price[price_highs_idx[0]]) / price[price_highs_idx[0]]
                histogram_diff = (histogram_highs.iloc[0] - histogram_highs.iloc[1]) / histogram_highs.iloc[1]
                if price_diff > threshold and histogram_diff > threshold:
                    divergence = 'regular_bearish'
            if histogram_lows.iloc[1] < histogram_lows.iloc[0]:  # Lower low in histogram
                # Check if divergence meets the threshold
                price_diff = (price[price_highs_idx[1]] - price[price_highs_idx[0]]) / price[price_highs_idx[0]]
                histogram_diff = (histogram_lows.iloc[1] - histogram_lows.iloc[0]) / histogram_lows.iloc[0]
                if price_diff > threshold and histogram_diff > threshold:
                    divergence = 'hidden_bearish'

    return divergence

def plot_signals(data):
    """Identifies and plots potential open long/short signals based on MACD divergence and RSI."""
    atr = calculate_atr(data, atr_period)
    macd, signal_line, histogram = calculate_macd(data, macd_fast, macd_slow, macd_signal)
    rsi = calculate_rsi(data, rsi_period)

    open_long_signals = []
    open_short_signals = []
    active_divergence = None

    for i in range(lookback_period, len(data)):
        current_data = data.iloc[i - lookback_period:i + 1]
        divergence = detect_divergence(current_data, macd.iloc[i - lookback_period:i + 1], signal_line.iloc[i - lookback_period:i + 1], histogram.iloc[i - lookback_period:i + 1], lookback_period, divergence_threshold)
        current_price = data['Close'].iloc[i]
        current_rsi = rsi.iloc[i]
        timestamp = data.index[i]
        current_histogram = histogram.iloc[i]
        prev_histogram = histogram.iloc[i - 1] if i > 0 else current_histogram

        if divergence in ('regular_bullish', 'hidden_bullish'):
            active_divergence = 'bullish'
        elif divergence in ('regular_bearish', 'hidden_bearish'):
            active_divergence = 'bearish'

        if active_divergence == 'bullish':
            # Check for MACD histogram and RSI conditions for open long
            if current_histogram < 0 and current_histogram > prev_histogram:# and current_rsi < rsi_oversold:
                open_long_signals.append((timestamp, current_price))
                active_divergence = None  # Reset active divergence after signal

        elif active_divergence == 'bearish':
            # Check for MACD histogram and RSI conditions for open short
            if current_histogram > 0 and current_histogram < prev_histogram:# and current_rsi > rsi_overbought:
                open_short_signals.append((timestamp, current_price))
                active_divergence = None  # Reset active divergence after signal

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Price Chart
    ax1.plot(data['Close'], label='Price')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Chart with Potential Open Long/Short Signals')

    # Plot Open Long/Short Signals
    if open_long_signals:
        long_timestamps, long_prices = zip(*open_long_signals)
        ax1.plot(long_timestamps, long_prices, '^', markersize=10, color='g', label='Open Long')

    if open_short_signals:
        short_timestamps, short_prices = zip(*open_short_signals)
        ax1.plot(short_timestamps, short_prices, 'v', markersize=10, color='r', label='Open Short')

    # MACD Histogram
    ax2.plot(macd, label='MACD')
    ax2.plot(signal_line, label='Signal Line')
    ax2.bar(histogram.index, histogram, label='Histogram', color='b')
    ax2.set_ylabel('MACD')
    ax2.set_title('MACD Histogram')

    # RSI
    ax3.plot(rsi, label='RSI')
    ax3.axhline(y=rsi_oversold, color='r', linestyle='--', label=f'Oversold ({rsi_oversold})')
    ax3.axhline(y=rsi_overbought, color='r', linestyle='--', label=f'Overbought ({rsi_overbought})')
    ax3.set_ylabel('RSI')
    ax3.set_title('Relative Strength Index (RSI)')

    # Format and display the plot
    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Save the figure
    plt.savefig('trading_signals.png')
    plt.close()  # Close the figure to free up memory

# Fetch historical data using yfinance
data = yf.download(ticker, period=period, interval=interval)

# Plot the signals and save the image
plot_signals(data)