import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Trading parameters
ticker = "BTC-USD"
period = "1mo"
interval = "1h"
macd_fast = 12
macd_slow = 26
macd_signal = 9
lookback_period = 144
atr_period = 14
divergence_threshold = 0
rsi_period = 13
rsi_oversold = 30
rsi_overbought = 70

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

def find_histogram_peaks(histogram):
    peak_indices = []

    # Pad the histogram with zeros at the beginning and end for edge cases
    padded_histogram = np.pad(histogram.values, (1, 1), 'constant', constant_values=(0, 0))

    # Find indices where the histogram crosses zero
    zero_crossings = np.where(np.diff(np.signbit(padded_histogram)))[0]
    #print(f'number of sections: {len(zero_crossings)}')

    # Iterate through each section and find the maximum or minimum
    for i in range(len(zero_crossings) - 1):
        start = zero_crossings[i]
        end = zero_crossings[i+1]
        section = histogram.iloc[start:end]

        if not section.empty:
            if section.iloc[0] >= 0:  # Positive section
                peak_index = section.idxmax()
            else:  # Negative section
                peak_index = section.idxmin()

            peak_indices.append(peak_index)

    return pd.Series(peak_indices)

def detect_divergence(data, macd, signal_line, histogram, lookback, threshold):
    """Detects regular and hidden bullish/bearish divergences with a threshold."""
    price = data['Close']

    divergence = 'none'

    peak_indices = find_histogram_peaks(histogram)
    histogram_peaks = histogram.loc[peak_indices]
    histogram_maxima = histogram_peaks[histogram_peaks > 0]
    histogram_minima = histogram_peaks[histogram_peaks < 0]

    # Iterate through all pairs of sequential histogram maxima for bearish divergences
    if len(histogram_maxima) >= 2:
        for i in range(len(histogram_maxima)):
            for j in range(i + 1, len(histogram_maxima)):
                # Corresponding price points for histogram maxima
                price_maxima = price.loc[histogram_maxima.index]

                # Regular Bearish Divergence
                if price_maxima.iloc[j] > price_maxima.iloc[i]:  # Higher high in price
                    if histogram_maxima.iloc[j] < histogram_maxima.iloc[i]:  # Lower high in histogram
                        price_diff = abs((price_maxima.iloc[j] - price_maxima.iloc[i]) / price_maxima.iloc[i])
                        histogram_diff = abs((histogram_maxima.iloc[i] - histogram_maxima.iloc[j]) / histogram_maxima.iloc[j])
                        if price_diff > threshold and histogram_diff > threshold:
                            divergence = 'regular_bearish'
                            break

                # Hidden Bearish Divergence
                if price_maxima.iloc[j] < price_maxima.iloc[i]:  # Lower high in price
                    if histogram_maxima.iloc[j] > histogram_maxima.iloc[i]:  # Higher high in histogram
                        price_diff = abs((price_maxima.iloc[i] - price_maxima.iloc[j]) / price_maxima.iloc[j])
                        histogram_diff = abs((histogram_maxima.iloc[j] - histogram_maxima.iloc[i]) / histogram_maxima.iloc[i])
                        if price_diff > threshold and histogram_diff > threshold:
                            divergence = 'hidden_bearish'
                            break
            if divergence != 'none':
                break

    if divergence != 'none':
        return divergence

    # Iterate through all pairs of sequential histogram minima for bullish divergences
    if len(histogram_minima) >= 2:
        for i in range(len(histogram_minima)):
            for j in range(i + 1, len(histogram_minima)):
                # Corresponding price points for histogram minima
                price_minima = price.loc[histogram_minima.index]

                # Regular Bullish Divergence
                if price_minima.iloc[j] < price_minima.iloc[i]:  # Lower low in price
                    #print("test1")
                    if histogram_minima.iloc[j] > histogram_minima.iloc[i]:  # Higher low in histogram
                        #print("test2")
                        #divergence = 'regular_bullish'
                        #break
                        price_diff = abs((price_minima.iloc[i] - price_minima.iloc[j]) / price_minima.iloc[j])
                        histogram_diff = abs((histogram_minima.iloc[j] - histogram_minima.iloc[i]) / histogram_minima.iloc[i])
                        if price_diff > threshold and histogram_diff > threshold:
                            divergence = 'regular_bullish'
                            break

                # Hidden Bullish Divergence
                if price_minima.iloc[j] > price_minima.iloc[i]:  # Higher low in price
                    if histogram_minima.iloc[j] < histogram_minima.iloc[i]:  # Lower low in histogram
                        price_diff = abs((price_minima.iloc[j] - price_minima.iloc[i]) / price_minima.iloc[i])
                        histogram_diff = abs((histogram_minima.iloc[i] - histogram_minima.iloc[j]) / histogram_minima.iloc[j])
                        if price_diff > threshold and histogram_diff > threshold:
                            divergence = 'hidden_bullish'
                            break
            if divergence != 'none':
                break

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
            #print(f'Bullish Divergence detected at {timestamp} with price {current_price} and histogram {current_histogram}')
        elif divergence in ('regular_bearish', 'hidden_bearish'):
            active_divergence = 'bearish'
            #print(f'Bearish Divergence detected at {timestamp} with price {current_price} and histogram {current_histogram}')

        if active_divergence == 'bullish':
            # Check for MACD histogram and RSI conditions for open long
            if current_histogram < 0 and current_histogram > prev_histogram and current_rsi < rsi_oversold:
                open_long_signals.append((timestamp, current_price))
                active_divergence = None  # Reset active divergence after signal

        elif active_divergence == 'bearish':
            # Check for MACD histogram and RSI conditions for open short
            if current_histogram > 0 and current_histogram < prev_histogram and current_rsi > rsi_overbought:
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