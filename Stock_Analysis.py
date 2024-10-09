import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

class StockAnalysis:
    def __init__(self, ticker, start, end):
        """Initialize with stock ticker, start and end date for data."""
        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = None  # Initialize data to None
        try:
            self.data = self.download_data()  # Download stock data at initialization
        except Exception as e:
            st.error(f"Error downloading data: {e}")
        self.indicators = {}  # To store calculated indicators

    def download_data(self):
        """Download stock data from yfinance with error handling."""
        try:
            st.write(f"Downloading data for **{self.ticker}** from **{self.start}** to **{self.end}**")
            data = yf.download(self.ticker, start=self.start, end=self.end)
            if data.empty:
                raise ValueError(f"No data found for {self.ticker}. Check the ticker symbol or date range.")
            return data
        except ValueError as ve:
            st.error(ve)
            raise
        except Exception as e:
            st.error(f"An error occurred: {e}")
            raise

    # Indicator Methods
    def calculate_moving_average(self, window):
        """Calculate Simple Moving Average (SMA) with error handling."""
        try:
            self.indicators[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
        except KeyError:
            st.error("Error: 'Close' column not found in the data.")
        except Exception as e:
            st.error(f"An error occurred while calculating SMA: {e}")

    def calculate_exponential_moving_average(self, window):
        """Calculate Exponential Moving Average (EMA) with error handling."""
        try:
            self.indicators[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        except KeyError:
            st.error("Error: 'Close' column not found in the data.")
        except Exception as e:
            st.error(f"An error occurred while calculating EMA: {e}")

    def calculate_rsi(self, window=14):
        """Calculate Relative Strength Index (RSI) with error handling."""
        try:
            delta = self.data['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            self.indicators['RSI'] = rsi
        except KeyError:
            st.error("Error: 'Close' column not found in the data.")
        except Exception as e:
            st.error(f"An error occurred while calculating RSI: {e}")

    def calculate_macd(self, short_window=12, long_window=26, signal_window=9):
        """Calculate MACD and Signal Line with error handling."""
        try:
            short_ema = self.data['Close'].ewm(span=short_window, adjust=False).mean()
            long_ema = self.data['Close'].ewm(span=long_window, adjust=False).mean()
            macd_line = short_ema - long_ema
            signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
            self.indicators['MACD'] = macd_line
            self.indicators['Signal_Line'] = signal_line
        except KeyError:
            st.error("Error: 'Close' column not found in the data.")
        except Exception as e:
            st.error(f"An error occurred while calculating MACD: {e}")

    def calculate_bollinger_bands(self, window=20, num_sd=2):
        """Calculate Bollinger Bands with error handling."""
        try:
            sma = self.data['Close'].rolling(window=window).mean()
            std_dev = self.data['Close'].rolling(window=window).std()
            upper_band = sma + (std_dev * num_sd)
            lower_band = sma - (std_dev * num_sd)
            self.indicators['Bollinger_SMA'] = sma
            self.indicators['Bollinger_Upper'] = upper_band
            self.indicators['Bollinger_Lower'] = lower_band
        except KeyError:
            st.error("Error: 'Close' column not found in the data.")
        except Exception as e:
            st.error(f"An error occurred while calculating Bollinger Bands: {e}")

    def calculate_atr(self, window=14):
        """Calculate Average True Range (ATR) with error handling."""
        try:
            high_low = self.data['High'] - self.data['Low']
            high_close = np.abs(self.data['High'] - self.data['Close'].shift(1))
            low_close = np.abs(self.data['Low'] - self.data['Close'].shift(1))
            true_range = pd.DataFrame([high_low, high_close, low_close]).max(axis=0)
            atr = true_range.rolling(window=window).mean()
            self.indicators['ATR'] = atr
        except KeyError:
            st.error("Error: Required columns ('High', 'Low', 'Close') not found in the data.")
        except Exception as e:
            st.error(f"An error occurred while calculating ATR: {e}")

    # Visualization
    def plot_price_and_indicators(self):
        """Plot the stock price along with selected indicators using Plotly."""
        if self.data is None:
            st.warning("No data available to plot.")
            return

        fig = go.Figure()
        
        # Plot the Closing Price
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], mode='lines', name='Closing Price', line=dict(color='blue', width=2)))

        # Plot Moving Averages (SMA and EMA)
        for indicator in self.indicators:
            if 'SMA' in indicator or 'EMA' in indicator:
                fig.add_trace(go.Scatter(x=self.indicators[indicator].index, y=self.indicators[indicator], mode='lines', name=indicator))

        # Plot Bollinger Bands
        if 'Bollinger_Upper' in self.indicators and 'Bollinger_Lower' in self.indicators:
            fig.add_trace(go.Scatter(x=self.indicators['Bollinger_Upper'].index, y=self.indicators['Bollinger_Upper'], mode='lines', name='Upper Bollinger Band', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=self.indicators['Bollinger_Lower'].index, y=self.indicators['Bollinger_Lower'], mode='lines', name='Lower Bollinger Band', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(
                x=self.indicators['Bollinger_Upper'].index,
                y=self.indicators['Bollinger_Upper'],
                mode='lines',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=self.indicators['Bollinger_Lower'].index,
                y=self.indicators['Bollinger_Lower'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(211,211,211,0.5)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Bollinger Bands',
                showlegend=False
            ))

        fig.update_layout(title=f'{self.ticker} Price with Moving Averages and Bollinger Bands', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)  # Display the plot in Streamlit

    def plot_rsi(self):
        """Plot the Relative Strength Index (RSI)."""
        fig = go.Figure()
        if 'RSI' in self.indicators:
            fig.add_trace(go.Scatter(x=self.indicators['RSI'].index, y=self.indicators['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
            fig.add_hline(y=70, line_color='red', line_dash='dash', annotation_text='Overbought', annotation_position='bottom right')
            fig.add_hline(y=30, line_color='green', line_dash='dash', annotation_text='Oversold', annotation_position='bottom right')

            fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)  # Display the plot in Streamlit
        else:
            st.warning("RSI data not available.")

    def plot_macd(self):
        """Plot MACD and Signal Line."""
        fig = go.Figure()
        if 'MACD' in self.indicators and 'Signal_Line' in self.indicators:
            fig.add_trace(go.Scatter(x=self.indicators['MACD'].index, y=self.indicators['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=self.indicators['Signal_Line'].index, y=self.indicators['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='orange')))

            fig.update_layout(title='MACD and Signal Line', xaxis_title='Date', yaxis_title='MACD', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)  # Display the plot in Streamlit
        else:
            st.warning("MACD data not available.")

    def plot_atr(self):
        """Plot the Average True Range (ATR)."""
        fig = go.Figure()
        if 'ATR' in self.indicators:
            fig.add_trace(go.Scatter(x=self.indicators['ATR'].index, y=self.indicators['ATR'], mode='lines', name='ATR', line=dict(color='orange')))

            fig.update_layout(title=f'Average True Range (ATR) for {self.ticker}', xaxis_title='Date', yaxis_title='ATR', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)  # Display the plot in Streamlit
        else:
            st.warning("ATR data not available.")

# Main Streamlit Application
def main():
    st.set_page_config(page_title="Stock Analysis Tool", layout="wide")
    st.title("📈 Stock Analysis Tool")
    st.markdown("Analyze stock trends and indicators with ease!")
    
    # User Inputs
    with st.sidebar:
        st.header("User Input")
        ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL, MSFT):", value='AAPL')
        start_date = st.date_input("Select the start date", pd.to_datetime("2022-01-01"))
        end_date = st.date_input("Select the end date", pd.to_datetime("today"))

        # SMA and EMA windows
        sma_window = st.number_input("SMA Window", min_value=1, value=20, step=1)
        ema_window = st.number_input("EMA Window", min_value=1, value=20, step=1)

    # Process the input and perform analysis
    if st.button("Analyze"):
        if ticker and start_date < end_date:
            analysis = StockAnalysis(ticker, start_date, end_date)
            analysis.calculate_moving_average(sma_window)
            analysis.calculate_exponential_moving_average(ema_window)
            analysis.calculate_rsi()
            analysis.calculate_macd()
            analysis.calculate_bollinger_bands()
            analysis.calculate_atr()
            
            # Visualize indicators
            analysis.plot_price_and_indicators()
            analysis.plot_rsi()
            analysis.plot_macd()
            analysis.plot_atr()
        else:
            if not ticker:
                st.error("Error: Please enter a valid stock ticker.")
            if start_date >= end_date:
                st.error("Error: Start date must be before end date.")

if __name__ == "__main__":
    main()
