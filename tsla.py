import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
ticker = 'TSLA'
df = yf.download(ticker, start="2010-01-01", end="2024-12-31")

# Drop the first row because it contains repeated column names
df = df.iloc[1:].reset_index(drop=True)
df["Date"] = pd.date_range(start="2010-01-01", periods=len(df), freq='D')
df.set_index("Date", inplace=True)

#######################################################################
####################  Time-Series Analysis    #########################
#######################################################################
# Decompose time series
decomposition = seasonal_decompose(df["Close"], model='additive', period=30)

# Plot decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df["Close"], label='Original')
plt.legend()
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend()
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend()
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend()
plt.tight_layout()
plt.show()

#######################################################################
#################### Event Study Methodology  #########################
#######################################################################

# Tesla Model 3 launch (July 7, 2017)
# Cybertruck reveal (Nov 21, 2019)
# Tesla added to S&P 500 (Dec 21, 2020)
events = ["2017-07-07", "2019-11-21", "2020-12-21"]
event_window = 10  # 10 days before and after
df["Date"] = pd.date_range(start="2010-01-01", periods=len(df), freq='D')
# Function to calculate returns
df["Return"] = df["Close"].pct_change()

# Analyze each event
event_results = {}

for event in events:
    event_date = pd.to_datetime(event)

    # Extract event window data
    mask = (df["Date"] >= event_date - pd.Timedelta(days=event_window)) & \
           (df["Date"] <= event_date + pd.Timedelta(days=event_window))
    event_data = df[mask].copy()

    # Calculate expected return (using average past returns)
    event_data["Expected Return"] = df["Return"].rolling(window=30, min_periods=1).mean()

    # Calculate Abnormal Returns (AR) and Cumulative Abnormal Returns (CAR)
    event_data["Abnormal Return"] = event_data["Return"] - event_data["Expected Return"]
    event_data["CAR"] = event_data["Abnormal Return"].cumsum()

    event_results[event] = event_data

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(event_data["Date"], event_data["CAR"], label=f"CAR for event {event}", marker="o")
    event_numeric = mdates.date2num(event_date)

    plt.axvline(event_numeric, color='red', linestyle='--', label="Event Date")
    plt.title(f"Cumulative Abnormal Return (CAR) Around {event}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Abnormal Return")
    plt.legend()
    plt.grid()
    plt.show()

# Display results for last event
print(event_results[events[-1]].head())


#######################################################################
####################   Regression Analysis    #########################
#######################################################################

# Convert numeric columns to float
numeric_columns = ["Close", "High", "Low", "Open", "Volume"]
df[numeric_columns] = df[numeric_columns].astype(float)

# Define independent (X) and dependent (Y) variables
X = df[["High", "Low", "Open", "Volume"]]  # Predictors
Y = df["Close"]  # Target variable

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Print the regression summary
print(model.summary())

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of TSLA Financial Data")
plt.show()
