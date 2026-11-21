# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 18/11/2026
# Data set : Tesla stock price



### AIM:
To implement the ARMA (AutoRegressive Moving Average) model using Python on the Tesla (TSLA) stock dataset and analyze the time series data using Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the TSLA dataset
data = pd.read_csv("/content/TSLA.csv")

# Use correct columns for TSLA: 'Date' and 'Close' instead of 'Month' and '#Passengers'
data["Date"] = pd.to_datetime(data["Date"])
data = data.dropna(subset=["Close"])
data = data.sort_values("Date")
data = data.reset_index(drop=True)

X = data["Close"]
print("Rows:", len(X))
print("Missing values:", X.isnull().sum())

# =====================================================
# Original TSLA Close Price Plot
# =====================================================

plt.figure(figsize=(12, 5))
plt.plot(data["Date"], X.values)
plt.title("Original TSLA Close Price Data")
plt.xlabel("Year")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# =====================================================
# Original ACF & PACF
# =====================================================

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(
    X,
    lags=20,
    ax=ax[0],
    zero=False
)
ax[0].set_title("TSLA Close ACF")
ax[0].set_xlim(0, 20)

plot_pacf(
    X,
    lags=20,
    ax=ax[1],
    method="ywm",
    zero=False
)
ax[1].set_title("TSLA Close PACF")
ax[1].set_xlim(0, 20)

plt.tight_layout()
plt.show()

# =====================================================
# ARMA(1,1)
# =====================================================

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print("\nARMA(1,1) Parameters")
print(arma11_model.params)

phi1 = arma11_model.params["ar.L1"]
theta1 = arma11_model.params["ma.L1"]

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

N = 1000
arma11 = ArmaProcess(ar1, ma1)
ARMA1 = arma11.generate_sample(nsample=N)

plt.figure(figsize=(12, 5))
plt.plot(ARMA1)
plt.title("Simulated ARMA(1,1)")
plt.xlabel("Observation")
plt.ylabel("Value")
plt.xlim(0, 500)
plt.show()

plt.figure(figsize=(10, 5))
plot_acf(
    ARMA1,
    lags=30
)
plt.xlim(0, 30)
plt.title("ACF - Simulated ARMA(1,1)")
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(
    ARMA1,
    lags=30,
    method="ywm"
)
plt.xlim(0, 30)
plt.title("PACF - Simulated ARMA(1,1)")
plt.show()

# =====================================================
# ARMA(2,2)
# =====================================================

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\nARMA(2,2) Parameters")
print(arma22_model.params)

phi1 = arma22_model.params["ar.L1"]
phi2 = arma22_model.params["ar.L2"]

theta1 = arma22_model.params["ma.L1"]
theta2 = arma22_model.params["ma.L2"]

ar2 = np.array([
    1,
    -phi1,
    -phi2
])

ma2 = np.array([
    1,
    theta1,
    theta2
])

arma22 = ArmaProcess(ar2, ma2)
ARMA2 = arma22.generate_sample(nsample=N * 10)

plt.figure(figsize=(12, 5))
plt.plot(ARMA2)
plt.title("Simulated ARMA(2,2)")
plt.xlabel("Observation")
plt.ylabel("Value")
plt.xlim(0, 500)
plt.show()

plt.figure(figsize=(10, 5))
plot_acf(
    ARMA2,
    lags=40
)
plt.xlim(0, 40)
plt.title("ACF - Simulated ARMA(2,2)")
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(
    ARMA2,
    lags=40,
    method="ywm"
)
plt.xlim(0, 40)
plt.title("PACF - Simulated ARMA(2,2)")
plt.show()
```

OUTPUT:

<img width="893" height="368" alt="image" src="https://github.com/user-attachments/assets/b55eec71-a271-4cf8-8630-6033b9101e5d" />


<img width="1025" height="316" alt="image" src="https://github.com/user-attachments/assets/28c285c5-dff1-4966-8697-d9a1097f746c" />


<img width="994" height="308" alt="image" src="https://github.com/user-attachments/assets/47b9f9e3-f18a-4935-b029-2f134ad41247" />


SIMULATED ARMA(1,1) PROCESS:

<img width="876" height="372" alt="image" src="https://github.com/user-attachments/assets/2e301ca2-a802-464a-a4b8-c92b82118545" />


Partial Autocorrelation:

<img width="531" height="355" alt="image" src="https://github.com/user-attachments/assets/7605d621-3902-4b96-9e4c-c4eadfd22e57" />


Autocorrelation :

<img width="515" height="338" alt="image" src="https://github.com/user-attachments/assets/c9bf0acc-98c3-4569-a688-1166262851c3" />



SIMULATED ARMA(2,2) PROCESS:
<img width="904" height="365" alt="image" src="https://github.com/user-attachments/assets/63680b08-c039-4046-bb4e-d2ae31cc0676" />


Partial Autocorrelation:
<img width="518" height="350" alt="image" src="https://github.com/user-attachments/assets/b9c22291-6a5e-4f6b-87da-d45f1b0214f9" />


Autocorrelation :
<img width="552" height="348" alt="image" src="https://github.com/user-attachments/assets/d12df1f7-878d-4747-af62-fcc4862c9b63" />


RESULT:
Thus, the ARMA(1,1) and ARMA(2,2) models were successfully implemented using Python on the Tesla (TSLA) stock dataset. The original closing-price data was analyzed using ACF and PACF plots, and simulated time series data was generated to study the autocorrelation patterns of both ARMA models.
