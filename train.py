from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
import pandas as pd

def sarimax_train(series):

    '''
    Takes time series and returns a fitted sarimax model with predefined parameters
    :param series: Training series to use as baseline for the prediction
    :return: Prediction series of length "predictionlengh"
    '''

    # TODO: Pass parameters as arguments
    
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 7), simple_differencing=False)
    model_fit = model.fit()
    return model_fit

def auto_arima_forecast(series, feature, split):
    '''
    Generation of model and fit for a given series with auto_arima.
    Function takes all series as input, splits it as training and testing, and returns:
    - Model
    - Forecast dataframe
    - Test timeseries to calculate errors

    :param series: Pandas dataframe containing the data to be predicted
    :param feature: Column name in the dataframe to predict
    :param split: Ratio of data to be used as training data [0,1]
    :return: trained model, forecast as DataFrame with "prediction" label, testing split of the original series
    '''

    series_train = series.iloc[:int(len(series.index) * split), :]
    series_test = series.iloc[int(len(series.index) * split):, :]

    model = auto_arima(series_train[feature], trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(series_train[feature])
    forecast = model.predict(n_periods=len(series_test))
    forecast = pd.DataFrame(forecast, index=series_test.index, columns=['prediction'])

    return model, forecast, series_test
