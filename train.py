from statsmodels.tsa.statespace.sarimax import SARIMAX

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