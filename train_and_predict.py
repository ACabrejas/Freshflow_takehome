from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_predict(series, prediction_length):

    '''
    Packaging of sarimax into a function with predetermined parameters
    :param series: Training series to use as baseline for the prediction
    :param prediction_length: Desired prediction length at the end of the training serioes
    :return: Prediction series of length "predictionlengh"
    '''

    # TODO: Pass parameters as arguments
    
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 7), simple_differencing=False)
    model_fit = model.fit()
    pred = model_fit.predict(prediction_length)
    return pred