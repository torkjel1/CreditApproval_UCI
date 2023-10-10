from pandas import Series


def standardize(series: Series):
    return (series - series.mean()) / series.std()
    
    
    
    