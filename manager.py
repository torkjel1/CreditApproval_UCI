import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

class ModelManager:
    _instance = None
    
    def __new__(cls, X: DataFrame, y: Series, *args, **kwargs):
       if not cls._instance:
           cls._instance = super(ModelManager, cls).__new__(cls)
           
           # Initialization logic moved to __new__
           cls._instance.X = X
           cls._instance.y = y
           cls._instance.X_train, cls._instance.X_test, cls._instance.y_train, cls._instance.y_test = train_test_split(X, y, train_size=0.2, random_state=1)
           cls._instance.models = set()

       return cls._instance

    def __init__(self, *args, **kwargs):
       # Empty constructor since the initialization logic has been moved to __new__
       pass
   
    @classmethod
    def get_instance(cls):
        if cls._instance:
            return cls._instance
        else:
            print("Singleton Instance has not been created")
            
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    def get_y_train(self):
        return self.y_train
    
    def get_y_test(self):
        return self.y_test
    
    def pick_model(self, metric = "f1"):
        
        if metric == "f1":
            key = lambda x: x.fscore
            
        elif metric == "recall":
            key = lambda x: x.recall
        
        else:
            key = lambda x: x.recall
            
        return max(ModelManager.get_instance().models, key = key)
            
            
            
        
        
        
        





 


    