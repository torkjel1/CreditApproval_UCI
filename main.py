import pandas as pd
import numpy as np 
import data
from ucimlrepo import fetch_ucirepo
from model import Model
from manager import ModelManager
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
def main():
    
    def data_fetching_and_preprocessing():
        # fetch dataset 
        credit_approval = fetch_ucirepo(id=27)
        
        
          
        # data (as pandas dataframes) 
        X = credit_approval.data.features 
        y = credit_approval.data.targets 
        y = y.replace({"+": 1, "-": 0})
        
        
        variable_types_X = ["continuous", "continuous", "categorical",
                        "categorical", "continuous", "categorical",
                        "categorical", "continuous", "categorical",
                        "categorical", "categorical", "categorical",
                        "continuous", "continuous", "categorical"]
        
        variable_types_X = dict(zip(X.columns, variable_types_X))
        
        
        for key, val in variable_types_X.items():
            if val != "continuous" and val != "categorical":
                raise ValueError("Incorrect variable type assignment")
                
        
        # Deal with categorical data
        X = pd.get_dummies(X, columns = [key for key, val in variable_types_X.items()
                                     if val == "categorical"
                                     ], drop_first = True)
        
        
        for key in X.columns:
            if not variable_types_X.get(key):
                variable_types_X[key] = "categorical"
        
        
        #Deal with missing values
        
        continuous_cols = [key for key, val in variable_types_X.items() if val == "continuous" and key in X.columns]
        categorical_cols = [key for key, val in variable_types_X.items() if val == "categorical" and key in X.columns]
        
        # Mean for continous cols
        X[continuous_cols] = X[continuous_cols].apply(lambda col: col.fillna(col.mean()))
        
        # Median for categorical
        X[categorical_cols] = X[categorical_cols].apply(lambda col: col.fillna(col.median()))
        
        #standardize continuous vars
        
        X[continuous_cols] = X[continuous_cols].apply(lambda col: data.standardize(col))
        
        y = y.iloc[:, 0]
        
        return (X, y)
    
    def create_models():
        
        X, y = data_fetching_and_preprocessing()
        
        model_manager = ModelManager(X, y)
        
        kernels = ["linear", "rbf", "sigmoid" ]
        gammas = [0.5, 1, 2, 3, 4]
        Cs = [0.01*10**i for i in range(3)] + [2]
        degrees = [1, 2, 3, 4, 5]
        param_grid = {"kernel": kernels, "gamma": gammas, "C": Cs, "degree":degrees}
        
        svc_model = Model(SVC, param_grid)
        
        svc_model.calculate_accuracy_measures_test_set()
        
        logistic_regression_model = Model(LogisticRegressionCV, penalty='l2', solver='lbfgs', max_iter=1000)
        
        logistic_regression_model.calculate_accuracy_measures_test_set()
        
        param_grid = {"n_estimators": np.arange(5, 100), "max_features": ["sqrt", "log2"]}
        
        random_forest_model = Model(RandomForestClassifier, param_grid)
        
        random_forest_model.calculate_accuracy_measures_test_set()
        
        param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['auto', 'sqrt', 'log2']
        }
        
        gradient_boost_model = Model(GradientBoostingClassifier, param_grid)
        
        gradient_boost_model.calculate_accuracy_measures_test_set()
        
        best_model = model_manager.pick_model()
        
        print(f"The best model is {best_model.model} and achieves an fscore of {best_model.fscore:.1%}")
            
        
        
     

        
        
    create_models()
    
    
if __name__ == "__main__":
   main()

