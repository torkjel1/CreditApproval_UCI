import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from manager import ModelManager



class Model():
    
  
    
    def __init__(self, model_class, scoring = "f1", param_grid = None, *args, **kwargs):
        
        self.model_manager = ModelManager.get_instance()
        self.predictions = {}
        self.model = self.create_model(model_class, param_grid, *args, **kwargs)
        self.precision = None
        self.recall = None
        self.fscore = None
        self.scoring = scoring
        
    
    def create_model(self, model_class, grid_search_param_grid = None, *args, **kwargs):
        
        
        model = model_class(*args, **kwargs)
        
        if grid_search_param_grid:
            model = GridSearchCV(model, param_grid = grid_search_param_grid, cv=5, n_jobs=-1, scoring = self.scoring)
        
        model.fit(self.model_manager.get_X_train(), self.model_manager.get_y_train())
        
        ModelManager.get_instance().models.add(self)
        
        return model
        
 
    def predict(self, X_features): #to allow for memoization
    
        if X_features is self.model_manager.X_train:
            
            if "train" not in self.predictions:
                self.predictions["train"] = self.model.predict(X_features)
            
            return self.predictions["train"]
        
        elif X_features is self.model_manager.X_test:
            
            if "test" not in self.predictions:
                self.predictions["test"] = self.model.predict(X_features)
            
            return self.predictions["test"]
        
        return None
    
            
    #Methods for accuracy mesaures
    def calculate_accuracy_measures(self, X_features_retriever, y_labels_retriever):
        
        if not self.model:
            return None
        
        y_pred = self.model.predict(X_features_retriever())
        
        precision, recall, fscore, _ = precision_recall_fscore_support(y_labels_retriever(), y_pred, average = "macro")
        
        return precision, recall, fscore
      
    def calculate_accuracy_measures_train_set(self):
        
        precision, recall, fscore = self.calculate_accuracy_measures(self.model_manager.get_X_train, self.model_manager.get_y_train)
        return precision, recall, fscore
    
    def calculate_accuracy_measures_test_set(self):
        
        if not (self.precision and self.recall and self.fscore):
        
            precision, recall, fscore = self.calculate_accuracy_measures(self.model_manager.get_X_test, self.model_manager.get_y_test)
            self.precision = precision
            self.recall = recall
            self.fscore = fscore
            
        return precision, recall, fscore
    
    
    def plot_PR_curve(self):
        
        precision, recall, _ = precision_recall_curve(self.model_manager.get_y_test(), self.predict(self.model_manager.get_X_test()))
    
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, color='blue', lw=2,)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend(loc="best")
        
    def plot_ROC_curve(self):
        
        fpr, tpr, _ = roc_curve(self.model_manager.get_y_test(), self.predict(self.model_manager.get_X_test()))
        
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="best")
        plt.tight_layout(pad=3.0)

       
