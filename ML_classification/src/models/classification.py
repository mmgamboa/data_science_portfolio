import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

class ClassificationModel:
    def __init__(self, which_model:str):
        self.which_model = which_model
            
    def train(self, x_train, y_train):
        if self.which_model == 'xgboost':
            return self.xgboost(x_train, y_train)
        else:
            raise ValueError(f"Model {self.model} not supported")
    
    def evaluate(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        # 5.3 Asses performance
        accuracy = accuracy_score(y_test, predictions)
        print(f"\n Model accuracy: {accuracy} \n")
        return predictions, accuracy
    
    def xgboost(self, x_train, y_train):
        # 5.1 Train the model
        self.model = xgb.XGBClassifier()
        self.model.fit(x_train.drop(columns=['PassengerId', 'Name']), 
                y_train)
        return self.model
