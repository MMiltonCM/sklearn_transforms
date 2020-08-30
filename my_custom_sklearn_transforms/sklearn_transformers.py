from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(type(X))
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        aux = X[['OBJETIVO']].transform(lambda x: x != "Sospechoso")
        y = aux.get('OBJETIVO')
        data = data.drop(columns=['OBJETIVO'])
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=10)
        selector.fit(data, y)
        support = selector.get_support()
        features = data.loc[:,support].columns.tolist()
        features.append('OBJETIVO')
        
        # Devolvemos un nuevo dataframe 
        return pd.DataFrame.from_records(data=X, columns=features)