from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pds
import numpy as np
import os

class Loader:
    def __init__(
        self,
        path,
        fold,
        scaling = None
    ):
        
        assert scaling in ['std','minmax',None], "please fill in the scaling type by 'std', 'minmax', or None"
            
        self.scaling = True if not scaling == None else False
        if self.scaling: self.scaling_type = scaling
            
        train_path = os.path.join(path,f'train_{fold}.csv')
        test_path = os.path.join(path,f'test_{fold}.csv')
        
        self.train_csv = pds.read_csv(train_path)
        self.test_csv = pds.read_csv(test_path)
        self.columns = self.train_csv.columns
        self.raw_train_x, self.train_y = self.train_csv.iloc[:,:-1].values,self.train_csv.iloc[:,-1].values
        self.raw_test_x, self.test_y = self.test_csv.iloc[:,:-1].values,self.test_csv.iloc[:,-1].values
        
        
    def __scaling__(self):
        
        if self.scaling:
            if self.scaling_type == 'std':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler((-1,1))
            scaled_train_x = self.scaler.fit_transform(self.raw_train_x)
            scaled_test_x = self.scaler.transform(self.raw_test_x)

            return scaled_train_x, self.train_y, scaled_test_x, self.test_y
        else:
            train_x,test_x = self.raw_train_x, self.raw_test_x
            return train_x, self.train_y, test_x, self.test_y
        
        
    def __call__(self):
        '''Return the train_x, train_y, test_x, test_y'''
        return self.__scaling__()
        