import numpy as np
from pygam import LinearGAM


class GAM:
    
    def __init__(self, n_splines):
        self.n_splines = n_splines
        
    def fit(self, x, y):
        # Check `x` dimensionality
        x = np.array(x)
        assert len(x.shape) == 2, f'`x` should be 2D array. Received {len(x.shape)} dimensional array.'
        
        # Fit the model
        self.model = LinearGAM(n_splines=self.n_splines).gridsearch(x, y) 
        
    def predict(self, x):
        return self.model.predict(x)