import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Results:

  def __init__(self, y_test, y_pred, seed, folder, name):
    self.y_test = y_test
    self.y_pred = y_pred
    self.seed = seed
    self.folder = folder
    self.name = name

  @property
  def mse(self):
    return mean_squared_error(self.y_test, self.y_pred)

  @property
  def mae(self):
    return mean_absolute_error(self.y_test, self.y_pred)

  @property
  def r2(self):
    return r2_score(self.y_test, self.y_pred)

  @property
  def mape(self): 
    y_test = np.array(self.y_test)
    y_pred = np.array(self.y_pred)
    return np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100

  def save_to_txt(self):
    # Saves the statistical results of method to a .txt file    
    with open('./results/txt/{}/OutputStatisticalResults_{}.txt'.format(self.folder, self.name), 'a') as f:
      print('\n#Seed {} - Method {}#'.format(self.seed, self.name), file=f)
      print('\n***Results***', file=f)
      print('MSE: %.2f' % self.mse, file=f) 
      print('MAE: %.2f' % self.mae, file=f) 
      print('MAPE: %.2f' % self.mape, file=f) 
      print('R2: %.2f' % self.r2, file=f) 

class Predictor:
  def __init__(self, regressor, x_train, y_train, x_test):
    self.regressor = regressor
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test

  def predict(self):
    # Fit the model to data matrix X and target(s) y.
    self.regressor.fit(self.x_train, self.y_train.ravel())
    # Predict using the multi-layer perceptron model.
    predicted = self.regressor.predict(self.x_test)
    return predicted

class Stats:

  def __init__(self):
    self.mse_array = np.array([])
    self.mae_array = np.array([])
    self.mape_array = np.array([])
    self.r2_array = np.array([])

  def append_array(self, mse, mae, mape, r2):
    self.mse_array = np.append(self.mse_array, mse)
    self.mae_array = np.append(self.mae_array, mae)
    self.mape_array = np.append(self.mape_array, mape)
    self.r2_array = np.append(self.r2_array, r2)