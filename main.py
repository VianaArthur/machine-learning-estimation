import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from model import Results, Predictor, Stats

#TODO:
# Add log

training_qnt = 3

def get_mlp_regressor(hidden_layer_sizes, activation, solver, max_iter, x_train, y_train, x_test):
  regressor = MLPRegressor(        
    hidden_layer_sizes = hidden_layer_sizes,      
    activation = activation,          
    solver = solver,             
    max_iter = max_iter,
    random_state = None) 
  return regressor

def get_rf_regressor(n_estimators, criterion, x_train, y_train, x_test):
  regressor = RandomForestRegressor(        
    n_estimators = n_estimators,      
    criterion = criterion,
    random_state = None)
  return regressor

#Reads excel.xlsx file
df = pd.concat(pd.read_excel('./data/Excel-Dataset.xlsx', sheet_name=None), ignore_index=True)
#Export to CSV file
df.to_csv('./data/Dataset.csv', encoding='utf-8', index=False)

# Dataset samples, read .CSV file
dataset = pd.read_csv('./data/Dataset.csv')
# dropping passed columns
dataset.drop(["Silica", "Fosfato", "Alumina", "Manganes", "Titanio", "Magnesio", "Carbonato"], axis = 1, inplace = True)  

x = dataset.iloc[:, 1:].values  #Predictors
y = dataset.iloc[:, 0].values  #To be Predicted
y = y.reshape(-1, 1) # needs 2D vector

arrays_RF5 = Stats()
arrays_RF20 = Stats()
arrays_MLP55 = Stats()
arrays_MLP2020 = Stats()

# For loop to vary the seed and train "i" times
for i in range(1, training_qnt+1):
  # splitting Dataset  samples
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=None)
  
  # Get Regressors
  regressor_RF5 = get_rf_regressor(5,'mse', x_train, y_train, x_test)
  regressor_RF20 = get_rf_regressor(20,'mse', x_train, y_train, x_test)
  regressor_MLP55 = get_mlp_regressor((5,5), 'tanh', 'lbfgs', 1000, x_train, y_train, x_test)
  regressor_MLP2020 = get_mlp_regressor((20,20), 'tanh', 'lbfgs', 1000, x_train, y_train, x_test)

  y_pred_RF5 = Predictor(regressor_RF5, x_train, y_train, x_test).predict()
  y_pred_RF20 = Predictor(regressor_RF20, x_train, y_train, x_test).predict()
  y_pred_MLP55 = Predictor(regressor_MLP55, x_train, y_train, x_test).predict()
  y_pred_MLP2020 = Predictor(regressor_MLP2020, x_train, y_train, x_test).predict()

  results_RF5 = Results(y_test, y_pred_RF5, i, 'RF', 'RF5')
  results_RF20 = Results(y_test, y_pred_RF20, i, 'RF', 'RF20')
  results_MLP55 = Results(y_test, y_pred_MLP55, i, 'MLP', 'MLP55')
  results_MLP2020 = Results(y_test, y_pred_MLP2020, i, 'MLP', 'MLP2020')

  arrays_RF5.append_array(results_RF5.mse, results_RF5.mae, results_RF5.mape, results_RF5.r2)
  arrays_RF20.append_array(results_RF20.mse, results_RF20.mae, results_RF20.mape, results_RF20.r2)
  arrays_MLP55.append_array(results_MLP55.mse, results_MLP55.mae, results_MLP55.mape, results_MLP55.r2)
  arrays_MLP2020.append_array(results_MLP2020.mse, results_MLP2020.mae, results_MLP2020.mape, results_MLP2020.r2)

  results_RF5.save_to_txt()
  results_RF20.save_to_txt()
  results_MLP55.save_to_txt()
  results_MLP2020.save_to_txt()
  
  # Create a workbook for each iteration and saves all the tests id different sheets
  with pd.ExcelWriter('./results/estimated/RF/RF_OutputSeed_{}.xlsx'.format(i)) as writer: # pylint: disable=abstract-class-instantiated
    pd.DataFrame(y_pred_RF5, columns=['Predicted']).to_excel(writer, sheet_name='RF5', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='RF5', startcol=1, index=False)
    
    pd.DataFrame(y_pred_RF20, columns=['Predicted']).to_excel(writer, sheet_name='RF20', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='RF20', startcol=1, index=False)
  
  # Create a workbook for each iteration and saves all the tests id different sheets
  with pd.ExcelWriter('./results/estimated/MLP/MLP_OutputSeed_{}.xlsx'.format(i)) as writer: # pylint: disable=abstract-class-instantiated
    pd.DataFrame(y_pred_MLP55, columns=['Predicted']).to_excel(writer, sheet_name='MLP55', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='MLP55', startcol=1, index=False)
    
    pd.DataFrame(y_pred_MLP2020, columns=['Predicted']).to_excel(writer, sheet_name='MLP2020', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='MLP2020', startcol=1, index=False)

# Generates Excel File saving the MSE results    
with pd.ExcelWriter('./results/statistics/MSE_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(arrays_RF5.mse_array, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(arrays_RF20.mse_array, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(arrays_MLP55.mse_array, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False) 
  pd.DataFrame(arrays_MLP2020.mse_array, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)   

# Generates Excel File saving the MAE results    
with pd.ExcelWriter('./results/statistics/MAE_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(arrays_RF5.mae_array, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(arrays_RF20.mae_array, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(arrays_MLP55.mae_array, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False) 
  pd.DataFrame(arrays_MLP2020.mae_array, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)   

# Generates Excel File saving the MAPE results    
with pd.ExcelWriter('./results/statistics/MAPE_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(arrays_RF5.mape_array, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(arrays_RF20.mape_array, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(arrays_MLP55.mape_array, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False)   
  pd.DataFrame(arrays_MLP2020.mape_array, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)   

# Generates Excel File saving the R2 results    
with pd.ExcelWriter('./results/statistics/R2_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(arrays_RF5.r2_array, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(arrays_RF20.r2_array, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(arrays_MLP55.r2_array, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False)   
  pd.DataFrame(arrays_MLP2020.r2_array, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)
