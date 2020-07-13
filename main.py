import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPRegressor

#TODO:
# Add log

# Using function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
  y_true= np.array(y_true)
  y_pred = np.array(y_pred)
  return np.mean( np.abs( (y_true - y_pred) / y_true ) ) * 100

# Array para armazenar os MSE de cada vez que o for iterar
result_array_mse_RF5 = np.array([])
result_array_mse_RF20 = np.array([])
result_array_mse_MLP55 = np.array([])
result_array_mse_MLP2020 = np.array([])

# Array para armazenar os MAE de cada vez que o for iterar
result_array_mae_RF5 = np.array([])
result_array_mae_RF20 = np.array([])
result_array_mae_MLP55 = np.array([])
result_array_mae_MLP2020 = np.array([])

# Array para armazenar os MAPE de cada vez que o for iterar
result_array_mape_RF5 = np.array([])
result_array_mape_RF20 = np.array([])
result_array_mape_MLP55 = np.array([])
result_array_mape_MLP2020 = np.array([])

# Array para armazenar os R2 de cada vez que o for iterar
result_array_r2_RF5 = np.array([])
result_array_r2_RF20 = np.array([])
result_array_r2_MLP55 = np.array([])
result_array_r2_MLP2020 = np.array([])

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

# For loop to vary the seed and train "i" times
for i in range(1,31):
  # splitting Dataset  samples
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=None)
  
  regressor_RF_5 = RandomForestRegressor(        
    n_estimators=5,      
    criterion='mse',
    random_state=None) 

  regressor_RF_20 = RandomForestRegressor(        
    n_estimators=20,      
    criterion='mse',
    random_state=None) 
  
  regressor_MLP_55 = MLPRegressor(        
    hidden_layer_sizes=(5,5),      
    activation='tanh',          
    solver='lbfgs',                
    max_iter=1000,
    random_state=None) 
  
  regressor_MLP_2020 = MLPRegressor(        
    hidden_layer_sizes=(20,20),      
    activation='tanh',          
    solver='lbfgs',                
    max_iter=1000,
    random_state=None) 
  
  # Fit the model to data matrix X and target(s) y.
  regressor_RF_5.fit(x_train, y_train.ravel())
  regressor_RF_20.fit(x_train, y_train.ravel())
  regressor_MLP_55.fit(x_train, y_train.ravel())
  regressor_MLP_2020.fit(x_train, y_train.ravel())
  
  # Predict using the multi-layer perceptron model.
  y_pred_RF5 = regressor_RF_5.predict(x_test)
  y_pred_RF20 = regressor_RF_20.predict(x_test)
  y_pred_MLP55 = regressor_MLP_55.predict(x_test)
  y_pred_MLP2020 = regressor_MLP_2020.predict(x_test)
  
  # Mean absolute error (MAE)
  mae_RF5 = mean_absolute_error(y_test, y_pred_RF5)
  mae_RF20 = mean_absolute_error(y_test, y_pred_RF20)
  mae_MLP55 = mean_absolute_error(y_test, y_pred_MLP55)
  mae_MLP2020 = mean_absolute_error(y_test, y_pred_MLP2020)
  
  # Mean squared error (MSE)
  mse_RF5 = mean_squared_error(y_test, y_pred_RF5)
  mse_RF20 = mean_squared_error(y_test, y_pred_RF20)
  mse_MLP55 = mean_squared_error(y_test, y_pred_MLP55)
  mse_MLP2020 = mean_squared_error(y_test, y_pred_MLP2020)
  
  # R2 Score (R2)
  r2_RF5 = r2_score(y_test, y_pred_RF5)
  r2_RF20 = r2_score(y_test, y_pred_RF20)
  r2_MLP55 = r2_score(y_test, y_pred_MLP55)
  r2_MLP2020 = r2_score(y_test, y_pred_MLP2020)
  
  # Mean absoulte percentage error (MAPE)
  mape_RF5 = mean_absolute_percentage_error(y_test, y_pred_RF5)
  mape_RF20 = mean_absolute_percentage_error(y_test, y_pred_RF20)
  mape_MLP55 = mean_absolute_percentage_error(y_test, y_pred_MLP55)
  mape_MLP2020 = mean_absolute_percentage_error(y_test, y_pred_MLP2020)
  
  # Saving MSE results to export to excel file
  result_array_mse_RF5 = np.append(result_array_mse_RF5, mse_RF5)
  result_array_mse_RF20 = np.append(result_array_mse_RF20, mse_RF20)
  result_array_mse_MLP55 = np.append(result_array_mse_MLP55, mse_MLP55)
  result_array_mse_MLP2020 = np.append(result_array_mse_MLP2020, mse_MLP2020)
  
  # Saving MAE results to export to excel file
  result_array_mae_RF5 = np.append(result_array_mae_RF5, mae_RF5)
  result_array_mae_RF20 = np.append(result_array_mae_RF20, mae_RF20)
  result_array_mae_MLP55 = np.append(result_array_mae_MLP55, mae_MLP55)
  result_array_mae_MLP2020 = np.append(result_array_mae_MLP2020, mae_MLP2020)
  
  # Saving MAPE results to export to excel file
  result_array_mape_RF5 = np.append(result_array_mape_RF5, mape_RF5)
  result_array_mape_RF20 = np.append(result_array_mape_RF20, mape_RF20)
  result_array_mape_MLP55 = np.append(result_array_mape_MLP55, mape_MLP55)
  result_array_mape_MLP2020 = np.append(result_array_mape_MLP2020, mape_MLP2020)
  
  # Saving R2 results to export to excel file
  result_array_r2_RF5 = np.append(result_array_r2_RF5, r2_RF5)
  result_array_r2_RF20 = np.append(result_array_r2_RF20, r2_RF20)
  result_array_r2_MLP55 = np.append(result_array_r2_MLP55, r2_MLP55)
  result_array_r2_MLP2020 = np.append(result_array_r2_MLP2020, r2_MLP2020)
  
  # Create a workook for each iteration and saves all the tests id different sheets
  with pd.ExcelWriter('./results/estimated/RF/RF_OutputSeed_{}.xlsx'.format(i)) as writer: # pylint: disable=abstract-class-instantiated
    pd.DataFrame(y_pred_RF5, columns=['Predicted']).to_excel(writer, sheet_name='RF5', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='RF5', startcol=1, index=False)
    
    pd.DataFrame(y_pred_RF20, columns=['Predicted']).to_excel(writer, sheet_name='RF20', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='RF20', startcol=1, index=False)
  
  # Create a workook for each iteration and saves all the tests id different sheets
  with pd.ExcelWriter('./results/estimated/MLP/MLP_OutputSeed_{}.xlsx'.format(i)) as writer: # pylint: disable=abstract-class-instantiated
    pd.DataFrame(y_pred_MLP55, columns=['Predicted']).to_excel(writer, sheet_name='MLP55', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='MLP55', startcol=1, index=False)
    
    pd.DataFrame(y_pred_MLP2020, columns=['Predicted']).to_excel(writer, sheet_name='MLP2020', index=False)
    pd.DataFrame(y_test, columns=['Measured']).to_excel(writer, sheet_name='MLP2020', startcol=1, index=False)

  # Saves the statistical results of RF5 to a .txt file    
  with open("./results/txt/RF/OutputStatisticalResults_RF5.txt", "a") as f:
    print("\n#Seed " + str(i) + " RESULTS#", file=f)
    print("\n*Results for  of  Samples RF5*", file=f)
    print("MSE SAMPLES RF5: %.2f" % mse_RF5, file=f) 
    print('R2 SAMPLES RF5: %.2f' % r2_RF5, file=f) 
    print("MAE SAMPLES RF5: %.2f" % mae_RF5, file=f) 
    print("MAPE SAMPLES RF5: %.2f" % mape_RF5, file=f) 

  # Saves the statistical results of RF20 to a .txt file    
  with open("./results/txt/RF/OutputStatisticalResults_RF20.txt", "a") as f:
    print("\n#Seed " + str(i) + " RESULTS#", file=f)
    print("\n*Results for  of  Samples RF20*", file=f)
    print("MSE SAMPLES RF20: %.2f" % mse_RF20, file=f) 
    print('R2 SAMPLES RF20: %.2f' % r2_RF20, file=f) 
    print("MAE SAMPLES RF20: %.2f" % mae_RF20, file=f) 
    print("MAPE SAMPLES RF20: %.2f" % mape_RF20, file=f) 

  # Saves the statistical results of MLP55 to a .txt file    
  with open("./results/txt/MLP/OutputStatisticalResults_MLP55.txt", "a") as f:
    print("\n#Seed " + str(i) + " RESULTS#", file=f)
    print("\n*Results for  of  Samples MLP55*", file=f)
    print("MSE SAMPLES MLP55: %.2f" % mse_MLP55, file=f) 
    print('R2 SAMPLES MLP55: %.2f' % r2_MLP55, file=f) 
    print("MAE SAMPLES MLP55: %.2f" % mae_MLP55, file=f) 
    print("MAPE SAMPLES MLP55: %.2f" % mape_MLP55, file=f) 

  # Saves the statistical results of MLP2020 to a .txt file    
  with open("./results/txt/MLP/OutputStatisticalResults_MLP2020.txt", "a") as f:
    print("\n#Seed " + str(i) + " RESULTS#", file=f)
    print("\n*Results for  of  Samples MLP2020*", file=f)
    print("MSE SAMPLES MLP2020: %.2f" % mse_MLP2020, file=f) 
    print('R2 SAMPLES MLP2020: %.2f' % r2_MLP2020, file=f) 
    print("MAE SAMPLES MLP2020: %.2f" % mae_MLP2020, file=f) 
    print("MAPE SAMPLES MLP2020: %.2f" % mape_MLP2020, file=f) 

# Generates Excel File saving the MSE results    
with pd.ExcelWriter('./results/statistics/MSE_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(result_array_mse_RF5, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(result_array_mse_RF20, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(result_array_mse_MLP55, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False) 
  pd.DataFrame(result_array_mse_MLP2020, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)   

# Generates Excel File saving the MAE results    
with pd.ExcelWriter('./results/statistics/MAE_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(result_array_mae_RF5, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(result_array_mae_RF20, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(result_array_mae_MLP55, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False) 
  pd.DataFrame(result_array_mae_MLP2020, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)   

# Generates Excel File saving the MAPE results    
with pd.ExcelWriter('./results/statistics/MAPE_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(result_array_mape_RF5, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(result_array_mape_RF20, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(result_array_mape_MLP55, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False)   
  pd.DataFrame(result_array_mape_MLP2020, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)   

# Generates Excel File saving the R2 results    
with pd.ExcelWriter('./results/statistics/R2_Output.xlsx') as writer:# pylint: disable=abstract-class-instantiated
  pd.DataFrame(result_array_r2_RF5, columns=['RF5']).to_excel(writer, sheet_name='Complete', index=False)
  pd.DataFrame(result_array_r2_RF20, columns=['RF20']).to_excel(writer, sheet_name='Complete', startcol=1, index=False)
  pd.DataFrame(result_array_r2_MLP55, columns=['MLP55']).to_excel(writer, sheet_name='Complete', startcol=2, index=False)   
  pd.DataFrame(result_array_r2_MLP2020, columns=['MLP2020']).to_excel(writer, sheet_name='Complete', startcol=3, index=False)
