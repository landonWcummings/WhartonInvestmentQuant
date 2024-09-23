import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna 
import tqdm 
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error



score = -1


class XGB:
    def __init__(self,
                X_train, y_train, X_test,y_test,savepath,optunadepth=20,params= []):
        
        self.optunadepth = optunadepth
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.savepath = savepath
        self.params = params

        
        
    def makemodel(self):
        print("Starting XGB")
        #uses optuna to find hyperparameters if no params are passed in on init
        if not self.params:

            def objective(trial):
                # Define the search space for hyperparameters
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),  
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),  
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),  
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True), 
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'enable_categorical': False
                }
                model = XGBRegressor(**params)
                
                #this minimizes randomness by training and testing multiple times on random sets
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []

                for train_index, test_index in tscv.split(self.X_train):
                    X_train_fold, X_test_fold = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                    y_train_fold, y_test_fold = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

                    model.fit(X_train_fold, y_train_fold)
                    bpredictions = model.predict(X_test_fold)

                    mse = mean_squared_error(y_test_fold, bpredictions)
                    scores.append(mse)

                score = np.mean(scores)
                
                return score

            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
            #looks to minimize mean squared error
            study = optuna.create_study(direction='minimize')


            with tqdm.tqdm(total=self.optunadepth, desc="Hyperparameter Optimization") as pbar:
                def tqdm_callback(study, trial):
                    pbar.update(1)
                study.optimize(objective, n_trials=self.optunadepth, callbacks=[tqdm_callback])

            best_params = study.best_params
            print(f"Best parameters: {best_params}")
        else:
            best_params = self.params

        xgb_model = xgb.XGBRegressor(**best_params)
        xgb_model.fit(self.X_train,self.y_train)

        predictions = xgb_model.predict(self.X_test)

        #cool visualization and P/L thing
        results = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': predictions
        })


        results['Close_t'] = results['Actual'].shift(1)
        results = results.dropna() 

        results['Price Change'] = results['Actual'] - results['Close_t']

        results['Predicted Change'] = results['Predicted'] - results['Close_t']

        results['Direction Correct'] = np.sign(results['Predicted Change']) == np.sign(results['Price Change'])
        #for the PL calculator - I just assume the bot either buys or sells the stock at the end of each trading day and holds it 1 trading day
        #this is using $100 dollars each day (so basically turns 100 into whatever final PL is)
        results['PL'] = np.where(results['Direction Correct'], results['Price Change'].abs(), -1 * results['Price Change'].abs())
        results['PL'] = (results['PL'] /self.X_test['Close']) * 100
        

        print(f'Max PL: {results["PL"].max()}')
        print(f'Min PL: {results["PL"].min()}')

        PL_base = results["Price Change"].sum()
        PL_sum = results["PL"].sum()
        print(f'Base Profit/Loss : {PL_base}')
        print(f'Profit/Loss: {PL_sum}')
        print(f'Market differential: {PL_sum - PL_base}')
        partial_sum = results['PL'][:10].sum()  
        print(f'Partial PL sum (first 100 rows): {partial_sum}')



        direction_counts = results['Direction Correct'].value_counts()
        print(direction_counts)

        X_test_df = pd.DataFrame(self.X_test, columns=self.X_test.columns)

        
        combined_df = X_test_df.copy()
        combined_df[results.columns] = results
        expel = combined_df[combined_df[f'Close_t'].isna()]
        combined_df = combined_df[~combined_df.index.isin(expel.index)]
        
        combined_df.to_csv(self.savepath, index=False)

        xgb.plot_importance(xgb_model)
        plt.show()
        return xgb_model





    def predict(self):
        print("predicting on dataset")
        #to do


