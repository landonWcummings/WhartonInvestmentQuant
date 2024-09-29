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
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import cupy as cp


score = -1


class XGB:
    def __init__(self,
                X_train, y_train, X_test,y_test,savepath,groups,optunadepth=20,gpu=False,params= []):
        
        self.optunadepth = optunadepth
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.groups = groups
        self.savepath = savepath
        self.params = params
        self.gpu = gpu

        
        
    def makemodel(self):
        
        print(self.y_train.isna().sum())
        print(np.isinf(self.y_train).sum())
        print("Starting XGB")
        #uses optuna to find hyperparameters if no params are passed in on init
        if not self.params:#optuna

            def objective(trial):
                # Define the search space for hyperparameters
                if self.gpu:
                    params = {
                        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.1, log=True),  
                        'max_depth': trial.suggest_int('max_depth', 4, 20),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),  
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True), 
                        'device':'cuda',
                        'max_bin': 256
                    }
                else:
                    params = {
                        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.1, log=True),  
                        'max_depth': trial.suggest_int('max_depth', 4, 20),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),  
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True), 
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500)
                    }

                #model = XGBRegressor(**params)
                
                #this minimizes randomness by training and testing multiple times on random sets
                tscv = TimeSeriesSplit(n_splits=4)
                scores = []

                for train_index, test_index in tscv.split(self.X_train):
                    X_train_fold, X_test_fold = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                    y_train_fold, y_test_fold = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
                    print(f"Train fold size: {X_train_fold.shape[0]} samples")
                    print(f"Test fold size: {X_test_fold.shape[0]} samples")

                    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
                    dtest = xgb.DMatrix(X_test_fold, label=y_test_fold)

                    # Train using xgb.train()
                    model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])

                    bpredictions = model.predict(dtest)
                    
                    mae = mean_absolute_error(y_test_fold, bpredictions)
                    scores.append(mae)

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
            'tag' : self.groups,
            'Actual': self.y_test,
            'Predicted': predictions
        })

        results['Direction predicted'] = results['Predicted'] >=0
        results['basePL'] = results['Actual'] 

        results = results.round(5)

        results['actual direction'] = results['Actual'] >= 0
        results['is direction correct'] = (results['actual direction'] == results['Direction predicted'])

        direction_counts = results['is direction correct'].value_counts()

        results['PL'] = np.where(results['is direction correct'], results['basePL'].abs(), -1 * results['basePL'].abs())
        
        print(direction_counts)

        print(f'Biggest P: {results["PL"].max()}')
        print(f'Biggest L: {results["PL"].min()}')

        PL_base = (results['basePL']).sum()
        PL_sum = results["PL"].sum()
        print(f'Base Profit/Loss : {PL_base}')
        print(f'Profit/Loss: {PL_sum}')
        print(f'Market differential: {PL_sum - PL_base}')


        results['Conviction'] = (results['Predicted'] < -0.1) | (results['Predicted'] > 0.3)
        results['HCPL'] = np.where(
            results['Conviction'],  # If 'high conviction' is True
            np.where(results['is direction correct'], results['basePL'].abs(), -1 * results['basePL'].abs()),  # Apply logic when True
            0 
        )
        print(f"High conviction PL: {results['HCPL'].sum()}")

        """#confusion matrix vis
        from sklearn.metrics import confusion_matrix

        # Map boolean values to integers (True -> 1, False -> 0)
        def get_confusion_label(row):
            if row['Direction predicted'] and row['actual direction']:
                return 'True Positive'  # Predicted Up, Actual Up
            elif row['Direction predicted'] and not row['actual direction']:
                return 'False Positive'  # Predicted Up, Actual Down
            elif not row['Direction predicted'] and not row['actual direction']:
                return 'True Negative'  # Predicted Down, Actual Down
            else:
                return 'False Negative'  # Predicted Down, Actual Up

        results['Confusion Label'] = results.apply(get_confusion_label, axis=1)

        # Map prediction direction to readable strings
        results['Prediction Direction'] = results['Direction predicted'].map({True: 'Predicted Up', False: 'Predicted Down'})

        # Calculate average PL per prediction direction
        avg_PL_by_prediction = results.groupby('Prediction Direction')['PL'].mean().reset_index()

        # Calculate average PL per confusion label
        avg_PL_by_confusion_label = results.groupby('Confusion Label')['PL'].mean().reset_index()

        # Visualize average PL per prediction direction
        plt.figure(figsize=(8,6))
        sns.barplot(x='Prediction Direction', y='PL', data=avg_PL_by_prediction, palette='viridis')
        plt.title('Average Profit/Loss per Trade by Prediction Direction')
        plt.xlabel('Prediction Direction')
        plt.ylabel('Average Profit/Loss ($)')
        plt.show()

        # Visualize average PL per confusion label
        confusion_order = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
        plt.figure(figsize=(10,6))
        sns.barplot(x='Confusion Label', y='PL', data=avg_PL_by_confusion_label, order=confusion_order, palette='coolwarm')
        plt.title('Average Profit/Loss per Trade by Confusion Matrix Category')
        plt.xlabel('Confusion Matrix Category')
        plt.ylabel('Average Profit/Loss ($)')
        plt.show()
        
        """

        """actual price change code
        results['Close_t'] = results.groupby('tag')['Actual'].shift(1)

        results = results.dropna(subset=['Close_t'])

        results['Price Change'] = results['Actual'] - results['Close_t']

        results['Predicted Change'] = results['Predicted'] - results['Close_t']
        results['Predicted Direction'] =  np.sign(results['Predicted Change'])
    

        results['Direction Correct'] = np.sign(results['Predicted Change']) == np.sign(results['Price Change'])
        #for the PL calculator - I just assume the bot either buys or sells the stock at the end of each trading day and holds it 1 trading day
        #this is using $100 dollars each day (so basically turns 100 into whatever final PL is)
        results['PL'] = np.where(results['Direction Correct'], results['Price Change'].abs(), -1 * results['Price Change'].abs())
        results['PL'] = (results['PL'] /self.X_test['Close']) * 100
        
        results = results.round(4)
        print(f'Percent predicted increase: {(results["Predicted Direction"].sum() / results.shape[0]).round(6) * 100}')
        print(f'Number of negative predictions: {(results['Predicted Direction'] == -1).sum()}')
        print(f'Max PL: {results["PL"].max()}')
        print(f'Min PL: {results["PL"].min()}')

        PL_base = ((results["Price Change"] / self.X_test['Close']) * 100).sum()
        PL_sum = results["PL"].sum()
        print(f'Base Profit/Loss : {PL_base}')
        print(f'Profit/Loss: {PL_sum}')
        print(f'Market differential: {PL_sum - PL_base}')
        #partial_sum = results['PL'][:10].sum()  
        #print(f'Partial PL sum (first 100 rows): {partial_sum}')



        direction_counts = results['Direction Correct'].value_counts()
        print(direction_counts)
        """
        X_test_df = pd.DataFrame(self.X_test, columns=self.X_test.columns)

        
        combined_df = X_test_df.copy()
        combined_df[results.columns] = results
        
        combined_df.to_csv(self.savepath, index=False)
        
        #xgb.plot_importance(xgb_model)
        #plt.show()
        return xgb_model, PL_sum-PL_base


    def modscore(self,datalocation,model,iterations=0):

        print("scoring")
        data = pd.read_csv(datalocation)
        print(data.shape)
        x = data.drop(["targ"],axis=1)
        y = data['targ']
        predictions = model.predict(x)
        results = pd.DataFrame({
            'Actual': y,
            'Predicted': predictions
        })

        results['Direction predicted'] = results['Predicted'] >=0
        results['basePL'] = results['Actual'] 

        results = results.round(5)

        results['actual direction'] = results['Actual'] >= 0
        results['is direction correct'] = (results['actual direction'] == results['Direction predicted'])

        direction_counts = results['is direction correct'].value_counts()

        results['PL'] = np.where(results['is direction correct'], results['basePL'].abs(), -1 * results['basePL'].abs())
        
        print(direction_counts)

        print(f'Biggest P: {results["PL"].max()}')
        print(f'Biggest L: {results["PL"].min()}')

        PL_base = (results['basePL']).sum()
        PL_sum = results["PL"].sum()
        print(f'Base Profit/Loss : {PL_base}')
        print(f'Profit/Loss: {PL_sum}')
        print(f'Market differential: {PL_sum - PL_base}')
        scores = []
        #neglimit = [-1.2,-1.1,-1,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65-0.6]
        #poslimit = [0,0.005,0.01,0.015,0.02,0.03]
        # best -0.8 , 0
        neg, pos = -0.8,0

        results['Conviction'] = (results['Predicted'] < neg) | (results['Predicted'] > pos)
        results['HCPL'] = np.where(
            results['Conviction'],  # If 'high conviction' is True
            np.where(results['is direction correct'], results['basePL'].abs(), -1 * results['basePL'].abs()),  # Apply logic when True
            0 
        )
        hpicks = results['Conviction'].sum()
        hscore = (results['HCPL'].sum())
        print(f"High conviction returns vs baseline {hscore}")


        results['notHCPL'] = np.where(
            ~results['Conviction'],  
            np.where(results['is direction correct'], results['basePL'].abs(), -1 * results['basePL'].abs()),  # Apply logic when True
            0 
        )
        lpicks = (~results['Conviction']).sum()
        lscore = (results['notHCPL'].sum() )
        print(f"Low conviction returns vs baseline {-lscore}")

        """
        for neg in neglimit:
            for pos in poslimit:
                results['Conviction'] = (results['Predicted'] < neg) | (results['Predicted'] > pos)
                results['HCPL'] = np.where(
                    results['Conviction'],  # If 'high conviction' is True
                    np.where(results['is direction correct'], results['basePL'].abs(), -1 * results['basePL'].abs()),  # Apply logic when True
                    0 
                )
                picks = results['Conviction'].sum()
                score = (results['HCPL'].sum() - picks/PL_base)
                found = False
                for i, (existing_neg, existing_pos, existing_score) in enumerate(scores):
                    if existing_neg == neg and existing_pos == pos:
                        # If found, update the score by adding the new score to the existing one
                        scores[i] = (existing_neg, existing_pos, existing_score + score)
                        found = True
                        break
                
                # If the (neg, pos) pair does not exist, append a new entry
                if not found:
                    scores.append((neg, pos, score))

        scores_df = pd.DataFrame(scores, columns=['Negative Limit', 'Positive Limit', 'Score'])
        plt.figure(figsize=(10, 6))
        plt.scatter(scores_df['Negative Limit'], scores_df['Positive Limit'], c=scores_df['Score'], cmap='coolwarm', s=100)
        plt.colorbar(label='Score')

        # Add labels for each point
        for i, row in scores_df.iterrows():
            plt.text(row['Negative Limit'], row['Positive Limit'], f'{row["Score"]:.1f}', fontsize=9, ha='right')

        # Set plot labels and title
        plt.xlabel('Negative Limit')
        plt.ylabel('Positive Limit')
        plt.title('Scores based on Negative and Positive Limit Combinations')
        plt.grid(True)

        # Show the plot
        plt.show()
        """

        hshort = (results['Predicted'] < -0.8).sum()
        hlong = (results['Predicted'] > 1).sum()
        print(f"Number of huge shorts: {hshort} longs: {hlong}")

        """
        from sklearn.metrics import confusion_matrix
        # Map boolean values to integers (True -> 1, False -> 0)
        def get_confusion_label(row):
            if row['Conviction'] and row['PL']>0:
                return 'True Positive'  
            elif not row['Conviction'] and row['PL']>0:
                return 'False Positive'  
            elif row['Conviction'] and not row['PL']>0:
                return 'True Negative'  
            else:
                return 'False Negative'  

        results['Confusion Label'] = results.apply(get_confusion_label, axis=1)

        # Calculate the average PL for each confusion label
        avg_PL_by_confusion_label = results.groupby('Confusion Label')['PL'].mean().reset_index()

        # Visualize the average PL per confusion label
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Confusion Label', y='PL', data=avg_PL_by_confusion_label, palette='coolwarm')
        plt.title('Average Profit/Loss per Trade by Confusion Label')
        plt.xlabel('Confusion Label')
        plt.ylabel('Average Profit/Loss ($)')
        plt.grid(True)
        plt.show()
        """

        print(f"---High conviction return differential {(hscore - lscore) - PL_base}")

        return hscore - PL_base

        #X_test_df = pd.DataFrame(x, columns=x.columns)

        
        #combined_df = X_test_df.copy()
        #combined_df[results.columns] = results
        
        #combined_df.to_csv(self.savepath, index=False)
        #return PL_sum - PL_base

    def predict(self):
        print("predicting on dataset")
        #to do


