import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

class BitcoinDataAnalysis:
    """
    A module for Predicting Bitcoin_Volatility_Magnitude data including loading, preprocessing, visualization, 
    statistical analysis, and machine learning modeling.
    """
    
    def __init__(self, file_path):
        """Initialize with a file path to load Bitcoin data."""
        self.df = self.load_data(file_path)
    
    def load_data(self, file_path):
        """Loads Bitcoin data from a CSV file."""
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            return df
        except FileNotFoundError:
            print("Error: File not found.")
            return None
        except pd.errors.ParserError:
            print("Error: Could not parse the CSV file. Please check the format.")
            return None
    
    def data_overview(self):
        """Displays basic statistics and missing values in the dataset."""
        if self.df is not None:
            print("Data Shape:", self.df.shape)
            print("\nData Types:\n", self.df.dtypes)
            print("\nMissing Values:\n", self.df.isnull().sum())
            print("\nDescriptive Statistics:\n", self.df.describe())
    
    def plot_closing_price(self):
        """Plots the Bitcoin closing price over time."""
        if self.df is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(self.df.index, self.df['Close'], label='Closing Price', color='blue')
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.title('Bitcoin Closing Price Over Time')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def compute_correlation_matrix(self):
        """Computes and visualizes the correlation matrix."""
        if self.df is not None:
            correlation_matrix = self.df.corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Bitcoin Features')
            plt.show()
    
    def compute_volatility(self):
        """Computes Bitcoin volatility using a 7-day window and handles outliers."""
        if self.df is not None:
            self.df['Volatility'] = (self.df['Close'].pct_change(periods=7).abs()) * 100
            self.df['Volatility'] = winsorize(self.df['Volatility'], limits=[0.01, 0.01])
    
    def test_stationarity(self):
        """Performs the Augmented Dickey-Fuller test for stationarity on volatility."""
        if self.df is not None and 'Volatility' in self.df:
            result = adfuller(self.df['Volatility'].dropna())
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        """Splits the dataset into train, validation, and test sets chronologically."""
        if self.df is not None:
            total_size = len(self.df)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            
            self.df_train = self.df.iloc[:train_size]
            self.df_val = self.df.iloc[train_size:train_size + val_size]
            self.df_test = self.df.iloc[train_size + val_size:]
            
            print(f"Training Set: {self.df_train.shape}")
            print(f"Validation Set: {self.df_val.shape}")
            print(f"Test Set: {self.df_test.shape}")
    
    def train_xgboost_model(self):
        """Trains an XGBoost regression model to predict Bitcoin volatility."""
        if self.df is not None and 'Volatility' in self.df:
            X_train = self.df_train.drop('Volatility', axis=1)
            y_train = self.df_train['Volatility']
            X_val = self.df_val.drop('Volatility', axis=1)
            y_val = self.df_val['Volatility']
            
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model.fit(X_train.fillna(0), y_train)
            
            y_pred_val = model.predict(X_val.fillna(0))
            mae = mean_absolute_error(y_val, y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            
            print(f"Mean Absolute Error (MAE): {mae}")
            print(f"Root Mean Squared Error (RMSE): {rmse}")
            
            return model
    
    def plot_actual_vs_predicted(self, model):
        """Plots actual vs. predicted Bitcoin volatility using the trained model."""
        if self.df is not None:
            X_test = self.df_test.drop('Volatility', axis=1).fillna(0)
            y_test = self.df_test['Volatility']
            y_pred_test = model.predict(X_test)
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.index, y_test, label='Actual Volatility', color='blue')
            plt.plot(y_test.index, y_pred_test, label='Predicted Volatility', color='red')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.title('Actual vs. Predicted Bitcoin Volatility')
            plt.legend()
            plt.grid(True)
            plt.show()

# Example Usage
# analysis = BitcoinDataAnalysis("Bitcoin Data March 13.csv")
# analysis.data_overview()
# analysis.plot_closing_price()
# analysis.compute_volatility()
# analysis.test_stationarity()
# analysis.split_data()
# model = analysis.train_xgboost_model()
# analysis.plot_actual_vs_predicted(model)

'''Results Obtained with Prototype Algorithm:

Mean Absolute Error (MAE): 3.9567609171334786
Root Mean Squared Error (RMSE): 5.37721743971324
Accuracy: 0.7378472222222222
Precision: 0.6286644951140065
Recall: 0.8391304347826087
AUC: 0.7548253329982408
Confusion Matrix:
[[232 114]
 [ 37 193]]
Volatility Threshold used for classification metrics: 5.000030557612966

 '''