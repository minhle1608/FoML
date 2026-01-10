import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import joblib
import os

from src.preprocessing import data_convert
from src.NN_setup import NN

data_path = "data/Car details v3.csv"

torch.manual_seed(42)
np.random.seed(42)

def print_metrics(model):
    model.fit(X_train, np.log(y_train))

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_pred_train = np.exp(y_pred_train)
    y_pred_test = np.exp(y_pred_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(r2_train, r2_test)
    print(rmse_train, rmse_test)
    print(mae_train, mae_test)

def NN_call(X_train, X_test, y_train, y_test):
    X_train_encode = X_train.astype(float)
    X_test_encode = X_test.astype(float)
    X_train_tensor = torch.tensor(X_train_encode.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_encode.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(np.log(y_train).values, dtype=torch.float32).unsqueeze(1)
    Y_test_tensor = torch.tensor(np.log(y_test).values, dtype=torch.float32).unsqueeze(1)

    num_node = X_train_encode.shape[1]
    NN_model = NN(num_node)

    calc = nn.MSELoss()
    optimizer = optim.Adam(NN_model.parameters(), lr=0.01)
    epochs = 500

    for epoch in range(epochs): 
        optimizer.zero_grad()
        current = NN_model(X_train_tensor)
        loss = calc(current, Y_train_tensor)
        loss.backward()
        optimizer.step()

    NN_model.eval()
    with torch.no_grad():
        preds_train = NN_model(X_train_tensor).detach().numpy()
        preds_test = NN_model(X_test_tensor).detach().numpy()
        
        r2_train = r2_score(y_train, np.exp(preds_train))
        r2_test = r2_score(y_test, np.exp(preds_test))

        rmse_train = np.sqrt(mean_squared_error(y_train, np.exp(preds_train)))
        rmse_test = np.sqrt(mean_squared_error(y_test, np.exp(preds_test)))

        mae_train = mean_absolute_error(y_train, np.exp(preds_train))
        mae_test = mean_absolute_error(y_test, np.exp(preds_test))

        print(r2_train, r2_test)
        print(rmse_train, rmse_test)
        print(mae_train, mae_test)

def cross_validation(model, X, Y, scaler, col_scale):
    y_split = pd.qcut(Y, q=4, labels = False)
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    ls = []
    ls2 = []
    ls3 = []

    for train_i, test_i in skf.split(X, y_split):
        X_train_fold, X_test_fold = X.iloc[train_i].copy(), X.iloc[test_i].copy()
        y_train_fold, y_test_fold = Y.iloc[train_i].copy(), Y.iloc[test_i].copy()

        X_train_fold[col_scale] = scaler.fit_transform(X_train_fold[col_scale])
        X_test_fold[col_scale] = scaler.transform(X_test_fold[col_scale])

        model.fit(X_train_fold, np.log(y_train_fold))
        y_pred_train = model.predict(X_train_fold)
        y_pred_test = model.predict(X_test_fold)

        y_pred_train = np.exp(y_pred_train)
        y_pred_test = np.exp(y_pred_test)

        r2_train = r2_score(y_train_fold, y_pred_train)
        r2_test = r2_score(y_test_fold, y_pred_test)

        rmse_train = np.sqrt(mean_squared_error(y_train_fold, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test_fold, y_pred_test))

        mae_train = mean_absolute_error(y_train_fold, y_pred_train)
        mae_test = mean_absolute_error(y_test_fold, y_pred_test)
        
        ls.append([r2_train, r2_test])
        ls2.append([mae_train, mae_test])
        ls3.append([float(rmse_train), float(rmse_test)])

    print(ls)
    print(ls2)
    print(ls3)

def cross_validation_NN(X, Y):
    y_split = pd.qcut(Y, q=5, labels = False)
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)
    ls = []
    ls2 = []
    ls3 = []

    for train_i, test_i in skf.split(X, y_split):
        X_train_fold, X_test_fold = X.iloc[train_i].copy(), X.iloc[test_i].copy()
        y_train_fold, y_test_fold = Y.iloc[train_i].copy(), Y.iloc[test_i].copy()

        X_train_fold[col_scale] = scaler.fit_transform(X_train_fold[col_scale])
        X_test_fold[col_scale] = scaler.transform(X_test_fold[col_scale])

        X_train_fold_tensor = torch.tensor(X_train_fold.astype(float).values, dtype=torch.float32)
        X_test_fold_tensor = torch.tensor(X_test_fold.astype(float).values, dtype=torch.float32)
        y_train_fold_tensor = torch.tensor(np.log(y_train_fold).values, dtype=torch.float32).unsqueeze(1)
        y_test_fold_tensor = torch.tensor(np.log(y_test_fold).values, dtype=torch.float32).unsqueeze(1)

        input_shape = X_train_fold.shape[1]
        cur = NN(input_shape)

        optimizer = optim.Adam(cur.parameters(), lr=0.01)
        calc = nn.MSELoss()

        for epoch in range(500):
            optimizer.zero_grad()
            current = cur(X_train_fold_tensor)
            loss = calc(current, y_train_fold_tensor)
            loss.backward()
            optimizer.step()

        cur.eval()
        with torch.no_grad():
            preds_train = cur(X_train_fold_tensor).detach().numpy()
            preds_test = cur(X_test_fold_tensor).detach().numpy()

            r2_train = r2_score(y_train_fold, np.exp(preds_train))
            r2_test = r2_score(y_test_fold, np.exp(preds_test))

            rmse_train = np.sqrt(mean_squared_error(y_train_fold, np.exp(preds_train)))
            rmse_test = np.sqrt(mean_squared_error(y_test_fold, np.exp(preds_test)))

            mae_train = mean_absolute_error(y_train_fold, np.exp(preds_train))
            mae_test = mean_absolute_error(y_test_fold, np.exp(preds_test))

            ls.append([r2_train, r2_test])
            ls2.append([mae_train, mae_test])
            ls3.append([float(rmse_train), float(rmse_test)])

    print(ls)
    print(ls2)
    print(ls3)

if __name__ == "__main__":
    #cleaned
    df = pd.read_csv(data_path)
    df = data_convert(df)

    #do one-hot encoding
    df_one_encode = pd.get_dummies(df, columns = ['name'])

    brands_col = []
    for value in df_one_encode.columns:
        if "name_" in value:
            brands_col.append(value)

    #We identified feature like max_power, transmission, year, and seller type matters most
    #Now onto normalizing these data to make it from 0 to 1 (except year)

    from sklearn.preprocessing import RobustScaler
    feature_chosen = ['year', 'seller_type', 'transmission', 'max_power', 'engine', 'owner', 'fuel'] + brands_col

    X = df_one_encode[feature_chosen].copy()
    Y = df_one_encode['selling_price'].copy()
    original_X = X.copy()
    original_Y = Y.copy()
    col_scale = ['max_power', 'engine']
    scaler = RobustScaler()
    #normalize max_power and engine using robust scaler

    X['year'] = 2020 - X['year']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train[col_scale] = scaler.fit_transform(X_train[col_scale])
    X_test[col_scale] = scaler.transform(X_test[col_scale])

    #Linear Regression
    model = LinearRegression()
    print("metrics for linear regression, with RMSE, MSE, and MAE for train_and_test")
    print_metrics(model)
    cross_validation(model, X, Y, scaler, col_scale)
    print()

    #Random Forest
    mrf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("metrics for random forest, with RMSE, MSE, and MAE for train_and_test")
    print_metrics(mrf_model)
    cross_validation(mrf_model, X, Y, scaler, col_scale)
    print()

    print("metrics for Neural Network, with RMSE, MSE, and MAE for train_and_test")
    NN_call(X_train, X_test, y_train, y_test)
    cross_validation_NN(X,Y)
    print("Random Forest performs best")

    model_with_data = {
        "model": mrf_model,
        "scaler": scaler,
        "features": X_train.columns,
        "scale_cols": col_scale
    }
    
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_extend.joblib")
    joblib.dump(model_with_data, model_path)
    print("(Random Forest) Model exported")
