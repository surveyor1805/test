import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def prepare_data():
    train = pd.read_csv("data/realty_data.csv")
    train = train.drop(columns = ['description', 'period', 'product_name', 'settlement', 'area', 'object_type', 'city',
            'district', 'address_name', 'source', 'lat', 'lon', 'postcode'])
    train = train.dropna()
    train = remove_outliers(train)
    train['price'] = train['price'].astype('int32')
    train['total_square'] = train['total_square'].astype('int16')
    train['rooms'] = train['rooms'].astype('int8')
    train['floor'] = train['floor'].astype('int8')
    train = train.reset_index(drop=True)
    return train

def train_model(train):
    x = train.drop('price', axis=1)
    y = train['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print("Mean Squared Error of our model:", round(mse))
    print("Mean Absolute Error of our model:", round(mae))
    with open('lr.pkl', 'wb') as file:
        pickle.dump(model, file)

def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("No such model")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model

def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[((df >= lower_bound) & (df <= upper_bound)).all(axis=1)]
    return df_filtered
