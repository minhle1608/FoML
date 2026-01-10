import pandas as pd

def transform_name(str):
    return str.split()[0]

def data_convert(df):
    df = df.drop_duplicates()
    #convert any text string into raw number for easier calculation
    df = df.drop(['torque'], axis = 1)
    df = df.copy()

    df['mileage'] = df['mileage'].str.replace(" kmpl", "")
    df['mileage'] = df['mileage'].str.replace(" km/kg", "")
    df['max_power'] = df['max_power'].str.replace(" bhp", "")
    df['engine'] = df['engine'].str.replace(" CC", "")
    df['mileage'] = pd.to_numeric(df['mileage'])
    df['max_power'] = pd.to_numeric(df['max_power'])

    df['engine'] = pd.to_numeric(df['engine'])
    #do this before splitting X and y

    df['seller_type'] = df['seller_type'].map({"Individual": 0, "Trustmark Dealer": 1, "Dealer": 2})
    df['transmission'] = df['transmission'].map({"Manual": 0, "Automatic": 1})
    df['fuel'] = df['fuel'].map({"CNG": 0, "LPG": 1, "Petrol": 2, "Diesel": 3})
    df['owner'] = df['owner'].map({"Fourth & Above Owner": 0, "Third Owner": 1, "Second Owner": 2, "First Owner": 3, "Test Drive Car": 4})

    df['name']=df['name'].apply(transform_name)

    df = df.dropna()

    return df