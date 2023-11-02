from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# fetch dataset 
def data_download():
    mushroom = fetch_ucirepo(id=73) 
    
    # data (as pandas dataframes) 
    mushroom_data = mushroom.data.features 
    mushroom_label = mushroom.data.targets 
    return mushroom_data, mushroom_label




def data_preprocessing(mushroom_data, mushroom_label):
    # Handling Missing Values
    # Assuming "?" denotes a missing value, replace it with a NaN for easier handling
    mushroom_data.replace("?", float("nan"), inplace=True)

    # Impute missing values with the most frequent value in each column
    imputer = SimpleImputer(strategy="most_frequent")
    mushroom_data_imputed = pd.DataFrame(imputer.fit_transform(mushroom_data), columns=mushroom_data.columns)
    
    # Encoding Categorical Variables
    # Since all the variables are categorical, we'll use Label Encoding
    encoder = LabelEncoder()
    mushroom_data_encoded = mushroom_data_imputed.apply(encoder.fit_transform)
    mushroom_label_encoded = encoder.fit_transform(mushroom_label)
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        mushroom_data_encoded, mushroom_label_encoded, test_size=0.3, random_state=445, stratify=mushroom_label_encoded, shuffle=True
    )
    
    # Standardization
    # Standardizing the data so that it follows a normal distribution
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
   
    # List of most important features to drop
    # Dropping least important features
    least_important_features = ['veil-type', 'gill-attachment', 'veil-color']
    X_train_scaled = X_train.drop(columns=least_important_features)
    X_test_scaled = X_test.drop(columns=least_important_features)

    return X_train_scaled, X_test_scaled, y_train, y_test

def reduce_dimensions(X_train, X_test):
    # Initialize PCA with 2 components
    pca = PCA(n_components=2)

    # Fit PCA to the training data and apply the transformation to both the training and testing data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca

if __name__ == "__main__":
    # Assuming you have downloaded the data using the data_download function
    mushroom_data, mushroom_label = data_download()
    X_train, X_test, y_train, y_test = data_preprocessing(mushroom_data, mushroom_label)
    print(f'Head of training set:\n{X_train.head()}')
    print(f'Head of test set:\n{X_test.head()}')
    print(f'Any NaN values in training set: {X_train.isnull().values.any()}')
    print(f'Any NaN values in test set: {X_test.isnull().values.any()}')
    print(f'Label distribution in training set:\n{pd.Series(y_train).value_counts(normalize=True)}')
    print(f'Label distribution in test set:\n{pd.Series(y_test).value_counts(normalize=True)}')

    print(y_test)