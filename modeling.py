from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_prepare import data_download, data_preprocessing, reduce_dimensions
from knn import MyKNN as KNN
from sklearn.model_selection import GridSearchCV

# Define the models
models = {
    "KNN": KNN(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}
mushroom_data, mushroom_label = data_download()
X_train, X_test, y_train, y_test = data_preprocessing(mushroom_data, mushroom_label)
X_train_pca, X_test_pca = reduce_dimensions(X_train, X_test)
# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_pca, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_pca)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print the evaluation results
    print(f"{model_name} Model:")
    print(f"Accuracy: {accuracy:.2f}")
    # print(f"Classification Report:\n{report}\n")



# Define the hyperparameter grids
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

def grid_serach():
    # Create the models
    svm = SVC()
    rf = RandomForestClassifier()

    # Create the GridSearchCV objects
    svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, verbose=2, n_jobs=-1)
    rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, verbose=2, n_jobs=-1)

    # Perform the Grid Search for SVM
    svm_grid_search.fit(X_train_pca, y_train)
    svm_best_params = svm_grid_search.best_params_
    svm_best_score = svm_grid_search.best_score_

    # Perform the Grid Search for Random Forest
    rf_grid_search.fit(X_train_pca, y_train)
    rf_best_params = rf_grid_search.best_params_
    rf_best_score = rf_grid_search.best_score_

    # Display the results
    print(f"SVM Best Parameters: {svm_best_params} - Best Score: {svm_best_score:.2f}")
    print(f"Random Forest Best Parameters: {rf_best_params} - Best Score: {rf_best_score:.2f}")


def feature_importance( X_train, y_train):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # Assuming your data is in X_train and y_train
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Create a DataFrame for the feature importances
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })

    # Sort the features based on importance
    features_df = features_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(features_df['Feature'], features_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have the feature with the highest importance at the top
    plt.show()

feature_importance(X_train, y_train)