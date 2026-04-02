from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,make_scorer,f1_score,confusion_matrix

def train_random_forest(X_train,y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs = -1
    )

    param_grid = {
    "n_estimators" : [100,200,300],
    "max_depth" : [None,4,5,6],
    "min_samples_split" : [2,3,5],
    "min_samples_leaf" : [1,2,5],
    "criterion" : ['gini','entropy']
    }

    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(
    estimator = rf,
    param_grid = param_grid,
    cv = 5,
    scoring = scorer,
    verbose = 2,
    n_jobs = -1
    )

    grid_search.fit(X_train,y_train)
    return grid_search

def evaluate_model(model,X_test_scaled,y_test,model_name:str):
    preds = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test,preds)
    class_report = classification_report(y_test,preds)
    cm=confusion_matrix(model.predict(X_test_scaled),y_test)

    
    print(f"\n{model_name} Performace")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n {class_report}")
    print(f"\nConfusion matrix: {cm}")