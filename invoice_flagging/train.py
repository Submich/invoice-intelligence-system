from model_evaluation import train_random_forest,evaluate_model
from data_preprocessing import (
                    load_invoice_data,
                    create_invoice_risk_label,
                    apply_labels,
                    split_data,scale_features)
import joblib

features = ['invoice_quantity','invoice_dollars','Freight','total_item_quantity','total_item_dollars']
target = 'flag_invoice'

def main():

    df = load_invoice_data()
    df = apply_labels(df)

    X_train,X_test,y_train,y_test = split_data(df,features,target)
    X_train_scaled,X_test_scaled = scale_features(X_train,X_test,'models/scaler.pkl')

    grid_search = train_random_forest(X_train_scaled,y_train)

    evaluate_model(grid_search.best_estimator_,X_test_scaled,y_test,'Random Forest Classifier')

    joblib.dump(grid_search.best_estimator_,'models/predict_flag_invoice.pkl')

if __name__ == "__main__":
    main()