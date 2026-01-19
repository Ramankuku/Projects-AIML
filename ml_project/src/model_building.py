from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
import mlflow
import mlflow.sklearn
from config.utils import TARGET_COLUMN, MODEL_DIR
import os
import joblib



class ModelBuilding:
    def __init__(self):
        self.target = TARGET_COLUMN

    def model_build(self, train_data, test_data):
        try:
            X_train = train_data.drop(columns=[self.target])
            y_train = train_data[self.target]

            X_test = test_data.drop(columns=[self.target])
            y_test = test_data[self.target]

            models = {
                "DecisionTree": DecisionTreeClassifier(criterion="entropy",max_depth=None,max_features="sqrt",min_samples_split=2),
                "RandomForest": RandomForestClassifier(max_depth=15, max_features="log2", min_samples_split=2,n_estimators=200,random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,min_samples_split=2),
                    n_estimators=300, learning_rate=0.05, random_state=42)
            }
            os.makedirs(MODEL_DIR, exist_ok=True)


            mlflow.set_experiment("Classifier Models")

            best_score = 0
            best_model = None

            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)

                    #Metrics 
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("precision", prec)
                    mlflow.log_metric("recall", rec)

                    mlflow.log_params(model.get_params())

                    # Model Saving
                    mlflow.sklearn.log_model(sk_model=model, artifact_path=model_name)
                    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))

                    print(f"{model_name} ---> Accuracy: {acc:.2f}")

                    if acc > best_score:
                        best_score = acc
                        best_model = model

            print("\nBest Model:", best_model.__class__.__name__)
            print("Best Accuracy:", best_score)

            return best_model

        except Exception as e:
            print("Error while building the model:", e)

