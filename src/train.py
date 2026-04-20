
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from pathlib import Path
import joblib

def main():
    # load dataset
    print("Loading dataset...")
    diabetes = load_diabetes()
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    feature_names = diabetes.feature_names

    # split data
    print("Split dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, 
                                                        y, 
                                                        test_size=0.3, 
                                                        random_state=42
                                                        )

    # train model
    print("Initialising model...")
    model = LinearRegression()
    print("Training model...")
    model.fit(X_train, y_train)
    print("Predicting test...")
    y_pred = model.predict(X_test)
    
    # evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE Score: {mse}")

    # create output directory
    output_dir = Path("outputs/model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save model bundle with joblib
    model_bundle = {
        "model": model,
        "feature_names": feature_names,
    }
    joblib.dump(model_bundle, output_dir / "model.joblib")
    print("Model saved...")


if __name__ == "__main__":
    main()
