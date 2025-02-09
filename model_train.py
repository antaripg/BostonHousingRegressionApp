import joblib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
from typing import Tuple, List
import os
import pytz
import logging
import argparse

# Paths for log file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
ist = pytz.timezone("Asia/Kolkata")
log_filename = datetime.now(ist).strftime("%Y%m%d_%H%M%S") + "_training.log"
log_filepath = os.path.join(log_dir, log_filename)

# Get Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File Handler (logs to file)
file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Stream Handler (logs to console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Start Logging
logging.info("Logging Initialized.")


# Argument Parser for Command-line execution
parser = argparse.ArgumentParser(description="Run the training pipeline with custom parameters.")
parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the Random Forest")
parser.add_argument("--split_size", type=float, default=0.2, help="Train-Test split size")
args = parser.parse_args()

class TrainingPipeline:

    def __init__(self, random_state: int=42, n_estimators: int=100, split_size: float=0.2):
        logging.info("Initializing Training Pipeline....")
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.split_size = split_size
        self.X, self.y, self.feature_names = self.get_data()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        logging.info("Training Pipeline Initialized...")
        
    def get_data(self)-> Tuple[np.ndarray, np.ndarray, List]:
        logging.info("Loading Dataset......")
        california = datasets.fetch_california_housing()
        feature_names = california.feature_names
        X = california.data
        y = california.target
        logging.info("Dataset Loaded Successfully.")
        return X, y, feature_names
    
    def run(self):
        logging.info("Starting the model training pipeline...")
        self.split_data()
        self.preprocess_data()
        self.model_train()
        self.model_predict()
        self.model_evaluate() 
        self.save_model()
        logging.info("Model training pipeline completed.")

    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Applying train test split to the data....")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.split_size,
                                                                                random_state=self.random_state)
        logger.info("Data splitting successful.")


    def preprocess_data(self):
        logger.info("Applying StandardScaler to features (X_train, X_test)...")
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        logger.info("Feature scaling completed successfully.")

    def model_train(self):
        logging.info("Training the Random Forest Model......")
        self.model.fit(self.X_train, self.y_train)
        logging.info("Model Training Completed")

    def model_predict(self):
        logging.info("Starting Model Prediction on test dataset.....")
        self.y_pred = self.model.predict(self.X_test)
        logging.info("Model Prediction Completed")

    def model_evaluate(self):
        logging.info("Model Evaluation Started...")
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        logging.info(f"Model Evaluation Completed. MSE: {mse: .4f}, RMSE: {rmse: .4f}")

    def save_model(self):
        logging.info("Saving the trained model...")
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"RF_california_{timestamp}.pkl"
        filepath = os.path.join(model_dir, filename)
        joblib.dump(self.model, filepath)
        logging.info(f"Model saved successfully as {filepath}")


if __name__ == "__main__":
    logger.info(f"Starting Training Pipeline with random_state={args.random_state}, "
                f"n_estimators={args.n_estimators}, split_size={args.split_size}")
    pipeline = TrainingPipeline(random_state=args.random_state, 
                                n_estimators=args.n_estimators, 
                                split_size=args.split_size)
    pipeline.run()
    logger.info("Training completed successfully.")