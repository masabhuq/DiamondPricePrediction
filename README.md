# Diamond Price Prediction
[Link to the dataset](https://www.kaggle.com/datasets/ronil8/diamond-price-prediction-dataset)
This project involves building a machine learning model to predict diamond prices based on various features. The model is deployed as a Flask API, allowing users to input diamond characteristics and receive price predictions.

## Project Structure

The repository is organized as follows:

- `notebooks/`: Contains Jupyter notebooks used for data exploration and initial model development.
- `src/`: Includes the source code for the project, structured into:
  - `components/`:
    - `data_ingestion.py`: Handles data collection and loading.
    - `data_transformation.py`: Manages data preprocessing and feature engineering.
    - `model_training.py`: Responsible for training the machine learning model.
  - `pipelines/`:
    - `training_pipeline.py`: Orchestrates the training pipeline by utilizing components from the `components` folder.
    - `prediction_pipeline.py`: Manages the prediction pipeline for new data inputs.
- `app.py`: A Flask application that serves the model as an API endpoint.
- `Dockerfile`: Defines the Docker image for containerizing the application.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `setup.py`: Configuration for packaging the project.

## Getting Started

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/masabhuq/DiamondPricePrediction.git
   cd DiamondPricePrediction
2. **Build the docker image:**
   ```bash
   docker build -t diamond-price-prediction .
3. **Run the docker container:**
   ```bash
   docker run -p 5000:5000 diamond-price-prediction
