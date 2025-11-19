# EstatePro: Housing Price Prediction Model

**Author:** Blake Thomas

## Overview

This project implements a Machine Learning model to predict housing prices based on the California Housing dataset. Unlike many standard implementations that rely on high-level libraries like Scikit-Learn for the core logic, this notebook implements **Linear Regression with Gradient Descent from scratch** using NumPy.

The model analyzes various features (such as location, number of rooms, and population) to predict the `medianHouseValue` (Price).

## Dataset

The project uses the [California Housing Train Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

### Features Included

| Feature | Description |
|---------|-------------|
| `longitude` | Measure of how far west a house is |
| `latitude` | Measure of how far north a house is |
| `housingMedianAge` | Median age of a house within a block |
| `totalRooms` | Total number of rooms within a block |
| `totalBedrooms` | Total number of bedrooms within a block |
| `population` | Total number of people residing within a block |
| `households` | Total number of households for a block |
| `medianIncome` | Median income for households (tens of thousands of US Dollars) |
| `medianHouseValue` | **Target** - Median house value (US Dollars) |

## Prerequisites

To run this notebook, you will need Python installed along with the following libraries:

- `pandas` - Data manipulation
- `numpy` - Matrix operations and math
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization

Install the dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn
```

## Project Structure & Implementation Details

The notebook is structured into the following logical sections:

### 1. Data Loading and Exploration

- Imports the dataset using Pandas
- Scales label data for better convergence
- Calculates descriptive statistics (Mean, Min, Max, Standard Deviation) for housing prices

### 2. Data Visualization

- **Histograms:** Visualizes the distribution of housing prices
- **Pairplots:** Uses Seaborn to visualize relationships between specific features (e.g., `GrLivArea`, `BedroomAbvGr`, etc.) to identify correlations

### 3. Linear Regression from Scratch

The core machine learning logic is defined via custom Python functions rather than pre-built libraries:

| Function | Description |
|----------|-------------|
| `pred(weights, features)` | Calculates the dot product of weights and features to generate a prediction |
| `loss(predictions, actual_prices)` | Calculates the Mean Squared Error (MSE) to evaluate model performance |
| `gradient(features, predictions, actual_prices)` | Computes the gradient of the loss function with respect to the weights |
| `update(weights, learning_rate, grad)` | Updates the model weights moving against the gradient |

### 4. Training Loop

The `housing_model(alpha, num_iterations)` function runs the Gradient Descent algorithm, iteratively updating weights to minimize the MSE.

### 5. Hyperparameter Tuning & Visualization

- The model is tested with different Learning Rates (`alpha`)
- A plot of MSE vs. Iterations is generated to visually verify that the model is converging correctly

## Usage

1. **Clone the repository** or download the `HousingPriceMLModel.ipynb` file

2. **Update Data Paths:**
   - Locate the cell where `pd.read_csv` is called
   - Update the filepath `"/Users/blakethomas/Desktop/MLproj/train.csv"` to the location of your local dataset or the direct URL

3. **Run the Notebook:** Execute the cells sequentially to train the model

4. **Testing:**
   - The final section of the notebook loads a test set
   - Ensure you have a `test.csv` available and update the path in the final cell before running

## Results

The model tracks the Mean Squared Error (MSE) over 500+ iterations. You can observe the reduction in error as the training progresses, demonstrating the effectiveness of the Gradient Descent implementation.

## License

This project is open source and available under the [MIT License](LICENSE).
