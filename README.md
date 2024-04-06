# Email Spam Detector

This project builds and compares two machine learning models - Logistic Regression and Random Forest - for detecting spam emails. The models are trained on a dataset of email features and labels indicating whether each email is spam or not.

## Dataset

The dataset used in this project is sourced from [here](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv). It contains 4,601 email samples, each represented by 57 features including word frequencies, character frequencies, and capital run length statistics. The target variable is a binary 'spam' label (1 for spam, 0 for non-spam).

## Project Structure

- `spam_detector.ipynb`: Jupyter Notebook containing the main code for data loading, preprocessing, model training, evaluation, and comparison.
- `spam-data.csv`: The raw dataset file.
- `README.md`: This readme file providing an overview of the project.

## Dependencies

- Python 3.7+
- pandas
- scikit-learn

Install the required Python packages using pip:
```
pip install pandas scikit-learn
```

## Usage

1. Clone this repository and navigate to the project directory.
2. Run the Jupyter Notebook `spam_detector.ipynb` and execute the cells in order.

The notebook will:
- Load and preprocess the dataset
- Split the data into training and testing sets
- Scale the feature data using StandardScaler
- Train and evaluate a Logistic Regression model
- Train and evaluate a Random Forest model
- Compare the performance of the two models
- Save the trained models for future use

## Results

On the test set:
- The Logistic Regression model achieved an accuracy of 92.0%.
- The Random Forest model achieved an accuracy of 95.7%.

The Random Forest model outperformed the Logistic Regression model for this specific task and dataset.

## Future Work

Potential enhancements to this project could include:

- Experimenting with additional features or feature engineering techniques
- Tuning the hyperparameters of the models to optimize performance
- Exploring other machine learning algorithms for spam detection
- Building a user interface or API to allow easy interaction with the trained models

Feel free to contribute to this project by submitting pull requests or reporting issues.

## License

This project is open source and available under the [MIT License](LICENSE).