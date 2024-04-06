import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
data = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv')

# Explore and preprocess the data
print(data.head())
print(data.info())
print(data.describe())

# Create feature matrix X and target vector y
feature_columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 
                   'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 
                   'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 
                   'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 
                   'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 
                   'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab',
                   'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415',
                   'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 
                   'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 
                   'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 
                   'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
                   'capital_run_length_longest', 'capital_run_length_total']

X = data[feature_columns]
y = data['spam']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the class balance
print("Labels counts in y_train:")
print(y_train.value_counts())

print("Labels counts in y_test:") 
print(y_test.value_counts())

# Scale the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a logistic regression model
logreg = LogisticRegression(random_state=1)
logreg.fit(X_train_scaled, y_train)

# Make predictions on test set and evaluate performance
logreg_pred = logreg.predict(X_test_scaled)
logreg_acc = accuracy_score(y_test, logreg_pred)
print(f'Logistic Regression Accuracy: {logreg_acc:.3f}')

# Fit a random forest model
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train_scaled, y_train)

# Make predictions on test set and evaluate performance
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f'Random Forest Accuracy: {rf_acc:.3f}')

# Compare model results
if logreg_acc > rf_acc:
    print('Logistic Regression performed better')
else:
    print('Random Forest performed better')

# Print classification report and confusion matrix
print("Logistic Regression:")
print(confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

print("Random Forest:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Save the model
import joblib
joblib.dump(logreg, 'logistic_regression_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')

# Print out the results with a summary of results and the models
print("Summary:")
print(f'Logistic Regression Accuracy: {logreg_acc:.3f}')
print(f'Random Forest Accuracy: {rf_acc:.3f}')
print("Models saved successfully!")
