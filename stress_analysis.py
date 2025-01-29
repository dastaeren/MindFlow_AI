!pip install pandas

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Stress.csv")
data

data.dtypes

data.describe()

print("Missing Values:\n", data.isnull().sum())

class_distribution = data['label'].value_counts(normalize=True) * 100
print("\nClass Distribution:\n", class_distribution)

sns.set_style("whitegrid")

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=class_distribution.index, y=class_distribution.values, palette=["blue", "red"])
plt.xticks(ticks=[0, 1], labels=["Non-Stress (0)", "Stress (1)"])
plt.ylabel("Percentage (%)")
plt.title("Class Distribution")
plt.show()

# Calculate text length and add it as a new column
data['text_length'] = data['text'].str.len()

# Plot text length distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['text_length'], bins=30, kde=True, color='purple')
plt.xlabel("Text Length (Characters)")
plt.ylabel("Frequency")
plt.title("Text Length Distribution")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Calculate text length and add it as a new column (MOVED HERE)
data['text_length'] = data['text'].str.len()

# Linear Regression
X = data[['text_length']]  # 'text_length' column now exists
y = data['confidence']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("R-squared Score:", r2)

# Plot regression line
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test['text_length'], y=y_test, label="Actual", color='blue')
sns.lineplot(x=X_test['text_length'], y=y_pred, label="Predicted", color='red')
plt.xlabel("Text Length")
plt.ylabel("Confidence Score")
plt.title("Linear Regression: Text Length vs Confidence Score")
plt.legend()
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Text Classification Model (Naive Bayes)
X_text = data['text']
y_label = data['label']

# Split dataset
X_train_text, X_test_text, y_train_label, y_test_label = train_test_split(X_text, y_label, test_size=0.2, random_state=42)

# Create text classification pipeline
text_clf = make_pipeline(TfidfVectorizer(), MultinomialNB())
text_clf.fit(X_train_text, y_train_label)

# Predictions
y_pred_label = text_clf.predict(X_test_text)

# Evaluate model
accuracy = accuracy_score(y_test_label, y_pred_label)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_label, y_pred_label)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Stress", "Stress"], yticklabels=["Non-Stress", "Stress"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

!pip install imblearn

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode text data as features and labels
X = data['text']  # Input features (text)
y = data['label']  # Target column (label)

# Convert the text data into numerical format (basic vectorization example)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000)  # Use top 5000 words
X_vectorized = vectorizer.fit_transform(X).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Display the new distribution of the target variable after oversampling
print("Original training set distribution:")
print(y_train.value_counts())

print("\nResampled training set distribution:")
# Calculate value counts first, then print and multiply
resampled_counts = pd.Series(y_train_resampled).value_counts()
print(resampled_counts * 6)  # Multiply the counts by 6

!pip install joblib

# Import necessary libraries
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
import pandas as pd # Make sure pandas is imported

# Assuming 'data' is your DataFrame containing the text and label columns

# Encode text data as features and labels
X = data['text']  # Input features (text)
y = data['label']  # Target column (label)

# Convert the text data into numerical format (basic vectorization example)
vectorizer = CountVectorizer(max_features=5000)  # Use top 5000 words
X_vectorized = vectorizer.fit_transform(X).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create and train a Logistic Regression model
logreg = LogisticRegression(random_state=42)  # Initialize the model
logreg.fit(X_train_resampled, y_train_resampled)  # Train the model

# Now you can save the trained model
joblib.dump(logreg, "logistic_regression_model.pkl")
print("Model saved as 'logistic_regression_model.pkl'.")