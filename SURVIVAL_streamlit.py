import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# library for PCA
from sklearn.decomposition import PCA

# libraries for ML classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# libraries for classification report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# libraries for roc curve
from sklearn.metrics import roc_curve, roc_auc_score

st.title("Heart Failure Survival Analysis")

# Load the dataset
file_path = 'heart_failure_clinical_records_dataset.csv'  # Just the filename

# Read the CSV file
st.header("Dataset Preview")
df = pd.read_csv(file_path)
st.write("Shape of data:", df.shape)
st.dataframe(df.head(10))

st.header("Dataset Info")
st.write(df.info())

st.header("Target Value Counts")
st.write(df['DEATH_EVENT'].value_counts())

# Bar plot
st.subheader("Survival Distribution")
fig, ax = plt.subplots()
indices1 = np.array([0]) 
count1 = np.array([203])
indices2 = np.array([0])
count2 = np.array([96])
ax.bar(indices1,count1,color='green',label='Patient survived')
ax.bar(indices2+1,count2,color='red',label='Patient died')
ax.legend()
st.pyplot(fig)

# Pie chart
st.subheader("Pie Chart of Survival")
fig, ax = plt.subplots()
labels = ['32.11% patients died', '67.89% patients survived']
values = [96, 203]
ax.pie(values, labels=labels, radius=1)
st.pyplot(fig)

# Boxplots 1
st.subheader("Boxplots: Age, Anaemia, Creatinine Phosphokinase")
fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(x='DEATH_EVENT', y='age', data=df, ax=axes[0])
sns.boxplot(x='DEATH_EVENT', y='anaemia', data=df, ax=axes[1])
sns.boxplot(x='DEATH_EVENT', y='creatinine_phosphokinase', data=df, ax=axes[2])
st.pyplot(fig)

# Boxplots 2
st.subheader("Boxplots: Diabetes, Ejection Fraction, High Blood Pressure")
fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(x='DEATH_EVENT', y='diabetes', data=df, ax=axes[0])
sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=df, ax=axes[1])
sns.boxplot(x='DEATH_EVENT', y='high_blood_pressure', data=df, ax=axes[2])
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, ax=ax)
st.pyplot(fig)

# Dropping some columns
df.drop(['diabetes', 'sex'], axis=1, inplace=True)

# Normalizing data
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

data = normalize(df)
st.subheader("Normalized Data Sample")
st.dataframe(data.head(10))

# PCA
st.subheader("PCA Analysis")
pca = PCA(n_components=3)
pca.fit(df)
components = pd.DataFrame(np.round(pca.components_, 5), columns=df.columns)
components.index = ['Dimension 1', 'Dimension 2', 'Dimension 3']
ratios = pd.DataFrame(np.round(pca.explained_variance_ratio_.reshape(len(pca.components_),1), 4), columns=['Explained Variance'])
ratios.index = components.index
st.dataframe(pd.concat([ratios, components], axis=1))

# Further dropping columns
df.drop(['anaemia', 'smoking', 'high_blood_pressure'], axis=1, inplace=True)
data.drop(['anaemia', 'smoking', 'high_blood_pressure'], axis=1, inplace=True)

# Splitting
x = data.iloc[:, 0:7]
y = data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)

# KNN Model k-score vs k
st.subheader("KNN Accuracy for Different k Values")
k_range = list(range(1,25))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(knn.score(X_train, y_train))
st.line_chart(pd.DataFrame({'k': k_range, 'Accuracy': k_scores}).set_index('k'))

# KNN Recall vs k
st.subheader("KNN Recall for Different k Values")
k_recall_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    k_recall_scores.append(recall_score(y_train, y_pred))
st.line_chart(pd.DataFrame({'k': k_range, 'Recall': k_recall_scores}).set_index('k'))

# KNN Confusion Matrix
st.subheader("KNN Confusion Matrix")
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
st.pyplot(fig)

# Logistic Regression
st.subheader("Logistic Regression Confusion Matrix")
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
log_predicted = logreg.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, log_predicted))
st.text("Classification Report:\n" + classification_report(y_test, log_predicted))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, log_predicted), annot=True, fmt="d", ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve Comparison")
X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=0)
classifier.fit(X_train1, y_train1)
logreg.fit(X_train1, y_train1)

y_score1 = classifier.predict_proba(X_test1)[:,1]
y_score2 = logreg.predict_proba(X_test1)[:,1]

fpr1, tpr1, _ = roc_curve(y_test1, y_score1)
fpr2, tpr2, _ = roc_curve(y_test1, y_score2)

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(fpr1, tpr1, color='orange', label=f'KNN (AUC = {roc_auc_score(y_test1, y_score1):.2f})')
ax.plot(fpr2, tpr2, color='blue', label=f'Logistic Regression (AUC = {roc_auc_score(y_test1, y_score2):.2f})')
ax.plot([0, 1], ls='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic Curve')
ax.legend()
st.pyplot(fig)

# ----------- Real-time Prediction Form ------------
st.header("🩺 Predict Heart Failure Risk (Real-Time)")

# Collect user input features
age = st.number_input('Age', min_value=0, max_value=130, value=60)
creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase (mcg/L)', min_value=0, max_value=8000, value=500)
ejection_fraction = st.number_input('Ejection Fraction (%)', min_value=10, max_value=100, value=38)
platelets = st.number_input('Platelets (kiloplatelets/mL)', min_value=0, max_value=1000000, value=265000)
serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', min_value=0.0, max_value=10.0, value=1.1)
serum_sodium = st.number_input('Serum Sodium (mEq/L)', min_value=100, max_value=150, value=137)
time = st.number_input('Follow-up Period (days)', min_value=0, max_value=400, value=130)

# When the user clicks the button
if st.button('Predict Survival'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'age': [age],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'ejection_fraction': [ejection_fraction],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'time': [time]
    })

    # Normalize the input data based on original dataset normalization
    input_normalized = (input_data - x.min()) / (x.max() - x.min())

    # Make prediction
    prediction = classifier.predict(input_normalized)

    # Show result
    if prediction[0] == 0:
        st.success("✅ The patient is likely to **Survive**.")
    else:
        st.error("⚠️ The patient is at risk of a **Death Event**.")

