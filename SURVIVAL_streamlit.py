#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries for graphs and plots
import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# library for PCA
from sklearn.decomposition import PCA

# libraries for ML classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# libraries for classification report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# libraries for roc curve
from sklearn.metrics import roc_curve, roc_auc_score

# Update for the deprecated warning
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')



# In[17]:


import pandas as pd

# Provide the full path to your CSV file (escaped backslashes)
file_path = 'heart_failure_clinical_records_dataset.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert DataFrame to numpy array
data = df.values

# Print the number of rows and columns
print(data.shape)


# In[19]:


# First 10 entries of the dataset
df.head(10)


# In[21]:


df.info()


# In[23]:


df['DEATH_EVENT'].value_counts()


# In[27]:


indices1 = np.array([0]) 
count1 = np.array([203])
indices2 = np.array([0])
count2 = np.array([96])
plt.bar(indices1,count1,color='green',label='Patient survived')
plt.bar(indices2+1,count2,color='red',label='Patient died') 
plt.legend()
plt.show()


# In[35]:


labels = ['32.11 % patients died due to heart failure', '67.89 % patients‹→survived']
values = [96, 203]

plt.pie(values, labels=labels, radius=1)
plt.show()


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplot parameters
left = 0.1
right = 3
bottom = 0.1
top = 1
wspace = 0.3
hspace = 0.2

# Create subplots
f, axes = plt.subplots(1, 3)
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

# Create boxplots
sns.boxplot(x='DEATH_EVENT', y='age', data=df, ax=axes[0])
sns.boxplot(x='DEATH_EVENT', y='anaemia', data=df, orient='v', ax=axes[1])
sns.boxplot(x='DEATH_EVENT', y='creatinine_phosphokinase', data=df, orient='v', ax=axes[2])

# Display the plot
plt.show()


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplot parameters
left = 0.1
right = 3
bottom = 0.1
top = 1
wspace = 0.3
hspace = 0.2

# Create subplots
f, axes = plt.subplots(1, 3)
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

# Create boxplots
sns.boxplot(x='DEATH_EVENT', y='diabetes', data=df, orient='v', ax=axes[0])
sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=df, orient='v', ax=axes[1])
sns.boxplot(x='DEATH_EVENT', y='high_blood_pressure', data=df, orient='v', ax=axes[2])

# Display the plot
plt.show()


# In[49]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(corr_matrix, annot=True, linewidth=.5, ax=ax)


# In[51]:


df.drop(['diabetes', 'sex'], axis=1, inplace=True)


# In[53]:


df.describe()


# In[57]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# Normalize the data and display the first 10 rows
data = normalize(df)
data.head(10)


# In[63]:


from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Using the useful_data DataFrame
useful_data = df

# Apply PCA with 3 components
pca = PCA(n_components=3)
pca.fit(useful_data)

# Dimension indexing
dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]

# Individual PCA components
components = pd.DataFrame(np.round(pca.components_, 5), columns=useful_data.columns)
components.index = dimensions

# Explained Variance in PCA
ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
variance_ratios.index = dimensions

# Printing required information
pd.set_option('display.max_columns', None)
print(pd.concat([variance_ratios, components], axis=1))


# In[65]:


# Dropping the specified attributes from df and data
df.drop(['anaemia', 'smoking', 'high_blood_pressure'], axis=1, inplace=True)
data.drop(['anaemia', 'smoking', 'high_blood_pressure'], axis=1, inplace=True)

# Displaying the first 10 rows of the modified data
data.head(10)


# In[67]:


from sklearn.model_selection import train_test_split

# Selecting features and target variable
x = data.iloc[:, 0:7]  # Selecting the first 7 columns as features
y = data['DEATH_EVENT']  # Target variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)


# In[93]:


knn = KNeighborsClassifier()
k_range = list(range(1, 25))
k_scores = []
for k in k_range:
knn = KNeighborsClassifier(n_neighbors=k) knn.fit(X_train, y_train) k_scores.append(knn.score(X_train, y_train))
print(np.round(k_scores, 4))


# In[71]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Initialize the list to store accuracy scores
k_range = list(range(1, 25))
k_scores = []

# Loop through the range of k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  # Train the model
    k_scores.append(knn.score(X_train, y_train))  # Store the accuracy score

# Print the rounded accuracy scores
print(np.round(k_scores, 4))


# In[75]:


import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

# Initialize a list to store recall scores
k_scores = []

# Loop through the range of k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)  # Get predictions
    recall = recall_score(y_train, y_pred)  # Calculate recall score
    k_scores.append(recall)  # Store the recall score

# Plot k-value vs recall
plt.plot(k_range, k_scores, color="Blue")
plt.xlabel('k values')
plt.ylabel('Recall')
plt.title('k-value vs Recall')
plt.show()


# In[79]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Fit the classifier on the training data
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Prediction on test data
y_pred = classifier.predict(X_test)

# Print accuracy score
print("Accuracy score: ", accuracy_score(y_test, y_pred), '\n')

# Print classification report
print("Classification report: ")
print(classification_report(y_test, y_pred))

# Plot confusion matrix as heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()


# In[85]:


#Logistic Regression
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict the output using the trained logistic regression model
log_predicted = logreg.predict(X_test)

# Calculate accuracy scores
logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)

# Accuracy and Classification Report
print('Accuracy: ', accuracy_score(y_test, log_predicted), '\n')
print('Classification Report: \n', classification_report(y_test, log_predicted))

# Confusion matrix heatmap
sns.heatmap(confusion_matrix(y_test, log_predicted), annot=True, fmt="d")
plt.show()

sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")


# In[91]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Preparing test and training sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=0)

# Training models
classifier.fit(X_train1, y_train1)  # KNN
logreg.fit(X_train1, y_train1)  # Logistic Regression

# Finding predicted probabilities
y_score1 = classifier.predict_proba(X_test1)[:, 1]  # KNN probabilities
y_score2 = logreg.predict_proba(X_test1)[:, 1]  # Logistic Regression probabilities

# Creating true and false positive rates for ROC curve
fpr1, tpr1, threshold1 = roc_curve(y_test1, y_score1)
fpr2, tpr2, threshold2 = roc_curve(y_test1, y_score2)

# ROC AUC scores
print('roc_auc_score for K-Nearest Neighbors (k=5): ', roc_auc_score(y_test1, y_score1))
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test1, y_score2))
print('\n')

# ROC Curves Plot
plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic Curve')
plt.plot(fpr1, tpr1, color='orange', label='KNN (k=5) (AUC = {0:0.2f})'.format(roc_auc_score(y_test1, y_score1)))
plt.plot(fpr2, tpr2, color='blue', label='Logistic Regression (AUC = {0:0.2f})'.format(roc_auc_score(y_test1, y_score2)))
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7")
plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(prop={'size': 15})
plt.show()


# In[ ]:




