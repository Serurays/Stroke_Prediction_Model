import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
from imblearn.over_sampling import SMOTE

# Importing the dataset
stroke_df = pd.read_csv('C:/Users/MSI/Desktop/Kaggle/Kaggle_Stroke_Prediction_ML/healthcare-dataset-stroke-data.csv')

# Exploratory Data Analysis
print(stroke_df.head())

stroke_df.info()
# We can see that there are some NaN values in the bmi column.
# We also have some categorical variables, that we need to encode.
# Plus, we won't use the id column, so we can drop it.
stroke_df.drop('id', axis=1, inplace=True)

# Filling the NaN Values
# For the NaN Values, there are various methods that can be applied, we can even delete the rows with the NaN Values.
# deleting the rows with NaN Values: dataset.dropna(axis=0)

# For this dataset, I'll go with replacing the NaN Values in the bmi column
# with the mean of the remaining row values in the bmi column they are in.
stroke_df['bmi'].fillna(stroke_df['bmi'].mean(),inplace=True)

# Now we don't have anymore NaN Values
print(stroke_df.isnull().sum())

# Now, we can look at the outliers,
# sometimes removing the outliers may improve the efficiency of the machine learning model.
stroke_df.boxplot(figsize=(10, 7))
plt.show()

# In this case, we have a lot of outliers, as we can see as a lot of black dots that are outside the boxplot.
# So, we won't be removing any.

# Label Encoding
# converting categorical data into numerical format
# becomes is for the algorithms to process the information effectively,
# since  In many machine learning algorithms, mathematical equations are used to make predictions or classifications,
# and these equations often work with numerical values.
lenc = LabelEncoder()

# So, let's find the categorical columns
categorical_columns = stroke_df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    stroke_df[col] = lenc.fit_transform(stroke_df[col])

stroke_df.info()

# Splitting the data into training and test set
# So, firstly you have to  divide your dataset into two: one that contains the dependent variables
# aka features which are the input variables used to make the prediction
# and the other one that contains the dependent variable aka target, which we're trying to predict

X = stroke_df.drop('stroke', axis=1)
y = stroke_df['stroke']

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
print(y.value_counts())
print(y_sm.value_counts())

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=0, stratify=y_sm)
print(y_train.value_counts())
print(y_test.value_counts())

# Now, finally, we can train our model.
# For this example, I'll use the Decision Tree Classifier with class weights.
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy Score
ac = accuracy_score(y_test, y_pred)
print(ac)

print(classification_report(y_test, y_pred))

# pickle
pickle_out = open('stroke_prediction_model.pkl', 'wb')
pickle.dump(rf, pickle_out)
pickle_out.close()