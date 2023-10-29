# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data_file = "golf_play2.csv"
golf_df = pd.read_csv(data_file, skipinitialspace=True)
print(golf_df)
# Remove the Row No column as it is not an important feature
golf_df = golf_df.drop("Row No", axis=1)
print("*** Original Dataset ***")
print(golf_df)

# Map string vales for Outlook column to numbers
code = {"Sunny": 1, "Overcast": 2, "Rain": 3}
golf_df["Outlook"] = [code[item] for item in golf_df["Outlook"]]
print("*** Transformed Dataset ***")
print(golf_df)

# split the data into training and test data
train, test = train_test_split(golf_df, test_size=0.3, random_state=0)

# initialize Gaussian Naive Bayes
clf = GaussianNB()

# Use all columns apart from the Play column as features
train_features = train.iloc[:, 0:4]
# Use the play column as the label
train_label = train.iloc[:, 4]

# Repeat above for test data
test_features = test.iloc[:, 0:4]
test_label = test.iloc[:, 4]

# Train the naive bayes model
clf.fit(train_features, train_label)

# build a dataframe to show the expected vs predicted values
test_data = pd.concat([test_features, test_label], axis=1)
test_data["prediction"] = clf.predict(test_features)

print("*** Test Results ***")
print(test_data)

# Use the score function and output the prediction accuracy
print("Naive Bayes Accuracy:", clf.score(test_features, test_label))
