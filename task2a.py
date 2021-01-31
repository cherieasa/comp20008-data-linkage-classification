# Code written by Terada Asavapakuna 1012869

# import libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
# for splitting data -> test/train
from sklearn.model_selection import train_test_split
# for imputing missing values
from sklearn.impute import SimpleImputer
# mean removal and variance scaling
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# ----------------- reading/ cleaning data -----------------
# read csv
world_df = pd.read_csv("world.csv")
life_df = pd.read_csv("life.csv")

# remove the extra lines underneath the last country
world_df = world_df.drop([264,265,266,267,268])
# list of values in world
world_row_list  = world_df.values.tolist()

# get list of country codes from life.csv
country_life = {}
# make dictionary
for rows in life_df.values:
    # country code as key and value as life expectancy
    country_life[rows[1]] = rows[3]

# remove rows where it doesn't exist in life.csv
remove_code = []
for country_row in world_row_list:
    if country_row[2] not in list(country_life.keys()):
        remove_code.append(country_row[2])
        
# remove in world_row_list
cleanse_world = []
for row in world_row_list:
    
    # if country exists in life.csv add to list
    if row[2] not in remove_code:
        cleanse_world.append(row)

# impute missing values ("..")

# change ".." string -> NaN 
nan_world = []
for each_data_set in cleanse_world:
    list_set = []
    for each_data in each_data_set:
        if each_data == '..':
            list_set.append(np.nan)
        else:
            list_set.append(each_data)
                
    nan_world.append(list_set)
        
# extract only numerical value
num_world = []
list_of_codes = []
for each_data_set in nan_world:
    list_of_codes.append(each_data_set[2])
    num_world.append(each_data_set[3:])
    
# impute nan values using median
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(num_world)

impute_x_train = imp.transform(num_world)

# get median used to impute each feature
x_median = np.median(impute_x_train, axis=0).round(3)

#--------------- put everything back in to a dataframe -------------------

final_data = []

for i in range(len(nan_world)):
    
    add = []
    add.append(nan_world[i][2])
    add.extend(impute_x_train[i])
    final_data.append(add)

# add life expectancy (label) to the countries in final_data using country code
full_table = []
each_country_data = []

for each_country in final_data:
    full_country = []
    curr_country = each_country[0]
    for each_data in each_country:
        full_country.append(each_data)
        
    full_country.append(country_life[curr_country])
    full_table.append(full_country)

# now remove country code
pure_data = []
for each_country in full_table:
    each_data = each_country[1:]
    pure_data.append(each_data)

# convert to dataframe - ONLY use numerical value (features) & life expectancy (label)
    
# remove Country Name,Time,Country Code
# get columns
cols = list(world_df.columns.values)[3:]
cols.append(list(life_df.columns.values)[-1])
no_cols = len(cols)

data_df = DataFrame(pure_data, columns = cols)

# ----------------- split data -> train 2/3 | test 1/3 -----------------

# x - features
x = data_df.iloc[:, :-1].values
# y - labels
y = data_df.iloc[:, no_cols-1].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=100)

# ------ scaling (remove mean and scale to unit variance) ------

scaler = StandardScaler()

# scale data to training data as basis
scaler.fit(x_train)
#print(x_train)
# get mean
x_train_mean = (scaler.mean_).round(3)

# get variance
x_train_var = (scaler.var_).round(3)

# get median of training set
x_train_median = np.median(x_train, axis=0).round(3)

x_train = scaler.transform(x_train).round(3)
x_test = scaler.transform(x_test).round(3)

x_train_df = DataFrame(x_train, columns = list(world_df.columns.values)[3:])
y_train_df = DataFrame(y_train, columns = [list(life_df.columns.values)[-1]])
x_test_df = DataFrame(x_test, columns = list(world_df.columns.values)[3:])
y_test_df = DataFrame(y_test, columns = [list(life_df.columns.values)[-1]])


x_train_df.to_csv('x_train_standardised.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
x_test_df.to_csv('x_test_standardised.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)


# ----------------- train classifier -----------------

# ~~~~~ using KNeighborsClassifier - k = 5 ~~~~~

classifier_k_5 = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
classifier_k_5.fit(x_train, y_train)

# predict using test set
y_pred_k_5 = classifier_k_5.predict(x_test)

# check accuracy of KNeighborsClassifier - k = 5
accuracy_k_5 = metrics.accuracy_score(y_test, y_pred_k_5)
accuracy_k_5 = accuracy_k_5*100
accuracy_k_5 = "{:.3f}".format(accuracy_k_5)
k_5_percent = str(accuracy_k_5) + "%"

# ~~~~~ using KNeighborsClassifier - k = 10 ~~~~~

classifier_k_10 = KNeighborsClassifier(n_neighbors=10)

# Train the model using the training sets

classifier_k_10.fit(x_train, y_train)

# predict using test set
y_pred_k_10 = classifier_k_10.predict(x_test)

# check accuracy of KNeighborsClassifier - k = 10
accuracy_k_10 = metrics.accuracy_score(y_test, y_pred_k_10)
accuracy_k_10 = accuracy_k_10*100
accuracy_k_10 = "{:.3f}".format(accuracy_k_10)
k_10_percent = str(accuracy_k_10) + "%"

# ~~~~~ using decisionTreeDecisionTreeClassfier - depth = 4 ~~~~~

clf = DecisionTreeClassifier(max_depth=4)# add random_state = 100 for consistent results)

# train decision tree classifer
clf = clf.fit(x_train,y_train)

# predict the response for test dataset
y_pred_decision = clf.predict(x_test)
accuracy_decision = metrics.accuracy_score(y_test, y_pred_decision)
accuracy_decision = accuracy_decision*100
accuracy_decision = "{:.3f}".format(accuracy_decision)
decision_percent = str(accuracy_decision) + "%"

# print standard output
print("Accuracy of decision tree:", decision_percent)
print("Accuracy of k-nn (k=5):", k_5_percent)
print("Accuracy of k-nn (k=10):", k_10_percent)

# ----------------- output to csv file -----------------

# extract feature names

feature_col = list(world_df.columns.values)[3:]

# append median/mean/var with respect to each feature 

feature_values = []

for i in range(len(feature_col)):
    
    # append feature name
    each_feature = []
    each_feature.append(feature_col[i])
    each_feature.append(x_train_median[i])
    each_feature.append(x_train_mean[i])
    each_feature.append(x_train_var[i])

    feature_values.append(each_feature)
    
feature_df = DataFrame(feature_values,columns=['feature', 'median', 'mean', 'variance'])

feature_df.to_csv('task2a.csv', index=False)

