
######################################
########### RANDOM FORESTS ###########
######################################



###### dataset to be used ######
df_rf = df_total.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,24]]
df_rf = df_rf.dropna()
df_rf.isna().sum()
df_rf = df_rf[np.isfinite(df_rf)]
df_rf_backup = df_rf
sc = StandardScaler()

###### RF on difference ######


# Notitie: zoals hieronder staat worden alle vars gebruikt
# ALS: var importance voor het verschil: gebruik alles behalve prefshares en marketshares

y = df_rf.is_want
X = df_rf
X['frequency'] = X['frequency'].astype('category')
X = X.drop(columns=['PrefSh_without','marketshare','study','is_want'], axis=1)
cols= {0:'respondents', 1:'tasks', 2:'choices', 3:'products', 4:'competitors',
       5:'frequency', 6:'std_price', 7:'std_distri', 8:'std_volume'}
X.isna().sum()
X = pd.DataFrame(sc.fit_transform(X))
X.rename(columns=cols, inplace=True)
random.seed(987654)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=30)

rf = RandomForestRegressor(n_estimators=600)

# Train the classifier using the train data
rf = rf.fit(X_train, y_train)
# MSE
prediction = rf.predict(X_test)
msetest = mean_squared_error(y_test,prediction)
msetrain = mean_squared_error(y_train, rf.predict(X_train))
print('MSE on train set: ' + str(msetrain))
print('MSE on test set: ' + str(msetest))

# Validate the classifier
accuracy2 = rf.score(X_train, y_train)
print('Accuracy on train set: ' + str(accuracy2))
accuracy = rf.score(X_test, y_test)
print('Accuracy on test set: ' + str(accuracy))


# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
# Calculate mean absolute percentage error (MAPE)
mape = abs((y_test - predictions)/y_test)
mape = mape[np.isfinite(mape)]
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Out-of-sample-Accuracy:', round(accuracy, 2), '%.')

#INSAMPLE
predictions2 = rf.predict(X_train)
# Calculate the absolute errors
errors2 = abs(predictions2 - y_train)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors2), 2))
# Calculate mean absolute percentage error (MAPE)
mape2 = abs((y_train - predictions2)/y_train)
mape2 = mape2[np.isfinite(mape2)]
# Calculate and display accuracy
accuracy2 = 100 - np.mean(mape2)
print('In-sample-Accuracy:', round(accuracy2, 2), '%.')




%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_list = list(X.columns)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = False)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# list of x locations for plotting
x_values = list(range(len(importances)))

important = pd.DataFrame(feature_importances)
# Make a bar chart
plt.barh(important[0],important[1], color='lightblue')
# Tick labels for x axis
#plt.xticks(feature_list)
#plt.yticks(feature_list)
# Axis labels and title
plt.ylabel('Variable'); plt.xlabel('Importance')#; plt.title('Variable Importances');
plt.show()