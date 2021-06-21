######################################
########### RANDOM FORESTS ###########
######################################
df_total.info()
df_rf = df_total.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,24]]
df_rf = df_rf.dropna()
df_rf.isna().sum()
df_rf = df_rf[np.isfinite(df_rf)]
df_rf_backup = df_rf
sc = StandardScaler()
###### RF on marketshares ######

# Notitie: zoals hieronder staat worden alle vars gebruikt
# ALS: var importance voor het kiezen van producten door consumenten en de daarbij volgende van marketshares: gebruik alleen products, comp, freq, price, distri and volume
# ALS: voorspellen van marketshares aan de hand van pref shares en uitkomsten: gebruik alles

y = df_rf.marketshare
X = df_rf
X['frequency'] = X['frequency'].astype('category')
X = X.drop(columns=['marketshare','study','is_want'], axis=1) 
X.isna().sum()
X = pd.DataFrame(sc.fit_transform(X))
X.rename(columns={0:'pref',1:'respondents',2:'tasks',3:'choices',4:'products', 5:'competitors',6:'frequency',7:'price',8:'distri',9:'volume' }, inplace=True)
random.seed(987654321)
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
#plt.style.use('ggplot')
plt.figure()
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_list = ['Preference Share','No. respondents','No. tasks',"No. SKU's per task","No. SKU's in market",'No. competitors',"Frequency","Price","Distribution","Volume"]
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = False)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# list of x locations for plotting
x_values = list(range(len(importances)))

important = pd.DataFrame(feature_importances)
# Make a bar chart
plt.barh(important[0],important[1], color='cornflowerblue')
for index, value in enumerate(important[1]):
    plt.text(value, index, str(value), fontsize = 13)
# Tick labels for x axis
#plt.xticks(feature_list)
#plt.yticks(feature_list)
# Axis labels and title
plt.ylabel('Variable',fontsize = 15); plt.xlabel('Importance',fontsize = 15); plt.title('Variable Importances',fontsize = 16);
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.show()