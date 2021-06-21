######################################
################# SVR ################
######################################
test = df_rf
test =test.dropna()
y = pd.DataFrame(test.marketshare)
X = test
X['frequency'] = X['frequency'].astype('category')
X = X.drop(columns=['marketshare','study','is_want'], axis=1)
X.isna().sum()
X = (sc.fit_transform(X))

random.seed(963)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=30)
# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

svr_rbf.fit(X_train,y_train)
svr_lin.fit(X_train,y_train)
svr_poly.fit(X_train,y_train)
# #############################################################################
# Look at the results NOT POSSIBLE< NOT LINEAR

y=np.array(y)

#RBF
predictions = svr_rbf.predict(X_train)
# Calculate the absolute errors
errors = abs(predictions - np.array(y_train))
# Print out the mean absolute error (mae)
print('Mean Absolute Error-rbf-IS:', round(np.mean(errors), 2)) # 0.08
# Calculate mean absolute percentage error (MAPE)
mape = abs((y_train - predictions)/y_train)
mape = mape[np.isfinite(mape)]
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy-rbf-IS:', round(accuracy, 2), '%.') #45.87

# Use the svr's predict method on the test data
predictions = svr_rbf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - np.array(y_test))
# Print out the mean absolute error (mae)
print('Mean Absolute Error-rbf-OOS:', round(np.mean(errors), 2)) #0.07
# Calculate mean absolute percentage error (MAPE)
mape = abs((y_test - predictions)/y_test)
mape = mape[np.isfinite(mape)]
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy-rbf-OOS:', round(accuracy, 2), '%.') #62.76

#LIN
predictions2 = svr_lin.predict(X_train)
# Calculate the absolute errors
errors2 = abs(predictions2 - y_train)
# Print out the mean absolute error (mae)
print('Mean Absolute Error-lin-OOS:', round(np.mean(errors2), 2)) #0.06
# Calculate mean absolute percentage error (MAPE)
mape2 = abs((y_train - predictions2)/y_train)
mape2 = mape2[np.isfinite(mape2)]
# Calculate and display accuracy
accuracy2 = 100 - np.mean(mape2)
print('Accuracy-lin-OOS:', round(accuracy2, 2), '%.') #65.97

predictions2 = svr_lin.predict(X_test)
# Calculate the absolute errors
errors2 = abs(predictions2 - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error-lin-OOS:', round(np.mean(errors2), 2)) #0.05
# Calculate mean absolute percentage error (MAPE)
mape2 = abs((y_test - predictions2)/y_test)
mape2 = mape2[np.isfinite(mape2)]
# Calculate and display accuracy
accuracy2 = 100 - np.mean(mape2)
print('Accuracy-lin-OOS:', round(accuracy2, 2), '%.') #77.11

#poly

predictions3 = svr_poly.predict(X_train)
# Calculate the absolute errors
errors3 = abs(predictions3 - y_train)
# Print out the mean absolute error (mae)
print('Mean Absolute Error-poly-IS:', round(np.mean(errors3), 2)) #0.04
# Calculate mean absolute percentage error (MAPE)
mape3 = abs((y_train - predictions3)/y_train)
mape3 = mape3[np.isfinite(mape3)]
# Calculate and display accuracy
accuracy3 = 100 - np.mean(mape3)
print('Accuracy-poly-IS:', round(accuracy3, 2), '%.') #71.2

predictions3 = svr_poly.predict(X_test)
# Calculate the absolute errors
errors3 = abs(predictions3 - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error-poly-OOS:', round(np.mean(errors3), 2)) #0.04
# Calculate mean absolute percentage error (MAPE)
mape3 = abs((y_test - predictions3)/y_test)
mape3 = mape3[np.isfinite(mape3)]
# Calculate and display accuracy
accuracy3 = 100 - np.mean(mape3)
print('Accuracy-poly-OOS:', round(accuracy3, 2), '%.') #78.47


#y_pred = regressor.predict(6.5)
#y_pred = sc_y.inverse_transform(y_pred) 

