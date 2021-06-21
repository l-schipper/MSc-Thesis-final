from tabulate import tabulate_formats
import b0dependencies

######################################
########## BLACK BOX predict #########
######################################

df_rf = df_total.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,24]]
df_rf = df_rf.dropna()
df_rf.isna().sum()
df_rf = df_rf[np.isfinite(df_rf)]
df_rf_backup = df_rf
sc = StandardScaler()

df_predict = df_rf.drop(columns=['study','is_want'])
df_predict['frequency'] = df_predict['frequency'].astype('category')
df_predict = pd.DataFrame(sc.fit_transform(df_predict))
df_predict.rename(columns={0:'pref',1:'marketshare',2:'respondents',3:'tasks',4:'choices',5:'products',6:'competitors',7:'frequency',8:'price',9:'distri',10:'volume' }, inplace=True)

df_predict.columns

df_predict.to_csv('df_predict.csv')

y = df_rf.marketshare
X = df_rf
X['frequency'] = X['frequency'].astype('category')
X = X.drop(columns=['marketshare','study','is_want'], axis=1) 
X.isna().sum()
X = pd.DataFrame(sc.fit_transform(X))
X.rename(columns={0:'pref',1:'respondents',2:'tasks',3:'choices',4:'products', 5:'competitors',6:'frequency',7:'price',8:'distri',9:'volume' }, inplace=True)
random.seed(987654321)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=30)



#* RF
rf=loaded_rf
rf = RandomForestRegressor(n_estimators=600)

rf2 = RandomForestRegressor(n_estimators=600, max_depth=5)
rf = rf.fit(X_train, y_train)
#rf= rf2.fit(X_train,y_train)
rf = svr_poly.fit(X_train,y_train)
predictiontest = rf.predict(X_test)
msetest = mean_squared_error(y_test,predictiontest)
msetrain = mean_squared_error(y_train, rf.predict(X_train))
mape_test = abs((y_test - predictiontest)/y_test)
mape_test = mape_test[np.isfinite(mape_test)]
accuracy = 100 - np.mean(mape_test)
predictiontrain = rf.predict(X_train)
mape_train = abs((y_train - predictiontrain)/y_train)
mape_train = mape_train[np.isfinite(mape_train)]
accuracy2 = 100 - np.mean(mape_train)
print([math.sqrt(msetrain), math.sqrt(msetest), np.mean(mape_train), np.mean(mape_test)])


score = rf.score(X_train, y_train)
print("R-squared:", score) 

rf.score(X_train, y_train)
rf.score(X_test, y_test)

rf.r2_score(y_test, rf.predict(X_test))

model.score(X_train,y_train)

np.mean(abs(predictiontrain-y_train))

back =rf
rf=loaded_rf
rf=back
acctabsecond = pd.DataFrame()
acctabsecond[2] = [msetrain, msetest, math.sqrt(msetrain), math.sqrt(msetest), np.mean(abs(predictiontrain-y_train)),np.mean(abs(predictiontest-y_test)), np.mean(mape_train), np.mean(mape_test)]

prediction[1]

X_train.info()


errortestd=prediction-y_test

plt.figure()
plt.hist(errortestd)
plt.hist(errord)
plt.show()

errord = errord[np.isfinite(errord)]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = prediction)
# Plot the actual values
plt.plot(y_test, 'b-', label = 'actual')
# Plot the predicted values
plt.plot(prediction, 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels

plt.show()





from sklearn import tree
from dtreeviz.trees import dtreeviz # will be used for tree visualization
from matplotlib import pyplot as plt

plt.figure()
tree.plot_tree(rf2.estimators_[0], feature_names=X_train.columns, filled=True)
plt.show() #HUUUGE
plt.savefig('rf_individualtree2.png')
rf.estimators_[0].tree_.max_depth #21 max depth
rf.estimators_[10].tree_.max_depth #23
rf.estimators_[500].tree_.max_depth #29

from treeinterpreter import treeinterpreter as ti
instances = X_train.data[[1, 9]]
print "Instance 0 prediction:", rf.predict(instances[0])
print "Instance 1 prediction:", rf.predict(instances[1])
rediction, bias, contributions = ti.predict(rf, instances)

for i in range(len(instances)):
    print "Instance", i
    print "Bias (trainset mean)", biases[i]
    print "Feature contributions:"
    for c, feature in sorted(zip(contributions[i], 
                                 boston.feature_names), 
                             key=lambda x: -abs(x[0])):
        print feature, round(c, 2)
    print "-"*20 


#* SVM

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

svr_poly = svr_poly.fit(X_train,y_train)

prediction = svr_lin.predict(X_train)
msetrain = mean_squared_error(y_train, prediction)
mape = abs((y_train - prediction)/y_train)
mape = mape[np.isfinite(mape)]
accuracy = 100 - np.mean(mape)

prediction2 = svr_lin.predict(X_test)
msetest = mean_squared_error(y_test, prediction2)
mapetest = abs((y_test - prediction2)/y_test)
mapetest = mapetest[np.isfinite(mapetest)]
accuracytest = 100 - np.mean(mapetest)

svr[3] = [math.sqrt(msetrain), math.sqrt(msetest), np.mean(mape), np.mean(mapetest)]
svr=pd.DataFrame()
print([msetrain, msetest, accuracy, accuracytest, math.sqrt(msetrain), math.sqrt(msetest), np.mean(mape), np.mean(mapetest)])

acctabsecond[2] = [math.sqrt(msetrain), math.sqrt(msetest), np.mean(mape), np.mean(mapetest)]

svr_poly2 = SVR(kernel='poly', C=100, gamma='auto', degree=5, epsilon=.1,
               coef0=1)

accuracytable = pd.DataFrame()
k=1
for i in list([rf, svr_rbf ,svr_lin,svr_poly]):
    i.fit(X_train,y_train)
    prediction = i.predict(X_test)
    msetest = mean_squared_error(y_test,prediction)
    msetrain = mean_squared_error(y_train, i.predict(X_train))
    mape = abs((y_test - prediction)/y_test)
    mape = mape[np.isfinite(mape)]
    accuracy = 100 - np.mean(mape)
    predictions2 = i.predict(X_train)
    mape2 = abs((y_train - predictions2)/y_train)
    mape2 = mape2[np.isfinite(mape2)]
    accuracy2 = 100 - np.mean(mape2)
    accuracytable[k] = [msetrain, msetest, accuracy2, accuracy]
    k=k+1

rf.summary()
#* Neural net
seed(1)

y1 = np.array(df_rf.marketshare)
X = df_rf
X['frequency'] = X['frequency'].astype('category')
X = X.drop(columns=['marketshare','study','is_want'], axis=1)
X = sm.add_constant(X)
random.seed(987654321)
X_train, X_val, y_train, y_val = train_test_split(X, y1, test_size=0.3 , random_state=30)

X_train = sm.add_constant(X_train)
X_val = sm.add_constant(X_test)
y_val = np.array(y_test)
y_train = np.array(y_train)

y_train=np.reshape(y_train, (-1,1))
y_val=np.reshape(y_val, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)
print(scaler_x.fit(X_val))
xval_scale=scaler_x.transform(X_val)
print(scaler_y.fit(y_train))
ytrain_scale=scaler_y.transform(y_train)
print(scaler_y.fit(y_val))
yval_scale=scaler_y.transform(y_val)

#! load model in loaded_model
model = Sequential()
model.add(Dense(8, input_dim=11, kernel_initializer='normal', activation='relu'))
model.add(Dense(2066, activation='relu'))
model.add(Dense(1200, activation='sigmoid'))
model.add(Dense(1000, activation='linear'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

import keras.backend as K
def custom_loss(y_true, y_pred):
    # Normal MSE loss
    mse = K.mean(K.square(y_true-y_pred), axis=-1)
    # Loss that penalizes differences between sum(predictions) and sum(labels)
    idxs,vals = torch.unique(y_pred[:,0],return_counts=True)
    vs = torch.split_with_sizes(y_pred[:,1],tuple(vals))
    d = {k.item():v for k,v in zip(idxs,vs)}

    idxs2,vals2 = torch.unique(y_true[:,0],return_counts=True)
    vs2 = torch.split_with_sizes(y_true[:,1],tuple(vals2))
    d2 = {k.item():v for k,v in zip(idxs2,vs2)}

    sum_constraint = d - d2

    return(mse+sum_constraint)

import torch



sum_constraint=K.square(K.mean(y_pred, axis = -1) - K.mean(y_true, axis = -1))
sum_constraint = y_pred.groupby(['respondents','tasks','choices','products','competitors','frequency'])['marketshare'].sum() - y_true.groupby(['respondents','tasks','choices','products','competitors','frequency'])['marketshare'].sum()
    
idxs,vals = torch.unique(df_total['respondents'],return_counts=True)
vs = torch.split_with_sizes(df_total[:,1],tuple(vals))
d = {k.item():v for k,v in zip(idxs,vs)}

test = df_total.groupby(['respondents','tasks','choices','products','competitors','frequency'])['marketshare'].sum()



model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

tf.keras.models.save_model(
    model,
    './NNmodel2',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
#! go on below
loaded_model = load_model(
    './NNmodel2',
    custom_objects=None,
    compile=True
)
model = loaded_model_fitted

history = model.fit(xtrain_scale, ytrain_scale, epochs=50, batch_size=150, verbose=1, validation_split=0.2)
predictions4 = model.predict(xval_scale)
predictions = scaler_y.inverse_transform(predictions4)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


predictions2tr = model.predict(xtrain_scale)
predictionstr = scaler_y.inverse_transform(predictions2tr)
msetrain = mean_squared_error(y_train, predictionstr)

predictions2te = model.predict(xval_scale)
predictionste = scaler_y.inverse_transform(predictions2te)
msetest = mean_squared_error(y_val, predictionste)

mapetr = abs((y_train - predictionstr)/y_train)
mapetr = mapetr[np.isfinite(mapetr)]
accuracytr = 100 - np.mean(mapetr)

mapete = abs((y_val - predictionste)/y_val)
mapete = mapete[np.isfinite(mapete)]
accuracyte = 100 - np.mean(mapete)

acctabsecond
print([msetrain, msetest, accuracytr, accuracyte])
acctabsecond[3]=([msetrain, msetest, math.sqrt(msetrain), math.sqrt(msetest), np.mean(abs(predictionstr-y_train)),np.mean(abs(predictionste-y_val)), np.mean(mapetr), np.mean(mapete)])

acctabsecond.rename(columns={1:'Random Forest',2:'SVR-Polynomial',3:'Neural Network'}, inplace=True)
acctabsecond.rename(index={0:'MSE-In sample',1:'MSE-Out of sample',2:'mape-In Sample',3:'mape-Out of sample'}, inplace=True)
acctabsecond = acctabsecond.round(4)


accuracytable.rename(columns={1:'Random Forest',2:'SVR-RBF',3:'SVR-Linear',4:'SVR-Polynomial',5:'Neural Network'}, inplace=True)
accuracytable.rename(index={0:'MSE-In sample',1:'MSE-Out of sample',2:'Accuracy-In Sample',3:'Accuracy-Out of sample'}, inplace=True)
accuracytable = accuracytable.round(4)
accuracytable.to_csv('blackboxtab.csv')
accuracytable = pd.read_csv('blackboxtab.csv')

print(tabulate(accuracytable,headers=['Random Forest','SVR-RBF','SVR-Linear','SVR-Polynomial','Neural Network'], tablefmt="latex_booktabs"))

# * Black box + rf comparison: only MSE, MAE

# treeinterpreter

# dave the trees
filename = 'rf_bettersplit.sav'
pickle.dump(rf, open(filename, 'wb'))
 
# load the model from disk
loaded_rf = pickle.load(open(filename, 'rb'))
result = loaded_rf.score(X_test, y_test)

rf.score(X_test,y_test)

# this one is fitted! Above is NOT
tf.keras.models.save_model(
    model,
    './NNmodelfitbettersplit',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

loaded_model_fitted = load_model(
    './NNmodelfit2',
    custom_objects=None,
    compile=True
)
loaded_model_fitted.summary()



# ALTERNATIVE SPLIT
df_total.info()
df_rf = df_total.iloc[:,[3,4,18,19,20,21,22,10,11,12,13,14]]
df_rf = df_rf.dropna()
df_rf.isna().sum()
df_rf = df_rf[np.isfinite(df_rf)]
df_rf['frequency'] = df_rf['frequency'].astype('category')
random.seed(741852963)
#sc = StandardScaler()
#df_rf = pd.DataFrame(sc.fit_transform(df_rf))
#df_rf.rename(columns={0:'pref',1:'marketshare',2:'respondents',3:'tasks',4:'choices',5:'products', 6:'competitors',7:'frequency',8:'study',9:'price',10:'distri',11:'volume' }, inplace=True)
df_rf.rename(columns={'PrefSh_without':'pref','std_price':'price','std_distri':'distri','std_volume':'volume','respondenstmm':'respondents','tasksmm':'tasks','choicesmm':'choices','productsmm':'products','competitorsmm':'competitors'}, inplace=True)

studies = df_rf.study.unique()
train_stud, test_stud = train_test_split(studies, train_size=0.7)
train_df, test_df = df_rf.loc[df_rf['study'].isin(train_stud)], df_rf.loc[df_rf['study'].isin(test_stud)]

X_train = train_df.drop(columns=['marketshare','study'], axis=1)
y_train = train_df.marketshare
X_test = test_df.drop(columns=['marketshare','study'], axis=1)
y_test = test_df.marketshare

# Now run models as above