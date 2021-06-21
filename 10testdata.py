# RF IS CREATED IN 9B

df_test.info()
df_test = pd.read_csv('testdata.csv')
df_test = df_test.drop(columns=['Unnamed: 0','respondents','tasks','choices','products','competitors','is_want'])
df_test.rename(columns={'PrefSh_without':'pref','std_price':'price','std_distri':'distri','std_volume':'volume','respondenstmm':'respondents','tasksmm':'tasks','choicesmm':'choices','productsmm':'products','competitorsmm':'competitors'},inplace=True)

y = df_test['marketshare']
X = df_test.drop(columns=['TotalVolume','Price_per_unit','Distribution','marketshare','study'])
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X))
X.rename(columns={0:'pref',2:'price',3:'distri',4:'volume',5:'respondents', 6:'tasks',7:'choices',8:'products',9:'competitors',1:'frequency'}, inplace=True)


X.loc[1]
y.loc[1]
rf = loaded_rf

prediction = rf.predict(X)
prediction[0]
msetest = mean_squared_error(y,prediction)
rmse = math.sqrt(msetest)
mape_test = abs((y - prediction)/y)
mape_test = mape_test[np.isfinite(mape_test)]
mape_value = np.mean(mape_test)
accuracy = 100-np.mean(mape_test)
print([msetest,rmse,mape_value,accuracy])

df_total[['is_want','PrefSh_without','marketshare','std_price','std_distri','std_volume']].loc[[1]]

df_total[['totalabsdiff']].loc[[222]]


# SVM

svmprediction = svr_poly.predict(X)
svmmse = mean_squared_error(y,svmprediction)
svmrmse = math.sqrt(svmmse)
svmmape = abs((y - svmprediction)/y)
svmmape = svmmape[np.isfinite(svmmape)]
svmmape_value = np.mean(svmmape)
svmaccuracy = 100-np.mean(svmmape)
print([svmmse,svmrmse,svmmape_value,svmaccuracy])



# NN

seed(2)

y1 = np.array(y)
X1 = X
X1 = sm.add_constant(X1)

y1=np.reshape(y1, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X1))
x_scale=scaler_x.transform(X1)
print(scaler_y.fit(y1))
y_scale=scaler_y.transform(y1)

model = loaded_model_fitted
nnprediction = model.predict(x_scale)
nnprediction2 = scaler_y.inverse_transform(nnprediction)
nnmse = mean_squared_error(y1, nnprediction2)
nnmape = abs((y1 - nnprediction2)/y1)
nnmape = nnmape[np.isfinite(nnmape)]
nnacc = 100 - np.mean(nnmape)

np.mean(y1-nnprediction2)
error=y1-nnprediction2


predtab = pd.DataFrame()
predtab[1] = [msetest,rmse,np.mean(abs(y-prediction)),mape_value]
predtab[2] = [svmmse,svmrmse,np.mean(abs(y-svmprediction)),svmmape_value]
predtab[3] = [nnmse,math.sqrt(nnmse),np.mean(abs(y1-nnprediction)),np.mean(nnmape)]


predtab.rename(columns={1:'Random Forest',2:'SVM',3:'NN'}, inplace=True)
predtab.rename(index={0:'MSE',1:'RMSE',2:'MAD',3:'MAPE'}, inplace=True)
predtab = predtab.round(4)
predtab.to_csv('predtab3.csv')
predtabcsv = pd.read_csv('predtab3.csv')

print(tabulate(predtabcsv,headers=['Random Forest','Neural Network'], tablefmt="latex_booktabs"))



prediction
y

error=prediction-y
errord = prediction/y
plt.figure()

#labels1 = df_reg1.iloc[:,2:].columns.values
bp1 = plt.hist(error)
#print(get_box_plot_data(labels1, bp1))
plt.show()

X.to_csv("xvars.csv")
y.to_csv("actualy.csv")
pd.DataFrame(nnprediction2).to_csv("nnpredictconstraint.csv")