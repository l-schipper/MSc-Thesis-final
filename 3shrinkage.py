#####################################################################
###################### SHRINKAGE REGRESSION #########################
#####################################################################

# First look at data
corrMatrix1 = df_reg1.iloc[:,2:].corr()
print(corrMatrix1)
sns.heatmap(corrMatrix1, annot=True)
plt.show()

labels1 = df_reg1.iloc[:,2:].columns.values
bp1 = plt.boxplot(df_reg1.iloc[:,2:], labels=labels1)
print(get_box_plot_data(labels1, bp1))
plt.show()

# Deleting duplicated, left with 1 obs per study
df_test = df_reg1
df_test = df_test.set_index(df_test.study)
df_test = df_test.drop_duplicates(subset='study', keep='first')
df_reg1study = df_test
df_reg1study = df_reg1study.drop(columns=['study'])
df_reg1study['totalabs100'] = df_reg1study.totalabsdiff*100

# Define X and y + train and test sets
y = df_reg1study.totalabsdiff
X = df_reg1study
X['frequency'] = X['frequency'].astype('category')
dummies = pd.get_dummies(X['frequency'])
X = pd.concat([X,dummies], axis=1)
X = X.drop(columns=['totalabsdiff','totalabs100','frequency'], axis=1)
X = X.drop(X.columns[5], axis=1)
X.info()
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1 , random_state=30)
np.shape(X_train)
np.shape(X_test) 
# Train set N=29; test N=4

######################################
################ RIDGE ###############
######################################
alphas = 10**np.linspace( 10,-5,100)
ridge = Ridge(normalize=False)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

ridgeplot_index = ("No. respondents","No. tasks","No. SKU's per task","No. SKU's in market","No. competitors","Frequency: weekly","Frequency: monthly","Frequency: quarterly")    
np.shape(coefs)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(right=10**5)
plt.vlines(x=4.03701725859655, colors='grey', ls=':', ymin=-0.45, ymax=0.5, label="Cross-validated generalization parameter")
#plt.axis('tight')
plt.xlabel('Generalization parameter, lambda')
plt.ylabel('weights')
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.legend(ridgeplot_index, prop={'size': 14})
plt.show()

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', store_cv_values=True)
ridgecv.fit(X_train,y_train)
ridgecv.alpha_
np.shape(ridgecv.cv_values_)
np.shape(y_train)
ridgecv.cv_values_.mean() #MSE on training set
ridgecv.score(X_train,y_train)
ridgecv.coef_
ridgecv.intercept_

ypred = ridgecv.predict(X_test)
mean_squared_error(y_test,ypred) # MSE 0.035

# Create table
coefs3 = np.append(ridgecv.intercept_, ridgecv.coef_)
add1 = ridgecv.alpha_, ridgecv.score(X_train,y_train)
add2 = abs(ridgecv.cv_values_.mean()), abs(mean_squared_error(y_test,ypred))
coefs4 = np.append(coefs3,add1)
coefs5 = np.append(coefs4,add2)
index = ("Intercept","No. respondents","No. tasks","No. SKU's per task","No. SKU's in market", "No. competitors", "Frequency: weekly", "Frequency: monthly", "Frequency: quarterly","Alpha","R^2","MSE - in sample", "MSE - out of sample")

tabs = zip(index, coefs5)
print(tabulate(tabs, headers=("Variable","Coefficient"), tablefmt="latex_booktabs"))


######################################
################ LASSO ###############
######################################
lasso = Lasso(max_iter = 10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)


lassoplot_index = ("No. respondents","No. tasks","No. SKU's per task","No. SKU's in market","No. competitors","Frequency: weekly","Frequency: monthly","Frequency: quarterly")    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
ax.set_xlim(right=10**1.5)
plt.vlines(x=0.00756463, colors='grey', ls=':', ymin=-0.5, ymax=0.62, label="Cross-validated generalization parameter")
#plt.axis('tight')
plt.xlabel('Generalization parameter, lambda', fontsize=14)
plt.ylabel('Weights', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(ridgeplot_index, prop={'size': 14})
plt.show()

lassocv = LassoCV(alphas = alphas, max_iter = 100000)
lassocv.fit(X_train, y_train)
lassocv.alpha_
lassocv.mse_path_
lassocv.mse_path_.mean()
lassocv.score(X_train,y_train)
lassocv.coef_
lassocv.intercept_

mean_squared_error(y_test, lasso.predict(X_test))
print(r2_score(y_test, lasso.predict(X_test)))

# Create table
lcoefs3 = np.append(lassocv.intercept_, lassocv.coef_)
ladd1 = lassocv.alpha_, lassocv.score(X_train,y_train)
ladd2 = abs(lassocv.mse_path_.mean()), abs(mean_squared_error(y_test,lasso.predict(X_test)))
lcoefs4 = np.append(lcoefs3,ladd1)
lcoefs5 = np.append(lcoefs4,ladd2)
index = ("Intercept","No. respondents","No. tasks","No. SKU's per task","No. SKU's in market", "No. competitors", "Frequency: weekly", "Frequency: monthly", "Frequency: quarterly","Alpha","R^2","MSE - in sample", "MSE - out of sample")

ltabs = zip(index, lcoefs5)
print(tabulate(ltabs, headers=("Variable","Coefficient"), tablefmt="latex_booktabs"))


######################################
############# ELASTIC NET ############
######################################
elastic = ElasticNet(max_iter = 10000)
coefs = []
#l1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for a in alphas:
    elastic.set_params(alpha=a)
    elastic.fit(X_train, y_train)
    coefs.append(elastic.coef_)
    
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

elasticcv = ElasticNetCV(alphas = alphas, max_iter = 100000)
elasticcv.fit(X_train, y_train)
elasticcv.alpha_
elasticcv.mse_path_
elasticcv.mse_path_.mean()
elasticcv.score(X_train,y_train)
elasticcv.coef_
elasticcv.intercept_
elasticcv.l1_ratio_

mean_squared_error(y_test, elasticcv.predict(X_test))
print(r2_score(y_test, elasticcv.predict(X_test)))

# Create table
encoefs3 = np.append(elasticcv.intercept_, elasticcv.coef_)
enadd1 = elasticcv.alpha_, elasticcv.score(X_train,y_train)
enadd2 = abs(elasticcv.mse_path_.mean()), abs(mean_squared_error(y_test,elasticcv.predict(X_test)))
encoefs4 = np.append(encoefs3,enadd1)
encoefs5 = np.append(encoefs4,enadd2)
index = ("Intercept","No. respondents","No. tasks","No. SKU's per task","No. SKU's in market", "No. competitors", "Frequency: weekly", "Frequency: monthly", "Frequency: quarterly","Alpha","R^2","MSE - in sample", "MSE - out of sample")

tabs = zip(index, encoefs5)
print(tabulate(tabs, headers=("Variable","Coefficient"), tablefmt="latex_booktabs"))

# Add linear regression to see changes
lr = LinearRegression().fit(X_train,y_train)
lr.intercept_
lr.coef_
lrpred = lr.predict(X_test)
print(pd.Series(lr.coef_, index = index)) # Print coefficients
print(mean_squared_error(y2_test, lrpred))           # Calculate the test MSE
print(r2_score(y_test, lrpred))
lrcombine = np.append(lr.intercept_, lr.coef_)
lradd1 = "", lr.score(X_train,y_train)
lradd2 = mean_squared_error(y_train, lr.predict(X_train)), mean_squared_error(y_test, lrpred)
lrcombine2 = np.append(lrcombine, lradd1)
lrcombine3 = np.append(lrcombine2, lradd2)
tabs = zip(index, lrcombine3)
print(tabulate(tabs, headers=("Variable","Coefficient"), tablefmt="latex_booktabs"))

######################################
################ TABLE ###############
######################################
totaltabs = zip(index, coefs5, lcoefs5, encoefs5, lrcombine3)
print(tabulate(totaltabs, headers=("Ridge","Lasso","Elastic Net","Linear"), tablefmt="latex_booktabs"))


# Save to file in the current working directory
pkl_ridge = "pickle_ridge.pkl"
with open(pkl_ridge, 'wb') as file:
    pickle.dump(ridgecv, file)
pkl_lasso = "pickle_lasso.pkl"
with open(pkl_lasso, 'wb') as file:
    pickle.dump(lassocv, file)
pkl_elastic = "pickle_elastic.pkl"
with open(pkl_elastic, 'wb') as file:
    pickle.dump(elasticcv, file)


# * Relationship tasks and difference to explain regression coef
x= df_total.tasks
x = x.dropna()
x = x[np.isfinite(x)]
y = abs(df_total.totalabsdiff)
y= y.dropna()
y= y[np.isfinite(y)]

fig, (axs1) = plt.subplots(1)
axs1.plot(x, y, 'o')
m, b = np.polyfit(x,y, 1)
axs1.plot(x, m*x + b,label="Linear relation", color="green")
x_new = np.linspace(x.min(), x.max(),500)
f = interp1d(x, y, kind='quadratic')
y_smooth=f(x_new)
axs1.plot(x_new, y_smooth, color="red", label="Quadratic relation")

#scatterplot_index = ("Observations","Linear relation","Quadratic relation")    
plt.axis('tight')
plt.xlabel('Number of tasks', fontsize=14)
plt.ylabel('Magnitude of difference', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(prop={'size': 14})
plt.show()


# exclude the outlier! 
x= df_total.tasks[df_total['totalabsdiff']<1.4]
x = x.dropna()
x = x[np.isfinite(x)]
y = abs(df_total.totalabsdiff[df_total['totalabsdiff']<1.4])
y= y.dropna()
y= y[np.isfinite(y)]

fig, (axs1) = plt.subplots(1)
axs1.plot(x, y, 'o')
m, b = np.polyfit(x,y, 1)
axs1.plot(x, m*x + b,label="Linear relation", color="green")
x_new = np.linspace(x.min(), x.max(),500)
f = interp1d(x, y, kind='quadratic')
y_smooth=f(x_new)
axs1.plot(x_new, y_smooth, color="red", label="Quadratic relation")

#scatterplot_index = ("Observations","Linear relation","Quadratic relation")    
plt.axis('tight')
plt.xlabel('Number of tasks', fontsize=14)
plt.ylabel('Magnitude of difference', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(prop={'size': 14})
plt.show()

# ! APPENDIX! PLOTS WITH MEDIAN AND MEAN
fig, (axs1) = plt.subplots(1)
x= df_total.tasks
x = x.dropna()
x = x[np.isfinite(x)]
y = abs(df_total.totalabsdiff)
y= y.dropna()
y= y[np.isfinite(y)]

fig,(axs1) = plt.subplots(1)
axs1.plot(x, y, 'o')
m, b = np.polyfit(x,y, 1)
axs1.plot(x, m*x + b,label="Linear relation", color="green")
#x_new = np.linspace(x.min(), x.max(),500)
#f = interp1d(x, y, kind='quadratic')
#y_smooth=f(x_new)
#axs1.plot(x_new, y_smooth, color="red", label="Quadratic relation - all datapoints")

x2=np.array([6,12,13,15,16])
y2=np.array([df_total.totalabsdiff[df_total['tasks']==6].mean(),df_total.totalabsdiff[df_total['tasks']==12].mean(),df_total.totalabsdiff[df_total['tasks']==13].mean(),df_total.totalabsdiff[df_total['tasks']==15].mean(),df_total.totalabsdiff[df_total['tasks']==16 ].mean()])
x2_new = np.linspace(x2.min(), x2.max(),500)
f2 = interp1d(x2, y2, kind='quadratic')
y2_smooth=f(x2_new)

axs1.scatter(x2,y2)
axs1.plot(x2_new,y2_smooth, linewidth=2, label="Quadratic relation - means")

plt.axis('tight')
plt.xlabel('Number of tasks', fontsize=14)
plt.ylabel('Magnitude of difference', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(prop={'size': 14})
plt.show()


fig, (axs1) = plt.subplots(1)
x3=np.array([6,12,13,15,16])
y3=np.array([df_total.totalabsdiff[df_total['tasks']==6].median(),df_total.totalabsdiff[df_total['tasks']==12].median(),df_total.totalabsdiff[df_total['tasks']==13].median(),df_total.totalabsdiff[df_total['tasks']==15].median(),df_total.totalabsdiff[df_total['tasks']==16 ].median()])
x3_new = np.linspace(x3.min(), x3.max(),500)
f3 = interp1d(x3, y3, kind='quadratic')
y3_smooth=f(x3_new)
axs1.plot(x3,y3, 'x',color="red")
axs1.plot(x, y, 'o')

axs1.plot(x3_new,y3_smooth,linewidth=2, label="Quadratic relation - medians")
axs1.plot(x, m*x + b,label="Linear relation", color="green")
plt.axis('tight')
plt.xlabel('Number of tasks', fontsize=14)
plt.ylabel('Magnitude of difference', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(prop={'size': 14})
plt.show()

