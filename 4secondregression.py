
df_reg2b.info()
df_reg2b = df_reg2b.dropna()
df_reg2b = df_reg2b[np.isfinite(df_reg2b)]
df_reg2b['frequency'] = df_reg2b['frequency'].astype('category')
freqdum = pd.get_dummies(df_reg2b['frequency'])
df = pd.concat([df_reg2b,freqdum[[1,2,3]]], axis=1)

df = df.drop(columns=['study'])
df = df.drop(columns=['frequency'])
df.info() # ! NB! In this df: 1.0=week. 2.0=month, 3.0=quarterly

df_y = abs(df['is_want'])
df_X = df.drop(columns=['difference', 'is_want'], axis=1)
df_X = add_constant(df_X)


X5 = add_constant(df)
#VIF CHECK
pd.Series([variance_inflation_factor(X5.values, i)
          for i in range(X5.shape[1])],
         index=X5.columns)


bptot = plt.boxplot(df_reg2b['is_want'])
plt.show()

plt.hist(df_reg2b['is_want'])
plt.show()




random.seed(1324)
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.3 , random_state=30)
totalreg = sm.OLS(df_y_train, df_X_train)
totalres = totalreg.fit()
print(totalres.summary().as_latex())

df_reg2.info()
df_reg2['study'] = df_reg2.loc['study'].astype('category')

plt.hist(df_reg2['difference'])
plt.show()
df_reg2.difference.min()

df_reg2 = df_reg2.loc[df_reg2['difference']!=-1]

df = df.dropna()
df = df[np.isfinite(df)]

df['is_want'].min()
df['is_want'].max()
df['is_want'].max()+ abs(df['is_want'].min()) #range is 0.625872
df['is_want'].mean()

y2 = df['is_want']
X3 = df[['std_price','std_distri','std_volume']]

#X3 = X3.drop(columns=['study','difference','totalabsdiff','is_want'], axis=1)
X2 = add_constant(X3)

random.seed(1234)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3 , random_state=30)
X3_train, X3_test, y2_train, y2_test = train_test_split(X3, y2, test_size=0.3 , random_state=30)




# Linear regression
lr = LinearRegression().fit(X3_train,y2_train)
lr.coef_

lrpred = lr.predict(X3_test)
print(pd.Series(lr.coef_, index = X3.columns)) # Print coefficients
print(mean_squared_error(y2_test, lrpred))     # Calculate the test MSE
math.sqrt(mean_squared_error(y2_test, lrpred))
print(r2_score(y2_test, lrpred))


# LIN REG USE
est = sm.OLS(y2_train, X2_train)
est2 = est.fit()
estsum = est2.summary()
print(est2.summary().as_latex())

linpred = est2.predict(X2_test)
math.sqrt(mean_squared_error(y2_test, linpred)) #0.01698895

pkl_lin1 = "linearmodel2.pkl"
with open(pkl_lin1, 'wb') as file:
    pickle.dump(est2, file)

# Load from file
with open("linearmodel1.pkl", 'rb') as file:
    linmod1 = pickle.load(file)

linmod1.summary()


# ABSOLUTE
y2 = abs(df['is_want'])
X3 = df[['std_price','std_distri','std_volume']]
X2 = add_constant(X3)

random.seed(1234)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3 , random_state=30)

est3 = sm.OLS(y2_train, X2_train)
est4 = est3.fit()
print(est4.summary().as_latex())

abspred = est4.predict(X2_test)
math.sqrt(mean_squared_error(y2_test, abspred)) #0.014671

# POS AND NEG SEP
dfpos = df.loc[(df['is_want'] >= 0)]
dfneg = df.loc[(df['is_want'] <= 0)]

dfpos['is_want'].max() - dfpos['is_want'].min() #range is 0.17403
abs(dfneg['is_want'].min()) - abs(dfneg['is_want'].max()) #range is 0.45182


# pos
y2 = dfpos['is_want']
X3 = dfpos[['std_price','std_distri','std_volume']]
X2 = add_constant(X3)
random.seed(1234)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3 , random_state=30)
estpos = sm.OLS(y2_train, X2_train)
estpos = estpos.fit()
print(estpos.summary().as_latex())
pospred= estpos.predict(X2_test)
math.sqrt(mean_squared_error(y2_test, pospred)) #0.01605050

# neg
y2 = (dfneg['is_want'])
X3 = dfneg[['std_price','std_distri','std_volume']]
X2 = add_constant(X3)
random.seed(1234)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3 , random_state=30)
estneg = sm.OLS(y2_train, X2_train)
estneg = estneg.fit()
print(estneg.summary().as_latex())

negpred = estneg.predict(X2_test)
math.sqrt(mean_squared_error(y2_test, negpred)) #0.0165468

# Classify to predict sign of difference
df['sign'] = np.where(df['is_want']>=0, 'Positive', 'Negative') #1=pos,0=neg

y2 = df['sign']
X3 = df[['std_price','std_distri','std_volume']]
X2 = add_constant(X3)

random.seed(1234)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3 , random_state=30)
X3_train, X3_test, y2_train, y2_test = train_test_split(X3, y2, test_size=0.3 , random_state=30)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

clf = clf.fit(X2_train,y2_train)
y_pred = clf.predict(X2_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y2_test, y_pred))
print(classification_report(y2_test, y_pred))


from sklearn import tree
from dtreeviz.trees import dtreeviz # will be used for tree visualization
from matplotlib import pyplot as plt
plt.figure()
tree.plot_tree(clf, feature_names=X2_train.columns, filled=True)
plt.show() 


dtree = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 10,  max_depth= 5)
dtree = dtree.fit(X2_train,y2_train)
y_pred = dtree.predict(X2_test)
print(confusion_matrix(y2_test, y_pred))
print(classification_report(y2_test, y_pred))

plt.figure()
tree.plot_tree(dtree, feature_names=X2_train.columns, class_names=['Positive','Negative'], filled=True)
plt.show() #think about if needed in thesis


text_representation = tree.export_text(dtree)
print(text_representation)


import dtreeviz


mod = smf.ols("difference ~ study:std_price + study:std_distri + study:std_volume", data=df_reg2)
res = mod.fit()
res.summary()

mod2 = smf.ols("difference ~ std_price + std_distri + std_volume + study:std_price + study:std_distri + study:std_volume", data=df_reg2b)
res2 = mod2.fit()
res2.summary()

df_reg2b['study'] = df_reg2b['study'].astype('category')
df_reg2b.info()


mod4 = smf.ols("is_want ~ frequency + respondenstmm + tasksmm+ choicesmm + productsmm+ competitorsmm+ std_price + std_distri + std_volume", data=df_reg2b)
res4 = mod4.fit()
res4.summary()

df_reg2b.info()

fig, ax = plt.subplots()
y = y2_train/y2_train.sum()
yhat = est2.predict(X2_train)
ax.scatter(yhat, y)
line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
abline_plot(model_results=line_fit, ax=ax)
ax.set_title('Model Fit Plot')
ax.set_ylabel('Observed values')
ax.set_xlabel('Fitted values');
#Text(0.5, 0, 'Fitted values')
plt.show()

fig, ax = plt.subplots()
ax.scatter(yhat, est2.resid_pearson)
ax.hlines(0, 0, 1)
ax.set_xlim(0, 1)
ax.set_title('Residual Dependence Plot')
ax.set_ylabel('Pearson Residuals')
ax.set_xlabel('Fitted values')
plt.show()

resid = est2.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
#Text(0.5, 1.0, 'Histogram of standardized deviance residuals')


graphics.gofplots.qqplot(resid, line='r')


conditions = [
    (df_reg2b.study == 1) | (df_reg2b.study == 2) |  (df_reg2b.study == 36),
    (df_reg2b.study == 3) | (df_reg2b.study == 5) |  (df_reg2b.study == 6) | (df_reg2b.study == 7) | (df_reg2b.study == 8) |  (df_reg2b.study == 14) | (df_reg2b.study == 26) | (df_reg2b.study == 27),
    (df_reg2b.study == 4) | (df_reg2b.study == 11),
    (df_reg2b.study == 9) | (df_reg2b.study == 10) |  (df_reg2b.study == 15) | (df_reg2b.study == 16) | (df_reg2b.study == 19) |  (df_reg2b.study == 24) | (df_reg2b.study == 25),
    (df_reg2b.study == 13) | (df_reg2b.study == 17) |  (df_reg2b.study == 20) | (df_reg2b.study == 28) | (df_reg2b.study == 30) |  (df_reg2b.study == 31),
    #(df_reg2b.study == 21),
    (df_reg2b.study == 29) | (df_reg2b.study == 37) | (df_reg2b.study == 32),
    (df_reg2b.study == 33) | (df_reg2b.study == 34)]
    #(df_reg2b.study ==35)]

values = ['beauty', 'sigarettes', 'snack', 'dairy', 'batteries', 'drink', 'home']

# create a list of the values we want to assign for each condition
values = ['beauty', 'sigarettes', 'snack', 'dairy', 'batteries', 'diapers', 'energy', 'beer', 'home','supps']

# create a new column and use np.select to assign values to it using our lists as arguments
df_reg2b['cat'] = np.select(conditions, values)

df_reg2b['cat'] = df_reg2b['cat'].astype('category')
df_reg2b['study'] = df_reg2b['study'].astype('category')
df_reg2b.info()

df_reg2b.loc[df_reg2b['cat']=='diapers']
df_fixedeff = df_reg2b.dropna()
freqdum = pd.get_dummies(df_fixedeff['frequency'])
df_fixedeff = pd.concat([df_fixedeff,freqdum[[2,3,4]]], axis=1) # freqdm[1] = daily

+ study:std_price + study:std_distri + study:std_volume
mod3 = smf.ols("is_want ~ C(frequency) + respondenstmm + tasksmm+ choicesmm + productsmm+ competitorsmm+ std_price + std_distri + std_volume + C(cat)", data=df_fixedeff)
res3 = mod3.fit()
res3.summary()
res3.coefs()



X=df_reg2b
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(interaction_only=True,include_bias = False)
study_price = pd.DataFrame(poly.fit_transform(X.iloc[:,[0,1]])[:,2])
study_distri = pd.DataFrame(poly.fit_transform(X.iloc[:,[0,2]])[:,2])
study_volume = pd.DataFrame(poly.fit_transform(X.iloc[:,[0,3]])[:,2])
X=X.merge(study_price, on=X.index)
X=X.drop(columns='key_0')
X=X.merge(study_distri, on=X.index)
X=X.drop(columns='key_0')
X=X.merge(study_volume, on=X.index)
X=X.drop(columns='key_0')
X.rename(columns={'0_x':'study:price','0_y':'study:distri',0:'study:volume'}, inplace=True)
.Series(dtype='float64')
stepwise_selection(X.iloc[:,[5,6,7,8,9,10,12,13,14]], X['is_want'])



from sklearn.feature_selection import SequentialFeatureSelector
sffs = SFS(LinearRegression(),
         k_features=(3,11),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_



# ! MOD6 as it's here, cannot be used as freq,resp etc are the same for each study. Interaction effects between study and these will not give any information
mod6 = smf.ols("is_want ~ study:freqdum[1] + study:freqdum[2] + study:freqdum[3] + study:respondenstmm + study:tasksmm+ study:choicesmm + study:productsmm+ study:competitorsmm+ study:std_price + study:std_distri + study:std_volume", data=df_reg2b)
res6 = mod6.fit()
res6.summary()
res6.pvalues


print(res3.summary())
print(res3.params)

results_as_html = res3.summary().tables[1].as_html()
tab = pd.read_html(results_as_html, header=0, index_col=0)[0]
tab[['coef','std err','P>|t|']]
print(tabulate(tab[['coef','std err','P>|t|']], tablefmt='latex_booktabs'))

# save the model
pkl_everythingres3 = "everythingres3.pkl"
with open(pkl_everythingres3, 'wb') as file:
    pickle.dump(res3, file)
with open("everythingres3.pkl", 'rb') as file:
    res3b = pickle.load(file)
