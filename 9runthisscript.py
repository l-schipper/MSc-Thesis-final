#* Dependencies functions
execfile(open('a0dependencies.py').read())
import a0dependencies


#* Load dataframes (fully prepped)
df_total.info()
df_total = pd.read_csv('df_total_5.csv')
df_total = df_total.drop(columns=['Unnamed: 0'])
df_total = df_total[np.isfinite(df_total['difference'])]
df_reg2 = df_total.iloc[:,[11,12,13,14,16,23]]       # study, stdprice, stddistri, stdvolume, difference, totalabsdiff
df_reg1 = df_total.iloc[:,[11,10,18,19,20,21,22,23]]    # study, freq, respondentsmm, tasksmm, choicesmm, productsmm, competitiorsmm, totalabsdiff
df_reg2b = df_total.iloc[:,[11,12,13,14,16,10,18,19,20,21,22,23,24]]

#* VIF regressions
execfile('2vif.py')

#* Shrinkage regressions

# Load from file
with open("pickle_ridge.pkl", 'rb') as file:
    ridgecv = pickle.load(file)

with open("pickle_lasso.pkl", 'rb') as file:
    lassocv = pickle.load(file)

with open("pickle_elastic.pkl", 'rb') as file:
    elasticcv = pickle.load(file)

# Deleting duplicated, left with 1 obs per study
df_test = df_reg1
df_test = df_test.set_index(df_test.study)
df_test = df_test.drop_duplicates(subset='study', keep='first')
df_reg1study = df_test
df_reg1study = df_reg1study.drop(columns=['study'])
df_reg1study['totalabs100'] = df_reg1study.totalabsdiff*100
df_reg1study = df_reg1study.dropna()
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

# Alpha plot
alphas = 10**np.linspace( 10,-5,100)
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
plt.legend(lassoplot_index, prop={'size': 14})
plt.show()

# Total table of all shrinkage regressions
coefs3 = np.append(ridgecv.intercept_, ridgecv.coef_)
add1 = ridgecv.alpha_, ridgecv.score(X_train,y_train)
add2 = abs(ridgecv.cv_values_.mean()), abs(mean_squared_error(y_test,ridgecv.predict(X_test)))
coefs4 = np.append(coefs3,add1)
coefs5 = np.append(coefs4,add2)

lcoefs3 = np.append(lassocv.intercept_, lassocv.coef_)
ladd1 = lassocv.alpha_, lassocv.score(X_train,y_train)
ladd2 = abs(lassocv.mse_path_.mean()), abs(mean_squared_error(y_test,lassocv.predict(X_test)))
lcoefs4 = np.append(lcoefs3,ladd1)
lcoefs5 = np.append(lcoefs4,ladd2)

encoefs3 = np.append(elasticcv.intercept_, elasticcv.coef_)
enadd1 = elasticcv.alpha_, elasticcv.score(X_train,y_train)
enadd2 = abs(elasticcv.mse_path_.mean()), abs(mean_squared_error(y_test,elasticcv.predict(X_test)))
encoefs4 = np.append(encoefs3,enadd1)
encoefs5 = np.append(encoefs4,enadd2)

index = ("Intercept","No. respondents","No. tasks","No. SKU's per task","No. SKU's in market", "No. competitors", "Frequency: weekly", "Frequency: monthly", "Frequency: quarterly","Alpha","R^2","MSE - in sample", "MSE - out of sample")
totaltabs = zip(index, coefs5, lcoefs5, encoefs5)
print(tabulate(totaltabs, headers=("Ridge","Lasso","Elastic Net"), tablefmt="latex_booktabs"))

# Relationship tasks and difference to explain regression coef
# #! > fixed
plot = df_total[['study','tasks','totalabsdiff']]
plot = plot.set_index(plot.study)
plot = plot.drop_duplicates(subset='study', keep='first')
plot = plot.dropna()
plot = plot[np.isfinite(plot)]

# ALL OBSERVATIONS
x = plot['tasks']
y = abs(plot.totalabsdiff)


ymean = plot.loc[plot['tasks']==6.0].mean()[2]
ymean2 = plot.loc[plot['tasks']==12.0].mean()[2]
ymean3 = plot.loc[plot['tasks']==13.0].mean()[2]
ymean4 = plot.loc[plot['tasks']==15.0].mean()[2]
ymean5 = plot.loc[plot['tasks']==16.0].mean()[2]
ymeandf=np.array([ymean,ymean2,ymean3,ymean4,ymean5])
xmean = np.array([6.0,12.0,13.0,15.0,16.0])

ymin = plot.loc[plot['tasks']==6.0].min()[2]
ymin2 = plot.loc[plot['tasks']==12.0].min()[2]
ymin3 = plot.loc[plot['tasks']==13.0].min()[2]
ymin4 = plot.loc[plot['tasks']==15.0].min()[2]
ymin5 = plot.loc[plot['tasks']==16.0].min()[2]
ymindf=np.array([ymin,ymin2,ymin3,ymin4,ymin5])
xmin = np.array([6.0,12.0,13.0,15.0,16.0])

ymax = plot.loc[plot['tasks']==6.0].max()[2]
ymax2 = plot.loc[plot['tasks']==12.0].max()[2]
ymax3 = plot.loc[plot['tasks']==13.0].max()[2]
ymax4 = plot.loc[plot['tasks']==15.0].max()[2]
ymax5 = plot.loc[plot['tasks']==16.0].max()[2]
ymaxdf=np.array([ymax,ymax2,ymax3,ymax4,ymax5])
xmax = np.array([6.0,12.0,13.0,15.0,16.0])

ymed = plot.loc[plot['tasks']==6.0].median()[2]
ymed2 = plot.loc[plot['tasks']==12.0].median()[2]
ymed3 = plot.loc[plot['tasks']==13.0].median()[2]
ymed4 = plot.loc[plot['tasks']==15.0].median()[2]
ymed5 = plot.loc[plot['tasks']==16.0].median()[2]
ymeddf=np.array([ymed,ymed2,ymed3,ymed4,ymed5])
xmed = np.array([6.0,12.0,13.0,15.0,16.0])


fig, (axs1) = plt.subplots(1)
axs1.plot(x, y, 'o')
m, b = np.polyfit(x,y, 1)
axs1.plot(x, m*x + b,label="Linear relation", color="green")
x_new = np.linspace(x.min(), x.max(),500)
f = interp1d(xmean, ymeandf, kind='quadratic')
y_smooth=f(x_new)
axs1.plot(x_new, y_smooth, color="lightcoral", label="Interpolated Mean")
axs1.plot()
f2 = interp1d(xmin, ymindf, kind='quadratic')
y_smooth=f2(x_new)
axs1.plot(x_new, y_smooth, color="maroon", label="Interpolated Minimum")
axs1.plot()
f3 = interp1d(xmax, ymaxdf, kind='quadratic')
y_smooth=f3(x_new)
axs1.plot(x_new, y_smooth, color="orangered", label="Interpolated Maximum")
axs1.plot()
f4 = interp1d(xmed, ymeddf, kind='quadratic')
y_smooth=f4(x_new)
axs1.plot(x_new, y_smooth, color="red", label="Interpolated Median")
axs1.plot()

plt.axis('tight')
plt.xlabel('Number of tasks', fontsize=14)
plt.ylabel('Magnitude of difference', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(prop={'size': 14})
plt.show()

# EXCLUSING OUTLIER
plot = df_total[df_total['totalabsdiff']<1.4]
plot = plot[['study','tasks','totalabsdiff']]
plot = plot.set_index(plot.study)
plot = plot.drop_duplicates(subset='study', keep='first')
plot = plot.dropna()
plot = plot[np.isfinite(plot)]

x = plot['tasks']
y = abs(plot.totalabsdiff)
#create plots with the code above


outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
        
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers = detect_outlier(plot['totalabsdiff'])
noout = plot[~plot['totalabsdiff'].isin(plot)]

plot = noout
plot = plot.set_index(plot.study)
plot = plot.drop_duplicates(subset='study', keep='first')
plot = plot.dropna()
plot = plot[np.isfinite(plot)]

x = plot['tasks']
y = abs(plot.totalabsdiff)


#outlier fucntion gives same as just dropping diff>1.4



#* Second regression

df_reg2b = df_reg2b.dropna()
df_reg2b = df_reg2b[np.isfinite(df_reg2b)]
df_reg2b['frequency'] = df_reg2b['frequency'].astype('category')
freqdum = pd.get_dummies(df_reg2b['frequency'])
df = pd.concat([df_reg2b,freqdum[[1,2,3]]], axis=1)
df['study'] = df['study'].astype('category')
df.info()

mod3 = smf.ols("is_want ~ freqdum[1] + freqdum[2] + freqdum[3] + respondenstmm + tasksmm+ choicesmm + productsmm+ competitorsmm+ study:std_price + study:std_distri + study:std_volume", data=df)
res3 = mod3.fit()
res3.summary()
res3.pvalues



results_as_html = res3.summary().tables[1].as_html()
tab = pd.read_html(results_as_html, header=0, index_col=0)[0]
tab[['coef','std err','P>|t|']]
print(tabulate(tab[['coef','std err','P>|t|']], tablefmt='latex_booktabs'))

#* Clustering results
df_cluster = pd.read_csv('df_cluster.csv')
df_cluster = df_cluster.set_index('key_0')

cluster_X = df_cluster.iloc[:,6:12]
zsigni_X = df_cluster.iloc[:,[6,8,10]]
cluster3 = df_cluster.iloc[:,:6]


model= KMeans()
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(cluster_X)
visualizer.show()  

kmeans = KMeans(n_clusters=10, random_state=0)
labels = pd.DataFrame(kmeans.fit_predict(cluster_X))
study = pd.DataFrame(cluster_X.index.values)
kmeansout = pd.concat([study,labels], axis=1)
kmeansout = kmeansout.set_index(kmeansout.iloc[:,0])

means = pd.DataFrame(kmeansout[0].values, index=kmeansout.index)
means = means.drop(columns=[0])
donecluster = means
donecluster.rename(columns={0:'study',1:'KMeans+sig'}, inplace=True) # check this

plotdf = cluster_X.merge(donecluster, how='outer',left_on=cluster_X.index, right_on=donecluster.index)
plotdf.rename(columns={'key_0':'study'}, inplace=True)
plotdf = plotdf.set_index('study')

#* VISUALIZE CLUSTERS
x= plotdf.iloc[:,:6]

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf = principalDf.set_index(plotdf.index)
#finalDf = pd.concat([principalDf, plotdf.iloc[:,6]], axis = 1)
finalDf = principalDf.merge(plotdf.iloc[:,6], how='outer',left_on=principalDf.index,right_on=plotdf.index)
finalDf['KMeans+sig'].round(0)
colors = ['#DF2020', '#81DF20', '#2095DF','blue','red','yellow','black','purple','orange','green']
targets=[0,1,2,3,4,5,6,7,8,9]

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['KMeans+sig'] == target
    scatter = ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50,
               label= 'Cluster '+ str(int(target+1)))
    scatter2 = ax.scatter(finalDf.loc[finalDf['KMeans+sig']==target].mean()[[1]]
               , finalDf.loc[finalDf['KMeans+sig']==target].mean()[[2]]
               , c = color
               , s = 50*4
               , marker="^",
               label = None)
leg1= ax.legend(scatter.legend_elements(),labels=['Observations','Cluster Means'],loc="lower right", frameon=True,prop={'size': 14})
leg2 = ax.legend(loc='upper right', frameon=True,prop={'size': 14})
ax.add_artist(leg1)
ax.add_artist(leg2)
leg1.get_frame().set_edgecolor('black')
leg2.get_frame().set_edgecolor('black')
leg1.legendHandles[0]._sizes = [50]
leg1.legendHandles[1]._sizes = [50]
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid()
plt.show()

pca.explained_variance_ratio_ 

# switch to tensorenvironment and to 6,7,9b.py script