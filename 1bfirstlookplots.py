import seaborn as sns
# Create the default pairplot
sns.pairplot(df_total[['respondents','tasks','choices','products','competitors','frequency','study','std_price','std_distri','std_volume','is_want']])
plt.show()

df_total.info()

x=df_total.respondents
y=df_total.totalabsdiff
fig, (axs1) = plt.subplots(1)
axs1.plot(x, y, 'o')
plt.axis('tight')
plt.xlabel('Number of resp', fontsize=14)
plt.ylabel('Magnitude of difference', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(prop={'size': 14})
plt.show()

# * Respondents and competitors and frequency
fig, axs = plt.subplots(1,3)
# plot
axs[0,0].plot(df_total.respondents,df_total.is_want, 'o')
axs[1,0].plot(df_total.respondents,df_total.totalabsdiff, 'o')
axs[0,1].plot(df_total.competitors,df_total.is_want, 'o')
axs[1,1].plot(df_total.competitors,df_total.totalabsdiff, 'o')
axs[0,2].plot(df_total.frequency,df_total.is_want, 'o')
axs[1,2].plot(df_total.frequency,df_total.totalabsdiff, 'o')
# set labels right
axs[0,0].set_ylabel('Difference', fontsize=14)
axs[1,0].set_ylabel('Magnitude of absolute difference', fontsize=14)
axs[1,0].set_xlabel('No. respondents', fontsize=14)
axs[1,1].set_xlabel('No. competitors', fontsize=14)
axs[1,2].set_xlabel('Frequency', fontsize=14)
# give caption
axs[0,0].title.set_text('a')
axs[0,1].title.set_text('b')
axs[1,0].title.set_text('d')
axs[1,1].title.set_text('e')
axs[0,2].title.set_text('c')
axs[1,2].title.set_text('f')
# looks
plt.axis('tight')
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.show()


# * Tasks,choices, products
fig, axs = plt.subplots(2,3)
# plot
axs[0,0].plot(df_total.tasks,df_total.is_want, 'o')
axs[1,0].plot(df_total.tasks,df_total.totalabsdiff, 'o')
axs[0,1].plot(df_total.choices,df_total.is_want, 'o')
axs[1,1].plot(df_total.choices,df_total.totalabsdiff, 'o')
axs[0,2].plot(df_total.products,df_total.is_want, 'o')
axs[1,2].plot(df_total.products,df_total.totalabsdiff, 'o')
# set labels right
axs[0,0].set_ylabel('Difference', fontsize=14)
axs[1,0].set_ylabel('Magnitude of absolute difference', fontsize=14)
axs[1,0].set_xlabel('No. tasks', fontsize=14)
axs[1,1].set_xlabel("No. SKU's per task", fontsize=14)
axs[1,2].set_xlabel("No. SKU's in market", fontsize=14)
# give caption
axs[0,0].title.set_text('a')
axs[0,1].title.set_text('b')
axs[1,0].title.set_text('d')
axs[1,1].title.set_text('e')
axs[0,2].title.set_text('c')
axs[1,2].title.set_text('f')
# looks
plt.axis('tight')
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.show()


# * PRICE DISTRI VOLUME
fig, axs = plt.subplots(1,3)
# plot
axs[0].plot(df_total.std_price,df_total.is_want, 'o')
axs[1].plot(df_total.std_volume,df_total.is_want, 'o')
axs[2].plot(df_total.std_distri,df_total.is_want, 'o')
# set labels right
axs[0].set_ylabel('Difference', fontsize=14)
axs[0].set_xlabel('Standardized Price', fontsize=14)
axs[1].set_xlabel("Standardized Volume", fontsize=14)
axs[2].set_xlabel("Standardized Distribution", fontsize=14)
# give caption
axs[0].title.set_text('a')
axs[1].title.set_text('b')
axs[2].title.set_text('c')
# looks
plt.axis('tight')
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.show()





x=df_total.frequency
y=df_total.totalabsdiff
fig, (axs1) = plt.subplots(1)
axs1.plot(x, y, 'o')
plt.axis('tight')
plt.xlabel('Number of resp', fontsize=14)
plt.ylabel('difference', fontsize=14)
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.legend(prop={'size': 14})
plt.show()

