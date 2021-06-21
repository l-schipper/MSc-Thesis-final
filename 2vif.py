#####################################################################
################### VARIANCE INFLATION FACTOR #######################
#####################################################################

# REG1
df_reg1 = df_reg1.dropna()
df_reg2 = df_reg2.dropna()

df_reg1a = add_constant(df_reg1)
df_reg1a = df_reg1a.drop(['study','totalabsdiff'], axis=1)
#VIF
VIFreg1 = pd.Series([variance_inflation_factor(df_reg1a.values, i)
for i in range(df_reg1a.shape[1])],
index=df_reg1a.columns)

reg1_index = ("Constant","Frequency","Number of respondents","Number of choice tasks","Number of SKU's per task","Number of SKU's in market","Number of competitors")

VIFreg1tab = zip(reg1_index,VIFreg1)
print(tabulate(VIFreg1tab, tablefmt="latex_booktabs"))
# * Lot of multicol > shrinkage method, regression
#  ? ridge/lasso/elastic net

# REG2
X= add_constant(df_reg2)
#VIF
pd.Series([variance_inflation_factor(X.values, i)
          for i in range(X.shape[1])],
         index=X.columns)
# * No multicol > regular regression

# FULL REG2B
df_reg2b2 = df_reg2b.drop(columns='study')
X = add_constant(df_reg2b2)
#VIF
pd.Series([variance_inflation_factor(X.values, i)
          for i in range(X.shape[1])],
         index=X.columns)
