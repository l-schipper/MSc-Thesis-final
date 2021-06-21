######################################
########### NEURAL NETWORK ###########
######################################
# basic
y = np.array(df_rf.is_want)
X = df_rf
X['frequency'] = X['frequency'].astype('category')
X = X.drop(columns=['PrefSh_without','marketshare','study','is_want'], axis=1)
X = sc.fit_transform(X)
X.isna().sum()
random.seed(654321)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=30)

# Neural network
model = Sequential()
model.add(Dense(16, input_dim=9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs=150, batch_size=10)

model2 = Sequential()
model2.add(Dense(2, input_dim=10))
model2.add(Dense(1))
model2.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])
# train model
history = model2.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)
y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()



# USED MODEL

seed(1)
tensorflow.random.set_seed(1)

y1 = np.array(df_rf.is_want)
X = df_rf
X['frequency'] = X['frequency'].astype('category')
X = X.drop(columns=['PrefSh_without','marketshare','study','is_want'], axis=1)
X_back = X
X = sm.add_constant(X)
x1 = np.column_stack(('const','respondents','tasks','choices','products','competitors','frequency','std_price','std_distri','std_volume'))
#x1 = sm.add_constant(x1)

X_train, X_val, y_train, y_val = train_test_split(X, y1)

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

# Layer sizes
# Input: Number of features in the training set + 1 > dont count the constant as a feature! So here it's 10 variables + 1 constant, first layer input=10
# Hidden: Training Data Samples/Factor * (Input Neurons + Output Neurons)
# Output:  As this is the result layer, the output layer takes a value of 1 by default.

model = Sequential()
model.add(Dense(8, input_dim=10, kernel_initializer='normal', activation='relu'))
model.add(Dense(2066, activation='relu'))
model.add(Dense(1200, activation='sigmoid'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(xtrain_scale, ytrain_scale, epochs=50, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(xval_scale)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = scaler_y.inverse_transform(predictions)
predictions

mean_absolute_error(y_val, predictions)
mean_squared_error(y_val, predictions)
np.mean(y_val)
np.mean(predictions)

math.sqrt(mean_squared_error(y_val, predictions))

tf.keras.models.save_model(
    model,
    './NNmodel',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

from tensorflow.keras.models import Sequential, save_model, load_model
# Load the model
loaded_model = load_model(
    './NNmodel',
    custom_objects=None,
    compile=True
)

fitted = loaded_model.fit(xtrain_scale, ytrain_scale, epochs=50, batch_size=150, verbose=1, validation_split=0.2)
predictions2 = loaded_model.predict(xval_scale)

print(fitted.history.keys())
# "Loss"
plt.plot(fitted.history['loss'])
plt.plot(fitted.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

