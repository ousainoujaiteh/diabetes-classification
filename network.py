from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
import numpy
import os
numpy.random.seed(7)

# DATASET

train_dataset = numpy.loadtxt("pima-indians-diabetes.train_data.txt", delimiter=",")
X = train_dataset[:,0:8]
Y = train_dataset[:,8]

eval_dataset = numpy.loadtxt("pima-indians-diabetes.eval_data.txt", delimiter=",")

evalX = eval_dataset[:,0:8]
evalY = eval_dataset[:,8]

# NETWORK
model = Sequential()

# 8 inputs, 12 outputs
# Rectifier activation layer
# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
model.add(Dense(12, input_dim=8, activation='relu'))

# 12 inputs, 8 outputs
model.add(Dense(8, activation='relu'))

# ensure output is in range (0,1)
# Sigmoid activation layer
# https://en.wikipedia.org/wiki/Sigmoid_function
model.add(Dense(1, activation='sigmoid'))

# COMPILE MODEL

# logarithmic loss = binary cross entropy (https://en.wikipedia.org/wiki/Cross_entropy)
# gradient descent algorithm = adam (http://arxiv.org/abs/1412.6980)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAINING

# epochs is the number of iterations
# batch size is the number of instances evaluated before weight update in network
model.fit(X, Y, epochs=150, batch_size=10)

# EVALUATION (should be another dataset than we trained on)
scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(evalX)
rounded = [round(x[0]) for x in predictions]
print("Result",rounded)


# Serialize the model to JSON
model_json = model.to_json()
with open("model/model.json","w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights("model/model.h5")
print("Saved model to disk")



