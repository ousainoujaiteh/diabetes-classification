# load Flask
import flask
from flask import  Response , render_template,request
from werkzeug.utils import secure_filename
app = flask.Flask(__name__)

import os

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy
numpy.random.seed(7)


@app.route("/")
def index():
    return render_template("index.html",data=[])

@app.route("/single", methods=["GET"])
def single():
    return render_template("single.html")

@app.route("/handle_single",methods=["POST"])
def handle_single():

    num_preg = request.form.get("num_preg")
    glucose_conc = request.form.get("glucose_conc")
    diastolic_bp = request.form.get("diastolic_bp")
    thickness = request.form.get("thickness")
    insulin = request.form.get("insulin")
    bmi = request.form.get("bmi")
    diab_pred = request.form.get("diab_pred")
    age = request.form.get("age")
    skin = request.form.get("skin")

    form_data = [num_preg,glucose_conc,diastolic_bp,thickness,insulin,bmi,diab_pred,age,skin]

    file = 'dump.txt'

    path = os.path.join(os.path.abspath(""), os.path.join("DATA", file))

    with open(path,'w') as file:
        for d in form_data:
            val = d
            if not d == form_data[-1]:
                val = val + ","
                file.write(val)
            else:
                file.write(val)
        file.write('\n')
        for d in form_data:
            val = d
            if not d == form_data[-1]:
                val = val + ","
                file.write(val)
            else:
                file.write(val)


    data = prediction(path)

    results = read_file(path)
    final = []
    i = 0
    for d in results:
        val = d.split(',')
        del val[-1]
        final.append(val)
        i = i + 1

    result = []
    for x, y in zip(data, final):
        result.append({"pred": x, "values": y})
        break


    return render_template("index.html", data=result)






@app.route("/upload", methods=["GET"])
def upload():
    return render_template("upload.html")

@app.route("/handle_upload",methods=["POST"])
def handle_upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        path = os.path.join(os.path.abspath(""), os.path.join("DATA", filename))
        file.save(path)
        data =  prediction(path)

        results = read_file(path)
        final = []
        i = 0
        for d in results:
            val = d.split(',')
            del val[-1]
            final.append(val)
            i = i + 1

        result = []
        for x, y in zip(data, final):
            result.append({"pred": x, "values": y})

    return render_template("index.html", data=result)


def prediction(filename):

    train_dataset = numpy.loadtxt("pima-indians-diabetes.train_data.txt", delimiter=",")
    X = train_dataset[:, 0:8]
    Y = train_dataset[:, 8]

    eval_dataset = numpy.loadtxt(filename, delimiter=",")
    evalX = eval_dataset[:, 0:8]

    # load json and create model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model/model.h5")


    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    scores = loaded_model.evaluate(X, Y)

    print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))

    predictions = loaded_model.predict(evalX)
    K.clear_session()

    data = list()
    for num in predictions:
        for n in num:
            data.append(round(n))

    results = read_file(filename)
    final = []
    i = 0
    for d in results:
        val = d.split(',')
        del val[-1]
        final.append(val)
        i = i + 1

    result = []
    for x, y in zip(data,final):
        result.append({"pred":x,"values":y})

    return data

        #render_template("index.html", data=result)

@app.route("/predict",methods=["GET","POST"])
def predict():
    filename = "pima-indians-diabetes.eval_data.txt";

    # data = read_file(filename)
    # final = []
    # i = 0
    # for d in data:
    #     val = d.split(',')
    #     del val[-1]
    #     #final.append({i: val})
    #     final.append(val)
    #     i = i+1
    data = prediction(filename)

    results = read_file(filename)
    final = []
    i = 0
    for d in results:
        val = d.split(',')
        del val[-1]
        final.append(val)
        i = i + 1

    result = []
    for x, y in zip(data, final):
        result.append({"pred": x, "values": y})


    return render_template("index.html", data=result)


def read_file(filename):
    results = []
    with open(filename,'r') as file:
        lines = file.readlines()
        for line in lines:
            results.append(line)
    return results


@app.route("/train",methods=["GET","POST"])
def train():

    # DATASET
    train_dataset = numpy.loadtxt("pima-indians-diabetes.train_data.txt", delimiter=",")
    X = train_dataset[:, 0:8]
    Y = train_dataset[:, 8]

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
    history = model.fit(X, Y, validation_split=0.33,epochs=150, batch_size=10)

    # list all data in history

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Serialize the model to JSON
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("model/model.h5")
    print("Saved model to disk")

    K.clear_session()

    return render_template("index.html",train=[])

# start the flask app, allow remote connections
if __name__ == '__main__':
    app.run(host='0.0.0.0')
