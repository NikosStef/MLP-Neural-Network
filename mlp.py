import pandas as pd
import keras as k
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_data():
    '''
    Read file and load necessary data in preferred format
    
    Returns a table
    '''
    
    dataset = pd.read_csv("pdspeech.csv", header=1)
    mydata = dataset.iloc[:].values
    return mydata


def process_data(mydata):
    '''
    Prepocess the data by replacing Null values, normalize/standarize
    and transform to categorical.
    
    Params: Table of data
    Returns dependent and independent (x, y) data
    '''
    
    imputer = preprocessing.Imputer(missing_values="NaN",
                      strategy="mean",
                      axis=0)
    imputer = imputer.fit(mydata)
    mydata = imputer.transform(mydata)
    ydata = mydata[:, 754]
    xdata = mydata[:, :-1]
    xdata = preprocessing.scale(xdata)
    ydata = k.utils.to_categorical(ydata)
    return xdata, ydata


def initialize_model():
    '''
    Build and compile model with 4 layers
    
    Returns model
    '''
    
    model = k.Sequential()
    model.add(k.layers.Dense(units=256,
                             activation="relu",
                             input_dim=754,
                             activity_regularizer=k.regularizers.l1(0.0001)))
    model.add(k.layers.Dense(units=64, activation="relu"))
    model.add(k.layers.Dense(units=16, activation="sigmoid"))
    model.add(k.layers.Dense(units=2, activation='softmax'))
    myoptimizer = k.optimizers.SGD(lr=0.1,
                                   momentum=0.4,
                                   decay=0.0,
                                   nesterov=False)
    model.compile(loss=k.losses.categorical_hinge,
                  optimizer=myoptimizer,
                  metrics=['accuracy'])
    return model


def fit_and_train(mymodel, xtrain, ytrain, xtest, ytest):
    '''
    Fit data on model automatically or manually, batch by batch/
    
    Params: Model and training data split in x and y
    Returns model after training
    '''
    
    history = mymodel.fit(xtrain, ytrain,validation_data=(xtest, ytest), epochs=55, batch_size=64)
    # model.train_on_batch(x_batch, y_batch)
    return mymodel, history


def evaluate_predict(mymodel, xtest, ytest):
    '''
    Evaluate and predict accuracy and performance of the model
    
    Params: Model and test data
    Returns loss and metrics
    '''
    
    loss_and_metrics = mymodel.evaluate(xtest, ytest, batch_size=128)
    # classes = mymodel.predict(xtest, batch_size=128)
    return loss_and_metrics


def visualize(mymodel, history):
    '''
    Visualize the model and its loss and accuracy
    
    Params: Model and History
    '''
    
    #k.utils.plot_model(mymodel, to_file='model.png')
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    #plt.savefig('acc_plot.png', bbox_inches='tight')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    #plt.savefig('loss_plot.png')



if __name__ == "__main__":
    model = initialize_model()
    data = load_data()
    xdata, ydata = process_data(data)
    x_train, x_test, y_train, y_test = train_test_split(xdata,
                                                        ydata,
                                                        test_size=0.23,
                                                        random_state=5)
    model, history = fit_and_train(model, x_train, y_train, x_test, y_test)
    visualize(model, history)
