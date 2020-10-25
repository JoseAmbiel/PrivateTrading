import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import datetime

class mycllbck(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%10==0:
            print('Epoch: {:4} at {}  -  loss: {:6.4f}  -  accuracy: {:6.4f}  -  val_loss: {:6.4f}  -  val_accuracy: {:6.4f}'. \
                  format(epoch, datetime.datetime.now().time(), logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']))

def earlystp(patience):
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    return early_stop

def chckpnt(filepath, name):
    checkpoint = ModelCheckpoint(filepath+'model-'+name+'.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    return checkpoint

def csvlog(filepath, name):
    csv_logger = CSVLogger(filepath+'logg-'+name+'.csv', append=True)
    return csv_logger

def my_callback(history, epoch, SOSSEGO, sossego_counter, sum_losses, min_val_loss, model, filepath, name):
    epoch += 1
    condit = True
    accu = history.history['accuracy'][0]
    loss = history.history['loss'][0]
    val_accu = history.history['val_accuracy'][0]
    val_loss = history.history['val_loss'][0]

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        sossego_counter = 0
    else:
        sossego_counter += 1
        if sossego_counter >= SOSSEGO:
            condit = False
            print()
            print('STOPPING TRANING ON EPOCH:' + str(epoch))
            print('sossego_counter = ' + str(sossego_counter))
    
    if (loss + 0.01) < val_loss:
        condit = False
        print()
        print('STOPPING TRANING ON EPOCH:' + str(epoch))
        print('loss = ' + str(loss) + ' and val_loss = ' + str(val_loss))
    else:
        if (loss+val_loss) < sum_losses:
            sum_losses = loss + val_loss
            model.save(filepath+name)
            print('Epoch: {:4} at {}  -  loss: {:6.4f}  -  accuracy: {:6.4f}  -  val_loss: {:6.4f}  -  val_accuracy: {:6.4f}'.format(epoch, datetime.datetime.now().time(), loss, accu, val_loss, val_accu))
    
    return epoch, condit, sossego_counter, sum_losses, min_val_loss


def MODL(modl, InputShape):
    Model = Sequential()
    Model.add(Dropout(0.2, input_shape=(InputShape)))
    if modl==20100:
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20110:
        Model.add(LSTM(8, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20120:
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20200:
        Model.add(LSTM(16, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20210:
        Model.add(LSTM(16, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.2, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20220:
        Model.add(LSTM(16, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=False))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20300:
        Model.add(LSTM(32, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20310:
        Model.add(LSTM(32, recurrent_dropout=0.4, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, recurrent_dropout=0.3, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20320:
        Model.add(LSTM(32, return_sequences=True))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, return_sequences=False))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20400:
        Model.add(LSTM(64, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(32, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20410:
        Model.add(LSTM(64, recurrent_dropout=0.6, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(32, recurrent_dropout=0.5, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==20420:
        Model.add(LSTM(64, return_sequences=True))
        Model.add(Dropout(0.6))
        Model.add(BatchNormalization())
        Model.add(LSTM(32, return_sequences=False))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30100:
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30110:
        Model.add(LSTM(8, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, recurrent_dropout=0.1, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30120:
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, return_sequences=False))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30200:
        Model.add(LSTM(16, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30210:
        Model.add(LSTM(16, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.3, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30220:
        Model.add(LSTM(16, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30300:
        Model.add(LSTM(32, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30310:
        Model.add(LSTM(32, recurrent_dropout=0.5, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, recurrent_dropout=0.4, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30320:
        Model.add(LSTM(32, return_sequences=True))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, return_sequences=True))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=False))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30400:
        Model.add(LSTM(50, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(30, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(10, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30410:
        Model.add(LSTM(50, recurrent_dropout=0.6, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(30, recurrent_dropout=0.5, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(10, recurrent_dropout=0.4, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==30420:
        Model.add(LSTM(50, return_sequences=True))
        Model.add(Dropout(0.6))
        Model.add(BatchNormalization())
        Model.add(LSTM(30, return_sequences=True))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
        Model.add(LSTM(10, return_sequences=False))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40100:
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40110:
        Model.add(LSTM(8, recurrent_dropout=0.2, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.2, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.1, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, recurrent_dropout=0.1, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40120:
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=True))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, return_sequences=False))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40200:
        Model.add(LSTM(16, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40210:
        Model.add(LSTM(16, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, recurrent_dropout=0.1, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40220:
        Model.add(LSTM(16, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, return_sequences=False))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40300:
        Model.add(LSTM(10, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(6, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40310:
        Model.add(LSTM(10, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.2, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(6, recurrent_dropout=0.1, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40320:
        Model.add(LSTM(10, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(6, return_sequences=True))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40400:
        Model.add(LSTM(32, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40410:
        Model.add(LSTM(32, recurrent_dropout=0.5, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, recurrent_dropout=0.4, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, return_sequences=False))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif modl==40420:
        Model.add(LSTM(32, return_sequences=True))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, return_sequences=True))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, return_sequences=False))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(Dense(2, activation='softmax'))
        Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return Model

