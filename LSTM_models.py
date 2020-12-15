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

def my_callback(history, epoch, SOSSEGO, sossego_counter, diff2stop, diff2nope, sum_losses, min_val_loss, model, filepath, name):
    epoch += 1
    condit = True
    accu = history.history['accuracy'][0]
    loss = history.history['loss'][0]
    val_accu = history.history['val_accuracy'][0]
    val_loss = history.history['val_loss'][0]

    if ((loss + diff2stop) < val_loss) and (val_loss > min_val_loss):
        condit = False
        print()
        print('STOPPING TRANING ON EPOCH:' + str(epoch))
        print('loss = ' + str(loss) + ' and val_loss = ' + str(val_loss))
    elif val_loss > min_val_loss:
        sossego_counter += 1
        if sossego_counter >= SOSSEGO:
            condit = False
            print()
            print('STOPPING TRANING ON EPOCH:' + str(epoch))
            print('sossego_counter = ' + str(sossego_counter))
    else:
        model.save(filepath+name+'.h5')
        print('Epoch: {:4} at {}  -  loss: {:6.4f}  -  accuracy: {:6.4f}  -  val_loss: {:6.4f}  -  val_accuracy: {:6.4f}'.format(epoch, datetime.datetime.now().time(), loss, accu, val_loss, val_accu))
        min_val_loss = val_loss
        sossego_counter = 0
    
    return epoch, condit, sossego_counter, sum_losses, min_val_loss


def MODL(modl, InputShape):
    if modl % 10 == 1:
        w_init = 'glorot_uniform'
        optim = 'adam'
    elif modl % 10 == 2:
        w_init = 'glorot_uniform'
        optim = 'sgd'
    elif modl % 10 == 3:
        w_init = 'he_uniform'
        optim = 'adam'
    elif modl % 10 == 4:
        w_init = 'he_uniform'
        optim = 'sgd'

    Model = Sequential()
    Model.add(Dropout(0.2, input_shape=(InputShape)))
    if int(modl/10) == 1010:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1011:
        Model.add(LSTM(8, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1012:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1020:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1021:
        Model.add(LSTM(16, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1022:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1030:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1031:
        Model.add(LSTM(32, recurrent_dropout=0.4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1032:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1040:
        Model.add(LSTM(64, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1041:
        Model.add(LSTM(64, recurrent_dropout=0.5, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1042:
        Model.add(LSTM(64, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2010:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2011:
        Model.add(LSTM(8, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2012:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2020:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2021:
        Model.add(LSTM(16, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2022:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2030:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2031:
        Model.add(LSTM(32, recurrent_dropout=0.4, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2032:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2040:
        Model.add(LSTM(64, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2041:
        Model.add(LSTM(64, recurrent_dropout=0.6, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(32, recurrent_dropout=0.5, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2042:
        Model.add(LSTM(64, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.6))
        Model.add(BatchNormalization())
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3010:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3011:
        Model.add(LSTM(8, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, recurrent_dropout=0.1, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3012:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3020:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3021:
        Model.add(LSTM(16, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3022:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3030:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3031:
        Model.add(LSTM(32, recurrent_dropout=0.5, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, recurrent_dropout=0.4, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3032:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3040:
        Model.add(LSTM(50, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(30, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(10, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3041:
        Model.add(LSTM(50, recurrent_dropout=0.6, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(30, recurrent_dropout=0.5, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(10, recurrent_dropout=0.4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3042:
        Model.add(LSTM(50, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.6))
        Model.add(BatchNormalization())
        Model.add(LSTM(30, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
        Model.add(LSTM(10, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4010:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4011:
        Model.add(LSTM(8, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.1, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, recurrent_dropout=0.1, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4012:
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4020:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4021:
        Model.add(LSTM(16, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, recurrent_dropout=0.1, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4022:
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(2, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4030:
        Model.add(LSTM(10, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(6, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4031:
        Model.add(LSTM(10, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(6, recurrent_dropout=0.1, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4032:
        Model.add(LSTM(10, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(LSTM(6, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4040:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4041:
        Model.add(LSTM(32, recurrent_dropout=0.5, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, recurrent_dropout=0.4, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, recurrent_dropout=0.3, kernel_initializer=w_init, return_sequences=True))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, recurrent_dropout=0.2, kernel_initializer=w_init, return_sequences=False))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4042:
        Model.add(LSTM(32, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.5))
        Model.add(BatchNormalization())
        Model.add(LSTM(16, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(LSTM(8, kernel_initializer=w_init, return_sequences=True))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(LSTM(4, kernel_initializer=w_init, return_sequences=False))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())

    Model.add(Dense(2, activation='softmax'))
    Model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return Model
