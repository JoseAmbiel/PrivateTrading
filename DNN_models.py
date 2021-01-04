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
        activ = 'relu'
    elif modl % 10 == 2:
        w_init = 'glorot_uniform'
        activ = 'tanh'
    elif modl % 10 == 3:
        w_init = 'he_uniform'
        activ = 'relu'
    elif modl % 10 == 4:
        w_init = 'he_uniform'
        activ = 'tanh'

    Model = Sequential()
    Model.add(Dropout(0.2, input_shape=(InputShape,)))
    if int(modl/10) == 1010:
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1011:
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1020:
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1021:
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1030:
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1031:
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1040:
        Model.add(Dense(8192, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 1041:
        Model.add(Dense(8192, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2010:
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2011:
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2020:
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2021:
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2030:
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2031:
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2040:
        Model.add(Dense(8192, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 2041:
        Model.add(Dense(8192, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3010:
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3011:
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3020:
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3021:
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3030:
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3031:
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3040:
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 3041:
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4010:
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(8, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4011:
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(Dense(8, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4020:
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4021:
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4030:
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4031:
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4040:
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 4041:
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5010:
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(8, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(4, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5011:
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(Dense(8, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
        Model.add(Dense(4, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.1))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5020:
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5021:
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(32, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
        Model.add(Dense(16, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.2))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5030:
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5031:
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(128, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(64, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.3))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5040:
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(BatchNormalization())
    elif int(modl/10) == 5041:
        Model.add(Dense(4096, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(2048, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(1024, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(512, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())
        Model.add(Dense(256, kernel_initializer=w_init, activation=activ))
        Model.add(Dropout(0.4))
        Model.add(BatchNormalization())

    Model.add(Dense(2, activation='softmax'))
    Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return Model
