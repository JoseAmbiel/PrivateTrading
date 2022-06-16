import time
start_runtime = time.time(); total_time = 12*50*60
from random import randint
from json import loads as json_loads

def call_run(ith_back, maxwaitingtime=20):
    keepgoing = True
    !git clone https://JoseAmbiel:password@github.com/JoseAmbiel/Runs.git
    !git config --global user.email 'email'
    !git config --global user.name 'JoseAmbiel'
    %cd Runs

    with open('MANO_random.txt', 'r') as f:
        run_case = f.readlines()
    f.close()
    with open('MANO_counter.txt', 'r') as f:
        run_case_number = f.readlines()
    f.close()
    !git rm 'MANO_counter.txt'

    ith = int(run_case_number[0].strip())
    if ith <= len(run_case):
        content = run_case[ith].split(';')
        MODELL = int(content[0].strip().replace('MODELL = ', ''))
        indicators = content[1].strip().replace('indicators = ', '').replace("'", '').strip('][').split(', ')
        indic_param = json_loads(content[2].strip().replace('indic_param = ', ''))
        scale_type = content[3].strip().replace('scale_type = ', '').replace("'", '').strip('][').split(', ')
        interv = content[4].strip().replace('interv = ', '').replace("'", '')
        BATCH_SIZE = int(content[5].strip().replace('BATCH_SIZE = ', ''))
        with open('MANO_counter.txt', 'w') as f:
            if ith_back >= 0:
                f.write(str(ith_back) + '\n')
                for lines in run_case_number:
                    f.write(lines)
            elif len(run_case_number) > 1:
                for lines in run_case_number[1:]:
                    f.write(lines)
            else:
                f.write(str(ith + 1))
        f.close()

        !git add 'MANO_counter.txt'
        !git commit -m 'Changed via Colaboratory'
        !git push origin main
        boraver = !git status

        if "Your branch is up to date with 'origin/main'." in boraver:
            ooocaralho = False
            print()
            print("Done")
        else:
            ooocaralho = True
            print()
            print("ooocaralho")
            print()
            time.sleep(randint(0, maxwaitingtime))
    
    else:
        ooocaralho = False
        keepgoing = False
    
    %cd /content
    !rm -r Runs

    return ith, keepgoing, ooocaralho, MODELL, indicators, indic_param, scale_type, interv, BATCH_SIZE

keepgoing = True
ooocaralho = True
while ooocaralho:

    ith, keepgoing, ooocaralho, MODELL, indicators, indic_param, scale_type, interv, BATCH_SIZE = call_run(-1, maxwaitingtime=60)

!git clone https://JoseAmbiel:password@github.com/JoseAmbiel/PrivateTrading.git
from google.colab import files
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dropout, LSTM, BatchNormalization
import datetime
import os
!cp "/content/PrivateTrading/LSTM_models.py" .
import LSTM_models as mdl
!cp "/content/PrivateTrading/Indicators.py" .
import Indicators as indct

while keepgoing:
    while ooocaralho:

        ith, keepgoing, ooocaralho, MODELL, indicators, indic_param, scale_type, interv, BATCH_SIZE = call_run(-1)

    if keepgoing:
        ##################################################################################################################################
        # --------------------------------------------------------- PARAMETERS --------------------------------------------------------- #
        ##################################################################################################################################
        CHARTS = ['btc1all_0']
        PREDIC = '5min'
        USING = '1min'
        SEQ_LEN = 61
        FUTURE_PERIOD_BASED = 0
        FUTURE_PERIOD_PREDICT = 5
        longer_indic = 200
        SOSSEGO = 30
        EPOCHS = 70
        diff2stop = 0.01
        diff2nope = 0.005

        first = True
        for chart in CHARTS:
            df = pd.read_csv('/content/PrivateTrading/'+chart+'.csv', names=['time', 'open', 'high', 'low', 'close', 'volume', 'volumeMA'], skiprows=1)

            for indic_n in range(1, len(indicators)):
                if indicators[indic_n] == 'klo':
                    df['klo'] = df['close']
                if indicators[indic_n] == 'vol':
                    df['vol'] = df['volume']
                if indicators[indic_n] == 'ema':
                    df['ema'+str(indic_param[indic_n])] = indct.EMA(df['close'].values, 2/(indic_param[indic_n]+1)) # alpha = 2/(emalen+1)
                if indicators[indic_n] == 'rsi':
                    df['rsi'+str(indic_param[indic_n])] = indct.RSI(df['close'].values, indic_param[indic_n])
                if indicators[indic_n] == 'wil':
                    df['wil'+str(indic_param[indic_n])] = indct.Williams_R(df[['high', 'low', 'close']].values, indic_param[indic_n])
                if indicators[indic_n] == 'mom':
                    df['mom'+str(indic_param[indic_n])] = indct.Momentum(df['close'].values, indic_param[indic_n])
                if indicators[indic_n] == 'sto':
                    df['sto'+str(indic_param[indic_n])], nope = indct.Stochastic(df[['high', 'low', 'close']].values, indic_param[indic_n])
                if indicators[indic_n] == 'mcd':
                    df['mcd'+str(indic_param[indic_n][0])+"'"+str(indic_param[indic_n][1])+"'"+str(indic_param[indic_n][2])] = indct.MACD(df['close'].values, indic_param[indic_n][0], indic_param[indic_n][1], indic_param[indic_n][2])

            df['sep'] = indct.separator(df['close'], interv)

            df.drop(columns=['time', 'open', 'high', 'low', 'volume', 'volumeMA'], inplace=True)
            df.drop(df.index[:longer_indic], inplace=True)

            close_pos = list(df.columns).index('close')
            n_indic = df.shape[1]
            oi1 = indct.xy_sample(df.values, close_pos, n_indic, SEQ_LEN, FUTURE_PERIOD_BASED, FUTURE_PERIOD_PREDICT)
            n_indic -= 1
            oi1 = np.concatenate((oi1[oi1[:, -2].astype(bool), 0:-(SEQ_LEN+1)], oi1[oi1[:, -2].astype(bool), -1:]), axis=1)
            if first:
                oi3 = oi1
                first = False
            else:
                oi3 = np.concatenate((oi3, oi1), axis=0)

        scale_vect = range(0,len(scale_type))

        namindics = 'klo'
        for i in df.columns[1:-1]:
            namindics = namindics + '_' + i

        namscales = ''
        for i in scale_type:
            namscales = namscales + i

        if '0020' in interv:
            interv_bot = 0.00001
            interv_top = 0.20
        elif '2040' in interv:
            interv_bot = 0.20
            interv_top = 0.40
        elif '4060' in interv:
            interv_bot = 0.40
            interv_top = 0.60
        elif '6080' in interv:
            interv_bot = 0.60
            interv_top = 0.80
        elif '8000' in interv:
            interv_bot = 0.80
            interv_top = 0.99999
        else:
            print('intervalo ta errado')

        idx_split1 = int(np.ceil([oi3.shape[0] * interv_bot]))
        idx_split2 = int(np.floor([oi3.shape[0] * interv_top]))
        oi3_train = oi3[np.concatenate((range(0, idx_split1), range(idx_split2, oi3.shape[0]))), :]
        oi3_validation = oi3[range(idx_split1, idx_split2), :]

        ##################################################################################################################################
        # ----------------------------------------------------- Balancing the data ----------------------------------------------------- #
        ##################################################################################################################################

        # ------------------- train ------------------ #
        buy = oi3_train[oi3_train[:,-1]==1.,:] # number of buys
        sell = oi3_train[oi3_train[:,-1]==0.,:] # number of sells
        lower = min(len(buy),len(sell));
        np.random.shuffle(buy)
        np.random.shuffle(sell)
        buy = buy[0:lower,:]
        sell = sell[0:lower,:]
        sample_train = np.concatenate((buy,sell),axis=0)
        np.random.shuffle(sample_train) # shuffling so buys and sells are shuffled

        # sample_train = oi3_train
        print('percentage of moon on train      sample: '+str(len(oi3_train[oi3_train[:,-1]==1.,:])/len(oi3_train)))

        y_train = sample_train[:, -1]
        sample_train_x = sample_train[:, 0:-1]
        sample_train_x = np.reshape(sample_train_x, (y_train.shape[0], n_indic, SEQ_LEN))
        sample_train_x = np.transpose(sample_train_x, (0, 2, 1))

        # ----------------- validation --------------- #
        sample_validation = oi3_validation
        y_validation = sample_validation[:, -1]
        sample_validation_x = sample_validation[:, 0:-1]
        sample_validation_x = np.reshape(sample_validation_x, (y_validation.shape[0], n_indic, SEQ_LEN))
        sample_validation_x = np.transpose(sample_validation_x, (0, 2, 1))
        print('percentage of moon on validation sample: '+str(np.sum(y_validation)/len(y_validation)))

        ##################################################################################################################################
        # ------------------------------------------------------ Scaling the data ------------------------------------------------------ #
        ##################################################################################################################################
        
        x_train = indct.scaling_data(sample_train_x, scale_type, scale_vect, close_pos)
        x_validation = indct.scaling_data(sample_validation_x, scale_type, scale_vect, close_pos)

        ##################################################################################################################################
        # --------------------------------------------------------- Dimensions --------------------------------------------------------- #
        ##################################################################################################################################
        print(x_train.shape)
        print(x_validation.shape)
        print(y_train.shape)
        print(y_validation.shape)
        print()

        ##################################################################################################################################
        # ------------------------------------------------------- LSTM MODEL-XX -------------------------------------------------------- #
        ##################################################################################################################################
        filepath = './'
        name = 'LSTM-' + str(SEQ_LEN-1) + 'x' + USING + '-' + namindics + '_' + namscales + '-' + interv + '-' + str(MODELL) + '-' + str(BATCH_SIZE)
        print(name)
        model = mdl.MODL(MODELL, x_train.shape[1:])

        epoch = 0
        sossego_counter = 0
        sum_losses = 2
        min_val_loss = 1
        condit = True
        start_training = time.time()
        while condit and epoch <= EPOCHS:
            history = model.fit(x_train, y_train, 
                                epochs=1, 
                                batch_size=BATCH_SIZE, 
                                validation_data=(x_validation, y_validation), 
                                verbose=0
                                )

            epoch, condit, sossego_counter, sum_losses, min_val_loss = mdl.my_callback(history, epoch, SOSSEGO, sossego_counter, diff2stop, diff2nope, sum_losses, min_val_loss, model, filepath, name)

            remaining_time = start_runtime + total_time - time.time()
            time4epoch = (time.time() - start_training) / epoch
            if ((epoch == 1) and (time4epoch * 20 > remaining_time)) or (time4epoch > remaining_time):
                ith, keepgoing, ooocaralho, MODELL, indicators, indic_param, scale_type, interv, BATCH_SIZE = call_run(ith)
                condit = False
                keepgoing = False
    
    ooocaralho = True
    print("------------------------------------------------------------------------------------------\n")

##################################################################################################################################
# ---------------------------------------------- SAVING LOGS AND MODELS ON GITHUB ---------------------------------------------- #
##################################################################################################################################
startcommit = time.time()

!git clone https://JoseAmbiel:password@github.com/JoseAmbiel/Models_brave.git
!git config --global user.email 'email'
!git config --global user.name 'JoseAmbiel'
dest = 'Models_brave/'

files = os.listdir(".")
filee = [fi for fi in files if '.h5' in fi]

for mode in filee:
    !cp {mode} {dest+mode}

%cd {dest}

!git add --all
!git commit -m 'Added via Colaboratory'
!git push origin master
boraver = !git status

if "Your branch is up to date with 'origin/master'." in boraver:
    ooocaralho = False
    print()
    print("Done")
else:
    ooocaralho = True
    print()
    print("ooocaralho")
    print()

while ooocaralho:
    !git pull
    !git commit -m 'Added via Colaboratory'
    !git push origin master
    boraver = !git status
    if "Your branch is up to date with 'origin/master'." in boraver:
        ooocaralho = False
        print()
        print("Done")
    else:
        ooocaralho = True
        print()
        print("ooocaralho")
        print()

%cd ..

print("Committing time: " + str( time.time() - startcommit ))
