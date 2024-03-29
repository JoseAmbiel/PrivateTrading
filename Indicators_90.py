import numpy as np
from sklearn import preprocessing

##################################################################################################################################
# --------------------------------------------------------- INDICATORS --------------------------------------------------------- #
##################################################################################################################################

# ---------------------- SMA ---------------------- #
def SMA(data, smalen):
    sma = np.zeros(data.shape)
    for i in range(smalen, len(data)):
        sma[i] = sum(data[i-smalen+1: i+1]) / smalen

    return sma

# ---------------------- EMA ---------------------- #
def EMA(data, alpha):
    ema = np.zeros(data.shape)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i]*alpha + ema[i-1]*(1-alpha)
    
    return ema

# ---------------------- RSI ---------------------- #
def RSI(data, rsilen):
    U   = np.zeros(data.shape)
    D   = np.zeros(data.shape); D[0] = 1
    U[1:] = np.maximum(data[1:]-data[0:-1], 0)
    D[1:] = np.maximum(data[0:-1]-data[1:], 0)
    emaU = EMA(U, 1/rsilen)
    emaD = EMA(D, 1/rsilen)
    rs = np.divide(emaU, emaD)
    rsi = 100 - 100 / (1+rs)

    return rsi

# ------------------- Williams_R ------------------ #
def Williams_R(data, willen):
    wil = np.zeros((data.shape[0], 1))
    for i in range(willen, len(data)):
        maxx = max(data[i-willen+1:i+1, 0])
        minn = min(data[i-willen+1:i+1, 1])
        wil[i] = 100* (data[i, 2] - maxx) / (maxx - minn)

    return wil

# -------------------- Momentum ------------------- #
def Momentum(data, momlen):
    mom = np.zeros(data.shape)
    mom[momlen:] = data[momlen:] - data[0:-momlen]

    return mom

# ------------------- Stochastic ------------------ #
def Stochastic(data, periodK):
    smoothK = 3
    periodD = 3
    sto = np.zeros((data.shape[0], 1))
    for i in range(periodK, data.shape[0]):
        maxx = max(data[i-periodK+1:i+1, 0])
        minn = min(data[i-periodK+1:i+1, 1])
        sto[i] = 100* (data[i, 2] - minn) / (maxx - minn)

    k = SMA(sto, smoothK)
    d = SMA(k, periodD)

    return k, d


##################################################################################################################################
# ------------------------------------------------------- TEMPORAL SERIES ------------------------------------------------------ #
##################################################################################################################################

def xy_sample(df_data, close_pos, n_indic, SEQ_LEN, FUTURE_PERIOD_BASED, FUTURE_PERIOD_PREDICT):

# ------------------- y vector ------------------ #
    y_aux1 = df_data[SEQ_LEN-1+FUTURE_PERIOD_BASED:-FUTURE_PERIOD_PREDICT+FUTURE_PERIOD_BASED, range(close_pos, df_data.shape[1], n_indic)]
    y_aux2 = df_data[SEQ_LEN-1+FUTURE_PERIOD_PREDICT:, range(close_pos, df_data.shape[1], n_indic)]
    y_sample = np.zeros((y_aux1.shape))
    y_sample[y_aux2>y_aux1] = 1

# ------------------- x vector ------------------ #
    cases = df_data.shape[0] - FUTURE_PERIOD_PREDICT - SEQ_LEN + 1
    x_sample = np.zeros((cases, SEQ_LEN, df_data.shape[1]))
    for i in range(SEQ_LEN, df_data.shape[0]-FUTURE_PERIOD_PREDICT+1):
        x_sample[i-SEQ_LEN, :, :] = df_data[i-SEQ_LEN:i, :]

    oi1 = np.transpose(x_sample, (0, 2, 1))
    oi1 = np.squeeze(np.reshape(oi1, (1, cases, n_indic*oi1.shape[2])))
    oi1 = np.concatenate((oi1, y_sample), axis=1)

    return oi1


##################################################################################################################################
# ----------------------------------------------------------- SCALING ---------------------------------------------------------- #
##################################################################################################################################

# ------------------- percentage scaling ------------------ #
def pct_scale(data):
    data1 = data.T
    data1 = np.divide(data1[1:, :]-data1[0:-1, :], data1[0:-1, :])
    data1 = preprocessing.scale(data1)
    data1 = data1.T

    return data1

# --------------------- normal scaling -------------------- #
def scl_scale(data):
    data1 = data.T
    data1 = data1[1:, :]
    data1 = preprocessing.scale(data1)
    data1 = data1.T

    return data1

# ------------------ normal scaling with ------------------ #
def scl_scale_group(data):
    data1 = data[:, 1:, :]
    data1 = np.transpose(data1, (0, 2, 1))
    data1 = np.reshape(data1, (data1.shape[0], data1.shape[1]*data1.shape[2]))
    data1 = preprocessing.scale(data1.T)
    data1 = data1.T
    data1 = np.reshape(data1, (data.shape[0], data.shape[2], data.shape[1]-1))
    data1 = np.transpose(data1, (0, 2, 1))

    return data1

# ------------------- difference scaling ------------------ #
def dif_scale(data):
    data1 = data.T
    data1 = data1[1:, :]-data1[0:-1, :]
    data1 = preprocessing.scale(data1)
    data1 = data1.T

    return data1

# -------------------- RSI-like scaling ------------------- #
def rsi_scale(data):
    data1 = data.T
    data1 = data1[1:, :]
    data1 = (data1 - 50) / 100
    data1 = data1.T

    return data1

# ----------------- Stochastic-like scaling --------------- #
def sto_scale(data):
    data1 = data.T
    data1 = data1[1:, :]
    data1 = (data1 - 50) / 100
    data1 = data1.T

    return data1

# -------------------- WIL-like scaling ------------------- #
def wil_scale(data):
    data1 = data.T
    data1 = data1[1:, :]
    data1 = (data1 + 50) / 100
    data1 = data1.T

    return data1

# --------------------- Model scaling --------------------- #
def mdl_scale(data):
    data1 = data.T
    data1 = data1[1:, :]
    data1 = data1 - 0.5
    data1 = data1.T

    return data1

dispatcher = {'pct':pct_scale, 'scl':scl_scale, 'scl_group':scl_scale_group, 'dif':dif_scale, 'rsi':rsi_scale, 'sto':sto_scale, 'wil':wil_scale, 'mdl':mdl_scale}

