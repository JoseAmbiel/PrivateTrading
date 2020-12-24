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
        if minn == maxx:
            wil[i] = -50
        else:
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
        if minn == maxx:
            sto[i] = 50
        else:
            sto[i] = 100* (data[i, 2] - minn) / (maxx - minn)
    
    k = SMA(sto, smoothK)
    d = SMA(k, periodD)

    return k, d

# ---------------------- MACD --------------------- #
def MACD(data, macd_fast, macd_slow, macd_sign):
    macd_auxi = EMA(data, 2/(macd_fast+1)) - EMA(data, 2/(macd_slow+1))
    macd_sign = EMA(macd_auxi, 2/(macd_sign+1))

    macd = macd_auxi - macd_sign

    return macd


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


def separator(df_close, interv):
    if '_1' in interv:
        sep00_n = 48; sep01_n = 104; sep02_n = 20
        sep10_n = 48; sep11_n = 104
        sep00 = EMA(df_close, 2/(sep00_n+1)) - EMA(df_close, 2/(sep01_n+1))
        sep01 = EMA(sep00, 2/(sep02_n+1))
        sep10 = EMA(df_close, 2/(sep10_n+1))
        sep11 = EMA(df_close, 2/(sep11_n+1))
        if 'uu' in interv:
            sep0 = np.greater_equal(sep00 - sep01, 0)
            sep1 = np.greater_equal(sep10 - sep11, 0)
        elif 'ud' in interv:
            sep0 = np.greater_equal(sep00 - sep01, 0)
            sep1 = np.less(sep10 - sep11, 0)
        elif 'du' in interv:
            sep0 = np.less(sep00 - sep01, 0)
            sep1 = np.greater_equal(sep10 - sep11, 0)
        elif 'dd' in interv:
            sep0 = np.less(sep00 - sep01, 0)
            sep1 = np.less(sep10 - sep11, 0)
        df_sep = np.logical_and(sep0, sep1)

    if '_2' in interv:
        sep000_n = 14; sep00_n = 20; sep01_n = 5
        sep10_n = 9; sep11_n = 21
        sep000 = RSI(df_close, sep000_n)
        sep00 = EMA(sep000, 2/(sep00_n+1))
        sep01 = EMA(sep000, 2/(sep01_n+1))
        sep10 = EMA(df_close, 2/(sep10_n+1))
        sep11 = EMA(df_close, 2/(sep11_n+1))
        if 'uu' in interv:
            sep0 = np.greater_equal(sep00 - sep01, 0)
            sep1 = np.greater_equal(sep10 - sep11, 0)
        elif 'ud' in interv:
            sep0 = np.greater_equal(sep00 - sep01, 0)
            sep1 = np.less(sep10 - sep11, 0)
        elif 'du' in interv:
            sep0 = np.less(sep00 - sep01, 0)
            sep1 = np.greater_equal(sep10 - sep11, 0)
        elif 'dd' in interv:
            sep0 = np.less(sep00 - sep01, 0)
            sep1 = np.less(sep10 - sep11, 0)
        df_sep = np.logical_and(sep0, sep1)

    if '_3' in interv:
        sep00_n = 12; sep01_n = 21; sep02_n = 45
        sep00 = EMA(df_close, 2/(sep00_n+1))
        sep01 = EMA(df_close, 2/(sep01_n+1))
        sep02 = EMA(df_close, 2/(sep02_n+1))
        if 'abc' in interv:
            sep0 = np.greater_equal(sep00 - sep01, 0)
            sep1 = np.greater_equal(sep00 - sep02, 0)
            sep2 = np.greater_equal(sep01 - sep02, 0)
        elif 'acb' in interv:
            sep0 = np.greater_equal(sep00 - sep01, 0)
            sep1 = np.greater_equal(sep00 - sep02, 0)
            sep2 = np.less(sep01 - sep02, 0)
        elif 'bac' in interv:
            sep0 = np.less(sep00 - sep01, 0)
            sep1 = np.greater_equal(sep00 - sep02, 0)
            sep2 = np.greater_equal(sep01 - sep02, 0)
        elif 'bca' in interv:
            sep0 = np.less(sep00 - sep01, 0)
            sep1 = np.less(sep00 - sep02, 0)
            sep2 = np.greater_equal(sep01 - sep02, 0)
        elif 'cab' in interv:
            sep0 = np.greater_equal(sep00 - sep01, 0)
            sep1 = np.less(sep00 - sep02, 0)
            sep2 = np.less(sep01 - sep02, 0)
        elif 'cba' in interv:
            sep0 = np.less(sep00 - sep01, 0)
            sep1 = np.less(sep00 - sep02, 0)
            sep2 = np.less(sep01 - sep02, 0)
        df_sep = np.logical_and(sep2, np.logical_and(sep0, sep1))

    return df_sep


##################################################################################################################################
# ----------------------------------------------------------- SCALING ---------------------------------------------------------- #
##################################################################################################################################

# ------------------- percentage scaling ------------------ #
def pct_scale(data):
    data1 = np.divide(data[:, 1:]-data[:, 0:-1], data[:, 0:-1])
    data1 = preprocessing.scale(data1, axis=1)

    return data1

# --------------------- normal scaling -------------------- #
def scl_scale(data):
    data1 = data[:, 1:]
    data1 = preprocessing.scale(data1, axis=1)

    return data1

# ------------------ normal scaling with ------------------ #
def scl_scale_group(data):
    data1 = data[:, 1:, :]
    data1 = np.transpose(data1, (0, 2, 1))
    data1 = np.reshape(data1, (data1.shape[0], data1.shape[1]*data1.shape[2]))
    data1 = preprocessing.scale(data1, axis=1)
    data1 = np.reshape(data1, (data.shape[0], data.shape[2], data.shape[1]-1))
    data1 = np.transpose(data1, (0, 2, 1))

    return data1

# ------------------- difference scaling ------------------ #
def dif_scale(data):
    data1 = data[:, 1:]-data[:, :-1]
    data1 = preprocessing.scale(data1, axis=1)

    return data1

# -------------------- RSI-like scaling ------------------- #
def rsi_scale(data):
    data1 = data[:, 1:]
    data1 = (data1 - 50) / 50

    return data1

# ----------------- Stochastic-like scaling --------------- #
def sto_scale(data):
    data1 = data[:, 1:]
    data1 = (data1 - 50) / 50

    return data1

# -------------------- WIL-like scaling ------------------- #
def wil_scale(data):
    data1 = data[:, 1:]
    data1 = (data1 + 50) / 50

    return data1

# --------------------- Model scaling --------------------- #
def mdl_scale(data):
    data1 = data[:, 1:]
    data1[data1>=0.5] = 1
    data1[data1< 0.5] = 0
#     data1 = data1 - 0.5

    return data1


dispatcher = {'pct':pct_scale, 'scl':scl_scale, 'scl_group':scl_scale_group, 'dif':dif_scale, 'rsi':rsi_scale, 'sto':sto_scale, 'wil':wil_scale, 'mdl':mdl_scale}


# -------------------- Momentum scaling ------------------- #
def mom_scale(data, mean):
    data1 = data[:, 1:]
    data1 = np.divide(data1, np.tile(mean.reshape((-1, 1)), (1, data1.shape[1])))
    data1 = preprocessing.scale(data1, axis=1, with_mean=False)
   
    return data1


# -------- Distance from the actual price scaling --------- #
def closediff_scale(data, last_price):
    data1 = data[:, 1:]
    data1 = data1 - np.tile(last_price.reshape((-1, 1)), (1, data1.shape[1]))
    data1 = preprocessing.scale(data1, axis=1, with_mean=False)

    return data1


# --------- Poercentage distance from close price --------- #
def pctdiff2seq_scale(data, close_seq):
    data1 = data[:, 1:]
    data1 = (data1 - close_seq) / close_seq
    data1 = preprocessing.scale(data1, axis=1, with_mean=False)

    return data1


def scaling_data(sample_x, scale_type, scale_vect, close_pos):
    x = np.zeros((sample_x.shape[0], sample_x.shape[1]-1, sample_x.shape[2]))
    for i in range(0, len(scale_type)):
        if scale_type[i] == 'cld':
            last_price = sample_x[:, -1, close_pos]
            oi2 = closediff_scale(sample_x[:, :, scale_vect[i]], last_price)
        elif scale_type[i] == 'pds':
            close_seq = sample_x[:, 1:, close_pos]
            oi2 = pctdiff2seq_scale(sample_x[:, :, scale_vect[i]], close_seq)
        elif scale_type[i] == 'mom':
            mean = np.mean(sample_x[:, 1:, close_pos], axis=1)
            oi2 = mom_scale(sample_x[:, :, scale_vect[i]], mean)
        else:
            oi2 = dispatcher[scale_type[i]](sample_x[:, :, scale_vect[i]])

        x[:, :, scale_vect[i]] = oi2

    return x


