import numpy as np
import pandas as pd
import sys
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse as sp
from xgboost.sklearn import XGBClassifier
import NDCG

np.random.seed(0)

#Loading data
df_train = pd.read_csv('./input/train_users.csv')
df_test = pd.read_csv('./input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['date_first_booking'], axis=1)

#####Feature engineering#######
#date_account_created
df_all['date_account_created'] = df_all.date_account_created.apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
df_all['dac_year'] = df_all.date_account_created.apply(lambda x: x.year)
df_all['dac_month'] = df_all.date_account_created.apply(lambda x: x.month)
df_all['dac_weekday'] = df_all.date_account_created.apply(lambda x: x.weekday())
df_all = df_all.drop(['date_account_created'], axis=1)
#add dfb feature that the score will be bad because of the lossing data is too many.
'''
def dfb_map(x):
    if x!= '-1':
        return list(map(int, x.split('-')))
    else:
        return [-1,-1,-1]
#date_first_booking
dfb = np.vstack(df_all.date_first_booking.astype(str).apply(dfb_map).values)
df_all['dfb_year'] = dfb[:,0]
df_all['dfb_month'] = dfb[:,1]
'''
#timestamp_first_active
df_all['timestamp_first_active'] = df_all.timestamp_first_active.astype(str).apply(lambda x: datetime.datetime.strptime(x[:8],'%Y%m%d'))
#tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = df_all.timestamp_first_active.apply(lambda x: x.year)
df_all['tfa_month'] = df_all.timestamp_first_active.apply(lambda x: x.month)
df_all['tfa_weekday'] = df_all.timestamp_first_active.apply(lambda x: x.weekday())
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#tuned the age's parameter from 60 to 100  and then I found the upperbound of age about 85 is better.
#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>85), float('nan'), av)
#df_all.age = df_all.age.fillna(df_all.age.median())

#Filling nan
df_all = df_all.fillna(-1)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Session features
df_sessions = pd.read_csv("./input/sessions.csv", encoding='utf8')
users = df_all.id
df_sessions = df_sessions.dropna(subset=["user_id"])
device_freq = df_sessions.groupby('user_id').device_type.value_counts()
action_freq = df_sessions.groupby('user_id').action.value_counts()

def feature_dict(df):
    f_dict = dict(list(df.groupby(level='user_id')))
    res = {}
    for k, v in f_dict.items():
        v.index = v.index.droplevel('user_id')
        res[k] = v.to_dict()
    return res

#make a dictionary with the frequencies { 'user_id' : {"IPhone": 2, "Windows": 1}}
action_dict = feature_dict(action_freq)
device_dict = feature_dict(device_freq)

#transform to a list of dictionaries
action_rows = [action_dict.get(k, {}) for k in users]
device_rows = [device_dict.get(k, {}) for k in users]

device_transf = DictVectorizer()
tf = device_transf.fit_transform(device_rows)

action_transf = DictVectorizer()
tf2 = action_transf.fit_transform(action_rows)

#concatenate the two datasets
features = sp.hstack([tf, tf2])

df_sess_features = pd.DataFrame(features.todense())
df_sess_features['id'] = users
df_all = pd.merge(df_all, df_sess_features, how='left', left_on='id', right_on='id')
df_all = df_all.drop(['id'], axis=1)
#release memory
del df_sessions
del device_freq
del action_freq

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.25, n_estimators=43,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)                  

print('scores:', NDCG.cross_validation_score(X, labels,xgb,5))
'''
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)  

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)
'''
