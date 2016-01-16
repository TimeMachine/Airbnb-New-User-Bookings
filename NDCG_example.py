import pandas as pd
from xgboost.sklearn import XGBClassifier
import NDCG

df_train = pd.read_csv('./input/train_users.csv')
# data output
truth = df_train['country_destination'].values
# format the data 
df_all = df_train.drop(['id', 'date_first_booking','date_account_created','timestamp_first_active','age','country_destination'], axis=1)
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
# data input
preds = df_all.values
# model
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
# call validation
print NDCG.cross_validation_score(preds,truth,xgb,3)
