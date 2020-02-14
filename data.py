import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

train_df = pd.read_csv("train_set.csv")
test_df = pd.read_csv("test_set.csv")
train_y_df = pd.read_csv("train_set.csv",usecols=["y"])
train_y = np.array(train_y_df)#np.ndarray()
# print(train_y.shape)
# train_y=train_y_np.tolist()#list
# # print(train_y)
# print(train_df.info())
# print(train_df["age"].describe())
# scalerstd_age = preprocessing.StandardScaler().fit(train_df["age"])#接收2D参数
# print(scalerstd_age.mean)
# print(scalerstd_age.var)
# print(train_df.groupby(["job"]).count())

train_df["age_scaler"] = (train_df["age"]-train_df["age"].min())/(train_df["age"].max()-train_df["age"].min())
test_df["age_scaler"] = (test_df["age"]-test_df["age"].min())/(test_df["age"].max()-test_df["age"].min())
train_df["balance_scaler"] = (train_df["balance"]-train_df["balance"].min())/(train_df["balance"].max()-train_df["balance"].min())
test_df["balance_scaler"] = (test_df["balance"]-test_df["balance"].min())/(test_df["balance"].max()-test_df["balance"].min())
# print(train_df["balance_scaler"].describe())
train_df["campaign_scaler"] = (train_df["campaign"]-train_df["campaign"].min())/(train_df["campaign"].max()-train_df["campaign"].min())
test_df["campaign_scaler"] = (test_df["campaign"]-test_df["campaign"].min())/(test_df["campaign"].max()-test_df["campaign"].min())
train_df["duration_scaler"] = (train_df["duration"]-train_df["duration"].min())/(train_df["duration"].max()-train_df["duration"].min())
test_df["duration_scaler"] = (test_df["duration"]-test_df["duration"].min())/(test_df["duration"].max()-test_df["duration"].min())
train_df["previous_scaler"] = (train_df["previous"]-train_df["previous"].min())/(train_df["previous"].max()-train_df["previous"].min())
test_df["previous_scaler"] = (test_df["previous"]-test_df["previous"].min())/(test_df["previous"].max()-test_df["previous"].min())

train_df["housing"] = train_df["housing"].map({"yes":0,"no":1})
test_df["housing"] = test_df["housing"].map({"yes":0,"no":1})
# print(train_df["housing"].head())
train_df["loan"] = train_df["loan"].map({"no":0,"yes":1})
test_df["loan"] = test_df["loan"].map({"no":0,"yes":1})
train_df["default"] = train_df["default"].map({"no":0,"yes":1})
test_df["default"] = test_df["default"].map({"no":0,"yes":1})
onehotcode = pd.get_dummies(train_df,columns=["contact","poutcome","education","marital","job"])
onehotcode_test = pd.get_dummies(test_df,columns=["contact","poutcome","education","marital","job"])
# print(onehotcode.info())
train_data = onehotcode.drop(columns=["ID","y","age","balance","campaign","duration","previous","day","month","pdays"])
test_data = onehotcode_test.drop(columns=["ID","age","balance","campaign","duration","previous","day","month","pdays"])
feature_train_np = np.array(train_data)
feature_test_np = np.array(test_data)
print(feature_train_np.shape)
print(feature_test_np.shape)
train_x =feature_train_np
test_x = feature_test_np

X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size=0.3, random_state=0)
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)
prepro = logreg.predict_proba(X_val)
y_pred = logreg.predict(X_val)
acc = logreg.score(X_val,Y_val)
print(acc)
print(classification_report(Y_val, y_pred))


'''
train_df["contact"] = train_df["contact"].map({"cellular":0,"unknown":1,"telephone":2})
test_df["contact"] = test_df["contact"].map({"cellular":0,"unknown":1,"telephone":2})
train_df["poutcome"] = train_df["poutcome"].map({"success":0,"unknown":1,"failure":2,"other":3})
test_df["poutcome"] = test_df["poutcome"].map({"success":0,"unknown":1,"failure":2,"other":3})
train_df["education"] = train_df["education"].map({"primary":0,"secondary":1,"tertiary":2})
test_df["education"] = test_df["education"].map({"primary":0,"secondary":1,"tertiary":2})
train_df["marital"] = train_df["marital"].map({"married":0,"divorced":1,"single":2})
test_df["marital"] = test_df["marital"].map({"married":0,"divorced":1,"single":2})
job_map = {"admin.":0,
"blue-collar":1,
"entrepreneur":2,
"housemaid":3,
"management":4,
"retired":5,
"self-employed":6,
"services":7,
"student":8,
"technician":9,
"unemployed":10,
"unknown":11}
train_df["job"] = train_df["job"].map(job_map)
test_df["job"] = test_df["job"].map(job_map)
print(train_df.info())

features_use_1 = ["age_scaler","balance_scaler","campaign_scaler","duration_scaler","previous_scaler"]
features_use_2 = ["housing","loan","default"]
features_use_3 = ["contact","poutcome","education","marital","job"]
contact_np = np.array(train_df.loc[:,['contact']]).tolist()
poutcome_np = np.array(train_df.loc[:,['poutcome']]).tolist()
education_np = np.array(train_df.loc[:,['education']]).tolist()
marital_np = np.array(train_df.loc[:,['marital']]).tolist()
job_np = np.array(train_df.loc[:,['job']]).tolist()
feature_contact = OneHotEncoder().fit(contact_np).transform(contact_np).toarray()
feature_poutcome = OneHotEncoder().fit(poutcome_np).transform(poutcome_np).toarray()
feature_marital = OneHotEncoder().fit(marital_np).transform(marital_np).toarray()
feature_education = OneHotEncoder().fit(education_np).transform(education_np).toarray()
feature_job = OneHotEncoder().fit(job_np).transform(job_np).toarray()

contact_tnp = np.array(test_df.loc[:,['contact']]).tolist()
poutcome_tnp = np.array(test_df.loc[:,['poutcome']]).tolist()
education_tnp = np.array(test_df.loc[:,['education']]).tolist()
marital_tnp = np.array(test_df.loc[:,['marital']]).tolist()
job_tnp = np.array(test_df.loc[:,['job']]).tolist()
feature_contact1 = OneHotEncoder().fit(contact_tnp).transform(contact_tnp).toarray()
feature_poutcome1 = OneHotEncoder().fit(poutcome_tnp).transform(poutcome_tnp).toarray()
feature_marital1 = OneHotEncoder().fit(marital_tnp).transform(marital_tnp).toarray()
feature_education1 = OneHotEncoder().fit(education_tnp).transform(education_tnp).toarray()
feature_job1 = OneHotEncoder().fit(job_tnp).transform(job_tnp).toarray()
# print(feature_contact)
'''
