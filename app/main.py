#%%
import numpy as np
import pandas as pd
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score,f1_score, \
    precision_score, recall_score, roc_auc_score, classification_report,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from plot_util import skew_plot, mod_sum

%load_ext autoreload
%autoreload 2

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(str(DATA) + "/data.csv")
print(df.shape)

#%%
df2 = df.copy()

# Rename label
df2.rename(columns={"default.payment.next.month": "default"}, inplace=True)

# Drop pay cols and id
cols_to_drop = ["PAY_" + str(x) for x in range(0, 7)] + ["ID"]
df2.drop(columns=cols_to_drop, inplace=True)

# Change cats to string for to make dummy col names
SEX_dict = {1: "male", 2: "female"}
EDUCATION_dict = {
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "education_others",
    5: "unknown",
    6: "unknown",
    0: "unknown",
}
MARRIAGE_dict = {1: "married", 2: "single", 3: "marriage_others", 0: "marriage_others"}
default_dict = {0: "no", 1: "yes"}

df2["SEX"] = df2["SEX"].replace(SEX_dict)
df2["EDUCATION"] = df2["EDUCATION"].replace(EDUCATION_dict)
df2["MARRIAGE"] = df2["MARRIAGE"].replace(MARRIAGE_dict)

print(df2.dtypes)
#Disp categorical counts
features = df2.select_dtypes(include=["object"]).columns
combinations_cats = df2[features].value_counts().to_frame('counts').reset_index()\
    .sort_values(by='counts', ascending=False)
display(combinations_cats)

combinations_cats2 = combinations_cats[combinations_cats['counts'] < 20]
#print(combinations_cats2)

cc = combinations_cats2.drop(columns='counts').to_numpy()

for sex, edu, mar in cc:
    df2.loc[((df2['SEX'] == sex) &
             (df2['EDUCATION'] == edu) &
             (df2['MARRIAGE'] == mar)),
            ['SEX', 'EDUCATION', 'MARRIAGE']] = 'trash'

#Disp categorical counts
features = df2.select_dtypes(include=["object"]).columns
combinations_cats = df2[features].value_counts().to_frame('counts').reset_index()\
    .sort_values(by='counts', ascending=False)
display(combinations_cats)

# Make dummy cols
df2 = pd.get_dummies(df2, columns=["SEX", "EDUCATION", "MARRIAGE"])

# Combine BILL_AMT and PAY_AMT to BAL_AMT
for month in range(1, 7):
    bill_col = "BILL_AMT" + str(month)
    pay_col = "PAY_AMT" + str(month)
    bal_col = "BAL_AMT" + str(month)
    df2[bal_col] = df2[bill_col] - df2[pay_col]
    df2 = df2.drop(columns=[bill_col, pay_col])

# Set age to float
df2["AGE"] = df2.AGE.astype("float")

# Get cont cols
cont_cols = df2.select_dtypes(include=["float"]).columns

#%%

df2.to_csv(str(DATA) + "/da_clean.csv", index=False)

#%% Print data

#print(cont_cols)
#print(df2.info())



# %%
# Check features 

print(df.columns)
#print(f"df shape: {df.shape}, df2 shape: {df2.shape}")
# print(df2.head(3))
#print(df2.dtypes)

#%% Check class balance

df_Y = df2['default']
df_class = df_Y.value_counts()

plt.figure()
plt.bar(df_Y.unique(),df_class/df_Y.shape[0])
plt.show()

#%% Explore qt transform

def qt_deskew(df, col_name):
    df_func = df.copy()
    #Raw skew
    print(df_func[col_name].skew())
    skew_plot(df_func,col_name)

    qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=0)
    df_func["new_col"] = qt.fit_transform(df_func[col_name].to_numpy().reshape(-1, 1))

    print(df_func["new_col"].skew())
    skew_plot(df_func,'new_col')

    return


for col in cont_cols:
    print(col)
    qt_deskew(df2,col)

#%%
# Do qt transform

df3 = df2.copy()
print(df3.shape)
df_Y = df3['default']
df3 = df3.drop(columns='default')
print(df3.shape)

(X_train,X_test, y_train, y_test) = train_test_split(df3, df_Y, test_size=0.2,
                                                     random_state=42, stratify=df_Y)

feat_names = cont_cols #X_train.columns

X_train_cont = X_train[cont_cols]
X_train_rest = X_train.drop(columns=cont_cols)
X_test_cont = X_test[cont_cols]
X_test_rest = X_test.drop(columns=cont_cols)


qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=0)
qt_arr = qt.fit_transform(X_train_cont.to_numpy())
X_train_qt_cont = pd.DataFrame(qt_arr, columns=[cont_cols])

X_train_qt = pd.concat([X_train_rest.reset_index(drop=True),
                        X_train_qt_cont.reset_index(drop=True)], axis=1)

X_test_qt_arr = qt.transform(X_test_cont.to_numpy())
X_test_qt_cont = pd.DataFrame(X_test_qt_arr, columns=[cont_cols])
X_test_qt = pd.concat([X_test_rest.reset_index(drop=True),
                        X_test_qt_cont.reset_index(drop=True)], axis=1)

sm = SMOTE()
X, y = sm.fit_resample(X_train_qt, y_train)


#%%

eval_set = [(X,y), (X_test_qt, y_test)]
eval_metrics = ['logloss']

clf = XGBClassifier(use_label_encoder=False, 
                        n_estimators = 100,random_state=42,
                        tree_method='auto', max_depth=5)

xgb = clf.fit(X,y,
             eval_set=eval_set,
             eval_metric=eval_metrics,
             #early_stopping_rounds=5,
             verbose=True)

y_pred = xgb.predict(X_test_qt)
y_pred_prob = xgb.predict_proba(X_test_qt)

#%%

mod_sum(y_test, y_pred, y_pred_prob[:, 1])

#%% ------------------------------------- RANDOM SEARCH --------------------------------
%%time
eval_set = [(X,y), (X_test_qt, y_test)]
eval_metric = ['logloss']
xgb_class = XGBClassifier(use_label_encoder=False, verbosity=1)#, tree_method="gpu_hist")
                    
params = {
    'min_child_weight': np.random.uniform(0.1, 10, 65), # [0.1, 1, 5],
    #'gamma': np.random.uniform(0.01, 0.7, 65), # [0.5, 1, 1.5, 2],
    'subsample':        np.random.uniform(0.5, 1, 65), # [0.6, 0.8],
    'colsample_bytree': [0.5, 0.65, 0.8, 0.95], #np.random.uniform(0.5, 0.9, 65),
    'max_depth':        np.random.randint(3, 12, 65), # [5, 10, 15],
    'learning_rate':    [0.01, 0.05, 0.1, 0.15, 0.2],
    #'n_estimators': [50, 100, 150],
    'reg_alpha':        [0.0001, 0.001, 0.1, 1, 10],
    'reg_lambda':       [0.0001, 0.001, 0.1, 1, 10]
    }

clf_rand = RandomizedSearchCV(xgb_class, params, scoring='accuracy',
                                n_iter=65, cv=5, n_jobs=7, return_train_score=True,
                                refit=True, random_state=42, verbose=2)

xgb_model_fit = clf_rand.fit(X, y , eval_metric = eval_metric,
                    eval_set=eval_set, 
                    early_stopping_rounds=5,
                    verbose=True)

#%%

print(xgb_model_fit.cv_results_.keys())


#%%

def rand_result(grid_result):
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print('\n')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        #print("%f (%f) with: %r" % (mean, stdev, param))
        print(f'Fold accuracy mean: {mean:.2f} ({stdev:.3f}) \n Params {param} \n')

rand_result(xgb_model_fit)

#%%

eval_set = [(X, y), (X_test_qt, y_test)]
eval_metric = ['auc']
clf = XGBClassifier(use_label_encoder=False, random_state=42,**xgb_model_fit.best_params_)
best_model = clf.fit(X, y,
                    eval_set=eval_set, eval_metric=eval_metric,
                    #early_stopping_rounds=5, 
                    verbose=True)

y_pred_qt = best_model.predict(X_test_qt)
y_pred_prob_qt = best_model.predict_proba(X_test_qt)

# %%
#training_plot(best_model)

mod_sum(y_test,y_pred_qt, y_pred_prob_qt[:, 1])




#%% ---------------------- Neural Network ----------------------------------
import tensorflow as tf
from tensorflow import keras
import os

print(tf.__version__)

#%%


#Common parts
n_cols = X_pca3.shape[1]
print(n_cols)
WORKING_DIR = os.getcwd() 

#Input numeric matrix
data_input = keras.Input(shape=(n_cols,), name="multi_data")
#Make deep model-----------------------------------------------------------
d_x = tf.keras.layers.Dense(256, activation='relu')(data_input)
d_x = tf.keras.layers.Dense(256, activation='relu')(d_x)
d_x = tf.keras.layers.Dense(128, activation='relu')(d_x)
d_x = tf.keras.layers.Dropout(.2)(d_x)
d_x = tf.keras.layers.Dense(128, activation='relu')(d_x)
d_x = tf.keras.layers.Dense(64, activation='relu')(d_x)
d_x = tf.keras.layers.Dense(64, activation='relu')(d_x)
d_x = tf.keras.layers.Dense(32, activation='relu')(d_x)
d_x = tf.keras.layers.Dropout(.2)(d_x)
d_x = tf.keras.layers.Dense(32, activation='relu')(d_x)

output = tf.keras.layers.Dense(8, activation='softmax')(d_x)

deep_model = keras.Model(inputs=data_input,
                        outputs=output)

deep_model.compile(optimizer='adam',
                loss=['categorical_crossentropy'],
                metrics=['accuracy'])



#%%

history = deep_model.fit(X, 
                y, 
                validation_data=(X_test_qt,y_test),
                batch_size=64, epochs=300,
                callbacks=[#tf.keras.callbacks.EarlyStopping(patience=5),
                        tf.keras.callbacks.ModelCheckpoint(WORKING_DIR,
                                                monitor='val_accuracy', verbose=2,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='auto')])

frame = pd.DataFrame(history.history)
