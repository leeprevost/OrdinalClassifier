from models.ordinal import OrdinalClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np


data = load_diabetes(as_frame=True)
kb_q = KBinsDiscretizer(3, encode='ordinal', strategy='quantile')
kb_k = KBinsDiscretizer(3, encode='ordinal', strategy="kmeans")
kb_u = KBinsDiscretizer(3, encode='ordinal', strategy='uniform')

y = data.frame.target
y_q = pd.Series(kb_q.fit_transform(y.to_frame()).reshape(-1), y.index, name = y.name + "_q")
y_k = pd.Series(kb_k.fit_transform(y.to_frame()).reshape(-1), y.index, name=y.name + "_k")
y_u = pd.Series(kb_u.fit_transform(y.to_frame()).reshape(-1), y.index, name=y.name + "_u")

#pd.concat((y_q, y_k, y_u), axis=1).hist()

disease_class_progression = ["low", "medium", "high"]
y_q_c = pd.qcut(y, 3, labels=disease_class_progression)  #named progression using pandas categories and quantile cut.

assert abs((y_q == y_q_c.cat.codes).value_counts().pct_change().loc[False]) > 0.99  # assert pd.qcut and sk qcut methods mostly agree
print(y_q_c) # note category order

X = data.frame.iloc[:, 0:-1]

ys = (y_k, y_k, y_u, y_q_c)

random_state = 34

# test it using cross validation against an SVC

svc_bal = SVC(kernel='linear', class_weight="balanced", probability=True)
svc_imb = SVC(kernel='linear', class_weight=None, probability=True)

oc_bal = OrdinalClassifier(svc_bal)
oc_imb = OrdinalClassifier(svc_imb)

models = [svc_bal, svc_imb, oc_bal, oc_imb]

oc_params = [{"reverse_classes": True}, {'reverse_classes': False}]

tests = []
for y_t in ys:
    X_train, X_test, y_train, y_test = train_test_split(X, y_t, shuffle=True, test_size=0.2, random_state=random_state, stratify=y_t)

    ord_pass = 0
    for model in models:
        param ={}
        if "Ordinal" in model.__repr__():
            if ord_pass == 0:
                param = oc_params[ord_pass]  #first time through
                ord_pass +=1
            elif ord_pass==1:
                param = oc_params[ord_pass]  # second pass
        model.set_params(**param)

        mod_name = model.__repr__()

        cv = cross_validate(model, X_train, y_train,  n_jobs=-1, scoring="f1_weighted")
        cv["clf"] = mod_name
        cv['param'] = param
        cv['y_name'] = y_t.name

        model.fit(X_test, y_test)

        print(model.__repr__())
        print("params are: ".format(param))
        try:
            #this gets into using pred_proba
            cr = classification_report(y_test, model.predict(X_test), output_dict=True)
            print(classification_report(y_test, model.predict(X_test)))
            print("Cross Validation:")
            print(cv)
            cv['cr on test'] = cr
            if cr.get("2.0"):
                pos_class = '2.0'
                neg_class = '0.0'

            else:
                pos_class = 'high'
                neg_class = 'low'
                cv['test_on_pos_class'] = cr['high']  # SVC sorts classes lexographically in OVR .  can';t look at last and first
                cv['test_on_neg_class'] = cr['low']
            cv['test_on_pos_class'] = cr[pos_class]  # SVC sorts classes lexographically in OVR .  can';t look at last and first
            cv['test_on_neg_class'] = cr[neg_class]
            cv['pos_class'] = pos_class
        except Exception as e:
            cv['error'] = e

        cv["test_score_mean"] = cv['test_score'].mean()
        cv['test_score_std'] = cv['test_score'].std()
        cv["classes_"] = model.classes_
        tests.append(cv)

test_df = pd.DataFrame(tests)
test_df.to_csv("evaluation_ordinal.csv")




