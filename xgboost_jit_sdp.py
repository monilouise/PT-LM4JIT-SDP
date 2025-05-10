from xgboost import XGBClassifier
import pickle
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def normalize_df(df, feature_columns):
    df["fix"] = df["fix"].apply(lambda x: float(bool(x)))
    df = df.astype({i: "float32" for i in feature_columns})
    return df[["commit_hash"] + feature_columns]
    
def load_data():
    with open('data/features_train.pkl', 'rb') as f:
        features_train = pickle.load(f)
        features_train = features_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_train = preprocess(features_train)
        y_train = features_train['is_buggy_commit']

    with open('data/features_test.pkl', 'rb') as f:
        features_test = pickle.load(f)
        X_test = preprocess(features_test)
        y_test = features_test['is_buggy_commit']

    return X_train, y_train, X_test, y_test

    
def preprocess(fedata):
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = preprocessing.scale(fedata[manual_features_columns].to_numpy())
    fedata[manual_features_columns] = manual_features
    fedata = fedata.drop('commit_hash', axis=1)

    return fedata

X_train, y_train, X_test, y_test = load_data()
bst = XGBClassifier()

total_pos = sum(y_train)
total_neg = len(y_train) - total_pos
ratio = total_neg/total_pos
print('scale_pos_weight = ', ratio)

# fit model
bst = XGBClassifier(scale_pos_weight=ratio)
bst.fit(X_train, y_train)

y_prob = bst.predict_proba(X_test)[:, 1]  
y_pred = bst.predict(X_test)  

roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

