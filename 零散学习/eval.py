from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, \
    fbeta_score, confusion_matrix

# confusion matrix

y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# P,R,f1
alg_conf = {}
alg_conf["Mmetrics"] = {
    'precision_score:': precision_score,  # MSE,
    'recall_score': recall_score,  # MSLE
    'f1_score:': f1_score  # MedAE
}

y_true = [0, 1, 0, 1]
y_pred = [0, 1, 0, 0]
for key, valid_metrics in alg_conf["Mmetrics"].items():
    print(key, valid_metrics(y_true, y_pred))

# log-loss
alg_conf = {}
alg_conf["Mmetrics"] = {
    'log_loss:': log_loss  # log_loss
}

y_true = [0, 0, 1, 1]
y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
for key, valid_metrics in alg_conf["Mmetrics"].items():
    print(key, valid_metrics(y_true, y_pred))
