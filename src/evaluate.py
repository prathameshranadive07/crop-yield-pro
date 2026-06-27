from sklearn.metrics import r2_score, mean_absolute_error

def evaluate(y_true, y_pred):
    print("R2 Score:", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))