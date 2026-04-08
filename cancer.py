"""
Optuna + LightGBM: Breast Cancer Classifier Hyperparameter Optimization
- Fixes data leakage (split done once outside objective)
- Adds early stopping
- Launches Optuna Dashboard on port 8089
- Saves study to SQLite for dashboard persistence
"""

import threading

import lightgbm as lgb
import numpy as np
import optuna
import optuna_dashboard
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Suppress verbose optuna/lgb logs ──────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load & split data ONCE (fixes per-trial leakage) ─────────────────────────
data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
data = scaler.fit_transform(data)

train_x, valid_x, train_y, valid_y = train_test_split(
    data, target, test_size=0.25, random_state=42, stratify=target
)
# NOTE: lgb.Dataset is created inside objective() — NOT here.
# If shared across trials, LightGBM caches it with the first trial's
# min_data_in_leaf and raises a fatal error when a later trial reduces it.


def objective(trial):
    # Recreate datasets each trial with free_raw_data=False so the numpy
    # arrays (train_x / valid_x) are not freed between calls.
    dtrain = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain, free_raw_data=False)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        # Prevents the fatal error when min_data_in_leaf shrinks between trials
        "feature_pre_filter": False,
        # Regularisation
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # Tree structure
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # Sampling
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # Learning rate (tuned)
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    gbm = lgb.train(
        param,
        dtrain,
        num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

    # Log extra metrics for dashboard
    trial.set_user_attr("auc", sklearn.metrics.roc_auc_score(valid_y, preds))
    trial.set_user_attr("n_estimators", gbm.num_trees())

    return accuracy


def launch_dashboard(storage_url: str, port: int = 8089):
    """Run Optuna Dashboard in a background thread."""
    app = optuna_dashboard.wsgi(storage_url)
    import wsgiref.simple_server as wss

    server = wss.make_server("0.0.0.0", port, app)
    print(f"\n🌐  Optuna Dashboard → http://localhost:{port}\n")
    server.serve_forever()


if __name__ == "__main__":
    STORAGE = "sqlite:///cancer_study.db"
    STUDY_NAME = "lgbm_cancer_v1"
    DASHBOARD_PORT = 8089
    N_TRIALS = 100

    # Persistent storage so dashboard survives restarts
    storage = optuna.storages.RDBStorage(STORAGE)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Launch dashboard before optimization starts
    t = threading.Thread(
        target=launch_dashboard, args=(STORAGE, DASHBOARD_PORT), daemon=True
    )
    t.start()

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nFinished trials : {len(study.trials)}")
    print(f"Best accuracy   : {study.best_trial.value:.4f}")
    print(f"Best AUC        : {study.best_trial.user_attrs['auc']:.4f}")
    print("\nBest params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Keep server alive after optimization
    print("\nDashboard still running — press Ctrl+C to stop.")
    t.join()
