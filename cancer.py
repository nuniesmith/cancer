"""
Optuna + LightGBM: Breast Cancer Classifier Hyperparameter Optimization
- Fixes data leakage (split done once outside objective)
- Adds early stopping
- Launches Optuna Dashboard on port 8089
- Saves study to SQLite for dashboard persistence
"""

import threading

import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import optuna_dashboard
import sklearn.datasets
import sklearn.decomposition
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

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


def save_plots(study, valid_y, best_preds):
    """Generate and save result + dataset overview plots after training."""
    ds = sklearn.datasets.load_breast_cancer()
    X_raw, y = ds.data, ds.target

    ACCENT = "#00d4ff"
    GREEN = "#00ff88"
    ORANGE = "#ff8c00"
    RED = "#ff4444"
    BG = "#1a1d2e"
    TEXT = "#e0e0e0"

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    trials = [t for t in study.trials if t.value is not None]
    accs = [t.value for t in trials]
    aucs = [t.user_attrs.get("auc", 0) for t in trials]
    n_ests = [t.user_attrs.get("n_estimators", 0) for t in trials]
    nums = [t.number for t in trials]
    best_acc = max(accs)
    running_best = [max(accs[: i + 1]) for i in range(len(accs))]

    # ── Plot 1: Optimization Dashboard ───────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1a. Optimization history
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(
        nums,
        accs,
        c=accs,
        cmap="RdYlGn",
        s=60,
        alpha=0.7,
        zorder=3,
        vmin=min(accs),
        vmax=1.0,
    )
    ax1.plot(nums, running_best, color=ACCENT, lw=2, label="Best so far", zorder=4)
    ax1.axhline(best_acc, color=GREEN, lw=1.2, ls="--", alpha=0.6)
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Accuracy")
    ax1.legend(facecolor=BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)
    style_ax(ax1, "Optimization History — Accuracy per Trial")

    # 1b. AUC vs Accuracy scatter
    ax2 = fig.add_subplot(gs[0, 2])
    sc = ax2.scatter(
        accs, aucs, c=n_ests, cmap="plasma", s=70, alpha=0.85, edgecolors="#333", lw=0.4
    )
    cb = fig.colorbar(sc, ax=ax2)
    cb.set_label("# Trees", color=TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)
    ax2.set_xlabel("Accuracy")
    ax2.set_ylabel("AUC-ROC")
    style_ax(ax2, "AUC vs Accuracy (color = # Trees)")

    # 1c. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 0])
    cm = sklearn.metrics.confusion_matrix(valid_y, np.rint(best_preds))
    cmap = LinearSegmentedColormap.from_list("custom", ["#0f1117", "#00d4ff"])
    ax3.imshow(cm, cmap=cmap, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax3.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] < cm.max() * 0.7 else "#0f1117",
                fontsize=18,
                fontweight="bold",
            )
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred Mal.", "Pred Ben."], color=TEXT, fontsize=9)
    ax3.set_yticklabels(["True Mal.", "True Ben."], color=TEXT, fontsize=9)
    style_ax(ax3, f"Confusion Matrix (Best Trial)\nAcc = {best_acc:.4f}")

    # 1d. ROC Curve
    ax4 = fig.add_subplot(gs[1, 1])
    fpr, tpr, _ = sklearn.metrics.roc_curve(valid_y, best_preds)
    auc_val = sklearn.metrics.roc_auc_score(valid_y, best_preds)
    ax4.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax4.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc_val:.4f}")
    ax4.plot([0, 1], [0, 1], color="#555", lw=1, ls="--")
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend(facecolor=BG, edgecolor=ACCENT, labelcolor=TEXT, fontsize=9)
    style_ax(ax4, "ROC Curve (Best Model)")

    # 1e. Learning rate: top vs bottom 25%
    ax5 = fig.add_subplot(gs[1, 2])
    lrs = [t.params.get("learning_rate", np.nan) for t in trials]
    p75 = np.percentile(accs, 75)
    p25 = np.percentile(accs, 25)
    lrs_top = [v for v, a in zip(lrs, accs) if a >= p75]
    lrs_bot = [v for v, a in zip(lrs, accs) if a < p25]
    ax5.hist(lrs_top, bins=10, color=GREEN, alpha=0.7, label="Top 25%", density=True)
    ax5.hist(lrs_bot, bins=10, color=RED, alpha=0.7, label="Bot 25%", density=True)
    ax5.set_xlabel("Learning Rate")
    ax5.set_ylabel("Density")
    ax5.legend(facecolor=BG, edgecolor="#555", labelcolor=TEXT, fontsize=9)
    ax5.set_xscale("log")
    style_ax(ax5, "Learning Rate: Top vs Bottom Trials")

    fig.suptitle(
        "LightGBM Cancer Classifier — Optuna Hyperparameter Search",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    p1 = "optuna_results.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"📊  Saved → {p1}")

    # ── Plot 2: Dataset Overview ──────────────────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")

    # PCA
    pca = sklearn.decomposition.PCA(n_components=2)
    X_sc = StandardScaler().fit_transform(X_raw)
    X_pca = pca.fit_transform(X_sc)
    for cls, color, label in [(0, RED, "Malignant"), (1, GREEN, "Benign")]:
        mask = y == cls
        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            s=25,
            alpha=0.7,
            label=label,
            edgecolors="none",
        )
    axes[0].set_facecolor(BG)
    axes[0].tick_params(colors=TEXT)
    axes[0].set_title("PCA — 2D Projection", color=TEXT, fontweight="bold")
    axes[0].legend(facecolor=BG, edgecolor="#555", labelcolor=TEXT)
    axes[0].set_xlabel("PC1", color=TEXT)
    axes[0].set_ylabel("PC2", color=TEXT)
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#333355")

    # Top 5 discriminative features
    t_scores = [
        abs(stats.ttest_ind(X_raw[y == 0, i], X_raw[y == 1, i]).statistic)
        for i in range(X_raw.shape[1])
    ]
    top5_idx = np.argsort(t_scores)[-5:][::-1]
    x_pos, w = np.arange(5), 0.35
    col_min = X_raw[:, top5_idx].min(0)
    col_range = X_raw[:, top5_idx].max(0) - col_min
    m0n = (X_raw[y == 0][:, top5_idx].mean(0) - col_min) / col_range
    m1n = (X_raw[y == 1][:, top5_idx].mean(0) - col_min) / col_range
    axes[1].bar(x_pos - w / 2, m0n, w, color=RED, alpha=0.8, label="Malignant")
    axes[1].bar(x_pos + w / 2, m1n, w, color=GREEN, alpha=0.8, label="Benign")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(
        [ds.feature_names[i][:14] for i in top5_idx],
        rotation=30,
        ha="right",
        fontsize=8,
        color=TEXT,
    )
    axes[1].set_facecolor(BG)
    axes[1].tick_params(colors=TEXT)
    axes[1].set_title(
        "Top 5 Discriminative Features\n(normalised mean)",
        color=TEXT,
        fontweight="bold",
    )
    axes[1].legend(facecolor=BG, edgecolor="#555", labelcolor=TEXT)
    axes[1].set_ylabel("Normalised Mean", color=TEXT)
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#333355")

    # Class balance
    counts = [np.sum(y == 0), np.sum(y == 1)]
    bars = axes[2].bar(
        ["Malignant (0)", "Benign (1)"],
        counts,
        color=[RED, GREEN],
        alpha=0.85,
        width=0.5,
        edgecolor="#333",
        linewidth=0.8,
    )
    for bar, cnt in zip(bars, counts):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(cnt),
            ha="center",
            color=TEXT,
            fontsize=12,
            fontweight="bold",
        )
    axes[2].set_facecolor(BG)
    axes[2].tick_params(colors=TEXT)
    axes[2].set_title(
        "Dataset Class Balance\n(569 samples total)", color=TEXT, fontweight="bold"
    )
    axes[2].set_ylabel("Count", color=TEXT)
    for spine in axes[2].spines.values():
        spine.set_edgecolor("#333355")

    fig2.suptitle(
        "Breast Cancer Wisconsin Dataset — Overview",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
    )
    fig2.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    p2 = "dataset_overview.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)
    print(f"📊  Saved → {p2}")


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

    # ── Retrain best model to get predictions for plots ───────────────────────
    print("\n⏳  Retraining best model for plots...")
    best_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        **study.best_trial.params,
    }
    dtrain_final = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    dvalid_final = lgb.Dataset(
        valid_x, label=valid_y, reference=dtrain_final, free_raw_data=False
    )
    best_gbm = lgb.train(
        best_params,
        dtrain_final,
        num_boost_round=500,
        valid_sets=[dvalid_final],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    best_preds = best_gbm.predict(valid_x)

    # ── Save plots ────────────────────────────────────────────────────────────
    save_plots(study, valid_y, best_preds)

    # Keep server alive after optimization
    print("\nDashboard still running — press Ctrl+C to stop.")
    t.join()
