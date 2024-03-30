import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from engineering import EngineeringTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate


def my_cross_validate(models, X_train, y_train, scoring, cv=6, n_jobs=-1):
    rows = len(models)
    cols = len(scoring)
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3 * rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    all_scores = {}
    base_results = cross_validate(
        DummyClassifier(strategy="uniform"),
        X_train,
        y_train,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    base_scores = [
        (name, base_results[name]) for name in base_results if name.startswith("test_")
    ]
    all_scores["base"] = base_scores

    labels = ["Fold {}".format(i) for i in range(1, cv + 1)]
    x = np.arange(len(labels))
    width = 0.35

    for j, model in enumerate(models):
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv, n_jobs=n_jobs, scoring=scoring
        )

        model_name = model.__class__.__name__.split(".")[-1]
        scores = [
            (name[len("test_") :], cv_results[name])
            for name in cv_results
            if name.startswith("test_")
        ]
        all_scores[model_name] = scores

        for i, (name, score) in enumerate(scores):
            org_i = i
            i += j * len(scoring)

            mean_score = np.mean(score)
            std_score = np.std(score)
            base_score = np.mean(base_scores[org_i][1])

            # print(f"{name:.10s} - mean score: {mean_score:2.6f}, std: {std_score:2.6f}. Precise: {score}")

            bars = sns.barplot(x=x, y=score, ax=axes[i], color="skyblue")
            axes[i].axhline(
                y=mean_score,
                color="red",
                linestyle="--",
                label=f"Mean ({mean_score:1.3f})",
            )
            axes[i].axhline(
                y=mean_score + std_score,
                color="orange",
                linestyle="--",
                label=f"+- std ({std_score:1.3f})",
            )
            axes[i].axhline(y=mean_score - std_score, color="orange", linestyle="--")
            axes[i].axhline(
                y=base_score,
                color="green",
                linestyle="-.",
                label=f"Base score ({base_score:1.3f})",
            )

            if org_i == 0:
                axes[i].set_ylabel(f"{model_name}")
            if j == 0:
                axes[i].set_title(f"{name}")
            axes[i].set_xticklabels(labels)
            axes[i].set_xticks(x)
            axes[i].set_ylim(0, 1)

            for bar, precise_score in zip(bars.patches, score):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.1,
                    f"{precise_score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                )

            axes[i].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.xlabel("Fold")
    plt.tight_layout()
    plt.show()

    return all_scores


def plot_scoring(scoring, scores):
    plt.figure(figsize=(12, 8))
    sns.set_palette(sns.color_palette("tab20"))

    cv = len(list(scores.values())[0][1][1])
    rows = int(np.ceil(len(scoring) / 2))
    cols = 2
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3 * rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for model_name, model_scores in scores.items():
        for i, (metric_name, metric_scores) in enumerate(model_scores):
            linestyle = "-."
            alpha = 0.5
            if model_name == "base":
                linestyle = "-"
                alpha = 1
            sns.lineplot(
                x=range(len(metric_scores)),
                y=metric_scores,
                label=f"{model_name}",
                ax=axes[i],
                linestyle=linestyle,
                alpha=alpha,
            )

            sns.scatterplot(
                x=[len(metric_scores)],
                y=[np.mean(metric_scores)],
                ax=axes[i],
                alpha=alpha,
            )

    for i in range(len(scoring)):
        for j, (model_name, _) in enumerate(scoring.items()):
            axes[j].set_title(f"{model_name}")

        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].set_ylim(0, 1)
        axes[i].get_legend().remove()
        axes[i].set_xlabel("Fold")
        axes[i].set_ylabel("score")

        axes[i].set_xticklabels(
            [f"Fold {i+1}" if i < cv else "Mean Score" for i in range(-1, cv + 1)]
        )
        axes[i].tick_params(axis="x", rotation=45)

    axes[min(len(scoring) - 1, cols - 1)].legend(
        fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left"
    )

    for i in range(rows * cols, len(scoring), -1):
        axes[i - 1].axis("off")

    plt.tight_layout()
    plt.show()


def print_scores(scores, score_name):
    print(f"Models {score_name}")

    l = []
    for model_name, score in scores.items():
        for s_name, s_values in score:
            if s_name == score_name:
                l.append((model_name, np.mean(s_values)))

    l.sort(key=lambda x: x[1], reverse=True)
    for model_name, mean_s in l:
        print(f"\t{model_name:30s}{mean_s}")


def create_best_estimator(
    study,
    X_train,
    y_train,
    X_valid,
    y_valid,
    estimator_class,
):
    trial = study.best_trial

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    estimator = estimator_class(**trial.params)
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_valid)
    score = f1_score(y_valid, y_pred, average="micro")

    print(f"Refitted best model f1-score: {score}")
    return estimator
