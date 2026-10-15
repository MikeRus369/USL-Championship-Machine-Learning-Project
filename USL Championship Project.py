"""USL Championship Match Outcome Neural Network

CS 435 - Applied Deep Learning
Predicts home loss / draw / home win from each team's previous five matches.

The program uses the Keras 3 API. It prefers TensorFlow when TensorFlow is
installed. In environments without TensorFlow, Keras uses the PyTorch backend
so the same submitted code remains runnable for verification.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
from collections import defaultdict, deque
from pathlib import Path

# Keras 3 normally defaults to TensorFlow. Keep that when available; otherwise
# use Keras' PyTorch backend so the script can still run in lightweight setups.
if importlib.util.find_spec("tensorflow") is None:
    os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
LOOKBACK = 5
CLASS_NAMES = ["Home loss", "Draw", "Home win"]
FEATURE_NAMES = [
    "home_points_per_match",
    "home_goals_for_per_match",
    "home_goals_against_per_match",
    "home_win_rate",
    "home_draw_rate",
    "home_goal_difference_per_match",
    "away_points_per_match",
    "away_goals_for_per_match",
    "away_goals_against_per_match",
    "away_win_rate",
    "away_draw_rate",
    "away_goal_difference_per_match",
    "points_per_match_difference",
    "goal_difference_difference",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uslc_2024.csv"
OUTPUT_DIR = BASE_DIR / "usl_model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def summarize_recent(games: deque[tuple[int, int, int]]) -> list[float]:
    goals_for = np.mean([g[0] for g in games])
    goals_against = np.mean([g[1] for g in games])
    points = np.mean([g[2] for g in games])
    win_rate = np.mean([g[2] == 3 for g in games])
    draw_rate = np.mean([g[2] == 1 for g in games])
    goal_difference = np.mean([g[0] - g[1] for g in games])
    return [
        float(points),
        float(goals_for),
        float(goals_against),
        float(win_rate),
        float(draw_rate),
        float(goal_difference),
    ]


def prepare_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name}. Place uslc_2024.csv in the same folder as this script."
        )

    raw = pd.read_csv(path)
    required = {"round", "date", "start_time", "home_team", "away_team", "home_score", "away_score", "match_id"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    raw = raw[raw["round"].eq("Regular Season")].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["home_score"] = pd.to_numeric(raw["home_score"], errors="coerce")
    raw["away_score"] = pd.to_numeric(raw["away_score"], errors="coerce")
    raw = raw.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    raw = raw.drop_duplicates(subset=["date", "home_team", "away_team"])
    raw = raw.sort_values(["date", "start_time", "match_id"]).reset_index(drop=True)

    histories: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=LOOKBACK))
    prepared_rows: list[dict] = []

    for row in raw.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_score = int(row.home_score)
        away_score = int(row.away_score)

        # Create a feature row BEFORE adding the current match to either history.
        # This is what prevents target leakage.
        if len(histories[home]) >= LOOKBACK and len(histories[away]) >= LOOKBACK:
            home_form = summarize_recent(histories[home])
            away_form = summarize_recent(histories[away])
            label = 0 if home_score < away_score else 1 if home_score == away_score else 2

            values = home_form + away_form + [
                home_form[0] - away_form[0],
                home_form[5] - away_form[5],
            ]
            prepared_rows.append(
                {
                    "date": row.date,
                    "home_team": home,
                    "away_team": away,
                    **dict(zip(FEATURE_NAMES, values)),
                    "outcome": label,
                    "outcome_name": CLASS_NAMES[label],
                }
            )

        home_points = 3 if home_score > away_score else 1 if home_score == away_score else 0
        away_points = 3 if away_score > home_score else 1 if home_score == away_score else 0
        histories[home].append((home_score, away_score, home_points))
        histories[away].append((away_score, home_score, away_points))

    prepared = pd.DataFrame(prepared_rows)
    prepared.to_csv(OUTPUT_DIR / "prepared_usl_match_data.csv", index=False)
    return prepared


def chronological_split(prepared: pd.DataFrame):
    x = prepared[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = prepared["outcome"].to_numpy(dtype=np.int64)
    n = len(prepared)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:validation_end], y[train_end:validation_end]
    x_test, y_test = x[validation_end:], y[validation_end:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), train_end, validation_end


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(len(FEATURE_NAMES),)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(3, activation="softmax"),
        ],
        name="usl_match_outcome_model",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_line_plot(history: dict, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(history[metric], label="Training")
    ax.plot(history[f"val_{metric}"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Training and Validation {ylabel}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title("Confusion Matrix")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_class_distribution(prepared: pd.DataFrame) -> None:
    counts = prepared["outcome_name"].value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Prepared matches")
    ax.set_title("Outcome Distribution in Prepared Dataset")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)


def save_per_class_metrics(precision, recall, f1) -> None:
    x = np.arange(3)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Test Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_metrics.png", dpi=180)
    plt.close(fig)


def save_probability_examples(test_rows: pd.DataFrame, probabilities: np.ndarray) -> None:
    rows = test_rows.tail(6).copy().reset_index(drop=True)
    probs = probabilities[-6:]
    labels = [f"{r.home_team}\nvs {r.away_team}" for r in rows.itertuples(index=False)]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bottom = np.zeros(len(rows))
    for class_index, name in enumerate(CLASS_NAMES):
        vals = probs[:, class_index]
        ax.bar(x, vals, bottom=bottom, label=name)
        bottom += vals
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted probability")
    ax.set_title("Example Predictions from the Final Test Date")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_probabilities.png", dpi=180)
    plt.close(fig)


def main() -> None:
    set_seeds()
    prepared = prepare_data(DATA_PATH)
    train, val, test, train_end, val_end = chronological_split(prepared)
    x_train, y_train = train
    x_val, y_val = val
    x_test, y_test = test

    model = build_model()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0,
    )

    probabilities = model.predict(x_test, verbose=0)
    predicted = probabilities.argmax(axis=1)
    accuracy = accuracy_score(y_test, predicted)
    cm = confusion_matrix(y_test, predicted, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=[0, 1, 2], zero_division=0
    )

    baseline_class = int(np.bincount(y_train).argmax())
    baseline_predictions = np.full_like(y_test, baseline_class)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    test_rows = prepared.iloc[val_end:].copy().reset_index(drop=True)
    test_rows["actual_outcome"] = [CLASS_NAMES[i] for i in y_test]
    test_rows["predicted_outcome"] = [CLASS_NAMES[i] for i in predicted]
    for i, name in enumerate(CLASS_NAMES):
        test_rows[f"prob_{name.lower().replace(' ', '_')}"] = probabilities[:, i]
    test_rows.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    metrics = {
        "keras_backend": keras.backend.backend(),
        "raw_matches": 408,
        "prepared_matches": int(len(prepared)),
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "epochs_completed": int(len(history.history["loss"])),
        "test_accuracy": float(accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        "correct_test_predictions": int((predicted == y_test).sum()),
        "confusion_matrix": cm.tolist(),
        "class_metrics": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "test_start_date": str(test_rows["date"].min().date()),
        "test_end_date": str(test_rows["date"].max().date()),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    save_class_distribution(prepared)
    save_line_plot(history.history, "accuracy", "Accuracy", "training_accuracy.png")
    save_line_plot(history.history, "loss", "Loss", "training_loss.png")
    save_confusion_matrix(cm)
    save_per_class_metrics(precision, recall, f1)
    save_probability_examples(test_rows, probabilities)
    model.save(OUTPUT_DIR / "usl_match_outcome_model.keras")

    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Prepared rows: {len(prepared)}")
    print(f"Train / validation / test: {len(y_train)} / {len(y_val)} / {len(y_test)}")
    print(f"Epochs completed: {len(history.history['loss'])}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
