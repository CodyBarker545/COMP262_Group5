"""
COMP262 - Phase 2: Machine Learning Sentiment Analysis

Dataset: Amazon Gift Cards Reviews

Phase 1:
Lexicon-based sentiment analysis (VADER + TextBlob)

Phase 2:
- Build ML models (Naive Bayes, MLP)
- Compare with lexicon models
- Enhance ratings using review text
- Save models, figures, metrics, and metadata
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


@dataclass
class Config:
    data_path: str = "Gift_Cards.json"
    output_dir: str = "artifacts"
    n_per_class: int = 2996 # change to how much you want to sample per class (max neutral is 2996) can use other techniques to balance if you want more data. Include in report for future work.
    random_state: int = 42


class TextProcessor:
    """
    Text preprocessing:
    - Lowercase
    - Basic negation handling
    - Remove non-alphabetic characters (keep !)
    - Remove stopwords (keep "not", "no")
    - Lemmatization
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english")) - {"not", "no", "never", "nor", "none", "nothing", "nowhere"}
        self.lemmatizer = WordNetLemmatizer()

    def handle_negation(self, text):
        words = str(text).split()
        negation_words = {"not", "no", "never", "nor", "none", "nothing", "nowhere"}

        result = []
        negate_window = 0

        for word in words:
            if word.lower() in negation_words:
                negate_window = 3
                result.append(word)
                continue

            if negate_window > 0:
                result.append("NOT_" + word)
                negate_window -= 1
            else:
                result.append(word)

        return " ".join(result)
    
    def expand_contractions(self, text):
        text = str(text)
        text = text.replace("n't", " not")
        return text
    
    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s!]", "", text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    def label_sentiment(self, rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"


class FeatureBuilder:
    """
    TF-IDF chosen because:
    - Captures word importance across documents
    - Works well with ML models (Naive Bayes, MLP)
    - Supports n-grams for phrase-level sentiment
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        self.scaler = MinMaxScaler()
        self.numeric_cols = [
            "review_length",
            "exclamation_count",
            "uppercase_count",
        ]

    def add_features(self, df):
        df = df.copy()

        df["reviewText"] = df["reviewText"].fillna("").astype(str)
        df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

        df["review_length"] = df["cleaned_text"].apply(len)
        df["exclamation_count"] = df["reviewText"].str.count("!").fillna(0)
        df["uppercase_count"] = df["reviewText"].apply(
            lambda x: sum(1 for w in str(x).split() if w.isupper())
        )

        df[self.numeric_cols] = df[self.numeric_cols].fillna(0)
        return df

    def fit_transform(self, df):
        df = self.add_features(df)
        x_text = self.vectorizer.fit_transform(df["cleaned_text"])
        x_extra = self.scaler.fit_transform(df[self.numeric_cols])
        return hstack([x_text, x_extra]), df["sentiment"]

    def transform(self, df):
        df = self.add_features(df)
        x_text = self.vectorizer.transform(df["cleaned_text"])
        x_extra = self.scaler.transform(df[self.numeric_cols])
        return hstack([x_text, x_extra]), df["sentiment"]


class ModelTrainer:
    """
    Models:
    1. Naive Bayes → strong baseline for text classification
    2. MLP → captures nonlinear relationships
    """

    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()

    def train_nb(self, x, y):
        grid = GridSearchCV(
            MultinomialNB(),
            {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0]},
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=self.config.random_state
            ),
            scoring="f1_weighted",
            n_jobs=-1
        )
        grid.fit(x, y)
        return grid.best_estimator_, grid

    def train_mlp(self, x, y):
        y_enc = self.label_encoder.fit_transform(y)

        grid = GridSearchCV(
            MLPClassifier(
                max_iter=200,
                early_stopping=True,
                random_state=self.config.random_state
            ),
            {
                "hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "alpha": [0.01, 0.1, 1.0]
            },
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=self.config.random_state
            ),
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid.fit(x, y_enc)
        return grid.best_estimator_, grid

    def predict_mlp(self, model, x):
        preds = model.predict(x)
        return self.label_encoder.inverse_transform(preds)


class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred, name):
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }


class SentimentPipeline:
    def __init__(self, config):
        self.config = config
        self.processor = TextProcessor()
        self.features = FeatureBuilder()
        self.trainer = ModelTrainer(config)
        self.eval = Evaluator()

    def ensure_dirs(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "reports"), exist_ok=True)

    def save_figure(self, fig, filename):
        figures_dir = os.path.join(self.config.output_dir, "figures")
        fig.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def save_text_reports(self, nb_report, mlp_report, vader_report, textblob_report):
        reports_dir = os.path.join(self.config.output_dir, "reports")

        with open(os.path.join(reports_dir, "classification_reports.txt"), "w", encoding="utf-8") as f:
            f.write("VADER\n")
            f.write(vader_report + "\n\n")
            f.write("TextBlob\n")
            f.write(textblob_report + "\n\n")
            f.write("Naive Bayes\n")
            f.write(nb_report + "\n\n")
            f.write("MLP\n")
            f.write(mlp_report + "\n")

    def compute_enhanced_rating(self, df_sample, alpha=0.6):
        analyzer = SentimentIntensityAnalyzer()
        df_sample = df_sample.copy()

        def compute_row(row):
            compound = analyzer.polarity_scores(str(row["reviewText"]))["compound"]
            v_norm = (compound + 1) / 2
            s_norm = (row["overall"] - 1) / 4
            blended = alpha * s_norm + (1 - alpha) * v_norm
            return round(blended * 4 + 1, 4)

        df_sample["enhanced_rating"] = df_sample.apply(compute_row, axis=1)
        df_sample["rating_diff"] = abs(df_sample["enhanced_rating"] - df_sample["overall"])
        return df_sample

    def run(self):
        self.ensure_dirs()

        df = pd.read_json(self.config.data_path, lines=True)
        df["reviewText"] = df["reviewText"].fillna("").astype(str)

        df["sentiment"] = df["overall"].apply(self.processor.label_sentiment)
        # Expand contractions first: didn't -> did not
        df["reviewText"] = df["reviewText"].apply(self.processor.expand_contractions)

        # Then apply negation handling
        df["reviewText"] = df["reviewText"].apply(self.processor.handle_negation)   
        
        #clean
        df["cleaned_text"] = df["reviewText"].apply(self.processor.clean)

        df_sample = pd.concat([
            df[df["sentiment"] == c].sample(
                self.config.n_per_class,
                random_state=self.config.random_state
            )
            for c in ["Positive", "Neutral", "Negative"]
        ]).sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)

        # Full dataset distribution
        fig, ax = plt.subplots(figsize=(9, 6))
        df["overall"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Distribution of Ratings - Full Dataset", pad=15)
        ax.set_xlabel("Star Rating")
        ax.set_ylabel("Count")
        plt.tight_layout()
        self.save_figure(fig, "full_rating_distribution.png")

        # Subset rating distribution
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        df_sample["overall"].value_counts().sort_index().plot(kind="bar", ax=axes[0])
        axes[0].set_title("Rating Distribution - Balanced Subset", pad=15)
        axes[0].set_xlabel("Star Rating")
        axes[0].set_ylabel("Count")

        df_sample["overall"].value_counts().sort_index().plot.pie(
            autopct="%1.1f%%", ax=axes[1]
        )
        axes[1].set_title("Rating Share - Balanced Subset", pad=15)
        axes[1].set_ylabel("")
        plt.tight_layout()
        self.save_figure(fig, "subset_rating_distribution.png")

        # Subset sentiment distribution
        fig, ax = plt.subplots(figsize=(9, 6))
        df_sample["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).plot(
            kind="bar", ax=ax
        )
        ax.set_title("Sentiment Distribution - Balanced Subset", pad=15)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        self.save_figure(fig, "subset_sentiment_distribution.png")

        df_train, df_test = train_test_split(
            df_sample,
            test_size=0.30,
            stratify=df_sample["sentiment"],
            random_state=self.config.random_state
        )

        x_train, y_train = self.features.fit_transform(df_train)
        x_test, y_test = self.features.transform(df_test)

        nb, nb_grid = self.trainer.train_nb(x_train, y_train)
        mlp, mlp_grid = self.trainer.train_mlp(x_train, y_train)

        nb_pred = nb.predict(x_test)
        mlp_pred = self.trainer.predict_mlp(mlp, x_test)

        vader = SentimentIntensityAnalyzer()

        def vader_pred(text):
            s = vader.polarity_scores(text)["compound"]
            return "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"

        df_test = df_test.copy()
        df_test["vader"] = df_test["cleaned_text"].apply(vader_pred)
        df_test["textblob"] = df_test["cleaned_text"].apply(
            lambda t: "Positive" if TextBlob(t).sentiment.polarity > 0
            else "Negative" if TextBlob(t).sentiment.polarity < 0
            else "Neutral"
        )

        nb_report = classification_report(y_test, nb_pred, zero_division=0)
        mlp_report = classification_report(y_test, mlp_pred, zero_division=0)
        vader_report = classification_report(y_test, df_test["vader"], zero_division=0)
        textblob_report = classification_report(y_test, df_test["textblob"], zero_division=0)

        results = pd.DataFrame([
            self.eval.evaluate(y_test, df_test["vader"], "VADER"),
            self.eval.evaluate(y_test, df_test["textblob"], "TextBlob"),
            self.eval.evaluate(y_test, nb_pred, "Naive Bayes"),
            self.eval.evaluate(y_test, mlp_pred, "MLP"),
        ])

        print(results)

        # Naive Bayes grid search figure
        cv_results_nb = pd.DataFrame(nb_grid.cv_results_)
        alpha_grid = [0.001, 0.01, 0.1, 0.5, 1.0]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alpha_grid, cv_results_nb["mean_test_score"], marker="o", color="steelblue")
        ax.fill_between(
            alpha_grid,
            cv_results_nb["mean_test_score"] - cv_results_nb["std_test_score"],
            cv_results_nb["mean_test_score"] + cv_results_nb["std_test_score"],
            alpha=0.2,
            color="steelblue"
        )
        ax.set_xscale("log")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("CV F1-Weighted Score")
        ax.set_title("Naive Bayes Grid Search Results", pad=15)
        plt.tight_layout()
        self.save_figure(fig, "nb_gridsearch.png")

        # NB confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, nb_pred,
            display_labels=["Negative", "Neutral", "Positive"],
            cmap="Blues", ax=ax
        )
        ax.set_title("Naive Bayes - Confusion Matrix", pad=15)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        self.save_figure(fig, "nb_confusion_matrix.png")

        # MLP confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, mlp_pred,
            display_labels=["Negative", "Neutral", "Positive"],
            cmap="Blues", ax=ax
        )
        ax.set_title("MLP - Confusion Matrix", pad=15)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        self.save_figure(fig, "mlp_confusion_matrix.png")

        # MLP loss curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(mlp.loss_curve_, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("MLP - Training Loss Curve", pad=15)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        self.save_figure(fig, "mlp_loss_curve.png")

        # Model comparison
        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        x = np.arange(len(metrics))
        width = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (_, row) in enumerate(results.iterrows()):
            ax.bar(x + i * width, [row[m] for m in metrics], width, label=row["Model"])
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison", pad=15)
        ax.legend(loc="lower right")
        plt.tight_layout()
        self.save_figure(fig, "model_comparison.png")

        # All confusion matrices
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        labels = ["Negative", "Neutral", "Positive"]
        model_data = [
            ("VADER", y_test, df_test["vader"]),
            ("TextBlob", y_test, df_test["textblob"]),
            ("Naive Bayes", y_test, nb_pred),
            ("MLP", y_test, mlp_pred),
        ]
        for ax, (name, yt, yp) in zip(axes, model_data):
            ConfusionMatrixDisplay.from_predictions(
                yt, yp,
                display_labels=labels,
                cmap="Blues",
                ax=ax,
                colorbar=False
            )
            ax.set_title(name, fontsize=12)
        plt.suptitle("All Model Confusion Matrices", fontsize=16, y=1.05)
        plt.tight_layout()
        self.save_figure(fig, "all_confusion_matrices.png")

        # Enhanced rating analysis
        df_sample_enhanced = self.compute_enhanced_rating(df_sample)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(df_sample_enhanced["overall"], bins=5, range=(0.5, 5.5), edgecolor="white")
        axes[0].set_title("Original Star Rating Distribution", pad=15)
        axes[0].set_xlabel("Rating")
        axes[0].set_xticks([1, 2, 3, 4, 5])

        axes[1].hist(df_sample_enhanced["enhanced_rating"], bins=30, edgecolor="white", alpha=0.85)
        axes[1].set_title("Enhanced Rating Distribution", pad=15)
        axes[1].set_xlabel("Enhanced Rating")

        scatter = axes[2].scatter(
            df_sample_enhanced["overall"],
            df_sample_enhanced["enhanced_rating"],
            c=df_sample_enhanced["overall"],
            cmap="Blues",
            alpha=0.35,
            s=10
        )
        axes[2].plot([1, 5], [1, 5], "r--", linewidth=1)
        axes[2].set_xlabel("Original Star Rating")
        axes[2].set_ylabel("Enhanced Rating")
        axes[2].set_title("Original vs Enhanced Ratings", pad=15)
        plt.colorbar(scatter, ax=axes[2], label="Star Rating")
        plt.tight_layout()
        self.save_figure(fig, "enhanced_rating_analysis.png")

        # Save models
        joblib.dump(nb, os.path.join(self.config.output_dir, "nb_model.joblib"))
        joblib.dump(mlp, os.path.join(self.config.output_dir, "mlp_model.joblib"))
        joblib.dump(self.features.vectorizer, os.path.join(self.config.output_dir, "tfidf.joblib"))
        joblib.dump(self.features.scaler, os.path.join(self.config.output_dir, "scaler.joblib"))
        joblib.dump(self.trainer.label_encoder, os.path.join(self.config.output_dir, "label_encoder.joblib"))
        print("Models saved successfully.")

        # Save reports
        reports_dir = os.path.join(self.config.output_dir, "reports")
        results.to_csv(os.path.join(reports_dir, "model_results.csv"), index=False)
        self.save_text_reports(nb_report, mlp_report, vader_report, textblob_report)

        metadata = {
            "config": asdict(self.config),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_rows": len(df),
            "subset_rows": len(df_sample),
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "subset_class_distribution": df_sample["sentiment"].value_counts().to_dict(),
            "train_class_distribution": df_train["sentiment"].value_counts().to_dict(),
            "test_class_distribution": df_test["sentiment"].value_counts().to_dict(),
            "nb_best_params": nb_grid.best_params_,
            "nb_best_cv_f1": float(nb_grid.best_score_),
            "mlp_best_params": mlp_grid.best_params_,
            "mlp_best_cv_f1": float(mlp_grid.best_score_),
            "results": results.to_dict(orient="records"),
        }

        with open(os.path.join(reports_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        df_sample_enhanced.to_csv(
            os.path.join(reports_dir, "enhanced_ratings.csv"),
            index=False
        )

        print(f"All artifacts saved to: {self.config.output_dir}")


if __name__ == "__main__":
    config = Config()
    pipeline = SentimentPipeline(config)
    pipeline.run()