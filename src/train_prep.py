import numpy as np
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

from utils.logger import LOG
from src.helper_model import soft_vote_proba, tune_class_thresholds


class TrainPrep:

    def __init__(self, weights=(0.5, 0.5)):
        self.cv = 5
        self.weights = weights
        LOG.info("TrainPrep initialized")

    def train_models(self, X_train, y_train_enc_enc):
        LOG.info("Training started")

        base = Pipeline([
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("clf", None),
        ])

        grids = {
            "LogisticRegression": {
                "clf": [LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1
                )],
                "tfidf__analyzer": ["word", "char"],
                "tfidf__ngram_range": [(1,2), (3,5)],
                "tfidf__min_df": [2],
                "tfidf__max_features": [20000],
                "clf__C": [0.1, 1, 5]
            },

            "LinearSVM": {
                "clf": [CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced"),
                    method="sigmoid",
                    cv=3
                )],
                "tfidf__analyzer": ["word", "char"],
                "tfidf__ngram_range": [(1,2), (3,5)],
                "tfidf__min_df": [2],
                "tfidf__max_features": [20000],
                "clf__estimator__C": [0.1, 1, 5]
            }
        }

        models = {}

        for name, params in grids.items():
            LOG.info(f"GridSearch for {name}")

            gs = GridSearchCV(
                estimator=clone(base),
                param_grid=params,
                scoring="f1_macro",
                cv=self.cv,
                n_jobs=-1,
                verbose=2,
            )

            gs.fit(X_train, y_train_enc_enc)
            models[name] = gs.best_estimator_

            LOG.info(
                f"{name} best CV F1: {gs.best_score_:.4f}"
            )

        return models


    def evaluate(
        self,
        models,
        X_train,
        y_train_enc,
        X_test,
        y_test_enc,
        label_encoder
    ):
        LOG.info("Evaluation started")

        metrics = {}
        for name, model in models.items():
            preds = model.predict(X_test)
            score = f1_score(y_test_enc, preds, average="macro")
            metrics[name] = score
            LOG.info(f"{name} validation F1: {score:.4f}")

        # Ensemble on validation
        probs = soft_vote_proba(
            models["LogisticRegression"],
            models["LinearSVM"],
            X_test,
            self.weights,
        )

        ensemble_preds = np.argmax(probs, axis=1)
        ensemble_f1 = f1_score(y_test_enc, ensemble_preds, average="macro")
        LOG.info(f"Ensemble F1: {ensemble_f1:.4f}")

        # Threshold tuning on validation
        thresholds = tune_class_thresholds(
            y_test_enc,
            probs,
            num_classes=len(label_encoder.classes_),
        )

        # FINAL FIT ON FULL DATASET
        LOG.info("Fitting final pipeline on full dataset")

        X_full = np.concatenate([X_train, X_test])
        y_full = np.concatenate([y_train_enc, y_test_enc])

        tfidf = clone(models["LogisticRegression"].named_steps["tfidf"])
        lr_clf = clone(models["LogisticRegression"].named_steps["clf"])
        svm_clf = clone(models["LinearSVM"].named_steps["clf"])

        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ("lr", lr_clf),
                ("svm", svm_clf)
            ],
            voting="soft",
            weights=self.weights
        )

        pipeline = Pipeline([
            ("tfidf", tfidf),
            ("ensemble", ensemble)
        ])

        pipeline.fit(X_full, y_full)

        return pipeline, thresholds, self.weights, metrics, ensemble_f1

    def save_artifact(self, pipeline, label_encoder, thresholds, weights, metrics, ensemble_f1, artifact_path="artifact/pipeline.joblib"):
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": pipeline,
            "label_encoder": label_encoder,
            "thresholds": thresholds,
            "weights": weights,
            "report": {
                "models_f1": metrics,
                "ensemble_f1": ensemble_f1
    }
        }
        joblib.dump(artifact, artifact_path)
        LOG.info(f"Artifact saved → {artifact_path}")
        return artifact

    def report(self, artifact):
        print("\nMODEL REPORT")
      
        for k, v in artifact["report"]["models_f1"].items():
            print(f"{k:10s}: F1={v:.4f}")
        
        print(f"ENSEMBLE   : F1={artifact['report']['ensemble_f1']:.4f}")