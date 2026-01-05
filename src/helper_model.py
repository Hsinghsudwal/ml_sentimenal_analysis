import numpy as np
from sklearn.metrics import f1_score


def soft_vote_proba(lr, svm, X, weights=(0.5, 0.5)):
    return (
        weights[0] * lr.predict_proba(X)
        + weights[1] * svm.predict_proba(X)
    )


def predict_with_thresholds(probs, thresholds):
    adjusted = probs - thresholds
    return np.argmax(adjusted, axis=1)


def soft_vote_predict(lr, svm, X, weights=(0.5, 0.5), thresholds=None):
    probs = soft_vote_proba(lr, svm, X, weights)
    if thresholds is None:
        return np.argmax(probs, axis=1)
    return predict_with_thresholds(probs, thresholds)


def tune_class_thresholds(y_true, y_probs, num_classes):
    thresholds = np.zeros(num_classes)

    for c in range(num_classes):
        best_f1, best_t = 0, 0.5

        for t in np.arange(0.2, 0.85, 0.05):
            preds = (y_probs[:, c] >= t).astype(int)
            y_bin = (y_true == c).astype(int)

            f1 = f1_score(y_bin, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        thresholds[c] = best_t

    return thresholds



# class SoftVotingClassifier:
#     def __init__(self, lr, svm, weights=(0.5, 0.5)):
#         self.lr = lr
#         self.svm = svm
#         self.weights = weights

#     def predict_proba(self, X):
#         return (
#             self.weights[0] * self.lr.predict_proba(X)
#             + self.weights[1] * self.svm.predict_proba(X)
#         )

#     def predict(self, X, thresholds=None):
#         probs = self.predict_proba(X)
#         if thresholds is None:
#             return np.argmax(probs, axis=1)
#         return self.predict_with_thresholds(probs, thresholds)



# def tune_class_thresholds(y_true, y_probs, num_classes):
#     thresholds = np.zeros(num_classes)

#     for c in range(num_classes):
#         best_f1, best_t = 0, 0.5
#         for t in np.arange(0.2, 0.85, 0.05):
#             preds = (y_probs[:, c] >= t).astype(int)
#             y_bin = (y_true == c).astype(int)

#             f1 = f1_score(y_bin, preds, zero_division=0)
#             if f1 > best_f1:
#                 best_f1, best_t = f1, t

#         thresholds[c] = best_t
#     return thresholds


# def predict_with_thresholds(probs, thresholds):
#     adjusted = probs - thresholds
#     return np.argmax(adjusted, axis=1)


# class TrainPrep:
#     def __init__(self, cv=5, ensemble_weights=(0.5, 0.5)):
#         self.cv = cv
#         self.ensemble_weights = ensemble_weights
#         LOG.info("TrainPrep initialized")

#     # --------------------------------------------------
#     # TRAIN
#     # --------------------------------------------------
#     def train(self, X_train, y_train):
#         base = Pipeline([
#             ("tfidf", TfidfVectorizer(sublinear_tf=True)),
#             ("clf", None),
#         ])

#         grids = {
#             "lr": {
#                 "clf": [LogisticRegression(
#                     max_iter=1000,
#                     class_weight="balanced",
#                     n_jobs=-1,
#                 )],
#                 "clf__C": [0.1, 1, 5],
#             },
#             "svm": {
#                 "clf": [CalibratedClassifierCV(
#                     estimator=LinearSVC(class_weight="balanced"),
#                     method="sigmoid",
#                     cv=3,
#                 )],
#                 "clf__estimator__C": [0.1, 1, 5],
#             },
#         }

#         models = {}

#         for name, params in grids.items():
#             LOG.info(f"GridSearch for {name}")

#             gs = GridSearchCV(
#                 estimator=clone(base),
#                 param_grid=params,
#                 scoring="f1_macro",
#                 cv=self.cv,
#                 n_jobs=-1,
#                 verbose=1,
#             )

#             gs.fit(X_train, y_train)
#             models[name] = gs.best_estimator_

#             LOG.info(f"{name} best CV F1: {gs.best_score_:.4f}")

#         return models

#     # --------------------------------------------------
#     # EVALUATE + ENSEMBLE + SAVE
#     # --------------------------------------------------
#     def evaluate(
#         self,
#         models,
#         X_train,
#         y_train,
#         X_val,
#         y_val,
#         label_encoder,
#         artifact_path="artifact/sentiment_pipeline.joblib",
#     ):
#         LOG.info("Evaluation started")

#         # Individual scores
#         report = {}
#         for name, model in models.items():
#             preds = model.predict(X_val)
#             score = f1_score(y_val, preds, average="macro")
#             report[name] = score
#             LOG.info(f"{name} validation F1: {score:.4f}")

#         # Ensemble (validation)
#         lr, svm = models["lr"], models["svm"]
#         lr_probs = lr.predict_proba(X_val)
#         svm_probs = svm.predict_proba(X_val)

#         ensemble_probs = (
#             self.ensemble_weights[0] * lr_probs
#             + self.ensemble_weights[1] * svm_probs
#         )

#         ensemble_preds = np.argmax(ensemble_probs, axis=1)
#         ensemble_f1 = f1_score(y_val, ensemble_preds, average="macro")

#         LOG.info(f"Ensemble F1: {ensemble_f1:.4f}")

#         # Threshold tuning (ON VALIDATION ONLY)
#         thresholds = tune_class_thresholds(
#             y_val,
#             ensemble_probs,
#             num_classes=len(label_encoder.classes_),
#         )

#         # FINAL FIT ON FULL DATASET
#         LOG.info("Fitting final pipeline on full dataset")

#         X_full = np.concatenate([X_train, X_val])
#         y_full = np.concatenate([y_train, y_val])

#         tfidf = clone(models["lr"].named_steps["tfidf"])
#         lr_clf = clone(models["lr"].named_steps["clf"])
#         svm_clf = clone(models["svm"].named_steps["clf"])

#         ensemble = SoftVotingClassifier(
#             lr_clf, svm_clf, self.ensemble_weights
#         )

#         pipeline = Pipeline([
#             ("tfidf", tfidf),
#             ("ensemble", ensemble),
#         ])

#         pipeline.fit(X_full, y_full)

#         artifact = {
#             "model": pipeline,
#             "label_encoder": label_encoder,
#             "thresholds": thresholds,
#             "weights": self.ensemble_weights,
#             "report": {
#                 "models_f1": report,
#                 "ensemble_f1": ensemble_f1,
#             },
#         }

#         joblib.dump(artifact, artifact_path)
#         LOG.info(f"Artifact saved → {artifact_path}")

#         return artifact

# def print_report(artifact):
#     print("\nMODEL REPORT")
#     print("-" * 40)
#     for k, v in artifact["report"]["models_f1"].items():
#         print(f"{k:10s}: F1={v:.4f}")

#     print("-" * 40)
#     print(f"ENSEMBLE   : F1={artifact['report']['ensemble_f1']:.4f}")


# # app.py
# import joblib
# import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel

# artifact = joblib.load("artifact/sentiment_pipeline.joblib")

# model = artifact["model"]
# label_encoder = artifact["label_encoder"]
# thresholds = artifact["thresholds"]

# app = FastAPI()


# class Request(BaseModel):
#     text: str


# @app.post("/predict")
# def predict(req: Request):
#     preds = model.named_steps["ensemble"].predict(
#         model.named_steps["tfidf"].transform([req.text]),
#         thresholds=thresholds,
#     )

#     label = label_encoder.inverse_transform(preds)[0]

#     return {"prediction": label}


# trainer = TrainPrep()

# models = trainer.train(X_train, y_train)

# artifact = trainer.evaluate(
#     models,
#     X_train,
#     y_train,
#     X_val,
#     y_val,
#     label_encoder,
# )

# print_report(artifact)

