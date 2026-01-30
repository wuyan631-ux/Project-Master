import numpy as np
np.set_printoptions(threshold=10000, suppress=True)

import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    make_scorer,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle
import time


# Question 8 : Orchestration
# Fonction d'évaluation
def evaluate_classifier(modele, X_test, y_test, nom):
    y_pred = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    # Affichage des métriques
    print(f"\n--- {nom} ---")
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    print("Accuracy :", accuracy_score(y_test, y_pred))
    # Dans le cas de la solvabilité client, on prend la précision.
    print("Précision :", precision_score(y_test, y_pred))
    # print("Rappel :", recall_score(y_test, y_pred))

    score_final = (accuracy_score(y_test, y_pred) + precision_score(y_test, y_pred)) / 2
    print("Score final :", score_final)

    # Courbe ROC avec l'AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, score_final


# Fonction d'apprentissage et de test
def run_classifiers_train_test(clf_dict, X_train, X_test, y_train, y_test):
    scores = {}
    plt.figure(figsize=(8, 6))

    for nom, model in clf_dict.items():
        # Entraînement de chaque algorithme de clf
        model.fit(X_train, y_train)

        # Affichage des résultats d'évaluation de la fonction evaluate_classifier
        fpr, tpr, roc_auc, score = evaluate_classifier(model, X_test, y_test, nom)
        scores[nom] = score

        # Pour afficher toutes les courbes ROC sur le même graphique
        plt.plot(fpr, tpr, label=f"{nom} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Faux positifs")
    plt.ylabel("Vrais positifs")
    plt.title("Courbes ROC")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Choix final du meilleur modèle selon le score final
    meilleur_modele = max(scores, key=scores.get)
    print(f"\nMeilleur modèle : {meilleur_modele} avec un score de {scores[meilleur_modele]}")
    return clf_dict[meilleur_modele]


# Fonction d'apprentissage et de test sur les données normalisées
def run_classifiers_normal_train_test(clf_dict, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model = run_classifiers_train_test(clf_dict, X_train_scaled, X_test_scaled, y_train, y_test)
    return best_model, X_train_scaled, X_test_scaled


# Fonction d'apprentissage et de test sur les données normalisées avec une ACP
def run_classifiers_acp_train_test(clf_dict, X_train_scaled, X_test_scaled, y_train, y_test):
    # ACP sur les données normalisées
    pca = PCA(n_components=3)

    # Fit uniquement sur le train
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Concaténation : données normalisées + 3 composantes PCA
    X_train_final = np.concatenate([X_train_scaled, X_train_pca], axis=1)
    X_test_final = np.concatenate([X_test_scaled, X_test_pca], axis=1)

    return run_classifiers_train_test(clf_dict, X_train_final, X_test_final, y_train, y_test)


# Fonction sur l'importance des variables
def importance_relative(Xtrain, Ytrain, nom_cols):
    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    clf.fit(Xtrain, Ytrain)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    sorted_idx = np.argsort(importances)[::-1]

    features = np.array(nom_cols)
    print(features[sorted_idx])

    padding = np.arange(Xtrain.shape[1]) + 0.5
    plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align="center")
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()

    return sorted_idx


# Fonction sur la sélection du nombre optimal de variables
def selection(modele, Xtrain, Xtest, Ytrain, Ytest, sorted_idx):
    scores = np.zeros(Xtrain.shape[1])

    for f in np.arange(0, Xtrain.shape[1]):
        X1_f = Xtrain[:, sorted_idx[: f + 1]]
        X2_f = Xtest[:, sorted_idx[: f + 1]]

        modele.fit(X1_f, Ytrain)
        Ypred = modele.predict(X2_f)

        scores[f] = np.round(accuracy_score(Ytest, Ypred), 3)

    plt.plot(scores)
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Accuracy")
    plt.title("Evolution de l'accuracy en fonction des variables")
    plt.show()

    k_opt = np.argmax(scores) + 1
    return k_opt


# Fonction qui optimise au mieux le critère: (accuracy + précision) / 2
def acc_prec_score(y_true, y_pred):
    return (accuracy_score(y_true, y_pred) + precision_score(y_true, y_pred, average="binary", zero_division=0)) / 2

def acc_prec_score1(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if (tp + fp) == 0:
        prec = 0.0
    else:
        prec = tp / (tp + fp)

    return (acc + prec) / 2

acc_prec_scorer1 = make_scorer(acc_prec_score1)

# Transforme la fonction en scorer compatible avec scikit-learn
acc_prec_scorer = make_scorer(acc_prec_score)


def tuning_mlp(Xtrain, Xtest, Ytrain, Ytest, sorted_idx, k_opt):
    # On se restreint aux k_opt meilleures variables
    Xtrain_sel = Xtrain[:, sorted_idx[:k_opt]]
    Xtest_sel = Xtest[:, sorted_idx[:k_opt]]

    mlp = MLPClassifier(random_state=1, max_iter=500)

    # Grille simple d'hyperparamètres
    param_grid = {
        "hidden_layer_sizes": [(40, 20), (50, 25)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # GridSearchCV pour optimiser (accuracy + précision) / 2
    grid = GridSearchCV(mlp, param_grid, scoring=acc_prec_scorer, cv=cv, n_jobs=-1, refit=True)
    grid.fit(Xtrain_sel, Ytrain)

    # Meilleur modèle et évaluation sur test
    best_mlp = grid.best_estimator_
    Ypred = best_mlp.predict(Xtest_sel)

    acc = accuracy_score(Ytest, Ypred)
    prec = precision_score(Ytest, Ypred, average="binary", zero_division=0)
    final_score = (acc + prec) / 2

    try:
        auc_score = roc_auc_score(Ytest, best_mlp.predict_proba(Xtest_sel)[:, 1])
    except Exception:
        auc_score = None

    print("Meilleurs paramètres:", grid.best_params_)
    print("Score CV moyen (acc+prec)/2:", grid.best_score_)
    print("Score test final:", final_score)
    print("Accuracy:", acc, "| Précision:", prec, "| AUC:", auc_score)
    print("Matrice de confusion:\n", confusion_matrix(Ytest, Ypred))

    return best_mlp


def creation_pipeline(Xtrain, Xtest, Ytrain, Ytest, sorted_idx, k_opt, best_model, filename="pipeline_final.pkl"):
    Xtrain_sel = Xtrain[:, sorted_idx[:k_opt]]
    Xtest_sel = Xtest[:, sorted_idx[:k_opt]]

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", best_model)])
    pipeline.fit(Xtrain_sel, Ytrain)

    score = pipeline.score(Xtest_sel, Ytest)
    print("Score pipeline:", score)

    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


def pipeline_generation_train_test_split(X, y, nom_cols, filename="pipeline_orchestration.pkl"):
    # Split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.5, random_state=1)

    # Dictionnaire de classifieurs
    clfs = {
        "CART": DecisionTreeClassifier(random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1),
    }

    # Choix du meilleur modèle
    best_model, X_train_scaled, X_test_scaled = run_classifiers_normal_train_test(clfs, Xtrain, Xtest, Ytrain, Ytest)

    # Importance des variables
    sorted_idx = importance_relative(X_train_scaled, Ytrain, nom_cols)

    # Sélection du nombre optimal de variables
    k_opt = selection(best_model, X_train_scaled, X_test_scaled, Ytrain, Ytest, sorted_idx)

    # Tuning du meilleur modèle (exemple avec MLP)
    tuned_model = tuning_mlp(X_train_scaled, X_test_scaled, Ytrain, Ytest, sorted_idx, k_opt)

    # Création du pipeline final
    pipeline = creation_pipeline(
        X_train_scaled,
        X_test_scaled,
        Ytrain,
        Ytest,
        sorted_idx,
        k_opt,
        tuned_model,
        filename,
    )

    print(f"\nPipeline final sauvegardé dans {filename}")
    return pipeline


# Question 9
def run_classifiers_cv(clfs, X, y, scoring=acc_prec_scorer):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    results = {}

    for name, clf in clfs.items():
        print(f"\n--- {name} ---")
        start = time.time()

        # Accuracy
        cv_acc = cross_val_score(clf, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
        print(f"Accuracy: {cv_acc.mean():.3f}+/- {cv_acc.std():.3f}")

        # AUC
        try:
            cv_auc = cross_val_score(clf, X, y, cv=kf, scoring="roc_auc", n_jobs=-1)
            print(f"AUC: {cv_auc.mean():.3f}+/- {cv_auc.std():.3f}")
        except Exception:
            cv_auc = None
            print("AUC: non disponible")

        # Accuracy + précision / 2
        cv_acc_prec = cross_val_score(clf, X, y, cv=kf, scoring=scoring, n_jobs=-1)
        print(f"(Accuracy+Précision)/2: {cv_acc_prec.mean():.3f}+/- {cv_acc_prec.std():.3f}")

        end = time.time()
        print(f"Temps d'exécution: {end-start:.2f} secondes")

        results[name] = {
            "accuracy": cv_acc.mean(),
            "auc": cv_auc.mean() if cv_auc is not None else None,
            "acc_prec": cv_acc_prec.mean(),
        }

    return results


def pipeline_generation_cv(X, y, nom_cols, filename="pipeline_cv.pkl"):
    # Dictionnaire de classifieurs
    clfs = {
        "CART": DecisionTreeClassifier(max_depth=3, random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(20, 10), random_state=1, max_iter=500),
        "Bagging": sklearn.ensemble.BaggingClassifier(n_estimators=200, random_state=1, n_jobs=-1),
        "AdaBoost": sklearn.ensemble.AdaBoostClassifier(n_estimators=200, random_state=1),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1),
    }

    # Comparaison par CV
    results = run_classifiers_cv(clfs, X, y)

    # Choix du meilleur modèle
    best_model_name = max(results, key=lambda k: results[k]["acc_prec"])
    print(f"\n>>> Meilleur modèle selon CV: {best_model_name}")

    best_model = clfs[best_model_name]

    # Importance des variables
    sorted_idx = importance_relative(X, y, nom_cols)

    # Sélection du nombre optimal de variables
    k_opt = selection(best_model, X, X, y, y, sorted_idx)

    # Tuning du meilleur modèle
    tuned_model = tuning_mlp(X, X, y, y, sorted_idx, k_opt)

    # Création du pipeline final
    pipeline = creation_pipeline(X, X, y, y, sorted_idx, k_opt, tuned_model, filename)

    print(f"\nPipeline CV final sauvegardé dans {filename}")
    return pipeline
