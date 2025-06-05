import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.utils import resample
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib


def load_and_prepare_data(percorso_file):
    """
    Carica il dataset e rimuove colonne non necessarie, eseguendo anche il bilanciamento tramite undersampling.

    Args:
        percorso_file (str): Percorso al file CSV.

    Returns:
        pd.DataFrame: DataFrame bilanciato con le sole feature e target.
    """
    df = pd.read_csv(percorso_file)
    df = df.drop(columns=['user_id', 'purchase_probability'])
    df = df.dropna()

    # Bilanciamento con undersampling
    df_majority = df[df['target'] == 0]
    df_minority = df[df['target'] == 1]
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


def train_test_splitting(df):
    """
    Suddivide il dataset in set di addestramento e di test.

    Args:
        df (pd.DataFrame): Dataset bilanciato.

    Returns:
        X_train, X_test, y_train, y_test: Feature e target per train/test.
    """
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_base_model(X_train, y_train, X_test, y_test):
    """
    Addestra un modello base di XGBoost e ne valuta l'accuratezza.

    Returns:
        model: Modello addestrato.
    """
    model = XGBClassifier(booster='gbtree', n_estimators=200, eta=0.1,
                          gamma=1, random_state=42, max_depth=15, tree_method='hist')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy modello base: {accuracy:.4f}')
    return model


def fine_tune_model(X_train, y_train):
    """
    Esegue GridSearchCV per ottimizzare i parametri dell'XGBClassifier.

    Returns:
        modello ottimizzato tramite grid search
    """
    params = {
        'n_estimators': [100, 200, 300],
        'eta': [0.1, 0.4, 0.7],
        'gamma': [0.1, 0.5, 1],
        'max_depth': [5, 10]
    }
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    grid = GridSearchCV(
        estimator=XGBClassifier(random_state=42, tree_method='hist'),
        param_grid=params,
        cv=4,
        scoring=scoring_metrics,
        refit='accuracy'
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def valuta_modello(model, X_test, y_test):
    """
    Esegue la valutazione completa del modello: cross-validation, classification report, confusion matrix.

    Args:
        model: Modello da valutare.
        X_test: Feature del set di test.
        y_test: Target del set di test.
    """
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    results = cross_validate(model, X_test, y_test, scoring=scoring_metrics, cv=5, return_train_score=True)

    print("Risultati cross validation:")
    for metric in scoring_metrics:
        print(f"{metric}: {results[f'test_{metric}'].mean():.4f}")

    y_pred = model.predict(X_test)
    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Matrice di Confusione")
    plt.show()


def save_model(model, path='models/xgboost.pkl'):
    """
    Salva il modello su disco.

    Args:
        model: Modello da salvare.
        path (str): Percorso di destinazione.
    """
    joblib.dump(model, path)
    print(f"Modello salvato in: {path}")


def main():
    """
    Funzione principale che esegue l'intero processo: caricamento dati, addestramento, ottimizzazione,
    valutazione e salvataggio del modello.
    """
    df = load_and_prepare_data("..\syntethic_data\data\ecommerce_synthetic_data.csv")
    X_train, X_test, y_train, y_test = train_test_splitting(df)
    _ = train_base_model(X_train, y_train, X_test, y_test)
    optimized_model = fine_tune_model(X_train, y_train)
    valuta_modello(optimized_model, X_test, y_test)
    save_model(optimized_model)


if __name__ == "__main__":
    main()
