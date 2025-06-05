import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


def load_and_clean_data(filepath):
    """
    Load dataset from a CSV file, drop specified columns and rows with missing values.

    Parameters:
    ----------
    filepath : str
        Path to the CSV file.

    Returns:
    -------
    pd.DataFrame
        Cleaned DataFrame with specified columns dropped and NaNs removed.
    """
    df = pd.read_csv(filepath)
    df = df.drop(columns=['user_id', 'purchase_probability'])
    df = df.dropna()
    return df


def balance_data(df):
    """
    Perform under-sampling on the majority class to balance the dataset.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the data with a 'target' column.

    Returns:
    -------
    pd.DataFrame
        Balanced DataFrame with equal number of majority and minority class samples.
    """
    df_majority = df[df['target'] == 0]
    df_minority = df[df['target'] == 1]
    df_majority_under_sampled = resample(df_majority,
                                         replace=False,
                                         n_samples=len(df_minority),
                                         random_state=42)
    df_balanced = pd.concat([df_majority_under_sampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced


def split_data(df):
    """
    Split features and target into train and test sets with stratification.

    Parameters:
    ----------
    df : pd.DataFrame
        Balanced DataFrame containing features and 'target' column.

    Returns:
    -------
    tuple
        X_train, X_test, y_train, y_test split with stratification.
    """
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """
    Train a logistic regression model using a pipeline with scaling and grid search for hyperparameter tuning.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels.

    Returns:
    -------
    sklearn.pipeline.Pipeline
        Best trained pipeline model after GridSearchCV.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))
    ])
    parameters = {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear'],
    }
    scoring_metrics = ['accuracy', 'f1_macro']

    grid_search = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring=scoring_metrics,
        refit='accuracy',
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using cross-validation on test data and display classification report and confusion matrix.

    Parameters:
    ----------
    model : sklearn.pipeline.Pipeline
        Trained pipeline model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels for test set.

    Returns:
    -------
    None
        Prints evaluation metrics and shows confusion matrix plot.
    """
    scoring_metrics = ['accuracy', 'f1_macro']
    results = cross_validate(
        model,
        X_test,
        y_test,
        scoring=scoring_metrics,
        cv=5,
        return_train_score=True
    )

    print("Risultati cross validation:")
    for metric in scoring_metrics:
        test_score = results[f'test_{metric}'].mean()
        print(f"{metric}: {test_score:.4f}")

    y_pred = model.predict(X_test)
    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Matrice di Confusione")
    plt.show()


def save_model(model, path='models/logistic_regression_model.pkl'):
    """
    Save the trained model to disk using joblib.

    Parameters:
    ----------
    model : sklearn.pipeline.Pipeline
        Trained pipeline model to be saved.
    path : str, optional
        File path where the model will be saved (default is 'models/logistic_regression_model.pkl').

    Returns:
    -------
    None
    """
    joblib.dump(model, path)


def main():
    """
    Main function to load data, preprocess, train, evaluate, and save the logistic regression model.
    """
    filepath = "..\syntethic_data\data\ecommerce_synthetic_data.csv"
    df = load_and_clean_data(filepath)
    df_balanced = balance_data(df)
    X_train, X_test, y_train, y_test = split_data(df_balanced)
    feature_order = X_train.columns.tolist()
    print(feature_order)
    best_model = train_model(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)
    save_model(best_model)


if __name__ == "__main__":
    main()