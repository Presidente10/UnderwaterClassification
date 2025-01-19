import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

###############################################################################
#                          SPLIT DATASET MULTICLASS
###############################################################################
def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Suddivide il dataset in Train, Validation e Test,
    mappando la 'Subclass' in numeri interi (multiclasse).
    Mantiene la coesione dei gruppi (Parent) grazie a GroupShuffleSplit.
    """

    # Encoding per la Subclass
    subclass_encoder = LabelEncoder()
    df['Subclass_encoded'] = subclass_encoder.fit_transform(df['Subclass'])

    # Split su base Subclass, mantenendo 'Parent' come group
    train_idx, val_idx, test_idx = [], [], []

    for subclass in df['Subclass'].unique():
        subset = df[df['Subclass'] == subclass]
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        groups = subset['Parent']
        train_local_idx, temp_local_idx = next(gss.split(subset, groups=groups))

        # Val e Test dallo "temp"
        subset_temp = subset.iloc[temp_local_idx]
        gss_temp = GroupShuffleSplit(n_splits=1,
                                     train_size=(val_size / (val_size + test_size)),
                                     random_state=42)
        val_local_idx, test_local_idx = next(gss_temp.split(subset_temp,
                                                            groups=subset_temp['Parent']))

        train_idx.extend(subset.index[train_local_idx])
        val_idx.extend(subset_temp.index[val_local_idx])
        test_idx.extend(subset_temp.index[test_local_idx])

    X_train = df.loc[train_idx].copy()
    X_val   = df.loc[val_idx].copy()
    X_test  = df.loc[test_idx].copy()

    for dataset in [X_train, X_val, X_test]:
        dataset['File_Prefix']    = dataset['File Name'].str.extract(r'(^[a-f0-9\-]+)')
        dataset['Segment_Number'] = dataset['File Name'].str.extract(r'_seg(\d+)').astype(int)
        dataset.sort_values(by=['File_Prefix', 'Segment_Number'], inplace=True)
        dataset.drop(columns=['File_Prefix', 'Segment_Number'], inplace=True)

    # Label numeric
    y_train = X_train['Subclass_encoded'].values
    y_val   = X_val['Subclass_encoded'].values
    y_test  = X_test['Subclass_encoded'].values

    # Rimuovi la colonna encoding
    X_train.drop(columns=['Subclass_encoded'], inplace=True)
    X_val.drop(columns=['Subclass_encoded'], inplace=True)
    X_test.drop(columns=['Subclass_encoded'], inplace=True)

    total_samples = len(df)
    print(f"\nTrain size: {len(X_train)} ({len(X_train)/total_samples:.2%})")
    print(f"Val size:   {len(X_val)} ({len(X_val)/total_samples:.2%})")
    print(f"Test size:  {len(X_test)} ({len(X_test)/total_samples:.2%})\n")

    print("Distribuzione di 'Subclass' nel train:")
    print(X_train['Subclass'].value_counts())

    print("\nDistribuzione di 'Subclass' nel val:")
    print(X_val['Subclass'].value_counts())

    print("\nDistribuzione di 'Subclass' nel test:")
    print(X_test['Subclass'].value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test, subclass_encoder


###############################################################################
#                           SMOTE MULTICLASS
###############################################################################
def apply_smote_multiclass(X_train, y_train, k_neighbors=1):
    """
    Applica SMOTE multiclasse su X_train, y_train (solo sulle feature numeriche).
    Rimuove eventuali classi con 1 solo campione e riduce k_neighbors se necessario.
    """
    y_series = pd.Series(y_train)

    # Rimuovi classi con un solo campione
    class_counts = y_series.value_counts()
    to_remove = class_counts[class_counts == 1].index
    if len(to_remove) > 0:
        print(f"Attenzione: classi con 1 campione rimosse: {list(to_remove)}")
        mask = ~y_series.isin(to_remove)
        X_train = X_train[mask]
        y_series = y_series[mask]

    # Gestisci k_neighbors
    class_counts = y_series.value_counts()
    min_samples = class_counts.min()
    if k_neighbors > min_samples - 1:
        print(f"Attenzione: k_neighbors={k_neighbors} > {min_samples-1}. Ridotto a {min_samples-1}.")
        k_neighbors = min_samples - 1 if (min_samples - 1) > 0 else 1

    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    X_train_num = X_train[numeric_columns]

    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled_num, y_resampled = smote.fit_resample(X_train_num, y_series)

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())

    return X_resampled_num, y_resampled


###############################################################################
#           MODELLI MULTICLASS: Random Forest, SVM, LightGBM
###############################################################################
def train_random_forest_multiclass(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Modello Random Forest multiclasse, con GridSearchCV e output a 4 cifre decimali.
    """
    X_train = X_train.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')
    X_val   = X_val.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')
    X_test  = X_test.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [5, 7],
        'min_samples_split': [10],
        'min_samples_leaf': [4]
    }

    model = RandomForestClassifier(random_state=42, max_features='sqrt')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Validation
    y_val_pred  = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    logloss_val  = log_loss(y_val, y_val_proba, labels=np.unique(y_val))

    print(f"Accuratezza (Val) = {accuracy_val:.4f}")
    print(f"Log Loss (Val)    = {logloss_val:.4f}")

    print("\n=== Report - Validation ===")
    print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))

    # Test
    y_test_pred  = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    logloss_test  = log_loss(y_test, y_test_proba, labels=np.unique(y_test))

    print(f"\nAccuratezza (Test) = {accuracy_test:.4f}")
    print(f"Log Loss (Test)    = {logloss_test:.4f}")

    print("\n=== Report - Test ===")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    return best_model


def train_svm_multiclass(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Modello SVM multiclasse, output a 4 cifre decimali.
    """
    X_train = X_train.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')
    X_val   = X_val.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')
    X_test  = X_test.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')

    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train, y_train)

    # Validazione
    y_val_pred = svm_model.predict(X_val)
    print("\n=== Report - Validation ===")
    print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza (Val) = {accuracy_val:.4f}")

    # Test
    y_test_pred = svm_model.predict(X_test)
    print("\n=== Report - Test ===")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Accuratezza (Test) = {accuracy_test:.4f}")

    return svm_model


def train_lightgbm_multiclass(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Modello LightGBM multiclasse, output a 4 cifre decimali.
    """
    # Rimuove valori NaN da y e X in train/val/test
    mask_train = ~pd.isna(y_train)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    mask_val = ~pd.isna(y_val)
    X_val = X_val[mask_val]
    y_val = y_val[mask_val]

    mask_test = ~pd.isna(y_test)
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    # Rimuovi colonne testuali
    X_train = X_train.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')
    X_val   = X_val.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')
    X_test  = X_test.drop(columns=['Class','Parent','Subclass','File Name'], errors='ignore')

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss',
        'random_state': 42,
        'verbose': -1
    }

    lgb_model = lgb.train(params, train_data, valid_sets=[val_data], valid_names=['valid'], num_boost_round=100)

    # Validation
    y_val_pred_proba = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza (Val) = {accuracy_val:.4f}")

    print("\n=== Report - Validation ===")
    print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))

    # Test
    y_test_pred_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"\nAccuratezza (Test) = {accuracy_test:.4f}")

    print("\n=== Report - Test ===")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    return lgb_model

###############################################################################
#               CONFUSION MATRIX MULTICLASS in PERCENTUALE
###############################################################################
def rf_plot_confusion_matrices(model,
                               X_val, y_val_encoded,
                               X_test, y_test_encoded,
                               subclass_encoder):
    """
    Mostra due confusion matrix (Val, Test) con percentuali.
    Sugli assi compaiono i nomi originali delle classi (non i numeri).
    """
    # Predizioni codici
    y_val_pred_encoded  = model.predict(X_val)
    y_test_pred_encoded = model.predict(X_test)

    # Convertiamo in stringhe
    y_val_str        = subclass_encoder.inverse_transform(y_val_encoded)
    y_val_pred_str   = subclass_encoder.inverse_transform(y_val_pred_encoded)
    y_test_str       = subclass_encoder.inverse_transform(y_test_encoded)
    y_test_pred_str  = subclass_encoder.inverse_transform(y_test_pred_encoded)

    all_labels = subclass_encoder.classes_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Validation
    ConfusionMatrixDisplay.from_predictions(
        y_val_str,
        y_val_pred_str,
        labels=all_labels,
        normalize='true',
        cmap='Blues',
        colorbar=False,
        ax=axes[0]
    )
    axes[0].set_title("Random Forest - CM (Validation)")

    # Test
    ConfusionMatrixDisplay.from_predictions(
        y_test_str,
        y_test_pred_str,
        labels=all_labels,
        normalize='true',
        cmap='Blues',
        colorbar=False,
        ax=axes[1]
    )
    axes[1].set_title("Random Forest - CM (Test)")

    for ax in axes:
        ax.grid(False)
    plt.tight_layout()
    plt.show()


def svm_plot_confusion_matrices(model,
                                X_val, y_val_encoded,
                                X_test, y_test_encoded,
                                subclass_encoder):
    y_val_pred_encoded  = model.predict(X_val)
    y_test_pred_encoded = model.predict(X_test)

    y_val_str       = subclass_encoder.inverse_transform(y_val_encoded)
    y_val_pred_str  = subclass_encoder.inverse_transform(y_val_pred_encoded)
    y_test_str      = subclass_encoder.inverse_transform(y_test_encoded)
    y_test_pred_str = subclass_encoder.inverse_transform(y_test_pred_encoded)

    all_labels = subclass_encoder.classes_

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # Validation
    ConfusionMatrixDisplay.from_predictions(
        y_val_str,
        y_val_pred_str,
        labels=all_labels,
        normalize='true',
        cmap='Blues',
        colorbar=False,
        ax=axes[0]
    )
    axes[0].set_title("SVM - CM (Val)")

    # Test
    ConfusionMatrixDisplay.from_predictions(
        y_test_str,
        y_test_pred_str,
        labels=all_labels,
        normalize='true',
        cmap='Blues',
        colorbar=False,
        ax=axes[1]
    )
    axes[1].set_title("SVM - CM (Test)")

    for ax in axes:
        ax.grid(False)
    plt.tight_layout()
    plt.show()


def lightgbm_plot_confusion_matrices(model,
                                     X_val, y_val_encoded,
                                     X_test, y_test_encoded,
                                     subclass_encoder):
    """
    Per LightGBM (Booster nativo), predict(X_val) di solito restituisce prob.
    """
    y_val_pred_proba  = model.predict(X_val)
    y_test_pred_proba = model.predict(X_test)

    y_val_pred_encoded  = np.argmax(y_val_pred_proba, axis=1)
    y_test_pred_encoded = np.argmax(y_test_pred_proba, axis=1)

    # Converti in stringhe
    y_val_str       = subclass_encoder.inverse_transform(y_val_encoded)
    y_val_pred_str  = subclass_encoder.inverse_transform(y_val_pred_encoded)
    y_test_str      = subclass_encoder.inverse_transform(y_test_encoded)
    y_test_pred_str = subclass_encoder.inverse_transform(y_test_pred_encoded)

    all_labels = subclass_encoder.classes_

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # Validation
    ConfusionMatrixDisplay.from_predictions(
        y_val_str,
        y_val_pred_str,
        labels=all_labels,
        normalize='true',
        cmap='Blues',
        colorbar=False,
        ax=axes[0]
    )
    axes[0].set_title("LightGBM - CM (Val)")

    # Test
    ConfusionMatrixDisplay.from_predictions(
        y_test_str,
        y_test_pred_str,
        labels=all_labels,
        normalize='true',
        cmap='Blues',
        colorbar=False,
        ax=axes[1]
    )
    axes[1].set_title("LightGBM - CM (Test)")

    for ax in axes:
        ax.grid(False)
    plt.tight_layout()
    plt.show()
