from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder


###############################################################################
#                               SPLIT DATASET
###############################################################################
def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Suddivide il dataset in train, validation e test, gestisce i NaN
    e applica un label encoder alle classi.
    """

    # 1. Gestione dei NaN (per ora su tutto il dataframe)
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # 2. Colonne per ordinamento (estratte da 'File Name')
    df['File_Prefix'] = df['File Name'].str.extract(r'(^[a-f0-9\-]+)')
    df['Segment_Number'] = df['File Name'].str.extract(r'_seg(\d+)').astype(int)

    # 3. Suddivisione per Subclass, mantenendo coerenza dei gruppi (Parent)
    train_idx, val_idx, test_idx = [], [], []
    for subclass in df['Subclass'].unique():
        subclass_data = df[df['Subclass'] == subclass]

        # Primo split: 80% train, 20% temporaneo (validation + test)
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        groups = subclass_data['Parent']
        train_local_idx, temp_local_idx = next(gss.split(subclass_data, groups=groups))

        temp_data = subclass_data.iloc[temp_local_idx]

        # Secondo split: 50% validation, 50% test dal "temp_data"
        gss_temp = GroupShuffleSplit(n_splits=1,
                                     train_size=val_size / (val_size + test_size),
                                     random_state=42)
        val_local_idx, test_local_idx = next(gss_temp.split(temp_data, groups=temp_data['Parent']))

        # Aggiungi gli indici al dataset globale
        train_idx.extend(subclass_data.index[train_local_idx])
        val_idx.extend(temp_data.index[val_local_idx])
        test_idx.extend(temp_data.index[test_local_idx])

    # 4. Creazione dei set
    X_train = df.loc[train_idx].copy()
    X_val = df.loc[val_idx].copy()
    X_test = df.loc[test_idx].copy()

    # 5. Ordina i set
    for dataset in [X_train, X_val, X_test]:
        dataset.sort_values(by=['File_Prefix', 'Segment_Number'], inplace=True)

    # 6. Separazione delle etichette
    y_train = X_train['Class'].values
    y_val = X_val['Class'].values
    y_test = X_test['Class'].values

    # 7. Encoder per le etichette 'Class'
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    # 8. Rimuovi colonne di ordinamento (opzionale)
    columns_to_drop = ['File_Prefix', 'Segment_Number']
    X_train.drop(columns=columns_to_drop, inplace=True)
    X_val.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)

    # 9. Report
    total_samples = len(df)
    print(f"\nDimensione del set di addestramento: {len(X_train)} campioni "
          f"({len(X_train) / total_samples:.2%})")
    print(f"Dimensione del set di validazione: {len(X_val)} campioni "
          f"({len(X_val) / total_samples:.2%})")
    print(f"Dimensione del set di test: {len(X_test)} campioni "
          f"({len(X_test) / total_samples:.2%})\n")

    print("Distribuzione delle classi nel set di addestramento:")
    print(X_train['Class'].value_counts())
    print("\nDistribuzione delle classi nel set di validazione:")
    print(X_val['Class'].value_counts())
    print("\nDistribuzione delle classi nel set di test:")
    print(X_test['Class'].value_counts())

    return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded


###############################################################################
#                    SMOTE PER SOTTOCLASSI E CLASSI PRINCIPALI
###############################################################################
def apply_smote_binary(X_train, y_train, k_neighbors=1):

   # Applica un unico SMOTE al dataset binario
    #(Class = Target/Non-Target) IGNORANDO la subdivisione Subclass.


    # Copia X_train per non sovrascrivere i dati originali
    X_train_copy = X_train.copy()

    # Se ci sono colonne numeriche:
    numeric_columns = X_train_copy.select_dtypes(include=[np.number]).columns

    # Imputazione dei NaN sulle sole colonne numeriche del train
    imputer = SimpleImputer(strategy='mean')
    X_train_num = pd.DataFrame(
        imputer.fit_transform(X_train_copy[numeric_columns]),
        columns=numeric_columns,
        index=X_train_copy.index
    )

    # SMOTE binario
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled_num, y_resampled = smote.fit_resample(X_train_num, y_train)

    # Se vuoi ri-aggiungere eventuali colonne non numeriche:
    # (tipicamente non servono per il training di un modello ML,
    #  ma se desideri mantenerle, devi "replicarle" per le righe sintetiche.)
    # In molti casi, si procede solo con le feature numeriche.
    # Per semplicità, restituiamo le sole feature numeriche bilanciate.

    X_resampled = pd.DataFrame(X_resampled_num, columns=numeric_columns)

    return X_resampled, y_resampled


###############################################################################
#                      MODELLI: RF, SVM, LightGBM (invariati)
###############################################################################
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    # Rimuovi colonne non necessarie
    X_train = X_train.drop(columns=['Class', 'Parent', 'Subclass', 'File Name'], errors='ignore')
    X_val = X_val.drop(columns=['Class', 'Parent', 'Subclass', 'File Name'], errors='ignore')
    X_test = X_test.drop(columns=['Class', 'Parent', 'Subclass', 'File Name'], errors='ignore')

    # Trova tutte le classi presenti nei set
    all_classes = np.unique(np.concatenate((y_train, y_val, y_test)))

    # Impostazioni per la Grid Search
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 7, 9],
        'min_samples_split': [10, 15],
        'min_samples_leaf': [4, 5]
    }

    model = RandomForestClassifier(max_features="sqrt", random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Valutazione su Validation Set
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    logloss_val = log_loss(y_val, y_val_proba, labels=np.unique(np.concatenate((y_val, y_val_pred))))
    print(f"Accuratezza sul Validation Set: {accuracy_val:.4f}")
    print(f"Log Loss sul Validation Set: {logloss_val:.4f}")

    # Report di classificazione per il Validation Set
    print("\n=== Report di Classificazione - Validation Set ===")
    print(classification_report(y_val, y_val_pred, digits=4,labels=np.unique(np.concatenate((y_val, y_val_pred))),
                                zero_division=0))

    # Valutazione su Test Set
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    logloss_test = log_loss(y_test, y_test_proba, labels=np.unique(np.concatenate((y_test, y_test_pred))))
    print(f"Accuratezza sul Test Set: {accuracy_test:.4f}")
    print(f"Log Loss sul Test Set: {logloss_test:.4f}")

    # Report di classificazione per il Test Set
    print("\n=== Report di Classificazione - Test Set ===")
    print(classification_report(y_test, y_test_pred,digits=4, labels=np.unique(np.concatenate((y_test, y_test_pred))),
                                zero_division=0))

    return best_model


def train_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # Codifica le etichette se non sono numeriche
    if y_train.dtype == 'O':
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train, y_train)

    y_val_pred = svm_model.predict(X_val)
    y_test_pred = svm_model.predict(X_test)

    print("Distribuzione delle classi reali nel set di validazione:", np.bincount(y_val))
    print("Distribuzione delle classi predette nel set di validazione:", np.bincount(y_val_pred))

    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val, y_val_pred,digits=4, zero_division=0))

    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    print("Distribuzione delle classi reali nel set di test:", np.bincount(y_test))
    print("Distribuzione delle classi predette nel set di test:", np.bincount(y_test_pred))

    print("Report di classificazione del set di test:")
    print(classification_report(y_test, y_test_pred, digits=4,zero_division=0))

    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f"Accuratezza sul set di test: {accuracy_test:.4f}")

    return svm_model


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    # Rimuove valori NaN da y e applica la maschera a X
    na_mask_train = ~pd.isna(y_train)
    X_train = X_train[na_mask_train]
    y_train = y_train[na_mask_train]
    print(f"Dimensioni dopo NaN nel set di addestramento: X_train: {X_train.shape}, y_train: {y_train.shape}")

    na_mask_val = ~pd.isna(y_val)
    X_val = X_val[na_mask_val]
    y_val = y_val[na_mask_val]
    print(f"Dimensioni dopo NaN nel set di validazione: X_val: {X_val.shape}, y_val: {y_val.shape}")

    na_mask_test = ~pd.isna(y_test)
    X_test = X_test[na_mask_test]
    y_test = y_test[na_mask_test]
    print(f"Dimensioni dopo NaN nel set di test: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Controlla le classi uniche
    print("Classi nel set di addestramento:", np.unique(y_train))
    print("Classi nel set di validazione:", np.unique(y_val))
    print("Classi nel set di test:", np.unique(y_test))

    # Mappa le etichette
    label_encoder = LabelEncoder()
    y_train_mapped = label_encoder.fit_transform(y_train)
    y_val_mapped = label_encoder.transform(y_val)
    y_test_mapped = label_encoder.transform(y_test)

    # Crea un dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train_mapped)
    val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)

    # Imposta i parametri del modello
    params = {
        'objective': 'multiclass',  # se binario, 'objective': 'binary'
        'num_class': len(np.unique(y_train)),  # per binary: 1 (ma LightGBM internamente gestisce 2 classi)
        'metric': 'multi_logloss',
        'random_state': 42,
        'verbose': -1
    }

    # Addestra il modello
    lgb_model = lgb.train(params, train_data, valid_sets=[val_data], valid_names=['valid'], num_boost_round=1000)

    # Previsioni sul set di validazione
    y_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)

    # Valuta il modello sul set di validazione
    print("Report di classificazione del set di validazione:")
    print(classification_report(y_val_mapped, y_val_pred_classes, digits=4,zero_division=1))

    accuracy_val = accuracy_score(y_val_mapped, y_val_pred_classes)
    print(f"Accuratezza sul set di validazione: {accuracy_val:.4f}")

    # Previsioni sul set di test
    y_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)

    # Valuta il modello sul set di test
    print("Report di classificazione del set di test:")
    print(classification_report(y_test_mapped, y_test_pred_classes, digits=4,zero_division=1))

    accuracy_test = accuracy_score(y_test_mapped, y_test_pred_classes)
    print(f"Accuratezza sul set di test: {accuracy_test:.4f}")

    return lgb_model


###############################################################################
#                  FUNZIONI PER PLOTTARE LE MATRICI DI CONFUSIONE
###############################################################################
def rf_plot_confusion_matrices(model,
                               X_val, y_val_encoded,
                               X_test, y_test_encoded,
                               display_labels=None):
    """
    Mostra due confusion matrix (val/test) in formato percentuale
    usando le previsioni di un modello (Random Forest).

    Parametri:
      - model: modello addestrato
      - X_val, y_val_encoded: dati e label (numeriche) di validazione
      - X_test, y_test_encoded: dati e label (numeriche) di test
      - display_labels: lista di etichette testuali da mostrare sugli assi
                        (es. ["Non-Target","Target"])
                        Se None, verranno usate etichette generiche.
    """
    # Predizioni del modello
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Se non fornisci etichette, usiamo placeholder
    if display_labels is None:
        display_labels = ["Class 0", "Class 1"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Validazione
    ConfusionMatrixDisplay.from_predictions(
        y_val_encoded,
        y_val_pred,
        display_labels=display_labels,
        normalize='true',  # normalizza in percentuale (per riga)
        cmap='Greens',
        colorbar=False,
        ax=axes[0]
    )
    axes[0].set_title("Random Forest - Confus. Matrix (Validation)")

    # Test
    ConfusionMatrixDisplay.from_predictions(
        y_test_encoded,
        y_test_pred,
        display_labels=display_labels,
        normalize='true',
        cmap='Greens',
        colorbar=False,
        ax=axes[1]
    )
    axes[1].set_title("Random Forest - Confus. Matrix (Test)")

    for ax in axes:
        ax.grid(False)

    plt.tight_layout()
    plt.show()


###############################################################################
#                  2) SVM - Confusion Matrix
###############################################################################
def svm_plot_confusion_matrices(model,
                                X_val, y_val,
                                X_test, y_test):
    """
    Genera due confusion matrix (validazione e test) in percentuali
    per un modello SVM.

    Qui y_val, y_test sono etichette testuali (es. 'Target','Non-Target').
    Vengono encodate con LabelEncoder all'interno.
    Le etichette reali compaiono sugli assi.
    """
    le = LabelEncoder()
    y_val_encoded = le.fit_transform(y_val)
    y_test_encoded = le.transform(y_test)

    # Predizioni del modello
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Convertiamo le predizioni in codici
    y_val_pred_encoded = le.transform(y_val_pred)
    y_test_pred_encoded = le.transform(y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Validazione
    ConfusionMatrixDisplay.from_predictions(
        y_val_encoded,
        y_val_pred_encoded,
        display_labels=le.classes_,  # etichette reali
        normalize='true',
        cmap='Greens',
        colorbar=False,
        ax=axes[0]
    )
    axes[0].set_title("SVM - Confus. Matrix (Validation)")

    # Test
    ConfusionMatrixDisplay.from_predictions(
        y_test_encoded,
        y_test_pred_encoded,
        display_labels=le.classes_,
        normalize='true',
        cmap='Greens',
        colorbar=False,
        ax=axes[1]
    )
    axes[1].set_title("SVM - Confus. Matrix (Test)")

    for ax in axes:
        ax.grid(False)

    plt.tight_layout()
    plt.show()


###############################################################################
#                  3) LightGBM - Confusion Matrix
###############################################################################
def lightgbm_plot_confusion_matrices(model,
                                     X_val, y_val_encoded,
                                     X_test, y_test_encoded,
                                     display_labels=None):
    """
    Genera due confusion matrix (validazione e test) in percentuali
    per un modello LightGBM. Le etichette (encoded) vengono confrontate
    con le argmax delle probabilità di previsione.

    Parametri:
      - model: modello LightGBM addestrato
      - X_val, y_val_encoded: dati e label numeriche di validazione
      - X_test, y_test_encoded: dati e label numeriche di test
      - display_labels: etichette testuali da mostrare sugli assi
    """
    # Predizioni di probabilità
    y_val_pred_proba = model.predict(X_val)
    y_test_pred_proba = model.predict(X_test)

    # Argmax per ottenere la classe predetta
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)

    if display_labels is None:
        display_labels = ["Class 0", "Class 1"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Validazione
    ConfusionMatrixDisplay.from_predictions(
        y_val_encoded,
        y_val_pred,
        display_labels=display_labels,
        normalize='true',
        cmap='Greens',
        colorbar=False,
        ax=axes[0]
    )
    axes[0].set_title("LightGBM - Confus. Matrix (Validation)")

    # Test
    ConfusionMatrixDisplay.from_predictions(
        y_test_encoded,
        y_test_pred,
        display_labels=display_labels,
        normalize='true',
        cmap='Greens',
        colorbar=False,
        ax=axes[1]
    )
    axes[1].set_title("LightGBM - Confus. Matrix (Test)")

    for ax in axes:
        ax.grid(False)

    plt.tight_layout()
    plt.show()