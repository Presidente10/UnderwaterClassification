import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#CSV
df = pd.read_csv(".csv")

# Separa le colonne non numeriche e quelle numeriche
metadata = df[['File Name', 'Class', 'subclass']]
features = df.drop(columns=['File Name', 'Class', 'subclass'])

# Standardizza le feature numeriche
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Applica la PCA mantenendo il 95% della varianza
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_scaled)

# Ricrea il DataFrame
features_pca_df = pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(features_pca.shape[1])])
final_df = pd.concat([metadata, features_pca_df], axis=1)

# Salva il nuovo dataset
final_df.to_csv("dataset_pca.csv", index=False)

# Salva i pesi delle feature sulle componenti principali
loading_matrix = pd.DataFrame(pca.components_, columns=features.columns, index=[f'PC{i+1}' for i in range(features_pca.shape[1])])
loading_matrix.to_csv("loading_matrix.csv")

# Salva la varianza spiegata da ogni componente
explained_variance = pd.DataFrame({
    'Componente': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Varianza Spiegata (%)': pca.explained_variance_ratio_ * 100
})
explained_variance.to_csv("explained_variance.csv")

# Stampa il numero di componenti principali mantenute
print(f"Numero di componenti principali mantenute: {features_pca.shape[1]}")
