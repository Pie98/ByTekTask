import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def analyze_feature_distributions(csv_file_path='../data/commerce_synthetic_data.csv'):
    """
    Analizza e visualizza le distribuzioni delle features del dataset e-commerce

    Args:
        csv_file_path (str): Percorso del file CSV da analizzare
    """

    # Carica il dataset
    print("Caricamento dataset...")
    df = pd.read_csv(csv_file_path).drop(columns=['user_id'])

    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    print(f"Tasso di conversione: {df['target'].mean():.2%}\n")

    # Identifica le colonne numeriche (escludi user_id e target)
    numeric_features = [col for col in df.columns
                        if col not in ['user_id', 'target', 'purchase_probability']]

    print("STATISTICHE DESCRITTIVE")
    print("=" * 50)
    stats_df = df[numeric_features].describe()
    print(stats_df.round(2))

    # Analisi delle distribuzioni
    print("\nANALISI DISTRIBUZIONI")
    print("=" * 50)

    distribution_info = []

    for feature in numeric_features:
        data = df[feature].dropna()

        # Statistiche di base
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        # Test di normalità
        _, p_value_shapiro = stats.shapiro(data[:5000] if len(data) > 5000 else data)
        is_normal = p_value_shapiro > 0.05

        distribution_info.append({
            'Feature': feature,
            'Media': round(mean_val, 2),
            'Mediana': round(median_val, 2),
            'Std Dev': round(std_val, 2),
            'Skewness': round(skewness, 2),
            'Kurtosis': round(kurtosis, 2),
            'Normale': 'Sì' if is_normal else 'No',
            'Min': round(data.min(), 2),
            'Max': round(data.max(), 2)
        })

    dist_df = pd.DataFrame(distribution_info)
    print(dist_df.to_string(index=False))

    # Correlazioni con il target
    print(f"\nCORRELAZIONI CON TARGET")
    print("=" * 50)
    correlations = []

    for feature in numeric_features:
        corr_pearson = df[feature].corr(df['target'])
        correlations.append({
            'Feature': feature,
            'Correlazione': round(corr_pearson, 3),
            'Forza': get_correlation_strength(abs(corr_pearson))
        })

    corr_df = pd.DataFrame(correlations).sort_values('Correlazione', key=abs, ascending=False)
    print(corr_df.to_string(index=False))

    from sklearn.feature_selection import mutual_info_classif
    # Calculate mutual information
    mi = mutual_info_classif(df[numeric_features], df['target'], random_state=42)

    # Create DataFrame for easy viewing
    mi_df = pd.DataFrame({
        'Feature': numeric_features,
        'Mutual Information': mi
    })

    # Add relevance labels
    def relevance(score):
        if score >= 0.02:
            return 'High'
        elif score >= 0.005:
            return 'Medium'
        else:
            return 'Low'

    mi_df['Relevance'] = mi_df['Mutual Information'].apply(relevance)

    # Sort by MI score
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)
    selected_features = mi_df[mi_df['Relevance'].isin(['Medium', 'High'])]['Feature'].tolist()

    # Output the list
    print("Selected Features (Medium or High Relevance):")
    print(selected_features)
    # Print table
    print("\nMutual Information Table:")
    print(mi_df.to_string(index=False))

    # Visualizzazioni
    create_distribution_plots(df, numeric_features)
    create_correlation_heatmap(df, numeric_features)
    create_target_comparison_plots(df, numeric_features)

    return df, dist_df, corr_df


def get_correlation_strength(corr_value):
    """Classifica la forza della correlazione"""
    if corr_value >= 0.7:
        return "Molto forte"
    elif corr_value >= 0.5:
        return "Forte"
    elif corr_value >= 0.3:
        return "Moderata"
    elif corr_value >= 0.1:
        return "Debole"
    else:
        return "Molto debole"


def create_distribution_plots(df, numeric_features):
    """Crea grafici delle distribuzioni"""
    print("\nCreazione grafici delle distribuzioni...")

    # Calcola il numero di righe e colonne per i subplot
    n_features = len(numeric_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, feature in enumerate(numeric_features):
        ax = axes[i]

        # Istogramma con density plot
        data = df[feature].dropna()
        ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')

        # Aggiungi curva di densità
        try:
            kde_data = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_data, kde(kde_data), 'r-', linewidth=2, label='Densità')
        except:
            pass

        ax.set_title(f'{feature}', fontsize=10, weight='bold')
        ax.set_xlabel('Valore')
        ax.set_ylabel('Densità')
        ax.grid(True, alpha=0.3)

        # Aggiungi statistiche nel grafico
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Media: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Mediana: {median_val:.1f}')
        ax.legend(fontsize=8)

    # Rimuovi subplot vuoti
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle('Distribuzioni delle Features', fontsize=16, y=1.02)
    plt.show()


def create_correlation_heatmap(df, numeric_features):
    """Crea heatmap delle correlazioni"""
    print("\n Creazione heatmap correlazioni...")

    # Matrice di correlazione
    features_for_corr = numeric_features + ['target']
    corr_matrix = df[features_for_corr].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlazione'})

    plt.title('Matrice di Correlazione Features vs Target', fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def create_target_comparison_plots(df, numeric_features):
    """Crea box plot per confrontare le distribuzioni per target"""
    print("\nCreazione box plot per confronto target...")

    # Seleziona le 8 feature più correlate con il target
    correlations = df[numeric_features].corrwith(df['target']).abs().sort_values(ascending=False)
    top_features = correlations.head(8).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        ax = axes[i]

        # Box plot per target 0 e 1
        data_0 = df[df['target'] == 0][feature]
        data_1 = df[df['target'] == 1][feature]

        box_data = [data_0, data_1]
        box = ax.boxplot(box_data, labels=['No Acquisto', 'Acquisto'], patch_artist=True)

        # Colora i box
        box['boxes'][0].set_facecolor('lightcoral')
        box['boxes'][1].set_facecolor('lightgreen')

        ax.set_title(f'{feature}', fontsize=10, weight='bold')
        ax.grid(True, alpha=0.3)

        # Aggiungi informazioni statistiche
        mean_0 = data_0.mean()
        mean_1 = data_1.mean()
        ax.text(1, ax.get_ylim()[1] * 0.9, f'Media: {mean_0:.1f}',
                ha='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax.text(2, ax.get_ylim()[1] * 0.9, f'Media: {mean_1:.1f}',
                ha='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    plt.tight_layout()
    plt.suptitle('Confronto Distribuzioni per Target (Top 8 Features)', fontsize=14, y=1.02)
    plt.show()


def quick_summary(df):
    """Genera un riassunto rapido del dataset"""
    print("\n⚡ RIASSUNTO RAPIDO")
    print("=" * 50)

    print(f"Dimensioni dataset: {df.shape[0]:,} righe × {df.shape[1]} colonne")
    print(f"Tasso conversione: {df['target'].mean():.2%}")
    print(f"Bilanciamento classi: {df['target'].value_counts().to_dict()}")

    # Feature con correlazione più alta
    numeric_features = [col for col in df.columns if col not in ['user_id', 'target', 'purchase_probability']]
    correlations = df[numeric_features].corrwith(df['target']).abs().sort_values(ascending=False)

    print(f"\nTop 3 feature più correlate:")
    for i, (feature, corr) in enumerate(correlations.head(3).items(), 1):
        print(f"   {i}. {feature}: {corr:.3f}")

    print(f"\nFeature con valori mancanti:")
    missing = df.isnull().sum()
    missing_features = missing[missing > 0]
    if len(missing_features) == 0:
        print("   Nessuna feature con valori mancanti ")
    else:
        for feature, count in missing_features.items():
            print(f"   {feature}: {count} ({count / len(df) * 100:.1f}%)")


# Esempio di utilizzo
if __name__ == "__main__":
    print("AVVIO ANALISI DISTRIBUZIONE FEATURES E-COMMERCE")
    print("=" * 60)

    # Analizza il dataset (modifica il percorso se necessario)
    try:
        df, dist_info, corr_info = analyze_feature_distributions('data/ecommerce_synthetic_data.csv')
        quick_summary(df)

        print(f"\nAnalisi completata con successo!")
        print(f"Grafici visualizzati e statistiche generate")


    except FileNotFoundError:
        print("File CSV non trovato!")
        print("Assicurati che 'ecommerce_synthetic_data.csv' sia nella directory corrente")
        print("oppure modifica il percorso nella funzione analyze_feature_distributions()")

    except Exception as e:
        print(f" Errore durante l'analisi: {str(e)}")