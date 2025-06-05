import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import uuid


class EcommerceSyntheticDataGenerator:
    def __init__(self, seed=42):
        """
        Generatore di dati sintetici per analisi di propensione all'acquisto e-commerce

        Args:
            seed (int): Seed per la riproducibilità
        """
        np.random.seed(seed)
        self.seed = seed

        # Definizione delle feature e dei loro parametri
        self.feature_params = {
            'tempo_ultimo_acquisto': {
                'type': 'lognormal',
                'mu': 3.0,
                'sigma': 1.2,
                'description': 'Giorni dall\'ultimo acquisto (0-365)',
                'range': (0, 365)
            },
            'frequenza_visita_30gg': {
                'type': 'negative_binomial',
                'n': 30,
                'p': 0.2,
                'description': 'Numero di visite negli ultimi 30 giorni (0-50)',
                'range': (0, 200)
            },
            'valore_medio_carrello': {
                'type': 'lognormal',
                'mu': 4.0,
                'sigma': 0.8,
                'description': 'Valore medio del carrello in euro (10-500)',
                'range': (1, 500)
            },
            'numero_acquisti_precedenti': {
                'type': 'negative_binomial',
                'n': 20,
                'p': 0.15,
                'description': 'Numero totale di acquisti precedenti (0-100)',
                'range': (0, 210)
            },
            'tempo_permanenza_sito': {
                'type': 'beta_scaled',
                'alpha': 2,
                'beta': 5,
                'scale': 60,
                'description': 'Tempo medio di permanenza in minuti (0-60)',
                'range': (0, 60)
            },
            'percentuale_abbandono_carrello': {
                'type': 'beta',
                'alpha': 2,
                'beta': 5,
                'description': 'Percentuale di abbandono del carrello (0-1)',
                'range': (0, 1)
            },
            'prodotti_visualizzati': {
                'type': 'negative_binomial',
                'n': 30,
                'p': 0.1,
                'description': 'Numero di prodotti visualizzati (0-200)',
                'range': (0, 410)
            },
            'interazioni_servizio_clienti': {
                'type': 'poisson',
                'lambda': 2.5,
                'description': 'Numero di interazioni con il servizio clienti (0-15)',
                'range': (0, 15)
            },
            'numero_resi': {
                'type': 'poisson',
                'lambda': 1.8,
                'description': 'Numero di resi effettuati (0-10)',
                'range': (0, 10)
            },
            'categorie_esplorate': {
                'type': 'poisson',
                'lambda': 4.2,
                'description': 'Numero di categorie di prodotti esplorate (1-20)',
                'range': (1, 20)
            },
            'valore_lifetime': {
                'type': 'lognormal',
                'mu': 5.5,
                'sigma': 1.2,
                'description': 'Valore totale acquisti lifetime in euro (0-5000)',
                'range': (0, 5000)
            },
            'orario_preferenziale': {
                'type': 'gaussian_mixture',
                'means': [10, 14, 20],
                'stds': [2, 1.5, 2.5],
                'weights': [0.3, 0.4, 0.3],
                'description': 'Orario preferenziale di navigazione (0-23)',
                'range': (0, 23)
            },
            'tasso_apertura_email': {
                'type': 'beta',
                'alpha': 3,
                'beta': 4,
                'description': 'Tasso di apertura delle email marketing (0-1)',
                'range': (0, 1)
            },
            'numero_dispositivi': {
                'type': 'poisson',
                'lambda': 2.1,
                'description': 'Numero di dispositivi utilizzati (1-8)',
                'range': (1, 8)
            },
        }

        # Pesi per la variabile target
        self.feature_weight_ranges = {
            'frequenza_visita_30gg': (0.75, 0.95),
            'valore_medio_carrello': (0.80, 1.00),
            'tasso_apertura_email': (0.65, 0.85),
            'numero_acquisti_precedenti': (0.70, 0.90),
            'tempo_ultimo_acquisto': (-0.75, -0.55),
            'tempo_permanenza_sito': (0.10, 0.30),
            'prodotti_visualizzati': (0.55, 0.75),
            'valore_lifetime': (0.35, 0.55),
            'percentuale_abbandono_carrello': (-0.50, -0.30),
            'categorie_esplorate': (0.25, 0.45),
            'interazioni_servizio_clienti': (0.10, 0.30),
            'numero_resi': (-0.35, -0.15),
            'orario_preferenziale': (0.05, 0.25),
            'numero_dispositivi': (0.10, 0.30)
        }

    @staticmethod
    def _generate_lognormal(n_samples, mu, sigma, range_limit=None):
        """Genera valori da distribuzione log-normale"""
        values = np.random.lognormal(mu, sigma, n_samples)
        if range_limit:
            values = np.clip(values, range_limit[0], range_limit[1])
        return values

    @staticmethod
    def _generate_negative_binomial(n_samples, n, p, range_limit=None):
        """Genera valori da distribuzione binomiale negativa"""
        values = np.random.negative_binomial(n, p, n_samples)
        if range_limit:
            values = np.clip(values, range_limit[0], range_limit[1])
        return values

    @staticmethod
    def _generate_beta(n_samples, alpha, beta, range_limit=None):
        """Genera valori da distribuzione beta"""
        values = np.random.beta(alpha, beta, n_samples)
        if range_limit:
            values = np.clip(values, range_limit[0], range_limit[1])
        return values

    @staticmethod
    def _generate_beta_scaled(n_samples, alpha, beta, scale, range_limit=None):
        """Genera valori da distribuzione beta scalata"""
        values = np.random.beta(alpha, beta, n_samples) * scale
        if range_limit:
            values = np.clip(values, range_limit[0], range_limit[1])
        return values

    @staticmethod
    def _generate_poisson( n_samples, lambda_param, range_limit=None):
        """Genera valori da distribuzione di Poisson"""
        values = np.random.poisson(lambda_param, n_samples)
        if range_limit:
            values = np.clip(values, range_limit[0], range_limit[1])
        return values

    @staticmethod
    def _generate_gaussian_mixture( n_samples, means, stds, weights, range_limit=None):
        """Genera valori da una mistura di gaussiane"""
        # Scegli la componente per ogni campione
        components = np.random.choice(len(means), n_samples, p=weights)
        values = np.zeros(n_samples)

        for i, comp in enumerate(components):
            values[i] = np.random.normal(means[comp], stds[comp])

        if range_limit:
            values = np.clip(values, range_limit[0], range_limit[1])
        return values

    def generate_features(self, n_samples):
        """
        Genera le features per n_samples utenti

        Args:
            n_samples (int): Numero di campioni da generare

        Returns:
            pd.DataFrame: DataFrame con le features generate
        """
        features_data = {}

        for feature_name, params in self.feature_params.items():
            if params['type'] == 'lognormal':
                values = self._generate_lognormal(
                    n_samples, params['mu'], params['sigma'], params['range']
                )
            elif params['type'] == 'negative_binomial':
                values = self._generate_negative_binomial(
                    n_samples, params['n'], params['p'], params['range']
                )
            elif params['type'] == 'beta':
                values = self._generate_beta(
                    n_samples, params['alpha'], params['beta'], params['range']
                )
            elif params['type'] == 'beta_scaled':
                values = self._generate_beta_scaled(
                    n_samples, params['alpha'], params['beta'], params['scale'], params['range']
                )
            elif params['type'] == 'poisson':
                values = self._generate_poisson(
                    n_samples, params['lambda'], params['range']
                )
            elif params['type'] == 'gaussian_mixture':
                values = self._generate_gaussian_mixture(
                    n_samples, params['means'], params['stds'], params['weights'], params['range']
                )

            features_data[feature_name] = values

        return pd.DataFrame(features_data)

    def generate_target(self, features_df, seed=None, return_weights=False):
        """
        Genera la variabile target basata sulle features con pesi casuali per ogni osservazione

        Args:
            features_df (pd.DataFrame): DataFrame con le features
            seed (int, optional): Seed per la riproducibilità
            return_weights (bool): Se True, restituisce anche i pesi medi utilizzati

        Returns:
            np.array: Array con la variabile target binaria
            np.array: Array con le probabilità (sempre restituito)
            dict: Dizionario con i pesi medi utilizzati (solo se return_weights=True)
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = len(features_df)

        # Standardizza le features per il calcolo dei pesi
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features_df),
            columns=features_df.columns
        )

        # Calcola il punteggio logit con pesi casuali per ogni osservazione
        logit_score = np.zeros(n_samples)
        for feature, (min_weight, max_weight) in self.feature_weight_ranges.items():
            if feature in features_scaled.columns:
                # Genera un peso casuale per ogni osservazione
                random_weights = np.random.uniform(min_weight, max_weight, n_samples)
                logit_score += features_scaled[feature] * random_weights

        # Creando la variabile binaria
        logit_score += -1.3 + np.random.normal(0, 0.4, n_samples)
        probabilities = 1 / (1 + np.exp(-logit_score))
        will_purchase = (probabilities > 0.5).astype(int)

        if return_weights:
            # Se richiesto, calcola i pesi medi utilizzati (per riferimento)
            avg_weights = {feature: (min_w + max_w) / 2
                           for feature, (min_w, max_w) in self.feature_weight_ranges.items()}
            return will_purchase, probabilities, avg_weights
        else:
            return will_purchase, probabilities

    def generate_dataset(self, n_samples=1000, output_format='json'):
        """
        Genera il dataset completo

        Args:
            n_samples (int): Numero di campioni da generare
            output_format (str): Formato di output ('json', 'dataframe', 'both')

        Returns:
            dict o pd.DataFrame: Dataset generato nel formato richiesto
        """
        print(f"Generando {n_samples} campioni...")

        # Genera le features
        features_df = self.generate_features(n_samples)

        # Genera la variabile target
        target, probabilities = self.generate_target(features_df)

        # Aggiungi target al DataFrame
        features_df['target'] = target
        features_df['purchase_probability'] = probabilities

        # Genera user_id
        user_ids = [f"user_{str(uuid.uuid4())[:8]}" for _ in range(n_samples)]

        print(f"Dataset generato con successo!")
        print(f"Distribuzione target: {np.sum(target)} acquisti su {n_samples} utenti ({np.mean(target):.2%})")

        if output_format == 'json':
            # Formato JSON richiesto
            json_data = []
            for i, user_id in enumerate(user_ids):
                user_data = {
                    'user_id': user_id,
                    'features': {
                        feature: float(features_df.iloc[i][feature])
                        for feature in features_df.columns if feature not in ['target', 'purchase_probability']
                    },
                    'target': int(target[i])
                }
                json_data.append(user_data)
            return json_data

        elif output_format == 'dataframe':
            features_df['user_id'] = user_ids
            return features_df

        elif output_format == 'both':

            # JSON
            json_data = []
            for i, user_id in enumerate(user_ids):
                user_data = {
                    'user_id': user_id,
                    'features': {
                        feature: float(features_df.iloc[i][feature])
                        for feature in features_df.columns if feature not in ['target', 'purchase_probability']
                    },
                    'target': int(target[i])
                }
                json_data.append(user_data)

            # DataFrame
            df = features_df.copy()
            df['user_id'] = user_ids

            return {'dataframe': df, 'json': json_data}

    @staticmethod
    def save_dataset(dataset, filename, format_type='json'):
        """
        Salva il dataset su file

        Args:
            dataset: Dataset da salvare
            filename (str): Nome del file
            format_type (str): Tipo di formato ('json', 'csv')
        """
        if format_type == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"Dataset salvato in {filename}")

        elif format_type == 'csv':
            if isinstance(dataset, pd.DataFrame):
                dataset.to_csv(filename, index=False)
                print(f"Dataset salvato in {filename}")
            else:
                print("Per salvare in CSV, il dataset deve essere un DataFrame")


# Esempio di utilizzo
if __name__ == "__main__":
    # Inizializza il generatore
    generator = EcommerceSyntheticDataGenerator(seed=42)

    # Genera in formato JSON
    print("\n GENERAZIONE DATASET ")
    n_samples = 100000
    dataset = generator.generate_dataset(n_samples, output_format='both')
    json_dataset = dataset['json']
    df_dataset = dataset['dataframe']

    # Salva i dataset
    generator.save_dataset(json_dataset, 'data/ecommerce_synthetic_data.json', 'json')
    generator.save_dataset(df_dataset, 'data/ecommerce_synthetic_data.csv', 'csv')

    print(f"\n STATISTICHE FINALI ")
    print(f"Totale utenti generati: {len(json_dataset)}")
    print(f"Utenti che effettueranno acquisti: {sum(1 for user in json_dataset if user['target'] == 1)}")
