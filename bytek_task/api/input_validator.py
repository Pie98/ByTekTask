from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from enum import Enum


class FeatureValidationLevel(Enum):
    """Livelli di validazione per diversi tipi di feature"""
    STRICT = "strict"  # Validazione rigida per percentuali, range fissi
    FLEXIBLE = "flexible"  # Validazione flessibile per contatori, metriche


class FeatureValidator:
    """Classe per definire le regole di validazione delle feature"""

    def __init__(self):
        # Mapping delle feature con le loro regole di validazione
        self.validation_rules = {
            # STRICT: Percentuali e valori con limiti logici rigidi
            'percentuale_abbandono_carrello': {
                'level': FeatureValidationLevel.STRICT,
                'min': 0.0,
                'max': 1.0,
                'description': 'Percentuale (0-1)'
            },
            'tasso_apertura_email': {
                'level': FeatureValidationLevel.STRICT,
                'min': 0.0,
                'max': 1.0,
                'description': 'Percentuale (0-1)'
            },
            'tempo_permanenza_sito': {
                'level': FeatureValidationLevel.STRICT,
                'min': 0.0,
                'max': 1440.0,  # Max 24 ore in minuti
                'description': 'Minuti (0-1440)'
            },
            'orario_preferenziale': {
                'level': FeatureValidationLevel.STRICT,
                'min': 0,
                'max': 23,
                'description': 'Ora del giorno (0-23)'
            },
            'tempo_ultimo_acquisto': {
                'level': FeatureValidationLevel.STRICT,
                'min': 0,
                'max': 3650,  # Max ~10 anni
                'description': 'Giorni (0-3650)'
            },

            # FLEXIBLE: Contatori e metriche che possono eccedere i range tipici
            'frequenza_visita_30gg': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0,
                'soft_max': 200,  # Warning oltre questo valore
                'description': 'Numero di visite'
            },
            'valore_medio_carrello': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0.0,
                'soft_max': 10000.0,
                'description': 'Euro'
            },
            'numero_acquisti_precedenti': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0,
                'soft_max': 1000,
                'description': 'Numero acquisti'
            },
            'prodotti_visualizzati': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0,
                'soft_max': 1000,
                'description': 'Numero prodotti'
            },
            'interazioni_servizio_clienti': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0,
                'soft_max': 100,
                'description': 'Numero interazioni'
            },
            'numero_resi': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0,
                'soft_max': 50,
                'description': 'Numero resi'
            },
            'categorie_esplorate': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0,
                'soft_max': 100,
                'description': 'Numero categorie'
            },
            'valore_lifetime': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 0.0,
                'soft_max': 100000.0,
                'description': 'Euro'
            },
            'numero_dispositivi': {
                'level': FeatureValidationLevel.FLEXIBLE,
                'min': 1,
                'soft_max': 20,
                'description': 'Numero dispositivi'
            }
        }

    def validate_feature(self, feature_name: str, value: float) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Valida una singola feature
        Returns: (is_valid, error_message, warning_message)
        """
        if feature_name not in self.validation_rules:
            return True, None, f"Feature '{feature_name}' non riconosciuta"

        rule = self.validation_rules[feature_name]

        # Validazione del tipo
        if not isinstance(value, (int, float)):
            return False, f"Feature '{feature_name}' deve essere numerica", None

        # Validazione del minimo (sempre rigida)
        if value < rule['min']:
            return False, f"Feature '{feature_name}' ({value}) sotto il minimo consentito ({rule['min']})", None

        # Validazione del massimo basata sul livello
        if rule['level'] == FeatureValidationLevel.STRICT:
            if 'max' in rule and value > rule['max']:
                return False, f"Feature '{feature_name}' ({value}) sopra il massimo consentito ({rule['max']})", None

        elif rule['level'] == FeatureValidationLevel.FLEXIBLE:
            if 'soft_max' in rule and value > rule['soft_max']:
                warning = f"Feature '{feature_name}' ({value}) sopra il range tipico ({rule['soft_max']})"
                return True, None, warning

        return True, None, None


class Features(BaseModel):
    """Modello per le features con validazione dinamica"""

    model_config = ConfigDict(
        extra="forbid",  # Non permette campi extra
        validate_assignment=True
    )

    tempo_ultimo_acquisto: Optional[float] = Field(None, description="Giorni dall'ultimo acquisto")
    frequenza_visita_30gg: Optional[float] = Field(None, description="Visite negli ultimi 30 giorni")
    valore_medio_carrello: Optional[float] = Field(None, description="Valore medio carrello in euro")
    numero_acquisti_precedenti: Optional[float] = Field(None, description="Numero acquisti precedenti")
    tempo_permanenza_sito: Optional[float] = Field(None, description="Tempo permanenza in minuti")
    percentuale_abbandono_carrello: Optional[float] = Field(None, description="Percentuale abbandono carrello")
    prodotti_visualizzati: Optional[float] = Field(None, description="Prodotti visualizzati")
    interazioni_servizio_clienti: Optional[float] = Field(None, description="Interazioni servizio clienti")
    numero_resi: Optional[float] = Field(None, description="Numero resi")
    categorie_esplorate: Optional[float] = Field(None, description="Categorie esplorate")
    valore_lifetime: Optional[float] = Field(None, description="Valore lifetime in euro")
    orario_preferenziale: Optional[float] = Field(None, description="Orario preferenziale")
    tasso_apertura_email: Optional[float] = Field(None, description="Tasso apertura email")
    numero_dispositivi: Optional[float] = Field(None, description="Numero dispositivi")


class APIRequest(BaseModel):
    """Modello principale per la richiesta API"""
    user_id: str = Field(..., min_length=1, description="ID utente")
    features: Features = Field(..., description="Features dell'utente")

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError("user_id non può essere vuoto")
        return v.strip()


class FeatureValidationService:
    """Servizio per la validazione delle features con gestione errori avanzata"""

    def __init__(self):
        self.validator = FeatureValidator()

    def validate_request(self, data: Dict[str, Any]) -> tuple[bool, APIRequest, list, list]:
        """
        Valida una richiesta completa
        Returns: (is_valid, validated_data, errors, warnings)
        """
        errors = []
        warnings = []

        try:
            # Validazione con Pydantic che la struttura del payload rispetti quella attesa
            validated_data = APIRequest(**data)

            # Validazione aggiuntiva delle features
            if validated_data.features:
                feature_data = validated_data.features.model_dump(exclude_none=True)
                for feature in list(self.validator.validation_rules.keys()):
                    if feature not in feature_data:
                        errors.append(f"Feature missing from input: {feature}")

                for feature_name, value in feature_data.items():
                    is_valid, error_msg, warning_msg = self.validator.validate_feature(feature_name, value)

                    if error_msg:
                        errors.append(error_msg)
                    if warning_msg:
                        warnings.append(warning_msg)

            return len(errors) == 0, validated_data, errors, warnings

        except ValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(x) for x in error['loc'])
                errors.append(f"Campo '{field_path}': {error['msg']}")

            return False, None, errors, warnings

        except Exception as e:
            errors.append(f"Errore di validazione: {str(e)}")
            return False, None, errors, warnings
