import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
import time
import logging
import joblib
import pandas as pd
from bytek_task.api.circuit_breakers import circuit_breaker
from bytek_task.api.rate_limiters import check_rate_limit
from bytek_task.api.input_validator import FeatureValidationService, Features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where main.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'logistic_regression_model.pkl')
predictive_model = joblib.load(model_path)


class PropensityRequest(BaseModel):
    user_id: str
    features: Features


class PropensityResponse(BaseModel):
    user_id: str
    propensity: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    timestamp: float


class ReadinessResponse(BaseModel):
    """
    Modello per la risposta del readiness check.

    Attributes:
        status (str): Stato di readiness del servizio (es. "ready")
        model_loaded (bool): Indica se il modello ML è caricato e pronto all'uso
        timestamp (float): Timestamp Unix del momento della verifica
    """
    status: str
    model_loaded: bool
    timestamp: float


def preprocess_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocessa le features limitando quelle che eccedono il soft_max
    """

    soft_max_limits = {
        'frequenza_visita_30gg': 200,
        'valore_medio_carrello': 10000.0,
        'numero_acquisti_precedenti': 1000,
        'prodotti_visualizzati': 1000,
        'interazioni_servizio_clienti': 100,
        'numero_resi': 50,
        'categorie_esplorate': 100,
        'valore_lifetime': 100000.0,
        'numero_dispositivi': 20
    }

    processed_features = features.copy()

    for feature_name, soft_max_value in soft_max_limits.items():
        original_value = processed_features[feature_name]
        if original_value > soft_max_value:
            processed_features[feature_name] = soft_max_value
            print(f"Feature '{feature_name}' limitata da {original_value} a {soft_max_value}")

    return processed_features


async def predict_propensity(features: Dict[str, Any]) -> float:
    processed_features = preprocess_features(features)

    # Simula un po' di latenza del modello
    feature_order = ['tempo_ultimo_acquisto', 'frequenza_visita_30gg', 'valore_medio_carrello',
                      'numero_acquisti_precedenti', 'tempo_permanenza_sito', 'percentuale_abbandono_carrello',
                      'prodotti_visualizzati', 'interazioni_servizio_clienti', 'numero_resi', 'categorie_esplorate',
                      'valore_lifetime', 'orario_preferenziale', 'tasso_apertura_email', 'numero_dispositivi']
    X_new = pd.DataFrame([processed_features])[feature_order]
    propensity = predictive_model.predict_proba(X_new)[0][1]

    return float(round(propensity, 2))


app = FastAPI(
    title="Propensity Prediction API",
    description="API per la predizione della propensione utente basata su features comportamentali",
    version="1.0.0",
)


@app.post("/predict-propensity", response_model=PropensityResponse)
async def predict_propensity_endpoint(
        request: PropensityRequest,
        _: None = Depends(check_rate_limit)
):
    """
    Endpoint principale per la predizione della propensione utente.

    Riceve le features di un utente e restituisce una predizione della propensione
    utilizzando un modello di machine learning. Include validazione dei dati,
    rate limiting, circuit breaker e timeout protection.

    Args:
        request (PropensityRequest): Dati dell'utente e features per la predizione
        _ (None): Dependency per il controllo del rate limit

    Returns:
        PropensityResponse: Oggetto contenente user_id e propensione predetta

    Raises:
        HTTPException 422: Se la validazione dei dati fallisce\n
        HTTPException 408: Se la richiesta supera il timeout (30 secondi)\n
        HTTPException 429: Se il rate limit è superato\n
        HTTPException 503: Se il circuit breaker è aperto\n
        HTTPException 500: Per errori interni del server\n

    Example:\n
        ```
        POST /predict-propensity
          {
            "user_id": "user_f8b7c861",\n
            "features": {\n
              "tempo_ultimo_acquisto": 38.51593502067873,
              "frequenza_visita_30gg": 71.0,
              "valore_medio_carrello": 44.86634368525073,
              "numero_acquisti_precedenti": 108.0,
              "tempo_permanenza_sito": 18.47988348990493,
              "percentuale_abbandono_carrello": 0.48687508605969587,
              "prodotti_visualizzati": 299.0,
              "interazioni_servizio_clienti": 0.0,
              "numero_resi": 2.0,
              "categorie_esplorate": 7.0,
              "valore_lifetime": 287.77539647619284,
              "orario_preferenziale": 10.203915755795173,
              "tasso_apertura_email": 0.6007745530830877,
              "numero_dispositivi": 3.0\n
            }
          }
        ```

        Response:
        ```
        {
            "user_id": "user123",
            "propensity": 0.98
        }
        ```
    """
    start_time = time.time()

    try:
        async def process_request():
            # Validazione delle features in input
            feature_validator = FeatureValidationService()
            is_valid, validated_data, errors, warnings = feature_validator.validate_request(
                request.dict()
            )

            if not is_valid:
                return JSONResponse(
                    status_code=422,
                    content={
                        "message": "Validation failed",
                        "errors": errors,
                        "warnings": warnings
                    }
                )

            # Log warnings se presenti
            if warnings:
                logger.warning(f"Validation warnings for user {request.user_id}: {warnings}")

            # Predizione usando il circuit breaker per resilienza
            def predict_sync():
                return asyncio.create_task(
                    predict_propensity(request.features)
                )

            prediction_task = circuit_breaker.call(predict_sync)
            propensity = await prediction_task

            return PropensityResponse(
                user_id=request.user_id,
                propensity=propensity
            )

        # Esegui con timeout per evitare richieste che si bloccano
        result = await asyncio.wait_for(process_request(), timeout=30.0)

        # Log della latenza per monitoring
        latency = time.time() - start_time
        logger.info(f"Request processed for user {request.user_id} in {latency:.3f}s")

        return result

    except asyncio.TimeoutError:
        logger.error(f"Timeout processing request for user {request.user_id}")
        raise HTTPException(
            status_code=408,
            detail="Request timeout. The operation took too long to complete."
        )
    except HTTPException as http_exception:
        if http_exception.status_code == 503:
            raise HTTPException(
                status_code=http_exception.status_code,
                detail="Service temporarily unavailable"
            )
        if http_exception.status_code == 429:
            raise HTTPException(
                status_code=http_exception.status_code,
                detail="Rate limit exceeded. Too many requests."
            )
        else:
            raise HTTPException(
                status_code=http_exception.status_code,
                detail=f"Http exception: {http_exception}"
            )
    except Exception as e:
        logger.error(f"Internal error processing request for user {request.user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing the request."
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )


@app.get("/readiness", response_model=ReadinessResponse)
async def readiness_check():
    joblib.load(model_path)
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        timestamp=time.time()
    )