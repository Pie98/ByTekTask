from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
import time
import logging
import joblib
import pandas as pd
from circuit_breakers import circuit_breaker
from rate_limiters import check_rate_limit
from input_validator import FeatureValidationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
predictive_model = joblib.load('model/logistic_regression_model.pkl')


class PropensityRequest(BaseModel):
    user_id: str
    features: Dict[str, float]


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


async def predict_propensity(features: Dict[str, Any]) -> float:
    """
    Simula la predizione del modello ML per calcolare la propensione dell'utente.

    Questa funzione simula l'inferenza di un modello di machine learning.
    In un ambiente di produzione, qui verrebbe caricato e utilizzato il modello reale.

    Args:
        features (Dict[str, Any]): Dizionario contenente le features dell'utente
                                 per la predizione

    Returns:
        float: Valore di propensione predetto, compreso tra 0.0 e 1.0

    Note:
        - Aggiunge una latenza simulata di 0.1 secondi
        - Calcola una propensione basata sulla media delle features
        - Il risultato è limitato all'intervallo [0.0, 1.0]
    """
    # Simula un po' di latenza del modello
    feature_order = ['tempo_ultimo_acquisto', 'frequenza_visita_30gg', 'valore_medio_carrello',
                      'numero_acquisti_precedenti', 'tempo_permanenza_sito', 'percentuale_abbandono_carrello',
                      'prodotti_visualizzati', 'interazioni_servizio_clienti', 'numero_resi', 'categorie_esplorate',
                      'valore_lifetime', 'orario_preferenziale', 'tasso_apertura_email', 'numero_dispositivi']
    X_new = pd.DataFrame([features])[feature_order]
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
        req: Request,
        _: None = Depends(check_rate_limit)
):
    """
    Endpoint principale per la predizione della propensione utente.

    Riceve le features di un utente e restituisce una predizione della propensione
    utilizzando un modello di machine learning. Include validazione dei dati,
    rate limiting, circuit breaker e timeout protection.

    Args:
        request (PropensityRequest): Dati dell'utente e features per la predizione
        req (Request): Oggetto request di FastAPI (per rate limiting)
        _ (None): Dependency per il controllo del rate limit

    Returns:
        PropensityResponse: Oggetto contenente user_id e propensione predetta

    Raises:
        HTTPException 422: Se la validazione dei dati fallisce
        HTTPException 408: Se la richiesta supera il timeout (30 secondi)
        HTTPException 429: Se il rate limit è superato
        HTTPException 503: Se il circuit breaker è aperto
        HTTPException 500: Per errori interni del server

    Example:
        ```
        POST /predict-propensity
        {
            "user_id": "user123",
            "features": {
                "age": 25.0,
                "income": 50000.0,
                "activity_score": 0.8
            }
        }
        ```

        Response:
        ```
        {
            "user_id": "user123",
            "propensity": 0.67
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
                raise HTTPException(
                    status_code=422,
                    detail={
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
    except HTTPException:
        raise
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
    joblib.load('model/logistic_regression_model.pkl')
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        timestamp=time.time()
    )


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "The provided input is not valid",
            "details": str(exc)
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request"
        }
    )


@app.exception_handler(503)
async def service_unavailable_handler(request: Request, exc):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "message": "The service is temporarily unavailable due to high load"
        }
    )