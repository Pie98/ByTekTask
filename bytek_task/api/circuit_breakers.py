from fastapi import HTTPException
import time


class CircuitBreaker:
    """
    Implementazione del pattern Circuit Breaker per gestire i fallimenti del servizio.

    Il Circuit Breaker previene chiamate ripetute a un servizio che sta fallendo,
    permettendo al sistema di recuperare prima di riprovare.

    States:
        - CLOSED: Normale funzionamento, le richieste passano
        - OPEN: Servizio in errore, le richieste vengono rifiutate
        - HALF_OPEN: Tentativo di recupero, una richiesta di test viene permessa

    Attributes:
        failure_threshold (int): Numero di fallimenti consecutivi prima di aprire il circuito
        recovery_timeout (int): Tempo in secondi prima di tentare il recupero
        failure_count (int): Contatore dei fallimenti correnti
        last_failure_time (float): Timestamp dell'ultimo fallimento
        state (str): Stato corrente del circuit breaker
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Args:
            failure_threshold (int): Numero di fallimenti consecutivi prima di aprire il circuito.
            recovery_timeout (int): Tempo in secondi prima di tentare il recupero.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def call(self, func, *args, **kwargs):
        """
        Esegue una funzione attraverso il circuit breaker.

        Args:
            func: Funzione da eseguire
            *args: Argomenti posizionali per la funzione
            **kwargs: Argomenti nominali per la funzione

        Returns:
            Il risultato della funzione se eseguita con successo

        Raises:
            HTTPException: Se il circuito è aperto (status 503)
            Exception: Se la funzione fallisce
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e


circuit_breaker = CircuitBreaker()