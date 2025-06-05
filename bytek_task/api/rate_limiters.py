from collections import deque
from collections import defaultdict
from fastapi import HTTPException, Request
import time


class RateLimiter:
    """
    Implementazione di un rate limiter basato su sliding window per limitare
    il numero di richieste per IP client.

    Attributes:
        max_requests (int): Numero massimo di richieste permesse nella finestra temporale
        window_seconds (int): Dimensione della finestra temporale in secondi
        requests (defaultdict): Dizionario che traccia le richieste per ogni IP
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Inizializza il Rate Limiter.

        Args:
            max_requests (int): Numero massimo di richieste permesse nella finestra.
                              Default: 60
            window_seconds (int): Dimensione della finestra temporale in secondi.
                                Default: 60
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """
        Verifica se una richiesta da un determinato IP è permessa.

        Args:
            client_ip (str): Indirizzo IP del client

        Returns:
            bool: True se la richiesta è permessa, False altrimenti
        """
        now = time.time()
        # Rimuove le richieste al di fuori della finestra temporale
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window_seconds
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        self.requests[client_ip].append(now)
        return True


class GlobalRateLimiter:
    def __init__(self, max_requests: int = 100000000, window_seconds: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_timestamps = deque()
        self.last_request = time.time()

    def is_allowed(self) -> bool:
        # Se l'ultima richiesta è avvenuta più tardi della finestra di secondi, questo vuol dire che tutte le richieste
        # sono da eliminare, pertanto request_timestamps viene ri-inizializzata
        if time.time() - self.last_request > self.window_seconds:
            self.request_timestamps = deque()
        self.last_request = time.time()

        # Rimuove timestamp vecchi, si usa un ciclo while e una deque con popleft così da non dover fare troppi controlli
        # dato che a differenza degli ip singoli qua ci possono essere milioni di richieste
        while self.request_timestamps and self.last_request - self.request_timestamps[0] > self.window_seconds:
            self.request_timestamps.popleft()

        if len(self.request_timestamps) >= self.max_requests:
            return False

        self.request_timestamps.append(self.last_request)
        return True


global_rate_limiter = GlobalRateLimiter()
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request):
    """
    Dependency function per verificare il rate limiting per IP client.

    Controlla se il client ha superato il limite di richieste permesse
    nella finestra temporale configurata.

    Args:
        request (Request): Oggetto request di FastAPI contenente i dettagli della richiesta

    Raises:
        HTTPException: Se il rate limit è stato superato (status 429)
    """
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip) or not global_rate_limiter.is_allowed():
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Too many requests."
        )