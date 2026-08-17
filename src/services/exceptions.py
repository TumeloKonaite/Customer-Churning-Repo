"""Framework-independent exceptions raised by endpoint services."""


class APIServiceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, errors=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.errors = errors


class BatchContractViolation(APIServiceError):
    """A malformed JSON or CSV batch contract."""


class ModelNotReadyError(APIServiceError):
    def __init__(self):
        super().__init__(
            "Model artifacts are not ready yet. Please wait for training to finish.",
            status_code=503,
        )


class PredictionExecutionError(APIServiceError):
    def __init__(self, message: str):
        super().__init__(message, status_code=500)
