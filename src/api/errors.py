"""HTTP error translation for service and request-validation failures."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.schemas.errors import BATCH_CONTRACT_VERSION
from src.services.exceptions import APIServiceError, BatchContractViolation


def json_error(message: str, status_code: int = 400, errors=None) -> JSONResponse:
    payload = {"status": "error", "message": message}
    if errors:
        payload["errors"] = errors
    return JSONResponse(payload, status_code=status_code)


def batch_contract_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        {
            "status": "error",
            "message": message,
            "contract_version": BATCH_CONTRACT_VERSION,
        },
        status_code=status_code,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BatchContractViolation)
    def batch_contract_violation_handler(
        request: Request, exc: BatchContractViolation
    ) -> JSONResponse:
        return batch_contract_error(exc.message, exc.status_code)

    @app.exception_handler(APIServiceError)
    def api_service_error_handler(request: Request, exc: APIServiceError) -> JSONResponse:
        return json_error(exc.message, exc.status_code, exc.errors)
