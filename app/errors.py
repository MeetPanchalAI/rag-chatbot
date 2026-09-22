"""Application errors. Each one carries the HTTP status the API should return."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class IngestError(AppError):
    """The PDF could not be read or contained no usable text."""

    status_code = 422
    code = "ingest_failed"


class UnsupportedFile(AppError):
    """The uploaded file is not a PDF."""

    status_code = 415
    code = "unsupported_file"


class DocumentNotFound(AppError):
    status_code = 404
    code = "document_not_found"


class ProviderError(AppError):
    """The embedding or LLM provider failed."""

    status_code = 502
    code = "provider_error"


class ProviderTimeout(AppError):
    status_code = 504
    code = "provider_timeout"


class FileTooLarge(AppError):
    status_code = 413
    code = "file_too_large"
