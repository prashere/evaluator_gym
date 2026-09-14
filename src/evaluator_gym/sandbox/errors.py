"""Sandbox service errors."""


class SubmitValidationError(ValueError):
    pass


class SubmissionConflictError(Exception):
    pass
