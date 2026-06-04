class NotFoundError(Exception):
    """Raised when a requested resource does not exist in the database."""


class SessionEndedError(Exception):
    """Raised when an operation is attempted on a session that has already ended."""
