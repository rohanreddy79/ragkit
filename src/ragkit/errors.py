"""Typed exceptions. Every failure ragkit raises on purpose is a subclass of RagkitError."""


class RagkitError(Exception):
    """Base class for all ragkit errors."""


class LoadError(RagkitError):
    """A document could not be read (unsupported type, unreadable PDF, missing path)."""


class StoreError(RagkitError):
    """The SQLite store is missing, inconsistent, or was built with incompatible settings."""


class ConfigurationError(RagkitError):
    """A required setting (for example an API key) is missing or invalid."""


class OptionalDependencyError(RagkitError):
    """An optional extra is required for the requested feature."""

    def __init__(self, extra: str, package: str, feature: str) -> None:
        self.extra = extra
        self.package = package
        super().__init__(
            f"{feature} needs the '{package}' package. Install it with: uv sync --extra {extra}"
        )


class GenerationError(RagkitError):
    """The generator could not produce an answer (API failure, refusal, empty context)."""
