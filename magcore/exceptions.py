from __future__ import annotations


class MagcoreError(Exception):
    """Base exception for magcore."""


class DomainValidationError(MagcoreError):
    """Raised when a domain object or full problem is invalid."""


class TopologyError(MagcoreError):
    """Raised when mesh-region topology is inconsistent."""


class MeshQualityError(MagcoreError):
    """Raised when mesh geometry is degenerate or unusable."""


class AssemblyError(MagcoreError):
    """Raised during BEM system assembly."""


class SolveError(MagcoreError):
    """Raised during linear system solve."""


class EvaluationError(MagcoreError):
    """Raised during field/potential evaluation."""


class SerializationError(MagcoreError):
    """Raised when DTO/JSON serialization fails."""