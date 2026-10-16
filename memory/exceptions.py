class MemoryError(Exception):
    pass


class ApprovalRequiredError(MemoryError):
    pass


class IdentityVerificationError(MemoryError):
    pass


class MemoryNotFoundError(MemoryError):
    pass


class MemoryConfigError(MemoryError):
    pass
