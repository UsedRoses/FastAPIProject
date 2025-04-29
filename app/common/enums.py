from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "Pending"
    PAID = "Paid"
    EXPIRED = "Expired"


class StatusEnum(int, Enum):
    DISABLED = 0
    ENABLED = 1


class ReturnCode(int, Enum):
    SUCCESS = 100000
    FAILED = 100001
