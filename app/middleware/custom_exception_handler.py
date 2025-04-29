import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from common.enums import ReturnCode
from common.public_configuration.log_configuration import LoggerData
from models.entity.exception import BusinessException, ServiceException, InfoException, SysException
from models.entity.response_model import ResponseModel


async def custom_exception_handler(request: Request, exc: Exception):
    code = ReturnCode.FAILED.value
    if isinstance(exc, StarletteHTTPException):
        status_code = exc.status_code
        detail = exc.detail
    elif isinstance(exc, RequestValidationError):
        status_code = 422
        detail = exc.errors()
    elif isinstance(exc, BusinessException) | isinstance(exc, ServiceException) | isinstance(exc, InfoException):
        status_code = 200
        detail = exc.message
        code = exc.code
    elif isinstance(exc, SysException):
        status_code = exc.status_code
        detail = exc.message
    else:
        status_code = 500
        traceback_info = get_traceback_info()
        detail = str(exc)
        LoggerData().set_message(detail).append_data(traceback=traceback_info).error()

    response_model = ResponseModel(code=code, message=detail, data=None)
    return JSONResponse(status_code=status_code, content=response_model.dict())


def get_traceback_info():
    error_message = traceback.format_exc()
    tb_lines = error_message.strip().split("\n")

    filtered_lines = []
    for i, line in enumerate(tb_lines):
        if "File" in line and "/code/app/" in line and "|" not in line or "Error" in line:
            # 捕获匹配行
            filtered_lines.append(line)
            # 捕获下一行（通常为代码行）
            if i + 1 < len(tb_lines) and not tb_lines[i + 1].strip().startswith("File"):
                filtered_lines.append(tb_lines[i + 1])

    return "\n".join(filtered_lines)