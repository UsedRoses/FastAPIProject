from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from common.enums import ReturnCode
from models.entity.response_model import ResponseModel
import json


class ResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if response.media_type:
            content_type = response.media_type
        else:
            content_type = response.headers.get("content-type", "")

        # 如果是 SSE 流式传输，直接返回
        if content_type.startswith("text/event-stream"):
            return response
        elif content_type.startswith("text/html"):
            return response

        if response.status_code in [200, 201, 204]:
            # 创建一个BytesIO流来保存修改后的内容
            response_body = b""

            # 异步读取原始流内容
            async for chunk in response.body_iterator:
                response_body += chunk

            if response_body:
                # 解码字节流
                decoded_body = response_body.decode('utf-8')
                # 如果数据是 JSON 格式，并且可能被字符串化，解析它
                try:
                    cleaned_data = json.loads(decoded_body)
                except json.JSONDecodeError:
                    cleaned_data = decoded_body.strip('"')
            else:
                cleaned_data = None

            headers = dict(response.headers)
            headers.pop("content-length")
            # 将流数据封装到 ResponseModel 中
            response_model = ResponseModel(code=ReturnCode.SUCCESS.value, message="Success", data=cleaned_data)
            return JSONResponse(status_code=200, content=response_model.model_dump(), headers=headers)
        return response
