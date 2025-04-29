from urllib import parse
import json
import phpserialize
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from common.constant import UserInfoContext, set_user_info_context, get_user_info_context
from models.table.user import ZUser, SocialAccount

"""
查询用户信息
"""
async def get_user_data_by_user_id(user_id):
    user_data = {
        "user_id": user_id,
        "username": "",
        "email": ""
    }
    user = await ZUser.filter(id=user_id).first().values('username', 'email')
    if user:
        user_data['username'] = user['username']
        user_data['email'] = user['email']
    if not user_data['email'] or not user_data['username']:
        s_user = await SocialAccount.filter(user_id=user_id).first().values('username', 'email')
        if s_user:
            user_data['username'] = s_user['username']
            user_data['email'] = s_user['email']
    return user_data


"""
解析获取user_id
"""
def get_user_id_from_identity(identity):
    # 提供的 _identity 信息
    # 1. 进行 URL 解码
    decoded = parse.unquote(identity)
    # 2. 修正序列化数据的格式，添加缺失的 'a:' 前缀
    php_serialized_part = 'a:' + decoded.split(':', 1)[-1]
    # 3. 解析 PHP 序列化数据
    try:
        data = phpserialize.loads(php_serialized_part.encode(), decode_strings=True)
        # 数据是一个包含两个元素的数组，索引1存储了JSON字符串
        identity_json = data[1]
        identity_data = json.loads(identity_json)
        return int(identity_data[0])
    except Exception as e:
        return 0


async def set_user_info(request: Request):
    _identity = request.cookies.get("_identity", "")
    if _identity:
        user_id = get_user_id_from_identity(_identity)
        if user_id:
            user_info_dict = await get_user_data_by_user_id(user_id)
            user_info_context = UserInfoContext.from_dict(user_info_dict)
            set_user_info_context(user_info_context)


class UserInfoContextMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        """
        解析user_info上下文数据,如果有的话
        :param request:
        :param call_next:
        :return:
        """
        await set_user_info(request)

        response: Response = await call_next(request)

        info_context = get_user_info_context()
        if info_context.user_id:
            response.headers["X-User-Id"] = str(info_context.user_id)
            response.headers["X-Username"] = str(info_context.username or "")
            response.headers["X-Email"] = str(info_context.email or "")
            response.headers["Access-Control-Expose-Headers"] = "X-User-Id, X-Username, X-Email"
        return response