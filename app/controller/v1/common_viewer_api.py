import json

from fastapi import APIRouter

from common.constant import ALIYUN_STS_TOKEN_KEY
from common.public_configuration.redis_configuration import get_redis_connection
from models.entity.exception import ServiceException
from utils.oss_util import get_sts_token

router = APIRouter(prefix="/api/v1/common", tags=["Common"])

@router.post("/sts-token")
async def get_sts_token_api():
    """
    获取阿里云 STS Token
    """
    # 查看 Redis 中是否已有缓存
    redis = await get_redis_connection()
    sts_token = await redis.get(ALIYUN_STS_TOKEN_KEY)
    if sts_token:
        return json.loads(sts_token)

    # 没有的话，临时生成
    aliyun_token = get_sts_token()
    if "Credentials" in aliyun_token:
        res = {
            'AccessKeyId': aliyun_token['Credentials']['AccessKeyId'],
            'AccessKeySecret': aliyun_token['Credentials']['AccessKeySecret'],
            'SecurityToken': aliyun_token['Credentials']['SecurityToken'],
            'Expiration': aliyun_token['Credentials']['Expiration'],
        }
        # 保存到 Redis，过期时间 3500s
        await redis.set(ALIYUN_STS_TOKEN_KEY, json.dumps(res), 3500)
        return res

    raise ServiceException(message='STS_TOKEN_NOT_EXIST')