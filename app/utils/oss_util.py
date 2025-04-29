import requests
import oss2
import json
import mimetypes
import urllib.parse

from common.constant import ALIYUN_STS_TOKEN_KEY
from common.public_configuration.redis_configuration import get_redis_connection
from models.entity.exception import ServiceException
from aliyunsdksts.request.v20150401.AssumeRoleRequest import AssumeRoleRequest

from utils.aiohttp_client_util import client

OSS_REGION = 'us-west-1'


async def upload_file(file_url, oss_path: str, upload_type: str = 'image', file_name: str = None):
    mime_type, ext_name = _get_media_info(file_url)
    if ext_name == 'public':
        ext_name = 'png'
        mime_type = 'image/png'

    token = await _get_access_token()
    try:
        url = 'https://sv-prod.remusiccloudflare.workers.dev/api/upload'
        data = {
            "type": upload_type,
            "image_url": file_url,
            "oss_path": oss_path,
            "extension_name": ext_name,
            "access_key_id": token['AccessKeyId'],
            "access_key_secret": token['AccessKeySecret'],
            "security_token": token['SecurityToken']
        }
        if file_name:
            data['file_name'] = file_name

        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        }

        response = await client.post(url=url, headers=headers, json=data)
        if response.status == 200:
            response_data = response.json
            # 返回的格式{'code': 200, 'message': 'success', 'data': {'type': 'image', 'image_url': 'https://cdn.storyviewer.ai/products/image/8c2b9cc871f2870ca542553b6eafdc8e.webp', 'author_profile_url': '', 'cost': 1665}}
            return_code = response_data.get('code')
            if return_code == 200:
                file_url_data = response_data.get('data')
                url = file_url_data.get('url')
                return url
            else:
                raise ServiceException(message=response_data.get("message"))
        else:
            raise ServiceException(message='Failed to upload file')
    except Exception as e:
        raise ServiceException(message='Failed to upload')


def _get_media_info(file_url):
    """
    解析URL

    返回：媒体类型和扩展名
    """
    # 获取文件名
    parsed_url = urllib.parse.urlparse(file_url)
    file_name = parsed_url.path.split('/')[-1]

    # 获取文件扩展名
    ext_name = file_name.split('.')[-1]

    # 获取MIME类型
    mime_type, _ = mimetypes.guess_type(file_url)

    return mime_type, ext_name


async def _get_access_token():
    res = {}
    redis = await get_redis_connection()
    json_str = await redis.get(f'{ALIYUN_STS_TOKEN_KEY}')
    if json_str is None:
        aliyun_token = get_sts_token()
        if "Credentials" in aliyun_token.keys():
            res = {
                'AccessKeyId': aliyun_token['Credentials']['AccessKeyId'],
                'AccessKeySecret': aliyun_token['Credentials']['AccessKeySecret'],
                'SecurityToken': aliyun_token['Credentials']['SecurityToken'],
                'Expiration': aliyun_token['Credentials']['Expiration']
            }
            # 从申请临时通信证明到失效需要耗费时间3600s（阿里云规定），预留100s，保证通行证在redis中被获取到到时候是有效的
            await redis.set(f'{ALIYUN_STS_TOKEN_KEY}', json.dumps(res), 3500)
    else:
        res = json.loads(json_str)
    return res


def get_access_key():
    file_ram_path = '/nas/zbase/security-credentials/ali_ram'
    with open(file_ram_path, 'r') as f:
        content = f.read()
        access_key_id, access_key_secret = content.split('\n')[0:2]
    return access_key_id, access_key_secret


def get_sts_token():
    # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    # endpoint = END_POINT
    # 阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
    # access_key_id = ACCESS_KEY_ID
    # access_key_secret = ACCESS_KEY_SECRET

    access_key_id, access_key_secret = get_access_key()

    # access_key_id = OSS_ACCESS_KEY_ID
    # access_key_secret = OSS_ACCESS_KEY_SECRET

    # 使用阿里云轮盘来进行初始化
    # 填写Bucket名称，例如examplebucket。
    # bucket_name = BUCKET_NAME
    # 填写Object完整路径，例如exampledir/exampleobject.txt。Object完整路径中不能包含Bucket名称。
    # object_name = 'waplus-crm/crm_workspace_chat/requirements.txt'
    # 您可以登录RAM控制台，在RAM角色管理页面，搜索创建的RAM角色后，单击RAM角色名称，在RAM角色详情界面查看和复制角色的ARN信息。
    # 填写角色的ARN信息。格式为acs:ram::$accountID:role/$roleName。
    # $accountID为阿里云账号ID。您可以通过登录阿里云控制台，将鼠标悬停在右上角头像的位置，直接查看和复制账号ID，或者单击基本资料查看账号ID。
    # $roleName为RAM角色名称。您可以通过登录RAM控制台，单击左侧导航栏的RAM角色管理，在RAM角色名称列表下进行查看。
    role_arn = 'acs:ram::1792795717556120:role/oss-full'

    # 创建权限策略。
    # 只允许对名称为examplebucket的Bucket下的所有资源执行GetObject操作。
    # policy_text = '''
    # {
    #     "Version":"1",
    #     "Statement":[
    #         {
    #             "Action":[
    #                 "oss:PutObject"
    #             ],
    #             "Effect":"Allow",
    #             "Resource":[
    #                 "acs:oss:*:*:/scrm-data-us-oss*"
    #             ]
    #         }
    #     ]
    # }
    # '''
    clt = client.AcsClient(access_key_id, access_key_secret, OSS_REGION)
    req = AssumeRoleRequest()

    # 设置返回值格式为JSON。
    req.set_accept_format('json')
    req.set_RoleArn(role_arn)
    # 自定义角色会话名称，用来区分不同的令牌，例如可填写为session-test。
    req.set_RoleSessionName('oss-full')
    # 设置有效时间为1个小时，范围最小值为900，最大值为3600。
    req.add_query_param('DurationSeconds', '3600')
    # req.set_Policy(policy_text)
    body = clt.do_action_with_exception(req)

    # 使用RAM用户的AccessKeyId和AccessKeySecret向STS申请临时访问凭证。
    token = json.loads(oss2.to_unicode(body))
    return token
