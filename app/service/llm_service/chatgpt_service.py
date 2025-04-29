import json
import uuid

import base64
from datetime import datetime

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5, AES

from models.entity.exception import ServiceException
from utils.aiohttp_client_util import client

RSA_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDa2oPxMZe71V4dw2r8rHWt59gH
W5INRmlhepe6GUanrHykqKdlIB4kcJiu8dHC/FJeppOXVoKz82pvwZCmSUrF/1yr
rnmUDjqUefDu8myjhcbio6CnG5TtQfwN2pz3g6yHkLgp8cFfyPSWwyOCMMMsTU9s
snOjvdDb4wiZI8x3UwIDAQAB
-----END PUBLIC KEY-----"""

GPT_API_URL = "https://extensiondock.com/chatgpt/v4/question"

def generate_aes_key(length=16):
    import random, string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def pkcs7padding(text):
    """
    明文使用PKCS7填充
    最终调用AES加密方法时，传入的是一个byte数组，要求是16的整数倍，因此需要对明文进行处理
    :param text: 待加密内容(明文)
    :return:
    """
    bs = AES.block_size  # 16
    length = len(text)
    bytes_length = len(bytes(text, encoding='utf-8'))
    # tips：utf-8编码时，英文占1个byte，而中文占3个byte
    padding_size = length if (bytes_length == length) else bytes_length
    padding = bs - padding_size % bs
    # tips：chr(padding)看与其它语言的约定，有的会使用'\0'
    padding_text = chr(padding) * padding
    return text + padding_text

def get_prod_ai_param():
    app_id = 'ng_yt_app'
    secret = 'NHGNy5YFz7HeFb'
    rsa_public_key = RSA_PUBLIC_KEY
    timestamp = int(datetime.utcnow().timestamp())
    nonce = uuid.uuid4()
    aes_secret = generate_aes_key(16)

    # rsa 加密
    rsa_key = RSA.importKey(rsa_public_key)
    cipher = Cipher_pkcs1_v1_5.new(rsa_key)  # 创建用于执行pkcs1_v1_5加密或解密的密码

    encoded_aes_secret = base64.b64encode(cipher.encrypt(aes_secret.encode('utf-8'))).decode("utf-8")
    # aes加密
    key_bytes = aes_secret.encode("utf-8")
    aes_cryptor = AES.new(key_bytes, AES.MODE_CBC, iv=key_bytes)
    # 加密的串：app_id + ":" + secret + ":" + timestamp + ":" + nonce + ":" + encodeAesSecret
    content = '{}:{}:{}:{}:{}'.format(app_id, secret, timestamp, nonce, encoded_aes_secret)
    content_padding = pkcs7padding(content)
    sign = aes_cryptor.encrypt(content_padding.encode("utf-8"))
    sign = base64.b64encode(sign).decode('utf-8')
    params = {
        "t": str(timestamp),
        "nonce": str(nonce),
        "sign": sign,
        "secret_key": encoded_aes_secret,
        "app_id": app_id,
    }
    return params

async def gpt_sse_stream(prompt: str):
    url = GPT_API_URL
    payload = {
        "text": prompt,
        "streaming": "True",
        "model": "gpt-4o-mini"
    }

    response = await client.session.post(
        url, params=get_prod_ai_param(), json=payload
    )

    try:
        if response.status != 200:
            raise ServiceException(message="GPT service error")

        async for line in iter_sse_lines(response):
            decoded_line = line
            if decoded_line.startswith("data:"):
                data = decoded_line[5:].strip()
                if data == "[DONE]":
                    yield f"data: {json.dumps({'text': '', 'status': 4})}\n\n"
                else:
                    try:
                        data_dict = json.loads(data)
                        if "message" in data_dict:
                            yield f"data: {json.dumps({'text': data_dict['message'], 'status': 3})}\n\n"
                        else:
                            yield f"data: {data}\n\n"
                    except json.JSONDecodeError:
                        print(f"JSON decode error: {data}")
    except Exception as e:
        print(f"Stream error: {str(e)}")
        yield f"data: {json.dumps({'text': {str(e)}, 'status': 4})}\n\n"


async def iter_sse_lines(response_res):
        async for line_content in response_res.content:
            if not line_content:
                continue
            yield line_content.decode('utf-8').strip()

