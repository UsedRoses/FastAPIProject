import json

from aiohttp_socks import ProxyConnector
import aiohttp

class ResponseData:
    """封装 HTTP 响应数据"""
    def __init__(self, status, headers, text, json_data):
        self.status = status  # HTTP 状态码
        self.headers = headers  # 响应头
        self.text = text  # 响应文本
        self.json = json_data  # 解析后的 JSON 数据

    def __repr__(self):
        return f"<ResponseData status={self.status}, json={self.json}>"

class HttpClient:
    def __init__(self, proxy_url=None):
        if proxy_url:
            connector = ProxyConnector.from_url(proxy_url, rdns=True)
            self.session = aiohttp.ClientSession(connector=connector)
        else:
            self.session = aiohttp.ClientSession()

    async def __response_data(self, response):
        text = await response.text()
        try:
            json_data = await response.json()
            if not json_data:
                json_data = json.loads(text)
        except Exception as e:
            print(str(e))
            json_data = json.loads(text)
        return ResponseData(response.status, dict(response.headers), text, json_data)

    async def get(self, url, params=None, headers=None, timeout=15) -> "ResponseData":
        async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
            return await self.__response_data(response)

    async def post(self, url, params=None, headers=None, json=None, timeout=15) -> "ResponseData":
        async with self.session.post(url, params=params, headers=headers, json=json, timeout=timeout, ssl=False) as response:
            return await self.__response_data(response)

    async def close(self):
        await self.session.close()


client = HttpClient()

async def close_client():
    await client.close()