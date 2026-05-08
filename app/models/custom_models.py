import os
from langchain_openai import ChatOpenAI
from configuration import settings


class CustomGPTChatModel(ChatOpenAI):
    """
    专门为 ExtensionDock 第三方平台定制的大模型类。
    继承 ChatOpenAI，从而原生获得最完美的 Tool Calling 和 astream 支持。
    """

    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        # 1. 强制覆盖为你自己的第三方 API 地址
        # 注意：通常兼容 OpenAI 的接口需要加上 /v1 后缀，请根据你的实际 API 文档调整
        custom_base_url = "https://oapi.extensiondock.com/v1"


        # 3. 将自定义的 URL 和 Key 注入给父类 ChatOpenAI
        kwargs["base_url"] = custom_base_url
        kwargs["api_key"] = 'sk-mKTVlvIemrc2lAKi49B2De2b6fC346A89d5b3e7a854e5b64'
        kwargs["model"] = 'gpt-4o'

        # 4. 强制开启流式输出，完美适配你的 message_generator
        kwargs.setdefault("streaming", True)
        kwargs.setdefault("temperature", 0.5)

        # 初始化父类
        super().__init__(**kwargs)