from functools import cache
from typing import Callable, Dict, Type
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from configuration import settings
from models.custom_models import CustomGPTChatModel
from models.llm_models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    VertexAIModelName,
    CustomModelName
)

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in FakeModelName}
    | {m: m.value for m in CustomModelName}
)

# 使用 LangChain 标准的基类替代你之前的 ModelT
ModelT = BaseChatModel

# 定义一个构建器函数的签名：接收模型名称字符串，返回一个实例化的 ChatModel
ModelBuilder = Callable[[str], ModelT]

# 核心注册表：将 Enum 类型映射到对应的构建器函数
_MODEL_FACTORIES: Dict[Type, ModelBuilder] = {}

def register_model_factory(enum_class: Type):
    """
    装饰器：用于注册不同平台的模型构建策略
    """
    def decorator(func: ModelBuilder):
        _MODEL_FACTORIES[enum_class] = func
        return func
    return decorator


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    """
    统一模型获取入口
    """
    # 1. 获取对应的字符串 value
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model enum value: {model_name}")

    # 2. 动态路由：通过 Enum 的 type 找到对应的构建工厂
    # 例如传入的是 OpenAIModelName.GPT4，那么 type(model_name) 就是 OpenAIModelName
    enum_type = type(model_name)
    factory_func = _MODEL_FACTORIES.get(enum_type)

    if not factory_func:
        raise NotImplementedError(f"No factory registered for model type: {enum_type.__name__}")

    # 3. 执行工厂函数，返回实例
    return factory_func(api_model_name)


@register_model_factory(OpenAIModelName)
def build_openai(api_model_name: str) -> ModelT:
    return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)

@register_model_factory(OpenAICompatibleName)
def build_compatible(api_model_name: str) -> ModelT:
    if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
        raise ValueError("OpenAICompatible base url and endpoint must be configured")
    return BaseChatOpenAI(
        model=api_model_name,
        temperature=0.5,
        streaming=True,
        openai_api_base=settings.COMPATIBLE_BASE_URL,
        openai_api_key=settings.COMPATIBLE_API_KEY,
    )

@register_model_factory(CustomModelName)
def build_custom(api_model_name: str) -> ModelT:
    return CustomGPTChatModel(model_name=api_model_name)
