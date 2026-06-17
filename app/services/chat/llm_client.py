import os
import ssl

import certifi
import httpx
from dotenv import load_dotenv
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

load_dotenv()

ZSCALER_CERT_PATH = os.environ.get(
    "ZSCALER_CERT_PATH",
    "C:\\Users\\jatin.parmar\\Documents\\zscaler.crt",
)


def build_ssl_context() -> ssl.SSLContext:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.load_verify_locations(cafile=ZSCALER_CERT_PATH)
    return ssl_context


custom_ssl_context = build_ssl_context()
custom_client = httpx.Client(verify=custom_ssl_context, trust_env=True)
custom_async_client = httpx.AsyncClient(
    verify=custom_ssl_context,
    trust_env=True,
)


def get_llm(model_name: str, temperature: float = 0) -> AzureAIOpenAIApiChatModel:
    return AzureAIOpenAIApiChatModel(
        model=model_name,
        temperature=temperature,
        endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        credential=os.environ["AZURE_API_KEY"],
        http_client=custom_client,
        http_async_client=custom_async_client,
        http_socket_options=(),
        use_responses_api=False,
    )
