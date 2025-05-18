import os
import sys
from typing import Dict, Any
import copy
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from dotenv import load_dotenv
from langchain_together import ChatTogether, Together
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import VertexAIModelGarden
from langchain_google_vertexai.model_garden_maas.llama import VertexModelGardenLlama
from langchain_google_vertexai import VertexAI
from google.oauth2 import service_account
from google.auth import default, transport
from google.cloud import aiplatform
from typing import Dict, Any
import vertexai
import os
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_openai import OpenAI

# Load environment variables
load_dotenv(override=True)

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
GCP_PROJECT = os.getenv('GCP_PROJECT')
GCP_REGION = os.getenv('GCP_REGION')
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# print(f"GCP_PROJECT: {GCP_PROJECT}, GCP_REGION: {GCP_REGION}, GCP_CREDENTIALS: {GCP_CREDENTIALS}")

# aiplatform.init(
#   project=GCP_PROJECT,
#   location=GCP_REGION,
#   credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS, scopes=SCOPES),
# )
# vertexai.init(project=GCP_PROJECT, location=GCP_REGION, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS, scopes=SCOPES))

# credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS, scopes=SCOPES)
# auth_request = transport.requests.Request()
# credentials.refresh(auth_request)
# # print(credentials.token)


engine_configs = {
    "gpt-4": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4", "temperature": 0}
    },
    "gpt-4.1-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4.1-mini", "temperature": 0}
    },
    "gpt-3.5-turbo": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo", "temperature": 0}
    },
    "gpt-4o-mini-2024-07-18": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o-mini-2024-07-18", "temperature": 0}
    },
    "gpt-3.5-turbo-0125": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-0125", "temperature": 0}
    },
    "gpt-4o": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o", "temperature": 0}
    },
    "ft_openai": {
        "constructor": ChatOpenAI,
        "params": {"temperature": 0}
    },
    "DeepSeek-R1": {
        "constructor": Together,
        "params": {"model": "deepseek-ai/DeepSeek-R1", "temperature": 0.6}
    },
    "gemini-1.5-pro": {
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-pro", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-pro-002": {
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-pro-002", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-2.0-flash-exp":{
        "constructor": VertexAI,
        "params": {"model": "gemini-2.0-flash-exp", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-flash":{
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-flash", "temperature": 0, "safety_settings": safety_settings}
    },
    "meta/llama-3.1-405b-instruct-maas": {
        "constructor": VertexModelGardenLlama,
        "params": {
            "model": "meta/llama-3.1-405b-instruct-maas", 
            "temperature": 0, 
            "safety_settings": safety_settings}
    },
    # "meta_llama_3.1_70b_instrct": {
    #     "constructor": ChatOpenAI,
    #     "params": {"model": "meta/llama-3.1-70b-instruct-maas",
    #                "base_url": f"https://{GCP_REGION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT}/locations/{GCP_REGION}/endpoints/openapi/chat/completions?", 
    #                "api_key": credentials.token,
    #                "temperature": 0}
    # },
    # model=MODEL_ID,
    # base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
    # api_key=credentials.token,
    # "claude-3-5-sonnet": {
    #     "constructor": ChatAnthropicVertex,
    #     "params": {
    #         "model": "claude-3-5-sonnet-v2@20241022", 
    #         "temperature": 0, 
    #         "location": "europe-west1", 
    #         "project": "sercan-v1", 
    #         "credentials": service_account.Credentials.from_service_account_file(GCP_CREDENTIALS, scopes=SCOPES)
    #     }
    # }
}

def get_engine(model_name: str, temperature: float, inference_server_url: str=None, **kwargs):
    """
    Creates and returns a language model engine based on the specified model name.

    Args:
        model_name (str): Name of the model to initialize
        **kwargs: Additional keyword arguments to pass to the model constructor

    Returns:
        LangChain chat model instance configured with the specified parameters

    Note:
        Handles special case for 'gpt-4o-mini' by mapping it to its full version name
        For 'gemini-1.5-pro', applies safety settings automatically
    """
    if model_name == "gpt-4o-mini":
        model_name = "gpt-4o-mini-2024-07-18"
    kwargs["model"] = model_name
    kwargs["temperature"] = temperature
    # kwargs["max_tokens"] = max_tokens
    # if model_name == "gemini-1.5-pro":
    #     kwargs["safety_settings"] = safety_settings

    if inference_server_url=="https://api.together.xyz/v1":
        params = {"model": model_name, "api_key":os.getenv("TOGETHER_API_KEY"), "temperature": temperature, "max_tokens": 2000}
        params.update(kwargs)
        return ChatTogether(**params)
    
    if inference_server_url=="https://openrouter.ai/api/v1":
        params = {"model_name": model_name, "openai_api_key":os.getenv("OPENROUTER_API_KEY"), "openai_api_base":inference_server_url, "temperature": temperature, "max_tokens": 5000}
        params.update(kwargs)
        return ChatOpenAI(**params)

    if "R1-Distill" in model_name and inference_server_url is not None:
        # Using OpenAI Completions API
        params = {"model": model_name, "openai_api_key":BEARER_TOKEN, "openai_api_base":inference_server_url, "temperature": temperature, "max_tokens": 8000}
        params.update(kwargs)
        return OpenAI(**params)
    
    if inference_server_url is not None:
        params = {"model": model_name, "openai_api_key":BEARER_TOKEN, "openai_api_base":inference_server_url, "temperature": temperature, "max_tokens": 36000}
        params.update(kwargs)
        return ChatOpenAI(**params)
    
    # Start with base params from config
    params = copy.deepcopy(engine_configs[model_name]["params"])
    # Override with any provided kwargs
    params.update(kwargs)
    
    return engine_configs[model_name]["constructor"](**params)

def invoke_with_log_probs(engine, prompt, **kwargs):
    """
    Invokes the language model and returns both the response content and log probability.

    Args:
        engine: The language model engine to use
        prompt: The input prompt to send to the model
        **kwargs: Additional keyword arguments for the model invocation

    Returns:
        tuple: (content, logprob) where content is the model's response text and
               logprob is the log probability of the first token

    Note:
        Handles different log probability formats for ChatOpenAI and ChatTogether models
    """
    engine = engine.bind(logprobs=True)
    response = engine.invoke(prompt, **kwargs)
    content = response.content
    if isinstance(engine.bound, ChatOpenAI):
        logprob = response.response_metadata['logprobs']['content'][0]['logprob']
    elif isinstance(engine.bound, ChatTogether):
        logprob = response.response_metadata['logprobs']['token_logprobs'][0]
    return content, logprob

def invoke_engine(engine, prompt, **kwargs):
    """
    Simple wrapper to invoke a language model engine and return its response.

    Args:
        engine: The language model engine to use
        prompt: The input prompt to send to the model
        **kwargs: Additional keyword arguments for the model invocation

    Returns:
        str: The model's response text. For gemini-1.5-pro, returns the raw response object;
             for other models, returns just the content
    """
    
    max_retries = 3
    base_wait = 5  # Start with 1 second
    max_wait = 60  # Max wait of 1 minute
    timeout = 180
    
    model = engine.model if hasattr(engine, 'model') else engine.model_name
    for attempt in range(max_retries):
        try:            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future = executor.submit(lambda: (
                    engine.invoke(prompt, **kwargs).content 
                    if "gemini" not in model and "R1-Distill" not in model
                    else engine.invoke(prompt, **kwargs)
                ))
                
                try:
                    result = future.result(timeout=timeout)
                    return result
                except FuturesTimeoutError:
                    raise TimeoutError(f"Request timed out after {timeout} seconds")
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise the exception on the last attempt
            
            # Calculate wait time with exponential backoff, capped at max_wait
            wait_time = min(base_wait * (2 ** attempt), max_wait)
            time.sleep(wait_time)