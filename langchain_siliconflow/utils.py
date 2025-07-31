from typing import Any, Dict

import openai
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
import os.path

def validate_environment(values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and prepare SiliconFlow API key and initialize clients."""
    values["siliconflow_api_key"] = convert_to_secret_str(
        get_from_dict_or_env(
            values,
            "siliconflow_api_key",
            "SILICONFLOW_API_KEY",
        )
    )
    api_key = values["siliconflow_api_key"].get_secret_value()
    base_url = get_from_dict_or_env(
        values,
        "base_url",
        "SILICONFLOW_BASE_URL",
        "https://api.siliconflow.com"
    )
    # Normalize URL - strip trailing slashes and ensure proper format
    base_url = base_url.rstrip("/")
    if "siliconflow" in base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    values["client"] = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    values["async_client"] = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return values
