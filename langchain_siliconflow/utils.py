from typing import Any, Dict

import openai
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


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
    base_url = "https://api.siliconflow.com/v1"
    values["client"] = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    values["async_client"] = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return values
