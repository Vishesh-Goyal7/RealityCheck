from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / '.env')


@dataclass(frozen=True)
class GenerationConfig:
    watsonx_url: str
    watsonx_api_key: str
    watsonx_project_id: str
    llama_model_id: str
    mistral_model_id: str
    granite_model_id: str
    timeout_seconds: int = 120
    max_completion_tokens: int = 80
    temperature: float = 0.1
    top_p: float = 1.0
    time_limit_ms: int = 120000
    use_chat_api: bool = True
    chat_api_version: str = '2024-10-08'
    text_api_version: str = '2024-10-10'


@dataclass(frozen=True)
class PathsConfig:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / 'data'
    outputs_dir: Path = ROOT_DIR / 'outputs'

    @property
    def processed_dataset_path(self) -> Path:
        return self.data_dir / 'truthfulqa_processed.json'


def _require_env(name: str) -> str:
    value = os.getenv(name, '').strip()
    if not value:
        raise ValueError(f'Missing required environment variable: {name}')
    return value


def load_generation_config() -> GenerationConfig:
    return GenerationConfig(
        watsonx_url=_require_env('WATSONX_URL').rstrip('/'),
        watsonx_api_key=_require_env('WATSONX_API_KEY'),
        watsonx_project_id=_require_env('WATSONX_PROJECT_ID'),
        llama_model_id=os.getenv('WATSONX_LLAMA_MODEL_ID', 'meta-llama/llama-3-3-70b-instruct').strip(),
        mistral_model_id=os.getenv('WATSONX_MISTRAL_MODEL_ID', 'mistralai/mistral-medium-2505').strip(),
        granite_model_id=os.getenv('WATSONX_GRANITE_MODEL_ID', 'ibm/granite-3-3-8b-instruct').strip(),
        timeout_seconds=int(os.getenv('WATSONX_TIMEOUT_SECONDS', '120')),
        max_completion_tokens=int(os.getenv('WATSONX_MAX_COMPLETION_TOKENS', '80')),
        temperature=float(os.getenv('WATSONX_TEMPERATURE', '0.1')),
        top_p=float(os.getenv('WATSONX_TOP_P', '1.0')),
        time_limit_ms=int(os.getenv('WATSONX_TIME_LIMIT_MS', '120000')),
        use_chat_api=os.getenv('WATSONX_USE_CHAT_API', 'true').lower() == 'true',
        chat_api_version=os.getenv('WATSONX_CHAT_API_VERSION', '2024-10-08').strip(),
        text_api_version=os.getenv('WATSONX_TEXT_API_VERSION', '2024-10-10').strip(),
    )


def load_paths_config() -> PathsConfig:
    return PathsConfig()
