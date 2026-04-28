from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from config import GenerationConfig

IAM_TOKEN_URL = 'https://iam.cloud.ibm.com/identity/token'


@dataclass
class WatsonxGenerationResult:
    model_id: str
    raw_text: str
    raw_api_response: dict[str, Any]


class WatsonxChatClient:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self._token: str | None = None
        self._token_expiry: float = 0.0

    def _get_bearer_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expiry - 60:
            return self._token

        response = requests.post(
            IAM_TOKEN_URL,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data={
                'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
                'apikey': self.cfg.watsonx_api_key,
            },
            timeout=self.cfg.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload['access_token']
        self._token_expiry = now + int(payload.get('expires_in', 3600))
        return self._token

    def _headers(self) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {self._get_bearer_token()}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

    def generate_answer(self, model_id: str, system_prompt: str, user_prompt: str) -> WatsonxGenerationResult:
        if self.cfg.use_chat_api:
            return self._chat_generate(model_id, system_prompt, user_prompt)
        return self._legacy_generate(model_id, system_prompt, user_prompt)

    def _chat_generate(self, model_id: str, system_prompt: str, user_prompt: str) -> WatsonxGenerationResult:
        url = f"{self.cfg.watsonx_url}/ml/v1/text/chat?version={self.cfg.chat_api_version}"
        payload = {
            'model_id': model_id,
            'project_id': self.cfg.watsonx_project_id,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': user_prompt,
                        }
                    ],
                },
            ],
            'max_tokens': self.cfg.max_completion_tokens,
            'temperature': self.cfg.temperature,
            'top_p': self.cfg.top_p,
            'time_limit': self.cfg.time_limit_ms,
        }
        response = requests.post(url, headers=self._headers(), json=payload, timeout=self.cfg.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        text = data['choices'][0]['message']['content']
        return WatsonxGenerationResult(model_id=model_id, raw_text=text, raw_api_response=data)

    def _legacy_generate(self, model_id: str, system_prompt: str, user_prompt: str) -> WatsonxGenerationResult:
        url = f"{self.cfg.watsonx_url}/ml/v1/text/generation?version={self.cfg.text_api_version}"
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            'model_id': model_id,
            'project_id': self.cfg.watsonx_project_id,
            'input': combined_prompt,
            'parameters': {
                'decoding_method': 'greedy',
                'max_new_tokens': self.cfg.max_completion_tokens,
                'temperature': self.cfg.temperature,
                'top_p': self.cfg.top_p,
                'return_options': {'input_text': False},
            },
        }
        response = requests.post(url, headers=self._headers(), json=payload, timeout=self.cfg.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        text = data['results'][0]['generated_text']
        return WatsonxGenerationResult(model_id=model_id, raw_text=text, raw_api_response=data)
