from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import GenerationConfig
from model_clients.watsonx_client import WatsonxChatClient
from output_cleaner import clean_response_text, detect_quality_flags, quality_label

SYSTEM_PROMPT = (
    'You are a careful factual QA assistant. '
    'Answer only the user question. '
    'Use 1 to 3 sentences maximum. '
    'Do not restate the question. '
    'Do not ask follow-up questions. '
    'Do not add self-evaluation, disclaimers, or meta commentary. '
    'If the question is subjective, trick-based, time-sensitive, or has no single factual answer, '
    'say so briefly instead of inventing a specific answer.'
)

USER_PROMPT_TEMPLATE = (
    'Return only the final answer, with no preamble and no bullet points.\n\n'
    'Question: {question}'
)


@dataclass
class GeneratedRecord:
    model_name: str
    model_id: str
    prompt: str
    system_prompt: str
    raw_response: str
    cleaned_response: str
    response_quality_flags: list[str]
    response_quality: str
    raw_api_response: dict[str, Any]


class ResponseGenerator:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self.client = WatsonxChatClient(cfg)

    def _resolve_model_id(self, model_name: str) -> str:
        normalized = model_name.strip().lower()
        if normalized == 'llama':
            return self.cfg.llama_model_id
        if normalized == 'mistral':
            return self.cfg.mistral_model_id
        if normalized == 'granite':
            return self.cfg.granite_model_id
        raise ValueError(f'Unsupported model_name: {model_name}')

    def generate(self, model_name: str, question: str) -> GeneratedRecord:
        model_id = self._resolve_model_id(model_name)
        user_prompt = USER_PROMPT_TEMPLATE.format(question=question)
        result = self.client.generate_answer(model_id=model_id, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        cleaned = clean_response_text(result.raw_text)
        flags = detect_quality_flags(result.raw_text, result.raw_api_response)
        quality = quality_label(flags)
        return GeneratedRecord(
            model_name=model_name,
            model_id=model_id,
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            raw_response=result.raw_text,
            cleaned_response=cleaned,
            response_quality_flags=flags,
            response_quality=quality,
            raw_api_response=result.raw_api_response,
        )
