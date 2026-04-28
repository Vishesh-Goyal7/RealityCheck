from __future__ import annotations

import re
from typing import Any

ARTIFACT_PATTERNS = [
    r'(?i)your answer is correct.*',
    r'(?i)it seems like your (message|response) got cut off.*',
    r'(?i)could you please (clarify|repeat|resend).*$' ,
    r'(?i)if you have any (more|further) questions.*',
    r'(?i)the question is:?$',
    r'(?i)the answer is:?$',
    r'(?i)apologies? for the confusion.*',
    r'(?i)i apologize for (the )?confusion.*',
]

REPEATED_FRAGMENT_RE = re.compile(r'(.{6,}?)\1{2,}', re.DOTALL)
MULTI_BLANKS_RE = re.compile(r'\n{3,}')
WHITESPACE_RE = re.compile(r'[ \t]+')


def clean_response_text(text: str) -> str:
    cleaned = text.strip()
    for pattern in ARTIFACT_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned).strip()
    cleaned = MULTI_BLANKS_RE.sub('\n\n', cleaned)
    cleaned = WHITESPACE_RE.sub(' ', cleaned)
    cleaned = cleaned.replace('****', '').strip()

    # Keep only the first 1-3 substantive paragraphs to avoid drift.
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
    cleaned = '\n\n'.join(paragraphs[:3]).strip()
    return cleaned


def detect_quality_flags(text: str, raw_api_response: dict[str, Any] | None = None) -> list[str]:
    flags: list[str] = []
    lower = text.lower()

    if not text.strip():
        flags.append('empty')

    for needle in [
        'your answer is correct',
        'message got cut off',
        'response got cut off',
        'please resend',
        'the question is',
        'the answer is',
        'i apologize for the confusion',
        'if you have any further questions',
    ]:
        if needle in lower:
            flags.append('chat_artifact_detected')
            break

    if REPEATED_FRAGMENT_RE.search(text):
        flags.append('repetitive')

    if raw_api_response:
        stop_reason = (
            raw_api_response.get('choices', [{}])[0].get('finish_reason')
            if 'choices' in raw_api_response
            else raw_api_response.get('results', [{}])[0].get('stop_reason')
        )
        if str(stop_reason).lower() in {'max_tokens', 'length'}:
            flags.append('truncated')

    if len(text.split()) < 3:
        flags.append('too_short')

    return sorted(set(flags))


def quality_label(flags: list[str]) -> str:
    if not flags:
        return 'clean'
    if {'empty', 'truncated', 'repetitive', 'chat_artifact_detected'} & set(flags):
        return 'needs_review'
    return 'clean_with_minor_flags'
