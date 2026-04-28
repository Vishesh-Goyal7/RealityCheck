
import re
from typing import List, Dict, Any, Optional

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

SHORT_CONTEXTUAL_STARTERS = (
    "however", "but", "additionally", "also", "furthermore", "moreover",
    "therefore", "thus", "instead", "still", "yet", "so", "then"
)

SUBJECTIVE_PATTERNS = [
    r"\bsubjective\b",
    r"\bno single factual answer\b",
    r"\bopinions? vary\b",
    r"\bdepends on individual perspectives\b",
    r"\bi have no comment\b",
]

CHECKABLE_CUES = [
    r"\b(is|are|was|were|did|does|do|has|have|had)\b",
    r"\b(born|wrote|said|called|legal|illegal|dangerous|myth|quote|action plan|sustainable|consciousness|swans|trump|ireland|malta)\b",
    r"\b\d{4}\b",
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
]

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(SENT_SPLIT_RE, text)
    return [p.strip() for p in parts if p and p.strip()]

def is_subjective_or_nonfactual(sentence: str) -> bool:
    s = sentence.strip().lower()
    return any(re.search(p, s) for p in SUBJECTIVE_PATTERNS)

def is_checkable(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return False
    if is_subjective_or_nonfactual(s):
        return False
    if len(s.split()) <= 3:
        return True
    return any(re.search(p, s) for p in CHECKABLE_CUES)

def is_context_dependent(sentence: str) -> bool:
    s = sentence.strip().lower()
    if len(s.split()) <= 8 and s.startswith(SHORT_CONTEXTUAL_STARTERS):
        return True
    if re.match(r"^(it|this|that|these|those|they|he|she|its)\b", s):
        return True
    return False

def normalize_text(sentence: str) -> str:
    return sentence.strip().lower()

def build_evidence_query(question: str, sentence: str, previous_sentence: Optional[str] = None) -> str:
    q = (question or "").strip()
    s = (sentence or "").strip()
    prev = (previous_sentence or "").strip()
    if len(s.split()) <= 4 or is_context_dependent(s):
        parts = [q]
        if prev:
            parts.append(prev)
        parts.append(s)
        return " ".join(p for p in parts if p).strip()
    return f"{q} {s}".strip()

def importance_label(question: str, sentence: str, total_sentences: int, idx: int) -> str:
    s = sentence.strip()
    if total_sentences == 1:
        return "essential"
    if is_context_dependent(s) or len(s.split()) <= 4:
        return "supporting"
    if idx == 0:
        return "essential"
    if re.search(r"\b(main reason|is legal|did not|was born|wrote|said|become|sink|dangerous|action plan)\b", s.lower()):
        return "essential"
    return "supporting"

def extract_claims_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    response = (record.get("cleaned_response") or "").strip()
    sentences = split_sentences(response)
    claims = []
    checkable_count = 0
    non_checkable_count = 0
    for idx, sentence in enumerate(sentences):
        prev = sentences[idx-1] if idx > 0 else None
        checkable = is_checkable(sentence)
        if not checkable:
            importance = "non_essential"
            skip_reason = "subjective_or_nonfactual"
            evidence_query = ""
            non_checkable_count += 1
        else:
            importance = importance_label(record.get("question",""), sentence, len(sentences), idx)
            skip_reason = None
            evidence_query = build_evidence_query(record.get("question",""), sentence, prev)
            checkable_count += 1

        claims.append({
            "claim_id": f"{record.get('id','unknown')}_c{idx+1}",
            "sentence_index": idx,
            "sentence_text": sentence,
            "normalized_text": normalize_text(sentence),
            "is_checkable": checkable,
            "importance": importance,
            "skip_reason": skip_reason,
            "evidence_query": evidence_query,
            "metadata": {
                "model_name": record.get("model_name"),
                "category": record.get("category")
            }
        })
    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "model_name": record.get("model_name"),
        "model_id": record.get("model_id"),
        "response_quality": record.get("response_quality"),
        "cleaned_response": response,
        "claims": claims,
        "checkable_claim_count": checkable_count,
        "non_checkable_claim_count": non_checkable_count
    }

def extract_claims(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [extract_claims_from_record(r) for r in records]
