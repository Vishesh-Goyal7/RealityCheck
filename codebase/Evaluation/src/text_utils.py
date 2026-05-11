import re
import math
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = {
    'the','a','an','and','or','but','if','then','else','of','to','in','on','at','by','for','with','as','is','are','was','were','be','been','being',
    'this','that','these','those','it','its','they','them','their','there','here','from','into','about','than','so','such','can','could','would','should',
    'may','might','must','do','does','did','not','no','yes','you','your','we','our','i','he','she','his','her','who','what','when','where','why','how'
}

def normalize_text(text: str) -> str:
    if text is None:
        return ''
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokens(text: str, keep_stopwords: bool = False):
    toks = _WORD_RE.findall(normalize_text(text))
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return toks

def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def cosine_bow(a: str, b: str) -> float:
    ca, cb = Counter(tokens(a)), Counter(tokens(b))
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in common)
    na = math.sqrt(sum(v*v for v in ca.values()))
    nb = math.sqrt(sum(v*v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def best_lexical_similarity(answer: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(0.55 * cosine_bow(answer, r) + 0.45 * jaccard(answer, r) for r in references)
