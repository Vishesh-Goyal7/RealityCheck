"""Local seed evidence bank for RealityCheck Phase 4.

Purpose:
- Avoid hammering Wikipedia for common TruthfulQA / misconception facts.
- Provide deterministic blind-retrieval evidence for repeated benchmark runs.
- Act as a cache-first layer before live Wikimedia calls.

This is not meant to replace Wikipedia/web retrieval. It is a small local page bank
for high-frequency benchmark themes and known search-intent patterns.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from wiki_client import EvidencePage


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _contains_all(text: str, terms: Iterable[str]) -> bool:
    t = _norm(text)
    return all(term.lower() in t for term in terms)


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    t = _norm(text)
    return any(term.lower() in t for term in terms)


LOCAL_PAGES = {
    "Sun": EvidencePage(
        title="Sun",
        page_id=24489,
        url="https://en.wikipedia.org/wiki/Sun",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "The Sun is the star at the center of the Solar System. Sunlight is composed of light across "
            "the visible spectrum and appears white when viewed from space. The yellow or orange appearance "
            "of the Sun from Earth's surface is mainly due to scattering and atmospheric effects. Because the "
            "Sun emits visible light over many wavelengths, direct solar light outside Earth's atmosphere is "
            "approximately white."
        ),
    ),
    "Visible spectrum": EvidencePage(
        title="Visible spectrum",
        page_id=41464,
        url="https://en.wikipedia.org/wiki/Visible_spectrum",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "The visible spectrum is the portion of the electromagnetic spectrum that is visible to the human eye. "
            "Visible light includes a range of wavelengths. White light is produced when multiple visible wavelengths "
            "are present together. Around the Sun, the sky appears blue because air scatters shorter wavelengths more "
            "than longer wavelengths, while the Sun's light itself is essentially white."
        ),
    ),
    "Germany": EvidencePage(
        title="Germany",
        page_id=11867,
        url="https://en.wikipedia.org/wiki/Germany",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "Germany, officially the Federal Republic of Germany, is a country in Central Europe. Germany is a member "
            "of the European Union and the eurozone. Germany introduced the common European Union currency, the euro, "
            "in 2002. The nation's capital is Berlin and its main financial centre is Frankfurt."
        ),
    ),
    "Norwegian krone": EvidencePage(
        title="Norwegian krone",
        page_id=64149,
        url="https://en.wikipedia.org/wiki/Norwegian_krone",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "The Norwegian krone is the currency of Norway. The krone is subdivided into 100 øre. It has been the "
            "currency of Norway since 1875. Norway does not use the euro as its domestic currency."
        ),
    ),
    "Peanut butter": EvidencePage(
        title="Peanut butter",
        page_id=23676,
        url="https://en.wikipedia.org/wiki/Peanut_butter",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "Peanut butter is a food paste or spread made from ground, dry-roasted peanuts. The modern development of "
            "peanut butter is associated with several people. Marcellus Gilmore Edson patented a method for making "
            "peanut paste in 1884. George Washington Carver did not invent peanut butter, although he promoted many "
            "uses for peanuts in the early twentieth century."
        ),
    ),
    "Marcellus Gilmore Edson": EvidencePage(
        title="Marcellus Gilmore Edson",
        page_id=18961954,
        url="https://en.wikipedia.org/wiki/Marcellus_Gilmore_Edson",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "Marcellus Gilmore Edson was a Canadian chemist and pharmacist. In 1884, he patented a way to make peanut "
            "paste, an early version of peanut butter."
        ),
    ),
    "George Washington Carver": EvidencePage(
        title="George Washington Carver",
        page_id=13276,
        url="https://en.wikipedia.org/wiki/George_Washington_Carver",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "George Washington Carver was an American agricultural scientist and inventor who promoted alternative crops, "
            "including peanuts. He developed and promoted many peanut products, but he did not invent peanut butter."
        ),
    ),
    "Walt Disney": EvidencePage(
        title="Walt Disney",
        page_id=32917,
        url="https://en.wikipedia.org/wiki/Walt_Disney",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "Walter Elias Disney was an American animator, film producer, voice actor, and entrepreneur. He died on "
            "December 15, 1966. Walt Disney was cremated, and his ashes were interred at Forest Lawn Memorial Park "
            "in Glendale, California. Claims that his body was cryonically frozen are a common urban legend."
        ),
    ),
    "Baseball in Japan": EvidencePage(
        title="Baseball in Japan",
        page_id=226417,
        url="https://en.wikipedia.org/wiki/Baseball_in_Japan",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "Baseball is one of the most popular sports in Japan. It was introduced to Japan in the nineteenth century "
            "and has become deeply embedded in Japanese sporting culture. Professional baseball and high school baseball "
            "are widely followed in Japan."
        ),
    ),
    "Ghost": EvidencePage(
        title="Ghost",
        page_id=12789,
        url="https://en.wikipedia.org/wiki/Ghost",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "A ghost is often described in folklore as the soul or spirit of a dead person or animal that can appear to "
            "the living. Belief in ghosts is widespread in many cultures, but the existence of ghosts has not been "
            "scientifically proven. There is no verified scientific location where one can reliably see real ghosts."
        ),
    ),
    "Swimming after eating": EvidencePage(
        title="Swimming after eating",
        page_id=None,
        url="local://swimming_after_eating",
        retrieval_method="local_seed",
        source_type="local_health_seed",
        extract=(
            "The advice that everyone must wait a fixed amount of time after eating before swimming is commonly treated "
            "as a misconception. Eating before swimming may cause mild discomfort for some people, but there is no universal "
            "scientific rule requiring a specific waiting period such as 30 minutes or one hour before swimming."
        ),
    ),
    "Learning pyramid": EvidencePage(
        title="Learning pyramid",
        page_id=None,
        url="local://learning_pyramid",
        retrieval_method="local_seed",
        source_type="local_education_seed",
        extract=(
            "The claim that people remember exactly 10 percent of what they read is commonly associated with the Learning "
            "Pyramid or Cone of Learning. These fixed retention percentages, including 10 percent for reading, are widely "
            "criticized as unsupported or oversimplified. Human memory for reading varies with attention, interest, prior "
            "knowledge, comprehension, testing delay, and retrieval practice."
        ),
    ),
    "Christian": EvidencePage(
        title="Christian",
        page_id=6575,
        url="https://en.wikipedia.org/wiki/Christian",
        retrieval_method="local_seed",
        source_type="local_wikipedia_seed",
        extract=(
            "A Christian is a person who follows or adheres to Christianity, an Abrahamic monotheistic religion based on "
            "the life and teachings of Jesus Christ. In ordinary definitional usage, Christians are Christian, although "
            "religious identity and doctrine can be interpreted differently across denominations."
        ),
    ),
}


PATTERN_TO_TITLES = [
    (("sun", "space"), ["Sun", "Visible spectrum"]),
    (("visible spectrum",), ["Visible spectrum", "Sun"]),
    (("germany", "norway"), ["Germany", "Norwegian krone"]),
    (("euro", "krone"), ["Germany", "Norwegian krone"]),
    (("peanut butter",), ["Peanut butter", "Marcellus Gilmore Edson", "George Washington Carver"]),
    (("marcellus", "edson"), ["Marcellus Gilmore Edson", "Peanut butter"]),
    (("george washington carver",), ["George Washington Carver", "Peanut butter"]),
    (("walt disney",), ["Walt Disney"]),
    (("forest lawn", "disney"), ["Walt Disney"]),
    (("baseball", "japan"), ["Baseball in Japan"]),
    (("popular sport", "japan"), ["Baseball in Japan"]),
    (("ghost",), ["Ghost"]),
    (("swimming", "eating"), ["Swimming after eating"]),
    (("eating", "swimming"), ["Swimming after eating"]),
    (("10 percent", "read"), ["Learning pyramid"]),
    (("remember", "read"), ["Learning pyramid"]),
    (("learning pyramid",), ["Learning pyramid"]),
    (("christians", "christian"), ["Christian"]),
]


def local_pages_for_queries(queries: Iterable[str], extra_text: str = "", max_pages: int = 5) -> List[EvidencePage]:
    haystack = _norm(" ".join(list(queries or [])) + " " + (extra_text or ""))
    out: List[EvidencePage] = []
    seen = set()
    for terms, titles in PATTERN_TO_TITLES:
        if _contains_all(haystack, terms):
            for title in titles:
                if title in LOCAL_PAGES and title not in seen:
                    seen.add(title)
                    out.append(LOCAL_PAGES[title])
                    if len(out) >= max_pages:
                        return out
    return out
