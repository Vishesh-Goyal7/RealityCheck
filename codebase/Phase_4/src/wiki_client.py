"""Cache-first retrieval client for RealityCheck Phase 4.

Patch goals:
1. Do not depend on live Wikipedia for every experiment run.
2. Use a small local seed evidence bank before network retrieval.
3. Use persistent disk cache for search/page/REST/external fetches.
4. Use polite Wikimedia networking: meaningful User-Agent, throttle, 429 backoff.
5. Avoid retry storms: when Wikimedia says 429, cool down and stop blasting fallback calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class EvidencePage:
    title: str
    page_id: Optional[int]
    url: str
    extract: str
    retrieval_method: str = "search"  # source_url/search/external_source/local_seed/rest_summary/cache
    source_type: str = "wikipedia"     # wikipedia/wikiquote/external/local_*


PSEUDO_SOURCE_VALUES = {"", "subjective", "indexical", "false stereotype", "unknown", "none", "null"}


class SourceAwareClient:
    def __init__(
        self,
        language: str = "en",
        timeout_seconds: int = 25,
        user_agent: Optional[str] = None,
        sleep_seconds: Optional[float] = None,
        max_external_chars: int = 30000,
        cache_dir: Optional[str] = None,
        max_429_per_claim: int = 2,
    ) -> None:
        self.language = language
        self.timeout_seconds = timeout_seconds
        self.sleep_seconds = float(os.getenv("REALITYCHECK_WIKI_DELAY", sleep_seconds if sleep_seconds is not None else 2.0))
        self.max_external_chars = max_external_chars
        self.max_429_per_claim = int(os.getenv("REALITYCHECK_MAX_429_PER_CLAIM", str(max_429_per_claim)))
        self.user_agent = user_agent or os.getenv(
            "REALITYCHECK_USER_AGENT",
            "RealityCheck/0.7 (student research project; cache-first retrieval; contact: please-set-REALITYCHECK_USER_AGENT)",
        )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        })
        self.cache_dir = Path(cache_dir or os.getenv("REALITYCHECK_WIKI_CACHE", ".phase4_wiki_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_at = 0.0
        self._claim_429_count = 0
        self.debug_events: List[Dict[str, Any]] = []

    # ----------------------------- public API -----------------------------

    def retrieve_pages(
        self,
        queries: Iterable[str],
        limit_per_query: int = 3,
        max_pages: int = 8,
        source_url: Optional[str] = None,
        extra_text: str = "",
    ) -> List[EvidencePage]:
        self.debug_events = []
        self._claim_429_count = 0
        queries = [q for q in (queries or []) if str(q).strip()]
        pages: List[EvidencePage] = []
        seen = set()

        def add(page: Optional[EvidencePage]) -> None:
            if not page or not (page.extract or "").strip():
                return
            key = (page.source_type, page.url.lower())
            if key in seen:
                return
            seen.add(key)
            pages.append(page)

        # 0. Local seed evidence. This avoids repeated live calls for common benchmark cases.
        try:
            from local_evidence_bank import local_pages_for_queries
            local_pages = local_pages_for_queries(queries, extra_text=extra_text or "", max_pages=max_pages)
            for page in local_pages:
                add(page)
            if local_pages:
                self.debug_events.append({
                    "stage": "local_seed_lookup",
                    "result_count": len(local_pages),
                    "page_titles": [p.title for p in local_pages],
                })
        except Exception as exc:  # local seed must never crash retrieval
            self.debug_events.append({"stage": "local_seed_error", "error": repr(exc)})

        # If local seed already gives enough pages, do not hit the network.
        if len(pages) >= min(max_pages, 2):
            self.debug_events.append({
                "stage": "retrieve_pages_final",
                "queries": queries,
                "source_url_present": bool(source_url),
                "pages_returned": len(pages[:max_pages]),
                "page_titles": [p.title for p in pages[:max_pages]],
                "network_used": False,
            })
            return pages[:max_pages]

        # 1. Direct source URL, if given.
        source_page = self.fetch_source_url(source_url)
        add(source_page)

        # 2. Wikipedia search fallback. Keep this conservative.
        for query in queries[: int(os.getenv("REALITYCHECK_MAX_QUERIES_PER_CLAIM", "3"))]:
            if len(pages) >= max_pages or self._claim_429_count >= self.max_429_per_claim:
                break
            results = self.search_wikipedia(query=query, limit=min(limit_per_query, 3))
            for result in results:
                if len(pages) >= max_pages or self._claim_429_count >= self.max_429_per_claim:
                    break
                title = result.get("title")
                if not title:
                    continue
                page = self.fetch_wiki_page(title=title, project="wikipedia", retrieval_method="search")
                add(page)

            # opensearch fallback only if regular search returns nothing and no throttling happened.
            if not results and self._claim_429_count < self.max_429_per_claim:
                for title in self.opensearch_wikipedia(query=query, limit=min(limit_per_query, 3)):
                    if len(pages) >= max_pages or self._claim_429_count >= self.max_429_per_claim:
                        break
                    add(self.fetch_wiki_page(title=title, project="wikipedia", retrieval_method="opensearch"))

        self.debug_events.append({
            "stage": "retrieve_pages_final",
            "queries": queries,
            "source_url_present": bool(source_url),
            "pages_returned": len(pages[:max_pages]),
            "page_titles": [p.title for p in pages[:max_pages]],
            "network_used": True,
            "claim_429_count": self._claim_429_count,
        })
        return pages[:max_pages]

    def fetch_source_url(self, source_url: Optional[str]) -> Optional[EvidencePage]:
        if not source_url or not isinstance(source_url, str):
            return None
        source_url = source_url.strip()
        if source_url.lower() in PSEUDO_SOURCE_VALUES:
            return None

        parsed = urlparse(source_url)
        host = parsed.netloc.lower()

        if "wikipedia.org" in host or "wikiquote.org" in host:
            title = self.title_from_wiki_like_url(source_url)
            if not title:
                return None
            project = "wikiquote" if "wikiquote.org" in host else "wikipedia"
            return self.fetch_wiki_page(title=title, project=project, retrieval_method="source_url")

        return self.fetch_external_page(source_url)

    # ----------------------------- Wikipedia APIs -----------------------------

    def search_wikipedia(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "utf8": 1,
        }
        key = self._cache_key("search", "wikipedia", query, str(limit))
        cached = self._cache_get(key)
        if cached is not None:
            self.debug_events.append({"stage": "wiki_search_cache", "query": query, "result_count": len(cached)})
            return cached
        data = self._get_json(self.api_url("wikipedia"), params=params, stage="wiki_search", meta={"query": query})
        results = data.get("query", {}).get("search", []) if data else []
        self._cache_set(key, results)
        self.debug_events.append({"stage": "wiki_search", "query": query, "result_count": len(results)})
        return results

    def opensearch_wikipedia(self, query: str, limit: int = 3) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []
        params = {
            "action": "opensearch",
            "format": "json",
            "search": query,
            "limit": limit,
            "namespace": 0,
        }
        key = self._cache_key("opensearch", "wikipedia", query, str(limit))
        cached = self._cache_get(key)
        if cached is not None:
            self.debug_events.append({"stage": "wiki_opensearch_cache", "query": query, "result_count": len(cached)})
            return cached
        data = self._get_json(self.api_url("wikipedia"), params=params, stage="wiki_opensearch", meta={"query": query})
        titles = data[1] if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list) else []
        self._cache_set(key, titles)
        self.debug_events.append({"stage": "wiki_opensearch", "query": query, "result_count": len(titles)})
        return titles

    def fetch_wiki_page(self, title: str, project: str = "wikipedia", retrieval_method: str = "search") -> EvidencePage:
        title = (title or "").strip()
        if not title:
            return EvidencePage(title="", page_id=None, url="", extract="", retrieval_method=retrieval_method, source_type=project)
        key = self._cache_key("page", project, title)
        cached = self._cache_get(key)
        if cached is not None:
            page = EvidencePage(**cached)
            page.retrieval_method = page.retrieval_method or retrieval_method
            self.debug_events.append({"stage": "wiki_page_cache", "title": title, "project": project})
            return page

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "utf8": 1,
        }
        data = self._get_json(self.api_url(project), params=params, stage="wiki_fetch", meta={"title": title, "project": project})
        if not data:
            return EvidencePage(title=title, page_id=None, url=self.page_url(title, project), extract="", retrieval_method=retrieval_method, source_type=project)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return EvidencePage(title=title, page_id=None, url=self.page_url(title, project), extract="", retrieval_method=retrieval_method, source_type=project)
        first_page = next(iter(pages.values()))
        if "missing" in first_page:
            return EvidencePage(title=title, page_id=None, url=self.page_url(title, project), extract="", retrieval_method=retrieval_method, source_type=project)
        resolved_title = first_page.get("title", title)
        page = EvidencePage(
            title=resolved_title,
            page_id=first_page.get("pageid"),
            url=self.page_url(resolved_title, project),
            extract=first_page.get("extract", "") or "",
            retrieval_method=retrieval_method,
            source_type=project,
        )
        if page.extract.strip():
            self._cache_set(key, asdict(page))
        return page

    def api_url(self, project: str) -> str:
        if project == "wikiquote":
            return f"https://{self.language}.wikiquote.org/w/api.php"
        return f"https://{self.language}.wikipedia.org/w/api.php"

    @staticmethod
    def title_from_wiki_like_url(url: Optional[str]) -> Optional[str]:
        if not url or not isinstance(url, str):
            return None
        parsed = urlparse(url)
        match = re.search(r"/wiki/([^#?]+)", parsed.path)
        if not match:
            return None
        title = unquote(match.group(1)).replace("_", " ").strip()
        return title or None

    @staticmethod
    def page_url(title: str, project: str = "wikipedia") -> str:
        host = "en.wikiquote.org" if project == "wikiquote" else "en.wikipedia.org"
        return f"https://{host}/wiki/" + quote(title.replace(" ", "_"))

    # ----------------------------- External pages -----------------------------

    def fetch_external_page(self, url: str) -> EvidencePage:
        key = self._cache_key("external", url)
        cached = self._cache_get(key)
        if cached is not None:
            return EvidencePage(**cached)
        try:
            response = self._request("GET", url, allow_redirects=True)
            html = response.text or ""
            text, title = self._html_to_text(html, fallback_title=urlparse(url).netloc)
            page = EvidencePage(
                title=title,
                page_id=None,
                url=response.url or url,
                extract=text[: self.max_external_chars],
                retrieval_method="external_source",
                source_type="external",
            )
            if page.extract.strip():
                self._cache_set(key, asdict(page))
            return page
        except requests.RequestException as exc:
            self.debug_events.append({"stage": "external_fetch_error", "url": url, "error": repr(exc)})
            return EvidencePage(title=urlparse(url).netloc or url, page_id=None, url=url, extract="", retrieval_method="external_source_failed", source_type="external")

    @staticmethod
    def _html_to_text(html: str, fallback_title: str = "external source") -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        title = (soup.title.string.strip() if soup.title and soup.title.string else fallback_title)
        parts: List[str] = []
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            txt = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
            if len(txt) >= 40:
                parts.append(txt)
        text = "\n".join(parts)
        if not text:
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text, title

    # ----------------------------- request/cache internals -----------------------------

    def _get_json(self, url: str, params: Dict[str, Any], stage: str, meta: Dict[str, Any]) -> Any:
        try:
            response = self._request("GET", url, params=params)
            return response.json()
        except requests.HTTPError as exc:
            self.debug_events.append({"stage": f"{stage}_error", **meta, "error": repr(exc)})
            return None
        except (requests.RequestException, ValueError) as exc:
            self.debug_events.append({"stage": f"{stage}_error", **meta, "error": repr(exc)})
            return None

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        max_attempts = int(os.getenv("REALITYCHECK_HTTP_ATTEMPTS", "2"))
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            self._polite_sleep()
            response = self.session.request(method, url, timeout=self.timeout_seconds, **kwargs)
            if response.status_code == 429:
                self._claim_429_count += 1
                retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                wait = retry_after if retry_after is not None else min(30.0, (2 ** attempt) * self.sleep_seconds * 2)
                wait += random.uniform(0.0, 0.75)
                self.debug_events.append({
                    "stage": "http_429_backoff",
                    "url": response.url,
                    "attempt": attempt + 1,
                    "wait_seconds": round(wait, 2),
                    "retry_after": response.headers.get("Retry-After"),
                })
                time.sleep(wait)
                last_exc = requests.HTTPError(f"429 Too Many Requests for url: {response.url}", response=response)
                if self._claim_429_count >= self.max_429_per_claim:
                    raise last_exc
                continue
            try:
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                last_exc = exc
                raise
        if last_exc:
            raise last_exc
        raise requests.RequestException("request failed without response")

    def _polite_sleep(self) -> None:
        elapsed = time.time() - self._last_request_at
        target = self.sleep_seconds + random.uniform(0.0, 0.4)
        if elapsed < target:
            time.sleep(target - elapsed)
        self._last_request_at = time.time()

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    def _cache_key(self, *parts: str) -> str:
        raw = "||".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest() + ".json"

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key

    def _cache_get(self, key: str) -> Any:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _cache_set(self, key: str, value: Any) -> None:
        path = self._cache_path(key)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            tmp.replace(path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
