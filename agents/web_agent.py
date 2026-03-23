"""
Web agent — search and fetch via DuckDuckGo + requests.

All tools are non-destructive: they only read from the web.
No local state is modified; no files are written.

Dispatch: action type QUERY → routed by params["query_type"]:
  "search" → web_search(query)
  "fetch"  → web_fetch(url)
"""
from __future__ import annotations

import re
from typing import Optional

from agents.base_agent import BaseAgent
from db.audit import log_action_started, log_action_completed
from errors import LeavesError, LeavesErrorCode


class WebAgent(BaseAgent):
    AGENT_TYPE = "web"

    def execute_action(self, action: dict) -> dict:
        """
        Dispatch QUERY actions by params["query_type"]:
          "search"  → web_search(query)
          "fetch"   → web_fetch(url)
        """
        action_type = action.get("type", "").upper()
        action_id = action.get("action_id", "unknown")
        params = action.get("params", {})

        if action_type != "QUERY":
            raise LeavesError(
                LeavesErrorCode.AGENT_NOT_AVAILABLE,
                detail=f"WebAgent only supports QUERY, got {action_type}",
            )

        query_type = params.get("query_type", "search")

        if query_type == "search":
            query = params.get("query", params.get("path", ""))
            if not query:
                raise LeavesError(
                    LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                    detail="No query provided for web search",
                )
            row_id = self._audit_start(action_id, "QUERY", params)
            try:
                result = self._web_search(query)
                self._audit_end(row_id, result)
                return result
            except LeavesError:
                raise
            except Exception as e:
                err = LeavesError(LeavesErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
                self._audit_end(row_id, error=err)
                raise err

        elif query_type == "fetch":
            url = params.get("url", params.get("path", ""))
            if not url:
                raise LeavesError(
                    LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                    detail="No URL provided for web fetch",
                )
            row_id = self._audit_start(action_id, "QUERY", params)
            try:
                result = self._web_fetch(url)
                self._audit_end(row_id, result)
                return result
            except LeavesError:
                raise
            except Exception as e:
                err = LeavesError(LeavesErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
                self._audit_end(row_id, error=err)
                raise err

        else:
            # Unknown query_type — default to search
            row_id = self._audit_start(action_id, "QUERY", params)
            try:
                result = self._web_search(
                    params.get("query", params.get("path", str(params))))
                self._audit_end(row_id, result)
                return result
            except LeavesError:
                raise
            except Exception as e:
                err = LeavesError(LeavesErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
                self._audit_end(row_id, error=err)
                raise err

    def _web_search(self, query: str) -> dict:
        """
        Search the web using DuckDuckGo and return top results.
        Returns structured results, NOT raw HTML.
        """
        try:
            from ddgs import DDGS
        except ImportError:
            return {"error": "ddgs not installed (pip install ddgs)",
                    "query": query, "results": []}

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:300],
                    })
            return {
                "query": query,
                "result_count": len(results),
                "results": results,
            }
        except Exception as e:
            return {"error": str(e), "query": query, "results": []}

    def _web_fetch(self, url: str) -> dict:
        """
        Fetch a URL and extract readable text content.
        Strips HTML, returns plain text.
        """
        import requests
        from bs4 import BeautifulSoup

        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script, style, nav, footer elements
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside"]):
                tag.decompose()

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string or ""

            # Extract main text
            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Truncate to 3000 chars — enough for context, not overwhelming
            if len(text) > 3000:
                text = text[:3000] + "\n[truncated]"

            return {
                "url": url,
                "title": title.strip(),
                "content": text,
                "content_length": len(text),
            }

        except requests.exceptions.Timeout:
            return {"error": "Request timed out", "url": url}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP {e.response.status_code}", "url": url}
        except Exception as e:
            return {"error": str(e), "url": url}

    # ------------------------------------------------------------------
    # Audit helpers (same pattern as SystemAgent)
    # ------------------------------------------------------------------

    def _audit_start(self, action_id: str, action_type: str, params: dict) -> Optional[int]:
        try:
            return log_action_started(
                self._db,
                intent_id=self.intent_id,
                action_id=action_id,
                action_type=action_type,
                agent=self.AGENT_TYPE,
                params=params,
            )
        except Exception:
            return None

    def _audit_end(
        self,
        row_id: Optional[int],
        result: Optional[dict] = None,
        error: Optional[LeavesError] = None,
    ) -> None:
        if row_id is None:
            return
        try:
            log_action_completed(
                self._db,
                row_id=row_id,
                result=result,
                error_code=error.code.value if error else None,
                error_detail=error.detail if error else None,
            )
        except Exception:
            pass
