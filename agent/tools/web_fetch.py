import urllib.request
import urllib.error
from agent.registry import register

DEFAULT_FETCH_BYTES = 32_768   # 32 KB
MAX_FETCH_BYTES = 4_194_304    # 4 MB hard cap

try:
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._skip_tags = {"script", "style", "head"}
            self._current_skip = 0
            self.parts = []

        def handle_starttag(self, tag, attrs):
            if tag in self._skip_tags:
                self._current_skip += 1

        def handle_endtag(self, tag):
            if tag in self._skip_tags and self._current_skip > 0:
                self._current_skip -= 1

        def handle_data(self, data):
            if self._current_skip == 0:
                text = data.strip()
                if text:
                    self.parts.append(text)

    def _extract_text(html: str) -> str:
        parser = _TextExtractor()
        parser.feed(html)
        return "\n".join(parser.parts)

    _HTML_PARSER_AVAILABLE = True
except ImportError:
    _HTML_PARSER_AVAILABLE = False


@register({
    "name": "web_fetch",
    "description": (
        "Fetch the content of a URL and return its text. "
        "Use this when you already have a URL and want to read the page. "
        "HTML pages are converted to plain text. "
        f"Default limit is {DEFAULT_FETCH_BYTES // 1024} KB; specify max_bytes (up to {MAX_FETCH_BYTES // 1024 // 1024} MB) for larger pages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
            "max_bytes": {
                "type": "integer",
                "description": (
                    f"Maximum bytes to read (default {DEFAULT_FETCH_BYTES // 1024} KB, "
                    f"max {MAX_FETCH_BYTES // 1024 // 1024} MB). "
                    "Increase for large pages."
                ),
            },
        },
        "required": ["url"],
    },
})
def web_fetch(url: str, max_bytes: int = DEFAULT_FETCH_BYTES) -> str:
    limit = min(max(1, max_bytes), MAX_FETCH_BYTES)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; arko-agent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get_content_type() or ""
            raw = resp.read(limit)
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Failed to fetch URL: {e.reason}"
    except Exception as e:
        return f"Error fetching URL: {e}"

    text = raw.decode("utf-8", errors="replace")

    if "html" in content_type and _HTML_PARSER_AVAILABLE:
        text = _extract_text(text)

    if len(raw) >= limit:
        text += f"\n\n[... truncated at {limit:,} bytes ...]"

    return text
