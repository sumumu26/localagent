import urllib.request
import urllib.error
from agent.registry import register

MAX_FETCH_BYTES = 32_768  # 32 KB hard cap to protect context window

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
        "Content is truncated at 32 KB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    },
})
def web_fetch(url: str) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; hakobune-agent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get_content_type() or ""
            raw = resp.read(MAX_FETCH_BYTES)
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Failed to fetch URL: {e.reason}"
    except Exception as e:
        return f"Error fetching URL: {e}"

    text = raw.decode("utf-8", errors="replace")

    if "html" in content_type and _HTML_PARSER_AVAILABLE:
        text = _extract_text(text)

    if len(raw) >= MAX_FETCH_BYTES:
        text += f"\n\n[... truncated at {MAX_FETCH_BYTES} bytes ...]"

    return text
