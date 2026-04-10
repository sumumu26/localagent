from agent.registry import register

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


@register({
    "name": "web_search",
    "description": (
        "Search the web using DuckDuckGo. "
        "Returns titles, URLs, and snippets from search results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
            },
        },
        "required": ["query"],
    },
})
def web_search(query: str, max_results: int = 5) -> str:
    if not _DDGS_AVAILABLE:
        return "Error: duckduckgo-search is not installed. Run: pip install duckduckgo-search"

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"Title: {r.get('title', 'N/A')}")
        lines.append(f"URL: {r.get('href', 'N/A')}")
        lines.append(f"Snippet: {r.get('body', 'N/A')}")
        lines.append("")
    return "\n".join(lines).strip()
