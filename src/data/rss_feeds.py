"""
src/data/rss_feeds.py  —  Simple Economic RSS Feed Fetcher
Fetches live economic/financial news headlines from public RSS feeds.
Used by the Streamlit dashboard sidebar and injected into AI context.
"""

import feedparser

# ── Feed sources ──────────────────────────────────────────────────────────────
RSS_SOURCES = {
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "Reuters Markets":  "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo Finance":    "https://finance.yahoo.com/rss/topstories",
    "CNBC Economy":     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
}


def fetch_rss_headlines(max_per_source: int = 5) -> list:
    """
    Fetch recent headlines from all RSS sources.

    Returns
    -------
    list of dict with keys: source, title, link, published, summary
    """
    items = []
    for source, url in RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_source]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                items.append({
                    "source":    source,
                    "title":     title,
                    "link":      entry.get("link", "#"),
                    "published": entry.get("published", ""),
                    "summary":   entry.get("summary", "")[:250].strip(),
                })
        except Exception:
            # Silently skip unavailable feeds — dashboard must not crash
            pass
    return items


def headlines_to_context_text(headlines: list) -> str:
    """
    Format headlines as a plain-text block for injection into the AI system prompt.
    """
    if not headlines:
        return ""
    lines = ["", "=== LIVE ECONOMIC NEWS HEADLINES ===",
             "Use these to answer questions about recent economic events or news."]
    for h in headlines:
        lines.append(f"[{h['source']}] {h['title']}")
        if h["summary"]:
            lines.append(f"  {h['summary']}")
    lines.append("")
    return "\n".join(lines)
