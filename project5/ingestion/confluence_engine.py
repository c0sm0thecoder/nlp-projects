from __future__ import annotations

import re

from atlassian import Confluence
from langchain_core.documents import Document

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_CONFLUENCE_AUTHORITY = 5


def fetch_confluence_documents(last_modified_date: str | None = None) -> list[Document]:
    """Fetch pages from configured Confluence spaces via CQL, optionally filtered by date."""
    settings = get_settings()
    cf = Confluence(
        url=settings.confluence_url,
        username=settings.confluence_user,
        password=settings.confluence_api_token,
        cloud=True,
    )

    spaces_in = ", ".join(f'"{s}"' for s in settings.confluence_space_list)
    cql = f"space IN ({spaces_in}) AND type = page"
    if last_modified_date:
        cql += f' AND lastModified > "{last_modified_date}"'

    logger.info("Confluence CQL: %s", cql)
    documents: list[Document] = []
    start = 0
    limit = 50

    while True:
        results = cf.cql(
            cql, start=start, limit=limit, expand="body.storage,version,space"
        )
        page_list = results.get("results", [])
        if not page_list:
            break

        for page in page_list:
            raw_body = page.get("body", {}).get("storage", {}).get("value", "")
            clean_text = _strip_html(raw_body)
            if not clean_text.strip():
                continue

            title = page.get("title", "Untitled")
            space_key = page.get("space", {}).get("key", "UNKNOWN")
            page_id = page.get("id", "")
            version = page.get("version", {})
            last_modified = version.get("when", "")
            author_display = version.get("by", {}).get("displayName", "Unknown Author")

            documents.append(
                Document(
                    page_content=f"{title}\n\n{clean_text}",
                    metadata={
                        "source": "confluence",
                        "url": f"{settings.confluence_url}/pages/{page_id}",
                        "author_role": f"Confluence Author ({space_key})",
                        "authority_score": _CONFLUENCE_AUTHORITY,
                        "timestamp": last_modified,
                        "namespace": "confluence",
                        "space": space_key,
                        "page_title": title,
                        "page_id": page_id,
                        "last_modified_by": author_display,
                    },
                )
            )

        if len(page_list) < limit:
            break
        start += limit

    logger.info("Confluence ingestion complete: %d documents.", len(documents))
    return documents


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    for entity, replacement in [
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"'),
    ]:
        text = text.replace(entity, replacement)
    return re.sub(r"\s{2,}", " ", text).strip()
