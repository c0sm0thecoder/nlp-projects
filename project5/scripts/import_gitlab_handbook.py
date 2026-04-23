"""
import_gitlab_handbook.py — Scrape GitLab handbook and populate Confluence + Pinecone.

Fetches key sections: HR, Engineering, Security, Legal, Finance.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlassian import Confluence
from brain.vector_store import upsert_documents
from core.config import get_settings
from core.logger import get_logger

logger = get_logger("gitlab_import")

BASE_URL = "https://handbook.gitlab.com"

# Key sections to scrape (focused on policies/regulations)
SECTIONS = [
    "/handbook/people-group/",  # HR policies
    "/handbook/people-group/paid-time-off/",
    "/handbook/people-group/benefits/",
    "/handbook/people-group/employment-solutions/",
    "/handbook/security/",  # Security policies
    "/handbook/security/security-assurance/",
    "/handbook/legal/",  # Legal/compliance
    "/handbook/legal/privacy/",
    "/handbook/finance/",  # Finance policies
    "/handbook/finance/expenses/",
    "/handbook/engineering/",  # Engineering guidelines
    "/handbook/engineering/development/",
    "/handbook/engineering/infrastructure/",
    "/handbook/engineering/workflow/",
    "/handbook/product/",  # Product
    "/handbook/values/",  # Company values
    "/handbook/communication/",  # Communication guidelines
    "/handbook/tools-and-tips/",  # Tools
]

MAX_PAGES = 150  # Limit total pages


def fetch_page(url: str) -> tuple[str, str] | None:
    """Fetch a page and extract title + content."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Get title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        # Get main content
        content_div = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
        if not content_div:
            return None

        # Remove scripts, styles, nav
        for tag in content_div.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get text
        text = content_div.get_text(separator="\n", strip=True)

        # Clean up
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        if len(text) < 200:  # Skip very short pages
            return None

        return title, text[:8000]  # Limit content length

    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


def get_section_links(section_url: str, visited: set) -> list[str]:
    """Get links from a section page."""
    links = []
    try:
        resp = requests.get(section_url, timeout=10)
        if resp.status_code != 200:
            return links

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/handbook/") and href not in visited:
                full_url = urljoin(BASE_URL, href)
                if full_url not in visited:
                    links.append(full_url)
                    visited.add(full_url)

                    if len(visited) >= MAX_PAGES:
                        break

    except Exception as e:
        logger.warning("Failed to get links from %s: %s", section_url, e)

    return links[:20]  # Limit links per section


def create_confluence_page(cf: Confluence, space: str, title: str, content: str) -> str | None:
    """Create a Confluence page."""
    # Clean title for Confluence
    safe_title = f"GitLab: {title[:100]}"

    # Check if exists
    existing = cf.get_page_by_title(space, safe_title)
    if existing:
        logger.info("Page '%s' already exists, skipping.", safe_title)
        return existing["id"]

    # Create page with simple formatting
    body = f"<p><em>Source: GitLab Handbook</em></p><hr/>"

    # Convert text to HTML paragraphs
    paragraphs = content.split("\n\n")
    for p in paragraphs[:50]:  # Limit paragraphs
        p = p.strip()
        if p:
            # Escape HTML
            p = p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            body += f"<p>{p}</p>"

    try:
        result = cf.create_page(space, safe_title, body, parent_id=None)
        logger.info("Created page: %s", safe_title)
        return result["id"]
    except Exception as e:
        logger.error("Failed to create page '%s': %s", safe_title, e)
        return None


def main():
    settings = get_settings()

    # Connect to Confluence
    cf = Confluence(
        url=settings.confluence_url,
        username=settings.confluence_user,
        password=settings.confluence_api_token,
        cloud=True,
    )

    space = settings.confluence_space_list[0] if settings.confluence_space_list else "WIKI"

    logger.info("=== GitLab Handbook Import ===")
    logger.info("Target Confluence space: %s", space)

    visited = set()
    pages_fetched = []

    # Fetch sections
    for section in SECTIONS:
        if len(visited) >= MAX_PAGES:
            break

        section_url = urljoin(BASE_URL, section)
        logger.info("Fetching section: %s", section)

        # Fetch main section page
        result = fetch_page(section_url)
        if result:
            title, content = result
            pages_fetched.append((section_url, title, content))
            visited.add(section_url)

        # Get sub-pages
        links = get_section_links(section_url, visited)
        for link in links:
            if len(visited) >= MAX_PAGES:
                break

            result = fetch_page(link)
            if result:
                title, content = result
                pages_fetched.append((link, title, content))
                visited.add(link)

            time.sleep(0.3)  # Be nice to GitLab

    logger.info("Fetched %d pages from GitLab handbook", len(pages_fetched))

    # Push to Confluence
    logger.info("--- Pushing to Confluence ---")
    confluence_docs = []

    for url, title, content in pages_fetched:
        page_id = create_confluence_page(cf, space, title, content)

        if page_id:
            doc = Document(
                page_content=f"{title}\n\n{content}",
                metadata={
                    "source": "confluence",
                    "url": url,
                    "page_title": f"GitLab: {title}",
                    "author_role": "GitLab Handbook",
                    "authority_score": 8,  # High authority - official docs
                    "namespace": "confluence",
                    "space": space,
                }
            )
            confluence_docs.append(doc)

        time.sleep(0.2)  # Rate limit

    # Index in Pinecone
    if confluence_docs:
        logger.info("--- Indexing in Pinecone ---")
        upsert_documents(confluence_docs, namespace="confluence")

    logger.info("=== Import complete: %d pages ===", len(confluence_docs))


if __name__ == "__main__":
    main()
