"""
entity_extractor.py — LLM-based entity extraction using Gemini.

Extracts structured entities (people, departments, projects, etc.) from text.
"""
from __future__ import annotations

import json
import re

from core.clients import get_llm
from core.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """\
Extract all named entities from the following text. Return a JSON object with these keys:
- people: list of objects with "name" and "role" (if mentioned)
- departments: list of department names
- projects: list of project names
- services: list of service names
- technologies: list of technology names
- policies: list of policy names or topics

Only extract entities that are explicitly mentioned. If a category has no entities, use an empty list.
Return ONLY valid JSON, no markdown formatting.

Text:
{text}

JSON:"""


def extract_entities(text: str) -> dict:
    """Extract entities from text using Gemini."""
    if not text or len(text.strip()) < 10:
        return _empty_result()

    llm = get_llm()
    prompt = EXTRACTION_PROMPT.format(text=text[:4000])

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return _parse_json_response(content)
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
        return _empty_result()


def extract_entities_from_question(question: str) -> list[str]:
    """Extract entity names from a user question for graph lookup."""
    entities = extract_entities(question)
    names = []

    for person in entities.get("people", []):
        if isinstance(person, dict) and person.get("name"):
            names.append(person["name"])
        elif isinstance(person, str):
            names.append(person)

    for key in ["departments", "projects", "services", "technologies", "policies"]:
        for item in entities.get(key, []):
            if isinstance(item, str) and item:
                names.append(item)

    return names


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        result = json.loads(content)
        return _validate_result(result)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return _validate_result(result)
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse entity extraction JSON")
        return _empty_result()


def _validate_result(result: dict) -> dict:
    """Ensure result has all expected keys with correct types."""
    validated = _empty_result()
    for key in validated:
        if key in result and isinstance(result[key], list):
            validated[key] = result[key]
    return validated


def _empty_result() -> dict:
    return {
        "people": [],
        "departments": [],
        "projects": [],
        "services": [],
        "technologies": [],
        "policies": [],
    }
