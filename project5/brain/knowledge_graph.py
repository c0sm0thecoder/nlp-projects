"""
knowledge_graph.py — Neo4j knowledge graph operations for Graph RAG.

Provides CRUD operations and graph queries for entities and relationships.
"""
from __future__ import annotations

from typing import Any

from core.clients import get_neo4j_driver
from core.logger import get_logger

logger = get_logger(__name__)


def create_constraints() -> None:
    """Create unique constraints on node IDs."""
    driver = get_neo4j_driver()
    constraints = [
        "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT department_id IF NOT EXISTS FOR (d:Department) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT service_id IF NOT EXISTS FOR (s:Service) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT technology_id IF NOT EXISTS FOR (t:Technology) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    ]
    with driver.session() as session:
        for cypher in constraints:
            session.run(cypher)
    logger.info("Graph constraints created.")


def clear_graph() -> None:
    """Delete all nodes and relationships."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Graph cleared.")


def upsert_person(
    id: str,
    name: str,
    role: str,
    department: str,
    authority_score: int,
    email: str = "",
) -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Person {id: $id})
            SET p.name = $name, p.role = $role, p.department = $department,
                p.authority_score = $authority_score, p.email = $email
            """,
            id=id, name=name, role=role, department=department,
            authority_score=authority_score, email=email,
        )


def upsert_department(id: str, name: str, description: str = "", head: str = "") -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (d:Department {id: $id})
            SET d.name = $name, d.description = $description, d.head = $head
            """,
            id=id, name=name, description=description, head=head,
        )


def upsert_project(
    id: str,
    name: str,
    status: str,
    owner_dept: str,
    tech_stack: list[str] | None = None,
    team_size: int = 0,
) -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Project {id: $id})
            SET p.name = $name, p.status = $status, p.owner_dept = $owner_dept,
                p.tech_stack = $tech_stack, p.team_size = $team_size
            """,
            id=id, name=name, status=status, owner_dept=owner_dept,
            tech_stack=tech_stack or [], team_size=team_size,
        )


def upsert_service(
    id: str,
    name: str,
    status: str,
    owner_team: str,
    tech_stack: list[str] | None = None,
) -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (s:Service {id: $id})
            SET s.name = $name, s.status = $status, s.owner_team = $owner_team,
                s.tech_stack = $tech_stack
            """,
            id=id, name=name, status=status, owner_team=owner_team,
            tech_stack=tech_stack or [],
        )


def upsert_technology(id: str, name: str, category: str = "") -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (t:Technology {id: $id})
            SET t.name = $name, t.category = $category
            """,
            id=id, name=name, category=category,
        )


def upsert_policy(
    id: str,
    title: str,
    category: str,
    owner: str = "",
    effective_date: str = "",
) -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Policy {id: $id})
            SET p.title = $title, p.category = $category, p.owner = $owner,
                p.effective_date = $effective_date
            """,
            id=id, title=title, category=category, owner=owner,
            effective_date=effective_date,
        )


def upsert_document(
    id: str,
    source: str,
    url: str,
    title: str,
    timestamp: str,
    namespace: str,
) -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.source = $source, d.url = $url, d.title = $title,
                d.timestamp = $timestamp, d.namespace = $namespace
            """,
            id=id, source=source, url=url, title=title,
            timestamp=timestamp, namespace=namespace,
        )


def create_relationship(
    from_label: str,
    from_id: str,
    rel_type: str,
    to_label: str,
    to_id: str,
    props: dict[str, Any] | None = None,
) -> None:
    driver = get_neo4j_driver()
    props_str = ""
    if props:
        props_items = ", ".join(f"{k}: ${k}" for k in props.keys())
        props_str = f" {{{props_items}}}"

    cypher = f"""
        MATCH (a:{from_label} {{id: $from_id}})
        MATCH (b:{to_label} {{id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        {"SET r += $props" if props else ""}
    """
    with driver.session() as session:
        session.run(cypher, from_id=from_id, to_id=to_id, props=props or {})


def query_related_entities(entity_names: list[str], hops: int = 2) -> list[dict]:
    """Find entities related to the given names within N hops."""
    if not entity_names:
        return []

    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            UNWIND $names AS name
            MATCH (start)
            WHERE toLower(start.name) CONTAINS toLower(name)
               OR toLower(start.title) CONTAINS toLower(name)
            CALL apoc.path.subgraphNodes(start, {maxLevel: $hops}) YIELD node
            RETURN DISTINCT labels(node)[0] AS label, node.id AS id, node.name AS name,
                   properties(node) AS props
            LIMIT 50
            """,
            names=entity_names, hops=hops,
        )
        return [dict(r) for r in result]


def get_documents_for_entities(entity_ids: list[str]) -> list[str]:
    """Get document IDs connected to the given entity IDs."""
    if not entity_ids:
        return []

    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            UNWIND $ids AS entity_id
            MATCH (e {id: entity_id})<-[:MENTIONS]-(d:Document)
            RETURN DISTINCT d.id AS doc_id
            """,
            ids=entity_ids,
        )
        return [r["doc_id"] for r in result]


def find_entity_by_name(name: str) -> list[dict]:
    """Search for entities by name (case-insensitive partial match)."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($name)
            RETURN labels(n)[0] AS label, n.id AS id, n.name AS name, properties(n) AS props
            LIMIT 20
            """,
            name=name,
        )
        return [dict(r) for r in result]


def get_department_with_projects(dept_id: str) -> dict:
    """Get a department with its projects and team members."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Department {id: $dept_id})
            OPTIONAL MATCH (d)<-[:WORKS_IN]-(p:Person)
            OPTIONAL MATCH (d)-[:OWNS]->(proj:Project)
            RETURN d AS department,
                   collect(DISTINCT p.name) AS members,
                   collect(DISTINCT proj.name) AS projects
            """,
            dept_id=dept_id,
        )
        record = result.single()
        if not record:
            return {}
        return {
            "department": dict(record["department"]),
            "members": record["members"],
            "projects": record["projects"],
        }


def get_project_dependencies(project_id: str) -> list[dict]:
    """Get all dependencies for a project."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {id: $project_id})-[:DEPENDS_ON]->(dep)
            RETURN labels(dep)[0] AS type, dep.id AS id, dep.name AS name
            """,
            project_id=project_id,
        )
        return [dict(r) for r in result]


def get_graph_stats() -> dict:
    """Get basic statistics about the graph."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WITH labels(n)[0] AS label, count(n) AS count
            RETURN collect({label: label, count: count}) AS node_counts
            """
        )
        record = result.single()
        return {"nodes": record["node_counts"] if record else []}


def index_document_in_graph(doc_id: str, content: str, metadata: dict) -> None:
    """
    Automatically index a document in the graph:
    1. Create Document node
    2. Link to author via AUTHORED relationship
    3. Extract entities and create MENTIONS relationships
    """
    from brain.entity_extractor import extract_entities

    driver = get_neo4j_driver()

    # Create document node
    upsert_document(
        id=doc_id,
        source=metadata.get("source", "unknown"),
        url=metadata.get("url", ""),
        title=metadata.get("page_title", content[:50]),
        timestamp=str(metadata.get("timestamp", "")),
        namespace=metadata.get("namespace", ""),
    )

    # Link to author if exists
    author_name = metadata.get("author_name", "")
    if author_name:
        with driver.session() as session:
            # Find person by name (fuzzy match)
            session.run(
                """
                MATCH (p:Person)
                WHERE toLower(p.name) = toLower($author_name)
                MATCH (d:Document {id: $doc_id})
                MERGE (p)-[:AUTHORED]->(d)
                """,
                author_name=author_name, doc_id=doc_id,
            )

    # Extract entities and create MENTIONS
    entities = extract_entities(content[:2000])

    with driver.session() as session:
        # Link to mentioned people
        for person in entities.get("people", []):
            name = person.get("name") if isinstance(person, dict) else person
            if name:
                session.run(
                    """
                    MATCH (p:Person)
                    WHERE toLower(p.name) CONTAINS toLower($name)
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:MENTIONS]->(p)
                    """,
                    name=name, doc_id=doc_id,
                )

        # Link to departments
        for dept in entities.get("departments", []):
            session.run(
                """
                MATCH (dept:Department)
                WHERE toLower(dept.name) CONTAINS toLower($name)
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:MENTIONS]->(dept)
                """,
                name=dept, doc_id=doc_id,
            )

        # Link to projects
        for proj in entities.get("projects", []):
            session.run(
                """
                MATCH (p:Project)
                WHERE toLower(p.name) CONTAINS toLower($name)
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:MENTIONS]->(p)
                """,
                name=proj, doc_id=doc_id,
            )

        # Link to technologies
        for tech in entities.get("technologies", []):
            session.run(
                """
                MATCH (t:Technology)
                WHERE toLower(t.name) CONTAINS toLower($name)
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:MENTIONS]->(t)
                """,
                name=tech, doc_id=doc_id,
            )

    logger.info("Indexed document %s in graph with entity links", doc_id[:16])
