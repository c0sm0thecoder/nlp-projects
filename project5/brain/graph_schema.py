"""
graph_schema.py — Strawberry GraphQL schema for the knowledge graph.

Defines types and queries for accessing the Neo4j knowledge graph.
"""
from __future__ import annotations

from typing import Optional

import strawberry

from brain import knowledge_graph as kg


@strawberry.type
class Person:
    id: str
    name: str
    role: str
    department: str
    authority_score: int
    email: str = ""


@strawberry.type
class Department:
    id: str
    name: str
    description: str = ""
    head: str = ""
    members: list[str] = strawberry.field(default_factory=list)
    projects: list[str] = strawberry.field(default_factory=list)


@strawberry.type
class Project:
    id: str
    name: str
    status: str
    owner_dept: str
    tech_stack: list[str] = strawberry.field(default_factory=list)
    team_size: int = 0


@strawberry.type
class Service:
    id: str
    name: str
    status: str
    owner_team: str
    tech_stack: list[str] = strawberry.field(default_factory=list)


@strawberry.type
class Technology:
    id: str
    name: str
    category: str = ""


@strawberry.type
class Policy:
    id: str
    title: str
    category: str
    owner: str = ""
    effective_date: str = ""


@strawberry.type
class Entity:
    label: str
    id: str
    name: str
    props: strawberry.scalars.JSON


@strawberry.type
class GraphStats:
    node_counts: list[strawberry.scalars.JSON]


@strawberry.type
class Query:
    @strawberry.field
    def person(self, name: str) -> Optional[Person]:
        results = kg.find_entity_by_name(name)
        for r in results:
            if r["label"] == "Person":
                props = r["props"]
                return Person(
                    id=props.get("id", ""),
                    name=props.get("name", ""),
                    role=props.get("role", ""),
                    department=props.get("department", ""),
                    authority_score=props.get("authority_score", 0),
                    email=props.get("email", ""),
                )
        return None

    @strawberry.field
    def department(self, id: str) -> Optional[Department]:
        result = kg.get_department_with_projects(id)
        if not result:
            return None
        dept = result["department"]
        return Department(
            id=dept.get("id", ""),
            name=dept.get("name", ""),
            description=dept.get("description", ""),
            head=dept.get("head", ""),
            members=result.get("members", []),
            projects=result.get("projects", []),
        )

    @strawberry.field
    def project_dependencies(self, project_id: str) -> list[Entity]:
        deps = kg.get_project_dependencies(project_id)
        return [
            Entity(
                label=d["type"],
                id=d["id"],
                name=d["name"],
                props={},
            )
            for d in deps
        ]

    @strawberry.field
    def related_entities(self, names: list[str], hops: int = 2) -> list[Entity]:
        results = kg.query_related_entities(names, hops)
        return [
            Entity(
                label=r["label"],
                id=r["id"],
                name=r["name"] or "",
                props=r["props"],
            )
            for r in results
        ]

    @strawberry.field
    def documents_for_entities(self, entity_ids: list[str]) -> list[str]:
        return kg.get_documents_for_entities(entity_ids)

    @strawberry.field
    def search_entities(self, name: str) -> list[Entity]:
        results = kg.find_entity_by_name(name)
        return [
            Entity(
                label=r["label"],
                id=r["id"],
                name=r["name"] or "",
                props=r["props"],
            )
            for r in results
        ]

    @strawberry.field
    def graph_stats(self) -> GraphStats:
        stats = kg.get_graph_stats()
        return GraphStats(node_counts=stats.get("nodes", []))


schema = strawberry.Schema(query=Query)


def execute_query(query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query and return the result."""
    result = schema.execute_sync(query, variable_values=variables)
    if result.errors:
        raise Exception(f"GraphQL errors: {result.errors}")
    return result.data
