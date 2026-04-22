"""
build_graph.py — Build the Neo4j knowledge graph from corporate data.

Creates nodes for people, departments, projects, services, technologies,
and establishes relationships between them.

Run from inside project5/:  python scripts/build_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import knowledge_graph as kg
from core.logger import get_logger

logger = get_logger("build_graph")


# ══════════════════════════════════════════════════════════════════════════════
# CORPORATE DATA DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

DEPARTMENTS = [
    {
        "id": "dept_engineering",
        "name": "Engineering",
        "description": "Product development, architecture, and technical infrastructure",
        "head": "Michael Torres",
    },
    {
        "id": "dept_product",
        "name": "Product",
        "description": "Roadmap, requirements, UX research and design",
        "head": "Lisa Nguyen",
    },
    {
        "id": "dept_hr",
        "name": "Human Resources",
        "description": "People, policies, culture, and talent management",
        "head": "Sarah Mitchell",
    },
    {
        "id": "dept_sales",
        "name": "Sales",
        "description": "Revenue, customer acquisition, and account management",
        "head": "James Wilson",
    },
    {
        "id": "dept_marketing",
        "name": "Marketing",
        "description": "Brand, content, demand generation, and communications",
        "head": "Amanda Lee",
    },
    {
        "id": "dept_finance",
        "name": "Finance",
        "description": "Budget, accounting, reporting, and financial planning",
        "head": "Robert Chen",
    },
    {
        "id": "dept_legal",
        "name": "Legal and Compliance",
        "description": "Contracts, compliance, intellectual property, and security",
        "head": "Diana Park",
    },
    {
        "id": "dept_it",
        "name": "IT Operations",
        "description": "Infrastructure, security, support, and enterprise applications",
        "head": "Ryan Patel",
    },
]

PEOPLE = [
    # Leadership
    {"id": "person_jennifer_wang", "name": "Jennifer Wang", "role": "CTO", "department": "Engineering", "authority_score": 10, "email": "jennifer.wang@athena-tech.com"},
    {"id": "person_michael_torres", "name": "Michael Torres", "role": "VP Engineering", "department": "Engineering", "authority_score": 10, "email": "michael.torres@athena-tech.com"},
    # Engineering
    {"id": "person_alex_chen", "name": "Alex Chen", "role": "Lead Architect", "department": "Engineering", "authority_score": 10, "email": "alex.chen@athena-tech.com"},
    {"id": "person_emily_johnson", "name": "Emily Johnson", "role": "Senior Engineer", "department": "Engineering", "authority_score": 7, "email": "emily.johnson@athena-tech.com"},
    {"id": "person_david_kim", "name": "David Kim", "role": "Senior Engineer", "department": "Engineering", "authority_score": 7, "email": "david.kim@athena-tech.com"},
    {"id": "person_ryan_patel", "name": "Ryan Patel", "role": "DevOps Engineer", "department": "Engineering", "authority_score": 7, "email": "ryan.patel@athena-tech.com"},
    {"id": "person_chris_lee", "name": "Chris Lee", "role": "Software Engineer", "department": "Engineering", "authority_score": 5, "email": "chris.lee@athena-tech.com"},
    {"id": "person_jordan_kim", "name": "Jordan Kim", "role": "Junior Developer", "department": "Engineering", "authority_score": 3, "email": "jordan.kim@athena-tech.com"},
    {"id": "person_taylor_smith", "name": "Taylor Smith", "role": "Junior Developer", "department": "Engineering", "authority_score": 3, "email": "taylor.smith@athena-tech.com"},
    {"id": "person_amanda_garcia", "name": "Amanda Garcia", "role": "Security Engineer", "department": "Engineering", "authority_score": 7, "email": "amanda.garcia@athena-tech.com"},
    # Product
    {"id": "person_lisa_nguyen", "name": "Lisa Nguyen", "role": "Director of Product", "department": "Product", "authority_score": 10, "email": "lisa.nguyen@athena-tech.com"},
    {"id": "person_marcus_chen", "name": "Marcus Chen", "role": "Product Manager", "department": "Product", "authority_score": 7, "email": "marcus.chen@athena-tech.com"},
    {"id": "person_sarah_kim", "name": "Sarah Kim", "role": "Product Manager", "department": "Product", "authority_score": 7, "email": "sarah.kim@athena-tech.com"},
    {"id": "person_rachel_park", "name": "Rachel Park", "role": "UX Lead", "department": "Product", "authority_score": 7, "email": "rachel.park@athena-tech.com"},
    # HR
    {"id": "person_sarah_mitchell", "name": "Sarah Mitchell", "role": "HR Lead", "department": "Human Resources", "authority_score": 10, "email": "sarah.mitchell@athena-tech.com"},
    # Sales
    {"id": "person_james_wilson", "name": "James Wilson", "role": "VP Sales", "department": "Sales", "authority_score": 10, "email": "james.wilson@athena-tech.com"},
    # Marketing
    {"id": "person_amanda_lee", "name": "Amanda Lee", "role": "Director of Marketing", "department": "Marketing", "authority_score": 10, "email": "amanda.lee@athena-tech.com"},
    # Finance
    {"id": "person_robert_chen", "name": "Robert Chen", "role": "CFO", "department": "Finance", "authority_score": 10, "email": "robert.chen@athena-tech.com"},
    # Legal
    {"id": "person_diana_park", "name": "Diana Park", "role": "General Counsel", "department": "Legal and Compliance", "authority_score": 10, "email": "diana.park@athena-tech.com"},
]

PROJECTS = [
    {
        "id": "proj_athena_core",
        "name": "Athena Core",
        "status": "Active",
        "owner_dept": "Engineering",
        "tech_stack": ["Python", "FastAPI", "LangChain", "Neo4j", "Pinecone"],
        "team_size": 5,
    },
    {
        "id": "proj_auth_service",
        "name": "Auth Service",
        "status": "Active",
        "owner_dept": "Engineering",
        "tech_stack": ["Go", "PostgreSQL", "Redis", "Okta"],
        "team_size": 3,
    },
    {
        "id": "proj_knowledge_service",
        "name": "Knowledge Service",
        "status": "Active",
        "owner_dept": "Engineering",
        "tech_stack": ["Python", "FastAPI", "Pinecone", "Gemini"],
        "team_size": 3,
    },
    {
        "id": "proj_customer_portal",
        "name": "Customer Portal",
        "status": "Active",
        "owner_dept": "Engineering",
        "tech_stack": ["Next.js", "TypeScript", "React", "Vercel"],
        "team_size": 4,
    },
    {
        "id": "proj_mobile_app_v2",
        "name": "Mobile App v2",
        "status": "Planning",
        "owner_dept": "Engineering",
        "tech_stack": ["React Native", "TypeScript", "Expo"],
        "team_size": 2,
    },
    {
        "id": "proj_data_pipeline",
        "name": "Data Pipeline",
        "status": "Active",
        "owner_dept": "Engineering",
        "tech_stack": ["Python", "Kafka", "Airflow", "Snowflake", "dbt"],
        "team_size": 3,
    },
    {
        "id": "proj_api_gateway",
        "name": "API Gateway",
        "status": "Active",
        "owner_dept": "Engineering",
        "tech_stack": ["Kong", "Lua", "Redis"],
        "team_size": 2,
    },
    {
        "id": "proj_billing_system",
        "name": "Billing System",
        "status": "Active",
        "owner_dept": "Finance",
        "tech_stack": ["Python", "FastAPI", "Stripe", "PostgreSQL"],
        "team_size": 2,
    },
    {
        "id": "proj_hr_portal",
        "name": "HR Portal",
        "status": "Active",
        "owner_dept": "Human Resources",
        "tech_stack": ["React", "Node.js", "Workday API"],
        "team_size": 2,
    },
    {
        "id": "proj_marketing_site",
        "name": "Marketing Site",
        "status": "Active",
        "owner_dept": "Marketing",
        "tech_stack": ["Next.js", "Contentful", "Vercel"],
        "team_size": 2,
    },
    {
        "id": "proj_sales_crm_integration",
        "name": "Sales CRM Integration",
        "status": "Planning",
        "owner_dept": "Sales",
        "tech_stack": ["Python", "Salesforce API", "Kafka"],
        "team_size": 1,
    },
    {
        "id": "proj_compliance_dashboard",
        "name": "Compliance Dashboard",
        "status": "Active",
        "owner_dept": "Legal and Compliance",
        "tech_stack": ["React", "Python", "PostgreSQL"],
        "team_size": 2,
    },
]

SERVICES = [
    {"id": "svc_auth", "name": "Auth Service", "status": "Active", "owner_team": "Core Services", "tech_stack": ["Go", "PostgreSQL"]},
    {"id": "svc_knowledge", "name": "Knowledge Service", "status": "Active", "owner_team": "Knowledge Team", "tech_stack": ["Python", "Pinecone"]},
    {"id": "svc_api_gateway", "name": "API Gateway", "status": "Active", "owner_team": "Platform", "tech_stack": ["Kong", "Redis"]},
    {"id": "svc_notification", "name": "Notification Service", "status": "Active", "owner_team": "Platform", "tech_stack": ["Node.js", "SQS"]},
    {"id": "svc_search", "name": "Search Service", "status": "Active", "owner_team": "Platform", "tech_stack": ["Elasticsearch"]},
    {"id": "svc_ingestion", "name": "Ingestion Service", "status": "Active", "owner_team": "Knowledge Team", "tech_stack": ["Python", "Redis"]},
]

TECHNOLOGIES = [
    {"id": "tech_python", "name": "Python", "category": "Language"},
    {"id": "tech_go", "name": "Go", "category": "Language"},
    {"id": "tech_typescript", "name": "TypeScript", "category": "Language"},
    {"id": "tech_react", "name": "React", "category": "Framework"},
    {"id": "tech_nextjs", "name": "Next.js", "category": "Framework"},
    {"id": "tech_fastapi", "name": "FastAPI", "category": "Framework"},
    {"id": "tech_langchain", "name": "LangChain", "category": "Framework"},
    {"id": "tech_neo4j", "name": "Neo4j", "category": "Database"},
    {"id": "tech_postgresql", "name": "PostgreSQL", "category": "Database"},
    {"id": "tech_redis", "name": "Redis", "category": "Database"},
    {"id": "tech_pinecone", "name": "Pinecone", "category": "Database"},
    {"id": "tech_kafka", "name": "Kafka", "category": "Infrastructure"},
    {"id": "tech_kubernetes", "name": "Kubernetes", "category": "Infrastructure"},
    {"id": "tech_aws", "name": "AWS", "category": "Cloud"},
    {"id": "tech_gemini", "name": "Gemini", "category": "AI"},
    {"id": "tech_stripe", "name": "Stripe", "category": "Integration"},
    {"id": "tech_okta", "name": "Okta", "category": "Integration"},
    {"id": "tech_salesforce", "name": "Salesforce", "category": "Integration"},
    {"id": "tech_kong", "name": "Kong", "category": "Infrastructure"},
    {"id": "tech_airflow", "name": "Airflow", "category": "Infrastructure"},
    {"id": "tech_snowflake", "name": "Snowflake", "category": "Database"},
]

POLICIES = [
    {"id": "policy_pto", "title": "PTO Policy", "category": "HR", "owner": "Sarah Mitchell", "effective_date": "2024-01-01"},
    {"id": "policy_remote_work", "title": "Remote Work Policy", "category": "HR", "owner": "Sarah Mitchell", "effective_date": "2024-03-01"},
    {"id": "policy_expense", "title": "Expense Reimbursement Policy", "category": "Finance", "owner": "Robert Chen", "effective_date": "2024-01-01"},
    {"id": "policy_security", "title": "Information Security Policy", "category": "Security", "owner": "Diana Park", "effective_date": "2024-01-01"},
    {"id": "policy_code_review", "title": "Code Review Guidelines", "category": "Engineering", "owner": "Alex Chen", "effective_date": "2024-01-01"},
    {"id": "policy_oncall", "title": "On-Call Procedures", "category": "Engineering", "owner": "Ryan Patel", "effective_date": "2024-01-01"},
    {"id": "policy_incident", "title": "Incident Response Policy", "category": "Engineering", "owner": "Ryan Patel", "effective_date": "2024-01-01"},
]


# ══════════════════════════════════════════════════════════════════════════════
# RELATIONSHIP DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

DEPARTMENT_DEPENDENCIES = [
    ("dept_engineering", "dept_product"),
    ("dept_engineering", "dept_it"),
    ("dept_engineering", "dept_legal"),
    ("dept_product", "dept_engineering"),
    ("dept_product", "dept_sales"),
    ("dept_product", "dept_marketing"),
    ("dept_sales", "dept_product"),
    ("dept_sales", "dept_legal"),
    ("dept_sales", "dept_marketing"),
    ("dept_marketing", "dept_product"),
    ("dept_marketing", "dept_engineering"),
    ("dept_hr", "dept_legal"),
    ("dept_hr", "dept_finance"),
    ("dept_it", "dept_engineering"),
    ("dept_it", "dept_legal"),
    ("dept_legal", "dept_engineering"),
]

PROJECT_DEPENDENCIES = [
    ("proj_athena_core", "proj_auth_service"),
    ("proj_athena_core", "proj_knowledge_service"),
    ("proj_athena_core", "proj_api_gateway"),
    ("proj_customer_portal", "proj_auth_service"),
    ("proj_customer_portal", "proj_api_gateway"),
    ("proj_customer_portal", "proj_billing_system"),
    ("proj_mobile_app_v2", "proj_auth_service"),
    ("proj_mobile_app_v2", "proj_api_gateway"),
    ("proj_data_pipeline", "proj_billing_system"),
    ("proj_hr_portal", "proj_auth_service"),
    ("proj_sales_crm_integration", "proj_data_pipeline"),
    ("proj_compliance_dashboard", "proj_data_pipeline"),
    ("proj_compliance_dashboard", "proj_auth_service"),
]

PERSON_LEADS = [
    ("person_jennifer_wang", "dept_engineering", "Department"),
    ("person_michael_torres", "dept_engineering", "Department"),
    ("person_alex_chen", "proj_auth_service", "Project"),
    ("person_alex_chen", "proj_api_gateway", "Project"),
    ("person_emily_johnson", "proj_athena_core", "Project"),
    ("person_emily_johnson", "proj_knowledge_service", "Project"),
    ("person_david_kim", "proj_customer_portal", "Project"),
    ("person_david_kim", "proj_mobile_app_v2", "Project"),
    ("person_ryan_patel", "proj_data_pipeline", "Project"),
    ("person_lisa_nguyen", "dept_product", "Department"),
    ("person_sarah_mitchell", "dept_hr", "Department"),
    ("person_james_wilson", "dept_sales", "Department"),
    ("person_amanda_lee", "dept_marketing", "Department"),
    ("person_robert_chen", "dept_finance", "Department"),
    ("person_diana_park", "dept_legal", "Department"),
    ("person_ryan_patel", "dept_it", "Department"),
]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_nodes() -> None:
    """Create all graph nodes."""
    logger.info("Creating department nodes...")
    for dept in DEPARTMENTS:
        kg.upsert_department(
            id=dept["id"],
            name=dept["name"],
            description=dept.get("description", ""),
            head=dept.get("head", ""),
        )

    logger.info("Creating person nodes...")
    for person in PEOPLE:
        kg.upsert_person(
            id=person["id"],
            name=person["name"],
            role=person["role"],
            department=person["department"],
            authority_score=person["authority_score"],
            email=person.get("email", ""),
        )

    logger.info("Creating project nodes...")
    for proj in PROJECTS:
        kg.upsert_project(
            id=proj["id"],
            name=proj["name"],
            status=proj["status"],
            owner_dept=proj["owner_dept"],
            tech_stack=proj.get("tech_stack", []),
            team_size=proj.get("team_size", 0),
        )

    logger.info("Creating service nodes...")
    for svc in SERVICES:
        kg.upsert_service(
            id=svc["id"],
            name=svc["name"],
            status=svc["status"],
            owner_team=svc["owner_team"],
            tech_stack=svc.get("tech_stack", []),
        )

    logger.info("Creating technology nodes...")
    for tech in TECHNOLOGIES:
        kg.upsert_technology(
            id=tech["id"],
            name=tech["name"],
            category=tech.get("category", ""),
        )

    logger.info("Creating policy nodes...")
    for policy in POLICIES:
        kg.upsert_policy(
            id=policy["id"],
            title=policy["title"],
            category=policy["category"],
            owner=policy.get("owner", ""),
            effective_date=policy.get("effective_date", ""),
        )


def build_relationships() -> None:
    """Create all graph relationships."""
    logger.info("Creating person -> department relationships...")
    dept_map = {d["name"]: d["id"] for d in DEPARTMENTS}
    for person in PEOPLE:
        dept_id = dept_map.get(person["department"])
        if dept_id:
            kg.create_relationship("Person", person["id"], "WORKS_IN", "Department", dept_id)

    logger.info("Creating person -> leads relationships...")
    for person_id, target_id, target_label in PERSON_LEADS:
        kg.create_relationship("Person", person_id, "LEADS", target_label, target_id)

    logger.info("Creating department -> owns project relationships...")
    for proj in PROJECTS:
        dept_id = dept_map.get(proj["owner_dept"])
        if dept_id:
            kg.create_relationship("Department", dept_id, "OWNS", "Project", proj["id"])

    logger.info("Creating department dependencies...")
    for from_dept, to_dept in DEPARTMENT_DEPENDENCIES:
        kg.create_relationship("Department", from_dept, "DEPENDS_ON", "Department", to_dept)

    logger.info("Creating project dependencies...")
    for from_proj, to_proj in PROJECT_DEPENDENCIES:
        kg.create_relationship("Project", from_proj, "DEPENDS_ON", "Project", to_proj)

    logger.info("Creating project -> technology relationships...")
    tech_map = {t["name"]: t["id"] for t in TECHNOLOGIES}
    for proj in PROJECTS:
        for tech_name in proj.get("tech_stack", []):
            tech_id = tech_map.get(tech_name)
            if tech_id:
                kg.create_relationship("Project", proj["id"], "USES", "Technology", tech_id)

    logger.info("Creating service -> technology relationships...")
    for svc in SERVICES:
        for tech_name in svc.get("tech_stack", []):
            tech_id = tech_map.get(tech_name)
            if tech_id:
                kg.create_relationship("Service", svc["id"], "USES", "Technology", tech_id)


def print_stats() -> None:
    """Print graph statistics."""
    stats = kg.get_graph_stats()
    logger.info("Graph statistics:")
    for item in stats.get("nodes", []):
        logger.info("  %s: %d nodes", item.get("label"), item.get("count", 0))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=== Building knowledge graph ===")

    logger.info("Creating constraints...")
    kg.create_constraints()

    logger.info("Clearing existing graph...")
    kg.clear_graph()

    logger.info("Building nodes...")
    build_nodes()

    logger.info("Building relationships...")
    build_relationships()

    logger.info("=== Graph build complete ===")
    print_stats()
