"""
seed_data.py — Seed Confluence pages and Slack messages with realistic corporate content.

Slack messages use chat:write.customize to simulate named personas.
Run from inside project5/:  python scripts/seed_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlassian import Confluence
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from brain.vector_store import upsert_documents
from core.config import get_settings
from core.logger import get_logger
from ingestion.confluence_engine import _strip_html

logger = get_logger("seed_data")


# ══════════════════════════════════════════════════════════════════════════════
# CONFLUENCE PAGES — Realistic Corporate Documentation
# ══════════════════════════════════════════════════════════════════════════════

def _get_confluence_pages(settings):
    spaces = settings.confluence_space_list
    space_wiki = spaces[0] if len(spaces) > 0 else "CW"
    space_eng = spaces[1] if len(spaces) > 1 else "ENG"

    return [
        # ─── Company Wiki Space ───────────────────────────────────────────────
        {
            "space": space_wiki,
            "title": "PTO Policy",
            "body": """
<h1>Paid Time Off (PTO) Policy</h1>
<p><strong>Effective Date:</strong> January 1, 2024</p>
<p><strong>Last Updated:</strong> January 15, 2024</p>

<h2>Overview</h2>
<p>At Athena Technologies, we believe in work-life balance. All full-time employees are entitled to paid time off as outlined below.</p>

<h2>PTO Allowance by Level</h2>
<table>
<tr><th>Employee Level</th><th>Annual PTO Days</th></tr>
<tr><td>Junior (L1-L2)</td><td>15 days</td></tr>
<tr><td>Mid-Level (L3-L4)</td><td>18 days</td></tr>
<tr><td>Senior (L5-L6)</td><td>20 days</td></tr>
<tr><td>Lead/Manager</td><td>20 days</td></tr>
<tr><td>Director+</td><td>25 days</td></tr>
</table>

<h2>Accrual</h2>
<p>PTO accrues monthly at a rate of 1/12th of your annual allowance. New employees begin accruing from their start date.</p>

<h2>Carryover</h2>
<p>Unused PTO can be carried over to the next year, up to a maximum of 5 days. Any excess will be forfeited on December 31st.</p>

<h2>Request Process</h2>
<ol>
<li>Submit requests through Workday at least 2 weeks in advance</li>
<li>Manager approval required for requests over 3 consecutive days</li>
<li>HR approval required for requests over 2 weeks</li>
</ol>

<h2>Blackout Periods</h2>
<p>PTO requests may be limited during critical business periods such as end-of-quarter, major releases, or company-wide events. Check with your manager.</p>

<p><em>Contact HR at hr@athena-tech.com for questions.</em></p>
""",
        },
        {
            "space": space_wiki,
            "title": "Remote Work Policy",
            "body": """
<h1>Remote Work Policy</h1>
<p><strong>Effective Date:</strong> March 1, 2024</p>

<h2>Hybrid Work Model</h2>
<p>Athena Technologies operates on a hybrid model. All employees are expected to be in-office a minimum of <strong>2 days per week</strong> (Tuesday and Thursday are anchor days).</p>

<h2>Fully Remote Exceptions</h2>
<p>Fully remote arrangements may be approved for:</p>
<ul>
<li>Employees located more than 50 miles from an office</li>
<li>Roles designated as "remote-first" in the job posting</li>
<li>Approved medical accommodations</li>
</ul>

<h2>Equipment</h2>
<p>Remote employees receive a one-time $500 home office stipend. IT will ship necessary equipment (laptop, monitors) within 5 business days of start date.</p>

<h2>Core Hours</h2>
<p>Regardless of location, all employees must be available during core hours: <strong>10 AM - 3 PM</strong> in their local timezone for meetings and collaboration.</p>

<h2>International Remote Work</h2>
<p>Working from outside your home country requires Legal and HR approval at least 30 days in advance due to tax and compliance implications.</p>
""",
        },
        {
            "space": space_wiki,
            "title": "Expense Reimbursement Policy",
            "body": """
<h1>Expense Reimbursement Policy</h1>

<h2>Eligible Expenses</h2>
<ul>
<li><strong>Travel:</strong> Flights (economy class, business for 6+ hour flights), hotels (up to $250/night), ground transportation</li>
<li><strong>Meals:</strong> Up to $75/day while traveling, $30 for client meals locally</li>
<li><strong>Software:</strong> Pre-approved tools and subscriptions up to $50/month</li>
<li><strong>Professional Development:</strong> Conferences, courses (up to $2,000/year with manager approval)</li>
</ul>

<h2>Submission Process</h2>
<ol>
<li>Submit expenses through Expensify within 30 days of incurring</li>
<li>Attach itemized receipts for all expenses over $25</li>
<li>Manager approval required for expenses over $500</li>
<li>Finance approval required for expenses over $2,000</li>
</ol>

<h2>Reimbursement Timeline</h2>
<p>Approved expenses are reimbursed within 10 business days via direct deposit.</p>

<h2>Corporate Card</h2>
<p>Employees at L5+ may request a corporate Amex. Contact finance@athena-tech.com.</p>

<h2>Non-Reimbursable Items</h2>
<p>Personal expenses, alcohol (except at approved client dinners), first-class upgrades, gym memberships, personal phone bills.</p>
""",
        },
        {
            "space": space_wiki,
            "title": "New Employee Onboarding Guide",
            "body": """
<h1>Welcome to Athena Technologies!</h1>
<p>Congratulations on joining our team. This guide will help you get started.</p>

<h2>Your First Day</h2>
<ol>
<li>Report to reception at 9:00 AM</li>
<li>HR orientation (9:30 AM - 11:00 AM)</li>
<li>IT setup - pick up laptop and credentials (11:00 AM - 12:00 PM)</li>
<li>Team lunch with your manager</li>
<li>Meet your onboarding buddy</li>
</ol>

<h2>First Week Checklist</h2>
<ul>
<li>Complete mandatory compliance training in Workday (4 hours)</li>
<li>Set up Slack, email, and calendar</li>
<li>Review team documentation in Confluence</li>
<li>Schedule 1:1s with key team members</li>
<li>Complete I-9 verification with HR</li>
</ul>

<h2>Key Systems</h2>
<table>
<tr><th>System</th><th>Purpose</th><th>Access</th></tr>
<tr><td>Workday</td><td>HR, Time Off, Expenses</td><td>SSO via Okta</td></tr>
<tr><td>Slack</td><td>Communication</td><td>your.name@athena-tech.com</td></tr>
<tr><td>GitHub</td><td>Code repositories</td><td>Request from Engineering</td></tr>
<tr><td>Jira</td><td>Project management</td><td>SSO via Okta</td></tr>
<tr><td>Confluence</td><td>Documentation</td><td>SSO via Okta</td></tr>
</table>

<h2>Important Contacts</h2>
<ul>
<li>HR: hr@athena-tech.com</li>
<li>IT Support: it-help@athena-tech.com or #it-support on Slack</li>
<li>Facilities: facilities@athena-tech.com</li>
</ul>
""",
        },
        {
            "space": space_wiki,
            "title": "Company Values and Culture",
            "body": """
<h1>Athena Technologies: Our Values</h1>

<h2>Mission</h2>
<p>To empower organizations with intelligent knowledge systems that unlock human potential.</p>

<h2>Core Values</h2>

<h3>1. Wisdom Over Speed</h3>
<p>We take time to make thoughtful decisions. Quick fixes create long-term debt. We'd rather ship the right thing slowly than the wrong thing fast.</p>

<h3>2. Radical Transparency</h3>
<p>Information should flow freely. We default to public channels, open documents, and shared context. Secrets slow us down.</p>

<h3>3. Own Your Impact</h3>
<p>Everyone is empowered to identify problems and fix them. Don't wait for permission. See something, do something.</p>

<h3>4. Customers Are Partners</h3>
<p>We succeed when our customers succeed. Their feedback shapes our roadmap. We build with them, not for them.</p>

<h3>5. Continuous Learning</h3>
<p>The best ideas win, regardless of source. We embrace failure as feedback. Every retrospective is an opportunity.</p>

<h2>Culture Norms</h2>
<ul>
<li>Meetings start on time and have agendas</li>
<li>Fridays are meeting-free (Focus Fridays)</li>
<li>Assume positive intent in all communications</li>
<li>Celebrate wins publicly, give feedback privately</li>
<li>Document decisions and reasoning</li>
</ul>
""",
        },
        {
            "space": space_wiki,
            "title": "Benefits Overview 2024",
            "body": """
<h1>Employee Benefits Guide 2024</h1>

<h2>Health Insurance</h2>
<p>We offer three medical plans through Aetna:</p>
<ul>
<li><strong>PPO Premium:</strong> Low deductible ($500), higher premium. Best for families.</li>
<li><strong>PPO Standard:</strong> Moderate deductible ($1,500), moderate premium.</li>
<li><strong>HDHP + HSA:</strong> High deductible ($3,000), lowest premium, company contributes $1,000 to HSA annually.</li>
</ul>
<p>Dental (Delta Dental) and Vision (VSP) included at no cost for employees.</p>

<h2>401(k) Retirement</h2>
<p>Company matches 100% of contributions up to 4% of salary. Immediate vesting. Managed through Fidelity.</p>

<h2>Equity</h2>
<p>All employees receive stock options. Standard vesting is 4-year with 1-year cliff. Refresh grants annually based on performance.</p>

<h2>Parental Leave</h2>
<ul>
<li>Primary caregiver: 16 weeks paid</li>
<li>Secondary caregiver: 6 weeks paid</li>
<li>Adoption/surrogacy: Same as above</li>
</ul>

<h2>Other Benefits</h2>
<ul>
<li>$100/month wellness stipend (gym, mental health apps, etc.)</li>
<li>$500/year learning & development budget</li>
<li>Commuter benefits (pre-tax transit/parking)</li>
<li>Free lunch in-office on Tuesdays and Thursdays</li>
<li>Annual company retreat</li>
</ul>
""",
        },
        {
            "space": space_wiki,
            "title": "Performance Review Process",
            "body": """
<h1>Performance Review Cycle</h1>

<h2>Timeline</h2>
<p>We run two review cycles per year:</p>
<ul>
<li><strong>Mid-Year Review:</strong> June (feedback only, no ratings)</li>
<li><strong>Annual Review:</strong> December (ratings, compensation, promotions)</li>
</ul>

<h2>Annual Review Process</h2>
<ol>
<li><strong>Self-Assessment</strong> (Dec 1-7): Complete your self-review in Workday</li>
<li><strong>Peer Feedback</strong> (Dec 1-14): Request and provide feedback for 3-5 peers</li>
<li><strong>Manager Review</strong> (Dec 15-21): Manager writes review incorporating all inputs</li>
<li><strong>Calibration</strong> (Dec 22-Jan 5): Leadership aligns on ratings and promotions</li>
<li><strong>Delivery</strong> (Jan 6-15): 1:1 meetings to discuss review and compensation</li>
</ol>

<h2>Rating Scale</h2>
<table>
<tr><th>Rating</th><th>Description</th><th>Distribution</th></tr>
<tr><td>Exceptional</td><td>Significantly exceeds expectations</td><td>~10%</td></tr>
<tr><td>Exceeds</td><td>Consistently exceeds expectations</td><td>~25%</td></tr>
<tr><td>Meets</td><td>Fully meets expectations</td><td>~50%</td></tr>
<tr><td>Developing</td><td>Partially meets expectations</td><td>~10%</td></tr>
<tr><td>Below</td><td>Does not meet expectations</td><td>~5%</td></tr>
</table>

<h2>Promotion Criteria</h2>
<p>Promotions require sustained performance at the next level for 6+ months, not just meeting current level expectations. See the Career Ladder document for level-specific competencies.</p>
""",
        },
        {
            "space": space_wiki,
            "title": "Security and Compliance Policies",
            "body": """
<h1>Information Security Policy</h1>

<h2>Data Classification</h2>
<ul>
<li><strong>Public:</strong> Marketing materials, public docs</li>
<li><strong>Internal:</strong> Most company documents, Slack messages</li>
<li><strong>Confidential:</strong> Customer data, financial reports, HR records</li>
<li><strong>Restricted:</strong> Security credentials, encryption keys, legal matters</li>
</ul>

<h2>Access Control</h2>
<ul>
<li>All systems require SSO through Okta</li>
<li>MFA mandatory for all employees</li>
<li>Access reviews conducted quarterly</li>
<li>Principle of least privilege applies</li>
</ul>

<h2>Acceptable Use</h2>
<ul>
<li>Company devices for business use primarily</li>
<li>No unauthorized software installations</li>
<li>Report lost/stolen devices immediately to IT</li>
<li>Do not share credentials or bypass security controls</li>
</ul>

<h2>Incident Response</h2>
<p>If you suspect a security incident:</p>
<ol>
<li>Do not attempt to investigate yourself</li>
<li>Report immediately to security@athena-tech.com</li>
<li>Call the Security Hotline for urgent issues: 1-888-SEC-RITY</li>
<li>Preserve evidence (don't delete logs, emails, etc.)</li>
</ol>

<h2>Compliance</h2>
<p>We are SOC 2 Type II certified and GDPR compliant. Annual security training is mandatory for all employees.</p>
""",
        },
        # ─── Engineering Space ────────────────────────────────────────────────
        {
            "space": space_eng,
            "title": "Deployment Guide",
            "body": """
<h1>Deployment Guide</h1>
<p><strong>Last Updated:</strong> January 2024</p>

<h2>Deployment Pipeline</h2>
<p>All services are deployed via <strong>Jenkins</strong>. Our CI/CD pipeline consists of:</p>
<ol>
<li>Code push to GitHub triggers Jenkins webhook</li>
<li>Jenkins runs unit tests and linting</li>
<li>Build Docker image, push to ECR</li>
<li>Deploy to staging environment</li>
<li>Run integration tests</li>
<li>Manual approval for production</li>
<li>Rolling deployment to production</li>
</ol>

<h2>Triggering Deployments</h2>
<ul>
<li><strong>Staging:</strong> Automatic on merge to <code>main</code> branch</li>
<li><strong>Production:</strong> Push a tag matching <code>v*.*.*</code> (e.g., <code>v2.3.1</code>)</li>
</ul>

<h2>Rollback Procedure</h2>
<p>If a production deployment fails:</p>
<ol>
<li>Go to Jenkins dashboard → Service → Production</li>
<li>Click "Rollback" and select the previous stable version</li>
<li>Notify #incidents channel with details</li>
</ol>

<h2>Environment Variables</h2>
<p>Secrets are managed in AWS Secrets Manager. Never commit secrets to git. Use the <code>secrets.yaml</code> template and configure in Jenkins.</p>

<h2>Service-Specific Notes</h2>
<ul>
<li><strong>API Gateway:</strong> Requires cache invalidation after deploy</li>
<li><strong>Auth Service:</strong> Zero-downtime deploy, uses blue-green</li>
<li><strong>Worker Services:</strong> Can tolerate brief downtime, uses rolling</li>
</ul>
""",
        },
        {
            "space": space_eng,
            "title": "Code Review Guidelines",
            "body": """
<h1>Code Review Best Practices</h1>

<h2>For Authors</h2>
<ul>
<li>Keep PRs small (< 400 lines preferred, never > 1000)</li>
<li>Write a clear description: What, Why, and How to test</li>
<li>Self-review before requesting reviews</li>
<li>Add tests for new functionality</li>
<li>Respond to feedback within 24 hours</li>
</ul>

<h2>For Reviewers</h2>
<ul>
<li>Review within 24 hours of request</li>
<li>Be constructive and kind — critique code, not people</li>
<li>Distinguish between "must fix" and "nit/suggestion"</li>
<li>Approve when good enough, don't block on perfection</li>
<li>Use "Request Changes" sparingly, only for critical issues</li>
</ul>

<h2>What to Look For</h2>
<ol>
<li><strong>Correctness:</strong> Does it work? Edge cases handled?</li>
<li><strong>Security:</strong> Input validation, auth checks, no secrets</li>
<li><strong>Performance:</strong> N+1 queries, unnecessary loops</li>
<li><strong>Readability:</strong> Clear names, reasonable complexity</li>
<li><strong>Tests:</strong> Adequate coverage, meaningful assertions</li>
</ol>

<h2>Approval Requirements</h2>
<ul>
<li>1 approval for most PRs</li>
<li>2 approvals for: security-sensitive code, database migrations, infrastructure changes</li>
<li>CODEOWNERS must approve changes to critical paths</li>
</ul>

<h2>Merge Strategy</h2>
<p>Use "Squash and Merge" for feature branches. Keep main history clean.</p>
""",
        },
        {
            "space": space_eng,
            "title": "System Architecture Overview",
            "body": """
<h1>Athena Platform Architecture</h1>

<h2>High-Level Overview</h2>
<p>Athena is a distributed microservices platform running on AWS EKS (Kubernetes).</p>

<h2>Core Services</h2>
<table>
<tr><th>Service</th><th>Purpose</th><th>Tech Stack</th></tr>
<tr><td>API Gateway</td><td>Request routing, auth, rate limiting</td><td>Kong, Lua</td></tr>
<tr><td>Auth Service</td><td>Authentication, OAuth, JWT</td><td>Go, PostgreSQL</td></tr>
<tr><td>User Service</td><td>User management, profiles</td><td>Python, PostgreSQL</td></tr>
<tr><td>Knowledge Service</td><td>RAG pipeline, embeddings</td><td>Python, Pinecone</td></tr>
<tr><td>Ingestion Service</td><td>Data connectors (Slack, Confluence)</td><td>Python, Redis</td></tr>
<tr><td>Search Service</td><td>Full-text search</td><td>Elasticsearch</td></tr>
<tr><td>Notification Service</td><td>Email, push, in-app</td><td>Node.js, SQS</td></tr>
</table>

<h2>Data Stores</h2>
<ul>
<li><strong>PostgreSQL (RDS):</strong> Primary relational data</li>
<li><strong>Redis (ElastiCache):</strong> Caching, sessions, rate limiting</li>
<li><strong>Pinecone:</strong> Vector embeddings for RAG</li>
<li><strong>S3:</strong> File storage, backups</li>
<li><strong>Elasticsearch:</strong> Full-text search, logs</li>
</ul>

<h2>Infrastructure</h2>
<ul>
<li><strong>Compute:</strong> EKS (Kubernetes) with Karpenter autoscaling</li>
<li><strong>Networking:</strong> VPC with public/private subnets, ALB</li>
<li><strong>CDN:</strong> CloudFront for static assets</li>
<li><strong>DNS:</strong> Route53</li>
<li><strong>Monitoring:</strong> Datadog, PagerDuty</li>
</ul>

<h2>Diagrams</h2>
<p>See Lucidchart: <a href="#">Athena Architecture Diagram</a></p>
""",
        },
        {
            "space": space_eng,
            "title": "Incident Response Runbook",
            "body": """
<h1>Incident Response Runbook</h1>

<h2>Severity Levels</h2>
<table>
<tr><th>Level</th><th>Description</th><th>Response Time</th><th>Example</th></tr>
<tr><td>SEV1</td><td>Complete outage, data breach</td><td>15 min</td><td>Site down, security incident</td></tr>
<tr><td>SEV2</td><td>Major feature broken</td><td>1 hour</td><td>Auth failing, data loss</td></tr>
<tr><td>SEV3</td><td>Minor feature degraded</td><td>4 hours</td><td>Slow search, UI bug</td></tr>
<tr><td>SEV4</td><td>Cosmetic / low impact</td><td>Next sprint</td><td>Typo, minor UI issue</td></tr>
</table>

<h2>Incident Commander Duties</h2>
<ol>
<li>Acknowledge incident in PagerDuty</li>
<li>Create incident Slack channel: #inc-YYYY-MM-DD-brief-desc</li>
<li>Assess severity and escalate if needed</li>
<li>Coordinate response, delegate tasks</li>
<li>Communicate status updates every 30 min</li>
<li>Declare resolution and schedule postmortem</li>
</ol>

<h2>Communication Templates</h2>
<p><strong>Initial:</strong> "We're investigating reports of [issue]. The team is engaged. Updates to follow."</p>
<p><strong>Update:</strong> "Update: We've identified [root cause]. Implementing [fix]. ETA: [time]."</p>
<p><strong>Resolution:</strong> "Resolved: [issue] has been fixed. Root cause was [cause]. Postmortem scheduled for [date]."</p>

<h2>Postmortem Process</h2>
<ul>
<li>Complete within 5 business days of resolution</li>
<li>Blameless — focus on systems, not individuals</li>
<li>Document timeline, root cause, action items</li>
<li>Share in #engineering and add to Incident Log</li>
</ul>

<h2>Escalation Contacts</h2>
<ul>
<li>Engineering Manager on-call: Check PagerDuty</li>
<li>VP Engineering: @michael.torres</li>
<li>CTO: @jennifer.wang</li>
</ul>
""",
        },
        {
            "space": space_eng,
            "title": "API Documentation Standards",
            "body": """
<h1>API Documentation Standards</h1>

<h2>OpenAPI Specification</h2>
<p>All APIs must have OpenAPI 3.0 specs in <code>/docs/openapi.yaml</code>. Auto-generate from code annotations where possible.</p>

<h2>Required Documentation</h2>
<ul>
<li>Endpoint description and purpose</li>
<li>Request/response schemas with examples</li>
<li>Authentication requirements</li>
<li>Rate limiting information</li>
<li>Error codes and meanings</li>
</ul>

<h2>Versioning</h2>
<p>APIs are versioned via URL path: <code>/api/v1/</code>, <code>/api/v2/</code>. Breaking changes require a new major version.</p>

<h2>Authentication</h2>
<p>All APIs require Bearer token authentication:</p>
<pre>
Authorization: Bearer &lt;jwt_token&gt;
</pre>
<p>Tokens are obtained from the Auth Service. See Auth API docs for details.</p>

<h2>Rate Limiting</h2>
<table>
<tr><th>Tier</th><th>Requests/min</th><th>Burst</th></tr>
<tr><td>Free</td><td>60</td><td>10</td></tr>
<tr><td>Pro</td><td>600</td><td>100</td></tr>
<tr><td>Enterprise</td><td>6000</td><td>1000</td></tr>
</table>
<p>Rate limit headers are included in all responses: <code>X-RateLimit-Remaining</code>, <code>X-RateLimit-Reset</code>.</p>

<h2>Error Response Format</h2>
<pre>
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email format",
    "details": {...}
  }
}
</pre>
""",
        },
        {
            "space": space_eng,
            "title": "On-Call Procedures",
            "body": """
<h1>On-Call Engineer Guide</h1>

<h2>On-Call Rotation</h2>
<ul>
<li>Rotation: Weekly, Sunday 9 AM to Sunday 9 AM</li>
<li>Primary and secondary on-call for each shift</li>
<li>Schedule managed in PagerDuty</li>
<li>Swap requests: DM your manager and update PagerDuty</li>
</ul>

<h2>Expectations</h2>
<ul>
<li>Respond to pages within 15 minutes</li>
<li>Laptop and internet access required at all times</li>
<li>Stay within cell service / WiFi range</li>
<li>No alcohol or substances that impair response</li>
<li>Hand off cleanly at rotation end</li>
</ul>

<h2>Compensation</h2>
<ul>
<li>$500/week on-call stipend</li>
<li>1.5x hourly rate for time actively incident-responding outside business hours</li>
<li>Comp time for extended incidents (manager discretion)</li>
</ul>

<h2>Common Issues and Runbooks</h2>
<ul>
<li><a href="#">API Gateway 5xx spike</a></li>
<li><a href="#">Database connection pool exhaustion</a></li>
<li><a href="#">Kubernetes pod crash loop</a></li>
<li><a href="#">Redis memory pressure</a></li>
<li><a href="#">Certificate expiration</a></li>
</ul>

<h2>Escalation</h2>
<p>If you cannot resolve within 30 minutes, escalate to secondary. If secondary unavailable, escalate to Engineering Manager on-call.</p>
""",
        },
        {
            "space": space_eng,
            "title": "Tech Stack and Tooling",
            "body": """
<h1>Engineering Tech Stack</h1>

<h2>Languages</h2>
<ul>
<li><strong>Backend:</strong> Python 3.11+ (FastAPI), Go 1.21+ (performance-critical)</li>
<li><strong>Frontend:</strong> TypeScript, React 18, Next.js 14</li>
<li><strong>Infrastructure:</strong> Terraform, Helm</li>
<li><strong>Scripts:</strong> Bash, Python</li>
</ul>

<h2>Frameworks</h2>
<ul>
<li><strong>API:</strong> FastAPI (Python), Gin (Go)</li>
<li><strong>ORM:</strong> SQLAlchemy 2.0</li>
<li><strong>Testing:</strong> pytest, Jest, Playwright</li>
<li><strong>Task Queue:</strong> Celery + Redis</li>
</ul>

<h2>Infrastructure</h2>
<ul>
<li><strong>Cloud:</strong> AWS (primary), some GCP for ML</li>
<li><strong>Containers:</strong> Docker, Kubernetes (EKS)</li>
<li><strong>CI/CD:</strong> Jenkins (migrating to GitHub Actions)</li>
<li><strong>IaC:</strong> Terraform, Helm charts</li>
</ul>

<h2>Observability</h2>
<ul>
<li><strong>Metrics:</strong> Datadog</li>
<li><strong>Logs:</strong> Datadog Logs (from Elasticsearch)</li>
<li><strong>Tracing:</strong> Datadog APM</li>
<li><strong>Alerting:</strong> PagerDuty</li>
<li><strong>Error Tracking:</strong> Sentry</li>
</ul>

<h2>Development Tools</h2>
<ul>
<li><strong>IDE:</strong> VS Code (recommended), JetBrains</li>
<li><strong>Version Control:</strong> GitHub Enterprise</li>
<li><strong>Package Managers:</strong> pip/poetry (Python), npm (JS)</li>
<li><strong>Local Dev:</strong> Docker Compose, Tilt</li>
</ul>

<h2>Approved Libraries</h2>
<p>Check the <a href="#">Approved Dependencies List</a> before adding new libraries. Security review required for new dependencies.</p>
""",
        },
        {
            "space": space_eng,
            "title": "Database Conventions",
            "body": """
<h1>Database Standards</h1>

<h2>Naming Conventions</h2>
<ul>
<li>Tables: snake_case, plural (e.g., <code>users</code>, <code>order_items</code>)</li>
<li>Columns: snake_case (e.g., <code>created_at</code>, <code>user_id</code>)</li>
<li>Primary keys: <code>id</code> (UUID preferred, BIGINT acceptable)</li>
<li>Foreign keys: <code>&lt;table_singular&gt;_id</code> (e.g., <code>user_id</code>)</li>
<li>Indexes: <code>idx_&lt;table&gt;_&lt;columns&gt;</code></li>
</ul>

<h2>Required Columns</h2>
<p>All tables must have:</p>
<ul>
<li><code>id</code> — Primary key</li>
<li><code>created_at</code> — Timestamp, default NOW()</li>
<li><code>updated_at</code> — Timestamp, auto-update on change</li>
</ul>

<h2>Migrations</h2>
<ul>
<li>Use Alembic for Python services</li>
<li>Migrations must be reversible (include downgrade)</li>
<li>Test migrations on staging before production</li>
<li>Large data migrations: coordinate with DBA team</li>
</ul>

<h2>Query Guidelines</h2>
<ul>
<li>Always use parameterized queries (no string interpolation)</li>
<li>Add indexes for columns in WHERE clauses</li>
<li>Avoid SELECT * in production code</li>
<li>Use EXPLAIN ANALYZE for slow queries</li>
<li>Connection pooling via PgBouncer</li>
</ul>

<h2>Backups</h2>
<p>RDS automated backups: 7-day retention, point-in-time recovery enabled. Monthly snapshots retained for 1 year.</p>
""",
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SLACK MESSAGES — Realistic Corporate Conversations
# ══════════════════════════════════════════════════════════════════════════════

_SLACK_PERSONAS = {
    "alex_chen": {"username": "Alex Chen", "icon_emoji": ":hammer_and_wrench:", "role": "Lead Architect", "authority": 10},
    "sarah_mitchell": {"username": "Sarah Mitchell", "icon_emoji": ":briefcase:", "role": "HR Lead", "authority": 10},
    "michael_torres": {"username": "Michael Torres", "icon_emoji": ":desktop_computer:", "role": "VP Engineering", "authority": 10},
    "jennifer_wang": {"username": "Jennifer Wang", "icon_emoji": ":rocket:", "role": "CTO", "authority": 10},
    "david_kim": {"username": "David Kim", "icon_emoji": ":male-technologist:", "role": "Senior Engineer", "authority": 7},
    "emily_johnson": {"username": "Emily Johnson", "icon_emoji": ":female-technologist:", "role": "Senior Engineer", "authority": 7},
    "ryan_patel": {"username": "Ryan Patel", "icon_emoji": ":gear:", "role": "DevOps Engineer", "authority": 7},
    "lisa_nguyen": {"username": "Lisa Nguyen", "icon_emoji": ":bar_chart:", "role": "Product Manager", "authority": 7},
    "jordan_kim": {"username": "Jordan Kim", "icon_emoji": ":technologist:", "role": "Junior Developer", "authority": 3},
    "taylor_smith": {"username": "Taylor Smith", "icon_emoji": ":woman-raising-hand:", "role": "Junior Developer", "authority": 3},
    "chris_lee": {"username": "Chris Lee", "icon_emoji": ":man-raising-hand:", "role": "Software Engineer", "authority": 5},
    "amanda_garcia": {"username": "Amanda Garcia", "icon_emoji": ":female_detective:", "role": "Security Engineer", "authority": 7},
}

_SLACK_MESSAGES = [
    # ─── General Channel ──────────────────────────────────────────────────────
    {
        "channel_name": "general",
        "persona": "jennifer_wang",
        "text": "Hey everyone! Quick update on our Q2 priorities. We're doubling down on the enterprise features — SSO, audit logs, and advanced permissions. The RAG accuracy improvements are also critical. Let's make this quarter count! :rocket:",
    },
    {
        "channel_name": "general",
        "persona": "michael_torres",
        "text": "Reminder: All-hands meeting tomorrow at 2 PM PT. We'll be covering the product roadmap, Q1 financials, and announcing some exciting team changes. See you there!",
    },
    {
        "channel_name": "general",
        "persona": "sarah_mitchell",
        "text": "HR Update: We're excited to welcome 5 new team members starting Monday! Please give a warm welcome to our new engineers and product folks. Onboarding schedule will be shared shortly.",
    },
    {
        "channel_name": "general",
        "persona": "alex_chen",
        "text": "PSA: I've completed the migration from Jenkins to GitHub Actions. All pipelines are now running on GHA. The Jenkins servers will be decommissioned next week. Please update any bookmarks or scripts that reference Jenkins directly. The wiki deployment guide is outdated — I'll update it this week, but for now, just know that GitHub Actions is the source of truth.",
    },
    {
        "channel_name": "general",
        "persona": "lisa_nguyen",
        "text": "Just shipped the new dashboard redesign to 10% of users! :tada: Early metrics look promising — 23% increase in engagement. Will share full report in #product once we have more data.",
    },
    # ─── Engineering Channel ──────────────────────────────────────────────────
    {
        "channel_name": "engineering",
        "persona": "jordan_kim",
        "text": "Hey team, quick question — how do I deploy the Auth-Service to staging? The wiki mentions Jenkins but I heard we switched to something else? Also, do I need any special permissions?",
    },
    {
        "channel_name": "engineering",
        "persona": "alex_chen",
        "text": "@jordan_kim Great question! We've fully migrated to GitHub Actions now. Just push to main and it auto-deploys to staging. For production, create a release tag like v2.3.1. You should have permissions already — check that you're in the 'engineers' GitHub team. Ping me if not.",
    },
    {
        "channel_name": "engineering",
        "persona": "david_kim",
        "text": "Heads up everyone — I'm seeing elevated error rates on the Knowledge Service. Investigating now. Looks like it might be related to the Pinecone rate limiting changes they announced last week. Will keep you posted.",
    },
    {
        "channel_name": "engineering",
        "persona": "emily_johnson",
        "text": "Just merged the new caching layer for embeddings. Should reduce our Gemini API costs by ~40% and improve p99 latency. Please keep an eye out for any regressions.",
    },
    {
        "channel_name": "engineering",
        "persona": "ryan_patel",
        "text": "Kubernetes cluster upgrade scheduled for Saturday 2 AM PT. Expecting ~10 minutes of read-only mode. All services should failover gracefully but I'll be monitoring. Runbook is updated if anyone wants to follow along.",
    },
    {
        "channel_name": "engineering",
        "persona": "chris_lee",
        "text": "Anyone have experience with the new LangChain 1.0 LCEL syntax? I'm trying to migrate our resolver chain and hitting some weird import errors. The docs are a bit sparse.",
    },
    {
        "channel_name": "engineering",
        "persona": "emily_johnson",
        "text": "@chris_lee Yeah the imports changed a lot. Everything moved from langchain.schema to langchain_core. I can pair with you tomorrow if you want — I just went through this last week.",
    },
    {
        "channel_name": "engineering",
        "persona": "taylor_smith",
        "text": "Is there a style guide for our Python code? I'm seeing different patterns in different services and want to make sure I'm following the right conventions.",
    },
    {
        "channel_name": "engineering",
        "persona": "david_kim",
        "text": "@taylor_smith Check out the Engineering space in Confluence — there's a Python Style Guide doc. TLDR: we follow PEP 8, use Black for formatting, and ruff for linting. The pre-commit hooks should catch most issues.",
    },
    {
        "channel_name": "engineering",
        "persona": "amanda_garcia",
        "text": "Security reminder: We've enabled mandatory branch protection on all repos. PRs now require at least one approval and passing CI before merge. Also, please rotate any credentials that are older than 90 days.",
    },
    # ─── HR Updates Channel ───────────────────────────────────────────────────
    {
        "channel_name": "hr-updates",
        "persona": "sarah_mitchell",
        "text": "📢 Important PTO Policy Update: Effective immediately, all Lead and Manager level employees now receive 25 days of PTO annually, up from 20 days. This change reflects our commitment to work-life balance for our senior team members. The Confluence wiki will be updated shortly, but please consider this Slack message as the official policy update until then.",
    },
    {
        "channel_name": "hr-updates",
        "persona": "sarah_mitchell",
        "text": "Benefits enrollment period opens next Monday! You'll receive an email with instructions to review and update your selections in Workday. Changes take effect January 1st. HR office hours available Wednesday 2-4 PM for questions.",
    },
    {
        "channel_name": "hr-updates",
        "persona": "sarah_mitchell",
        "text": "Reminder: Q4 performance review self-assessments are due by December 7th. Please complete your self-review in Workday. Peer feedback nominations should also be submitted by then. Reach out to your manager if you have questions about the process.",
    },
    {
        "channel_name": "hr-updates",
        "persona": "sarah_mitchell",
        "text": "We're updating our remote work policy starting Q2. The new policy requires 3 days in-office per week (up from 2). Tuesday, Wednesday, and Thursday will be anchor days. Fully remote exceptions will still be considered case-by-case. FAQ coming soon.",
    },
    # ─── Product Channel ──────────────────────────────────────────────────────
    {
        "channel_name": "product",
        "persona": "lisa_nguyen",
        "text": "Sprint 23 planning complete! Key items: 1) Slack connector improvements (thread support), 2) Confluence space filtering, 3) Authority scoring UI. Engineering estimates look good — should be achievable in 2 weeks.",
    },
    {
        "channel_name": "product",
        "persona": "jennifer_wang",
        "text": "Customer feedback from Acme Corp: They love the RAG accuracy but need better citation tracking. When Athena gives an answer, they want to see exactly which Slack message or Confluence page it came from, with links. Can we prioritize this for next sprint?",
    },
    {
        "channel_name": "product",
        "persona": "lisa_nguyen",
        "text": "@jennifer_wang Absolutely, we've heard this from multiple customers. I'll add it to the backlog and we can discuss prioritization in tomorrow's planning meeting.",
    },
    {
        "channel_name": "product",
        "persona": "michael_torres",
        "text": "FYI — we're seeing strong interest from enterprise prospects on the compliance features. SOC 2 certification has been a key selling point. Let's make sure we maintain that bar.",
    },
    # ─── Incidents Channel ────────────────────────────────────────────────────
    {
        "channel_name": "incidents",
        "persona": "ryan_patel",
        "text": ":red_circle: INCIDENT: API response times degraded. P95 latency at 2.5s (normal is 200ms). Investigating — initial suspicion is database connection pool exhaustion. Will update in 15 min.",
    },
    {
        "channel_name": "incidents",
        "persona": "ryan_patel",
        "text": ":yellow_circle: UPDATE: Confirmed — PgBouncer connection pool was saturated due to a long-running analytics query. Killed the query and pool is recovering. Latency returning to normal.",
    },
    {
        "channel_name": "incidents",
        "persona": "ryan_patel",
        "text": ":green_circle: RESOLVED: API latency back to normal. Root cause: runaway analytics query holding connections. Action items: 1) Add query timeout to analytics role, 2) Improve monitoring for pool saturation. Postmortem scheduled for Thursday.",
    },
    {
        "channel_name": "incidents",
        "persona": "david_kim",
        "text": ":yellow_circle: INCIDENT: Elevated error rates on Knowledge Service. ~5% of embedding requests failing with rate limit errors from Gemini API. Implementing backoff and retry logic. Non-blocking but degraded experience.",
    },
    {
        "channel_name": "incidents",
        "persona": "david_kim",
        "text": ":green_circle: RESOLVED: Deployed retry logic with exponential backoff. Error rate back to baseline. Will look into request batching to reduce API calls long-term.",
    },
    # ─── Random Channel ───────────────────────────────────────────────────────
    {
        "channel_name": "random",
        "persona": "chris_lee",
        "text": "Anyone up for board games after work on Friday? I'm bringing Catan and Wingspan. Lobby at 6 PM.",
    },
    {
        "channel_name": "random",
        "persona": "taylor_smith",
        "text": "The new coffee machine in the kitchen is amazing! Highly recommend the oat milk latte setting. :coffee:",
    },
    {
        "channel_name": "random",
        "persona": "jordan_kim",
        "text": "Has anyone tried that new ramen place on 3rd street? Looking for lunch recommendations.",
    },
    {
        "channel_name": "random",
        "persona": "emily_johnson",
        "text": "@jordan_kim Yesss it's so good! Get the spicy miso with extra chashu. Totally worth the wait.",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# SEED FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _seed_confluence(cf: Confluence, settings) -> list[Document]:
    docs: list[Document] = []
    pages = _get_confluence_pages(settings)

    for page in pages:
        try:
            existing = cf.get_page_by_title(space=page["space"], title=page["title"])
            if existing:
                logger.info("Page '%s' already exists (id=%s), skipping create.", page["title"], existing["id"])
                page_id = existing["id"]
            else:
                created = cf.create_page(space=page["space"], title=page["title"], body=page["body"])
                page_id = created.get("id", "")
                logger.info("Created Confluence page '%s' (id=%s).", page["title"], page_id)
        except Exception as e:
            logger.error("Failed to create page '%s': %s", page["title"], e)
            continue

        docs.append(
            Document(
                page_content=f"{page['title']}\n\n{_strip_html(page['body'])}",
                metadata={
                    "source": "confluence",
                    "url": f"{settings.confluence_url}/pages/{page_id}",
                    "author_role": f"Confluence Author ({page['space']})",
                    "authority_score": 5,
                    "timestamp": "2024-01-15T09:00:00.000Z",
                    "namespace": "confluence",
                    "space": page["space"],
                    "page_title": page["title"],
                    "page_id": page_id,
                    "last_modified_by": "System",
                },
            )
        )
    return docs


def _get_channel_id(client: WebClient, name: str) -> str | None:
    cursor = None
    while True:
        kwargs: dict = {"types": "public_channel", "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.conversations_list(**kwargs)
        for ch in resp.get("channels", []):
            if ch.get("name") == name:
                return ch["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            return None


def _seed_slack(client: WebClient) -> list[Document]:
    docs: list[Document] = []
    channel_cache: dict[str, str | None] = {}

    for msg in _SLACK_MESSAGES:
        channel_name = msg["channel_name"]

        if channel_name not in channel_cache:
            channel_cache[channel_name] = _get_channel_id(client, channel_name)

        channel_id = channel_cache[channel_name]
        if not channel_id:
            logger.warning("Channel '#%s' not found — skipping.", channel_name)
            continue

        persona = _SLACK_PERSONAS[msg["persona"]]
        ts = "0"

        try:
            resp = client.chat_postMessage(
                channel=channel_id,
                text=msg["text"],
                username=persona["username"],
                icon_emoji=persona["icon_emoji"],
            )
            ts = resp["ts"]
            logger.info("Posted as '%s' in #%s.", persona["username"], channel_name)
        except SlackApiError as exc:
            logger.error("Failed to post to #%s: %s", channel_name, exc)

        url = f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}"
        docs.append(
            Document(
                page_content=msg["text"],
                metadata={
                    "source": "slack",
                    "url": url,
                    "author_role": persona["role"],
                    "authority_score": persona["authority"],
                    "timestamp": ts,
                    "namespace": "slack",
                    "channel_id": channel_id,
                    "author_name": persona["username"],
                },
            )
        )
    return docs


def _ensure_index(settings) -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [i.name for i in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info("Creating Pinecone index '%s'...", settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_region),
        )
        logger.info("Index created.")
    else:
        logger.info("Pinecone index '%s' already exists.", settings.pinecone_index_name)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    settings = get_settings()

    logger.info("=== Athena seed starting ===")
    _ensure_index(settings)

    logger.info("--- Seeding Confluence (%d pages) ---", len(_get_confluence_pages(settings)))
    cf = Confluence(
        url=settings.confluence_url,
        username=settings.confluence_user,
        password=settings.confluence_api_token,
        cloud=True,
    )
    cf_docs = _seed_confluence(cf, settings)
    upsert_documents(cf_docs, namespace="confluence")

    logger.info("--- Seeding Slack (%d messages) ---", len(_SLACK_MESSAGES))
    slack_client = WebClient(token=settings.slack_bot_token)
    slack_docs = _seed_slack(slack_client)
    upsert_documents(slack_docs, namespace="slack")

    logger.info("=== Seed complete. Confluence: %d, Slack: %d ===", len(cf_docs), len(slack_docs))
