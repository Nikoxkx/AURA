"""
AURA - Static Website Generator
Generates a static site for browsing reports, insights, and the knowledge graph.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.config import get_config
from src.common.logging import get_logger

logger = get_logger(__name__)


class WebsiteGenerator:
    """
    Generates a static website from AURA reports and insights.
    Outputs HTML files that can be served by any static file server.
    """

    def __init__(self, output_dir: Path | None = None, memory_store: Any = None):
        config = get_config()
        self.output_dir = output_dir or config.publication.reports_dir / "site"
        self.memory = memory_store

    async def generate(self) -> Path:
        """Generate the full static website."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate pages
        await self._generate_index()
        await self._generate_reports_page()
        await self._generate_insights_page()
        await self._generate_trends_page()
        await self._generate_stats_page()

        # Generate assets
        self._generate_css()

        logger.info("website_generated", output_dir=str(self.output_dir))
        return self.output_dir

    async def _generate_index(self) -> None:
        """Generate the homepage."""
        stats = {}
        if self.memory:
            try:
                stats = await self.memory.get_stats()
            except Exception:
                pass

        html = self._page_template(
            title="AURA - Autonomous Research Agent",
            content=f"""
            <section class="hero">
                <h1>🔬 AURA</h1>
                <p class="subtitle">Autonomous Research Understanding & Reporting Agent</p>
                <p>Continuously discovering, analyzing, and synthesizing AI research.</p>
            </section>

            <section class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_documents', 0):,}</div>
                    <div class="stat-label">Documents Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_insights', 0):,}</div>
                    <div class="stat-label">Insights Generated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_trends', 0):,}</div>
                    <div class="stat-label">Trends Tracked</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_reports', 0):,}</div>
                    <div class="stat-label">Reports Published</div>
                </div>
            </section>

            <section class="nav-grid">
                <a href="reports.html" class="nav-card">📄 <span>Reports</span></a>
                <a href="insights.html" class="nav-card">💡 <span>Insights</span></a>
                <a href="trends.html" class="nav-card">📈 <span>Trends</span></a>
                <a href="stats.html" class="nav-card">📊 <span>Statistics</span></a>
            </section>
            """,
        )

        (self.output_dir / "index.html").write_text(html)

    async def _generate_reports_page(self) -> None:
        """Generate the reports listing page."""
        reports = []
        if self.memory:
            try:
                records = await self.memory.get_recent_reports(limit=50)
                for r in records:
                    reports.append(f"""
                    <article class="report-card">
                        <h3><a href="#">{r.title}</a></h3>
                        <div class="meta">{r.report_type.title()} · {r.generated_at.strftime('%Y-%m-%d %H:%M')}</div>
                        <p>{r.content_markdown[:300].split(chr(10))[0]}...</p>
                    </article>
                    """)
            except Exception:
                pass

        content = f'<h2>Research Reports</h2><div class="reports-list">{"".join(reports) if reports else "<p>No reports yet.</p>"}</div>'
        html = self._page_template(title="Reports - AURA", content=content)
        (self.output_dir / "reports.html").write_text(html)

    async def _generate_insights_page(self) -> None:
        """Generate the insights listing page."""
        insights = []
        if self.memory:
            try:
                records = await self.memory.get_recent_insights(limit=50)
                for r in records:
                    icon = {"synthesis": "🔗", "contradiction": "⚡", "trend": "📈", "opportunity": "💡"}.get(r.insight_type, "📌")
                    insights.append(f"""
                    <article class="insight-card {r.importance}">
                        <div class="insight-header">
                            <span class="icon">{icon}</span>
                            <span class="type">{r.insight_type.title()}</span>
                            <span class="confidence">{r.confidence:.0%} confidence</span>
                        </div>
                        <h3>{r.title}</h3>
                        <p>{r.content[:300]}</p>
                        <div class="meta">{r.created_at.strftime('%Y-%m-%d')}</div>
                    </article>
                    """)
            except Exception:
                pass

        content = f'<h2>Research Insights</h2><div class="insights-list">{"".join(insights) if insights else "<p>No insights yet.</p>"}</div>'
        html = self._page_template(title="Insights - AURA", content=content)
        (self.output_dir / "insights.html").write_text(html)

    async def _generate_trends_page(self) -> None:
        """Generate the trends page."""
        trends = []
        if self.memory:
            try:
                records = await self.memory.get_trending_topics(min_growth=0, limit=30)
                for t in records:
                    growth_class = "positive" if t.growth_rate > 0 else "negative"
                    growth_icon = "📈" if t.growth_rate > 0 else "📉"
                    trends.append(f"""
                    <div class="trend-row">
                        <span class="trend-name">{t.topic}</span>
                        <span class="trend-growth {growth_class}">{growth_icon} {t.growth_rate:+.0f}%</span>
                        <span class="trend-count">{t.mention_count} mentions</span>
                    </div>
                    """)
            except Exception:
                pass

        content = f'<h2>Research Trends</h2><div class="trends-list">{"".join(trends) if trends else "<p>No trends detected yet.</p>"}</div>'
        html = self._page_template(title="Trends - AURA", content=content)
        (self.output_dir / "trends.html").write_text(html)

    async def _generate_stats_page(self) -> None:
        """Generate the statistics page."""
        stats = {}
        graph_stats = {}
        if self.memory:
            try:
                stats = await self.memory.get_stats()
            except Exception:
                pass

        rows = []
        for key, value in stats.items():
            rows.append(f'<tr><td>{key.replace("_", " ").title()}</td><td>{value:,}</td></tr>')

        content = f"""
        <h2>System Statistics</h2>
        <table class="stats-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        """
        html = self._page_template(title="Statistics - AURA", content=content)
        (self.output_dir / "stats.html").write_text(html)

    def _generate_css(self) -> None:
        """Generate the CSS stylesheet."""
        css_path = self.output_dir / "style.css"
        css_path.write_text(CSS_CONTENT)

    def _page_template(self, title: str, content: str) -> str:
        """Generate an HTML page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <nav>
        <div class="nav-brand"><a href="index.html">🔬 AURA</a></div>
        <div class="nav-links">
            <a href="reports.html">Reports</a>
            <a href="insights.html">Insights</a>
            <a href="trends.html">Trends</a>
            <a href="stats.html">Stats</a>
        </div>
    </nav>
    <main>
        {content}
    </main>
    <footer>
        <p>Generated by AURA · {datetime.utcnow().strftime('%Y-%m-%d')}</p>
    </footer>
</body>
</html>"""


CSS_CONTENT = """
/* AURA Website Styles */
:root {
    --bg: #0a0a0a;
    --surface: #141414;
    --surface2: #1e1e1e;
    --border: #2a2a2a;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #6366f1;
    --accent-dim: #4f46e5;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
}

nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
}

.nav-brand a { color: var(--text); text-decoration: none; font-weight: 700; font-size: 1.2rem; }
.nav-links a { color: var(--text-dim); text-decoration: none; margin-left: 1.5rem; transition: color 0.2s; }
.nav-links a:hover { color: var(--accent); }

main { max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }

.hero { text-align: center; padding: 3rem 0; }
.hero h1 { font-size: 3rem; margin-bottom: 0.5rem; }
.hero .subtitle { color: var(--accent); font-size: 1.2rem; margin-bottom: 1rem; }

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.stat-value { font-size: 2.5rem; font-weight: 700; color: var(--accent); }
.stat-label { color: var(--text-dim); font-size: 0.9rem; margin-top: 0.5rem; }

.nav-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.nav-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-decoration: none;
    color: var(--text);
    font-size: 1.1rem;
    transition: border-color 0.2s, transform 0.2s;
}

.nav-card:hover { border-color: var(--accent); transform: translateY(-2px); }

h2 { margin: 2rem 0 1rem; font-size: 1.5rem; }

.report-card, .insight-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.report-card h3 a { color: var(--accent); text-decoration: none; }
.report-card h3 a:hover { text-decoration: underline; }
.meta { color: var(--text-dim); font-size: 0.85rem; margin: 0.5rem 0; }

.insight-header { display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.5rem; }
.insight-header .type { background: var(--surface2); padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
.insight-header .confidence { color: var(--text-dim); font-size: 0.8rem; }
.insight-card.critical { border-left: 3px solid var(--red); }
.insight-card.high { border-left: 3px solid var(--yellow); }

.trend-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: var(--surface);
    border-radius: 6px;
    margin-bottom: 0.5rem;
}

.trend-growth.positive { color: var(--green); }
.trend-growth.negative { color: var(--red); }
.trend-count { color: var(--text-dim); font-size: 0.9rem; }

.stats-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
}

.stats-table th, .stats-table td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

.stats-table th { background: var(--surface2); font-weight: 600; }

footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}

@media (max-width: 768px) {
    .hero h1 { font-size: 2rem; }
    main { padding: 0 1rem; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
"""
