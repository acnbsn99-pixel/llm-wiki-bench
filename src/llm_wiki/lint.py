"""Wiki Linter - adapted from llm-wiki-agent tools/lint.py.

This module faithfully preserves the original linting logic while adapting:
- LLM calls use src/llm_client.py instead of direct litellm
- Added tracking for token usage and latency
"""

import re
from datetime import date
from pathlib import Path
from typing import Optional
from collections import defaultdict

from ..llm_client import LLMClient, CallResult
from .tracking import TrajectoryLogger


def read_file(path: Path) -> str:
    """Read file content, returns empty string if not exists."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from content."""
    return re.findall(r'\[\[([^\]]+)\]\]', content)


class WikiLinter:
    """Lints the LLM Wiki for health issues.
    
    Faithfully adapted from the original llm-wiki-agent tools/lint.py.
    
    The original agent checks for:
    - Orphan pages (no inbound wikilinks from other pages)
    - Broken wikilinks (pointing to pages that don't exist)
    - Missing entity pages (entities mentioned in 3+ pages but no page)
    - Contradictions between pages (via LLM)
    - Data gaps and suggested new sources (via LLM)
    
    This adaptation preserves that exact logic while:
    - Using our LLMClient for API calls
    - Tracking token usage and latency
    """
    
    def __init__(
        self,
        wiki_dir: Optional[Path] = None,
        client: Optional[LLMClient] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None
    ):
        """Initialize the WikiLinter.
        
        Args:
            wiki_dir: Directory for wiki files. Defaults to project root / wiki
            client: LLMClient instance. If None, creates new one
            trajectory_logger: Logger for tracking. If None, creates new one
        """
        # Set up paths
        self.repo_root = Path(__file__).parent.parent.parent
        self.wiki_dir = wiki_dir or self.repo_root / "wiki"
        self.log_file = self.wiki_dir / "log.md"
        self.lint_report_file = self.wiki_dir / "lint-report.md"
        
        # Initialize clients
        self.client = client or LLMClient()
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()
    
    def all_wiki_pages(self) -> list[Path]:
        """Get all wiki pages excluding index, log, and lint-report."""
        return [
            p for p in self.wiki_dir.rglob("*.md")
            if p.name not in ("index.md", "log.md", "lint-report.md")
        ]
    
    def page_name_to_path(self, name: str) -> list[Path]:
        """Try to resolve a [[WikiLink]] to a file path."""
        candidates = []
        for p in self.all_wiki_pages():
            if p.stem.lower() == name.lower() or p.stem == name:
                candidates.append(p)
        return candidates
    
    def find_orphans(self, pages: list[Path]) -> list[Path]:
        """Find pages with no inbound wikilinks."""
        inbound = defaultdict(int)
        for p in pages:
            content = read_file(p)
            for link in extract_wikilinks(content):
                resolved = self.page_name_to_path(link)
                for r in resolved:
                    inbound[r] += 1
        return [
            p for p in pages
            if inbound[p] == 0 and p != self.wiki_dir / "overview.md"
        ]
    
    def find_broken_links(self, pages: list[Path]) -> list[tuple[Path, str]]:
        """Find wikilinks pointing to non-existent pages."""
        broken = []
        for p in pages:
            content = read_file(p)
            for link in extract_wikilinks(content):
                if not self.page_name_to_path(link):
                    broken.append((p, link))
        return broken
    
    def find_missing_entities(self, pages: list[Path]) -> list[str]:
        """Find entity-like names mentioned in 3+ pages but lacking their own page."""
        mention_counts = defaultdict(int)
        existing_pages = {p.stem.lower() for p in pages}
        
        for p in pages:
            content = read_file(p)
            links = extract_wikilinks(content)
            for link in links:
                if link.lower() not in existing_pages:
                    mention_counts[link] += 1
        
        return [name for name, count in mention_counts.items() if count >= 3]
    
    def run_lint(self) -> str:
        """Run full lint check and return report.
        
        Returns:
            Markdown lint report string
        """
        pages = self.all_wiki_pages()
        today = date.today().isoformat()
        
        if not pages:
            print("Wiki is empty. Nothing to lint.")
            return ""
        
        print(f"Linting {len(pages)} wiki pages...")
        
        # Start trajectory tracking
        self.trajectory_logger.start_query(f"lint-{today}")
        
        # Deterministic checks
        orphans = self.find_orphans(pages)
        broken = self.find_broken_links(pages)
        missing_entities = self.find_missing_entities(pages)
        
        print(f"  orphans: {len(orphans)}")
        print(f"  broken links: {len(broken)}")
        print(f"  missing entity pages: {len(missing_entities)}")
        
        # Build context for semantic checks (contradictions, gaps)
        # Use a sample of pages to stay within context limits
        sample = pages[:20]
        pages_context = ""
        for p in sample:
            rel = p.relative_to(self.repo_root)
            pages_context += f"\n\n### {rel}\n{read_file(p)[:1500]}"  # truncate long pages
        
        print("  running semantic lint via API...")
        prompt = f"""You are linting an LLM Wiki. Review the pages below and identify:
1. Contradictions between pages (claims that conflict)
2. Stale content (summaries that newer sources have superseded)
3. Data gaps (important questions the wiki can't answer — suggest specific sources to find)
4. Concepts mentioned but lacking depth

Wiki pages (sample of {len(sample)} pages):
{pages_context}

Return a markdown lint report with these sections:
## Contradictions
## Stale Content
## Data Gaps & Suggested Sources
## Concepts Needing More Depth

Be specific — name the exact pages and claims involved.
"""
        # Log the cycle
        self.trajectory_logger.log_cycle(thought=prompt, action="semantic_lint")
        
        result: CallResult = self.client.call(prompt=prompt, max_tokens=3000)
        
        # Update metrics
        self.trajectory_logger.update_metrics(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            latency_ms=result.latency_ms
        )
        
        semantic_report = result.content
        
        # Log observation
        self.trajectory_logger.log_cycle(
            thought="",
            action="lint_complete",
            observation=f"Found {len(orphans)} orphans, {len(broken)} broken links, {len(missing_entities)} missing entities"
        )
        
        # Compose full report
        report_lines = [
            f"# Wiki Lint Report — {today}",
            "",
            f"Scanned {len(pages)} pages.",
            "",
            "## Structural Issues",
            "",
        ]
        
        if orphans:
            report_lines.append("### Orphan Pages (no inbound links)")
            for p in orphans:
                report_lines.append(f"- `{p.relative_to(self.repo_root)}`")
            report_lines.append("")
        
        if broken:
            report_lines.append("### Broken Wikilinks")
            for page, link in broken:
                report_lines.append(
                    f"- `{page.relative_to(self.repo_root)}` links to `[[{link}]]` — not found"
                )
            report_lines.append("")
        
        if missing_entities:
            report_lines.append("### Missing Entity Pages (mentioned 3+ times but no page)")
            for name in missing_entities:
                report_lines.append(f"- `[[{name}]]`")
            report_lines.append("")
        
        if not orphans and not broken and not missing_entities:
            report_lines.append("No structural issues found.")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        report_lines.append(semantic_report)
        
        report = "\n".join(report_lines)
        print("\n" + report)
        
        # End trajectory tracking
        self.trajectory_logger.end_query()
        
        return report
    
    def save_report(self, report: str) -> Path:
        """Save lint report to wiki/lint-report.md.
        
        Args:
            report: Markdown report string
            
        Returns:
            Path to saved report
        """
        self.lint_report_file.write_text(report, encoding="utf-8")
        print(f"\nSaved: {self.lint_report_file.relative_to(self.repo_root)}")
        return self.lint_report_file
    
    def append_log(self, entry: str):
        """Prepend new entry to wiki/log.md."""
        existing = read_file(self.log_file)
        write_file(self.log_file, entry.strip() + "\n\n" + existing)


def write_file(path: Path, content: str):
    """Create parent directories if needed and write content to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
