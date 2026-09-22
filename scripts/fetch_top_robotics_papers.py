#!/usr/bin/env python3
"""Fetch abstracts of the most-cited open-access robotics papers of the past decade.

How the corpus is acquired
--------------------------
Source: the Semantic Scholar Academic Graph API, `paper/search/bulk` endpoint
(https://api.semanticscholar.org/api-docs/graph).
It is free, needs no API key, sorts server side by citation count, and returns
title, authors, venue, year, citation count, abstract and open-access PDF
metadata in one call, so no second scraping step is needed.

Query sent to the API:
  - keyword query `robot*` (matches robot, robots, robotic, robotics, ...)
    against title and abstract;
  - `fieldsOfStudy=Computer Science,Engineering`;
  - `year=<start>-<end>` (default 2016-2026, the past decade);
  - `openAccessPdf` (only papers Semantic Scholar lists an open-access PDF for);
  - `sort=citationCount:desc`.

Each returned paper, taken in descending citation order, is then kept only if
it passes every check below, until `--limit` papers (default 500) are kept:
  1. Open access: it has an open-access PDF URL that is either an arXiv copy
     or has a Semantic Scholar open-access status other than CLOSED (CLOSED
     describes the publisher version, which can coexist with an arXiv
     preprint), and Semantic Scholar serves a non-empty abstract for it.
     Paywalled papers and papers whose abstract the publisher withholds are
     skipped, so no paywalled abstract is ever fetched or stored.
  2. Robotics relevance, the first matching rule decides:
     a. published in a robotics venue (venue name mentions robot/robotics, or is
        ICRA, IROS, RSS, CoRL, RA-L, T-RO, IJRR, Humanoids) -> keep;
     b. its fields of study include a social-science or business field
        (Business, Economics, Psychology, Sociology, Political Science,
        Education, Law, Philosophy, History, Art, Geography) -> drop;
        this removes labour-economics and service-marketing papers about robots;
     c. title mentions a robot term (and is not about "robotic process
        automation", i.e. software bots), and its fields of study include
        Computer Science, Engineering, Materials Science, Physics or
        Mathematics -> keep;
     d. abstract mentions a robot term at least twice and its fields of study
        include Computer Science -> keep;
     otherwise drop.
     Fields of study are Semantic Scholar's `fieldsOfStudy`, falling back to
     `s2FieldsOfStudy` categories when the former is empty.
  3. Deduplication: preprint and published versions that Semantic Scholar keeps
     as separate records are merged by normalised title; the higher-cited record
     wins (it always comes first because results are citation sorted).

Outputs (written to `--out-dir`, default `corpus/`):
  - `abstracts.md`: one section per paper with title, authors, venue, year,
    citation count and abstract (inline JATS/HTML tags stripped, whitespace
    collapsed), under a header recording this provenance and
    the date the pull was run.
  - `manifest.csv`: one row per paper with rank, Semantic Scholar paper ID, DOI,
    arXiv ID, title, authors, year, venue, citation count, open-access status
    and a link to the open-access source (the arXiv abstract page when the paper
    has an arXiv copy, otherwise the open-access PDF URL).
Citation counts are a snapshot as of the pull date.

Usage
-----
    pip install -r requirements.txt
    python scripts/fetch_top_robotics_papers.py              # full 500-paper pull
    python scripts/fetch_top_robotics_papers.py --limit 5 --out-dir /tmp/smoke

No API key is needed.
If `S2_API_KEY` is set in the environment it is sent as `x-api-key` for a higher
rate limit, but it is never required.
Unauthenticated requests share a global rate limit, so HTTP 429 and 5xx
responses are retried with exponential backoff that honours `Retry-After`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
QUERY = "robot*"
API_FIELDS_OF_STUDY = "Computer Science,Engineering"
FIELDS = (
    "paperId,title,abstract,authors,year,venue,publicationVenue,journal,"
    "citationCount,externalIds,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy"
)
DEFAULT_LIMIT = 500
DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2026
PAGE_PAUSE_SECONDS = 1.5
MAX_RETRIES = 8
REQUEST_TIMEOUT_SECONDS = 120

ROBOT_TERM = re.compile(r"\brobot", re.IGNORECASE)
# JATS/HTML inline markup some publishers leave in abstracts, e.g. <italic>, <jats:p>.
MARKUP_TAG = re.compile(r"</?(?:(?:jats|mml):[\w.-]+(?:\s[^<>]*)?|[a-z][a-z0-9-]*)/?>")
ARXIV_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([^/?#]+?)(?:v\d+)?(?:\.pdf)?(?:[?#]|$)", re.IGNORECASE
)
NOT_ROBOTICS_TITLE = re.compile(r"robotic process automation", re.IGNORECASE)
ROBOTICS_VENUE = re.compile(
    r"robot"
    r"|\b(ICRA|IROS|RSS|CoRL|RA-?L|T-?RO|IJRR|Humanoids)\b"
    r"|Intelligent Robots and Systems",
    re.IGNORECASE,
)
SOCIAL_SCIENCE_FIELDS = {
    "Business",
    "Economics",
    "Psychology",
    "Sociology",
    "Political Science",
    "Education",
    "Law",
    "Philosophy",
    "History",
    "Art",
    "Geography",
}
TECHNICAL_FIELDS = {
    "Computer Science",
    "Engineering",
    "Materials Science",
    "Physics",
    "Mathematics",
}

MANIFEST_COLUMNS = [
    "rank",
    "s2_paper_id",
    "doi",
    "arxiv_id",
    "title",
    "authors",
    "year",
    "venue",
    "citation_count",
    "open_access_status",
    "open_access_url",
]


def request_page(session: requests.Session, params: dict) -> dict:
    """GET one page of bulk-search results, retrying rate limits and server errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except (requests.ConnectionError, requests.Timeout) as error:
            wait = 2**attempt
            print(f"request failed ({error}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 or response.status_code >= 500:
            retry_after = response.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else 2**attempt
            print(f"HTTP {response.status_code}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        response.raise_for_status()
    raise RuntimeError(f"giving up after {MAX_RETRIES} attempts against {API_URL}")


def iter_candidates(start_year: int, end_year: int) -> Iterator[dict]:
    """Yield search results in descending citation order across all pages."""
    session = requests.Session()
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        session.headers["x-api-key"] = api_key
    params = {
        "query": QUERY,
        "fieldsOfStudy": API_FIELDS_OF_STUDY,
        "year": f"{start_year}-{end_year}",
        "openAccessPdf": "",
        "sort": "citationCount:desc",
        "fields": FIELDS,
    }
    while True:
        page = request_page(session, params)
        yield from page.get("data") or []
        token = page.get("token")
        if not token:
            return
        params["token"] = token
        time.sleep(PAGE_PAUSE_SECONDS)


def fields_of_study(paper: dict) -> set[str]:
    fields = set(paper.get("fieldsOfStudy") or [])
    if not fields:
        fields = {
            f["category"]
            for f in paper.get("s2FieldsOfStudy") or []
            if f.get("category")
        }
    return fields


def venue_names(paper: dict) -> list[str]:
    names = [paper.get("venue") or ""]
    names.append((paper.get("publicationVenue") or {}).get("name") or "")
    names.append((paper.get("journal") or {}).get("name") or "")
    return [name.strip() for name in names if name and name.strip()]


def arxiv_id(paper: dict) -> str:
    """The paper's arXiv ID, from its external IDs or its open-access URL."""
    known = (paper.get("externalIds") or {}).get("ArXiv")
    if known:
        return known
    match = ARXIV_URL.search((paper.get("openAccessPdf") or {}).get("url") or "")
    return match.group(1) if match else ""


def is_open_access(paper: dict) -> bool:
    pdf = paper.get("openAccessPdf") or {}
    if not pdf.get("url") or not clean_abstract(paper.get("abstract")):
        return False
    return pdf.get("status") != "CLOSED" or bool(arxiv_id(paper))


def is_robotics(paper: dict) -> bool:
    if any(ROBOTICS_VENUE.search(name) for name in venue_names(paper)):
        return True
    fields = fields_of_study(paper)
    if fields & SOCIAL_SCIENCE_FIELDS:
        return False
    title = paper.get("title") or ""
    if ROBOT_TERM.search(title) and not NOT_ROBOTICS_TITLE.search(title):
        return bool(fields & TECHNICAL_FIELDS)
    abstract = paper.get("abstract") or ""
    return len(ROBOT_TERM.findall(abstract)) >= 2 and "Computer Science" in fields


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def select_papers(candidates: Iterable[dict], limit: int) -> list[dict]:
    """Keep the first `limit` open-access robotics papers, dropping duplicate titles."""
    selected: list[dict] = []
    seen_titles: set[str] = set()
    for paper in candidates:
        if (
            not paper.get("title")
            or not is_open_access(paper)
            or not is_robotics(paper)
        ):
            continue
        key = normalized_title(paper["title"])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        selected.append(paper)
        if len(selected) >= limit:
            break
    return selected


def clean(text: str | None) -> str:
    """Collapse whitespace so fetched text cannot break the markdown structure."""
    return re.sub(r"\s+", " ", text or "").strip()


def clean_abstract(text: str | None) -> str:
    """Strip inline markup tags, keeping their text, then collapse whitespace."""
    return clean(MARKUP_TAG.sub("", text or ""))


def display_venue(paper: dict) -> str:
    names = venue_names(paper)
    if names:
        return clean(names[0])
    if arxiv_id(paper):
        return "arXiv"
    return "Unknown venue"


def author_list(paper: dict) -> str:
    names = [clean(author.get("name")) for author in paper.get("authors") or []]
    return ", ".join(name for name in names if name) or "Unknown authors"


def open_access_url(paper: dict) -> str:
    paper_arxiv_id = arxiv_id(paper)
    if paper_arxiv_id:
        return f"https://arxiv.org/abs/{paper_arxiv_id}"
    return (paper.get("openAccessPdf") or {}).get("url") or ""


def render_markdown(
    papers: list[dict], pulled_on: str, start_year: int, end_year: int, limit: int
) -> str:
    lines = [
        "# Most-cited open-access robotics papers, abstracts",
        "",
        "<!-- Generated by scripts/fetch_top_robotics_papers.py. Do not edit by hand; re-run the script instead. -->",
        "",
        "## How this corpus was acquired",
        "",
        f"- Pulled on: {pulled_on} (UTC).",
        "- Source: Semantic Scholar Academic Graph API, `paper/search/bulk` endpoint (https://api.semanticscholar.org/api-docs/graph), queried without an API key.",
        f"- Query: keyword `{QUERY}` over title and abstract, `fieldsOfStudy={API_FIELDS_OF_STUDY}`, `year={start_year}-{end_year}`, only papers with an open-access PDF, sorted by citation count descending.",
        "- Open access only: a paper is kept only if Semantic Scholar lists an open-access PDF for it that is an arXiv copy or has an open-access status other than CLOSED, and serves its abstract; paywalled papers and withheld abstracts are skipped.",
        "- Robotics relevance: kept if published in a robotics venue (ICRA, IROS, RSS, CoRL, RA-L, T-RO, IJRR, Humanoids, or any venue named after robotics); otherwise dropped if tagged with a social-science or business field; otherwise kept if the title mentions a robot term (excluding robotic process automation) and the paper is tagged with a technical field, or if the abstract mentions a robot term at least twice and the paper is tagged Computer Science.",
        "- Deduplication: records with the same normalised title are merged, keeping the higher-cited one.",
        f"- Selection: the top {limit} papers passing these checks, in descending citation order.",
        f"- Citation counts are a snapshot as of {pulled_on}.",
        "- `manifest.csv` next to this file lists every paper with its Semantic Scholar ID, DOI, arXiv ID and open-access link.",
        "",
        f"Papers in this file: {len(papers)}.",
        "",
    ]
    for rank, paper in enumerate(papers, start=1):
        lines += [
            f"## {rank}. {clean(paper['title'])}",
            "",
            f"- Authors: {author_list(paper)}",
            f"- Venue: {display_venue(paper)}",
            f"- Year: {paper.get('year') or 'Unknown'}",
            f"- Citations: {paper.get('citationCount') or 0}",
            f"- Open access: {open_access_url(paper)}",
            "",
            clean_abstract(paper["abstract"]),
            "",
        ]
    return "\n".join(lines)


def manifest_rows(papers: list[dict]) -> Iterator[dict]:
    for rank, paper in enumerate(papers, start=1):
        ids = paper.get("externalIds") or {}
        yield {
            "rank": rank,
            "s2_paper_id": paper.get("paperId") or "",
            "doi": ids.get("DOI") or "",
            "arxiv_id": arxiv_id(paper),
            "title": clean(paper["title"]),
            "authors": "; ".join(
                clean(a.get("name"))
                for a in paper.get("authors") or []
                if a.get("name")
            ),
            "year": paper.get("year") or "",
            "venue": display_venue(paper),
            "citation_count": paper.get("citationCount") or 0,
            "open_access_status": (paper.get("openAccessPdf") or {}).get("status")
            or "",
            "open_access_url": open_access_url(paper),
        }


def write_outputs(
    papers: list[dict],
    out_dir: Path,
    pulled_on: str,
    start_year: int,
    end_year: int,
    limit: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(papers, pulled_on, start_year, end_year, limit)
    (out_dir / "abstracts.md").write_text(markdown, encoding="utf-8")
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows(papers))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="number of papers to keep (default 500)",
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "corpus",
        help="directory for abstracts.md and manifest.csv (default: corpus/ at the repo root)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pulled_on = dt.datetime.now(dt.UTC).date().isoformat()
    papers = select_papers(iter_candidates(args.start_year, args.end_year), args.limit)
    if len(papers) < args.limit:
        print(
            f"warning: only {len(papers)} papers matched, fewer than --limit {args.limit}",
            file=sys.stderr,
        )
    write_outputs(
        papers, args.out_dir, pulled_on, args.start_year, args.end_year, args.limit
    )
    print(
        f"wrote {len(papers)} papers to {args.out_dir / 'abstracts.md'} and {args.out_dir / 'manifest.csv'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
