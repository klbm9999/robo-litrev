"""Offline tests for scripts/fetch_top_robotics_papers.py. Run: python -m unittest discover tests"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_top_robotics_papers as fetch


def paper(**overrides):
    record = {
        "paperId": "abc123",
        "title": "Learning Robot Grasping",
        "abstract": "We teach a robot\nto grasp.",
        "authors": [{"name": "Ada Lovelace"}, {"name": "Alan Turing"}],
        "year": 2020,
        "venue": "Some Journal",
        "publicationVenue": None,
        "journal": None,
        "citationCount": 100,
        "externalIds": {"DOI": "10.1/x"},
        "openAccessPdf": {"url": "https://example.org/paper.pdf", "status": "GOLD"},
        "fieldsOfStudy": ["Computer Science"],
        "s2FieldsOfStudy": [],
    }
    record.update(overrides)
    return record


class OpenAccessTest(unittest.TestCase):
    def test_open_status_with_abstract_is_open(self):
        self.assertTrue(fetch.is_open_access(paper()))

    def test_missing_abstract_is_rejected(self):
        self.assertFalse(fetch.is_open_access(paper(abstract=None)))
        self.assertFalse(fetch.is_open_access(paper(abstract="  ")))

    def test_missing_pdf_is_rejected(self):
        self.assertFalse(fetch.is_open_access(paper(openAccessPdf=None)))

    def test_closed_without_arxiv_is_rejected(self):
        self.assertFalse(
            fetch.is_open_access(
                paper(
                    openAccessPdf={
                        "url": "https://pub.example/x.pdf",
                        "status": "CLOSED",
                    }
                )
            )
        )

    def test_closed_with_arxiv_copy_is_open(self):
        self.assertTrue(
            fetch.is_open_access(
                paper(
                    openAccessPdf={
                        "url": "http://arxiv.org/pdf/2302.03648",
                        "status": "CLOSED",
                    }
                )
            )
        )
        self.assertTrue(
            fetch.is_open_access(
                paper(
                    externalIds={"ArXiv": "2307.15818"},
                    openAccessPdf={
                        "url": "https://pub.example/x.pdf",
                        "status": "CLOSED",
                    },
                )
            )
        )


class CleanAbstractTest(unittest.TestCase):
    def test_strips_inline_markup_and_keeps_text(self):
        raw = "<jats:p>A <italic>robust</italic> solver,\n<monospace>GNC</monospace>.</jats:p>"
        self.assertEqual(fetch.clean_abstract(raw), "A robust solver, GNC.")

    def test_keeps_comparisons_that_are_not_tags(self):
        self.assertEqual(
            fetch.clean_abstract("error < 5 and x <y and z> w"),
            "error < 5 and x <y and z> w",
        )

    def test_markup_only_abstract_counts_as_missing(self):
        self.assertFalse(fetch.is_open_access(paper(abstract="<jats:p> </jats:p>")))


class ArxivIdTest(unittest.TestCase):
    def test_prefers_external_id(self):
        self.assertEqual(
            fetch.arxiv_id(paper(externalIds={"ArXiv": "1708.03852"})), "1708.03852"
        )

    def test_parses_url_variants(self):
        for url in [
            "http://arxiv.org/pdf/2302.03648",
            "https://arxiv.org/pdf/2302.03648v2",
            "https://arxiv.org/abs/2302.03648?x=1",
            "https://arxiv.org/pdf/2302.03648.pdf",
        ]:
            self.assertEqual(
                fetch.arxiv_id(paper(openAccessPdf={"url": url})), "2302.03648", url
            )

    def test_non_arxiv_url(self):
        self.assertEqual(fetch.arxiv_id(paper()), "")


class RoboticsRelevanceTest(unittest.TestCase):
    def test_robotics_venue_wins_over_field_tags(self):
        self.assertTrue(
            fetch.is_robotics(
                paper(
                    title="Social robots for education",
                    venue="Science Robotics",
                    fieldsOfStudy=["Psychology"],
                )
            )
        )
        self.assertTrue(
            fetch.is_robotics(
                paper(title="Diffusion policy", venue="Robotics: Science and Systems")
            )
        )
        self.assertTrue(
            fetch.is_robotics(
                paper(
                    title="Code as Policies",
                    venue="IEEE International Conference on Robotics and Automation",
                )
            )
        )

    def test_social_science_papers_about_robots_are_dropped(self):
        self.assertFalse(
            fetch.is_robotics(
                paper(title="Robots and Jobs", fieldsOfStudy=["Economics"])
            )
        )
        self.assertFalse(
            fetch.is_robotics(
                paper(
                    title="Service robots in the frontline",
                    fieldsOfStudy=["Business", "Computer Science"],
                )
            )
        )

    def test_falls_back_to_s2_fields_when_fields_of_study_empty(self):
        record = paper(
            title="Anthropomorphism of service robots",
            fieldsOfStudy=[],
            s2FieldsOfStudy=[{"category": "Business"}],
        )
        self.assertFalse(fetch.is_robotics(record))

    def test_robot_title_needs_technical_field(self):
        self.assertTrue(
            fetch.is_robotics(
                paper(
                    title="Soft Robotic Grippers",
                    fieldsOfStudy=["Materials Science", "Medicine"],
                )
            )
        )
        self.assertFalse(
            fetch.is_robotics(
                paper(
                    title="Robotic knee arthroplasty outcomes",
                    fieldsOfStudy=["Medicine"],
                )
            )
        )

    def test_robotic_process_automation_is_not_robotics(self):
        self.assertFalse(
            fetch.is_robotics(
                paper(
                    title="Robotic Process Automation for Auditing",
                    abstract="Software bots.",
                )
            )
        )

    def test_abstract_mentions_need_computer_science(self):
        abstract = "An embodied model for robot planning evaluated on real robots."
        self.assertTrue(fetch.is_robotics(paper(title="PaLM-E", abstract=abstract)))
        self.assertFalse(
            fetch.is_robotics(
                paper(
                    title="Hydrogel skin",
                    abstract=abstract,
                    fieldsOfStudy=["Materials Science"],
                )
            )
        )
        self.assertFalse(
            fetch.is_robotics(
                paper(title="Deep learning review", abstract="Used in robotics.")
            )
        )


class SelectPapersTest(unittest.TestCase):
    def test_filters_dedupes_and_limits_in_input_order(self):
        candidates = [
            paper(paperId="1", title="Robot A", citationCount=50),
            paper(paperId="2", title="Unrelated vision paper", abstract="Images."),
            paper(paperId="3", title="Robot  a!", citationCount=40),
            paper(paperId="4", title="Robot B", abstract=None),
            paper(paperId="5", title="Robot C", citationCount=30),
            paper(paperId="6", title="Robot D", citationCount=20),
        ]
        selected = fetch.select_papers(candidates, 2)
        self.assertEqual([p["paperId"] for p in selected], ["1", "5"])

    def test_stops_consuming_candidates_at_limit(self):
        consumed = []

        def candidates():
            for index in range(10):
                consumed.append(index)
                yield paper(paperId=str(index), title=f"Robot {index}")

        fetch.select_papers(candidates(), 3)
        self.assertEqual(consumed, [0, 1, 2])


class OutputTest(unittest.TestCase):
    def test_writes_markdown_and_manifest(self):
        papers = [
            paper(
                externalIds={"DOI": "10.1/x", "ArXiv": "2302.03648"},
                venue="",
                journal={"name": "Science Robotics"},
            ),
            paper(paperId="def", title="Robot\ntitle  two", venue="", externalIds={}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fetch.write_outputs(papers, out, "2026-09-22", 2016, 2026, 500)
            markdown = (out / "abstracts.md").read_text(encoding="utf-8")
            with open(out / "manifest.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("## How this corpus was acquired", markdown)
        self.assertIn("Pulled on: 2026-09-22 (UTC).", markdown)
        self.assertIn("`year=2016-2026`", markdown)
        self.assertIn("Papers in this file: 2.", markdown)
        self.assertIn("## 1. Learning Robot Grasping", markdown)
        self.assertIn("- Authors: Ada Lovelace, Alan Turing", markdown)
        self.assertIn("- Venue: Science Robotics", markdown)
        self.assertIn("- Year: 2020", markdown)
        self.assertIn("- Citations: 100", markdown)
        self.assertIn("- Open access: https://arxiv.org/abs/2302.03648", markdown)
        self.assertIn("We teach a robot to grasp.", markdown)
        self.assertIn("## 2. Robot title two", markdown)
        self.assertIn("- Venue: Unknown venue", markdown)
        self.assertIn("- Open access: https://example.org/paper.pdf", markdown)

        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), fetch.MANIFEST_COLUMNS)
        self.assertEqual(rows[0]["rank"], "1")
        self.assertEqual(rows[0]["doi"], "10.1/x")
        self.assertEqual(rows[0]["arxiv_id"], "2302.03648")
        self.assertEqual(rows[0]["authors"], "Ada Lovelace; Alan Turing")
        self.assertEqual(rows[0]["open_access_url"], "https://arxiv.org/abs/2302.03648")
        self.assertEqual(rows[1]["s2_paper_id"], "def")
        self.assertEqual(rows[1]["title"], "Robot title two")


class RequestPageTest(unittest.TestCase):
    def response(self, status, headers=None, body=None):
        result = mock.Mock(status_code=status, headers=headers or {})
        result.json.return_value = body or {}
        return result

    def test_retries_rate_limit_honouring_retry_after(self):
        session = mock.Mock()
        session.get.side_effect = [
            self.response(429, {"Retry-After": "3"}),
            self.response(503),
            self.response(200, body={"data": [1]}),
        ]
        with mock.patch.object(fetch.time, "sleep") as sleep:
            page = fetch.request_page(session, {})
        self.assertEqual(page, {"data": [1]})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [3, 2])

    def test_gives_up_after_max_retries(self):
        session = mock.Mock()
        session.get.return_value = self.response(429)
        with mock.patch.object(fetch.time, "sleep"), self.assertRaises(RuntimeError):
            fetch.request_page(session, {})
        self.assertEqual(session.get.call_count, fetch.MAX_RETRIES)


class PaginationTest(unittest.TestCase):
    def test_follows_continuation_tokens(self):
        pages = [{"data": [1, 2], "token": "t1"}, {"data": [3], "token": None}]
        seen_params = []

        def fake_request(session, params):
            seen_params.append(dict(params))
            return pages[len(seen_params) - 1]

        with (
            mock.patch.object(fetch, "request_page", side_effect=fake_request),
            mock.patch.object(fetch.time, "sleep"),
        ):
            results = list(fetch.iter_candidates(2016, 2026))
        self.assertEqual(results, [1, 2, 3])
        self.assertNotIn("token", seen_params[0])
        self.assertEqual(seen_params[1]["token"], "t1")
        self.assertEqual(seen_params[0]["year"], "2016-2026")
        self.assertEqual(seen_params[0]["sort"], "citationCount:desc")


if __name__ == "__main__":
    unittest.main()
