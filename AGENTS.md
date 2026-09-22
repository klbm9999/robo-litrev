# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- `corpus/abstracts.md` and `corpus/manifest.csv` are the RAG document set and are generated; never hand-edit them, re-run `python scripts/fetch_top_robotics_papers.py` (selection rules live in its docstring).
- Tests are stdlib `unittest` and run offline: `python -m unittest discover tests`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
