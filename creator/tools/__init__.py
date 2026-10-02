"""Nupen's tools (owner, 2 Oct 2026: shells, scratchpads, bash, monitors - everything Claude Code can run, the AI should run too).

Each module is small and loaded on demand; this package imports none of them. Reach them by name through creator.registry
(`registry.get("tools")` -> creator.tools.toolbox) or import the module you need. policy.py is the single gate."""
MODULES = ("policy", "shell", "jobs", "scratch", "files", "toolbox")
