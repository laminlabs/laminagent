import os

import nox
from laminci.nox import install_lamindb, run_pre_commit

# we'd like to aggregate coverage information across sessions
# and for this the code needs to be located in the same
# directory in every github action runner
# this also allows to break out an installation section
nox.options.default_venv_backend = "none"

IS_PR = os.getenv("GITHUB_EVENT_NAME") != "push"


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
@nox.parametrize("group", ["unit", "tasks"])
def test(session: nox.Session, group: str) -> None:
    branch = (
        "main" if IS_PR else "main"
    )  # point to "main" for PRs, to "release" for main
    install_lamindb(session, branch=branch)
    if group == "tasks":
        session.run("npm", "install", "-g", "@openai/codex", external=True)
        if os.environ.get("OPENAI_API_KEY"):
            session.run(
                "bash",
                "-c",
                'codex login --with-api-key <<< "$OPENAI_API_KEY"',
                external=True,
            )
        session.run(
            "uv",
            "pip",
            "install",
            "--system",
            "matplotlib",
            "seaborn",
            "pandas",
            "numpy",
            "openpyxl",
            "requests",
            "scanpy",
            external=True,
        )
        session.run("npm", "install", "-g", "@anthropic-ai/claude-code", external=True)
        # ANTHROPIC_API_KEY in the environment is sufficient auth for headless mode —
        # unlike codex, no separate non-interactive login step is needed.

        coverage_args = []
    else:
        coverage_args = [
            "--cov=laminagent",
            "--cov-append",
            "--cov-report=term-missing",
        ]
    session.run("pytest", "-s", f"tests/{group}", *coverage_args)
