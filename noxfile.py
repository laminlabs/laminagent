import nox
from laminci.nox import run_pre_commit, run_pytest

# we'd like to aggregate coverage information across sessions
# and for this the code needs to be located in the same
# directory in every github action runner
# this also allows to break out an installation section
nox.options.default_venv_backend = "none"
from laminci.nox import (
    run,
)


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
def install(session):
    run(session, "uv pip install --system litellm")
    run_pytest(session)
