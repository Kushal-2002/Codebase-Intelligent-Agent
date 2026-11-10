"""
Clone a GitHub repository and index it into its own workspace folder.

    workspaces/<owner>__<repo>/
        <repo>/          shallow clone of the source
        chroma/          vector index
        data/            call graph + function lookup pickles
        meta.json        url, commit, stats, indexed_at

The cloned code is only parsed, never built or executed.
"""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime

from indexer import index_repository
from tools import reset_index

WORKSPACES_DIR = "workspaces"

GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:/.*)?$"
)


def parse_github_url(url):

    # accepts https://github.com/owner/repo, .git suffix, or a /tree/... link

    match = GITHUB_URL.match(url.strip())

    if not match:
        raise ValueError("Enter a GitHub repository link like https://github.com/owner/repo")

    owner, repo = match.groups()

    if repo in (".", ".."):
        raise ValueError("That does not look like a repository name")

    return owner, repo


def workspace_paths(owner, repo):

    root = os.path.join(WORKSPACES_DIR, f"{owner}__{repo}")

    return {
        "root": root,
        "source": os.path.join(root, repo),
        "chroma": os.path.join(root, "chroma"),
        "data": os.path.join(root, "data"),
        "meta": os.path.join(root, "meta.json"),
    }


def load_meta(paths):

    try:
        with open(paths["meta"]) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def clone(url, dest):

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # fail instead of asking for a password

    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, dest],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"git clone failed: {message} (is the repository public?)")

    commit = subprocess.run(
        ["git", "-C", dest, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    return commit


def prepare_repository(url, refresh=False, progress=None):

    # returns a workspace dict ready for tools.load_index(chroma, data)

    def report(message):
        if progress:
            progress(message)

    owner, repo = parse_github_url(url)
    paths = workspace_paths(owner, repo)

    meta = load_meta(paths)

    if meta and not refresh:
        report("Using the existing index")
        return {**paths, "meta": meta}

    # only ever delete inside workspaces/
    workspaces_root = os.path.abspath(WORKSPACES_DIR)

    if not os.path.abspath(paths["root"]).startswith(workspaces_root + os.sep):
        raise RuntimeError("Refusing to write outside the workspaces folder")

    if os.path.exists(paths["root"]):
        reset_index()  # the open index may be this one; release it before deleting
        shutil.rmtree(paths["root"])

    os.makedirs(paths["root"])

    clone_url = f"https://github.com/{owner}/{repo}.git"

    try:
        report(f"Cloning {owner}/{repo}")
        commit = clone(clone_url, paths["source"])

        stats = index_repository(paths["source"], paths["chroma"], paths["data"], progress=report)

        if stats["functions"] == 0:
            raise RuntimeError("No C/C++ functions found. Only C and C++ repositories are supported for now.")

    except Exception:
        # don't leave a half-built workspace that would show up later
        reset_index()
        shutil.rmtree(paths["root"], ignore_errors=True)
        raise

    meta = {
        "name": f"{owner}/{repo}",
        "url": f"https://github.com/{owner}/{repo}",
        "commit": commit,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
        **stats,
    }

    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    return {**paths, "meta": meta}


def list_workspaces():

    found = []

    if not os.path.isdir(WORKSPACES_DIR):
        return found

    for entry in sorted(os.listdir(WORKSPACES_DIR)):
        if "__" not in entry:
            continue

        owner, repo = entry.split("__", 1)
        paths = workspace_paths(owner, repo)
        meta = load_meta(paths)

        if meta:
            found.append({**paths, "meta": meta})

    return found
