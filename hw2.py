#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()
    
def extract_score(content: Any) -> float | None:
    """Robustly extract a score in [0, 1] from the agent's final message.

    Tries, in order:
    1. An explicit "SCORE: X" / "score is X" / "rating: X" pattern.
    2. A standalone number on the last non-empty line.
    3. The last number in [0, 1] found anywhere in the text.
    Returns None if no valid score is found.
    """
    text = str(getattr(content, "content", content))

    # 1. Explicit "score/rating/confidence: X"
    m = re.search(
        r"(?:score|rating|confidence)\s*(?:is|:|=)?\s*([01](?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            return val

    # 2. Standalone number on the last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        nums = re.findall(r"\b([01](?:\.\d+)?)\b", lines[-1])
        if len(nums) == 1:
            val = float(nums[0])
            if 0.0 <= val <= 1.0:
                return val

    # 3. Last number in [0, 1] found anywhere
    nums = re.findall(r"\b([01](?:\.\d+)?)\b", text)
    if nums:
        val = float(nums[-1])
        if 0.0 <= val <= 1.0:
            return val

    return None


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    ### YOUR CODE HERE
    from langchain_deepseek import ChatDeepSeek
    from langchain.agents import create_agent

    model = ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0,
    )

    system_prompt = """You are a CV verification agent for KYC compliance.

Your job: given a candidate's CV text, verify whether its claims agree with the
candidate's LinkedIn and Facebook profiles using the provided SocialGraph tools.

IMPORTANT RULES:
1. LinkedIn is the primary source. For every test candidate, Facebook agrees with LinkedIn.
2. Discrepancies can ONLY be in these fields: name, city, jobs (company, title,
   seniority, start and end years), education (degree, school, field, graduation
   year), and skills.
3. Job descriptions, headline, and hometown are NEVER the source of a discrepancy.
4. Wording differences are NOT discrepancies, e.g. "Bachelor of Science" vs "BSc",
   "UI/UX Design" vs "UI/UX", "Senior Engineer" for an Engineer role with seniority
   senior, or listing fewer skills than the profile.
5. Many candidates share the same name. You MUST verify the RIGHT person by narrowing
   searches with location (and industry on LinkedIn).
6. If the CV contains at least one false claim (inflated job title, shifted employment
   or graduation years, upgraded degree, fake school or employer, wrong location, or
   a skill the candidate does not have), the CV has a discrepancy.

OUTPUT FORMAT (STRICT):
On the very last line of your response, output exactly this format and nothing else:
SCORE: <number>
where <number> is between 0.0 and 1.0 (e.g., SCORE: 0.9).
Do not write any other number on that line.
Do not add punctuation or explanation after the number.
"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    ### YOUR CODE HERE
    semaphore = asyncio.Semaphore(3)

    async def score_one(name: str, text: str) -> tuple[str, float | None]:
        async with semaphore:
            try:
                user_message = (
                    f"Verify the following CV against the candidate's LinkedIn "
                    f"and Facebook profiles using the SocialGraph tools.\n\n"
                    f"CV file name: {name}\n"
                    f"CV text:\n{text}\n\n"
                    f"After your investigation, output ONLY a single number "
                    f"between 0 and 1 (1 = fully consistent, 0 = clear discrepancy)."
                )
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_message}]}
                )
                messages = result.get("messages", [])
                if not messages:
                    return name, None
                last = messages[-1]
                content = getattr(last, "content", last)
                score = extract_score(content)                
                return name, score
            except Exception:
                return name, 0.0

    tasks = [score_one(name, text) for name, text in cvs.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
