from pathlib import Path
import sys

from studyrag.retrieval import E5Embedder, search_index


def load_cases(path: Path) -> list[dict[str, str]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty JSON list.")
    for case in cases:
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip()
            for key in ("query", "expected_filename_contains")
        ):
            raise ValueError("Each case needs a query and expected_filename_contains string.")
    return cases


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Check retrieval against your own labeled examples.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="Your labeled JSON cases")
    parser.add_argument("--top-k", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        cases = load_cases(arguments.cases)
        if not arguments.database.is_file():
            raise ValueError("The index database does not exist. Prepare your library first.")
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    embedder = E5Embedder(local_files_only=True)
    failed = False
    for case in cases:
        query = case["query"]
        expected_filename_part = case["expected_filename_contains"].casefold()
        results = search_index(arguments.database, embedder, query, top_k=arguments.top_k)
        top_sources = [result.source_path.name for result in results]
        matched = any(expected_filename_part in name.casefold() for name in top_sources)
        failed = failed or not matched
        print(query)
        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. {result.source_path.name}, page {result.page_number}, score={result.score:.4f}")
        print(f"  expected source in top {arguments.top_k}: {matched}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
import argparse
import json
