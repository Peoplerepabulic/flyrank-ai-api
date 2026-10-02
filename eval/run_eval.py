#!/usr/bin/env python3
"""Eval: classify each case in eval/cases.json against the REAL provider.

Usage (from the project root):
    python eval/run_eval.py

Prints per-case PASS/FAIL and a final PASS RATE line. Honors LLM_PROVIDER and
friends via the normal provider_from_env() path. Exits non-zero if any case fails.
"""

import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.classifier import ClassificationError, classify_message  # noqa: E402
from app.providers import ProviderError, provider_from_env  # noqa: E402

# Seconds to wait between cases; the free pollinations endpoint rate-limits
# aggressively per egress IP. Override with EVAL_DELAY_S (0 to disable).
DELAY_S = float(os.environ.get("EVAL_DELAY_S", "20"))


def main() -> int:
    cases_path = os.path.join(ROOT, "eval", "cases.json")
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    provider = provider_from_env()
    print(f"provider={provider.name} model={provider.model} cases={len(cases)}\n")

    passed = 0
    for case in cases:
        cid = case.get("id", "?")
        expected = case["expected_category"]
        try:
            result, _prompt, _raw = classify_message(case["message"], provider)
            ok = result.category == expected
            status = "PASS" if ok else "FAIL"
            detail = f"got={result.category} conf={result.confidence:.2f}"
        except (ProviderError, ClassificationError, Exception) as exc:  # noqa: BLE001
            ok, status, detail = False, "FAIL", f"error: {exc}"
            traceback.print_exc()
        if ok:
            passed += 1
        print(f"[{status}] {cid}: expected={expected} {detail}", flush=True)
        print(f"       msg={case['message'][:80]}", flush=True)
        if DELAY_S > 0 and case is not cases[-1]:
            time.sleep(DELAY_S)

    total = len(cases)
    rate = 100.0 * passed / total if total else 0.0
    print(f"\nPASS RATE: {passed}/{total} ({rate:.1f}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
