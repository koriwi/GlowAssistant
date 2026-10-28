import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from prototype import MODEL, call_decider, error_message, setting
from numeric_values import INSTRUCTIONS, decode_numeric as decode_probe, finite_number, numeric_request as probe_request




async def run():
    parser = argparse.ArgumentParser(description="Isolated Jev numeric extraction experiment; no HA calls.")
    parser.add_argument("prompt")
    parser.add_argument("--parameter", default="brightness")
    parser.add_argument("--current", type=float)
    parser.add_argument("--min", dest="minimum", type=float, default=0)
    parser.add_argument("--max", dest="maximum", type=float, default=100)
    parser.add_argument("--unit", default="%")
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--method", choices=("score", "choice", "both"), default="both")
    parser.add_argument("--default-step", type=float, help="Step in parameter units for vague increase/decrease requests.")
    parser.add_argument("--expected", type=float, help="Reference value for error reporting; never sent to the model.")
    parser.add_argument("--instructions", default=INSTRUCTIONS, help="Replace the experimental extraction instructions.")
    parser.add_argument("--json", action="store_true", help="Show the request, raw response and decoded estimates.")
    args = parser.parse_args()
    if args.expected is not None and not finite_number(args.expected):
        parser.error("Expected value must be finite.")
    anchors, payload = probe_request(
        args.prompt, args.parameter, args.current, args.minimum, args.maximum,
        args.levels, args.unit, args.method, args.default_step, args.instructions, model=MODEL,
    )
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = setting("OPENROUTER_API_KEY")
    started = perf_counter()
    response = await call_decider(payload, api_key)
    results = decode_probe(response, anchors, args.method, args.expected)
    result = {
        "request": payload, "response": response, "anchors": anchors, "estimates": results,
        "expected": args.expected, "usage": response.get("usage", {}),
        "latency_ms": round((perf_counter() - started) * 1000),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"parameter: {args.parameter} ({args.unit})")
        print(f"current: {args.current if args.current is not None else 'not supplied'}")
        print(f"anchors: {', '.join(f'{value:g}' for value in anchors)}")
        for method, estimate in results.items():
            raw = f", raw score {estimate['raw_score']:.4g}" if method == "score" else ""
            print(f"{method}: {estimate['value']:.4g} {args.unit} (confidence {estimate['confidence']:.3f}{raw})")
            print(f"{method} probability mean: {estimate['weighted_value']:.4g} {args.unit}")
            if args.expected is not None:
                print(f"{method} absolute error: {estimate['absolute_error']:.4g}; mean error: {estimate['weighted_absolute_error']:.4g}")
        if args.expected is not None:
            print(f"expected: {args.expected:g} {args.unit}")
        print(f"latency: {result['latency_ms']} ms; usage: {json.dumps(result['usage'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run()))
    except Exception as error:
        print(f"Error: {error_message(error)}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
