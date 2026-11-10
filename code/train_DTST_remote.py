"""DT-ST B2/IRRG adaptation from source weights, with optional GSD selection."""
import argparse
import json
from pathlib import Path
import sys

from scripts.dtst_remote import validate_settings
from train_SND_remote import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dtst-config", type=Path, default=Path(__file__).parent / "configs/dtst_sfuda/code_b2_irrg.json")
    parser.add_argument("--stop-updates", type=int)
    args, remaining = parser.parse_known_args()
    settings = validate_settings(json.loads(args.dtst_config.read_text(encoding="utf-8-sig")))
    if args.stop_updates is not None:
        if args.stop_updates < 1:
            parser.error("--stop-updates must be positive")
        remaining += ["--gsd-stop-updates", str(args.stop_updates)]
    sys.argv = [sys.argv[0], "warm", *remaining]
    main(dtst_settings=settings)
