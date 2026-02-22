"""
Download and cache TIMIT, LibriSpeech, VCTK, and Common Voice via HuggingFace.

Usage:
    python data/scripts/download_datasets.py --datasets librispeech vctk common_voice
    python data/scripts/download_datasets.py --datasets timit --timit-dir /path/to/timit
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["timit", "librispeech", "vctk", "common_voice", "l2_arctic"],
        default=["librispeech", "vctk"],
    )
    parser.add_argument("--timit-dir", default=None, help="Path to locally downloaded TIMIT.")
    parser.add_argument("--output-dir", default="data/raw", help="Where to cache datasets.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if "librispeech" in args.datasets:
        logger.info("Downloading LibriSpeech …")
        from datasets import load_dataset
        for split in ["train.clean.100", "validation.clean", "test.clean"]:
            logger.info("  split: %s", split)
            ds = load_dataset("librispeech_asr", split=split, trust_remote_code=True)
            ds.save_to_disk(str(output_dir / "librispeech" / split.replace(".", "_")))
        logger.info("LibriSpeech saved to %s", output_dir / "librispeech")

    if "vctk" in args.datasets:
        logger.info("Downloading VCTK …")
        from datasets import load_dataset
        ds = load_dataset("vctk", trust_remote_code=True)
        ds.save_to_disk(str(output_dir / "vctk"))
        logger.info("VCTK saved to %s", output_dir / "vctk")

    if "common_voice" in args.datasets:
        logger.info("Downloading Common Voice (en) …")
        from datasets import load_dataset
        ds = load_dataset("mozilla-foundation/common_voice_16_1", "en", trust_remote_code=True)
        ds.save_to_disk(str(output_dir / "common_voice"))
        logger.info("Common Voice saved to %s", output_dir / "common_voice")

    if "timit" in args.datasets:
        if args.timit_dir:
            logger.info("Loading TIMIT from %s …", args.timit_dir)
            from datasets import load_dataset
            ds = load_dataset("timit_asr", data_dir=args.timit_dir, trust_remote_code=True)
            ds.save_to_disk(str(output_dir / "timit"))
        else:
            logger.warning(
                "TIMIT requires an LDC licence. Download the corpus from https://catalog.ldc.upenn.edu/LDC93S1 "
                "and re-run with --timit-dir /path/to/TIMIT"
            )

    if "l2_arctic" in args.datasets:
        logger.info(
            "L2-ARCTIC must be downloaded manually from https://psi.engr.tamu.edu/l2-arctic-corpus/ "
            "Place extracted files in data/raw/l2_arctic/"
        )


if __name__ == "__main__":
    main()
