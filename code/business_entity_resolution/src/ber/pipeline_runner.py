"""Master Pipeline Runner with tqdm progress tracking.

Executes all 6 stages of the Business Entity Resolution pipeline:
  [1/6] Normalise sources & learn transliteration
  [2/6] Candidate generation & pair features (train)
  [3/6] Candidate generation & pair features (test)
  [4/6] Stage-1 cross-fitted pairwise models (5 folds)
  [5/6] Stage-2 group reranking model & decision policy
  [6/6] Test inference & final submission generation
  [Validation] Validate output format against competition rules
"""
import argparse
import os
import subprocess
import sys
import time
from tqdm import tqdm


def run_command(cmd, desc):
    print(f"\n{'='*70}\n>>> {desc}\n{'='*70}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f">>> Completed in {elapsed:.1f}s\n", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="End-to-end BER Pipeline with tqdm progress tracking")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--work-dir", default="work", help="Path to work/cache directory")
    parser.add_argument("--out-dir", default="output", help="Path to output directory")
    parser.add_argument("--tag", default="final", help="Experiment / model tag")
    parser.add_argument("--name-block", action="store_true", default=True, help="Enable exact-name-block channel")
    parser.add_argument("--no-name-block", dest="name_block", action="store_false")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    py = sys.executable
    nb_flag = ["--name-block"] if args.name_block else []

    pipeline_stages = [
        (
            "Step 1/6: Normalise sources & learn transliteration",
            [py, "-m", "ber.prepare", "--data-dir", args.data_dir, "--work-dir", args.work_dir]
        ),
        (
            "Step 2/6: Candidate generation & pair features (train)",
            [py, "-m", "ber.build_pairs", "--split", "train", "--data-dir", args.data_dir, "--work-dir", args.work_dir] + nb_flag
        ),
        (
            "Step 3/6: Candidate generation & pair features (test)",
            [py, "-m", "ber.build_pairs", "--split", "test", "--data-dir", args.data_dir, "--work-dir", args.work_dir] + nb_flag
        ),
        (
            "Step 4/6: Stage-1 cross-fitting (5 folds) + OOF",
            [py, "-m", "ber.cv", "stage1", "--tag", args.tag, "--pairs", os.path.join(args.work_dir, "pairs", "train"),
             "--work-dir", args.work_dir, "--data-dir", args.data_dir]
        ),
        (
            "Step 5/6: Stage-2 group reranker & decision policy",
            [py, "-m", "ber.cv", "stage2", "--tag", args.tag, "--pairs", os.path.join(args.work_dir, "pairs", "train"),
             "--work-dir", args.work_dir, "--data-dir", args.data_dir]
        ),
        (
            "Step 6/6: Test inference & submission generation",
            [py, "-m", "ber.predict", "--models", os.path.join(args.work_dir, "models", args.tag),
             "--pairs", os.path.join(args.work_dir, "pairs", "test"), "--out", args.out_dir,
             "--work-dir", args.work_dir, "--data-dir", args.data_dir]
        ),
    ]

    total_t0 = time.time()
    print("\n" + "=" * 70)
    print("STARTING BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Data directory:   {args.data_dir}")
    print(f"Work directory:   {args.work_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Model tag:        {args.tag}")
    print("=" * 70 + "\n")

    with tqdm(pipeline_stages, desc="Total Pipeline Progress", unit="stage", ncols=90) as pbar:
        for desc, cmd in pbar:
            pbar.set_postfix_str(desc.split(":")[0])
            run_command(cmd, desc)

    # Submission validation step
    val_script = os.path.join(args.data_dir, "..", "utils", "validate_submission.py")
    if os.path.exists(val_script):
        val_cmd = [
            py, val_script,
            "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
            "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
            "--test-dir", os.path.join(args.data_dir, "test"),
        ]
        print("\n" + "=" * 70)
        print(">>> Validating submission files against competition rules...")
        print("=" * 70)
        subprocess.run(val_cmd, check=False)

    total_time = time.time() - total_t0
    print("\n" + "=" * 70)
    print(f"✓ ALL OPERATIONS COMPLETED SUCCESSFULLY in {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Outputs generated in {args.out_dir}:")
    print(f"  1. {os.path.join(args.out_dir, 'matching_results.tsv')}")
    print(f"  2. {os.path.join(args.out_dir, 'candidate_pairs.tsv')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
