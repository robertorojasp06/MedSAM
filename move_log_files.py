import shutil
import argparse
from tqdm import tqdm
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="""Move log files to corresponding
        destination folders. Make sure the number of log
        files and destination folders is the same. The
        correspondence is due to filename sorting.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="""Path to the directory containing the output
        folders."""
    )
    parser.add_argument(
        '--path_to_source',
        type=str,
        default=Path.cwd(),
        help="""Path to the folder containing the .log
        files."""
    )
    parser.add_argument(
        "--delete",
        action='store_true',
        help="Add this flag delete log files after copying."
    )
    args = parser.parse_args()
    paths_to_logs = list(Path(args.path_to_source).glob("*.log"))
    paths_to_logs = sorted(
        paths_to_logs,
        key= lambda x: int(Path(x).stem.split('-')[-1])
    )
    paths_to_folders = [
        path
        for path in Path(args.path_to_output).glob('*')
        if path.is_dir()
    ]
    assert len(paths_to_folders) == len(paths_to_logs), f"{len(paths_to_folders)} folders, {len(paths_to_logs)} log files. Must be the same number."
    for idx, path in enumerate(sorted(paths_to_folders)):
        shutil.copy2(
            paths_to_logs[idx],
            path / paths_to_logs[idx].name
        )
        if paths_to_logs[idx].exists() and args.delete:
            paths_to_logs[idx].unlink()


if __name__ == "__main__":
    main()
