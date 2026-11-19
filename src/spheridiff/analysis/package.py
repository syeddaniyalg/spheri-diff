import shutil
from pathlib import Path


def package_runs(runs_root, output):
    target = Path(output)
    archive = shutil.make_archive(str(target.with_suffix("")), "zip", root_dir=runs_root)
    return Path(archive)
