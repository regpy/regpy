import sys
from pathlib import Path

def import_example_package(example_dir):
    example_dir = Path(example_dir).resolve()
    print("Path to folder",example_dir)
    sys.path.insert(0, str(example_dir))