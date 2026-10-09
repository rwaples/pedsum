"""``python -m pedsum``: the same entry point as the ``pedsum`` command."""

import sys

from pedsum.cli import main

if __name__ == "__main__":
    sys.exit(main())
