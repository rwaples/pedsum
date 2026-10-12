#!/usr/bin/env python3
"""Run the pedsum CLI from an uninstalled checkout: ``python pedigree_summary.py summarize|validate ...``."""

import sys

from pedsum.cli import main

if __name__ == "__main__":
    sys.exit(main())
