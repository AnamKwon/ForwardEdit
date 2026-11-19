"""Inference-only project MCP launcher for restricted stdio worker contexts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_stdio_compat import run_stdio_compatible
from weight_space_lab.server import build_server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()
    run_stdio_compatible(build_server(Path(args.workspace).resolve()))
