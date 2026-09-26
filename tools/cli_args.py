"""Shared argument-error contract for the public Python commands (including subparsers)."""
import argparse
import json
import sys


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f'{self.prog}: {message}', file=sys.stderr)
        if '--json' in sys.argv[1:]:
            print(json.dumps({'error': message}))
        raise SystemExit(1)
