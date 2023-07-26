import argparse
import sys 

from .create import CreateCommand
from .decode import DecodeCommand

COMMANDS = [
    CreateCommand(), DecodeCommand()
]

__version__ = "0.0.1"

def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="Displays the current version of Memray",
    )

    subparsers = parser.add_subparsers(
        help="Mode of operation",
        dest="command",
        required=True,
    )

    for command in COMMANDS:
        assert command.__class__.__name__.endswith("Command")
        name = command.__class__.__name__[: -len("Command")].lower()
        print(name)
        command_parser = subparsers.add_parser(
            name#, help=command.__doc__, epilog=_EPILOG
        )

        command_parser.set_defaults(entrypoint=command.run)
        command.prepare_parser(command_parser)
    
    return parser

def main(args=None) -> int:
    if args is None:
        args = sys.argv[1:]

    parser = get_argument_parser()
    arg_values = parser.parse_args(args)
    arg_values.entrypoint(arg_values, parser)
    # set_log_level(determine_logging_level_from_verbosity(arg_values.verbose))