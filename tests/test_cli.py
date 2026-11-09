import pytest

from pm4bench import __version__
from pm4bench.cli import build_parser


def test_version_flag_prints_the_release(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"pm4bench {__version__}"
