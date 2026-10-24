import csv

from nba_edge import config


def test_paths_exist():
    assert config.DATA_RAW.is_dir()
    assert config.DATA_PROCESSED.is_dir()


def test_bet_log_has_expected_header():
    with config.BET_LOG.open() as f:
        header = next(csv.reader(f))
    assert header[:4] == ["date", "player", "team", "opponent"]
