import csv

from nba_ladders import config


def test_paths_exist():
    assert config.DATA_RAW.is_dir()
    assert config.DATA_PROCESSED.is_dir()


def test_bet_log_has_expected_header():
    with config.BET_LOG.open() as f:
        header = next(csv.reader(f))
    assert header[:4] == ["date", "player", "team", "opponent"]


def test_bet_log_scenarios_are_known_labels():
    with config.BET_LOG.open() as f:
        scenarios = {row["scenario"] for row in csv.DictReader(f)}
    assert scenarios - {""} <= config.SCENARIOS
