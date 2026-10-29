"""Parse raw league game logs into a tidy player-game table."""

import polars as pl

COLUMNS = {
    "SEASON_ID": "season_id",
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "MATCHUP": "matchup",
    "MIN": "minutes",
    "PTS": "pts",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "FGM": "fgm",
    "FGA": "fga",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FTM": "ftm",
    "FTA": "fta",
    "PF": "pf",
    "PLUS_MINUS": "plus_minus",
}


def parse_league_gamelog(payload: dict) -> pl.DataFrame:
    result = payload["resultSets"][0]
    df = pl.DataFrame(result["rowSet"], schema=result["headers"], orient="row")
    return (
        df.select(list(COLUMNS))
        .rename(COLUMNS)
        .with_columns(
            pl.col("game_date").str.to_date(),
            pl.col("matchup").str.contains("@").not_().alias("is_home"),
            pl.col("matchup").str.slice(-3).alias("opponent"),
        )
    )
