"""Ladder-tail check: blind Sportsbet ROI by price band, as-at minutes variance, and thesis angle.
Run: uv run python -m nba_ladders.ladder_tails_check. Results: docs/market_backtest_findings.md."""
import polars as pl

from nba_ladders.market_backtest import information_test, roi

df = pl.read_parquet('data/processed/market_backtest.parquet')
pg = pl.read_parquet('data/processed/player_games.parquet').sort('player_id', 'game_date')
# as-at variance: last 20 games BEFORE the game (shifted)
v = pg.with_columns(
    pl.col('minutes').shift(1).rolling_std(20, min_samples=10).over('player_id').alias('min_sd'),
    pl.col('minutes').shift(1).rolling_mean(20, min_samples=10).over('player_id').alias('min_mean'),
    ((pl.col('pts') + pl.col('reb')).shift(1).rolling_std(20, min_samples=10) /
     (pl.col('pts') + pl.col('reb')).shift(1).rolling_mean(20, min_samples=10)).over('player_id').alias('pr_cv'),
).select('player_id', 'game_date', 'min_sd', 'min_mean', 'pr_cv')
d = df.join(v, on=['player_id', 'game_date'], how='left').drop_nulls(['min_sd'])
d = d.with_columns(
    pl.col('price').cut([2.0, 3.0, 5.0, 10.0, 20.0], labels=['<2', '2-3', '3-5', '5-10', '10-20', '20+']).alias('band'),
    pl.col('min_sd').qcut(3, labels=['low', 'mid', 'high']).alias('min_var'),
)
print(f'{d.height:,} rungs with as-at variance')
q = d.select(pl.col('min_sd').quantile(0.333).alias('q1'), pl.col('min_sd').quantile(0.667).alias('q2')).row(0)
print('minutes sd tercile cut points: {:.1f} / {:.1f}'.format(*q))
print('\n1) BLIND: bet every Sportsbet rung, by price band')
for b in ['<2', '2-3', '3-5', '5-10', '10-20']:
    x = d.filter(pl.col('band') == b)
    print(f'  {b:6s}', roi(x, n_boot=500), '| implied {:.3f} actual {:.3f}'.format(x['p_book'].mean(), x['hit'].mean()))
print('\n2) BLIND by band x minutes variance (as-at)')
for b in ['3-5', '5-10', '10-20']:
    for mv in ['low', 'mid', 'high']:
        x = d.filter(pl.col('band') == b, pl.col('min_var') == mv)
        r = roi(x, n_boot=500)
        print(f'  {b:6s} {mv:5s} bets {r["bets"]:6d} roi {r["roi"]:+.3f} ci90 {r["roi_ci90"]}')
print('\n   high-variance tails by stat (price >= 5):')
for s in ['pts', 'reb', 'ast', 'pts+reb']:
    x = d.filter(pl.col('price') >= 5, pl.col('min_var') == 'high', pl.col('stat') == s)
    print(f'  {s:8s}', roi(x, n_boot=500))
print('\n3) does the model add info in tails? (price >= 5)')
for mv in ['low', 'mid', 'high']:
    x = d.filter(pl.col('price') >= 5, pl.col('min_var') == mv)
    it = information_test(x)
    print(f'  {mv:5s} n={it["n"]:6d} coef_model {it["coef_model"]:.3f} (se {it["coef_model_se"]:.3f}) | ll book {it["logloss_book_only"]:.5f} -> book+model {it["logloss_book+model"]:.5f}')

print('\n4) thesis spots: blind tails (price >= 3) by angle x variance')
for ang in ['starter_out', 'luck', 'other']:
    for mv in ['low', 'high']:
        x = d.filter(pl.col('price') >= 3, pl.col('angle') == ang, pl.col('min_var') == mv)
        r = roi(x, n_boot=500)
        if r['bets']:
            print(f'  {ang:12s} {mv:5s} bets {r["bets"]:6d} hit {r["hit_rate"]:.3f} avg ${r["avg_price"]:.2f} roi {r["roi"]:+.3f} ci90 {r["roi_ci90"]}')
x = d.filter(pl.col('price') >= 3, pl.col('angle') == 'starter_out', pl.col('stat').is_in(['reb', 'pts+reb']))
print('  starter_out reb & P+R tails (all variance):', roi(x, n_boot=500))
