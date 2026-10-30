"""One-off held-out check: market-anchored blend (Sportsbet + model, per angle/stat) and best-AU-price
line shopping. Run: uv run python -m nba_ladders.market_blend_check. Results: docs/market_backtest_findings.md."""
import random

import numpy as np
import polars as pl
from scipy.optimize import minimize

from nba_ladders.market_backtest import _logit, roi

df = pl.read_parquet('data/processed/market_backtest.parquet')
days = sorted(df['game_date'].unique().to_list()); random.Random(1).shuffle(days)
tune_days = set(days[: len(days)//2])
def X(d):
    lb, lm = _logit(d['p_book'].to_numpy()), _logit(d['p_model'].to_numpy())
    cols = [np.ones(d.height), lb, lm]
    for ang in ['starter_out', 'luck']:
        a = (d['angle'] == ang).to_numpy().astype(float); cols += [a, a * lm]
    for s in ['reb', 'ast', 'pts+reb']:
        a = (d['stat'] == s).to_numpy().astype(float); cols += [a, a * lb, a * lm]
    return np.column_stack(cols)
tune = df.filter(pl.col('game_date').is_in(list(tune_days))); test = df.filter(~pl.col('game_date').is_in(list(tune_days)))
xt, yt = X(tune), tune['hit'].to_numpy().astype(float)
def nll(w): z = xt @ w; return np.mean(np.logaddexp(0, z) - yt * z) + 1e-4 * np.sum(w[1:]**2)
w = minimize(nll, np.zeros(xt.shape[1]), method='BFGS').x
def pblend(d): return 1 / (1 + np.exp(-(X(d) @ w)))
eps = 1e-4
def ll(p, y): p = np.clip(p, eps, 1-eps); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
ytest = test['hit'].to_numpy().astype(float)
pb_test = pblend(test)
print('TEST logloss: book %.5f | blend %.5f' % (ll(test['p_book'].to_numpy(), ytest), ll(pb_test, ytest)))
test = test.with_columns(pl.Series('p_blend', pb_test)).with_columns((pl.col('price')*pl.col('p_blend')-1).alias('ev'))
for c in (0.02, 0.05, 0.10):
    bets = test.filter(pl.col('ev') >= c, (1/pl.col('p_blend')) <= 15)
    print(f'cushion {c:.0%}:', roi(bets))
    for ang in ['starter_out','luck','new_or_early','other']:
        print(f'    {ang:13s}', roi(bets.filter(pl.col('angle')==ang)))
    for s in ['reb','ast','pts','pts+reb']:
        print(f'    {s:13s}', roi(bets.filter(pl.col('stat')==s)))

# --- line shopping: best AU price per rung (books with history: sportsbet, tab, dabble, neds/ladbrokes, pointsbet)
from nba_ladders.sources.odds_api import load_cached

odds = load_cached('historical').filter(pl.col('side') == 'Over', pl.col('threshold').is_not_null())
au = ['sportsbet', 'tab', 'dabble_au', 'neds', 'ladbrokes_au', 'pointsbetau']
best = (odds.filter(pl.col('bookmaker').is_in(au)).group_by('event_id', 'player', 'stat', 'threshold')
        .agg(pl.col('price').max().alias('best_au'), pl.col('bookmaker').sort_by('price').last().alias('best_book'),
             pl.col('bookmaker').n_unique().alias('n_books')))
t2 = test.join(best.rename({'player': 'player_api'}), on=['event_id', 'player_api', 'stat', 'threshold'], how='left')
print('\nrungs with 2+ AU books:', t2.filter(pl.col('n_books') >= 2).height, '/', t2.height,
      '| avg best/Sportsbet price ratio %.3f' % t2.filter(pl.col('n_books') >= 2).select((pl.col('best_au') / pl.col('price')).mean()).item())
t2 = t2.with_columns((pl.col('best_au') * pl.col('p_blend') - 1).alias('ev_best'))
for c in (0.02, 0.05, 0.10):
    bets = t2.filter(pl.col('ev_best') >= c, (1 / pl.col('p_blend')) <= 15).with_columns(pl.col('best_au').alias('price'))
    print(f'BEST AU price, cushion {c:.0%}:', roi(bets))
    for s in ['reb', 'ast', 'pts', 'pts+reb']:
        print(f'    {s:8s}', roi(bets.filter(pl.col('stat') == s)))
    print('    by book:', bets.group_by('best_book').len().sort('len', descending=True).to_dicts())
