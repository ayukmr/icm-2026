from collections import defaultdict
from unidecode import unidecode
import csv
import joblib
import numpy as np
from sklearn import linear_model

FEATS = [
    'blocks', 'steals',
    'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
    'reboundsTotal', 'foulsPersonal',
]

PTS_FEATS = [
    'q1Points', 'q2Points', 'q3Points', 'q4Points',
    'benchPoints', 'pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance'
]

GAMES = defaultdict(list)
TIMED = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

PM25 = defaultdict(list)
MINS25 = defaultdict(list)

with open('data/PlayerStatistics.csv') as f:
    for p in csv.DictReader(f):
        name = p['firstName'] + ' ' + p['lastName']
        g = p['gameId']
        if any(p[k] == '' for k in ['plusMinusPoints', 'numMinutes']):
            continue
        for k in ['plusMinusPoints', 'numMinutes']:
            TIMED[name][g][k].append(float(p[k]))
        GAMES[name].append((g, p['playerteamCity']))

        date = p['gameDateTimeEst']

        if date.startswith('2025-10') or date.startswith('2025-11') or date.startswith('2025-12') or date.startswith('2026'):
            PM25[name].append(float(p['plusMinusPoints']))
            MINS25[name].append(float(p['numMinutes']))

with open('tr/players.csv') as f:
    players = [unidecode(line.strip()) for line in f if line.strip()]
    for p in players:
        if len(PM25[p]) == 0 or len(MINS25[p]) == 0:
            print(f'{p} IS BROKEN')

STATS = defaultdict(list)

with open('data/TeamStatistics.csv') as f:
    for s in csv.DictReader(f):
        STATS[s['teamCity']].append(s)

def norm_x(xs, mn=None, mx=None):
    xs = np.asarray(xs, dtype=float)
    if mn is None or mx is None:
        mn = np.nanmin(xs, axis=0)
        mx = np.nanmax(xs, axis=0)

    denom = mx - mn
    denom = np.where(denom == 0, 1.0, denom)
    x = (xs - mn) / denom
    return mn, mx, x

def norm_y(y):
    y = np.asarray(y, dtype=float)
    col = y[:, 0]
    mn, mx = np.nanmin(col), np.nanmax(col)
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    return ((col - mn) / denom).reshape(-1, 1)

def train(player, label):
    game_city = {gid: city for (gid, city) in GAMES[player]}

    g_stats = []
    for gid, city in game_city.items():
        for row in STATS[city]:
            if row["gameId"] != gid:
                continue
            if any(row[f] == '' for f in FEATS + PTS_FEATS):
                continue
            if row.get("teamScore", "") in ("", "0"):
                continue
            g_stats.append(row)
            break

    xg = [
        [float(g[f]) for f in FEATS]
        + [float(g[f]) / float(g["teamScore"]) for f in PTS_FEATS]
        for g in g_stats
    ]

    p_stats = [np.mean(TIMED[player][g["gameId"]][label]) for g in g_stats]

    mn, mx, x = norm_x(np.array(xg, dtype=float))
    y = norm_y(np.array(p_stats, dtype=float).reshape(-1, 1))

    mdl = linear_model.LinearRegression()
    mdl.fit(x, y)

    joblib.dump({"mdl": mdl, "mn": mn, "mx": mx}, f"tr2/{player}-{label}.joblib")

def run(tm):
    game_counts = {k: len(v) for k, v in PM25.items()}
    for dat in [PM25, MINS25]:
        for k, v in dat.items():
            dat[k] = sum(v) / len(v)

    def minmax_norm(d):
        vals = np.array(list(d.values()), dtype=float)
        mn, mx = vals.min(), vals.max()
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    pm25n = minmax_norm(PM25)
    mins25n = minmax_norm(MINS25)

    xth = []

    for s in STATS[tm]:
        date = s['gameDateTimeEst']
        if date.startswith('2025-10') or date.startswith('2025-11') or date.startswith('2025-12') or date.startswith('2026'):
            xth.append([float(s[f]) for f in FEATS] + [float(s[f]) / float(s['teamScore']) for f in PTS_FEATS])

    x = np.mean(xth, axis=0)

    ps = []

    with open('tr/players.csv') as f:
        players = [unidecode(line.strip()) for line in f if line.strip()]
        for p in players:
            if game_counts[p] < 5 or MINS25[p] < 15:
                continue

            mdls = [
                joblib.load(f'tr2/{p}-plusMinusPoints.joblib'),
                joblib.load(f'tr2/{p}-numMinutes.joblib')
            ]

            preds = []

            for dat in mdls:
                mdl = dat['mdl']
                mn = dat['mn']
                mx = dat['mx']
                _, _, xn = norm_x(x, mn=mn, mx=mx)
                pred = mdl.predict(xn.reshape(1, -1)).item()
                preds.append(pred)

            team = preds[0] * preds[1]
            impact = pm25n[p] * mins25n[p]

            ps.append([p, team, impact])

    with open('out.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(ps)

import os
from concurrent.futures import ProcessPoolExecutor

def train_player(p):
    if not os.path.exists(f'tr2/{p}-plusMinusPoints.joblib') or not os.path.exists(f'tr2/{p}-numMinutes.joblib'):
        train(p, 'plusMinusPoints')
        train(p, 'numMinutes')
    return p

def main():
    with open('tr/players.csv') as f:
        players = [unidecode(line.strip()) for line in f if line.strip()]

    players = [p for p in players if p in GAMES]

    workers = os.cpu_count() or 4

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p in ex.map(train_player, players, chunksize=10):
            print(f'done: {p}')
