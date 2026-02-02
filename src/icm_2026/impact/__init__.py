from collections import defaultdict
import csv
import joblib
import numpy as np
from sklearn import linear_model

FEATS = [
    'blocks', 'steals',
    'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
    'reboundsTotal', 'foulsPersonal',
]
C_FEATS = [
    'BLK', 'STL',
    'FG%', '3P%', 'FT%',
    'TRB', 'PF'
]

TRAIN = True
RETRAIN = False

if TRAIN:
    with open('data/Players.csv') as f:
        PLAYERS = {p['personId']: p for p in csv.DictReader(f)}

    with open('data/PlayerStatistics.csv') as f:
        STATS = list(csv.DictReader(f))

    DATA = {
        'guard': defaultdict(list),
        'forward': defaultdict(list),
        'center': defaultdict(list),
    }
    ALL_X = []

    for s in STATS:
        if any(s[f] == '' for f in FEATS):
            continue

        x = [float(s[f]) for f in FEATS]
        ALL_X.append(x)

        for pos in ['guard', 'forward', 'center']:
            if PLAYERS[s['personId']][pos] != 'True':
                continue

            for label in ['plusMinusPoints', 'numMinutes']:
                if s[label] == '':
                    continue

                DATA[pos][label].append((x, float(s[label])))

    ALL_X = np.array(ALL_X)
    GLOBAL_MIN = ALL_X.min(axis=0)
    GLOBAL_MAX = ALL_X.max(axis=0)
    joblib.dump({'min': GLOBAL_MIN, 'max': GLOBAL_MAX}, 'global-minmax.joblib')

def norm_x(xs, mins, maxs):
    return (xs - mins) / (maxs - mins)

def norm_y(y):
    col = y[:, 0]
    mn = col.min()
    mx = col.max()
    return ((col - mn) / (mx - mn)).reshape(-1, 1)

def rmse(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    return np.sqrt(np.mean((a - b) ** 2))

def rmse_eval():
    mm = joblib.load('global-minmax.joblib')

    gmin = mm['min']
    gmax = mm['max']

    for pos in ['forward', 'guard', 'center']:
        for label in ['plusMinusPoints', 'numMinutes']:
            pairs = DATA[pos][label]
            if not pairs:
                print(f'{pos},{label},n=0,rmse_norm=NA,rmse_raw=NA')
                continue

            x_raw = np.array([p[0] for p in pairs], dtype=float)
            x = norm_x(x_raw, gmin, gmax)

            y_raw = np.array([[p[1]] for p in pairs], dtype=float)
            y = norm_y(y_raw)

            mdl = joblib.load(f'{pos}_{label}.joblib')
            pred = mdl.predict(x)
            rmse_norm = rmse(y, pred)

            col = y_raw[:, 0]

            mn = col.min()
            mx = col.max()

            pred_raw = pred.ravel() * (mx - mn) + mn
            rmse_raw = rmse(col, pred_raw)

            print(f'{pos} {label} rmse_norm:{rmse_norm:.5f} rmse_raw:{rmse_raw:.5f}')

def inference():
    mm = joblib.load('global-minmax.joblib')
    gmin = mm['min']
    gmax = mm['max']

    stats = {}

    for path in ['ncaa.csv', 'ncaa2.csv', 'intl.csv']:
        with open(f'c2/{path}') as f:
            for p in csv.DictReader(f):
                stats[p['Player']] = p

    pos_map = {
        'Wing': 'forward',
        'Guard': 'guard',
        'Forward': 'forward',
        'Big': 'center',
        'Wing/Forward': 'forward'
    }

    xs = defaultdict(list)
    names = defaultdict(list)

    with open('c2/draft.csv') as f:
        for p in csv.DictReader(f):
            name = p['name']
            pos = pos_map[p['position']]
            xs[pos].append([float(stats[name][f]) for f in C_FEATS])
            names[pos].append(name)

    for pos in ['forward', 'guard', 'center']:
        x = norm_x(np.array(xs[pos]), gmin, gmax)

        mdls = [
            joblib.load(f'{pos}_plusMinusPoints.joblib'),
            joblib.load(f'{pos}_numMinutes.joblib'),
        ]

        preds = [mdl.predict(x).ravel() for mdl in mdls]

        for i, name in enumerate(names[pos]):
            print(f'{name},{preds[0][i]:.5f},{preds[1][i]:.5f}')

def train(pos, label):
    pairs = DATA[pos][label]

    x_raw = np.array([p[0] for p in pairs])
    x = norm_x(x_raw, GLOBAL_MIN, GLOBAL_MAX)
    y = norm_y(np.array([[p[1]] for p in pairs]))

    mdl = linear_model.LinearRegression()
    mdl.fit(x, y)

    joblib.dump(mdl, f'{pos}_{label}.joblib')

def stat(pos, label):
    mdl = joblib.load(f'imp/{pos}_{label}.joblib')

    with open(f'{pos}_{label}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(FEATS)
        writer.writerow([f'{x:.5f}' for x in mdl.coef_[0]])

def main():
    if TRAIN:
        if RETRAIN:
            for x in ['forward', 'guard', 'center']:
                for label in ['plusMinusPoints', 'numMinutes']:
                    print(f'{x} {label}...')

                    train(x, label)
                    stat(x, label)

        rmse_eval()
    else:
        inference()
