
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
warnings.filterwarnings("ignore")


csv_file = "beijing_aqi_2022_2024_combined.csv"
df = pd.read_csv(csv_file)

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']


for p in pollutants:
    df[f'{p}_t-1'] = df[p].shift(1)
    df[f'{p}_t-2'] = df[p].shift(2)


df['AQI_t'] = df['AQI']


df['target_AQI_next'] = df['AQI'].shift(-1)
df.dropna(inplace=True)


feature_cols = ['AQI_t'] + pollutants + \
               [f'{p}_t-1' for p in pollutants] + \
               [f'{p}_t-2' for p in pollutants]
X = df[feature_cols].values
y = df['target_AQI_next'].values
d = X.shape[1]


split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
date_test = df['date'].iloc[split_idx:].values


POP_SIZE, N_GEN  = 40, 100
CROSSOVER_RATE   = 0.6
MUTATE_RATE      = 0.03
PATIENCE         = 20
KFOLD            = 4
LAMBDA_PENALTY   = 0.4
np.random.seed(42)


def init_pop():
    pop = np.random.randint(0, 2, size=(POP_SIZE, d))
    for ind in pop:
        if ind.sum() == 0:
            ind[np.random.randint(0, d)] = 1
    return pop

def cv_mse(Xsel):
    cv = KFold(n_splits=KFOLD, shuffle=False)
    return np.mean([
        mean_squared_error(
            y_train[val],
            LinearRegression().fit(Xsel[tr], y_train[tr]).predict(Xsel[val])
        )
        for tr, val in cv.split(Xsel)
    ])

def fitness(ind):
    mask = ind.astype(bool)
    if mask.sum() == 0:
        return 1e9
    mse = cv_mse(X_train[:, mask])
    return mse + LAMBDA_PENALTY * mask.sum()

def select(pop, fit):
    elite_k = POP_SIZE // 4
    elite = pop[np.argsort(fit)[:elite_k]]
    prob = 1 / (fit + 1e-6)
    prob /= prob.sum()
    chosen = pop[np.random.choice(POP_SIZE, POP_SIZE - elite_k, p=prob)]
    return np.vstack([elite, chosen])

def crossover(parents):
    off = []
    for _ in range(POP_SIZE // 2):
        p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < CROSSOVER_RATE:
            cut = np.random.randint(1, d)
            c1[:cut], c2[:cut] = p2[:cut], p1[:cut]
        off += [c1, c2]
    return np.array(off)

def mutate(pop):
    for ind in pop:
        flip = np.random.rand(d) < MUTATE_RATE
        ind[flip] ^= 1
        if ind.sum() == 0:                              # 保证非空
            ind[np.random.randint(0, d)] = 1
    return pop


pop = init_pop()
best_mask, best_mse, stall = None, float('inf'), 0

print("Start GA evolution (target: next-day AQI, expect R2≈0.75)…")
for gen in range(1, N_GEN + 1):
    fit = np.array([fitness(ind) for ind in pop])
    if fit.min() < best_mse:
        best_mse = fit.min()
        best_mask = pop[fit.argmin()].copy()
        stall = 0
    else:
        stall += 1
    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:3d} | Best CV-MSE: {best_mse:.2f}")
    if stall >= PATIENCE:
        print(f"Early stop at gen {gen}")
        break
    pop = mutate(crossover(select(pop, fit)))

sel_cols = np.array(feature_cols)[best_mask.astype(bool)]
print("\nSelected features:", list(sel_cols))


lr = LinearRegression().fit(X_train[:, best_mask.astype(bool)], y_train)
y_pred = lr.predict(X_test[:, best_mask.astype(bool)])

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2   = np.corrcoef(y_test, y_pred)[0, 1] ** 2



print(f"\nTest  MSE={mse:.2f}  RMSE={rmse:.2f}  MAPE={mape:.3f}  R2={r2:.3f}")


out_csv = "lr_ga_nextday_results.csv"
pd.DataFrame({
    "date": date_test,
    "True AQI": y_test,
    "Predicted AQI": y_pred
}).to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Saved → {out_csv}")



plt.figure(figsize=(14, 6))
plt.plot(date_test, y_test, label='True AQI', marker='o')
plt.plot(date_test, y_pred, label='Predicted AQI (LR-GA)', linestyle='--', marker='x')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("LR-GA Predicted vs True AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lr_selfreg_aqi_plot.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title(f'LR $R^2$ = {r2:.4f}')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.grid(True)
plt.tight_layout()
plt.savefig("lr_selfreg_r2_scatter.png", dpi=300)
plt.show()
