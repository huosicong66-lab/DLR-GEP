import numpy as np, pandas as pd, operator, re, warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from deap import gp, base, tools, algorithms, creator
warnings.filterwarnings("ignore")

df = pd.read_csv("beijing_aqi_2022_2024_combined.csv")
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
for p in pollutants:
    df[f"{p}_t-1"] = df[p].shift(1)
    df[f"{p}_t-2"] = df[p].shift(2)

df['AQI_t']      = df['AQI']
df['target_next'] = df['AQI'].shift(-1)
df.dropna(inplace=True)

raw_cols = ['AQI_t'] + pollutants + \
           [f"{p}_t-1" for p in pollutants] + \
           [f"{p}_t-2" for p in pollutants]

X = df[raw_cols].values
y = df['target_next'].values

split = int(len(df)*0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
date_te    = df['date'].iloc[split:].values

lr = LinearRegression().fit(X_tr, y_tr)
base_tr   = lr.predict(X_tr)
base_te   = lr.predict(X_te)
residuals = y_tr - base_tr

def safe_name(col):
    return re.sub(r'\W', '_', col)

safe_cols = [safe_name(c) for c in raw_cols]
idx2var   = {f"ARG{i}": v for i, v in enumerate(safe_cols)}

pset = gp.PrimitiveSet("MAIN", len(raw_cols))
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
def pdiv(x, y): return x/y if abs(y)>1e-6 else x
pset.addPrimitive(pdiv, 2)
pset.addEphemeralConstant("rand", lambda: np.random.uniform(-2,2))

pset.renameArguments(**idx2var)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def eval_ind(ind):
    func = toolbox.compile(expr=ind)
    preds = np.array([func(*row) for row in X_tr])
    return mean_squared_error(residuals, preds),

toolbox.register("evaluate", eval_ind)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=17))


pop  = toolbox.population(n=200)
hof  = tools.HallOfFame(1)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                    ngen=40, stats=None, halloffame=hof, verbose=False)

best = hof[0]
print("\nBest GP expression:\n", best)


func_best = toolbox.compile(expr=best)
gp_res_te = np.array([func_best(*row) for row in X_te])
y_pred    = base_te + gp_res_te

mse  = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_te, y_pred)
r2   = np.corrcoef(y_te, y_pred)[0,1]**2
print(f"\nHybrid LR-GP  MSE={mse:.2f}  RMSE={rmse:.2f}  MAPE={mape:.3f}  R2={r2:.3f}")

pd.DataFrame({
    "date": date_te,
    "True AQI": y_te,
    "Predicted AQI": y_pred
}).to_csv("lr_gp_nextday_results.csv", index=False, encoding="utf-8-sig")
print("Saved → lr_gp_nextday_results.csv")
