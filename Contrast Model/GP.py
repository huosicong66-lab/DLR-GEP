import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.dates as mdates
from deap import gp, base, tools, algorithms, creator


df = pd.read_csv("beijing_aqi_2022_2024_combined.csv")
df = df[['date', 'AQI']]
df['AQI_t-1'] = df['AQI'].shift(1)
df['AQI_t-2'] = df['AQI'].shift(2)
df.dropna(inplace=True)

split_index = int(len(df) * 0.8)
X_train = df[['AQI_t-1', 'AQI_t-2']].iloc[:split_index].values
X_test = df[['AQI_t-1', 'AQI_t-2']].iloc[split_index:].values
y_train = df['AQI'].iloc[:split_index].values
y_test = df['AQI'].iloc[split_index:].values
date_test = pd.to_datetime(df['date'].iloc[split_index:])


pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

def protected_div(left, right):
    try:
        return left / right if abs(right) > 1e-6 else left
    except:
        return left
pset.addPrimitive(protected_div, 2)

pset.addEphemeralConstant("rand101", lambda: np.random.uniform(-2, 2))
pset.renameArguments(ARG0='AQI_t1')
pset.renameArguments(ARG1='AQI_t2')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def eval_individual(ind):
    func = toolbox.compile(expr=ind)
    y_pred = np.array([func(*x) for x in X_train])
    return mean_squared_error(y_train, y_pred),

toolbox.register("evaluate", eval_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=17))


np.random.seed(42)
pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=None, halloffame=hof, verbose=True)


best_ind = hof[0]
func = toolbox.compile(expr=best_ind)
y_pred = np.array([func(*x) for x in X_test])

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2

print("\nGP ")
print(f"gp：{str(best_ind)}")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")


results = pd.DataFrame({
    'date': date_test,
    'True AQI': y_test,
    'Predicted AQI': y_pred
})
results.to_csv("gp_selfreg_aqi.csv", index=False, encoding='utf-8-sig')


plt.figure(figsize=(14, 6))
plt.plot(date_test, y_test, label='True AQI', marker='o')
plt.plot(date_test, y_pred, label='Predicted AQI (GP)', linestyle='--', marker='x')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("GP Predicted vs True AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gp_selfreg_aqi_plot.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title(f'GP $R^2$ = {r2:.4f}')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.grid(True)
plt.tight_layout()
plt.savefig("gp_selfreg_r2_scatter.png", dpi=300)
plt.show()
