import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


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


POP_SIZE = 100
N_GEN = 200
MUTATE_RATE = 0.1
CROSSOVER_RATE = 0.7
PATIENCE = 10


def init_population():
    return np.random.uniform(-1, 1, size=(POP_SIZE, 3))


def predict(X, individual):
    return np.dot(X, individual[:2]) + individual[2]


def evaluate(pop, X, y):
    return np.array([mean_squared_error(y, predict(X, ind)) for ind in pop])


def select(pop, fitness):
    elite_idx = np.argsort(fitness)[:POP_SIZE // 2]
    elite = pop[elite_idx]
    probs = 1 / (fitness + 1e-6)
    probs /= probs.sum()
    chosen_idx = np.random.choice(len(pop), POP_SIZE // 2, p=probs)
    return np.vstack([elite, pop[chosen_idx]])

def crossover(parents):
    offspring = []
    for _ in range(POP_SIZE // 2):
        p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
        if np.random.rand() < CROSSOVER_RATE:
            pt = np.random.randint(1, 3)
            child = np.concatenate([p1[:pt], p2[pt:]])
        else:
            child = p1.copy()
        offspring.append(child)
    return np.array(offspring)

def mutate(pop):
    for i in range(len(pop)):
        for j in range(3):
            if np.random.rand() < MUTATE_RATE:
                pop[i][j] += np.random.normal(0, 0.1)
    return pop


np.random.seed(42)
pop = init_population()
best_model, best_mse = None, float('inf')
patience_counter = 0

for gen in range(N_GEN):
    fitness = evaluate(pop, X_train, y_train)
    gen_best = np.min(fitness)
    if gen_best < best_mse:
        best_mse = gen_best
        best_model = pop[np.argmin(fitness)].copy()
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"ealy{gen} ")
        break
    selected = select(pop, fitness)
    children = crossover(selected)
    pop = mutate(children)


y_pred = predict(X_test, best_model)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2

print("\nGA ：")
print(f"ga: w1 = {best_model[0]:.4f}, w2 = {best_model[1]:.4f}, b = {best_model[2]:.4f}")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")


results = pd.DataFrame({
    'date': date_test,
    'True AQI': y_test,
    'Predicted AQI': y_pred
})
results.to_csv("ga_selfreg_aqi.csv", index=False, encoding='utf-8-sig')


plt.figure(figsize=(14, 6))
plt.plot(date_test, y_test, label='True AQI', marker='o')
plt.plot(date_test, y_pred, label='Predicted AQI (GA)', linestyle='--', marker='x')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("GA Predicted vs True AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ga_selfreg_aqi_plot.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title(f"GA $R^2$ = {r2:.4f}")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.grid(True)
plt.tight_layout()
plt.savefig("ga_selfreg_r2_scatter.png", dpi=300)
plt.show()
