from __future__ import annotations

import json
import math
import random
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "prepared_data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "DLR-GEP-Nested"

CITY_DIRECTORIES = {
    "Beijing": DATA_ROOT / "beijing",
    "Nanning": DATA_ROOT / "nanning",
}

SEEDS = [42, 52, 62, 72, 82]
INNER_TRAIN_RATIO = 0.80
VALIDATION_SEGMENTS = 3
STABILITY_PENALTY = 0.50
ENSEMBLE_SIZE = 3

DLR_ALPHA = 1.0
CONTRIBUTION_COUNT = 24

POPULATION_SIZE = 100
GENERATIONS = 100
GENE_COUNT = 6
MAX_DEPTH = 3
ELITE_COUNT = 8
TOURNAMENT_SIZE = 4
PATIENCE = 30
GENE_RIDGE_ALPHA = 0.10

PRESSURE_ALPHA = 2.0
MUTATION_MAX = 0.25
MUTATION_MIN = 0.04
CROSSOVER_MIN = 0.40
CROSSOVER_MAX = 0.80

EPS = 1e-8
MAX_VALUE = 1e6


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    valid = np.abs(y_true) > EPS
    mape = (
        float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100)
        if np.any(valid)
        else float("nan")
    )
    return {
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": mape,
        "R2": float(r2_score(y_true, y_pred)),
    }


def segmented_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: int = VALIDATION_SEGMENTS,
) -> Tuple[float, float, float, float, List[float]]:

    indices = np.array_split(np.arange(len(y_true)), segments)
    values = [
        math.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))
        for idx in indices
        if len(idx)
    ]
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=0))
    standard_error = float(std_value / math.sqrt(max(len(values), 1)))
    robust = float(mean_value + STABILITY_PENALTY * std_value)
    return robust, mean_value, std_value, standard_error, [float(v) for v in values]


def load_feature_names(city_dir: Path) -> List[str]:
    path = city_dir / "feature_names.json"
    with path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    if isinstance(content, list):
        return list(content)
    if isinstance(content, dict):
        names = content.get("feature_names") or content.get("features") or content.get("columns")
        if names:
            return list(names)
    raise ValueError(f"Unable to read feature names from  {path} feature names")


def load_X(path: Path, names: List[str]) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.loc[:, ~frame.columns.astype(str).str.lower().str.startswith("unnamed")]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}  is missing  {len(missing)} features: {missing}")
    array = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path.name}  contains missing or nonnumeric feature values")
    return array


def load_y(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "Target_AQI" not in frame.columns:
        raise ValueError(f"{path.name}  is missing  Target_AQI")
    array = pd.to_numeric(frame["Target_AQI"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path.name} contains invalid targets")
    return array.reshape(-1)


def load_dates(city_dir: Path, split: str, n: int) -> np.ndarray:
    path = city_dir / f"{split}.csv"
    if path.exists():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "Target_Date" in frame.columns and len(frame) == n:
            return frame["Target_Date"].astype(str).to_numpy()
    return np.arange(n).astype(str)


def load_city(city_dir: Path) -> Tuple:
    names = load_feature_names(city_dir)
    X_train = load_X(city_dir / "X_train.csv", names)
    y_train = load_y(city_dir / "y_train.csv")
    X_val = load_X(city_dir / "X_validation.csv", names)
    y_val = load_y(city_dir / "y_validation.csv")
    X_test = load_X(city_dir / "X_test.csv", names)
    y_test = load_y(city_dir / "y_test.csv")
    if not (len(X_train) == len(y_train) and len(X_val) == len(y_val) and len(X_test) == len(y_test)):
        raise ValueError("X and y have different sample counts")
    forbidden = {"target_aqi", "targetaqi", "aqi_t+1", "aqit1"}
    if any(name.lower() in forbidden for name in names):
        raise ValueError("Future AQI target features detected; execution stopped to prevent leakage")
    return (
        X_train, y_train, X_val, y_val, X_test, y_test, names,
        load_dates(city_dir, "validation", len(y_val)),
        load_dates(city_dir, "test", len(y_test)),
    )


def fit_dlr(X: np.ndarray, y: np.ndarray) -> Tuple[Ridge, np.ndarray]:
    model = Ridge(alpha=DLR_ALPHA, fit_intercept=True)
    model.fit(X, y)
    beta = np.concatenate([[float(model.intercept_)], np.asarray(model.coef_, dtype=float)])
    return model, beta


def contribution_ranking(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    importance = np.mean(np.abs(X * beta[1:].reshape(1, -1)), axis=0)
    return np.argsort(importance)[::-1]


def enhance(
    X: np.ndarray,
    dlr_prediction: np.ndarray,
    beta: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    contributions = X[:, selected] * beta[1:][selected].reshape(1, -1)

    return np.column_stack([X, contributions, dlr_prediction])


UNARY = ("sin", "cos", "tanh", "sqrt", "log", "abs")
BINARY = ("add", "sub", "mul", "div")


@dataclass
class Node:
    op: str
    feature: Optional[int] = None
    constant: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


def safe(array: np.ndarray) -> np.ndarray:
    return np.clip(
        np.nan_to_num(array, nan=0.0, posinf=MAX_VALUE, neginf=-MAX_VALUE),
        -MAX_VALUE,
        MAX_VALUE,
    )


def evaluate(node: Node, X: np.ndarray) -> np.ndarray:
    if node.op == "feature":
        return X[:, node.feature]
    if node.op == "constant":
        return np.full(len(X), node.constant, dtype=float)
    if node.op in UNARY:
        value = evaluate(node.left, X)
        if node.op == "sin":
            result = np.sin(np.clip(value, -30, 30))
        elif node.op == "cos":
            result = np.cos(np.clip(value, -30, 30))
        elif node.op == "tanh":
            result = np.tanh(value)
        elif node.op == "sqrt":
            result = np.sqrt(np.abs(value) + EPS)
        elif node.op == "log":
            result = np.log1p(np.abs(value))
        else:
            result = np.abs(value)
        return safe(result)
    left, right = evaluate(node.left, X), evaluate(node.right, X)
    if node.op == "add":
        result = left + right
    elif node.op == "sub":
        result = left - right
    elif node.op == "mul":
        result = left * right
    else:
        denominator = np.where(np.abs(right) < EPS, np.where(right >= 0, EPS, -EPS), right)
        result = left / denominator
    return safe(result)


def count_nodes(node: Node) -> int:
    return 1 + (count_nodes(node.left) if node.left else 0) + (count_nodes(node.right) if node.right else 0)


def choose_feature(weights: np.ndarray, rng: random.Random) -> int:
    value = rng.random() * float(np.sum(weights))
    return min(int(np.searchsorted(np.cumsum(weights), value, side="right")), len(weights) - 1)


def random_tree(n_features: int, weights: np.ndarray, depth: int, rng: random.Random) -> Node:
    if depth <= 0 or rng.random() < 0.30:
        if rng.random() < 0.90:
            return Node("feature", feature=choose_feature(weights, rng))
        return Node("constant", constant=rng.uniform(-2.0, 2.0))
    if rng.random() < 0.35:
        return Node(rng.choice(UNARY), left=random_tree(n_features, weights, depth - 1, rng))
    return Node(
        rng.choice(BINARY),
        left=random_tree(n_features, weights, depth - 1, rng),
        right=random_tree(n_features, weights, depth - 1, rng),
    )


def node_text(node: Node, names: List[str]) -> str:
    if node.op == "feature":
        return names[node.feature]
    if node.op == "constant":
        return f"{node.constant:.6f}"
    if node.op in UNARY:
        child = node_text(node.left, names)
        forms = {
            "sin": f"sin({child})", "cos": f"cos({child})", "tanh": f"tanh({child})",
            "sqrt": f"sqrt(abs({child}))", "log": f"log(1+abs({child}))", "abs": f"abs({child})",
        }
        return forms[node.op]
    symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[node.op]
    return f"({node_text(node.left, names)} {symbol} {node_text(node.right, names)})"


@dataclass
class Chromosome:
    genes: List[Node]
    coefficients: Optional[np.ndarray] = None
    gene_mean: Optional[np.ndarray] = None
    gene_std: Optional[np.ndarray] = None
    train_rmse: float = np.inf
    inner_rmse: float = np.inf
    robust_score: float = np.inf
    complexity: int = 0
    objective: float = np.inf


def gene_matrix(chromosome: Chromosome, X: np.ndarray) -> np.ndarray:
    return np.column_stack([evaluate(gene, X) for gene in chromosome.genes])


def new_chromosome(n_features: int, weights: np.ndarray, rng: random.Random) -> Chromosome:
    return Chromosome([random_tree(n_features, weights, MAX_DEPTH, rng) for _ in range(GENE_COUNT)])


def seeded_chromosome(indices: np.ndarray, n_features: int, weights: np.ndarray, rng: random.Random) -> Chromosome:
    genes = [Node("feature", feature=int(index)) for index in indices[:GENE_COUNT]]
    while len(genes) < GENE_COUNT:
        genes.append(random_tree(n_features, weights, MAX_DEPTH, rng))
    return Chromosome(genes)


def fit_chromosome(
    chromosome: Chromosome,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_select: np.ndarray,
    y_select: np.ndarray,
) -> Chromosome:
    raw_fit, raw_select = gene_matrix(chromosome, X_fit), gene_matrix(chromosome, X_select)
    mean = np.mean(raw_fit, axis=0)
    std = np.where(np.std(raw_fit, axis=0) < EPS, 1.0, np.std(raw_fit, axis=0))
    Z_fit, Z_select = (raw_fit - mean) / std, (raw_select - mean) / std
    model = Ridge(alpha=GENE_RIDGE_ALPHA)
    model.fit(Z_fit, y_fit)
    pred_fit, pred_select = model.predict(Z_fit), model.predict(Z_select)
    train_rmse = math.sqrt(mean_squared_error(y_fit, pred_fit))
    inner_rmse = math.sqrt(mean_squared_error(y_select, pred_select))
    robust, _, _, _, _ = segmented_score(y_select, pred_select)
    complexity = sum(count_nodes(gene) for gene in chromosome.genes)
    gap = max(0.0, inner_rmse - train_rmse)
    objective = robust + 0.03 * gap + 0.0005 * max(np.std(y_fit), 1.0) * complexity
    chromosome.coefficients = np.concatenate([[model.intercept_], model.coef_])
    chromosome.gene_mean, chromosome.gene_std = mean, std
    chromosome.train_rmse, chromosome.inner_rmse = train_rmse, inner_rmse
    chromosome.robust_score, chromosome.complexity, chromosome.objective = robust, complexity, objective
    return chromosome


def refit_chromosome(chromosome: Chromosome, X: np.ndarray, y: np.ndarray) -> Chromosome:

    result = deepcopy(chromosome)
    raw = gene_matrix(result, X)
    mean = np.mean(raw, axis=0)
    std = np.where(np.std(raw, axis=0) < EPS, 1.0, np.std(raw, axis=0))
    Z = (raw - mean) / std
    model = Ridge(alpha=GENE_RIDGE_ALPHA).fit(Z, y)
    result.coefficients = np.concatenate([[model.intercept_], model.coef_])
    result.gene_mean, result.gene_std = mean, std
    return result


def predict_chromosome(chromosome: Chromosome, X: np.ndarray) -> np.ndarray:
    raw = gene_matrix(chromosome, X)
    Z = (raw - chromosome.gene_mean) / chromosome.gene_std
    return safe(chromosome.coefficients[0] + Z @ chromosome.coefficients[1:])


def tournament(population: List[Chromosome], rng: random.Random) -> Chromosome:
    candidates = rng.sample(population, min(TOURNAMENT_SIZE, len(population)))
    return min(candidates, key=lambda item: item.objective)


def probabilities(generation: int) -> Tuple[float, float]:
    pressure = (generation / max(GENERATIONS - 1, 1)) ** PRESSURE_ALPHA
    mutation = MUTATION_MAX - (MUTATION_MAX - MUTATION_MIN) * pressure
    crossover = CROSSOVER_MIN + (CROSSOVER_MAX - CROSSOVER_MIN) * pressure
    return mutation, crossover


def offspring(
    parent: Chromosome,
    elites: List[Chromosome],
    n_features: int,
    weights: np.ndarray,
    mutation: float,
    crossover: float,
    rng: random.Random,
) -> Chromosome:
    genes = []
    for index in range(GENE_COUNT):
        guide = rng.choice(elites)
        source = guide.genes[index] if rng.random() < crossover else parent.genes[index]
        gene = random_tree(n_features, weights, MAX_DEPTH, rng) if rng.random() < mutation else deepcopy(source)
        genes.append(gene)
    return Chromosome(genes)


def evolve(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_select: np.ndarray,
    y_select: np.ndarray,
    weights: np.ndarray,
    important: np.ndarray,
    seed: int,
) -> Tuple[Chromosome, pd.DataFrame]:
    rng = random.Random(seed)
    np.random.seed(seed)
    population = [seeded_chromosome(np.roll(important, shift), X_fit.shape[1], weights, rng) for shift in range(10)]
    population.extend(new_chromosome(X_fit.shape[1], weights, rng) for _ in range(POPULATION_SIZE - len(population)))
    population = [fit_chromosome(item, X_fit, y_fit, X_select, y_select) for item in population]
    best = deepcopy(min(population, key=lambda item: item.objective))
    history, unchanged = [], 0
    for generation in range(GENERATIONS):
        population.sort(key=lambda item: item.objective)
        elites = [deepcopy(item) for item in population[:ELITE_COUNT]]
        if elites[0].objective < best.objective - 1e-10:
            best, unchanged = deepcopy(elites[0]), 0
        else:
            unchanged += 1
        mutation, crossover = probabilities(generation)
        history.append({
            "Generation": generation + 1,
            "Training_RMSE": best.train_rmse,
            "Inner_Validation_RMSE": best.inner_rmse,
            "Robust_Score": best.robust_score,
            "Complexity": best.complexity,
            "Mutation": mutation,
            "Crossover": crossover,
        })
        if generation == 0 or (generation + 1) % 10 == 0:
            print(f"    Generation {generation + 1:3d}/{GENERATIONS}, inner={best.inner_rmse:.6f}, score={best.robust_score:.6f}")
        if unchanged >= PATIENCE:
            print(f"    Early stopping at generation {generation + 1}")
            break
        next_population = elites.copy()
        while len(next_population) < POPULATION_SIZE:
            parent = tournament(population, rng)
            child = offspring(parent, elites, X_fit.shape[1], weights, mutation, crossover, rng)
            child = fit_chromosome(child, X_fit, y_fit, X_select, y_select)
            next_population.append(child if np.isfinite(child.objective) and child.objective <= parent.objective * 1.20 else deepcopy(parent))
        population = next_population
    return best, pd.DataFrame(history)


def terminal_weights(X: np.ndarray, y: np.ndarray, original_count: int, contribution_count: int) -> np.ndarray:
    values = []
    for column in range(X.shape[1]):
        corr = 0.0 if np.std(X[:, column]) < EPS else np.corrcoef(X[:, column], y)[0, 1]
        values.append(abs(float(corr)) + 0.05 if np.isfinite(corr) else 0.05)
    weights = np.asarray(values)
    weights[original_count:original_count + contribution_count] *= 1.25
    weights[-1] *= 2.0
    return weights / np.sum(weights)


def expression(chromosome: Chromosome, names: List[str]) -> str:
    terms = [f"{chromosome.coefficients[0]:.10f}"]
    for index, gene in enumerate(chromosome.genes):
        terms.append(f"({chromosome.coefficients[index + 1]:.10f})*({node_text(gene, names)})")
    return " + ".join(terms)


def process_city(city: str, city_dir: Path) -> Tuple[List[Dict], Dict]:
    print("\n" + "=" * 78)
    print(f"Processing city: {city}")
    print("=" * 78)
    X_train, y_train, X_val, y_val, X_test, y_test, names, val_dates, test_dates = load_city(city_dir)
    split = int(len(y_train) * INNER_TRAIN_RATIO)
    if split < 50 or len(y_train) - split < 30:
        raise ValueError("The training set is too small for the inner chronological split")
    print(f"Features: {X_train.shape[1]}; inner train: {split}; inner validation: {len(y_train)-split}; official validation: {len(y_val)}; test: {len(y_test)}")
    output = OUTPUT_ROOT / city
    output.mkdir(parents=True, exist_ok=True)


    dlr, beta = fit_dlr(X_train, y_train)
    dlr_train = np.maximum(dlr.predict(X_train), 0.0)
    dlr_val = np.maximum(dlr.predict(X_val), 0.0)
    dlr_test = np.maximum(dlr.predict(X_test), 0.0)
    ranking = contribution_ranking(X_train, beta)
    selected = ranking[:min(CONTRIBUTION_COUNT, X_train.shape[1])]

    enhanced_train = enhance(X_train, dlr_train, beta, selected)
    enhanced_val = enhance(X_val, dlr_val, beta, selected)
    enhanced_test = enhance(X_test, dlr_test, beta, selected)
    enhanced_names = names + [f"DLR_contribution_{names[i]}" for i in selected] + ["DLR_prediction"]
    scaler = StandardScaler().fit(enhanced_train[:split])
    E_train, E_val, E_test = scaler.transform(enhanced_train), scaler.transform(enhanced_val), scaler.transform(enhanced_test)
    weights = terminal_weights(E_train[:split], y_train[:split], len(names), len(selected))
    important = np.argsort(weights)[::-1]
    lower, upper = 0.0, float(np.max(y_train) + 0.25 * np.std(y_train))

    importance = np.mean(np.abs(X_train * beta[1:].reshape(1, -1)), axis=0)
    pd.DataFrame({"Feature": names, "DLR_Importance": importance, "Selected": [i in set(selected) for i in range(len(names))]}).sort_values("DLR_Importance", ascending=False).to_csv(output / "dlr_feature_importance.csv", index=False, encoding="utf-8-sig")

    runs = []
    for run_number, seed in enumerate(SEEDS, 1):
        print(f"\nRun {run_number}/{len(SEEDS)}, seed={seed}")
        started = time.perf_counter()
        structure, history = evolve(E_train[:split], y_train[:split], E_train[split:], y_train[split:], weights, important, seed)

        model = refit_chromosome(structure, E_train, y_train)
        val_prediction = np.clip(predict_chromosome(model, E_val), lower, upper)
        test_prediction = np.clip(predict_chromosome(model, E_test), lower, upper)
        score, mean_rmse, std_rmse, se, _ = segmented_score(y_val, val_prediction)
        runs.append({
            "Run": run_number, "Seed": seed, "Model": model,
            "Validation_Prediction": val_prediction, "Test_Prediction": test_prediction,
            "Robust_Score": score, "Segment_Mean": mean_rmse,
            "Segment_Std": std_rmse, "Standard_Error": se,
            "Validation_Metrics": metrics(y_val, val_prediction),
            "Time": time.perf_counter() - started,
        })
        history.to_csv(output / f"convergence_seed_{seed}.csv", index=False, encoding="utf-8-sig")
        (output / f"expression_seed_{seed}.txt").write_text(expression(model, enhanced_names), encoding="utf-8")
        print(f"  Official validation RMSE={runs[-1]['Validation_Metrics']['RMSE']:.6f}, robust score={score:.6f}")

    runs.sort(key=lambda item: (item["Robust_Score"], item["Segment_Std"]))
    selected_runs = runs[:min(ENSEMBLE_SIZE, len(runs))]
    selected_seeds = [item["Seed"] for item in selected_runs]
    gep_val = np.median(np.column_stack([item["Validation_Prediction"] for item in selected_runs]), axis=1)
    gep_test = np.median(np.column_stack([item["Test_Prediction"] for item in selected_runs]), axis=1)


    final_val = gep_val.copy()
    final_test = gep_test.copy()

    dlr_metrics, gep_metrics, final_metrics = metrics(y_test, dlr_test), metrics(y_test, gep_test), metrics(y_test, final_test)
    print(f"\nSelected seeds: {selected_seeds}")
    print("Prediction mode: direct improved-GEP prediction using DLR-enhanced features")
    print("Test results:")
    print(f"  DLR: RMSE={dlr_metrics['RMSE']:.6f}, R2={dlr_metrics['R2']:.6f}")
    print(f"  Improved-GEP: RMSE={gep_metrics['RMSE']:.6f}, R2={gep_metrics['R2']:.6f}")
    print(f"  DLR-GEP: RMSE={final_metrics['RMSE']:.6f}, R2={final_metrics['R2']:.6f}")

    pd.DataFrame([{
        "City": city, "Run": item["Run"], "Seed": item["Seed"],
        "Validation_RMSE": item["Validation_Metrics"]["RMSE"],
        "Validation_R2": item["Validation_Metrics"]["R2"],
        "Robust_Score": item["Robust_Score"], "Segment_Std": item["Segment_Std"],
        "Selected": item["Seed"] in selected_seeds, "Time_seconds": item["Time"],
    } for item in runs]).to_csv(output / "independent_runs_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Target_Date": test_dates, "Observed_AQI": y_test,
        "DLR_Prediction": dlr_test, "Improved_GEP_Prediction": gep_test,
        "DLR_GEP_Prediction": final_test, "DLR_GEP_Error": y_test - final_test,
    }).to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")

    comparison = [
        {"City": city, "Model": "DLR", "Prediction_Mode": "Linear component only", **dlr_metrics},
        {"City": city, "Model": "DLR-GEP", "Prediction_Mode": "Direct improved-GEP prediction using DLR-enhanced features", **final_metrics},
    ]
    pd.DataFrame(comparison).to_csv(output / "component_comparison.csv", index=False, encoding="utf-8-sig")
    final_row = {
        "City": city, "Model": "DLR-GEP", "Original_Feature_Count": len(names),
        "Enhanced_Feature_Count": enhanced_train.shape[1],
        "Prediction_Mode": "Direct improved-GEP prediction using DLR-enhanced features",
        "Selected_Seeds": ",".join(map(str, selected_seeds)), **final_metrics,
    }
    metadata = {
        "City": city, "Inner_Train_Size": split, "Inner_Validation_Size": len(y_train) - split,
        "Official_Validation_Size": len(y_val), "Test_Size": len(y_test),
        "Selected_Seeds": selected_seeds,
        "Prediction_Mode": "Direct improved-GEP prediction using DLR-enhanced features",
        "Selection_Protocol": "nested chronological validation",
        "Test_Used_For_Selection": False,
    }
    (output / "model_information.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return comparison, final_row


def main() -> None:
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Input directory: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparisons, final_rows = [], []
    for city, city_dir in CITY_DIRECTORIES.items():
        if not city_dir.exists():
            print(f"Skipping  {city}: not found: {city_dir}")
            continue
        city_comparison, final_row = process_city(city, city_dir)
        comparisons.extend(city_comparison)
        final_rows.append(final_row)
    comparison_frame, final_frame = pd.DataFrame(comparisons), pd.DataFrame(final_rows)
    comparison_frame.to_csv(OUTPUT_ROOT / "DLR_GEP_component_comparison.csv", index=False, encoding="utf-8-sig")
    final_frame.to_csv(OUTPUT_ROOT / "DLR_GEP_final_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 78)
    print("DLR-GEP component comparison")
    print("=" * 78)
    print(comparison_frame.to_string(index=False))
    print("\n" + "=" * 78)
    print("DLR-GEP final summary")
    print("=" * 78)
    print(final_frame.to_string(index=False))


if __name__ == "__main__":
    main()
