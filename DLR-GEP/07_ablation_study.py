from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "results" / "DLR-GEP-Ablation"


QUICK_TEST = False
QUICK_POPULATION = 20
QUICK_GENERATIONS = 10

VARIANTS = [
    {"code": "A0", "name": "Full DLR-GEP", "dlr": True,  "multi": True,  "dynamic": True,  "suppress": True},
    {"code": "A1", "name": "w/o DLR enhancement", "dlr": False, "multi": True,  "dynamic": True,  "suppress": True},
    {"code": "A2", "name": "w/o Multi-Elite", "dlr": True,  "multi": False, "dynamic": True,  "suppress": True},
    {"code": "A3", "name": "w/o Dynamic Pressure", "dlr": True,  "multi": True,  "dynamic": False, "suppress": True},
    {"code": "A4", "name": "w/o Disturbance Suppression", "dlr": True, "multi": True, "dynamic": True, "suppress": False},
    {"code": "A5", "name": "Standard GEP", "dlr": True, "multi": False, "dynamic": False, "suppress": False},
    {"code": "A6", "name": "DLR only", "dlr": True, "dlr_only": True},
]


def load_core():
    candidates = [
        SCRIPT_DIR / "06_dlr_gep_nested.py",
        SCRIPT_DIR / "06_dlr_gep_core.py",
        SCRIPT_DIR / "06_dlr_gep.py",
    ]
    required = {
        "load_city", "fit_dlr", "contribution_ranking", "enhance",
        "seeded_chromosome", "new_chromosome", "fit_chromosome",
        "refit_chromosome", "predict_chromosome", "terminal_weights",
        "segmented_score", "metrics", "expression", "random_tree",
        "Chromosome",
    }
    path = None
    module = None
    incompatible = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            f"dlr_gep_core_{candidate.stem}", candidate
        )
        candidate_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            continue
        sys.modules[spec.name] = candidate_module
        spec.loader.exec_module(candidate_module)
        missing = sorted(name for name in required if not hasattr(candidate_module, name))
        if not missing:
            path, module = candidate, candidate_module
            break
        incompatible.append(f"{candidate.name} is missing: {', '.join(missing)}")
    if path is None:
        raise FileNotFoundError(
            "No compatible core module was found. Place 06_dlr_gep_nested.py in the same directory as 07_ablation_study.py.\n"
            + "\n".join(incompatible)
        )
    print(f"Using core module: {path}")
    return module


core = load_core()


def population_size() -> int:
    return QUICK_POPULATION if QUICK_TEST else core.POPULATION_SIZE


def generations() -> int:
    return QUICK_GENERATIONS if QUICK_TEST else core.GENERATIONS


def fixed_probabilities() -> Tuple[float, float]:

    return 0.10, 0.40


def dynamic_probabilities(generation: int) -> Tuple[float, float]:
    total = generations()
    pressure = (generation / max(total - 1, 1)) ** core.PRESSURE_ALPHA
    mutation = core.MUTATION_MAX - (core.MUTATION_MAX - core.MUTATION_MIN) * pressure
    crossover = core.CROSSOVER_MIN + (core.CROSSOVER_MAX - core.CROSSOVER_MIN) * pressure
    return float(mutation), float(crossover)


def generic_terminal_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = []
    for column in range(X.shape[1]):
        if np.std(X[:, column]) < core.EPS:
            corr = 0.0
        else:
            corr = np.corrcoef(X[:, column], y)[0, 1]
        values.append(abs(float(corr)) + 0.05 if np.isfinite(corr) else 0.05)
    result = np.asarray(values, dtype=float)
    return result / result.sum()


def make_child(parent, elites, n_features, weights, mutation, crossover, rng, multi_elite):
    genes = []

    single_guide = elites[0]
    for index in range(core.GENE_COUNT):
        guide = rng.choice(elites) if multi_elite else single_guide
        source = guide.genes[index] if rng.random() < crossover else parent.genes[index]
        gene = (
            core.random_tree(n_features, weights, core.MAX_DEPTH, rng)
            if rng.random() < mutation else deepcopy(source)
        )
        genes.append(gene)
    return core.Chromosome(genes)


def evolve_variant(X_fit, y_fit, X_select, y_select, weights, important, seed, variant):
    rng = random.Random(seed)
    np.random.seed(seed)
    size = population_size()
    seeded_count = min(10, size)
    population = [
        core.seeded_chromosome(np.roll(important, shift), X_fit.shape[1], weights, rng)
        for shift in range(seeded_count)
    ]
    population.extend(core.new_chromosome(X_fit.shape[1], weights, rng) for _ in range(size - seeded_count))
    population = [core.fit_chromosome(x, X_fit, y_fit, X_select, y_select) for x in population]
    best = deepcopy(min(population, key=lambda x: x.objective))
    history, unchanged = [], 0

    for generation in range(generations()):
        population.sort(key=lambda x: x.objective)
        keep = core.ELITE_COUNT if variant["multi"] else 1
        elites = [deepcopy(x) for x in population[:min(keep, len(population))]]
        if elites[0].objective < best.objective - 1e-10:
            best, unchanged = deepcopy(elites[0]), 0
        else:
            unchanged += 1

        mutation, crossover = (
            dynamic_probabilities(generation) if variant["dynamic"] else fixed_probabilities()
        )
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
            print(
                f"      Generation {generation+1:3d}/{generations()}, "
                f"inner={best.inner_rmse:.6f}, score={best.robust_score:.6f}"
            )
        if unchanged >= core.PATIENCE:
            print(f"      Early stopping at generation {generation+1}")
            break

        next_population = elites.copy()
        while len(next_population) < size:
            parent = core.tournament(population, rng)
            child = make_child(
                parent, elites, X_fit.shape[1], weights, mutation, crossover,
                rng, variant["multi"],
            )
            child = core.fit_chromosome(child, X_fit, y_fit, X_select, y_select)
            finite = np.isfinite(child.objective)
            if variant["suppress"]:
                accepted = finite and child.objective <= parent.objective * 1.20
            else:
                accepted = finite
            next_population.append(child if accepted else deepcopy(parent))
        population = next_population
    return best, pd.DataFrame(history)


def construct_inputs(X_train, y_train, X_val, X_test, names, split, dlr, beta, selected, use_dlr):
    if use_dlr:
        p_train = np.maximum(dlr.predict(X_train), 0.0)
        p_val = np.maximum(dlr.predict(X_val), 0.0)
        p_test = np.maximum(dlr.predict(X_test), 0.0)
        raw_train = core.enhance(X_train, p_train, beta, selected)
        raw_val = core.enhance(X_val, p_val, beta, selected)
        raw_test = core.enhance(X_test, p_test, beta, selected)
        enhanced_names = names + [f"DLR_contribution_{names[i]}" for i in selected] + ["DLR_prediction"]
    else:
        raw_train, raw_val, raw_test = X_train, X_val, X_test
        enhanced_names = list(names)

    scaler = StandardScaler().fit(raw_train[:split])
    E_train = scaler.transform(raw_train)
    E_val = scaler.transform(raw_val)
    E_test = scaler.transform(raw_test)
    if use_dlr:
        weights = core.terminal_weights(E_train[:split], y_train[:split], len(names), len(selected))
    else:
        weights = generic_terminal_weights(E_train[:split], y_train[:split])
    return E_train, E_val, E_test, enhanced_names, weights


def run_gep_variant(city, variant, data, dlr, beta, selected, split, lower, upper, out_dir):
    X_train, y_train, X_val, y_val, X_test, y_test, names, _, test_dates = data
    E_train, E_val, E_test, enhanced_names, weights = construct_inputs(
        X_train, y_train, X_val, X_test, names, split, dlr, beta, selected, variant["dlr"]
    )
    important = np.argsort(weights)[::-1]
    runs = []
    for number, seed in enumerate(core.SEEDS, 1):
        print(f"    Run {number}/{len(core.SEEDS)}, seed={seed}")
        started = time.perf_counter()
        structure, history = evolve_variant(
            E_train[:split], y_train[:split], E_train[split:], y_train[split:],
            weights, important, seed, variant,
        )
        model = core.refit_chromosome(structure, E_train, y_train)
        val_pred = np.clip(core.predict_chromosome(model, E_val), lower, upper)
        test_pred = np.clip(core.predict_chromosome(model, E_test), lower, upper)
        robust, seg_mean, seg_std, se, _ = core.segmented_score(y_val, val_pred)
        runs.append({
            "Run": number, "Seed": seed, "Model": model,
            "Validation_Prediction": val_pred, "Test_Prediction": test_pred,
            "Robust_Score": robust, "Segment_Mean": seg_mean,
            "Segment_Std": seg_std, "Standard_Error": se,
            "Validation_Metrics": core.metrics(y_val, val_pred),
            "Test_Metrics": core.metrics(y_test, test_pred),
            "Time_seconds": time.perf_counter() - started,
        })
        history.to_csv(out_dir / f"convergence_seed_{seed}.csv", index=False, encoding="utf-8-sig")
        (out_dir / f"expression_seed_{seed}.txt").write_text(
            core.expression(model, enhanced_names), encoding="utf-8"
        )
        print(f"      validation RMSE={runs[-1]['Validation_Metrics']['RMSE']:.6f}")

    runs.sort(key=lambda x: (x["Robust_Score"], x["Segment_Std"]))
    chosen = runs[:min(core.ENSEMBLE_SIZE, len(runs))]
    selected_seeds = [x["Seed"] for x in chosen]
    test_prediction = np.median(
        np.column_stack([x["Test_Prediction"] for x in chosen]), axis=1
    )
    final = core.metrics(y_test, test_prediction)

    pd.DataFrame([{
        "City": city, "Variant_Code": variant["code"], "Variant": variant["name"],
        "Run": x["Run"], "Seed": x["Seed"],
        "Validation_RMSE": x["Validation_Metrics"]["RMSE"],
        "Validation_R2": x["Validation_Metrics"]["R2"],
        "Robust_Score": x["Robust_Score"], "Segment_Std": x["Segment_Std"],
        "Selected": x["Seed"] in selected_seeds, "Time_seconds": x["Time_seconds"],
        **{f"Test_{k}": v for k, v in x["Test_Metrics"].items()},
    } for x in runs]).to_csv(out_dir / "independent_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Target_Date": test_dates, "Observed_AQI": y_test,
        "Predicted_AQI": test_prediction, "Error": y_test - test_prediction,
    }).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    stability = []
    for metric in ["MSE", "RMSE", "MAE", "MAPE(%)", "R2"]:
        values = np.asarray([x["Test_Metrics"][metric] for x in runs], dtype=float)
        stability.append({
            "City": city, "Variant_Code": variant["code"], "Variant": variant["name"],
            "Metric": metric, "Mean": values.mean(), "Std": values.std(ddof=0),
            "Minimum": values.min(), "Maximum": values.max(), "Number_of_Runs": len(values),
        })
    row = {
        "City": city, "Variant_Code": variant["code"], "Variant": variant["name"],
        "Original_Feature_Count": len(names), "Model_Feature_Count": E_train.shape[1],
        "Selected_Seeds": ",".join(map(str, selected_seeds)), **final,
    }
    return row, stability


def process_city(city: str, city_dir: Path):
    print("\n" + "=" * 82)
    print(f"Ablation study: {city}")
    print("=" * 82)
    data = core.load_city(city_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, names, _, test_dates = data
    split = int(len(y_train) * core.INNER_TRAIN_RATIO)
    dlr, beta = core.fit_dlr(X_train, y_train)
    selected = core.contribution_ranking(X_train, beta)[:min(core.CONTRIBUTION_COUNT, X_train.shape[1])]
    lower = 0.0
    upper = float(np.max(y_train) + 0.25 * np.std(y_train))
    rows, stability = [], []

    for variant in VARIANTS:
        print(f"\n  {variant['code']}: {variant['name']}")
        out_dir = OUTPUT_ROOT / city / f"{variant['code']}_{variant['name'].replace(' ', '_').replace('/', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if variant.get("dlr_only"):
            pred = np.clip(dlr.predict(X_test), lower, upper)
            result = core.metrics(y_test, pred)
            row = {
                "City": city, "Variant_Code": variant["code"], "Variant": variant["name"],
                "Original_Feature_Count": len(names), "Model_Feature_Count": len(names),
                "Selected_Seeds": "", **result,
            }
            pd.DataFrame({
                "Target_Date": test_dates, "Observed_AQI": y_test,
                "Predicted_AQI": pred, "Error": y_test - pred,
            }).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        else:
            row, variant_stability = run_gep_variant(
                city, variant, data, dlr, beta, selected, split, lower, upper, out_dir
            )
            stability.extend(variant_stability)
        rows.append(row)
        print(f"    Test RMSE={row['RMSE']:.6f}, MAE={row['MAE']:.6f}, R2={row['R2']:.6f}")
    return rows, stability


def add_degradation(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Delta_RMSE_vs_Full(%)"] = np.nan
    result["Delta_MAE_vs_Full(%)"] = np.nan
    result["Delta_R2_vs_Full"] = np.nan
    for city in result["City"].unique():
        mask = result["City"] == city
        full = result[mask & (result["Variant_Code"] == "A0")].iloc[0]
        result.loc[mask, "Delta_RMSE_vs_Full(%)"] = (
            (result.loc[mask, "RMSE"] - full["RMSE"]) / full["RMSE"] * 100.0
        )
        result.loc[mask, "Delta_MAE_vs_Full(%)"] = (
            (result.loc[mask, "MAE"] - full["MAE"]) / full["MAE"] * 100.0
        )
        result.loc[mask, "Delta_R2_vs_Full"] = result.loc[mask, "R2"] - full["R2"]
    return result


def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Input directory: {core.DATA_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Run mode: {'quick test (not for reported results)' if QUICK_TEST else 'full ablation study'}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows, all_stability = [], []
    for city, city_dir in core.CITY_DIRECTORIES.items():
        if not city_dir.exists():
            print(f"Skipping  {city}: not found: {city_dir}")
            continue
        rows, stability = process_city(city, city_dir)
        all_rows.extend(rows)
        all_stability.extend(stability)

    if not all_rows:
        raise RuntimeError("No runnable city dataset was found")
    final = pd.DataFrame(all_rows)
    degradation = add_degradation(final)
    stability = pd.DataFrame(all_stability)
    final.to_csv(OUTPUT_ROOT / "ablation_final_results.csv", index=False, encoding="utf-8-sig")
    degradation.to_csv(OUTPUT_ROOT / "ablation_degradation_vs_full.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(OUTPUT_ROOT / "ablation_stability_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "quick_test": QUICK_TEST,
        "seeds": core.SEEDS,
        "population_size": population_size(),
        "generations": generations(),
        "ensemble_size": core.ENSEMBLE_SIZE,
        "selection": "nested chronological validation; test excluded from selection",
        "variants": VARIANTS,
    }
    (OUTPUT_ROOT / "ablation_configuration.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 82)
    print("DLR-GEP ablation summary")
    print("=" * 82)
    columns = ["City", "Variant_Code", "Variant", "RMSE", "MAE", "MAPE(%)", "R2"]
    print(final[columns].to_string(index=False))
    print("\nMain result: ", OUTPUT_ROOT / "ablation_degradation_vs_full.csv")


if __name__ == "__main__":
    main()
