from pathlib import Path
import copy
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
PREPARED_ROOT = SCRIPT_DIR / "prepared_data"
RESULT_ROOT = SCRIPT_DIR / "results" / "GEP"

TARGET = "Target_AQI"

CITY_NAMES = {
    "Beijing": ["beijing", "Beijing"],
    "Nanning": ["nanning", "Nanning"],
}

RANDOM_SEEDS = [42, 52, 62, 72, 82]

POPULATION_SIZE = 100
GENERATIONS = 100
NUMBER_OF_GENES = 3
HEAD_LENGTH = 10
TAIL_LENGTH = HEAD_LENGTH + 1

TOURNAMENT_SIZE = 5
ELITE_SIZE = 2


MUTATION_PROBABILITY = 0.03
CROSSOVER_PROBABILITY = 0.40
PARSIMONY_COEFFICIENT = 1e-5
EARLY_STOPPING_PATIENCE = 20
MIN_GENERATIONS = 30
VALIDATION_CANDIDATES = 10


CONSTANT_RANGE = (-1.0, 1.0)

FEATURES = []
TERMINALS = []


FUNCTION_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "sqrt": 1,
    "sin": 1,
    "cos": 1,
}

FUNCTIONS = list(FUNCTION_ARITY.keys())


def configure_features(feature_names):
    global FEATURES, TERMINALS

    FEATURES = list(feature_names)

    TERMINALS = [
        f"X{index}"
        for index in range(len(FEATURES))
    ]


def safe_value(value):
    value = np.nan_to_num(
        value,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    return np.clip(
        value,
        -1e4,
        1e4,
    )


def protected_division(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros_like(a, dtype=float),
        where=np.abs(b) > 1e-8,
    )


def random_terminal(rng):
    if not TERMINALS:
        raise ValueError(
            "Input features have not been configured"
        )

    if rng.random() < 0.85:
        return rng.choice(TERMINALS)

    constant = rng.uniform(
        *CONSTANT_RANGE
    )

    return f"C:{constant:.6f}"


def random_head_symbol(rng):
    if rng.random() < 0.50:
        return rng.choice(FUNCTIONS)

    return random_terminal(rng)


def create_gene(rng):
    head = [
        random_head_symbol(rng)
        for _ in range(HEAD_LENGTH)
    ]

    tail = [
        random_terminal(rng)
        for _ in range(TAIL_LENGTH)
    ]

    return head + tail


def create_chromosome(rng):
    return [
        create_gene(rng)
        for _ in range(NUMBER_OF_GENES)
    ]


def evaluate_symbol(
    symbol,
    children,
    x,
):
    if symbol.startswith("X"):
        feature_index = int(symbol[1:])

        return x[:, feature_index]

    if symbol.startswith("C:"):
        constant = float(
            symbol.split(":")[1]
        )

        return np.full(
            len(x),
            constant,
            dtype=float,
        )

    if symbol == "add":
        value = children[0] + children[1]

    elif symbol == "sub":
        value = children[0] - children[1]

    elif symbol == "mul":
        value = children[0] * children[1]

    elif symbol == "div":
        value = protected_division(
            children[0],
            children[1],
        )

    elif symbol == "sqrt":
        value = np.sqrt(
            np.abs(children[0])
        )

    elif symbol == "sin":
        value = np.sin(children[0])

    elif symbol == "cos":
        value = np.cos(children[0])

    else:
        raise ValueError(
            f"Unknown symbol: {symbol}"
        )

    return safe_value(value)


def decode_gene(gene, x):
    position = [0]
    active_nodes = [0]

    def parse():
        if position[0] >= len(gene):
            return np.zeros(len(x))

        symbol = gene[position[0]]

        position[0] += 1
        active_nodes[0] += 1

        if symbol in FUNCTION_ARITY:
            children = [
                parse()
                for _ in range(
                    FUNCTION_ARITY[symbol]
                )
            ]

            return evaluate_symbol(
                symbol,
                children,
                x,
            )

        return evaluate_symbol(
            symbol,
            [],
            x,
        )

    return (
        safe_value(parse()),
        active_nodes[0],
    )


def predict_chromosome(chromosome, x):
    outputs = []
    active_nodes = 0

    for gene in chromosome:
        output, nodes = decode_gene(
            gene,
            x,
        )

        outputs.append(output)
        active_nodes += nodes


    prediction = np.sum(
        outputs,
        axis=0,
    )

    return (
        safe_value(prediction),
        active_nodes,
    )


def fit_linear_scaling(raw_prediction, target):

    z = np.asarray(raw_prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    z_mean = float(np.mean(z))
    y_mean = float(np.mean(y))
    denominator = float(np.sum((z - z_mean) ** 2))

    if denominator <= 1e-12:
        return y_mean, 0.0

    slope = float(np.sum((z - z_mean) * (y - y_mean)) / denominator)
    intercept = y_mean - slope * z_mean
    return intercept, slope


def apply_linear_scaling(raw_prediction, intercept, slope):
    return safe_value(intercept + slope * np.asarray(raw_prediction, dtype=float))


def gene_to_expression(gene):
    position = [0]

    def parse():
        if position[0] >= len(gene):
            return "0"

        symbol = gene[position[0]]
        position[0] += 1

        if symbol.startswith("X"):
            index = int(symbol[1:])

            return FEATURES[index]

        if symbol.startswith("C:"):
            return symbol.split(":")[1]

        children = [
            parse()
            for _ in range(
                FUNCTION_ARITY[symbol]
            )
        ]

        if symbol == "add":
            return (
                f"({children[0]} + "
                f"{children[1]})"
            )

        if symbol == "sub":
            return (
                f"({children[0]} - "
                f"{children[1]})"
            )

        if symbol == "mul":
            return (
                f"({children[0]} * "
                f"{children[1]})"
            )

        if symbol == "div":
            return (
                f"pdiv({children[0]}, "
                f"{children[1]})"
            )

        if symbol == "sqrt":
            return (
                f"sqrt(abs({children[0]}))"
            )

        if symbol == "sin":
            return f"sin({children[0]})"

        if symbol == "cos":
            return f"cos({children[0]})"

        return "0"

    return parse()


def chromosome_to_expression(chromosome):
    expressions = [
        gene_to_expression(gene)
        for gene in chromosome
    ]

    return " + ".join(
        f"({expression})"
        for expression in expressions
    )


def evaluate_fitness(
    chromosome,
    x,
    y,
):
    raw_prediction, active_nodes = (
        predict_chromosome(
            chromosome,
            x,
        )
    )

    intercept, slope = fit_linear_scaling(raw_prediction, y)
    prediction = apply_linear_scaling(raw_prediction, intercept, slope)

    rmse = float(
        np.sqrt(
            np.mean(
                (y - prediction) ** 2
            )
        )
    )

    penalized_rmse = (
        rmse
        + PARSIMONY_COEFFICIENT
        * active_nodes
    )

    fitness = (
        1.0 / (1.0 + penalized_rmse)
    )

    return fitness, rmse, active_nodes, intercept, slope


def tournament_selection(
    population,
    fitness_values,
    rng,
):
    indices = rng.choice(
        len(population),
        size=TOURNAMENT_SIZE,
        replace=False,
    )

    best_index = max(
        indices,
        key=lambda index: (
            fitness_values[index]
        ),
    )

    return copy.deepcopy(
        population[best_index]
    )


def mutate_chromosome(chromosome, rng):
    child = copy.deepcopy(chromosome)

    for gene_index in range(NUMBER_OF_GENES):
        for position in range(HEAD_LENGTH + TAIL_LENGTH):
            if rng.random() < MUTATION_PROBABILITY:
                if position < HEAD_LENGTH:
                    child[gene_index][position] = random_head_symbol(rng)
                else:
                    child[gene_index][position] = random_terminal(rng)

    return child


def crossover(parent_1, parent_2, rng):
    child_1 = copy.deepcopy(parent_1)
    child_2 = copy.deepcopy(parent_2)

    if (
        rng.random()
        >= CROSSOVER_PROBABILITY
    ):
        return child_1, child_2

    gene_index = rng.integers(
        NUMBER_OF_GENES
    )

    point = rng.integers(
        1,
        HEAD_LENGTH + TAIL_LENGTH,
    )

    tail_1 = child_1[
        gene_index
    ][point:]

    tail_2 = child_2[
        gene_index
    ][point:]

    child_1[gene_index][point:] = tail_2
    child_2[gene_index][point:] = tail_1

    return child_1, child_2


class StandardGEPRegressor:

    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.best_chromosome = None
        self.best_training_rmse = np.inf
        self.best_validation_rmse = np.inf
        self.scaling_intercept = 0.0
        self.scaling_slope = 1.0
        self.history = []

    def fit(
        self,
        x_train,
        y_train,
        x_validation,
        y_validation,
    ):
        rng = np.random.default_rng(
            self.random_seed
        )

        population = [
            create_chromosome(rng)
            for _ in range(POPULATION_SIZE)
        ]

        generations_without_improvement = 0

        for generation in range(
            1,
            GENERATIONS + 1,
        ):
            evaluations = [
                evaluate_fitness(
                    chromosome,
                    x_train,
                    y_train,
                )
                for chromosome in population
            ]

            fitness_values = np.asarray(
                [
                    result[0]
                    for result in evaluations
                ]
            )

            training_rmse_values = (
                np.asarray(
                    [
                        result[1]
                        for result in evaluations
                    ]
                )
            )

            ranked_indices = np.argsort(
                fitness_values
            )[::-1]

            generation_improved = False


            for index in ranked_indices[:VALIDATION_CANDIDATES]:
                chromosome = population[index]

                training_raw, _ = predict_chromosome(chromosome, x_train)
                intercept, slope = fit_linear_scaling(training_raw, y_train)
                validation_raw, _ = (
                    predict_chromosome(
                        chromosome,
                        x_validation,
                    )
                )
                validation_prediction = apply_linear_scaling(
                    validation_raw, intercept, slope
                )

                validation_rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                y_validation
                                - validation_prediction
                            ) ** 2
                        )
                    )
                )

                if (
                    validation_rmse
                    < self.best_validation_rmse
                ):
                    self.best_validation_rmse = (
                        validation_rmse
                    )

                    self.best_training_rmse = (
                        training_rmse_values[index]
                    )

                    self.best_chromosome = (
                        copy.deepcopy(
                            chromosome
                        )
                    )
                    self.scaling_intercept = intercept
                    self.scaling_slope = slope
                    generation_improved = True

            if generation_improved:
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            self.history.append(
                {
                    "Generation": generation,
                    "Best_Training_RMSE": (
                        self.best_training_rmse
                    ),
                    "Best_Validation_RMSE": (
                        self.best_validation_rmse
                    ),
                    "Population_Diversity": len(
                        {
                            str(chromosome)
                            for chromosome in population
                        }
                    ) / POPULATION_SIZE,
                }
            )

            next_population = [
                copy.deepcopy(
                    population[index]
                )
                for index in ranked_indices[
                    :ELITE_SIZE
                ]
            ]

            while (
                len(next_population)
                < POPULATION_SIZE
            ):
                parent_1 = tournament_selection(
                    population,
                    fitness_values,
                    rng,
                )

                parent_2 = tournament_selection(
                    population,
                    fitness_values,
                    rng,
                )

                child_1, child_2 = crossover(
                    parent_1,
                    parent_2,
                    rng,
                )

                child_1 = mutate_chromosome(
                    child_1,
                    rng,
                )

                child_2 = mutate_chromosome(
                    child_2,
                    rng,
                )

                next_population.append(child_1)

                if (
                    len(next_population)
                    < POPULATION_SIZE
                ):
                    next_population.append(
                        child_2
                    )

            population = next_population

            if (
                generation == 1
                or generation % 10 == 0
                or generation == GENERATIONS
            ):
                print(
                    f"    Generation "
                    f"{generation:3d}/{GENERATIONS}, "
                    f"train="
                    f"{self.best_training_rmse:.6f}, "
                    f"validation="
                    f"{self.best_validation_rmse:.6f}"
                )

            if (
                generation >= MIN_GENERATIONS
                and generations_without_improvement >= EARLY_STOPPING_PATIENCE
            ):
                print(f"    Early stopping at generation {generation}")
                break

        return self

    def predict(self, x):
        raw_prediction, _ = (
            predict_chromosome(
                self.best_chromosome,
                x,
            )
        )

        prediction = apply_linear_scaling(
            raw_prediction,
            self.scaling_intercept,
            self.scaling_slope,
        )


        return np.clip(prediction, 0.0, 1.0)

    def get_expression(self):
        return chromosome_to_expression(
            self.best_chromosome
        )


def find_city_directory(city):
    for name in CITY_NAMES[city]:
        directory = PREPARED_ROOT / name

        if directory.exists():
            return directory

    return None


def load_feature_names(directory):
    with open(
        directory / "feature_names.json",
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_numeric_column(path, preferred_columns):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for column in preferred_columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path.name} contains missing or nonnumeric values in {column}")
            return values
    raise ValueError(f"{path.name}  does not contain a target column. Available columns: {list(frame.columns)}")


def scale_target(values, target_minimum, target_range):
    return (np.asarray(values, dtype=float) - target_minimum) / target_range


def inverse_target(values, target_minimum, target_range):
    return np.asarray(values, dtype=float) * target_range + target_minimum


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    y_true = np.asarray(y_true)
    mask = np.abs(y_true) > 1e-8

    mape = np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "R2": r2,
    }


def process_city(city):
    directory = find_city_directory(city)

    if directory is None:
        return None

    print("\n" + "=" * 70)
    print(f"Processing city: {city}")
    print("=" * 70)

    feature_names = load_feature_names(
        directory
    )

    configure_features(feature_names)


    train = pd.read_csv(directory / "X_train.csv", encoding="utf-8-sig")
    validation = pd.read_csv(directory / "X_validation.csv", encoding="utf-8-sig")
    test = pd.read_csv(directory / "X_test.csv", encoding="utf-8-sig")

    test_raw = pd.read_csv(
        directory / "test_raw.csv"
    )

    x_train = train[
        feature_names
    ].to_numpy(dtype=float)

    y_train_original = load_numeric_column(
        directory / "y_train.csv", [TARGET, "Observed_AQI", "AQI"]
    )

    x_validation = validation[
        feature_names
    ].to_numpy(dtype=float)

    y_validation_original = load_numeric_column(
        directory / "y_validation.csv", [TARGET, "Observed_AQI", "AQI"]
    )

    x_test = test[
        feature_names
    ].to_numpy(dtype=float)

    y_test_original = load_numeric_column(
        directory / "y_test.csv", [TARGET, "Observed_AQI", "AQI"]
    )


    target_minimum = float(np.min(y_train_original))
    target_maximum = float(np.max(y_train_original))
    target_range = target_maximum - target_minimum
    if target_range <= 1e-12:
        raise ValueError("The training AQI range is zero; Min-Max scaling cannot be applied")

    y_train = scale_target(y_train_original, target_minimum, target_range)
    y_validation = scale_target(y_validation_original, target_minimum, target_range)

    output_directory = RESULT_ROOT / city

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_records = []
    expression_records = []
    run_predictions = []

    for run, seed in enumerate(
        RANDOM_SEEDS,
        start=1,
    ):
        print(f"\nRun {run}/5, seed={seed}")

        start_time = time.time()

        model = StandardGEPRegressor(seed)

        model.fit(
            x_train,
            y_train,
            x_validation,
            y_validation,
        )

        prediction_scaled = (
            model.predict(x_test)
        )

        prediction = inverse_target(
            prediction_scaled,
            target_minimum,
            target_range,
        )
        run_predictions.append(prediction)

        metrics = calculate_metrics(
            y_test_original,
            prediction,
        )

        runtime = time.time() - start_time

        metric_records.append(
            {
                "City": city,
                "Model": "GEP",
                "Run": run,
                "Random_Seed": seed,
                "Number_of_Features": len(
                    feature_names
                ),
                "Runtime_seconds": runtime,
                **metrics,
            }
        )

        expression_records.append(
            {
                "City": city,
                "Run": run,
                "Random_Seed": seed,
                "Training_RMSE_Scaled": (
                    model.best_training_rmse
                ),
                "Validation_RMSE_Scaled": (
                    model.best_validation_rmse
                ),
                "Scaling_Intercept": model.scaling_intercept,
                "Scaling_Slope": model.scaling_slope,
                "Expression": (
                    model.get_expression()
                ),
            }
        )

        prediction_table = pd.DataFrame(
            {
                "Target_Date": (
                    test_raw["Target_Date"]
                ),
                "Observed_AQI": (
                    y_test_original
                ),
                "Predicted_AQI": prediction,
            }
        )

        prediction_table["Residual"] = (
            prediction_table["Observed_AQI"]
            - prediction_table["Predicted_AQI"]
        )

        prediction_table["Absolute_Error"] = (
            np.abs(
                prediction_table["Residual"]
            )
        )

        prediction_table["Squared_Error"] = (
            prediction_table["Residual"] ** 2
        )

        prediction_table["Model"] = "GEP"
        prediction_table["Random_Seed"] = seed

        prediction_table.to_csv(
            output_directory
            / f"GEP_test_predictions_seed_{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            model.history
        ).to_csv(
            output_directory
            / f"GEP_history_seed_{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"  RMSE={metrics['RMSE']:.6f}, "
            f"MAE={metrics['MAE']:.6f}, "
            f"R2={metrics['R2']:.6f}"
        )


    ensemble_prediction = np.median(
        np.vstack(run_predictions),
        axis=0,
    )
    ensemble_metrics = calculate_metrics(
        y_test_original,
        ensemble_prediction,
    )

    ensemble_table = pd.DataFrame(
        {
            "Target_Date": test_raw["Target_Date"],
            "Observed_AQI": y_test_original,
            "Predicted_AQI": ensemble_prediction,
        }
    )
    ensemble_table["Residual"] = (
        ensemble_table["Observed_AQI"]
        - ensemble_table["Predicted_AQI"]
    )
    ensemble_table["Absolute_Error"] = np.abs(ensemble_table["Residual"])
    ensemble_table["Squared_Error"] = ensemble_table["Residual"] ** 2
    ensemble_table["Model"] = "GEP-Ensemble"
    ensemble_table.to_csv(
        output_directory / "GEP_ensemble_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [{
            "City": city,
            "Model": "GEP-Ensemble",
            "Number_of_Runs": len(RANDOM_SEEDS),
            "Aggregation": "Median",
            **ensemble_metrics,
        }]
    ).to_csv(
        output_directory / "GEP_ensemble_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nGEP ensemble test metrics:")
    print(
        f"  RMSE={ensemble_metrics['RMSE']:.6f}, "
        f"MAE={ensemble_metrics['MAE']:.6f}, "
        f"MAPE={ensemble_metrics['MAPE(%)']:.6f}, "
        f"R2={ensemble_metrics['R2']:.6f}"
    )

    metrics_table = pd.DataFrame(
        metric_records
    )

    metrics_table.to_csv(
        output_directory
        / "GEP_test_metrics_all_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        expression_records
    ).to_csv(
        output_directory
        / "GEP_symbolic_expressions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_records = []

    for metric in [
        "MSE",
        "RMSE",
        "MAE",
        "MAPE(%)",
        "R2",
    ]:
        values = metrics_table[
            metric
        ].to_numpy()

        summary_records.append(
            {
                "City": city,
                "Model": "GEP",
                "Metric": metric,
                "Mean": np.mean(values),
                "Std": np.std(values, ddof=1),
                "Minimum": np.min(values),
                "Maximum": np.max(values),
                "Number_of_Runs": len(values),
            }
        )

    summary = pd.DataFrame(
        summary_records
    )

    summary.to_csv(
        output_directory
        / "GEP_test_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nGEP summary:")
    print(summary.to_string(index=False))

    return metrics_table, summary


def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Input directory: {PREPARED_ROOT}")
    print(f"Output directory: {RESULT_ROOT}")

    all_metrics = []
    all_summaries = []

    for city in CITY_NAMES:
        try:
            result = process_city(city)

            if result is not None:
                metrics, summary = result
                all_metrics.append(metrics)
                all_summaries.append(summary)

        except Exception as error:
            print(
                f"\n{city} failed: {error}"
            )

    if not all_metrics:
        print("\nNo GEP results were generated.")
        return

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.concat(
        all_metrics,
        ignore_index=True,
    ).to_csv(
        RESULT_ROOT
        / "GEP_all_cities_all_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined_summary.to_csv(
        RESULT_ROOT
        / "GEP_all_cities_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("GEP baseline summary")
    print("=" * 70)

    print(combined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
