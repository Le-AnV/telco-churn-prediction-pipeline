import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# ===================== MODEL LIST =====================


# Tập hợp các mô hình baseline để thử nghiệm
def get_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced"),
        "SVM": SVC(probability=True),
    }
    return models


# ===================== TRAINING & EVALUATION =====================


# Train 1 mô hình với preprocessor
def train_single_model(preprocessor, model, X_train, y_train):
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


# Tính các chỉ số đánh giá
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {"accuracy": acc, "precision": pre, "recall": rec, "f1": f1}
    return metrics


# Train toàn bộ model và trả về dict model + dict metric
def train_all_models(preprocessor, X_train, y_train, X_test, y_test):
    models = get_models()
    trained_models = {}
    results = {}

    for name, model in models.items():
        trained = train_single_model(preprocessor, model, X_train, y_train)
        trained_models[name] = trained
        results[name] = evaluate_model(trained, X_test, y_test)

    return trained_models, results


# ===================== PRINT RESULT TABLE =====================


# In kết quả theo format
def print_result_table(results):

    print(f"{'Model':<18} {'Acc':<10} {'Prec':<10} {'Recall':<10} {'F1':<10}")

    for name, metric in results.items():
        print(
            f"{name:<18} "
            f"{metric['accuracy']:.6f}   "
            f"{metric['precision']:.6f}   "
            f"{metric['recall']:.6f}   "
            f"{metric['f1']:.6f}"
        )
