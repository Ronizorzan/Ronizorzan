"""
Harness de testes para a API FastAPI de risco de crédito.

Cobre práticas de mercado para APIs de ML:
- health/readiness
- contrato de schema (válido e inválido)
- smoke de inferência + determinismo
- métricas de latência (p50/p95) e taxa de sucesso

Uso:
    python teste_api_fastapi.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from api_fastapi import app

METRICS_DIR = Path("interface_and_data")
EXPECTED_UNIQUE_KEYS = {
    "profissao",
    "tiporesidencia",
    "escolaridade",
    "score",
    "estadocivil",
    "produto",
}
PROBABILITY_TOLERANCE = 1e-6

# Amostras derivadas do conjunto histórico de teste
SAMPLE_PAYLOADS = [
    {
        "profissao": "Advogado",
        "tempoprofissao": 39.0,
        "renda": 20860.0,
        "tiporesidencia": "Alugada",
        "escolaridade": "Ens.Fundamental",
        "score": "Baixo",
        "idade": 36,
        "dependentes": 0,
        "estadocivil": "Víuvo",
        "produto": "DoubleDuty",
        "valorsolicitado": 139244.0,
        "valortotalbem": 320000.0,
        "proporcaosolicitadototal": 0.4351375
    },
    {
        "profissao": "Médico",
        "tempoprofissao": 37.0,
        "renda": 5000.0,
        "tiporesidencia": "Própria",
        "escolaridade": "PósouMais",
        "score": "Baixo",
        "idade": 25,
        "dependentes": 0,
        "estadocivil": "Casado",
        "produto": "SpeedFury",
        "valorsolicitado": 100000.0,
        "valortotalbem": 200000.0,
        "proporcaosolicitadototal": 0.5
    },
    {
        "profissao": "Dentista",
        "tempoprofissao": 16.0,
        "renda": 20000.0,
        "tiporesidencia": "Própria",
        "escolaridade": "Superior",
        "score": "MuitoBom",
        "idade": 19,
        "dependentes": 4,
        "estadocivil": "Casado",
        "produto": "ElegantCruise",
        "valorsolicitado": 50000.0,
        "valortotalbem": 200000.0,
        "proporcaosolicitadototal": 0.25
    },
    {
        "profissao": "Contador",
        "tempoprofissao": 0.0,
        "renda": 7000.0,
        "tiporesidencia": "Alugada",
        "escolaridade": "Ens.Fundamental",
        "score": "MuitoBom",
        "idade": 24,
        "dependentes": 2,
        "estadocivil": "Solteiro",
        "produto": "TrailConqueror",
        "valorsolicitado": 200000.0,
        "valortotalbem": 300000.0,
        "proporcaosolicitadototal": 0.6666666666666666
    },
]


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def _record_check(passed: list[str], failed: list[str], name: str, ok: bool, detail: str = "") -> None:
    message = name if not detail else f"{name}: {detail}"
    if ok:
        passed.append(message)
        print(f"[PASS] {message}")
    else:
        failed.append(message)
        print(f"[FAIL] {message}")


def test_health(client: TestClient, passed: list[str], failed: list[str]) -> None:
    response = client.get("/health")
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    ok = (
        response.status_code == 200
        and "healthy" in str(body.get("status", "")).lower()
        and "operational" in str(body.get("reason", "")).lower()
    )
    _record_check(passed, failed, "health_check", ok, f"status={response.status_code} body={body}")


def test_unique_values(client: TestClient, passed: list[str], failed: list[str]) -> None:
    response = client.get("/unique-values")
    body = response.json() if response.status_code == 200 else {}
    has_keys = EXPECTED_UNIQUE_KEYS.issubset(set(body.keys())) if isinstance(body, dict) else False
    ok = response.status_code == 200 and has_keys
    _record_check(
        passed,
        failed,
        "unique_values_contract",
        ok,
        f"status={response.status_code} keys_ok={has_keys}",
    )


def test_predict_valid(client: TestClient, passed: list[str], failed: list[str]) -> dict | None:
    response = client.post("/predict", json=SAMPLE_PAYLOADS[0])
    body = response.json() if response.status_code == 200 else {}
    schema_ok = (
        isinstance(body, dict)
        and isinstance(body.get("prediction"), int)
        and body["prediction"] in (0, 1)
        and isinstance(body.get("probability"), (int, float))
        and 0.0 <= float(body["probability"]) <= 1.0
    )
    ok = response.status_code == 200 and schema_ok
    _record_check(passed, failed, "predict_valid_schema", ok, f"status={response.status_code} body={body}")
    return body if ok else None


def test_predict_invalid(client: TestClient, passed: list[str], failed: list[str]) -> None:
    invalid_payload = {"profissao": "Advogado", "renda": "nao-numerico"}
    response = client.post("/predict", json=invalid_payload)
    ok = response.status_code == 422
    _record_check(passed, failed, "predict_invalid_returns_422", ok, f"status={response.status_code}")


def test_determinism(client: TestClient, passed: list[str], failed: list[str]) -> None:
    payload = SAMPLE_PAYLOADS[1]
    first = client.post("/predict", json=payload)
    second = client.post("/predict", json=payload)
    if first.status_code != 200 or second.status_code != 200:
        _record_check(passed, failed, "predict_determinism", False, "requests não retornaram 200")
        return

    p1 = float(first.json()["probability"])
    p2 = float(second.json()["probability"])
    ok = abs(p1 - p2) <= PROBABILITY_TOLERANCE and first.json()["prediction"] == second.json()["prediction"]
    _record_check(passed, failed, "predict_determinism", ok, f"p1={p1} p2={p2}")


def run_latency_suite(client: TestClient, repeats: int = 3) -> dict:
    """Executa várias inferências e coleta métricas de serving."""
    latencies_ms: list[float] = []
    status_counts: Counter[int] = Counter()
    predictions: list[int] = []
    probabilities: list[float] = []
    errors = 0
    total = 0

    # Aquecimento (não entra nas métricas de latência)
    client.post("/predict", json=SAMPLE_PAYLOADS[0])

    for _ in range(repeats):
        for payload in SAMPLE_PAYLOADS:
            total += 1
            started = time.perf_counter()
            response = client.post("/predict", json=payload)
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies_ms.append(elapsed_ms)
            status_counts[response.status_code] += 1

            if response.status_code != 200:
                errors += 1
                continue

            body = response.json()
            predictions.append(int(body["prediction"]))
            probabilities.append(float(body["probability"]))

    sorted_lat = sorted(latencies_ms)
    success = total - errors
    metrics = {
        "n_requests": total,
        "success_count": success,
        "error_count": errors,
        "success_rate": round(success / total, 4) if total else 0.0,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "http_status_counts": {str(k): v for k, v in sorted(status_counts.items())},
        "latency_ms": {
            "mean": round(statistics.mean(sorted_lat), 3) if sorted_lat else 0.0,
            "p50": round(_percentile(sorted_lat, 0.50), 3) if sorted_lat else 0.0,
            "p95": round(_percentile(sorted_lat, 0.95), 3) if sorted_lat else 0.0,
            "min": round(min(sorted_lat), 3) if sorted_lat else 0.0,
            "max": round(max(sorted_lat), 3) if sorted_lat else 0.0,
        },
        "prediction_distribution": dict(Counter(predictions)),
        "probability_stats": {
            "min": round(min(probabilities), 6) if probabilities else None,
            "mean": round(statistics.mean(probabilities), 6) if probabilities else None,
            "max": round(max(probabilities), 6) if probabilities else None,
        },
    }
    return metrics


def persist_metrics(report: dict) -> tuple[Path, Path]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped_path = METRICS_DIR / f"api_ml_test_metrics_{stamp}.json"
    latest_path = METRICS_DIR / "api_ml_test_metrics_latest.json"

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    stamped_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return stamped_path, latest_path


def main() -> int:
    passed: list[str] = []
    failed: list[str] = []

    print("Iniciando testes da Credit Risk Prediction API...")
    with TestClient(app) as client:
        test_health(client, passed, failed)
        test_unique_values(client, passed, failed)
        test_predict_valid(client, passed, failed)
        test_predict_invalid(client, passed, failed)
        test_determinism(client, passed, failed)
        serving_metrics = run_latency_suite(client, repeats=3)

    latency_ok = serving_metrics["error_count"] == 0 and serving_metrics["n_requests"] > 0
    _record_check(
        passed,
        failed,
        "latency_suite_all_success",
        latency_ok,
        f"errors={serving_metrics['error_count']} n={serving_metrics['n_requests']}",
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "api_version": "1.0.0",
        "passed_checks": passed,
        "failed_checks": failed,
        "all_passed": len(failed) == 0,
        **serving_metrics,
    }

    stamped_path, latest_path = persist_metrics(report)
    print("\n=== Resumo ===")
    print(f"Checks OK: {len(passed)} | Falhas: {len(failed)}")
    print(f"Latência p50/p95 (ms): {serving_metrics['latency_ms']['p50']} / {serving_metrics['latency_ms']['p95']}")
    print(f"Success rate: {serving_metrics['success_rate']}")
    print(f"Métricas salvas em:\n  - {stamped_path}\n  - {latest_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
