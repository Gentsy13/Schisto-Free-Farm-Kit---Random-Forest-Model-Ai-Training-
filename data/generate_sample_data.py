"""
Generate the sample schistosomiasis risk dataset (data/schisto_risk.csv).

Run once to create the CSV used by the training pipeline:
    python data/generate_sample_data.py
"""

import random
import csv
import os

random.seed(42)

FIELDNAMES = [
    "proximity_to_water_m",
    "water_contact_hrs_per_day",
    "irrigation_type",        # 0 = drip, 1 = flood, 2 = sprinkler
    "soil_moisture_pct",
    "temperature_c",
    "rainfall_mm",
    "snail_habitat_index",    # 0.0 – 1.0 (higher = more suitable for snails)
    "sanitation_score",       # 0 – 10 (higher = better sanitation)
    "age_group",              # 0 = child (<15), 1 = adult (15-60), 2 = elder (>60)
    "ppe_usage",              # 0 = no, 1 = yes
    "schisto_risk",           # TARGET: 0 = low risk, 1 = high risk
]

ROWS = 1000


def _risk_label(row: dict) -> int:
    """Simple deterministic rule to assign a risk label for the synthetic data."""
    score = 0
    if row["proximity_to_water_m"] < 100:
        score += 2
    if row["water_contact_hrs_per_day"] > 2:
        score += 2
    if row["irrigation_type"] == 1:  # flood irrigation
        score += 1
    if row["snail_habitat_index"] > 0.6:
        score += 2
    if row["sanitation_score"] < 5:
        score += 1
    if row["ppe_usage"] == 0:
        score += 1
    if row["age_group"] == 0:  # children are more vulnerable
        score += 1
    # Add a small random component
    score += random.randint(0, 1)
    return 1 if score >= 5 else 0


def generate(output_path: str) -> None:
    rows = []
    for _ in range(ROWS):
        row = {
            "proximity_to_water_m": random.randint(5, 500),
            "water_contact_hrs_per_day": round(random.uniform(0, 8), 1),
            "irrigation_type": random.randint(0, 2),
            "soil_moisture_pct": round(random.uniform(10, 90), 1),
            "temperature_c": round(random.uniform(18, 40), 1),
            "rainfall_mm": round(random.uniform(0, 300), 1),
            "snail_habitat_index": round(random.uniform(0, 1), 2),
            "sanitation_score": random.randint(0, 10),
            "age_group": random.randint(0, 2),
            "ppe_usage": random.randint(0, 1),
        }
        row["schisto_risk"] = _risk_label(row)
        rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sample dataset written to {output_path} ({ROWS} rows)")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate(os.path.join(script_dir, "schisto_risk.csv"))
