import math


def _validate(rows, expected_seeds, rho, prompt_id):
    expected = set(expected_seeds)
    try:
        got = {int(row["seed"]) for row in rows}
        return (
            len(rows) == len(expected)
            and got == expected
            and all(math.isclose(float(row["rho"]), rho) for row in rows)
            and all(int(row["prompt_id"]) == prompt_id for row in rows)
            and all(
                math.isfinite(float(row["lag1_cdf"]))
                and math.isfinite(float(row["lag2_cdf"]))
                for row in rows
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def test_result_validation_accepts_complete_finite_rows():
    rows = [{"seed": i, "rho": .5, "prompt_id": 2, "lag1_cdf": .1, "lag2_cdf": .05} for i in range(10)]
    assert _validate(rows, range(10), .5, 2)


def test_result_validation_rejects_nonfinite_secondary_metric():
    rows = [{"seed": i, "rho": .5, "prompt_id": 2, "lag1_cdf": .1, "lag2_cdf": .05} for i in range(10)]
    rows[4]["lag2_cdf"] = float("nan")
    assert not _validate(rows, range(10), .5, 2)


def test_result_validation_rejects_wrong_seed_set():
    rows = [{"seed": i, "rho": .5, "prompt_id": 2, "lag1_cdf": .1, "lag2_cdf": .05} for i in range(9)]
    rows.append({"seed": 99, "rho": .5, "prompt_id": 2, "lag1_cdf": .1, "lag2_cdf": .05})
    assert not _validate(rows, range(10), .5, 2)
