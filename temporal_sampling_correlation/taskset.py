import json
import math

import verifiers.v1 as vf

from .prompts import PROMPTS, RHO_GRID, SEEDS, N_NEW, TEMPERATURE, TOP_P


class TemporalData(vf.TaskData):
    prompt_id: int
    rho: float
    seeds: list[int]
    model_name: str
    n_new: int
    temperature: float
    top_p: float


class TemporalTaskConfig(vf.TaskConfig):
    model_name: str = "gpt2"
    n_new: int = N_NEW
    temperature: float = TEMPERATURE
    top_p: float = TOP_P


class TemporalTask(vf.Task[TemporalData, vf.State, TemporalTaskConfig]):
    """One prompt × rho condition; ten source seeds are generated together."""

    async def _rows(self, runtime: vf.Runtime) -> list[dict]:
        raw = await runtime.read("/tmp/tsc_result.json")
        rows = json.loads(raw.decode("utf-8"))
        if not isinstance(rows, list):
            raise ValueError("/tmp/tsc_result.json must contain a JSON list")
        return rows

    def _validate(self, rows: list[dict]) -> bool:
        expected = set(self.data.seeds)
        try:
            got = {int(row["seed"]) for row in rows}
            return (
                len(rows) == len(expected)
                and got == expected
                and all(math.isclose(float(row["rho"]), self.data.rho) for row in rows)
                and all(int(row["prompt_id"]) == self.data.prompt_id for row in rows)
                and all(math.isfinite(float(row["lag1_cdf"])) and math.isfinite(float(row["lag2_cdf"])) for row in rows)
            )
        except (KeyError, TypeError, ValueError):
            return False

    @vf.reward
    async def lag1_cdf_reward(self, runtime: vf.Runtime) -> float:
        """Primary diagnostic reward: mean lag-1 CDF correlation, normalized to [0, 1]."""
        rows = await self._rows(runtime)
        if not self._validate(rows):
            return 0.0
        mean_corr = sum(float(row["lag1_cdf"]) for row in rows) / len(rows)
        return float((mean_corr + 1.0) / 2.0)

    @vf.metric
    async def mean_lag1_cdf(self, runtime: vf.Runtime) -> float:
        rows = await self._rows(runtime)
        if not self._validate(rows):
            return float("nan")
        return float(sum(float(row["lag1_cdf"]) for row in rows) / len(rows))

    @vf.metric
    async def mean_lag2_cdf(self, runtime: vf.Runtime) -> float:
        rows = await self._rows(runtime)
        if not self._validate(rows):
            return float("nan")
        return float(sum(float(row["lag2_cdf"]) for row in rows) / len(rows))


class TemporalConfig(vf.TasksetConfig):
    """User-facing environment configuration."""

    rho: float | None = None
    prompt_ids: list[int] = list(range(len(PROMPTS)))
    rho_grid: list[float] = list(RHO_GRID)
    seeds: list[int] = list(SEEDS)
    task: TemporalTaskConfig = TemporalTaskConfig()


class TemporalSamplingCorrelationTaskset(vf.Taskset[TemporalTask, TemporalConfig]):
    """20 prompts × configurable rho (or the five-value default sweep)."""

    def load(self):
        rho_values = [self.config.rho] if self.config.rho is not None else self.config.rho_grid
        for prompt_id in self.config.prompt_ids:
            for rho in rho_values:
                yield TemporalTask(
                    TemporalData(
                        id=f"p{prompt_id}_rho{rho:g}",
                        prompt=PROMPTS[prompt_id],
                        prompt_id=prompt_id,
                        rho=float(rho),
                        seeds=list(self.config.seeds),
                        model_name=self.config.task.model_name,
                        n_new=self.config.task.n_new,
                        temperature=self.config.task.temperature,
                        top_p=self.config.task.top_p,
                        artifacts=[
                            vf.Artifact(source="/tmp/tsc_result.json"),
                            vf.Artifact(source="/tmp/tsc_result.csv"),
                        ],
                    ),
                    self.config.task,
                )
