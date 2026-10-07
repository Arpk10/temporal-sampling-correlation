import json

import verifiers.v1 as vf
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.clients import ModelContext
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class TemporalHarnessConfig(HarnessConfig):
    device: str = "cuda"


class TemporalSamplingCorrelationHarness(Harness[TemporalHarnessConfig]):
    """Runs the supplied GPT-2 sampler directly inside the rollout runtime."""

    SUPPORTS_MCP = False
    SUPPORTS_RESUME = False

    async def setup(self, runtime: Runtime) -> None:
        return None

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        _, prompt = self.resolve_text_prompt(data)
        payload = {
            "prompt": prompt,
            "prompt_id": data.prompt_id,
            "rho": data.rho,
            "seeds": data.seeds,
            "model_name": data.model_name,
            "n_new": data.n_new,
            "temperature": data.temperature,
            "top_p": data.top_p,
            "device": self.config.device,
        }
        await runtime.write(
            "/tmp/tsc_task.json",
            json.dumps(payload).encode("utf-8"),
        )
        return await runtime.run_program(
            ["python", "-m", "temporal_sampling_correlation.runner"],
            {},
        )


__all__ = ["TemporalSamplingCorrelationHarness"]
