from genmodel_agent import hardware
from genmodel_agent.config import load_config


def test_detect_returns_known_device():
    info = hardware.detect()
    assert info.device in {"cuda", "mps", "cpu"}
    assert hardware.detect("cpu").device == "cpu"


def test_quick_budget_is_smaller_and_scaled():
    full = load_config().budget("mps")
    quick = load_config(quick=True).budget("mps")
    cpu = load_config().budget("cpu")
    assert quick.train_seconds_image < full.train_seconds_image
    assert quick.max_epochs <= 3
    assert cpu.train_seconds_image < full.train_seconds_image


def test_config_override():
    cfg = load_config(seed=1, llm={"enabled": False})
    assert cfg.seed == 1 and cfg.llm.enabled is False
