import json

import pytest

from ioqm.detect import detect_run, score_run
from ioqm.generate import REGISTRY, ModelConfig, completed, read_manifest, run_generation
from ioqm.prompts import make_spec


class FakeImage:
    def __init__(self, text):
        self.text = text

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.text)


class FakeBackend:
    def __init__(self, fail_after=None):
        self.config = ModelConfig("fake", "fake/fake", call_kwargs={"steps": 1})
        self.calls = 0
        self.fail_after = fail_after
        self.offload = "none"

    def generate(self, prompt, seed):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated Colab disconnect")
        self.calls += 1
        return FakeImage(f"{prompt}|{seed}")


SPECS = [make_spec([("dog", 2)]), make_spec([("cup", 1), ("apple", 3)]), make_spec([("person", 2)])]


def test_generation_writes_manifest_and_images(tmp_path):
    n = run_generation(FakeBackend(), SPECS, [0, 1], tmp_path, log_every=0)
    assert n == 6
    rows = read_manifest(tmp_path)
    assert len(rows) == 6
    r = rows[0]
    assert r["prompt"] == "2 dogs" and r["objects"] == [["dog", 2]] and r["model"] == "fake"
    assert (tmp_path / "images" / r["image"]).read_text() == "2 dogs|0"
    assert json.loads((tmp_path / "run_config.json").read_text())["model"]["key"] == "fake"


def test_resume_after_disconnect(tmp_path):
    with pytest.raises(RuntimeError):
        run_generation(FakeBackend(fail_after=4), SPECS, [0, 1], tmp_path, log_every=0)
    assert len(completed(tmp_path)) == 4
    b = FakeBackend()
    assert run_generation(b, SPECS, [0, 1], tmp_path, log_every=0) == 2
    assert b.calls == 2
    assert len(read_manifest(tmp_path)) == 6


def test_missing_image_is_regenerated(tmp_path):
    run_generation(FakeBackend(), SPECS, [0], tmp_path, log_every=0)
    victim = read_manifest(tmp_path)[0]["image"]
    (tmp_path / "images" / victim).unlink()
    assert run_generation(FakeBackend(), SPECS, [0], tmp_path, log_every=0) == 1


def test_truncated_manifest_line_tolerated(tmp_path):
    run_generation(FakeBackend(), SPECS, [0], tmp_path, log_every=0)
    with open(tmp_path / "manifest.jsonl", "a") as f:
        f.write('{"image": "trunc')
    assert len(completed(tmp_path)) == 3


def test_config_change_refused(tmp_path):
    run_generation(FakeBackend(), SPECS, [0], tmp_path, log_every=0)
    with pytest.raises(ValueError):
        run_generation(FakeBackend(), SPECS, [0], tmp_path, template="a photo of {objects}", log_every=0)


def test_detect_and_score_end_to_end(tmp_path):
    run_generation(FakeBackend(), SPECS, [0, 1], tmp_path, log_every=0)

    def fake_detector(path, threshold):
        prompt = open(path).read().split("|")[0]
        if prompt == "2 dogs":
            return [{"label": "dog", "score": 0.9, "box": {}}, {"label": "dog", "score": 0.6, "box": {}}]
        if prompt == "2 people":
            return [{"label": "person", "score": 0.9, "box": {}}] * 2
        return [{"label": "cup", "score": 0.9, "box": {}}]

    assert detect_run(tmp_path, fake_detector, "fake_det") == 6
    assert detect_run(tmp_path, fake_detector, "fake_det") == 0  # resumable
    rows = score_run(tmp_path, "fake_det", threshold=0.5)
    by_prompt = {r["image"].split("_s")[0]: r for r in rows}
    assert by_prompt[SPECS[0].prompt_id]["prompt_exact"] == 1.0
    assert by_prompt[SPECS[2].prompt_id]["prompt_exact"] == 1.0  # "people" scored as person
    assert by_prompt[SPECS[1].prompt_id]["type_exact"] == 0.5
    rows_hi = score_run(tmp_path, "fake_det", threshold=0.7, tag="t0.7", save=False)
    assert {r["image"].split("_s")[0]: r for r in rows_hi}[SPECS[0].prompt_id]["prompt_exact"] == 0.0


def test_registry_sane():
    for key, m in REGISTRY.items():
        assert m.key == key and "/" in m.hf_id
        assert m.kind in ("diffusers", "sdxl_ensemble", "api")


def test_images_path_pointer(tmp_path):
    src = tmp_path / "legacy_imgs"
    run_generation(FakeBackend(), SPECS, [0], src, log_every=0)
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.jsonl").write_text((src / "manifest.jsonl").read_text())
    (run / "images_path.txt").write_text(str(src / "images"))
    assert len(completed(run)) == 3
    assert detect_run(run, lambda p, threshold: [{"label": "dog", "score": 0.9, "box": {}}], "d") == 3
