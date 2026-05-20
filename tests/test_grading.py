from dataclasses import replace
import json

from xdl_densecaps import test_grading
from xdl_densecaps.config import ExperimentConfig, ModelConfig


def test_grading_evaluation_routes_stage1_model_to_single_image_runner(monkeypatch, tmp_path):
    config = replace(ExperimentConfig(), model=ModelConfig(name="densenet121_capsnet"))
    calls = []

    def fake_single_runner(config, *, config_path, split_name):
        calls.append(("single", config.model.name, config_path, split_name))
        return 0

    monkeypatch.setattr(test_grading, "run_single_image_grading_evaluation", fake_single_runner)

    result = test_grading.run_grading_evaluation(config, config_path=tmp_path / "stage1.yaml")

    assert result == 0
    assert calls == [("single", "densenet121_capsnet", tmp_path / "stage1.yaml", "test")]


def test_grading_evaluation_routes_stage2_model_to_paired_runner(monkeypatch, tmp_path):
    config = replace(ExperimentConfig(), model=ModelConfig(name="paired_densenet121_capsnet"))
    calls = []

    def fake_paired_runner(config, *, config_path, split_name):
        calls.append(("paired", config.model.name, config_path, split_name))
        return 0

    monkeypatch.setattr(test_grading, "run_paired_grading_evaluation", fake_paired_runner)

    result = test_grading.run_grading_evaluation(config, config_path=tmp_path / "stage2.yaml")

    assert result == 0
    assert calls == [("paired", "paired_densenet121_capsnet", tmp_path / "stage2.yaml", "test")]


def test_grading_cli_runs_both_stages_in_order_and_applies_path_overrides(monkeypatch, tmp_path):
    stage1_config = _write_config(
        tmp_path / "grading-1.yaml",
        model_name="densenet121_capsnet",
        root_dir="missing-stage1-root",
        output_dir=tmp_path / "stage1-output",
    )
    stage2_config = _write_config(
        tmp_path / "grading-2.yaml",
        model_name="paired_densenet121_capsnet",
        root_dir="missing-stage2-root",
        output_dir=tmp_path / "stage2-output",
        pair_metadata_path="missing-stage2-metadata.json",
    )
    stage1_root = tmp_path / "real-stage1-root"
    stage2_root = tmp_path / "real-stage2-root"
    stage2_metadata = tmp_path / "real-stage2-metadata.json"
    calls = []

    def fake_single_runner(config, *, config_path, split_name):
        calls.append(("single", config_path, config.data.root_dir, config.data.pair_metadata_path, split_name))
        return 0

    def fake_paired_runner(config, *, config_path, split_name):
        calls.append(("paired", config_path, config.data.root_dir, config.data.pair_metadata_path, split_name))
        return 0

    monkeypatch.setattr(test_grading, "run_single_image_grading_evaluation", fake_single_runner)
    monkeypatch.setattr(test_grading, "run_paired_grading_evaluation", fake_paired_runner)

    result = test_grading.main(
        [
            "--stage",
            "all",
            "--stage1-config",
            str(stage1_config),
            "--stage2-config",
            str(stage2_config),
            "--stage1-root",
            str(stage1_root),
            "--stage2-root",
            str(stage2_root),
            "--stage2-metadata",
            str(stage2_metadata),
        ]
    )

    assert result == 0
    assert calls == [
        ("single", stage1_config, str(stage1_root), None, "test"),
        ("paired", stage2_config, str(stage2_root), str(stage2_metadata), "test"),
    ]


def test_portable_pair_metadata_normalizes_windows_style_relative_paths(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    output_dir = tmp_path / "output"
    metadata_path.write_text(
        json.dumps(
            {
                "normal_records": [
                    {
                        "original_path": "data\\uc-grading\\normal\\normal\\image.jpg",
                        "output_path": "artifacts\\grading\\filtered_lesion_regions\\normal_images\\image.png",
                    }
                ],
                "records": [
                    {
                        "original_path": "data\\uc-grading\\lesion\\uc_1\\image.jpg",
                        "output_path": "artifacts\\grading\\filtered_lesion_regions\\images\\image.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = replace(ExperimentConfig(), data=replace(ExperimentConfig().data, pair_metadata_path=str(metadata_path)))

    portable_config = test_grading._with_portable_pair_metadata(config, output_dir=output_dir)
    portable_path = output_dir / "portable_pair_metadata.json"
    payload = json.loads(portable_path.read_text(encoding="utf-8"))

    assert portable_config.data.pair_metadata_path == str(portable_path)
    assert payload["normal_records"][0]["original_path"] == "data/uc-grading/normal/normal/image.jpg"
    assert payload["normal_records"][0]["output_path"] == (
        "artifacts/grading/filtered_lesion_regions/normal_images/image.png"
    )
    assert payload["records"][0]["original_path"] == "data/uc-grading/lesion/uc_1/image.jpg"
    assert payload["records"][0]["output_path"] == "artifacts/grading/filtered_lesion_regions/images/image.png"


def _write_config(
    path,
    *,
    model_name,
    root_dir,
    output_dir,
    pair_metadata_path=None,
):
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  root_dir: {root_dir}",
                f"  pair_metadata_path: {pair_metadata_path or 'null'}",
                "model:",
                f"  name: {model_name}",
                "runtime:",
                f"  output_dir: {output_dir}",
                "  checkpoint_name: best.pt",
            ]
        ),
        encoding="utf-8",
    )
    return path
