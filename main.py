from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import modal

from class_convert import convert_classes

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install(["wget", "libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics", "roboflow", "opencv-python", "pandas"])
    .pip_install("term-image")
    .run_commands(
        "wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && chmod +x /usr/bin/yq"
    )
    .add_local_python_source("class_convert")
)

volume = modal.Volume.from_name("yolo-finetune", create_if_missing=True)
volume_path = Path("/root") / "data"

app = modal.App("detect-holds", image=image, volumes={volume_path.as_posix(): volume})


@dataclass
class DataSetConfig:
    """Information required to download a dataset from Roboflow"""

    workspace_id: str
    project_id: str
    version: int
    format: str
    target_class: str

    @property
    def id(self) -> str:
        return f"{self.workspace_id}/{self.project_id}/{self.version}"


@app.function(
    secrets=[
        modal.Secret.from_name("roboflow-api-key", required_keys=["ROBOFLOW_API_KEY"])
    ]
)
def download_dataset(config: DataSetConfig):
    import os
    import subprocess

    from roboflow import Roboflow

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = (
        rf.workspace(config.workspace_id)
        .project(config.project_id)
        .version(config.version)
    )
    raw_dataset_dir = volume_path / "raw" / "dataset" / config.id
    project.download(config.format, location=str(raw_dataset_dir), overwrite=True)
    # use yq to change the format of the classes from array of classes to index-key dict
    result = subprocess.run(
        ["yq", "-i", ".names |= with_entries(.)", str(raw_dataset_dir / "data.yaml")],
        capture_output=True,
    )
    print(f"yq outputted: {result.stdout}")
    print(f"yq returned {result.returncode} status code")

    # This reclassifies the hold types (of which there are N classes) to 2 classes
    # holds and volumes. This is done to make the model better at detecting hold shapes,
    # and not what type it is.
    convert_classes(
        raw_dataset_dir / "data.yaml",
        {
            "blocker": "hold",
            "crimp": "hold",
            "volume": "volume",
            "jug": "hold",
            "edge": "hold",
            "foothold": "hold",
            "pinch": "hold",
            "pocket": "hold",
            "sloper": "hold",
            "wrap": "hold",
        },
        output_path=volume_path / "dataset" / config.id,
    )


MINUTES = 60
TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"A100:{TRAIN_GPU_COUNT}"
TRAIN_CPU_COUNT = 4


@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    timeout=60 * MINUTES,
)
def train(
    model_id: str,
    dataset: DataSetConfig,
    model_size="yolo11m-seg.pt",
    quick_check=False,
):
    from ultralytics import YOLO

    volume.reload()
    model_path = volume_path / "runs" / model_id
    model_path.mkdir(parents=True, exist_ok=True)

    data_path = volume_path / "dataset" / dataset.id / "data.yaml"

    model = YOLO(model_size)
    model.train(
        data=data_path,
        # fraction=0.4 if not quick_check else 0.04,
        device=list(range(TRAIN_GPU_COUNT)),
        epochs=20,
        imgsz=640,
        # batch=0.95,
        # seed=117,
        workers=max(TRAIN_CPU_COUNT // TRAIN_GPU_COUNT, 1),
        cache=False,
        project=f"{volume_path}/runs",
        name=model_id,
        exist_ok=True,
        verbose=True,
    )


@app.cls(gpu="a10g")
class Inference:
    weights_path: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)

    @modal.method()
    def predict(self, model_id: str, image_path: str, display: bool = False):
        """A simple method for running inference on one image at a time."""
        results = self.model.predict(
            image_path,
            half=True,
            save=True,
            exist_ok=True,
            project=f"{volume_path}/predictions/{model_id}",
        )
        if display:
            from term_image.image import from_file

            terminal_image = from_file(results[0].path)
            terminal_image.draw()


@app.local_entrypoint()
def main(quick_check: bool = True, inference_only: bool = False):
    """Run fine tuning and inference on two datasets.

    Args:
        quick_check: fine-tune on a small subset. Lower quality results, but fast iteration.
        inference_only: skip fine-tuning and only run inference.
    """

    holds = DataSetConfig(
        workspace_id="climbingholds-yasmq",
        project_id="climbingholds-yfkwa",
        version=1,
        format="yolov11",
        target_class="edge",
    )
    datasets = [holds]
    if not inference_only:
        download_dataset.for_each(datasets)

    today = datetime.now().strftime("%Y-%m-%d")
    model_ids = [dataset.id + f"/{today}" for dataset in datasets]

    if not inference_only:
        train.for_each(model_ids, datasets, kwargs={"quick_check": quick_check})

    for model_id, dataset in zip(model_ids, datasets):
        inference = Inference(
            weights_path=str(volume_path / "runs" / model_id / "weights" / "best.pt")
        )

        test_images = volume.listdir(
            str(Path("dataset") / dataset.id / "test" / "images")
        )

        for ii, image in enumerate(test_images):
            print(f"{model_id}: Single image prediction on image", image.path)
            inference.predict.remote(
                model_id=model_id,
                image_path=f"{volume_path}/{image.path}",
                display=(ii == 0),
            )
