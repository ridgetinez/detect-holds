import yaml
from pathlib import Path
import shutil

DEFAULT_SPEC_FILENAME = "data.yaml"

def convert_classes(dataset_spec_path: Path, mappings: dict[str, str], output_path=None):
    # read the spec file into an in-memory rep
    with open(dataset_spec_path, 'r') as dataset_spec_file:
        dataset_spec = yaml.safe_load(dataset_spec_file)
        print(dataset_spec)

        # create index based mappings array
        dest_classes = list(set(mappings.values()))
        class_mappings = create_integer_class_mapping(
            dataset_spec['names'],
            mappings,
            dest_classes
        )

        if class_mappings == {}:
            print("Already converted.")
            return

        dataset_spec['nc'] = len(dest_classes)
        dataset_spec['names'] = {i:c for i,c in enumerate(dest_classes)}

        label_directories = [dataset_spec['train'], dataset_spec['val'], dataset_spec['test']]
        # print(label_directories)
        for set_relative_path in label_directories:
            # print(dataset_spec_path, set_relative_path, make_set_path(dataset_spec_path, set_relative_path))
            convert_labels(
                class_mappings,
                make_set_path(dataset_spec_path, set_relative_path),
                output_path=make_set_path(output_path / DEFAULT_SPEC_FILENAME, set_relative_path) if output_path else None
            )
        print(dataset_spec)
        if output_path:
            with open(output_path / DEFAULT_SPEC_FILENAME, 'w') as out:
                yaml.dump(dataset_spec, out)

def convert_labels(mappings: dict[int,int], set_dir_path: Path, output_path=None):
    label_dir_path = set_dir_path / "labels"
    if output_path:
        # print(output_path)
        (output_path / "labels").mkdir(parents=True, exist_ok=True)
        (output_path / "images").mkdir(parents=True, exist_ok=True)
    for label_file_path in label_dir_path.iterdir():
        with open(label_file_path, "r") as label_file:
            converted_lines = []
            for line in label_file.readlines():
                src_class, *rest = line.strip().split(" ")
                # potential to write to the file / tmp file directly instead of holding it in memory
                dst_line = " ".join([str(mappings[int(src_class)]), *rest])
                converted_lines.append(dst_line)

            # copy image filepath
            if output_path:
                image_filename = label_file_path.with_suffix(".jpg").name
                output_label_path = output_path / "labels" / label_file_path.name
                output_image_path = output_path / "images" / image_filename
                with open(output_label_path, 'w') as fout:
                    fout.write('\n'.join(converted_lines))
                shutil.copy(Path.resolve(set_dir_path / "images" / image_filename), output_image_path)


def make_set_path(dataset_spec_path: Path, relative_label_path: Path):
    return Path.resolve(dataset_spec_path / relative_label_path / "..")

def create_integer_class_mapping(int_to_src_mapping: dict[int,str], src_to_dest_mapping: dict[str,str], dest_labels: list[str]) -> dict[int,int]:
    return {
        src_int: dest_labels.index(dst)
        for src,dst in src_to_dest_mapping.items()
        for src_int,src_class in int_to_src_mapping.items()
        if src == src_class
    }

if __name__ == '__main__':
    convert_classes(
        Path("/Users/adrianmartinez/Downloads/ClimbingHolds.v1i.yolov11") / "data.yaml",
        {'blocker': 'hold', 'crimp': 'hold', 'volume': 'volume', 'jug': 'hold', 'edge': 'hold', 'foothold': 'hold', 'pinch': 'hold', 'pocket': 'hold', 'sloper': 'hold', 'wrap': 'hold'},
        output_path=Path("/Users/adrianmartinez/projects/detect-holds/dataset-copy")
    )
