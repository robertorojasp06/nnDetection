import argparse
import shutil
import json
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class Formatter:
    def __init__(self, num_workers=4) -> None:
        self.image_extension = '.nii.gz'
        self.num_workers = num_workers

    def _format_single_case(self, case):
        """
        case : dict
            'path_to_img': path to input image
            'path_to_gt': path to gt mask
            'path_to_labels': path to json file with labels
            'case_id': id to identify the case/sample.
        """
        print(f"filename: {Path(case['path_to_img']).name}")
        path_to_output_img = self.path_to_output_imgs / f"case_{case['case_id']}_0000{self.image_extension}"
        path_to_output_gt = self.path_to_output_gts / f"case_{case['case_id']}{self.image_extension}"
        path_to_output_label = self.path_to_output_gts / f"case_{case['case_id']}.json"
        # Copy image
        shutil.copyfile(case["path_to_img"], path_to_output_img)
        # Transform mask to have consecutive integers
        with open(case["path_to_labels"], 'r') as file:
            labels = json.load(file)
        # shutil.copyfile(case["path_to_gt"], path_to_output_gt)
        mask_img = sitk.ReadImage(case["path_to_gt"])
        mask_array = sitk.GetArrayFromImage(mask_img)
        new_mask_array = np.zeros.astype(mask_array.dtype)
        # for value, label in label.items():
        #     rows, c

        # Class 0 for tumors, class 1 for adenopathy
        new_labels = {
            value: 0 if label.split(',')[0].strip() in ['p', 'm'] else 1
            for value, label in labels.items()
        }
        new_labels = {"instances": new_labels}
        with open(path_to_output_label, 'w') as file:
            json.dump(new_labels, file, indent=4)

    def format_data(self, path_to_imgs, path_to_gts, path_to_labels,
                    path_to_output, subset='Tr'):
        path_to_output_imgs = Path(path_to_output) / f"images{subset}"
        path_to_output_gts = Path(path_to_output) / f"labels{subset}"
        self.path_to_output_imgs = path_to_output_imgs
        self.path_to_output_gts = path_to_output_gts
        for path in (path_to_output_imgs, path_to_output_gts):            
            path.mkdir(exist_ok=True)
        series_uuids = [
            path.name.split(self.image_extension)[0]
            for path in Path(path_to_imgs).glob(f"*{self.image_extension}")
        ]
        mapping = {
            f"{idx:05d}": uuid
            for idx, uuid in enumerate(series_uuids)
        }
        cases = [
            {
                "path_to_img": Path(path_to_imgs) / f"{uuid}{self.image_extension}",
                "path_to_gt": Path(path_to_gts) / f"{uuid}{self.image_extension}",
                "path_to_labels": Path(path_to_labels) / f"{uuid}.json",
                "case_id": case_id
            }
            for case_id, uuid in mapping.items()
        ]
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            list(tqdm(executor.map(self._format_single_case, cases), total=len(cases)))
        with open(Path(path_to_output) / f"case_to_series_mapping_{subset}.json", 'w') as file:
            json.dump(mapping, file, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="""Give hcuch original data the proper format
        for nnDetection""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--path_to_images_train',
        type=str,
        default="/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-H1R1/data/original/train/nifti/imgs",
        help="Path to the train images."
    )
    parser.add_argument(
        '--path_to_gts_train',
        type=str,
        default="/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-H1R1/data/lesion-instances/train/preprocessed-masks",
        help="Path to the train ground truth masks."
    )
    parser.add_argument(
        '--path_to_labels_train',
        type=str,
        default="/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-H1R1/data/lesion-instances/train/preprocessed-labels",
        help="Path to the train mask labels."
    )
    parser.add_argument(
        '--path_to_images_test',
        type=str,
        default="/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-H1R1/data/original/test/nifti/imgs",
        help="Path to the test images."
    )
    parser.add_argument(
        '--path_to_gts_test',
        type=str,
        default="/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-H1R1/data/lesion-instances/test/preprocessed-masks",
        help="Path to the test ground truth masks."
    )
    parser.add_argument(
        '--path_to_labels_test',
        type=str,
        default="/home/robber/fondef-ID23I10337/resultados/medsam-finetuning/MedSAM-HITL-H1R1/data/lesion-instances/test/preprocessed-labels",
        help="Path to the test mask labels."
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd().parent / 'data',
        help="Path to the output directory."
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default="Task100_fondef",
        help="Task name. Generally, Task[Number]_[Name]."
    )
    args = parser.parse_args()
    # Copy the data
    path_to_output_task = Path(args.path_to_output) / args.task_name
    path_to_output_splitted = Path(path_to_output_task) / "raw_splitted"
    path_to_output_splitted.mkdir(exist_ok=True, parents=True)
    formatter = Formatter()
    formatter.format_data(
        args.path_to_images_train,
        args.path_to_gts_train,
        args.path_to_labels_train,
        path_to_output_splitted,
        subset='Tr'
    )
    formatter.format_data(
        args.path_to_images_test,
        args.path_to_gts_test,
        args.path_to_labels_test,
        path_to_output_splitted,
        subset='Ts'
    )
    # Create dataset json file
    dataset_json = {
        "task": args.task_name,
        "name": "fondef",
        "target_class": None,
        "test_labels": True,
        "labels": {
            "0": "tumor",
            "1": "adenopathy"
        },
        "modalities": {
            "0": "CT"
        },
        "dim": 3
    }
    with open(Path(path_to_output_task) / "dataset.json", 'w') as file:
        json.dump(dataset_json, file, indent=4)


if __name__ == "__main__":
    main()
