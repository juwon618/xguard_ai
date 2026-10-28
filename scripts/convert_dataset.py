
from pathlib import Path
from collections import defaultdict
from PIL import Image
import random
import shutil
import yaml

# 1. 경로 설정
ROOT = Path("dataset/croppedgdxray/cropped_castings_128_256")
OUTPUT = Path("dataset/yolo_gdxray")

# 2. 설정
SEED = 42
VAL_RATIO = 0.2
CLASS_NAME = "casting_defect"


def read_series(filename):
    """시리즈 목록 읽기"""
    path = ROOT / filename

    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_ground_truth(path):
    """시리즈별 바운딩박스 읽기"""
    boxes = defaultdict(list)

    if not path.exists():
        raise FileNotFoundError(f"라벨 없음: {path}")

    for line in path.read_text().splitlines():
        values = line.split()

        if not values:
            continue

        if len(values) != 5:
            raise ValueError(f"라벨 형식 오류: {line}")

        image_id, ymin, ymax, xmin, xmax = map(float, values)

        boxes[int(image_id)].append(
            (xmin, ymin, xmax, ymax)
        )

    return boxes


def convert_box(box, width, height):
    """픽셀 좌표를 YOLO 좌표로 변환"""
    xmin, ymin, xmax, ymax = box

    if not (
        0 <= xmin < xmax <= width
        and 0 <= ymin < ymax <= height
    ):
        raise ValueError(f"잘못된 좌표: {box}")

    x_center = (xmin + xmax) / 2 / width
    y_center = (ymin + ymax) / 2 / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return (
        f"0 {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def main():
    train_series = read_series("series.txt")
    test_series = read_series("test-series.txt")

    if set(train_series) & set(test_series):
        raise ValueError("학습/테스트 시리즈가 겹칩니다.")

    random.seed(SEED)
    random.shuffle(train_series)

    val_count = max(1, round(len(train_series) * VAL_RATIO))

    val_series = train_series[:val_count]
    train_series = train_series[val_count:]

    if not train_series:
        raise ValueError("학습 시리즈가 없습니다.")

    splits = {
        "train": train_series,
        "val": val_series,
        "test": test_series
    }

    # 변환 전에 폴더 및 라벨 무결성 확인
    all_series = train_series + val_series + test_series

    for series in all_series:
        folder = ROOT / series
        if not folder.is_dir():
            raise FileNotFoundError(f"시리즈 폴더 없음: {folder}")

        gt = load_ground_truth(folder / "ground_truth.txt")

        images = list(folder.glob("*.png"))
        image_ids = {int(p.stem.split("_")[-1]) for p in images}

        missing = set(gt.keys()) - image_ids
        if missing:
            raise ValueError(
                f"{series}: 라벨 ID에 해당하는 이미지 없음: {sorted(missing)[:10]}"
            )

        for image_path in images:
            image_id = int(image_path.stem.split("_")[-1])
            with Image.open(image_path) as img:
                w, h = img.size

            for box in gt.get(image_id, []):
                convert_box(box, w, h)

    # 출력 폴더 생성
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} 폴더가 이미 있습니다. "
            "기존 결과를 확인하고 이름을 바꾸거나 삭제하세요."
        )

    stats = {}

    for split, series_list in splits.items():
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        box_count = 0
        negative_count = 0

        for series in series_list:
            folder = ROOT / series
            gt = load_ground_truth(folder / "ground_truth.txt")

            for image_path in sorted(folder.glob("*.png")):
                image_id = int(image_path.stem.split("_")[-1])

                with Image.open(image_path) as img:
                    width, height = img.size

                label_lines = [
                    convert_box(box, width, height)
                    for box in gt.get(image_id, [])
                ]

                # 같은 이름으로 이미지 복사
                shutil.copy2(image_path, image_dir / image_path.name)

                # 이미지별 YOLO 라벨 생성
                label_path = label_dir / f"{image_path.stem}.txt"
                label_path.write_text(
                    "\n".join(label_lines),
                    encoding="utf-8"
                )

                image_count += 1
                box_count += len(label_lines)

                if not label_lines:
                    negative_count += 1

        stats[split] = (image_count, box_count, negative_count)

    # YOLO 설정 파일 생성
    config = {
        "path": str(OUTPUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: CLASS_NAME}
    }

    with open(OUTPUT / "data.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\n=== 변환 완료 ===")

    for split, (images, boxes, negatives) in stats.items():
        print(
            f"{split}: {images} images, "
            f"{boxes} boxes, {negatives} unlabeled images"
        )

    print(f"\n설정 파일: {OUTPUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
