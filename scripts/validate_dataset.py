
from pathlib import Path
from PIL import Image
from collections import Counter

ROOT = Path("dataset/yolo_gdxray")

for split in ["train", "val", "test"]:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = list(image_dir.glob("*.png"))
    errors = []
    positive = 0
    negative = 0

    for image in images:
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            errors.append(f"라벨 없음: {image.name}")
            continue

        lines = label.read_text().splitlines()

        if lines:
            positive += 1
        else:
            negative += 1

        for line in lines:
            values = line.split()

            if len(values) != 5:
                errors.append(f"형식 오류: {label.name}")
                continue

            cls, x, y, w, h = map(float, values)

            if not (
                cls == 0
                and 0 < w <= 1
                and 0 < h <= 1
                and 0 <= x - w / 2
                and x + w / 2 <= 1
                and 0 <= y - h / 2
                and y + h / 2 <= 1
            ):
                errors.append(f"좌표 오류: {label.name}")

    print(f"\n[{split}]")
    print("전체 이미지:", len(images))
    print("결함 이미지:", positive)
    print("라벨 없는 이미지:", negative)
    print("오류 개수:", len(errors))
    print("오류 예시:", errors[:5])

# 서로 다른 분할에 같은 파일명이 들어갔는지 확인
split_names = {
    split: {p.name for p in (ROOT / "images" / split).glob("*.png")}
    for split in ["train", "val", "test"]
}

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = split_names[a] & split_names[b]
    print(f"{a}/{b} 파일명 중복:", len(overlap))
