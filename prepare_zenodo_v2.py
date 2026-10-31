
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

ZIP = Path.home() / "Downloads/XrayObjectDetectionDataset.zip"
OUT = Path("dataset/yolo_zenodo_v2")

# Instance 단위 분할
SPLITS = {
    "train": {
        "FewStones": range(1, 31),
        "ManyStones": range(1, 9),
    },
    "val": {
        "FewStones": range(31, 36),
        "ManyStones": range(9, 11),
    },
    "test": {
        "FewStones": range(36, 41),
        "ManyStones": range(11, 13),
    },
}

ANGLES_PER_INSTANCE = 40


def to_png(img):
    img = img.astype(np.float32)
    low, high = np.percentile(img, [1, 99])

    if high <= low:
        return np.zeros(img.shape, dtype=np.uint8)

    img = np.clip((img - low) / (high - low), 0, 1)
    return (img * 255).astype(np.uint8)


def mask_to_labels(mask):
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    labels = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if area == 0:
            continue

        cx = (x + w / 2) / width
        cy = (y + h / 2) / height

        labels.append(
            f"0 {cx:.6f} {cy:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )

    return labels


counts = {}

with ZipFile(ZIP, "r") as zf:
    # ZIP 파일 목록을 메모리에서 조회
    members = set(zf.namelist())

    for split, categories in SPLITS.items():
        image_out = OUT / "images" / split
        label_out = OUT / "labels" / split

        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        images_count = 0
        boxes_count = 0
        empty_count = 0

        for category, instance_ids in categories.items():
            xray_folder = (
                f"{category}InPlayDoh_"
                "XrayProjectionsCorrectedSmall"
            )
            gt_folder = (
                f"{category}InPlayDoh_GTProjectionsSmall"
            )

            for number in instance_ids:
                instance = f"{number:03d}"

                prefix = (
                    f"{xray_folder}/Instance{instance}/"
                )

                pairs = []

                # 동일한 Instance와 Angle의 X-ray / GT 매칭
                for name in members:
                    if not name.startswith(prefix):
                        continue

                    if not name.endswith("Data.tiff"):
                        continue

                    try:
                        angle = int(
                            name.split("Angle")[-1]
                            .replace("Data.tiff", "")
                        )
                    except ValueError:
                        continue

                    gt_name = (
                        f"{gt_folder}/Instance{instance}/"
                        f"Instance{instance}"
                        f"Angle{angle:04d}.tiff"
                    )

                    if gt_name in members:
                        pairs.append((angle, name, gt_name))

                pairs.sort(key=lambda x: x[0])

                if not pairs:
                    print(f"매칭 파일 없음: {category} {instance}")
                    continue

                # 전체 촬영 각도에 걸쳐 고르게 추출
                indices = np.linspace(
                    0,
                    len(pairs) - 1,
                    min(ANGLES_PER_INSTANCE, len(pairs)),
                    dtype=int
                )

                for idx in indices:
                    angle, xray_name, gt_name = pairs[idx]

                    # ZIP에서 직접 읽기 (중간 TIFF 저장 없음)
                    with Image.open(
                        BytesIO(zf.read(xray_name))
                    ) as img:
                        xray = np.array(img)

                    with Image.open(
                        BytesIO(zf.read(gt_name))
                    ) as img:
                        gt = np.array(img)

                    if xray.shape != gt.shape:
                        raise ValueError(
                            f"이미지 크기 불일치: {xray_name}"
                        )

                    png = to_png(xray)
                    labels = mask_to_labels(gt)

                    filename = (
                        f"{category}_Instance{instance}"
                        f"_Angle{angle:05d}"
                    )

                    Image.fromarray(png).save(
                        image_out / f"{filename}.png"
                    )

                    (label_out / f"{filename}.txt").write_text(
                        "\n".join(labels),
                        encoding="utf-8"
                    )

                    images_count += 1
                    boxes_count += len(labels)

                    if not labels:
                        empty_count += 1

                print(
                    f"{split}: {category} "
                    f"Instance{instance} 완료"
                )

        counts[split] = images_count

        print(
            f"\n[{split}] 이미지 {images_count}장, "
            f"박스 {boxes_count}개, "
            f"빈 라벨 {empty_count}장\n"
        )

# YOLO 데이터셋 설정
(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: stone\n",
    encoding="utf-8"
)

print("===== 데이터셋 생성 완료 =====")
for split, count in counts.items():
    print(f"{split}: {count}장")

print("저장 경로:", OUT)
