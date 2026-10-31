
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# 경로 설정
SOURCE = Path("dataset/zenodo_subset")
OUTPUT = Path("dataset/yolo_zenodo")

# Instance별 분할
SPLITS = {
    "train": ["001", "002"],
    "val": ["003"]
}

CATEGORIES = ["FewStones", "ManyStones"]


def convert_image(img):
    """TIFF 이미지를 8-bit PNG로 변환"""
    img = img.astype(np.float32)

    # X-ray 밝기 범위를 0~255로 변환
    low, high = np.percentile(img, [1, 99])

    if high <= low:
        return np.zeros(img.shape, dtype=np.uint8)

    img = np.clip((img - low) / (high - low), 0, 1)
    return (img * 255).astype(np.uint8)


def mask_to_yolo(mask):
    """GT 마스크에서 돌 영역별 바운딩박스 생성"""
    binary = (mask > 0).astype(np.uint8)

    height, width = binary.shape

    # 연결된 흰색 영역을 각각 분리
    n, labels, stats, centers = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    yolo_labels = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if area == 0:
            continue

        # YOLO 정규화 좌표
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        nw = w / width
        nh = h / height

        yolo_labels.append(
            f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    return yolo_labels


total_images = 0
total_boxes = 0
empty_images = 0

for split, instances in SPLITS.items():

    image_out = OUTPUT / "images" / split
    label_out = OUTPUT / "labels" / split

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        xray_root = SOURCE / (
            f"{category}InPlayDoh_XrayProjectionsCorrectedSmall"
        )
        gt_root = SOURCE / (
            f"{category}InPlayDoh_GTProjectionsSmall"
        )

        for instance in instances:

            xray_dir = xray_root / f"Instance{instance}"
            gt_dir = gt_root / f"Instance{instance}"

            for xray_path in sorted(xray_dir.glob("*.tiff")):

                # X-ray Angle 번호 추출
                angle = int(
                    xray_path.stem.split("Angle")[1].split("Data")[0]
                )

                gt_path = gt_dir / (
                    f"Instance{instance}Angle{angle:04d}.tiff"
                )

                if not gt_path.exists():
                    print("GT 없음:", gt_path)
                    continue

                # 이미지 읽기
                xray = np.array(Image.open(xray_path))
                gt = np.array(Image.open(gt_path))

                if xray.shape != gt.shape:
                    raise ValueError(
                        f"이미지와 GT 크기 불일치: {xray_path}"
                    )

                # TIFF -> PNG
                png = convert_image(xray)

                # 파일명 중복 방지
                name = f"{category}_Instance{instance}_Angle{angle:05d}"

                Image.fromarray(png).save(
                    image_out / f"{name}.png"
                )

                # GT -> YOLO 라벨
                labels = mask_to_yolo(gt)

                (label_out / f"{name}.txt").write_text(
                    "\n".join(labels),
                    encoding="utf-8"
                )

                total_images += 1
                total_boxes += len(labels)

                if not labels:
                    empty_images += 1

    print(f"{split} 변환 완료")

# data.yaml 생성
yaml_path = OUTPUT / "data.yaml"

yaml_path.write_text(
    f"path: {OUTPUT.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: stone\n",
    encoding="utf-8"
)

print("\n===== 변환 결과 =====")
print("총 이미지:", total_images)
print("총 바운딩박스:", total_boxes)
print("라벨 없는 이미지:", empty_images)
print("저장 위치:", OUTPUT)
