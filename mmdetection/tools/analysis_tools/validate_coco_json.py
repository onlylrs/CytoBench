import json
import numbers
import sys
from typing import Any, Dict, List


def is_polygon(segmentation: Any) -> bool:
    if not isinstance(segmentation, list):
        return False
    if len(segmentation) == 0:
        return False
    for polygon in segmentation:
        if not isinstance(polygon, list):
            return False
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            return False
        for coordinate in polygon:
            if not isinstance(coordinate, numbers.Number):
                return False
    return True


def is_rle(segmentation: Any) -> bool:
    if not isinstance(segmentation, dict):
        return False
    if 'size' not in segmentation or 'counts' not in segmentation:
        return False
    size = segmentation['size']
    counts = segmentation['counts']
    if not isinstance(size, list) or len(size) != 2:
        return False
    # pycocotools accepts compressed (str/bytes) and uncompressed (list of ints) RLE
    if not (isinstance(counts, (str, bytes)) or isinstance(counts, list)):
        return False
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python validate_coco_json.py /path/to/annotations.json')
        sys.exit(1)

    ann_path = sys.argv[1]
    with open(ann_path, 'r') as f:
        data = json.load(f)

    annotations: List[Dict[str, Any]] = data.get('annotations', [])
    bad: List[Dict[str, Any]] = []
    for ann in annotations:
        seg = ann.get('segmentation')
        iscrowd = ann.get('iscrowd', 0)
        ok = is_rle(seg) if iscrowd == 1 else is_polygon(seg)
        if not ok:
            bad.append({
                'ann_id': ann.get('id'),
                'image_id': ann.get('image_id'),
                'iscrowd': iscrowd,
                'seg_type': type(seg).__name__,
                'seg_preview': str(seg)[:240]
            })

    print(f'bad_count={len(bad)}')
    for entry in bad[:50]:
        print(entry)


if __name__ == '__main__':
    main()



