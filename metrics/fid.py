from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

fid, _ = calculate_fid(
    '/home/lixiang/data/fid_kolors_torch',
    get_coco_fid_stats()
)
