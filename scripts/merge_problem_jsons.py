import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_instances_from_file(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{path} 内容既不是 list 也不是 dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="合并多个单题 json 为一个 instances_all.json")
    parser.add_argument(
        "--input-dir",
        default="d:/Code_development/gitproduct/sisinf_challenge2026/saiti1/sis_inf_problems_json",
        help="单题 json 所在目录",
    )
    parser.add_argument(
        "--output",
        default="d:/Code_development/gitproduct/sisinf_challenge2026/saiti1/instances_all.json",
        help="合并输出文件路径",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"未找到 json 文件：{input_dir}")

    merged: List[Dict[str, Any]] = []
    for f in files:
        merged.extend(load_instances_from_file(f))

    merged.sort(key=lambda x: int(x.get("id", 10**9)))
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：合并 {len(files)} 个文件，共 {len(merged)} 个实例 -> {output_path}")


if __name__ == "__main__":
    main()
