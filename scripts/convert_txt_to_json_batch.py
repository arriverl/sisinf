"""
将赛题原始 txt（含 ``A=...``、``t=...`` 段落）批量转为官方 JSON 列表格式。

用于数据预处理，不参与求解。输出字段含 id, A, t, q, gamma 等。
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_txt_case(text: str) -> Tuple[List[List[int]], List[int]]:
    """从单文件文本正则提取 A 与 t 的 Python 字面量列表。"""
    a_match = re.search(r"A\s*=\s*(\[[\s\S]*?\])\s*t\s*=", text)
    t_match = re.search(r"t\s*=\s*(\[[\s\S]*\])\s*$", text)
    if not a_match or not t_match:
        raise ValueError("未找到合法的 A=... 或 t=... 段落")
    a_raw = ast.literal_eval(a_match.group(1))
    t = ast.literal_eval(t_match.group(1))
    return a_raw, t


def build_instance(
    case_id: int,
    a_raw: List[List[int]],
    t: List[int],
    q: int,
    gamma: int,
    transpose_a: bool,
    require_norm_ge_q2: bool,
) -> Dict:
    if transpose_a:
        a = [list(row) for row in zip(*a_raw)]
    else:
        a = a_raw

    n = len(a)
    m = len(a[0]) if n > 0 else 0
    if len(t) != n:
        raise ValueError(f"t 维度 {len(t)} 与 A 行数 {n} 不一致")

    return {
        "id": case_id,
        "n": n,
        "m": m,
        "q": q,
        "gamma": gamma,
        "require_norm_ge_q2": require_norm_ge_q2,
        "A": a,
        "t": t,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="批量将 sis_inf_problems/*.txt 转成同名 json")
    parser.add_argument(
        "--input-dir",
        default="d:/Code_development/gitproduct/sisinf_challenge2026/saiti1/sis_inf_problems",
        help="txt 输入目录",
    )
    parser.add_argument(
        "--output-dir",
        default="d:/Code_development/gitproduct/sisinf_challenge2026/saiti1/sis_inf_problems_json",
        help="json 输出目录",
    )
    parser.add_argument("--q", type=int, default=100, help="默认模数 q")
    parser.add_argument("--gamma", type=int, default=15, help="默认 gamma")
    parser.add_argument(
        "--transpose-a",
        action="store_true",
        help="若原始 A 按列向量给出，则启用转置为按行矩阵",
    )
    parser.add_argument(
        "--require-norm-ge-q2",
        action="store_true",
        help="是否写入 require_norm_ge_q2=true",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"未在 {input_dir} 找到任何 .txt 文件")

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        a_raw, t = parse_txt_case(text)

        # 从文件名抽取 id，如 problem10.txt -> 10
        m = re.search(r"(\d+)", txt_file.stem)
        case_id = int(m.group(1)) if m else 0

        instance = build_instance(
            case_id=case_id,
            a_raw=a_raw,
            t=t,
            q=args.q,
            gamma=args.gamma,
            transpose_a=args.transpose_a,
            require_norm_ge_q2=args.require_norm_ge_q2,
        )

        out_path = output_dir / f"{txt_file.stem}.json"
        out_path.write_text(
            json.dumps([instance], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] {txt_file.name} -> {out_path.name}")

    print(f"完成：共转换 {len(txt_files)} 个文件，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
