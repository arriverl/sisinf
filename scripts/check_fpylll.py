"""检测 fpylll 是否可用，并试跑一小次 BKZ。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        from fpylll import BKZ, IntegerMatrix, LLL

        print("OK: fpylll 已安装，可使用真 BKZ/LLL。")
        M = IntegerMatrix(4, 4)
        for i in range(4):
            M[i, i] = 1
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=2))
        print("OK: LLL + BKZ 试跑成功。")
        return
    except ImportError as e:
        print("未安装 fpylll:", e)
    except Exception as e:
        print("fpylll 已导入但运行失败:", e)

    print("\n安装建议（任选其一）：")
    print("  conda install -c conda-forge fpylll")
    print("  WSL: pip install fpylll")
    print("  或运行: ..\\setup_fpylll.ps1")
    sys.exit(1)


if __name__ == "__main__":
    main()
