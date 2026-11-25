"""Small device-portability fixes for the CUDA rerun copy (~/thesis-cuda)."""
from pathlib import Path

root = Path.home() / "thesis-cuda"


def patch(name, old, new, count=1):
    path = root / name
    text = path.read_text()
    if new in text and old not in text:
        print(f"{name}: already patched")
        return
    assert text.count(old) == count, (name, old, text.count(old))
    path.write_text(text.replace(old, new))
    print(f"{name}: patched")


patch("final_summary.py", 'def rescore_h_zero(device="mps"):', 'def rescore_h_zero(device="auto"):')
patch("final_summary.py", '    parser.add_argument("--rescore-h-zero", action="store_true")\n',
      '    parser.add_argument("--rescore-h-zero", action="store_true")\n'
      '    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")\n')
patch("final_summary.py", "        rescore_h_zero()\n", "        rescore_h_zero(args.device)\n")
# The reference columns hold the same configurations measured earlier in this ledger, not MPS values.
patch("smoothing_study.py", "旧MPS速度成功", "既存同一設定の速度成功")
patch("smoothing_study.py", "旧MPS列は同一設定の既存記録（別デバイス）。", "既存同一設定の列はステージ1〜3の同一設定の記録。")
