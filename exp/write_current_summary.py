#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_result(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def make_summary(pop1024: dict, pop4096: dict) -> str:
    rows = [
        (1024, pop1024),
        (4096, pop4096),
    ]

    def t(result: dict, key: str) -> float:
        return result["timings_ms"][key]

    lines: list[str] = []
    lines.append("# GPU Reproduction Prototype Current Summary")
    lines.append("")
    lines.append("這份 summary 只保留現行無 `validate` 口徑。`exp/repro_proto_bench.cu` 現在量的是：")
    lines.append("")
    lines.append("- CPU preprocess")
    lines.append("- host packing")
    lines.append("- H2D")
    lines.append("- GPU evaluation")
    lines.append("- GPU selection + variation")
    lines.append("- D2H child copy")
    lines.append("- CPU 對照版 selection + variation")
    lines.append("- sequential / overlap wall time")
    lines.append("")
    lines.append("它不再量：")
    lines.append("")
    lines.append("- `validate_genome()`")
    lines.append("- cheap validate")
    lines.append("- fallback to parent")
    lines.append("")
    lines.append("## 1. Current result files")
    lines.append("")
    lines.append("- pop=1024: [current_pop1024.json](/home/hschi1106/g3p-vm-gpu/exp/results/current_pop1024.json)")
    lines.append("- pop=4096: [current_pop4096.json](/home/hschi1106/g3p-vm-gpu/exp/results/current_pop4096.json)")
    lines.append("")
    lines.append("## 2. Raw results")
    lines.append("")
    lines.append("| population | gpu_eval_ms | cpu_preprocess_ms | pack_total_ms | h2d_total_ms | gpu_sel+var_ms | gpu_sel+var+d2h_ms | cpu_sel+var_ms | sequential_wall_ms | overlap_wall_ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pop, result in rows:
        lines.append(
            "| {pop} | {gpu_eval:.3f} | {cpu_pre:.3f} | {pack:.3f} | {h2d:.3f} | {gpu_var:.3f} | {gpu_var_d2h:.3f} | {cpu_var:.3f} | {seq:.3f} | {ov:.3f} |".format(
                pop=pop,
                gpu_eval=t(result, "gpu_eval"),
                cpu_pre=t(result, "cpu_preprocess_total"),
                pack=t(result, "pack_total"),
                h2d=t(result, "h2d_total"),
                gpu_var=t(result, "gpu_selection_variation_total"),
                gpu_var_d2h=t(result, "gpu_selection_variation_d2h_total"),
                cpu_var=t(result, "cpu_selection_variation_total"),
                seq=t(result, "sequential_wall"),
                ov=t(result, "overlap_wall"),
            )
        )
    lines.append("")
    lines.append("## 3. Speedup")
    lines.append("")
    lines.append("### 3.1 Pure selection + variation")
    lines.append("")
    lines.append("公式：")
    lines.append("")
    lines.append("`cpu_sel+var_ms / gpu_sel+var_ms`")
    lines.append("")
    lines.append("| population | speedup |")
    lines.append("| --- | ---: |")
    for pop, result in rows:
        lines.append(
            f"| {pop} | {ratio(t(result, 'cpu_selection_variation_total'), t(result, 'gpu_selection_variation_total')):.3f}x |"
        )
    lines.append("")
    lines.append("### 3.2 Including D2H")
    lines.append("")
    lines.append("公式：")
    lines.append("")
    lines.append("`cpu_sel+var_ms / gpu_sel+var+d2h_ms`")
    lines.append("")
    lines.append("| population | speedup |")
    lines.append("| --- | ---: |")
    for pop, result in rows:
        lines.append(
            f"| {pop} | {ratio(t(result, 'cpu_selection_variation_total'), t(result, 'gpu_selection_variation_d2h_total')):.3f}x |"
        )
    lines.append("")
    lines.append("解讀：")
    lines.append("")
    lines.append("- GPU kernel 本體仍然非常快。")
    lines.append("- `DtoH` 會明顯吃掉一部分優勢，但在 current 口徑下仍保留兩位數 speedup。")
    lines.append("")
    lines.append("## 4. Overlap")
    lines.append("")
    lines.append("### 4.1 Evaluation hide preprocess")
    lines.append("")
    lines.append("公式：")
    lines.append("")
    lines.append("`gpu_eval_ms / cpu_preprocess_ms`")
    lines.append("")
    lines.append("| population | ratio |")
    lines.append("| --- | ---: |")
    for pop, result in rows:
        lines.append(f"| {pop} | {ratio(t(result, 'gpu_eval'), t(result, 'cpu_preprocess_total')):.3f}x |")
    lines.append("")
    lines.append("### 4.2 Evaluation hide preprocess + pack + H2D")
    lines.append("")
    lines.append("公式：")
    lines.append("")
    lines.append("`gpu_eval_ms / (cpu_preprocess_ms + pack_total_ms + h2d_total_ms)`")
    lines.append("")
    lines.append("| population | ratio |")
    lines.append("| --- | ---: |")
    for pop, result in rows:
        denom = t(result, "cpu_preprocess_total") + t(result, "pack_total") + t(result, "h2d_total")
        lines.append(f"| {pop} | {ratio(t(result, 'gpu_eval'), denom):.3f}x |")
    lines.append("")
    lines.append("### 4.3 Sequential vs overlap wall time")
    lines.append("")
    lines.append("公式：")
    lines.append("")
    lines.append("`sequential_wall_ms / overlap_wall_ms`")
    lines.append("")
    lines.append("| population | speedup |")
    lines.append("| --- | ---: |")
    for pop, result in rows:
        lines.append(f"| {pop} | {ratio(t(result, 'sequential_wall'), t(result, 'overlap_wall')):.3f}x |")
    lines.append("")
    lines.append("解讀：")
    lines.append("")
    lines.append("- 在兩個 population 下，`CPU preprocess + packing + H2D` 都仍可被 `GPU evaluation` 掩蓋一部分。")
    lines.append("- `population=4096` 的 margin 較小，但 current benchmark 下仍然成立。")
    lines.append("")
    lines.append("## 5. Current conclusions")
    lines.append("")
    lines.append("1. 現行無 validate 口徑下，GPU reproduction-style variation 有很強的速度優勢。")
    lines.append("2. 單看 kernel，本體仍是高倍數 speedup。")
    lines.append("3. 把 `DtoH` 算進去後，GPU reproduction path 仍然明顯快於 CPU。")
    lines.append("4. `CPU preprocess + packing + H2D` 目前仍能與 `GPU evaluation` overlap。")
    lines.append("5. current 敘事應該是：GPU reproduction path 的主要價值在於把 selection + variation 壓到極低成本，並且前處理仍可被 evaluation 掩蓋一部分。")
    lines.append("")
    lines.append("## 6. Next work")
    lines.append("")
    lines.append("1. 讓 `exp/repro_proto_bench.cu` 與正式 evolution loop 共用更多 preprocessing / packing 元件，減少 bench-only 邏輯。")
    lines.append("2. 針對更大的 population 再量一次，確認 overlap margin 何時開始明顯收斂。")
    lines.append("3. 若要往正式整合前進，優先量 child copyback 與 host-side compile 接入後的成本，而不是重新引入 validate path。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    pop1024 = load_result("current_pop1024.json")
    pop4096 = load_result("current_pop4096.json")
    summary = make_summary(pop1024, pop4096)
    out_path = RESULTS / "complete_summary.md"
    out_path.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
