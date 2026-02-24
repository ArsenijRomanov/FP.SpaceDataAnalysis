#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import platform
import statistics as stats
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


def count_objects_in_csv(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0
    return max(0, len(rows) - 1)


def run_once(
    analyzer_path: Path,
    images: list[Path],
    tile_size: int,
    output_csv: Path,
    workers: int | None,
    no_parallel: bool,
) -> tuple[float, int, str, str, int]:
    cmd = [sys.executable, str(analyzer_path)]
    cmd += [str(p) for p in images]
    cmd += ["--tile-size", str(tile_size), "--output", str(output_csv)]
    if no_parallel:
        cmd.append("--no-parallel")
    if workers is not None:
        cmd += ["--workers", str(workers)]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()

    secs = t1 - t0
    found = count_objects_in_csv(output_csv) if proc.returncode == 0 else 0
    return secs, found, proc.stdout, proc.stderr, proc.returncode


def format_seconds(x: float) -> str:
    return f"{x:.3f} сек"


def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк для astro_analyzer.py: разные tile_size, 3 прогона, время + число объектов."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Пути к TIFF-изображениям (можно несколько).",
    )
    parser.add_argument(
        "--tile-sizes",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="Список размеров тайла. По умолчанию: 256 512 1024",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Сколько повторов на каждый tile_size (по умолчанию 3).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Число воркеров для astro_analyzer.py (если не задано — как в ядре по умолчанию).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Отключить параллельную обработку (передаётся как --no-parallel).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="tile_benchmark_report.txt",
        help="Файл отчёта (по умолчанию tile_benchmark_report.txt).",
    )
    parser.add_argument(
        "--analyzer",
        type=str,
        default=None,
        help="Путь к astro_analyzer.py (по умолчанию ищется рядом со скриптом бенчмарка).",
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="Не удалять CSV с результатами (иначе они во временной папке и удаляются).",
    )

    args = parser.parse_args()

    images = [Path(p).resolve() for p in args.images]
    for p in images:
        if not p.exists():
            raise FileNotFoundError(f"Не найден файл изображения: {p}")

    if args.analyzer is None:
        analyzer_path = (Path(__file__).resolve().parent / "astro_analyzer.py").resolve()
    else:
        analyzer_path = Path(args.analyzer).resolve()

    if not analyzer_path.exists():
        raise FileNotFoundError(
            f"Не найден astro_analyzer.py: {analyzer_path}\n"
            f"Либо положи tile_benchmark.py рядом с astro_analyzer.py, либо укажи --analyzer"
        )

    report_path = Path(args.report).resolve()

    if args.keep_csv:
        out_dir = Path.cwd() / "tile_benchmark_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="tile_benchmark_")
        out_dir = Path(temp_ctx.name)

    started = datetime.now()

    lines: list[str] = []
    lines.append("=== Бенчмарк tile_size для astro_analyzer.py ===")
    lines.append(f"Дата/время запуска: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append(f"OS: {platform.platform()}")
    lines.append(f"Анализатор: {analyzer_path}")
    lines.append(f"Режим: {'SEQUENTIAL (--no-parallel)' if args.no_parallel else 'PARALLEL'}")
    if args.workers is not None and not args.no_parallel:
        lines.append(f"Workers: {args.workers}")
    lines.append("Изображения:")
    for p in images:
        try:
            size_mb = os.path.getsize(p) / (1024 * 1024)
            lines.append(f"  - {p.name} ({size_mb:.1f} MB)")
        except OSError:
            lines.append(f"  - {p.name}")
    lines.append("")

    summary_rows = []

    for tile in args.tile_sizes:
        lines.append(f"--- tile_size = {tile} ---")
        times = []
        counts = []
        errors = []

        for r in range(1, args.runs + 1):
            out_csv = out_dir / f"objects_tile{tile}_run{r}.csv"

            secs, found, stdout, stderr, code = run_once(
                analyzer_path=analyzer_path,
                images=images,
                tile_size=tile,
                output_csv=out_csv,
                workers=args.workers,
                no_parallel=args.no_parallel,
            )

            if code != 0:
                errors.append((r, code, stderr.strip()[:500]))
                lines.append(f"  Run {r}: ERROR (code={code}), время={format_seconds(secs)}")
            else:
                times.append(secs)
                counts.append(found)
                lines.append(f"  Run {r}: время={format_seconds(secs)}, объектов={found}")

        if times:
            mean_t = stats.mean(times)
            stdev_t = stats.stdev(times) if len(times) >= 2 else 0.0
            mean_c = int(round(stats.mean(counts))) if counts else 0

            lines.append(f"  Среднее время: {format_seconds(mean_t)} (σ={stdev_t:.3f})")
            lines.append(f"  Среднее число объектов: {mean_c}")
            summary_rows.append((tile, mean_t, stdev_t, mean_c, len(times), args.runs))
        else:
            lines.append("  Нет успешных прогонов для этого tile_size.")
            summary_rows.append((tile, float("nan"), float("nan"), 0, 0, args.runs))

        if errors:
            lines.append("  Ошибки:")
            for (r, code, msg) in errors:
                msg = msg if msg else "(stderr пуст)"
                lines.append(f"    - Run {r}: code={code}, stderr: {msg}")

        lines.append("")

    lines.append("=== Сводка ===")
    lines.append("tile_size | avg_time_sec | std_sec | avg_objects | ok_runs/total")
    for tile, mean_t, stdev_t, mean_c, ok, total in summary_rows:
        if mean_t == mean_t:  # not NaN
            lines.append(
                f"{tile:8d} | {mean_t:12.3f} | {stdev_t:7.3f} | {mean_c:11d} | {ok}/{total}"
            )
        else:
            lines.append(f"{tile:8d} | {'-':>12} | {'-':>7} | {mean_c:11d} | {ok}/{total}")

    finished = datetime.now()
    lines.append("")
    lines.append(f"Завершено: {finished.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Длительность (стенка): {(finished - started).total_seconds():.1f} сек")
    if args.keep_csv:
        lines.append(f"CSV сохранены в папке: {out_dir.resolve()}")
    else:
        lines.append("CSV были во временной папке и удалены после завершения.")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Готово. Отчёт записан в: {report_path}")
    if temp_ctx is not None:
        temp_ctx.cleanup()


if __name__ == "__main__":
    main()