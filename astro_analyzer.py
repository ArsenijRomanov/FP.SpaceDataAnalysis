import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from PIL import Image
from skimage import filters, morphology, measure
import pandas as pd


# ==========================
# Функции для обработки изображений
# ==========================

def load_image_as_gray(path):
    """
    Загрузка TIFF и приведение к градациям серого в диапазоне [0, 1].
    """
    img = Image.open(path).convert("F")  # 32‑битный float grayscale
    arr = np.array(img, dtype=np.float32)
    arr -= arr.min()
    maxv = arr.max()
    if maxv > 0:
        arr /= maxv
    return arr


def generate_tiles(image, tile_size):
    """
    Разбивает изображение на квадраты tile_size x tile_size.
    Возвращает (y0, x0, tile).
    """
    h, w = image.shape
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            tile = image[y0:y0 + tile_size, x0:x0 + tile_size]
            yield y0, x0, tile


def robust_threshold(data, sigma_level=3.0):
    """
    Оценка порога на основе медианы и MAD (устойчиво к выбросам).
    """
    flat = data[np.isfinite(data)].ravel()
    if flat.size == 0:
        return 0.0
    median = np.median(flat)
    mad = np.median(np.abs(flat - median))
    if mad == 0:
        sigma = flat.std() if flat.size > 1 else 0.0
    else:
        sigma = 1.4826 * mad
    return median + sigma_level * sigma


def detect_objects_in_tile(tile_info):
    """
    Обработка одного тайла (вызывается в отдельном процессе).

    tile_info = (image_path, tile_index, y0, x0, tile_array)
    """
    image_path, tile_index, y0, x0, tile = tile_info

    # 1. Размытие с малым и большим масштабом
    smooth_small = filters.gaussian(tile, sigma=1.0, preserve_range=True)
    smooth_large = filters.gaussian(tile, sigma=8.0, preserve_range=True)

    # 2. Высокочастотная составляющая (усиливаем звёзды)
    hp = smooth_small - smooth_large
    hp[hp < 0] = 0

    # 3. Порог для звёзд (robust по MAD)
    thr_star = robust_threshold(hp, sigma_level=3.0)
    stars_mask = hp > thr_star

    # 4. Порог для туманностей (работаем с "медленным" полем яркости)
    try:
        thr_neb = filters.threshold_yen(smooth_small)
        neb_mask = smooth_small > thr_neb
    except Exception:
        neb_mask = np.zeros_like(stars_mask, dtype=bool)

    # 5. Объединяем маски
    mask = stars_mask | neb_mask
    if not mask.any():
        return []

    # 6. Морфологическая очистка
    mask = morphology.remove_small_objects(mask, min_size=5)
    mask = morphology.remove_small_holes(mask, area_threshold=5)
    if not mask.any():
        return []

    # 7. Поиск связных компонент
    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels, intensity_image=smooth_small)

    objects = []
    for region in props:
        if region.area < 5:
            continue

        cy, cx = region.centroid
        global_y = y0 + cy
        global_x = x0 + cx

        flux = region.mean_intensity * region.area
        eccentricity = region.eccentricity
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col
        height = max_row - min_row
        elongation = float(max(width, height)) / max(1.0, min(width, height))

        # Простая классификация по площади и форме
        if region.area < 40 and eccentricity < 0.8 and elongation < 1.6:
            obj_type = "star"
        elif region.area > 200 and elongation > 2.0 and eccentricity > 0.9:
            obj_type = "trail"
        else:
            obj_type = "nebula"

        objects.append(
            {
                "image": os.path.basename(image_path),
                "y": float(global_y),
                "x": float(global_x),
                "area_px": int(region.area),
                "flux": float(flux),
                "mean_intensity": float(region.mean_intensity),
                "max_intensity": float(region.max_intensity),
                "eccentricity": float(eccentricity),
                "elongation": float(elongation),
                "object_type": obj_type,
            }
        )
    return objects


# ==========================
# Параллельная обработка
# ==========================

def process_image_sequential(image_path, tile_size=512):
    """
    Последовательная обработка одного изображения (без параллелизма).
    Удобно для отладки.
    """
    image = load_image_as_gray(image_path)
    all_objects = []
    for idx, (y0, x0, tile) in enumerate(generate_tiles(image, tile_size)):
        tile_info = (image_path, idx, y0, x0, tile)
        objs = detect_objects_in_tile(tile_info)
        all_objects.extend(objs)
    return all_objects


def process_image_parallel(image_path, tile_size=512, max_workers=None):
    """
    Параллельная обработка одного изображения:
    каждый тайл анализируется в отдельном процессе.
    """
    image = load_image_as_gray(image_path)
    tiles = list(generate_tiles(image, tile_size))

    tasks = []
    for idx, (y0, x0, tile) in enumerate(tiles):
        tasks.append((image_path, idx, y0, x0, tile))

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    all_objects = []

    # Контекст нужен для Windows, но и на Linux работает корректно
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(detect_objects_in_tile, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                all_objects.extend(res)

    return all_objects


# ==========================
# Постобработка статистики
# ==========================

def assign_brightness_classes(objects):
    """
    Делим объекты на "faint / medium / bright" по суммарному потоку.
    """
    if not objects:
        return objects

    fluxes = np.array([o["flux"] for o in objects])
    q1 = np.quantile(fluxes, 0.33)
    q2 = np.quantile(fluxes, 0.66)

    for o in objects:
        f = o["flux"]
        if f <= q1:
            cls = "faint"
        elif f <= q2:
            cls = "medium"
        else:
            cls = "bright"
        o["brightness_class"] = cls
    return objects


def run_pipeline(image_paths, tile_size=512, output_csv="objects_stats.csv",
                 parallel=True, max_workers=None):
    """
    Полный конвейер:
    - обработка всех изображений (параллельно по тайлам)
    - объединение результатов
    - вычисление классов по яркости
    - сохранение в CSV
    """
    all_objects = []

    for path in image_paths:
        if parallel:
            objs = process_image_parallel(path, tile_size=tile_size,
                                          max_workers=max_workers)
        else:
            objs = process_image_sequential(path, tile_size=tile_size)

        all_objects.extend(objs)

    assign_brightness_classes(all_objects)
    df = pd.DataFrame(all_objects)
    df = df.drop(columns=["tile_index", "label"], errors="ignore")
    df.to_csv(output_csv, index=False)
    return df


def main():
    """
    Пример запуска:
    python astro_lab.py data/hs-1995-45-a-full_tif.tif data/StarTrails.tif \
           --tile-size 512 --output result.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Параллельный анализ астрономических TIFF‑изображений."
    )
    parser.add_argument("images", nargs="+", help="Пути к входным TIFF‑файлам")
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Размер тайла (по умолчанию 512)"
    )
    parser.add_argument(
        "--output", type=str, default="objects_stats.csv",
        help="Имя выходного CSV‑файла"
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Отключить параллельную обработку (для отладки)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Число воркеров (по умолчанию: число ядер минус 1)"
    )

    args = parser.parse_args()

    df = run_pipeline(
        args.images,
        tile_size=args.tile_size,
        output_csv=args.output,
        parallel=not args.no_parallel,
        max_workers=args.workers,
    )

    print("Найдено объектов:", len(df))
    print("Первые строки таблицы:")
    print(df.head())


if __name__ == "__main__":
    main()
