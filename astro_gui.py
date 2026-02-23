#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astro Analyzer GUI (PySide6)
- Выбор папки/файла с TIFF
- Запуск анализа (запускает astro_analyzer.py как отдельный процесс)
- Индикатор выполнения (спиннер)
- Просмотр CSV-таблицы с сортировкой и фильтрами по колонкам
- Просмотр изображения рядом с таблицей, фильтрация строк по выбранному файлу
- Кроссхейр (вертикальная/горизонтальная линия) и координаты курсора поверх изображения
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PySide6.QtCore import QRectF

from PySide6.QtCore import (
    QAbstractTableModel,
    QProcess,
    QProcessEnvironment,
    QModelIndex,
    QObject,
    Qt,
    QSortFilterProxyModel,
    QTimer,
)
from PySide6.QtGui import (
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QColor,
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QFrame,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

# Hide noisy runtime warnings in the GUI process.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Lines matching this regex will be dropped from the GUI log (to remove noisy warnings).
_LOG_DROP_RE = re.compile(
    r"(DeprecationWarning|FutureWarning|UserWarning|RuntimeWarning):|^\s*WARNING\b",
    re.IGNORECASE,
)

# Optional deps (for robust TIFF load)
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # noqa: N816


SUPPORTED_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}


def human_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def list_tiff_images(folder: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix in SUPPORTED_EXTS:
            files.append(p)
    return files


def pil_to_qpixmap(path: Path, max_side: int = 4096, return_meta: bool = False):
    """
    Loads TIFF (or any PIL-readable image) using Pillow and converts to QPixmap.
    Downscales very large images to max_side for smooth GUI work.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required to load TIFF reliably. Install: pip install pillow")

    img = Image.open(path)

    orig_w, orig_h = img.size

    # Some TIFFs are multi-page; show first page by default.
    try:
        img.seek(0)
    except Exception:
        pass

    # Convert to RGB for simpler display
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        # grayscale -> RGB to draw nicely
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)))

    w, h = img.size
    data = img.tobytes("raw", "RGB")
    bytes_per_line = 3 * w
    qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)

    if return_meta:
        sx = 1.0 if orig_w <= 0 else (w / float(orig_w))
        sy = 1.0 if orig_h <= 0 else (h / float(orig_h))
        return pix, (sx, sy)

    return pix


class SpinnerWidget(QWidget):
    """
    Small rotating spinner (busy indicator) without external assets.
    """
    def __init__(self, parent: Optional[QWidget] = None, size: int = 20) -> None:
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._running = False
        self._size = size
        self.setFixedSize(size, size)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._timer.start(50)
        self.update()

    def stop(self) -> None:
        self._running = False
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        if not self._running:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        cx = self.width() / 2
        cy = self.height() / 2
        radius = min(self.width(), self.height()) / 2 - 2

        # Draw 12 ticks with fading alpha
        for i in range(12):
            alpha = int(255 * (i + 1) / 12)
            pen = QPen(self.palette().windowText().color())
            pen.setWidth(2)
            c = pen.color()
            c.setAlpha(alpha)
            pen.setColor(c)
            painter.setPen(pen)

            a = (self._angle + i * 30) * 3.14159 / 180.0
            x1 = cx + radius * 0.55 * (float(__import__("math").cos(a)))
            y1 = cy + radius * 0.55 * (float(__import__("math").sin(a)))
            x2 = cx + radius * 0.95 * (float(__import__("math").cos(a)))
            y2 = cy + radius * 0.95 * (float(__import__("math").sin(a)))
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


class PandasTableModel(QAbstractTableModel):
    """
    Lightweight model for a pandas DataFrame-like object.
    We avoid importing pandas here; we accept any object with:
      - .columns (list-like)
      - .shape (tuple)
      - .iloc[row, col]
    """
    def __init__(self, df: Any, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(self._df.shape[0])

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(self._df.shape[1])

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: ANN401
        if not index.isValid():
            return None
        val = self._df.iloc[index.row(), index.column()]

        # Raw value (for numeric filtering, etc.)
        if role == Qt.UserRole:
            return val

        if role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        if val is None:
            return ""
        # Pretty formatting
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:  # noqa: ANN401, N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        # Sorting is handled by proxy; keep stable.
        return


def _is_number_str(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # float/exp numbers
    return bool(re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$", s))


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            # NaN -> treat as None
            if isinstance(v, float) and (v != v):
                return None
        except Exception:
            pass
        return float(v)
    s = str(v).strip()
    if not _is_number_str(s):
        return None
    try:
        return float(s)
    except Exception:
        return None


class _CompiledFilter:
    __slots__ = ("raw", "pred")

    def __init__(self, raw: str, pred) -> None:  # noqa: ANN001
        self.raw = raw
        self.pred = pred


def _compile_filter(expr: str):
    """Compile user filter expression into a predicate.

    Supported:
      - 10..20           (inclusive numeric range)
      - >=5, <=3.2, >1, <2
      - !=0, ==1, =1
      - list: 1,2,3 (membership)
      - fallback: substring match (case-insensitive)
    """
    s = (expr or "").strip()
    if not s:
        return None

    # Membership list: "a,b,c"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        nums = [float(p) for p in parts if _is_number_str(p)]
        strs = [p.lower() for p in parts if not _is_number_str(p)]

        def _pred(v: Any) -> bool:
            if v is None:
                return False
            fv = _to_float(v)
            if fv is not None and nums:
                return fv in nums
            sv = str(v).strip().lower()
            if strs:
                return sv in strs
            # Mixed list but value couldn't be typed -> string compare against all
            return sv in [p.lower() for p in parts]

        return _CompiledFilter(s, _pred)

    # Range: "a..b"
    if ".." in s:
        a, b = s.split("..", 1)
        a = a.strip()
        b = b.strip()
        fa = _to_float(a)
        fb = _to_float(b)
        if fa is not None and fb is not None:
            lo, hi = (fa, fb) if fa <= fb else (fb, fa)

            def _pred(v: Any) -> bool:
                fv = _to_float(v)
                if fv is None:
                    return False
                return lo <= fv <= hi

            return _CompiledFilter(s, _pred)

    # Comparisons
    ops = [">=", "<=", "!=", "==", "=", ">", "<"]
    for op in ops:
        if s.startswith(op):
            rhs = s[len(op) :].strip()
            if not rhs:
                break
            rhs_num = _to_float(rhs)
            rhs_str = rhs.lower()

            def _pred(v: Any, _op=op, _rhs_num=rhs_num, _rhs_str=rhs_str) -> bool:
                if v is None:
                    return False
                fv = _to_float(v)
                if _rhs_num is not None and fv is not None:
                    if _op in ("=", "=="):
                        return fv == _rhs_num
                    if _op == "!=":
                        return fv != _rhs_num
                    if _op == ">=":
                        return fv >= _rhs_num
                    if _op == "<=":
                        return fv <= _rhs_num
                    if _op == ">":
                        return fv > _rhs_num
                    if _op == "<":
                        return fv < _rhs_num
                # String compare (case-insensitive)
                sv = str(v).strip().lower()
                if _op in ("=", "=="):
                    return sv == _rhs_str
                if _op == "!=":
                    return sv != _rhs_str
                # For non-numeric comparisons on strings: not supported => false
                return False

            return _CompiledFilter(s, _pred)

    # Fallback: substring match
    needle = s.lower()

    def _pred(v: Any) -> bool:
        if v is None:
            return False
        return needle in str(v).lower()

    return _CompiledFilter(s, _pred)


class MultiFilterProxyModel(QSortFilterProxyModel):
    """Column-wise filters with simple expressions + optional image filter by basename."""
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.setDynamicSortFilter(True)
        self._column_filters: Dict[int, _CompiledFilter] = {}
        self._image_filter: Optional[str] = None

    def _invalidate(self) -> None:
        # Qt6: invalidateFilter() is deprecated; use invalidateRowsFilter() when available.
        if hasattr(self, "invalidateRowsFilter"):
            try:
                self.invalidateRowsFilter()  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        self.invalidateFilter()

    def set_column_filter(self, column: int, text: str) -> None:
        t = (text or "").strip()
        if not t:
            self._column_filters.pop(column, None)
            self._invalidate()
            return

        compiled = _compile_filter(t)
        if compiled is None:
            self._column_filters.pop(column, None)
        else:
            self._column_filters[column] = compiled
        self._invalidate()

    def clear_filters(self) -> None:
        self._column_filters.clear()
        self._invalidate()

    def set_image_filter(self, image_basename: Optional[str]) -> None:
        self._image_filter = image_basename
        self._invalidate()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        if model is None:
            return True

        # Image filter: expects a column named "image"
        if self._image_filter:
            # Find column index of "image" (cached via header lookup)
            image_col = None
            for c in range(model.columnCount()):
                h = model.headerData(c, Qt.Horizontal, Qt.DisplayRole)
                if str(h) == "image":
                    image_col = c
                    break
            if image_col is not None:
                idx = model.index(source_row, image_col, source_parent)
                v = str(model.data(idx, Qt.DisplayRole) or "")
                if v != self._image_filter:
                    return False

        # Column filters
        for col, compiled in self._column_filters.items():
            if col < 0 or col >= model.columnCount():
                continue
            idx = model.index(source_row, col, source_parent)
            raw = model.data(idx, Qt.UserRole)
            if raw is None:
                raw = model.data(idx, Qt.DisplayRole)
            try:
                if not compiled.pred(raw):
                    return False
            except Exception:
                return False

        return True


class PathPickerDialog(QDialog):
    """One dialog to select either a TIFF file or a directory."""

    def __init__(self, parent: Optional[QWidget] = None, start_dir: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Выберите TIFF-файл или папку")
        self.resize(900, 540)

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Выберите TIFF (*.tif/*.tiff) или директорию…")
        layout.addWidget(self.path_edit)

        self.view = QTreeView()
        self.view.setAlternatingRowColors(True)
        self.view.setUniformRowHeights(True)
        layout.addWidget(self.view, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

        self.model = QFileSystemModel(self)
        self.model.setNameFilters(["*.tif", "*.tiff", "*.TIF", "*.TIFF"])
        self.model.setNameFilterDisables(False)
        self.model.setRootPath("")
        self.view.setModel(self.model)

        # Show only the name column
        for c in range(1, self.model.columnCount()):
            self.view.hideColumn(c)

        self.view.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.view.selectionModel().selectionChanged.connect(self._on_sel_changed)
        self.view.doubleClicked.connect(self._on_double_clicked)

        start = start_dir if start_dir and start_dir.exists() else Path.home()
        idx = self.model.index(str(start))
        if idx.isValid():
            self.view.setRootIndex(idx)

        self._selected_path: Optional[Path] = None

    def selected_path(self) -> Optional[Path]:
        return self._selected_path

    def _on_sel_changed(self, selected, deselected) -> None:  # noqa: ANN001
        idxs = selected.indexes()
        if not idxs:
            self._selected_path = None
            self.path_edit.setText("")
            return
        idx = idxs[0]
        p = Path(self.model.filePath(idx))
        self._selected_path = p
        self.path_edit.setText(human_path(p))

    def _on_double_clicked(self, idx) -> None:  # noqa: ANN001
        p = Path(self.model.filePath(idx))
        if p.is_file() and p.suffix in SUPPORTED_EXTS:
            self._selected_path = p
            self.accept()

    def accept(self) -> None:
        p = self._selected_path
        if p is None:
            QMessageBox.warning(self, "Не выбрано", "Выберите TIFF-файл или папку.")
            return

        if p.is_dir():
            super().accept()
            return

        if p.is_file() and p.suffix in SUPPORTED_EXTS:
            super().accept()
            return

        QMessageBox.warning(self, "Неверный выбор", "Нужно выбрать папку или TIFF-файл (*.tif/*.tiff).")


class CrosshairImageView(QGraphicsView):
    """
    Displays an image with crosshair lines following the mouse + coordinate reporting.
    """
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.StyledPanel)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._vline = QGraphicsLineItem()
        self._hline = QGraphicsLineItem()

        # Make crosshair always visible above the image.
        # Keep a thin cosmetic pen but ensure it draws on top (z-value).
        c = QColor(self.palette().highlight().color())
        c.setAlpha(220)
        pen = QPen(c)
        pen.setWidth(0)  # cosmetic
        self._vline.setPen(pen)
        self._hline.setPen(pen)
        self._vline.setZValue(10)
        self._hline.setZValue(10)

        self._scene.addItem(self._vline)
        self._scene.addItem(self._hline)
        self._vline.hide()
        self._hline.hide()

        self._on_coords = None  # type: Optional[callable]

    def set_on_coords(self, fn) -> None:  # noqa: ANN001
        self._on_coords = fn

    def clear(self) -> None:
        """Clear the displayed image but keep crosshair items alive."""
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None

        self._scene.setSceneRect(QRectF())
        self._vline.hide()
        self._hline.hide()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._vline.show()
        self._hline.show()
        # Put crosshair to the center by default so it is visible immediately.
        rect = self._pixmap_item.boundingRect()
        self.set_crosshair(rect.center().x(), rect.center().y(), emit=False)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        super().mouseMoveEvent(event)
        if not self._pixmap_item:
            return

        pos = self.mapToScene(event.position().toPoint())
        rect = self._pixmap_item.boundingRect()

        self.set_crosshair(pos.x(), pos.y())

    def set_crosshair(self, x: float, y: float, emit: bool = True) -> None:
        """Set crosshair position in *image (scene) coordinates*."""
        if not self._pixmap_item:
            return

        rect = self._pixmap_item.boundingRect()

        # Clamp to image bounds
        xx = min(max(float(x), rect.left()), rect.right())
        yy = min(max(float(y), rect.top()), rect.bottom())

        self._vline.setLine(xx, rect.top(), xx, rect.bottom())
        self._hline.setLine(rect.left(), yy, rect.right(), yy)
        self._vline.show()
        self._hline.show()

        if emit and self._on_coords:
            self._on_coords(int(xx), int(yy))


@dataclass
class AnalysisConfig:
    input_paths: List[Path]
    output_csv: Path


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Astro Analyzer — GUI")
        self.resize(1400, 820)

        self._project_dir = Path(__file__).resolve().parent
        self._analyzer_script = self._project_dir / "astro_analyzer.py"

        self._process: Optional[QProcess] = None
        self._current_images: List[Path] = []
        self._df = None
        self._last_dir: Optional[Path] = None
        self._shown_image: Optional[str] = None
        self._shown_scale: Tuple[float, float] = (1.0, 1.0)

        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        # Status bar
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.status_label = QLabel("Готово")
        sb.addPermanentWidget(self.status_label)

        self.coords_label = QLabel("x: —  y: —")
        sb.addPermanentWidget(self.coords_label)

        # Top controls
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(8, 8, 8, 8)

        gb = QGroupBox("Вход / Выход")
        gbl = QHBoxLayout(gb)

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Выберите директорию или TIFF-файл…")
        self.input_path_edit.setReadOnly(True)

        self.btn_pick_path = QPushButton("Выбрать…")
        self.btn_pick_path.clicked.connect(self.choose_path)

        self.output_name_edit = QLineEdit("objects_stats.csv")
        self.output_name_edit.setPlaceholderText("Имя CSV-файла (например: objects.csv)")

        self.btn_run = QPushButton("Запустить анализ")
        self.btn_run.clicked.connect(self.start_analysis)

        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)

        gbl.addWidget(QLabel("Вход:"))
        gbl.addWidget(self.input_path_edit, 2)
        gbl.addWidget(self.btn_pick_path)
        gbl.addSpacing(12)
        gbl.addWidget(QLabel("CSV имя:"))
        gbl.addWidget(self.output_name_edit, 1)
        gbl.addWidget(self.btn_run)
        gbl.addWidget(self.btn_stop)

        top_layout.addWidget(gb)

        # Busy indicator
        busy_box = QGroupBox("Состояние")
        busy_layout = QHBoxLayout(busy_box)
        self.spinner = SpinnerWidget(size=18)
        self.spinner.stop()
        self.busy_text = QLabel("Ожидание")
        busy_layout.addWidget(self.spinner)
        busy_layout.addWidget(self.busy_text)
        busy_layout.addStretch(1)
        top_layout.addWidget(busy_box)

        # Main area: splitter (image list | table | image)
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(240)
        self.image_list.itemClicked.connect(self.on_image_selected)

        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.clicked.connect(self.on_table_row_clicked)

        self.image_view = CrosshairImageView()
        self.image_view.set_on_coords(self._update_coords)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.image_list)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.image_view)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setStretchFactor(2, 2)

        # Central layout
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(top)
        v.addWidget(self.splitter, 1)
        self.setCentralWidget(central)

        # Logs dock
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        log_dock = QDockWidget("Логи", self)
        log_dock.setWidget(self.log_edit)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        # Filters dock (без возможности закрыть — иначе не вернуть)
        self.filters_dock = QDockWidget("Фильтры", self)
        self.filters_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.filters_root = QWidget()
        self.filters_layout = QVBoxLayout(self.filters_root)
        self.filters_layout.setContentsMargins(8, 8, 8, 8)

        # Column filters area
        self.col_filters_area = QScrollArea()
        self.col_filters_area.setWidgetResizable(True)
        self.col_filters_widget = QWidget()
        self.col_filters_form = QFormLayout(self.col_filters_widget)
        self.col_filters_form.setLabelAlignment(Qt.AlignRight)
        self.col_filters_area.setWidget(self.col_filters_widget)
        self.filters_layout.addWidget(self.col_filters_area, 1)

        # Clear button
        self.btn_clear_filters = QPushButton("Очистить фильтры")
        self.btn_clear_filters.clicked.connect(self.clear_filters)
        self.filters_layout.addWidget(self.btn_clear_filters)

        self.filters_dock.setWidget(self.filters_root)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filters_dock)

        # Proxy model
        self.proxy = MultiFilterProxyModel(self)
        self.table.setModel(self.proxy)

        # Initial list state
        self._set_images([])

    # ---------------- Helpers ----------------

    def _update_coords(self, x: int, y: int) -> None:
        self.coords_label.setText(f"x: {x}  y: {y}")

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_edit.appendPlainText(text.rstrip("\n"))

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy_text.setText(text)
        self.status_label.setText(text)
        if busy:
            self.spinner.start()
        else:
            self.spinner.stop()

        # Disable/enable controls
        for w in (self.btn_pick_path, self.output_name_edit, self.btn_run):
            w.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)

    def _set_images(self, images: List[Path]) -> None:
        self._current_images = images
        self.image_list.clear()

        item_all = QListWidgetItem("Все изображения")
        item_all.setData(Qt.UserRole, None)
        self.image_list.addItem(item_all)

        for p in images:
            it = QListWidgetItem(p.name)
            it.setToolTip(human_path(p))
            it.setData(Qt.UserRole, str(p))
            self.image_list.addItem(it)

        self.image_list.setCurrentRow(0)

    def _resolve_output_path(self) -> Optional[Path]:
        name = (self.output_name_edit.text() or "").strip()
        if not name:
            return None
        if not name.lower().endswith(".csv"):
            name += ".csv"

        # Save near chosen input (folder or file)
        in_text = (self.input_path_edit.text() or "").strip()
        if not in_text:
            # fallback to project dir
            out_dir = self._project_dir
        else:
            p = Path(in_text)
            out_dir = p if p.is_dir() else p.parent

        return out_dir / name

    def _build_column_filters(self) -> None:
        # Clear form
        while self.col_filters_form.rowCount():
            self.col_filters_form.removeRow(0)

        src = self.proxy.sourceModel()
        if src is None:
            return

        for c in range(src.columnCount()):
            col_name = str(src.headerData(c, Qt.Horizontal, Qt.DisplayRole))
            edit = QLineEdit()
            edit.textChanged.connect(lambda txt, col=c: self.proxy.set_column_filter(col, txt))
            self.col_filters_form.addRow(col_name + ":", edit)

    # ---------------- Input selection ----------------

    def choose_path(self) -> None:
        start = self._last_dir or Path.home()
        dlg = PathPickerDialog(self, start)
        if dlg.exec() != QDialog.Accepted:
            return
        p = dlg.selected_path()
        if p is None:
            return

        self._last_dir = p if p.is_dir() else p.parent

        if p.is_dir():
            images = list_tiff_images(p)
            if not images:
                QMessageBox.warning(self, "Нет TIFF", "В выбранной директории не найдено TIFF (*.tif/*.tiff).")
            self.input_path_edit.setText(human_path(p))
            self._set_images(images)
        else:
            self.input_path_edit.setText(human_path(p))
            self._set_images([p])

        self.image_view.clear()
        self.proxy.set_image_filter(None)

    # ---------------- Analysis ----------------

    def start_analysis(self) -> None:
        if not self._analyzer_script.exists():
            QMessageBox.critical(
                self,
                "Не найдено ядро",
                f"Не найден файл: {human_path(self._analyzer_script)}\n"
                "Убедитесь, что astro_analyzer.py лежит рядом с astro_gui.py.",
            )
            return

        if not self._current_images:
            QMessageBox.warning(self, "Нет входных данных", "Сначала выберите папку или TIFF-файл.")
            return

        out_path = self._resolve_output_path()
        if out_path is None:
            QMessageBox.warning(self, "Нет имени CSV", "Введите имя CSV-файла.")
            return

        # Clear previous results
        self._df = None
        self.proxy.setSourceModel(None)
        self.proxy.clear_filters()
        self.proxy.set_image_filter(None)
        self.image_view.clear()
        self._shown_image = None
        self._shown_scale = (1.0, 1.0)
        self._append_log("\n=== Запуск анализа ===")

        # Prepare process
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        # Suppress Python warnings from the analyzer subprocess (keeps logs clean).
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONWARNINGS", "ignore")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(lambda: self._on_proc_output(proc))
        proc.finished.connect(lambda code, status: self._on_proc_finished(code, status, out_path))

        # Build args: python astro_analyzer.py <images...> --output <csv>
        args = [str(self._analyzer_script)]
        args.extend([str(p) for p in self._current_images])
        args.extend(["--output", str(out_path)])

        # Start
        self._process = proc
        self._set_busy(True, "Анализ выполняется…")
        proc.start(sys.executable, args)

        if not proc.waitForStarted(3000):
            self._set_busy(False, "Ошибка запуска")
            self._append_log("Не удалось запустить процесс анализа.")
            QMessageBox.critical(self, "Ошибка запуска", "Не удалось запустить процесс анализа.")
            self._process = None

    def stop_analysis(self) -> None:
        if self._process is None:
            return
        self._append_log("\n=== Остановка анализа пользователем ===")
        self._process.kill()
        self._process = None
        self._set_busy(False, "Остановлено")

    def _on_proc_output(self, proc: QProcess) -> None:
        data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if not data:
            return

        out_lines: List[str] = []
        for line in data.splitlines():
            if _LOG_DROP_RE.search(line):
                continue
            out_lines.append(line)

        if out_lines:
            self._append_log("\n".join(out_lines) + "\n")

    def _on_proc_finished(self, code: int, status: QProcess.ExitStatus, out_path: Path) -> None:
        self._set_busy(False, "Готово")
        self._append_log(f"\n=== Завершено (код={code}, статус={status}) ===")
        self._process = None

        if code != 0:
            QMessageBox.critical(
                self,
                "Ошибка анализа",
                "Процесс анализа завершился с ошибкой. Смотрите вкладку логов.",
            )
            return

        if not out_path.exists():
            QMessageBox.critical(
                self,
                "Нет результата",
                f"CSV файл не найден: {human_path(out_path)}",
            )
            return

        # Load CSV into table
        try:
            import pandas as pd  # local import
            df = pd.read_csv(out_path)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения CSV", f"Не удалось прочитать CSV:\n{e}")
            return

        self._df = df
        model = PandasTableModel(df, self)
        self.proxy.setSourceModel(model)
        self.table.resizeColumnsToContents()
        self._build_column_filters()

        QMessageBox.information(
            self,
            "Анализ завершён",
            f"Результат сохранён в:\n{human_path(out_path)}\n"
            f"Найдено объектов: {len(df)}",
        )

    # ---------------- Table filtering / images ----------------

    def clear_filters(self) -> None:
        self.proxy.clear_filters()
        # Reset column filter widgets
        for i in range(self.col_filters_form.rowCount()):
            field = self.col_filters_form.itemAt(i, QFormLayout.FieldRole)
            if field and field.widget():
                w = field.widget()
                if isinstance(w, QLineEdit):
                    w.blockSignals(True)
                    w.setText("")
                    w.blockSignals(False)

    def on_image_selected(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data is None:
            # All images
            self.proxy.set_image_filter(None)
            self.image_view.clear()
            self._shown_image = None
            self._shown_scale = (1.0, 1.0)
            return

        path = Path(str(data))
        # Show image
        try:
            pix, scale = pil_to_qpixmap(path, return_meta=True)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка изображения", f"Не удалось загрузить изображение:\n{e}")
            return
        self.image_view.set_pixmap(pix)
        self._shown_image = path.name
        self._shown_scale = scale

        # Filter table by this image (by basename, matches core output column "image")
        self.proxy.set_image_filter(path.name)

    def _source_col_index(self, name: str) -> Optional[int]:
        m = self.proxy.sourceModel()
        if m is None:
            return None
        for c in range(m.columnCount()):
            h = m.headerData(c, Qt.Horizontal, Qt.DisplayRole)
            if str(h) == name:
                return c
        return None

    def _show_image_for_basename(self, basename: str) -> None:
        # Find image path among loaded inputs
        p: Optional[Path] = None
        for cand in self._current_images:
            if cand.name == basename:
                p = cand
                break
        if p is None:
            return
        try:
            pix, scale = pil_to_qpixmap(p, return_meta=True)
        except Exception:
            return
        self.image_view.set_pixmap(pix)
        self._shown_image = p.name
        self._shown_scale = scale

    def on_table_row_clicked(self, index: QModelIndex) -> None:
        """When a row is clicked, move crosshair to (x,y) from the table."""
        if not index.isValid():
            return
        src_model = self.proxy.sourceModel()
        if src_model is None:
            return
        src_index = self.proxy.mapToSource(index)
        row = src_index.row()

        col_x = self._source_col_index("x")
        col_y = self._source_col_index("y")
        col_img = self._source_col_index("image")
        if col_x is None or col_y is None:
            return

        # Extract values (prefer raw via UserRole)
        idx_x = src_model.index(row, col_x)
        idx_y = src_model.index(row, col_y)
        vx = src_model.data(idx_x, Qt.UserRole)
        vy = src_model.data(idx_y, Qt.UserRole)
        if vx is None:
            vx = src_model.data(idx_x, Qt.DisplayRole)
        if vy is None:
            vy = src_model.data(idx_y, Qt.DisplayRole)

        fx = _to_float(vx)
        fy = _to_float(vy)
        if fx is None or fy is None:
            return

        img_name = ""
        if col_img is not None:
            idx_img = src_model.index(row, col_img)
            img_name = str(src_model.data(idx_img, Qt.DisplayRole) or "")

        # If the clicked row belongs to another image, show it (do NOT change filters/list selection).
        if img_name and img_name != (self._shown_image or ""):
            self._show_image_for_basename(img_name)

        # Convert analyzer coords (original pixels) to displayed pixmap coords (if image was downscaled).
        sx, sy = self._shown_scale
        self.image_view.set_crosshair(fx * sx, fy * sy)


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()