# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>


# Create a QApplication, must be done before creating any GUI elements
import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox, QLineEdit
app = QApplication.instance() or QApplication(sys.argv)

from glob import glob
from typing import Any, List, Tuple

def selection_window(title: str, values: List[Any]) -> Tuple[List[Any], List[int]]:
    dialog = QDialog()
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(500, 400)
    layout = QVBoxLayout(dialog)

    search_bar = QLineEdit()
    search_bar.setPlaceholderText("Search...")
    layout.addWidget(search_bar)

    list_widget = QListWidget()
    list_widget.setSelectionMode(QListWidget.ExtendedSelection)
    layout.addWidget(list_widget)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    # Function to populate list based on current filter
    def populate_list(filter_text: str = ""):
        list_widget.clear()
        for index, value in enumerate(values):
            if filter_text.lower() in str(value).lower():
                item = QListWidgetItem(str(value))
                item.setData(Qt.UserRole, index)
                list_widget.addItem(item)

    populate_list()

    search_bar.textChanged.connect(lambda text: populate_list(text))

    selected_indexes = []
    if dialog.exec_() == QDialog.Accepted:
        for item in list_widget.selectedItems():
            selected_indexes.append(item.data(Qt.UserRole))

    selected_indexes.sort()
    selected_values = [values[i] for i in selected_indexes]

    return selected_values, selected_indexes

def select_runs_folders(base_dir: str, query_file: str | List[str]) -> Tuple[List[str], List[str]]:
    if isinstance(query_file, str):
        query_file = [query_file]
    
    base_dir = os.path.expandvars(base_dir)

    if not os.path.isdir(base_dir):
        raise ValueError(f"Base directory doesn't exist: {base_dir}")

    all_files = {
        qf: glob(os.path.join(base_dir, "**", qf), recursive=True) for qf in query_file
    }
    all_folders = {
        qf: set([os.path.dirname(file) for file in files]) for qf, files in all_files.items()
    }
    folders_with_all = sorted(set.intersection(*all_folders.values()))

    if not folders_with_all:
        raise ValueError(f"No folder containing '{query_file}' found in: {base_dir}")

    runs_names = [os.path.relpath(path, base_dir) for path in folders_with_all]

    selected_names, indexes = selection_window(title="Select Runs Folders", values=runs_names)
    selected_paths = [folders_with_all[i] for i in indexes]

    return selected_paths, selected_names

def select_files(base_dir: str, query_file: str, keep_full_names: bool = True, title: str = "Select Files") -> Tuple[List[str], List[str]]:
    config_filenames = sorted(glob(os.path.expandvars(os.path.join(base_dir, "**", query_file)), recursive=True))
    if not keep_full_names:
        config_names = [os.path.basename(file) for file in config_filenames]
    else:
        config_names = [os.path.relpath(file, os.path.expandvars(base_dir)) for file in config_filenames]

    selected_names, indexes = selection_window(title=title, values=config_names)
    selected_paths = [config_filenames[i] for i in indexes]

    return selected_paths, selected_names
    