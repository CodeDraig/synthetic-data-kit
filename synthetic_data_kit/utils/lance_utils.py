# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import lance
import pyarrow as pa
from typing import List, Dict, Any, Optional
import os

def create_lance_dataset(
    data: List[Dict[str, Any]],
    output_path: str,
    schema: Optional[pa.Schema] = None
) -> None:
    """Create a Lance dataset from a list of dictionaries.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a row.
        output_path (str): The path to save the Lance dataset.
        schema (Optional[pa.Schema], optional): The PyArrow schema. If not provided, it will be inferred. Defaults to None.
    """
    if not data:
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    table = pa.Table.from_pylist(data, schema=schema)
    lance.write_dataset(table, output_path, mode="overwrite")

def _resolve_lance_root(path: str) -> Optional[str]:
    """Resolve the root directory of a Lance dataset.

    If the provided path is inside a Lance dataset (e.g., .../foo.lance/data/<uuid>.lance),
    this walks up the directory tree to find the nearest ancestor directory that ends with
    ".lance" and returns that as the dataset root. If the path is already the root, it is
    returned unchanged. Returns None if no suitable root is found.
    """
    p = os.path.abspath(path)
    # Always walk up to find a .lance ancestor, but if starting inside
    # .../<root>.lance/data/<uuid>.lance ensure we pick <root>.lance
    # and not the nested <uuid>.lance.
    cur = p
    candidate = None
    while True:
        if os.path.isdir(cur) and cur.endswith(".lance"):
            parent_dir = os.path.basename(os.path.dirname(cur))
            # If parent is 'data', this is a nested version dir; keep walking up
            if parent_dir.lower() != "data":
                candidate = cur
                break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return candidate

def load_lance_dataset(
    dataset_path: str
):
    """Load a Lance dataset.

    Args:
        dataset_path (str): The path to the Lance dataset.

    Returns:
        The loaded Lance dataset, or None if the dataset does not exist.
    """
    # Resolve to the dataset root if a nested internal path was provided
    resolved = _resolve_lance_root(dataset_path) or dataset_path
    if not os.path.exists(resolved) or not (os.path.isdir(resolved) and resolved.endswith(".lance")):
        return None
    return lance.dataset(resolved)
