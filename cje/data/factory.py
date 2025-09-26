"""Dataset factory for loading datasets.

This module follows SOLID principles by using dependency injection
and separating concerns into focused classes.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .models import Dataset
from .loaders import DatasetLoader, DataSource, JsonlDataSource, InMemoryDataSource


class DatasetFactory:
    """Factory for creating Datasets from various sources.

    Follows SOLID principles:
    - Single Responsibility: Coordinates data loading
    - Open/Closed: Easy to extend with new loaders
    - Dependency Injection: Takes loader as dependency
    """

    def __init__(
        self,
        loader: Optional[DatasetLoader] = None,
    ):
        """Initialize factory with optional custom loader.

        Args:
            loader: DatasetLoader instance. If None, uses default.
        """
        self.loader = loader or DatasetLoader()

    def create_from_jsonl(
        self, file_path: str, target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Create Dataset from JSONL file.

        Args:
            file_path: Path to JSONL file
            target_policies: Optional list of target policy names

        Returns:
            Dataset instance
        """
        source = JsonlDataSource(file_path)
        return self.loader.load_from_source(source, target_policies)

    def create_from_data(
        self, data: List[Dict[str, Any]], target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Create Dataset from in-memory data.

        Args:
            data: List of dictionaries with data
            target_policies: Optional list of target policy names

        Returns:
            Dataset instance
        """
        source = InMemoryDataSource(data)
        return self.loader.load_from_source(source, target_policies)


# Convenience factory instance with default configuration
default_factory = DatasetFactory()
