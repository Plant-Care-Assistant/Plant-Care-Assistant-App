"""Class selection unit tests module.

Copyright 2025 Plant Care Assistant
"""

from pathlib import Path

from plant_care_ai.training.class_selection import get_most_popular_classes

# Test constants
TOP_K_SMALL = 2
TOP_K_DEFAULT = 10
TOP_K_LARGE = 1000
MIN_CLASSES_FOR_SORT_TEST = 2


class TestGetMostPopularClasses:
    """Tests for the get_most_popular_classes function."""

    @staticmethod
    def test_returns_top_classes(sample_data_dir: Path) -> None:
        """Test that function returns the most popular classes.

        Args:
            sample_data_dir: Path to sample data directory

        """
        top_classes, class_counts = get_most_popular_classes(
            data_dir=sample_data_dir,
            top_k=TOP_K_SMALL,
            split="train",
        )

        assert isinstance(top_classes, list)
        assert isinstance(class_counts, dict)
        assert len(top_classes) <= TOP_K_SMALL

    @staticmethod
    def test_returns_all_classes_when_top_k_exceeds(sample_data_dir: Path) -> None:
        """Test that function returns all classes when top_k exceeds total.

        Args:
            sample_data_dir: Path to sample data directory

        """
        top_classes, class_counts = get_most_popular_classes(
            data_dir=sample_data_dir,
            top_k=TOP_K_LARGE,
            split="train",
        )

        # Should return all available classes (not more than top_k)
        assert len(top_classes) <= TOP_K_LARGE
        assert len(top_classes) == len(class_counts)

    @staticmethod
    def test_class_counts_dict_values_are_positive(sample_data_dir: Path) -> None:
        """Test that class counts are positive integers.

        Args:
            sample_data_dir: Path to sample data directory

        """
        _, class_counts = get_most_popular_classes(
            data_dir=sample_data_dir,
            top_k=TOP_K_DEFAULT,
            split="train",
        )

        for count in class_counts.values():
            assert isinstance(count, int)
            assert count > 0

    @staticmethod
    def test_top_classes_sorted_by_popularity(sample_data_dir: Path) -> None:
        """Test that top classes are sorted by popularity (descending).

        Args:
            sample_data_dir: Path to sample data directory

        """
        top_classes, class_counts = get_most_popular_classes(
            data_dir=sample_data_dir,
            top_k=TOP_K_LARGE,
            split="train",
        )

        if len(top_classes) >= MIN_CLASSES_FOR_SORT_TEST:
            counts = [class_counts[c] for c in top_classes]
            # Verify sorted in descending order
            assert counts == sorted(counts, reverse=True)

    @staticmethod
    def test_works_with_different_splits(sample_data_dir: Path) -> None:
        """Test that function works with val and test splits.

        Args:
            sample_data_dir: Path to sample data directory

        """
        val_classes, val_counts = get_most_popular_classes(
            data_dir=sample_data_dir,
            top_k=TOP_K_DEFAULT,
            split="val",
        )
        test_classes, test_counts = get_most_popular_classes(
            data_dir=sample_data_dir,
            top_k=TOP_K_DEFAULT,
            split="test",
        )

        assert isinstance(val_classes, list)
        assert isinstance(test_classes, list)
        assert isinstance(val_counts, dict)
        assert isinstance(test_counts, dict)
