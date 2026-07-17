import json
from pathlib import Path

import pytest

from dtxt.backends import MockBackend
from dtxt.entities import (
    Entity,
    EntityExtractionError,
    EntityNormalizationError,
    EntityTypeNormalizer,
    FlatEntityExtractor,
)


def test_extract_returns_flat_entity_list() -> None:
    raw = json.dumps(
        [
            {"type": "person_name", "value": "Alice"},
            {"type": "person_name", "value": "Bob"},
            {"type": "date", "value": "2026-07-17"},
        ]
    )
    backend = MockBackend(responses=[raw])
    extractor = FlatEntityExtractor(backend)

    entities = extractor.extract("Alice met Bob on 2026-07-17.")

    assert entities == [
        Entity(type="person_name", value="Alice"),
        Entity(type="person_name", value="Bob"),
        Entity(type="date", value="2026-07-17"),
    ]


def test_extract_skips_malformed_items() -> None:
    raw = json.dumps([{"type": "email"}, {"value": "no type"}, {"type": "city", "value": "Kyoto"}])
    backend = MockBackend(responses=[raw])
    extractor = FlatEntityExtractor(backend)

    entities = extractor.extract("Kyoto")

    assert entities == [Entity(type="city", value="Kyoto")]


def test_extract_raises_on_invalid_json() -> None:
    backend = MockBackend(responses=["not json"])
    extractor = FlatEntityExtractor(backend)

    with pytest.raises(EntityExtractionError):
        extractor.extract("some text")


def test_extract_raises_when_response_is_not_a_list() -> None:
    backend = MockBackend(responses=[json.dumps({"type": "city", "value": "Kyoto"})])
    extractor = FlatEntityExtractor(backend)

    with pytest.raises(EntityExtractionError):
        extractor.extract("some text")


def test_fit_merges_synonymous_types_via_backend() -> None:
    backend = MockBackend(
        responses=[json.dumps({"name": "person_name", "full_name": "person_name"})]
    )
    normalizer = EntityTypeNormalizer(backend)
    entity_lists = [
        [Entity(type="name", value="Alice")],
        [Entity(type="full_name", value="Bob Smith")],
    ]

    normalizer.fit(entity_lists)

    assert normalizer.mapping == {"name": "person_name", "full_name": "person_name"}


def test_fit_passes_example_values_for_each_normalized_type() -> None:
    backend = MockBackend(responses=[json.dumps({})])
    normalizer = EntityTypeNormalizer(backend)

    normalizer.fit([[Entity(type="Name", value="Alice"), Entity(type="name", value="Bob")]])

    prompt, _ = backend.calls[0]
    assert "name" in prompt
    assert "Alice" in prompt
    assert "Bob" in prompt


def test_fit_on_empty_input_produces_empty_mapping_without_calling_backend() -> None:
    backend = MockBackend()
    normalizer = EntityTypeNormalizer(backend)

    normalizer.fit([])

    assert normalizer.mapping == {}
    assert backend.calls == []


def test_fit_raises_on_invalid_json() -> None:
    backend = MockBackend(responses=["not json"])
    normalizer = EntityTypeNormalizer(backend)

    with pytest.raises(EntityNormalizationError):
        normalizer.fit([[Entity(type="name", value="Alice")]])


def test_transform_applies_fitted_mapping() -> None:
    normalizer = EntityTypeNormalizer(MockBackend())
    normalizer.mapping = {"name": "person_name", "full_name": "person_name"}

    result = normalizer.transform(
        [
            [Entity(type="Name", value="Alice"), Entity(type="date", value="2026-07-17")],
            [Entity(type="full_name", value="Bob")],
        ]
    )

    assert result == [
        [Entity(type="person_name", value="Alice"), Entity(type="date", value="2026-07-17")],
        [Entity(type="person_name", value="Bob")],
    ]


def test_transform_falls_back_to_rule_normalized_type_for_unseen_types() -> None:
    normalizer = EntityTypeNormalizer(MockBackend())
    normalizer.mapping = {}

    result = normalizer.transform([[Entity(type="Email Address", value="a@example.com")]])

    assert result == [[Entity(type="email_address", value="a@example.com")]]


def test_transform_does_not_call_backend() -> None:
    backend = MockBackend()
    normalizer = EntityTypeNormalizer(backend)
    normalizer.mapping = {"name": "person_name"}

    normalizer.transform([[Entity(type="name", value="Alice")]])

    assert backend.calls == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    normalizer = EntityTypeNormalizer(MockBackend())
    normalizer.mapping = {"name": "person_name", "full_name": "person_name"}
    normalizer.save(path)

    loaded = EntityTypeNormalizer.load(path, MockBackend())

    assert loaded.mapping == normalizer.mapping
