import json
from pathlib import Path

import pytest

from dtxt.backends import MockBackend
from dtxt.entities import (
    Entity,
    EntityExtractionError,
    EntityNormalizationError,
    EntityRenderer,
    EntityRenderError,
    EntityTypeNormalizer,
    FlatEntityExtractor,
    NestedEntityExtractor,
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


def test_extract_without_entity_schema_calls_backend_with_no_schema() -> None:
    raw = json.dumps([{"type": "city", "value": "Kyoto"}])
    backend = MockBackend(responses=[raw])
    extractor = FlatEntityExtractor(backend)

    extractor.extract("Kyoto")

    _, schema = backend.calls[0]
    assert schema is None


def test_extract_with_entity_schema_includes_allowed_types_in_prompt() -> None:
    raw = json.dumps([{"type": "city", "value": "Kyoto"}])
    backend = MockBackend(responses=[raw])
    extractor = FlatEntityExtractor(
        backend, entity_schema={"type": "string", "enum": ["city", "date"]}
    )

    extractor.extract("Kyoto")

    prompt, _ = backend.calls[0]
    assert "city" in prompt
    assert "date" in prompt


def test_extract_with_entity_schema_passes_wrapped_output_schema_to_backend() -> None:
    raw = json.dumps([{"type": "city", "value": "Kyoto"}])
    backend = MockBackend(responses=[raw])
    entity_schema = {"type": "string", "enum": ["city", "date"]}
    extractor = FlatEntityExtractor(backend, entity_schema=entity_schema)

    extractor.extract("Kyoto")

    _, schema = backend.calls[0]
    assert schema == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"type": entity_schema, "value": {"type": "string"}},
            "required": ["type", "value"],
        },
    }


def test_extract_with_entity_schema_drops_out_of_vocabulary_types() -> None:
    raw = json.dumps(
        [
            {"type": "city", "value": "Kyoto"},
            {"type": "country", "value": "Japan"},
        ]
    )
    backend = MockBackend(responses=[raw])
    extractor = FlatEntityExtractor(
        backend, entity_schema={"type": "string", "enum": ["city", "date"]}
    )

    entities = extractor.extract("Kyoto, Japan")

    assert entities == [Entity(type="city", value="Kyoto")]


def test_nested_extract_returns_flat_entities_when_no_groups() -> None:
    raw = json.dumps([{"type": "city", "value": "Kyoto"}, {"type": "date", "value": "2026-07-17"}])
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    entities = extractor.extract("Kyoto, 2026-07-17")

    assert entities == [
        Entity(type="city", value="Kyoto"),
        Entity(type="date", value="2026-07-17"),
    ]


def test_nested_extract_returns_group_entity_with_children() -> None:
    raw = json.dumps(
        [
            {"type": "order_id", "value": "42"},
            {
                "type": "line_item",
                "children": [
                    {"type": "item_name", "value": "Widget"},
                    {"type": "quantity", "value": "3"},
                ],
            },
        ]
    )
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    entities = extractor.extract("Order #42: 3x Widget.")

    assert entities == [
        Entity(type="order_id", value="42"),
        Entity(
            type="line_item",
            children=[
                Entity(type="item_name", value="Widget"),
                Entity(type="quantity", value="3"),
            ],
        ),
    ]


def test_nested_extract_repeated_group_type_signals_array() -> None:
    raw = json.dumps(
        [
            {"type": "line_item", "children": [{"type": "item_name", "value": "Widget"}]},
            {"type": "line_item", "children": [{"type": "item_name", "value": "Gadget"}]},
        ]
    )
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    entities = extractor.extract("3x Widget, 1x Gadget.")

    assert [entity.type for entity in entities] == ["line_item", "line_item"]


def test_nested_extract_skips_malformed_children() -> None:
    raw = json.dumps(
        [
            {
                "type": "line_item",
                "children": [
                    {"type": "item_name", "value": "Widget"},
                    {"type": "quantity"},
                    {"value": "no type"},
                ],
            }
        ]
    )
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    entities = extractor.extract("3x Widget.")

    assert entities == [
        Entity(type="line_item", children=[Entity(type="item_name", value="Widget")])
    ]


def test_nested_extract_drops_nested_children_beyond_one_level() -> None:
    raw = json.dumps(
        [
            {
                "type": "line_item",
                "children": [
                    {
                        "type": "sub_item",
                        "value": "Widget",
                        "children": [{"type": "nested", "value": "too deep"}],
                    }
                ],
            }
        ]
    )
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    entities = extractor.extract("3x Widget.")

    assert entities == [
        Entity(type="line_item", children=[Entity(type="sub_item", value="Widget")])
    ]


def test_nested_extract_skips_items_without_type() -> None:
    raw = json.dumps([{"value": "no type"}, {"type": "city", "value": "Kyoto"}])
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    entities = extractor.extract("Kyoto")

    assert entities == [Entity(type="city", value="Kyoto")]


def test_nested_extract_skips_items_with_neither_value_nor_children() -> None:
    raw = json.dumps([{"type": "mystery"}, {"type": "city", "value": "Kyoto"}])
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    entities = extractor.extract("Kyoto")

    assert entities == [Entity(type="city", value="Kyoto")]


def test_nested_extract_raises_on_invalid_json() -> None:
    backend = MockBackend(responses=["not json"])
    extractor = NestedEntityExtractor(backend)

    with pytest.raises(EntityExtractionError):
        extractor.extract("some text")


def test_nested_extract_raises_when_response_is_not_a_list() -> None:
    backend = MockBackend(responses=[json.dumps({"type": "city", "value": "Kyoto"})])
    extractor = NestedEntityExtractor(backend)

    with pytest.raises(EntityExtractionError):
        extractor.extract("some text")


def test_nested_extract_passes_non_recursive_output_schema_to_backend() -> None:
    raw = json.dumps([{"type": "city", "value": "Kyoto"}])
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend)

    extractor.extract("Kyoto")

    _, schema = backend.calls[0]
    assert schema is not None
    child_schema = schema["items"]["properties"]["children"]["items"]
    assert "children" not in child_schema["properties"]


def test_nested_extract_supports_deeper_nesting_when_max_depth_is_increased() -> None:
    raw = json.dumps(
        [
            {
                "type": "line_item",
                "children": [
                    {
                        "type": "sub_item",
                        "children": [{"type": "nested", "value": "still here"}],
                    }
                ],
            }
        ]
    )
    backend = MockBackend(responses=[raw])
    extractor = NestedEntityExtractor(backend, max_depth=2)

    entities = extractor.extract("deeply nested text")

    assert entities == [
        Entity(
            type="line_item",
            children=[
                Entity(type="sub_item", children=[Entity(type="nested", value="still here")])
            ],
        )
    ]


def test_nested_extract_output_schema_reflects_max_depth() -> None:
    backend = MockBackend(responses=[json.dumps([{"type": "city", "value": "Kyoto"}])])
    extractor = NestedEntityExtractor(backend, max_depth=2)

    extractor.extract("Kyoto")

    _, schema = backend.calls[0]
    assert schema is not None
    level1 = schema["items"]["properties"]["children"]["items"]
    level2 = level1["properties"]["children"]["items"]
    assert "children" not in level2["properties"]


def test_nested_extract_render_re_extract_round_trip() -> None:
    extracted_raw = json.dumps(
        [
            {"type": "order_id", "value": "42"},
            {
                "type": "line_item",
                "children": [
                    {"type": "item_name", "value": "Widget"},
                    {"type": "quantity", "value": "3"},
                ],
            },
        ]
    )
    extractor = NestedEntityExtractor(MockBackend(responses=[extracted_raw]))
    entities = extractor.extract("Order #42: 3x Widget.")

    renderer = EntityRenderer(MockBackend(responses=["Order #42: 3x Widget."]))
    rendered_text = renderer.render(entities)

    re_extractor = NestedEntityExtractor(MockBackend(responses=[extracted_raw]))
    re_entities = re_extractor.extract(rendered_text)

    assert re_entities == entities


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


def test_entity_schema_is_none_before_fit() -> None:
    normalizer = EntityTypeNormalizer(MockBackend())

    assert normalizer.entity_schema() is None


def test_entity_schema_is_none_after_fit_on_empty_input() -> None:
    normalizer = EntityTypeNormalizer(MockBackend())

    normalizer.fit([])

    assert normalizer.entity_schema() is None


def test_entity_schema_returns_sorted_canonical_types_after_fit() -> None:
    backend = MockBackend(
        responses=[json.dumps({"name": "person_name", "full_name": "person_name", "date": "date"})]
    )
    normalizer = EntityTypeNormalizer(backend)

    normalizer.fit(
        [
            [Entity(type="name", value="Alice"), Entity(type="date", value="2026-07-17")],
            [Entity(type="full_name", value="Bob Smith")],
        ]
    )

    assert normalizer.entity_schema() == {"type": "string", "enum": ["date", "person_name"]}


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


def test_render_returns_backend_text() -> None:
    backend = MockBackend(responses=["Alice met Bob on 2026-07-17."])
    renderer = EntityRenderer(backend)

    text = renderer.render(
        [
            Entity(type="person_name", value="Alice"),
            Entity(type="person_name", value="Bob"),
            Entity(type="date", value="2026-07-17"),
        ]
    )

    assert text == "Alice met Bob on 2026-07-17."


def test_render_passes_every_entity_in_prompt() -> None:
    backend = MockBackend(responses=["some text"])
    renderer = EntityRenderer(backend)

    renderer.render([Entity(type="city", value="Kyoto"), Entity(type="date", value="2026-07-17")])

    prompt, _ = backend.calls[0]
    assert "city" in prompt
    assert "Kyoto" in prompt
    assert "date" in prompt
    assert "2026-07-17" in prompt


def test_render_passes_group_entity_children_in_prompt() -> None:
    backend = MockBackend(responses=["some text"])
    renderer = EntityRenderer(backend)

    renderer.render(
        [
            Entity(type="order_id", value="42"),
            Entity(
                type="line_item",
                children=[
                    Entity(type="item_name", value="Widget"),
                    Entity(type="quantity", value="3"),
                ],
            ),
        ]
    )

    prompt, _ = backend.calls[0]
    assert "order_id" in prompt
    assert "42" in prompt
    assert "line_item" in prompt
    assert "item_name" in prompt
    assert "Widget" in prompt
    assert "quantity" in prompt
    assert "3" in prompt


def test_render_raises_on_empty_backend_output() -> None:
    backend = MockBackend(responses=["   "])
    renderer = EntityRenderer(backend)

    with pytest.raises(EntityRenderError):
        renderer.render([Entity(type="city", value="Kyoto")])


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    normalizer = EntityTypeNormalizer(MockBackend())
    normalizer.mapping = {"name": "person_name", "full_name": "person_name"}
    normalizer.save(path)

    loaded = EntityTypeNormalizer.load(path, MockBackend())

    assert loaded.mapping == normalizer.mapping


def test_fit_recursively_normalizes_group_children() -> None:
    backend = MockBackend(
        responses=[
            json.dumps({"line_item": "line_item"}),
            json.dumps({"name": "item_name", "item_name": "item_name"}),
        ]
    )
    normalizer = EntityTypeNormalizer(backend)
    entity_lists = [
        [Entity(type="line_item", children=[Entity(type="name", value="Widget")])],
        [Entity(type="line_item", children=[Entity(type="item_name", value="Gadget")])],
    ]

    normalizer.fit(entity_lists)

    assert normalizer.mapping == {"line_item": "line_item"}
    assert normalizer.children["line_item"].mapping == {
        "name": "item_name",
        "item_name": "item_name",
    }


def test_fit_does_not_recurse_into_leaf_only_entities() -> None:
    backend = MockBackend(responses=[json.dumps({"name": "person_name"})])
    normalizer = EntityTypeNormalizer(backend)

    normalizer.fit([[Entity(type="name", value="Alice")]])

    assert normalizer.children == {}


def test_transform_preserves_group_children_recursively() -> None:
    normalizer = EntityTypeNormalizer(MockBackend())
    normalizer.mapping = {"line_item": "line_item"}
    child_normalizer = EntityTypeNormalizer(MockBackend())
    child_normalizer.mapping = {"name": "item_name"}
    normalizer.children = {"line_item": child_normalizer}

    result = normalizer.transform(
        [[Entity(type="line_item", children=[Entity(type="name", value="Widget")])]]
    )

    assert result == [
        [Entity(type="line_item", children=[Entity(type="item_name", value="Widget")])]
    ]


def test_transform_falls_back_to_rule_normalization_for_unfitted_group_children() -> None:
    normalizer = EntityTypeNormalizer(MockBackend())
    normalizer.mapping = {}

    result = normalizer.transform(
        [[Entity(type="Line Item", children=[Entity(type="Item Name", value="Widget")])]]
    )

    assert result == [
        [Entity(type="line_item", children=[Entity(type="item_name", value="Widget")])]
    ]


def test_save_and_load_round_trip_preserves_children(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    normalizer = EntityTypeNormalizer(MockBackend())
    normalizer.mapping = {"line_item": "line_item"}
    child_normalizer = EntityTypeNormalizer(MockBackend())
    child_normalizer.mapping = {"name": "item_name"}
    normalizer.children = {"line_item": child_normalizer}
    normalizer.save(path)

    loaded = EntityTypeNormalizer.load(path, MockBackend())

    assert loaded.mapping == normalizer.mapping
    assert loaded.children["line_item"].mapping == child_normalizer.mapping
