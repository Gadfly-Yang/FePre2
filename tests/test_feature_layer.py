import pytest

from model import resolve_feature_layer, resolve_visual_model_config


def test_resolve_feature_layer_aliases():
    assert resolve_feature_layer("final") == "final"
    assert resolve_feature_layer("last") == "final"
    assert resolve_feature_layer("output") == "final"


def test_resolve_feature_layer_projected_cls_layers():
    assert resolve_feature_layer("L1") == 0
    assert resolve_feature_layer("L12") == 11
    assert resolve_feature_layer("11") == 11
    assert resolve_feature_layer(11) == 11


def test_resolve_feature_layer_raw_cls_layers():
    assert resolve_feature_layer("L1_CLS") == ("cls", 0)
    assert resolve_feature_layer("L6_CLS") == ("cls", 5)
    assert resolve_feature_layer("L12_CLS") == ("cls", 11)
    assert resolve_feature_layer("l12-cls") == ("cls", 11)
    assert resolve_feature_layer("layer12_cls") == ("cls", 11)
    assert resolve_feature_layer("layer12-cls") == ("cls", 11)


@pytest.mark.parametrize("feature_layer", ["", "   "])
def test_resolve_feature_layer_rejects_empty_strings(feature_layer):
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_feature_layer(feature_layer)

@pytest.mark.parametrize("feature_layer", ["L0", "L0_CLS"])
def test_resolve_feature_layer_rejects_zero_one_based_layers(feature_layer):
    with pytest.raises(ValueError, match=">= 1"):
        resolve_feature_layer(feature_layer)


def test_resolve_visual_model_config_defaults_to_legacy_clip():
    cfg = resolve_visual_model_config({})
    assert cfg == {
        "type": "clip",
        "name": "openai/clip-vit-large-patch14",
        "feature_pooling": "default",
    }


def test_resolve_visual_model_config_uses_legacy_clip_model_name():
    cfg = resolve_visual_model_config({"clip_model": "openai/clip-vit-base-patch16"})
    assert cfg["type"] == "clip"
    assert cfg["name"] == "openai/clip-vit-base-patch16"
    assert cfg["feature_pooling"] == "default"


def test_resolve_visual_model_config_accepts_new_visual_model_block():
    cfg = resolve_visual_model_config(
        {
            "clip_model": "openai/clip-vit-large-patch14",
            "visual_model": {
                "type": "dinov2",
                "name": "facebook/dinov2-base",
                "feature_pooling": "cls",
            },
        }
    )
    assert cfg["type"] == "dinov2"
    assert cfg["name"] == "facebook/dinov2-base"
    assert cfg["feature_pooling"] == "cls"
