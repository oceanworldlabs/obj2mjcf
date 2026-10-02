from pathlib import Path

import pytest
from PIL import Image

from obj2mjcf import convert
from obj2mjcf.cli import Args, copy_textures, process_obj
from obj2mjcf.material import Material


def _existing_output(tmp_path):
    obj = tmp_path / "groups.obj"
    obj.write_bytes((Path(__file__).parent / "groups.obj").read_bytes())
    output = tmp_path / "groups"
    output.mkdir()
    sentinel = output / "previous-output.txt"
    sentinel.write_text("retain this output")
    return obj, sentinel


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
@pytest.mark.parametrize("invalid_option", ["axis", "format", "missing_input"])
def test_invalid_request_preserves_existing_output(
    tmp_path, entrypoint, invalid_option
):
    obj, sentinel = _existing_output(tmp_path)
    if invalid_option == "missing_input":
        obj.unlink()
    axis = "invalid" if invalid_option == "axis" else "Z"
    export = "bogus" if invalid_option == "format" else "mjcf"
    with pytest.raises((ValueError, FileNotFoundError)):
        if entrypoint == "api":
            convert(obj, export=export, up_axis=axis, overwrite=True)
        else:
            process_obj(
                obj, Args(str(tmp_path), export=export, up_axis=axis, overwrite=True)
            )
    assert sentinel.read_text() == "retain this output"


@pytest.mark.parametrize("across_materials", [False, True])
def test_same_basename_textures_keep_distinct_pixels(tmp_path, across_materials):
    output = tmp_path / "output"
    output.mkdir()
    for folder, color in [("a", (255, 0, 0)), ("b", (0, 255, 0))]:
        directory = tmp_path / folder
        directory.mkdir()
        Image.new("RGB", (4, 4), color).save(directory / "shared.png")
    albedo = Material(name="first", map_Kd="a/shared.png")
    second = Material(name="second") if across_materials else albedo
    second_attr = "map_Kd" if across_materials else "map_Pr"
    setattr(second, second_attr, "b/shared.png")
    copy_textures(albedo, tmp_path, output, 1.0)
    if across_materials:
        copy_textures(second, tmp_path, output, 1.0)
    second_texture = getattr(second, second_attr)
    assert albedo.map_Kd != second_texture
    with Image.open(output / albedo.map_Kd) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    with Image.open(output / second_texture) as image:
        assert image.getpixel((0, 0)) == (0, 255, 0)


def test_shared_texture_resizing_does_not_modify_data_map(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "shared.png")
    material = Material(name="shared", map_Kd="shared.png", map_Pr="shared.png")
    copy_textures(material, tmp_path, output, 0.5)
    assert material.map_Kd != material.map_Pr
    with Image.open(output / material.map_Kd) as image:
        assert image.size == (4, 4)
    with Image.open(output / material.map_Pr) as image:
        assert image.size == (8, 8)


def test_jpeg_conversion_keeps_distinct_png_data_map(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(tmp_path / "shared.jpg")
    Image.new("RGB", (4, 4), (0, 255, 0)).save(tmp_path / "shared.png")
    with Image.open(tmp_path / "shared.jpg") as image:
        expected_albedo = image.getpixel((0, 0))
    material = Material(name="jpeg", map_Kd="shared.jpg", map_Pr="shared.png")
    copy_textures(material, tmp_path, output, 1.0)
    assert Path(material.map_Kd).suffix == ".png"
    assert material.map_Kd != material.map_Pr
    with Image.open(output / material.map_Kd) as image:
        assert image.getpixel((0, 0)) == expected_albedo
    with Image.open(output / material.map_Pr) as image:
        assert image.getpixel((0, 0)) == (0, 255, 0)


def test_texture_outputs_are_stable_across_source_roots(tmp_path):
    outputs = []
    for run in ("first", "second"):
        source = tmp_path / run
        source.mkdir()
        output = source / "output"
        output.mkdir()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(source / "shared.png")
        material = Material(name="stable", map_Kd="shared.png")
        copy_textures(material, source, output, 1.0)
        outputs.append((material.map_Kd, (output / material.map_Kd).read_bytes()))
    assert outputs[0] == outputs[1]
