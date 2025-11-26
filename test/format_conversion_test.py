import os
import sxpb
import yaml
import json
from src.generate_data import generate_jsonl, generate_txtpb, generate_xml
from sxpb.jsonutil import to_plain_types


def get_benchmark_names():
    """Returns a list of all benchmark names."""
    return ["solarsystem"]


def test_format_conversions():
    """
    Tests that the data generated from data.sxpb matches the content
    of the other data format files.
    """
    for benchmark_name in get_benchmark_names():
        # Load the sxpb data, which is the source of truth
        sxpb_file_path = os.path.join(
            os.path.dirname(__file__), "../data", benchmark_name, "data.sxpb"
        )
        with open(sxpb_file_path, "r", encoding="utf-8") as f:
            sxpb_content = f.read()

        # Parse the sxpb data to a Python object
        parsed_data = sxpb.loads(sxpb_content, precise=True)
        assert parsed_data is not None

        # Define the test data directory
        test_data_dir = os.path.join(os.path.dirname(__file__), "data", benchmark_name)
        plain_data = to_plain_types(parsed_data)

        # 1. Test SxPB formatting
        sxpb_oneline_file_path = os.path.join(test_data_dir, "data.oneline.sxpb")
        with open(sxpb_oneline_file_path, "r", encoding="utf-8") as f:
            expected_sxpb_oneline_content = f.read()
        assert sxpb.dumps(parsed_data, indent=0) == expected_sxpb_oneline_content

        sxpb_compact_file_path = os.path.join(test_data_dir, "data.compact.sxpb")
        with open(sxpb_compact_file_path, "r", encoding="utf-8") as f:
            expected_sxpb_compact_content = f.read()
        assert sxpb.dumps(parsed_data, indent=-1) == expected_sxpb_compact_content

        # 2. Test JSON conversion
        json_file_path = os.path.join(test_data_dir, "data.json")
        with open(json_file_path, "r", encoding="utf-8") as f:
            expected_json_data = json.load(f)
        assert plain_data == expected_json_data

        json_compact_file_path = os.path.join(test_data_dir, "data.compact.json")
        with open(json_compact_file_path, "r", encoding="utf-8") as f:
            expected_json_compact_content = f.read()
        generated_json_compact_content = json.dumps(
            plain_data, ensure_ascii=False, separators=(",", ":")
        )
        assert generated_json_compact_content == expected_json_compact_content

        json_oneline_file_path = os.path.join(test_data_dir, "data.oneline.json")
        with open(json_oneline_file_path, "r", encoding="utf-8") as f:
            expected_json_oneline_content = f.read()
        generated_json_oneline_content = json.dumps(plain_data, ensure_ascii=False)
        assert generated_json_oneline_content == expected_json_oneline_content

        # 3. Test YAML conversion
        yaml_file_path = os.path.join(test_data_dir, "data.yaml")
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            expected_yaml_data = yaml.load(f, Loader=yaml.UnsafeLoader)
        assert plain_data == expected_yaml_data

        # Get the root element name and the data list.
        assert isinstance(plain_data, dict)
        root_element_name = list(plain_data.keys())[0]
        data_list = plain_data[root_element_name]

        # 4. Test XML conversion
        xml_file_path = os.path.join(test_data_dir, "data.xml")
        with open(xml_file_path, "r", encoding="utf-8") as f:
            expected_xml_content = f.read()
        generated_xml_content = generate_xml(data_list, root_element_name)
        assert generated_xml_content == expected_xml_content

        xml_compact_file_path = os.path.join(test_data_dir, "data.compact.xml")
        with open(xml_compact_file_path, "r", encoding="utf-8") as f:
            expected_xml_compact_content = f.read()
        generated_xml_compact_content = "".join(
            line.strip() for line in generated_xml_content.splitlines()
        )
        assert generated_xml_compact_content == expected_xml_compact_content

        # 5. Test TXTPB conversion
        txtpb_file_path = os.path.join(test_data_dir, "data.txtpb")
        with open(txtpb_file_path, "r", encoding="utf-8") as f:
            expected_txtpb_content = f.read()
        generated_txtpb_content = generate_txtpb(
            data_list, root_element_name, mode="pretty"
        )
        assert generated_txtpb_content == expected_txtpb_content

        txtpb_oneline_file_path = os.path.join(test_data_dir, "data.oneline.txtpb")
        with open(txtpb_oneline_file_path, "r", encoding="utf-8") as f:
            expected_txtpb_oneline_content = f.read()
        generated_txtpb_oneline_content = generate_txtpb(
            data_list, root_element_name, mode="oneline"
        )
        assert generated_txtpb_oneline_content == expected_txtpb_oneline_content

        txtpb_compact_file_path = os.path.join(test_data_dir, "data.compact.txtpb")
        with open(txtpb_compact_file_path, "r", encoding="utf-8") as f:
            expected_txtpb_compact_content = f.read()
        generated_txtpb_compact_content = generate_txtpb(
            data_list, root_element_name, mode="compact"
        )
        assert generated_txtpb_compact_content == expected_txtpb_compact_content

        # 6. Test JSONL conversion
        jsonl_file_path = os.path.join(test_data_dir, "data.jsonl")
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            expected_jsonl_content = f.read()
        generated_jsonl_content = generate_jsonl(data_list)
        assert generated_jsonl_content == expected_jsonl_content
