import yaml
import os
import sxpb
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import OrderedDict
import json
from typing import Any, Dict, List

script_dir = os.path.dirname(os.path.abspath(__file__))

DataType = List[Dict[str, Any]]


# From https://stackoverflow.com/a/21912744
def represent_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.MappingNode("tag:yaml.org,2002:map", value)


yaml.add_representer(OrderedDict, represent_ordereddict)


class IndentedDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentedDumper, self).increase_indent(flow, False)


def generate_yaml(data: Dict[str, Any]) -> str:
    """
    Generates a canonical YAML representation from a dictionary of Python objects.
    The data is first converted to a JSON string and then parsed back into an
    OrderedDict to ensure that custom sxpb types are converted to plain types
    and the order is preserved.
    """
    json_str = json.dumps(data)
    ordered_data = json.loads(json_str, object_pairs_hook=OrderedDict)
    return yaml.dump(
        ordered_data,
        Dumper=IndentedDumper,
        indent=2,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip()


def generate_json(data: Dict[str, Any], mode: str = "pretty") -> str:
    """
    Generates a JSON representation from a dictionary of Python objects.
    - 'pretty': Standard human-readable format with 2-space indent.
    - 'compact': A single line with no extra spaces.
    - 'oneline': A single line with standard spacing.
    """
    if mode == "pretty":
        return json.dumps(data, indent=2, ensure_ascii=False)
    elif mode == "compact":
        return json.dumps(data, indent=None, separators=(",", ":"), ensure_ascii=False)
    elif mode == "oneline":
        return json.dumps(data, indent=None, ensure_ascii=False)
    raise ValueError(f"Unknown JSON generation mode: {mode}")


def generate_jsonl(data: DataType, mode: str = "pretty") -> str:
    """
    Generates a JSONL representation from a list of Python objects.
    - 'pretty': Standard human-readable format.
    - 'compact': No extra spaces.
    - 'oneline': A single line with standard spacing.
    """
    if mode == "pretty":
        return "\n".join(
            [json.dumps(item, ensure_ascii=False, indent=None) for item in data]
        )
    elif mode == "compact":
        return "\n".join(
            [
                json.dumps(item, ensure_ascii=False, indent=None, separators=(",", ":"))
                for item in data
            ]
        )
    elif mode == "oneline":
        return "\n".join(
            [json.dumps(item, ensure_ascii=False, indent=None) for item in data]
        )
    raise ValueError(f"Unknown JSONL generation mode: {mode}")


def generate_xml(data: DataType, root_element_name: str, mode: str = "pretty") -> str:
    """
    Generates a canonical XML representation from a list of Python objects.
    - 'pretty': Standard human-readable format.
    - 'compact': No extra spaces.
    """
    # Create a dummy root to hold all items, will be stripped later
    root = ET.Element("root")
    for item_data in data:
        item_element = ET.Element(root_element_name)
        _build_xml_element_recursive(item_element, item_data)
        root.append(item_element)

    if mode == "pretty":
        # Pretty-print the XML
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Remove the XML declaration and the dummy <root> tags
        pretty_xml_lines = pretty_xml.splitlines()
        if pretty_xml_lines and "<?xml" in pretty_xml_lines[0]:
            pretty_xml_lines.pop(0)

        # Join non-empty lines, skipping the <root> and </root> tags
        xml_content = "\n".join(
            line
            for line in pretty_xml_lines
            if line.strip() and line.strip() not in ("<root>", "</root>")
        )

        # The pretty printer adds extra indentation, remove one level
        xml_content = "\n".join(
            line[2:] if line.startswith("  ") else line
            for line in xml_content.splitlines()
        )
        return xml_content
    elif mode == "compact":
        # Create a single string with no newlines or extra spaces
        rough_string = ET.tostring(root, "utf-8", short_empty_elements=True).decode()
        # Remove the dummy root tags and the XML declaration
        no_root = rough_string.replace("<root>", "").replace("</root>", "")
        return "".join(no_root.splitlines()).strip()
    raise ValueError(f"Unknown XML generation mode: {mode}")


def _build_xml_element_recursive(parent: ET.Element, data: Any) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            element = ET.SubElement(parent, key)
            _build_xml_element_recursive(element, value)
    elif isinstance(data, list):
        # Determine the item name. For a key like "moons", the sub-items are "moon".
        # This is a convention. If the parent tag is plural, singularize it for the child.
        item_name = parent.tag
        if item_name.endswith("s"):
            item_name = item_name[:-1]

        for item_data in data:
            item_element = ET.SubElement(parent, item_name)
            _build_xml_element_recursive(item_element, item_data)
    else:
        parent.text = str(data)


def generate_txtpb(data: DataType, root_element_name: str, mode: str = "pretty") -> str:
    """Generates a TXTPB representation from a dictionary of Python objects."""
    is_pretty = mode == "pretty"
    is_oneline = mode == "oneline"

    items = []
    for item in data:
        space_before_brace = " " if is_pretty or is_oneline else ""
        newline = "\n" if is_pretty else ""

        content = _build_txtpb_string_recursive(item, 1, mode)

        # Handle spacing inside braces for oneline
        if is_oneline and content:
            content = f" {content} "

        items.append(
            f"{root_element_name}{space_before_brace}{{{newline}{content}{newline if is_pretty and content else ''}}}"
        )

    separator = "\n" if is_pretty else " "
    result = separator.join(items)
    if is_pretty and result:
        result += "\n"
    return result


def _build_txtpb_string_recursive(data: Any, indent_level: int, mode: str) -> str:
    """Recursive helper to build the string representation of the data."""
    if not isinstance(data, dict):
        return ""

    is_pretty = mode == "pretty"
    is_oneline = mode == "oneline"
    is_compact = mode == "compact"

    indent_str = "  " * indent_level if is_pretty else ""
    newline = "\n" if is_pretty else ""
    space_after_colon = " " if is_pretty or is_oneline else ""
    space_before_brace = " " if is_pretty or is_oneline else ""
    field_separator = " " if is_oneline or is_compact else ""

    parts = []
    for key, value in data.items():
        if isinstance(value, dict):
            content = _build_txtpb_string_recursive(value, indent_level + 1, mode)
            if is_oneline and content:
                content = f" {content} "
            closer = f"{newline}{indent_str}" if is_pretty and content else ""
            parts.append(
                f"{indent_str}{key}{space_before_brace}{{{newline}{content}{closer}}}"
            )
        elif isinstance(value, list):
            for item in value:
                content = _build_txtpb_string_recursive(item, indent_level + 1, mode)
                if is_oneline and content:
                    content = f" {content} "
                closer = f"{newline}{indent_str}" if is_pretty and content else ""
                parts.append(
                    f"{indent_str}{key}{space_before_brace}{{{newline}{content}{closer}}}"
                )
        else:
            val_str = str(value).replace('"', r"\"")
            parts.append(f'{indent_str}{key}:{space_after_colon}"{val_str}"')

    if is_pretty:
        return (newline).join(parts)
    return field_separator.join(parts)


def generate_sxpb(data: Any, mode: str = "pretty") -> str:
    """
    Generates an SXPB representation from a list of Python objects.
    - 'pretty': Standard human-readable format.
    - 'oneline': A single line with minimal spacing.
    - 'compact': A single line with no extra spaces.
    """
    if mode == "pretty":
        return sxpb.dumps(data, indent=1)
    elif mode == "oneline":
        return sxpb.dumps(data, indent=0)
    elif mode == "compact":
        return sxpb.dumps(data, indent=-1)
    raise ValueError(f"Unknown SXPB generation mode: {mode}")


if __name__ == "__main__":
    data: DataType = [
        {
            "type": "planet",
            "name": "Kepler-186f",
            "properties": {
                "mass": "1.47 M⊕",
                "radius": "1.17 R⊕",
                "gravity": "1.1 g",
            },
            "question": "What is the mass of Kepler-186f?",
            "answer": "1.47 M⊕",
        },
        {
            "type": "star",
            "name": "TRAPPIST-1",
            "properties": {
                "distance": "40.66 light-years",
                "spectral_type": "M8V",
                "mass": "0.089 M☉",
            },
            "question": "What is the spectral type of TRAPPIST-1?",
            "answer": "M8V",
        },
    ]

    # This script is not meant to be run directly anymore.
    pass

    print("Data generated successfully in all formats.")
