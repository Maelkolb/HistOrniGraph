"""Generate PAGE XML (PAGE Content Schema) for each page.

Produces a simplified but valid PAGE XML 2019 document with layout regions
and their transcription results.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom

log = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# Map our region types → PAGE XML element names
_PAGE_XML_TYPE = {
    "ParagraphRegion": "TextRegion",
    "ListRegion": "TextRegion",
    "FootnoteRegion": "TextRegion",
    "MarginaliaRegion": "TextRegion",
    "PageNumberRegion": "TextRegion",
    "TableRegion": "TableRegion",
    "ImageRegion": "ImageRegion",
    "ObjectRegion": "GraphicRegion",
}


def _coords_str(bbox: Dict[str, int]) -> str:
    """Convert bbox dict → PAGE XML Points string (polygon)."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    return f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"


def generate_pagexml(
    page_id: str,
    regions: List[Dict[str, Any]],
    image_dims: Dict[str, int],
    image_filename: str,
    output_dir: Path,
) -> Path:
    """Write a PAGE XML file for one page."""

    root = Element("PcGts", xmlns=PAGE_NS)
    metadata = SubElement(root, "Metadata")
    SubElement(metadata, "Creator").text = "journal_processor"
    SubElement(metadata, "Created").text = datetime.now(timezone.utc).isoformat()

    page = SubElement(
        root,
        "Page",
        imageFilename=image_filename,
        imageWidth=str(image_dims["width"]),
        imageHeight=str(image_dims["height"]),
    )

    reading_order_el = SubElement(page, "ReadingOrder")
    og = SubElement(reading_order_el, "OrderedGroup", id="reading_order")

    for r in sorted(regions, key=lambda r: r["reading_order"]):
        rid = r["id"]
        rtype = r["type"]
        xml_tag = _PAGE_XML_TYPE.get(rtype, "TextRegion")
        bbox = r["bbox"]

        SubElement(og, "RegionRefIndexed", index=str(r["reading_order"]), regionRef=rid)

        region_el = SubElement(page, xml_tag, id=rid, custom=f"type:{rtype}")
        SubElement(region_el, "Coords", points=_coords_str(bbox))

        # Transcription
        text = r.get("transcription", {}).get("text", "")
        if text and xml_tag == "TextRegion":
            te = SubElement(region_el, "TextEquiv")
            SubElement(te, "Unicode").text = text
        elif text and xml_tag == "TableRegion":
            te = SubElement(region_el, "TextEquiv")
            SubElement(te, "Unicode").text = text

    xml_str = tostring(root, encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding=None)
    # Remove extra xml declaration minidom adds
    pretty = "\n".join(pretty.splitlines()[1:])

    out_path = output_dir / f"{page_id}.xml"
    out_path.write_text(pretty, encoding="utf-8")
    log.debug("Wrote %s", out_path.name)
    return out_path
