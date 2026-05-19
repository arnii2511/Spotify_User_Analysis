from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = ROOT / "tableau" / "extracts"
PRIMARY_WORKBOOK = ROOT / "tableau" / "User_Data_Quality_Conversion_Analytics.twb"

INJECT_START = "    <!-- STORYTELLING_DATASOURCES_START -->"
INJECT_END = "    <!-- STORYTELLING_DATASOURCES_END -->"

STORY_SOURCES = [
    ("kpi_summary.csv", "KPI Summary"),
    ("conversion_by_plan.csv", "Conversion by Plan"),
    ("conversion_by_frequency.csv", "Conversion by Frequency"),
    ("conversion_by_recommendation_rating.csv", "Conversion by Recommendation Rating"),
    ("conversion_by_time_slot.csv", "Conversion by Time Slot"),
    ("segment_scorecard.csv", "Segment Scorecard"),
    ("data_quality_checks.csv", "Data Quality Checks"),
    ("top_genres.csv", "Top Genres"),
    ("device_mix.csv", "Device Mix"),
    ("story_points.csv", "Story Points"),
]


def tableau_datatype(dtype: object) -> str:
    text = str(dtype)
    if text.startswith("int"):
        return "integer"
    if text.startswith("float"):
        return "real"
    return "string"


def tableau_role(datatype: str) -> str:
    return "measure" if datatype in {"integer", "real"} else "dimension"


def tableau_type(datatype: str) -> str:
    return "quantitative" if datatype in {"integer", "real"} else "nominal"


def title_caption(name: str) -> str:
    return name.replace("_", " ").title()


def source_block(filename: str, caption: str, index: int) -> str:
    path = EXTRACT_DIR / filename
    df = pd.read_csv(path, nrows=25)
    stem = Path(filename).stem
    datasource_name = f"federated.story_{index}_{stem}"
    connection_name = f"textscan.story_{index}_{stem}"
    object_id = f"{filename}_story_{index}"
    table = f"[{stem}#csv]"
    directory = str(EXTRACT_DIR).replace("\\", "/")

    relation_columns = []
    field_columns = []
    metadata_records = [
        "        <metadata-records>",
        "          <metadata-record class='capability'>",
        "            <remote-name />",
        "            <remote-type>0</remote-type>",
        f"            <parent-name>[{html.escape(filename)}]</parent-name>",
        "            <remote-alias />",
        "            <aggregation>Count</aggregation>",
        "            <contains-null>true</contains-null>",
        "            <attributes>",
        "              <attribute datatype='string' name='character-set'>&quot;UTF-8&quot;</attribute>",
        "              <attribute datatype='string' name='field-delimiter'>&quot;,&quot;</attribute>",
        "              <attribute datatype='string' name='header-row'>&quot;true&quot;</attribute>",
        "              <attribute datatype='string' name='locale'>&quot;en_IN&quot;</attribute>",
        "            </attributes>",
        "          </metadata-record>",
    ]

    for ordinal, (column, dtype) in enumerate(df.dtypes.items()):
        datatype = tableau_datatype(dtype)
        escaped_column = html.escape(str(column))
        local_name = f"[{escaped_column}]"
        relation_columns.append(f"            <column datatype='{datatype}' name='{escaped_column}' ordinal='{ordinal}' />")
        field_columns.append(
            "      "
            f"<column caption='{html.escape(title_caption(str(column)))}' datatype='{datatype}' name='{local_name}' "
            f"role='{tableau_role(datatype)}' type='{tableau_type(datatype)}' />"
        )
        metadata_records.extend(
            [
                "          <metadata-record class='column'>",
                f"            <remote-name>{escaped_column}</remote-name>",
                "            <remote-type>5</remote-type>",
                f"            <local-name>{local_name}</local-name>",
                f"            <parent-name>[{html.escape(filename)}]</parent-name>",
                f"            <remote-alias>{escaped_column}</remote-alias>",
                f"            <ordinal>{ordinal}</ordinal>",
                f"            <local-type>{datatype}</local-type>",
                "            <aggregation>Count</aggregation>",
                "            <contains-null>true</contains-null>",
                f"            <object-id>[{html.escape(object_id)}]</object-id>",
                "          </metadata-record>",
            ]
        )
    metadata_records.append("        </metadata-records>")

    return "\n".join(
        [
            f"    <datasource caption='{html.escape(caption)}' inline='true' name='{datasource_name}' version='18.1'>",
            "      <connection class='federated'>",
            "        <named-connections>",
            f"          <named-connection caption='{html.escape(stem)}' name='{connection_name}'>",
            f"            <connection class='textscan' directory='{html.escape(directory)}' filename='{html.escape(filename)}' password='' server='' />",
            "          </named-connection>",
            "        </named-connections>",
            f"        <relation connection='{connection_name}' name='{html.escape(filename)}' table='{html.escape(table)}' type='table'>",
            "          <columns character-set='UTF-8' header='yes' locale='en_IN' separator=','>",
            *relation_columns,
            "          </columns>",
            "        </relation>",
            *metadata_records,
            "      </connection>",
            *field_columns,
            "    </datasource>",
        ]
    )


def inject_sources(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    existing = re.compile(
        r"\n\s*<!-- STORYTELLING_DATASOURCES_START -->.*?<!-- STORYTELLING_DATASOURCES_END -->\n",
        flags=re.DOTALL,
    )
    text = existing.sub("\n", text)
    blocks = [INJECT_START]
    blocks.extend(source_block(filename, caption, index) for index, (filename, caption) in enumerate(STORY_SOURCES, 1))
    blocks.append(INJECT_END)
    injection = "\n".join(blocks) + "\n"
    text = text.replace("  </datasources>", injection + "  </datasources>", 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    inject_sources(PRIMARY_WORKBOOK)


if __name__ == "__main__":
    main()
