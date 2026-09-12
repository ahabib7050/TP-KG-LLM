"""Programmatic introspection of the graph schema.

The schema handed to the LLM is read from the live database rather than written
by hand, so it cannot drift away from the data after a re-ingestion. Three
procedures are combined, because no single one returns everything:

- db.schema.nodeTypeProperties : node properties and their types
- db.schema.relTypeProperties  : relationship properties and their types
- db.schema.visualization      : which labels each relationship actually connects
"""

from dataclasses import dataclass

from neo4j import Driver

from graphrag.config import database_name


@dataclass(frozen=True)
class GraphSchema:
    """The parts of the schema a Cypher generator needs.

    Attributes:
        node_properties: Label to {property name: type}.
        rel_properties: Relationship type to {property name: type}.
        connections: Triples of (start label, relationship type, end label).
    """

    node_properties: dict[str, dict[str, str]]
    rel_properties: dict[str, dict[str, str]]
    connections: list[tuple[str, str, str]]

    def render(self) -> str:
        """Render the schema as compact Cypher-shaped text for a prompt.

        Ascii-art patterns are used rather than JSON because the model has to
        produce Cypher, and showing the schema in the target syntax reduces the
        distance between what it reads and what it must write.

        Returns:
            A description listing node patterns then relationship patterns.
        """
        lines = ["Node labels and properties:"]
        for label in sorted(self.node_properties):
            props = self.node_properties[label]
            rendered = ", ".join(f"{name}: {kind}" for name, kind in sorted(props.items()))
            lines.append(f"  (:{label} {{{rendered}}})" if rendered else f"  (:{label})")

        lines.append("")
        lines.append("Relationships (direction matters):")
        for start, rel_type, end in sorted(self.connections):
            props = self.rel_properties.get(rel_type, {})
            rendered = ", ".join(f"{name}: {kind}" for name, kind in sorted(props.items()))
            middle = f"[:{rel_type} {{{rendered}}}]" if rendered else f"[:{rel_type}]"
            lines.append(f"  (:{start})-{middle}->(:{end})")

        return "\n".join(lines)


def _clean_type(raw: list[str] | None) -> str:
    """Normalise the propertyTypes list returned by Neo4j into one name.

    Args:
        raw: The propertyTypes value, for example ["STRING NOT NULL"].

    Returns:
        A short type name such as "STRING", or "ANY" when unknown.
    """
    if not raw:
        return "ANY"
    return raw[0].replace(" NOT NULL", "").strip()


def _strip_label(raw: str) -> str:
    """Turn a Neo4j schema identifier such as ``:`Match``` into ``Match``.

    Args:
        raw: The nodeType or relType string reported by Neo4j.

    Returns:
        The bare label or relationship type.
    """
    return raw.strip().lstrip(":").strip("`")


def introspect(driver: Driver) -> GraphSchema:
    """Read the live schema from the database.

    Args:
        driver: An open Neo4j driver.

    Returns:
        The node properties, relationship properties and connection triples.
    """
    db = database_name()

    node_properties: dict[str, dict[str, str]] = {}
    records, _, _ = driver.execute_query("CALL db.schema.nodeTypeProperties()", database_=db)
    for record in records:
        for label in record["nodeLabels"]:
            bucket = node_properties.setdefault(label, {})
            if record["propertyName"]:
                bucket[record["propertyName"]] = _clean_type(record["propertyTypes"])

    rel_properties: dict[str, dict[str, str]] = {}
    records, _, _ = driver.execute_query("CALL db.schema.relTypeProperties()", database_=db)
    for record in records:
        rel_type = _strip_label(record["relType"])
        bucket = rel_properties.setdefault(rel_type, {})
        if record["propertyName"]:
            bucket[record["propertyName"]] = _clean_type(record["propertyTypes"])

    connections: list[tuple[str, str, str]] = []
    records, _, _ = driver.execute_query("CALL db.schema.visualization()", database_=db)
    for record in records:
        for relationship in record["relationships"]:
            start, end = relationship.nodes
            start_label = next(iter(start.labels), "?")
            end_label = next(iter(end.labels), "?")
            connections.append((start_label, relationship.type, end_label))

    return GraphSchema(
        node_properties=node_properties,
        rel_properties=rel_properties,
        connections=sorted(set(connections)),
    )
