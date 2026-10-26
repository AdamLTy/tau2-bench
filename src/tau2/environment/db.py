import json
from typing import Any, Optional

from tau2.utils import dump_file, get_dict_hash, load_file
from tau2.utils.pydantic_utils import BaseModelNoExtra

# Lists whose element order carries no meaning, keyed by DB class. Each path
# walks ``model_dump()`` keys, with "*" visiting every value of a dict. Listed
# here rather than on the domain classes so that editing it leaves the domain
# runtime-rules digests pinned by released benchmark cells unchanged.
ORDER_INSENSITIVE_LISTS: dict[str, tuple[tuple[str, ...], ...]] = {
    # Airline tools never read payment_history by position.
    "tau2.domains.airline.data_model.FlightDB": (
        ("reservations", "*", "payment_history"),
    ),
}


def _sort_lists_at(node: Any, path: tuple[str, ...]) -> None:
    """Sort, in place, every list reached by following ``path`` from ``node``."""
    if not isinstance(node, dict):
        return
    head, rest = path[0], path[1:]
    keys = list(node) if head == "*" else [head] if head in node else []
    for key in keys:
        if rest:
            _sort_lists_at(node[key], rest)
        elif isinstance(node[key], list):
            node[key].sort(
                key=lambda item: json.dumps(item, sort_keys=True, default=str)
            )


class DB(BaseModelNoExtra):
    """Domain database.

    This is a base class for all domain databases.
    """

    @classmethod
    def load(cls, path: str) -> "DB":
        """Load the database from a structured file like JSON, YAML, or TOML."""
        data = load_file(path)
        return cls.model_validate(data)

    def dump(self, path: str, exclude_defaults: bool = False, **kwargs: Any) -> None:
        """Dump the database to a file."""
        data = self.model_dump(exclude_defaults=exclude_defaults)
        dump_file(path, data, **kwargs)

    def get_json_schema(self) -> dict[str, Any]:
        """Get the JSON schema of the database."""
        return self.model_json_schema()

    @classmethod
    def order_insensitive_lists(cls) -> tuple[tuple[str, ...], ...]:
        """Paths of this DB's lists that are compared as multisets."""
        return tuple(
            path
            for klass in cls.__mro__
            for path in ORDER_INSENSITIVE_LISTS.get(
                f"{klass.__module__}.{klass.__qualname__}", ()
            )
        )

    def get_hash(self) -> str:
        """Get the hash of the database.

        Lists named by ``order_insensitive_lists()`` are sorted first, so states
        that differ only in the order of those lists hash the same.
        """
        data = self.model_dump()
        for path in self.order_insensitive_lists():
            _sort_lists_at(data, path)
        return get_dict_hash(data)

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        return {}


def get_db_json_schema(db: Optional[DB] = None) -> dict[str, Any]:
    """Get the JSONschema of the database."""
    if db is None:
        return {}
    return db.get_json_schema()
