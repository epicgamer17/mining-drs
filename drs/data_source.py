from typing import Any

"""
Generic data point for data yielded by DataSource.

Note on alternatives:
If you prefer standard Python semantics, you could potentially bypass `DataSource` entirely 
and use a plain Python generator (e.g. `yield DataPoint(...)`). However, the `DataSource` 
class is retained to seamlessly integrate with `Module`, making it a first-class citizen 
in the simulation graph and tracking execution contexts and telemetry.
"""


class DataPoint:
    """A single batch of data yielded by a DataSource.

    Fields are accessed as attributes. Any keyword argument passed to
    ``__init__`` is stored and accessible via ``.name``::

        point = DataPoint(weight=40000.0, priority=1)
        point.weight    # → 40000.0
        point.priority  # → 1
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize a new DataPoint with arbitrary attributes.

        Args:
            **kwargs: Arbitrary keyword arguments that will be stored and exposed 
                as attributes on the DataPoint instance.
        """
        self._data = dict(kwargs)
        self._source = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no field '{name}'. "
                f"Available fields: {list(self._data)}"
            )

    def __repr__(self):
        items = ", ".join(f"{k}={v}" for k, v in self._data.items())
        return f"DataPoint({items})"
