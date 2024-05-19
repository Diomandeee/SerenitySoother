from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from uuid import uuid4


class ChainCoordinate(BaseModel):
    id: str = Field(
        str(uuid4()),
        description="The ID of the node.",
    )

    x: Any = Field(0, description="The x-coordinate of the coordinate.")

    y: Any = Field(0, description="The y-coordinate of the coordinate.")

    z: Any = Field(0, description="The z-coordinate of the coordinate.")

    t: Any = Field(0, description="The t-coordinate of the coordinate.")

    n_parts: int = Field(0, description="The number of parts of the coordinate.")

    parent: Optional[str] = Field(
        None,
        description="The ID of the parent node.",
    )

    children: List["ChainCoordinate"] = Field(
        [],
        description="The list of children nodes.",
    )

    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "id": "1234",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "t": 0.0,
            "n_parts": 0,
            "parent": "5678",
            "children": [],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ChainCoordinate to a dictionary."""
        return self.dict(exclude_none=True)


class BaseOperations(ChainCoordinate):
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        t: float = 0.0,
        n_parts: float = 0.0,
        **kwargs,
    ):
        super().__init__(x=x, y=y, z=z, t=t, n_parts=n_parts, **kwargs)

    def __getitem__(self, key: str) -> float:
        return self.fetch_value(key)

    def __setitem__(self, key: str, value: float) -> None:
        setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.dict())

    def __contains__(self, key: str) -> bool:
        return key in self.dict()

    def __iter__(self):
        return iter(self.dict().values())

    def fetch_value(self, field: str) -> float:
        """Fetch a value from the coordinate fields."""
        return getattr(self, field, 0.0)

    @classmethod
    def get_coordinate_fields(cls) -> List[str]:
        """Return names of the coordinate fields."""
        return [
            "x",
            "y",
            "z",
            "t",
            "n_parts",
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseOperations":
        """Initialize Coordinate from a dictionary."""
        return cls(**data)

    @staticmethod
    def get_coordinate_names() -> List[str]:
        """Return names of the coordinate dimensions."""
        return [
            "depth_x",
            "sibling_y",
            "sibling_count_z",
            "time_t",
            "n_parts",
        ]

    @classmethod
    def get_default_coordinates(cls) -> Dict[str, Any]:
        """Get the default values for the coordinates."""
        return {
            "depth_x": 0.0,
            "sibling_y": 0.0,
            "sibling_count_z": 0.0,
            "time_t": 0.0,
            "n_parts": 0.0,
        }

    @staticmethod
    def from_tuple(values: tuple) -> "BaseOperations":
        """Initialize Coordinate from a tuple."""
        return BaseOperations(
            x=values[0],
            y=values[1],
            z=values[2],
            t=values[3] if len(values) > 3 else 0.0,
            n_parts=values[4] if len(values) > 4 else 0.0,
        )

    def to_list(self) -> List[float]:
        """Convert Coordinate to list."""
        return [self.x, self.y, self.z, self.t, self.n_parts]

    def to_dict(self) -> dict:
        """Convert Coordinate to dict."""
        return self.dict()

    def tuple(self) -> tuple:
        """Convert Coordinate to tuple."""
        return tuple(self.dict().values())
