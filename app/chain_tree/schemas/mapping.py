from typing import Dict, Optional, List, Any
from app.chain_tree.schemas.chain import Chain
from pydantic import BaseModel, Field
import pandas as pd


class ChainMap(BaseModel):
    """
    Represents a mapping between a message and its relationships.
    id (str): Unique identifier for the mapping.
    message (Optional[Message]): The message associated with the mapping.
    parent (Optional[str]): The ID of the parent message.
    children (List[str]): The IDs of the child messages.
    """

    id: str = Field(..., description="Unique identifier for the mapping.")

    message: Optional[Chain] = Field(
        None, description="The message associated with the mapping."
    )

    parent: Optional[str] = Field(None, description="The ID of the parent message.")

    children: Optional[List[str]] = Field(
        [], description="The IDs of the child messages."
    )

    references: Optional[List[str]] = Field(
        [], description="The IDs of the referenced messages."
    )

    relationships: Optional[Any] = Field(
        None, description="Relationships between the message and other messages."
    )

    depth: Optional[int] = Field(
        None, description="Depth of the message in the conversation."
    )

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"<ChainMap id={self.id} message={self.message}>"

    def to_dict(self) -> Dict:
        """Converts the ChainMap to a dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict) -> "ChainMap":
        """Converts a dictionary to a ChainMap."""
        return cls(**data)

    @classmethod
    def from_dataframe_row(cls, row: Dict) -> "ChainMap":
        """Converts a DataFrame row to a ChainMap."""
        return cls.from_dict(row)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["ChainMap"]:
        """Converts a DataFrame to a list of ChainMaps."""
        return [cls.from_dataframe_row(row) for _, row in df.iterrows()]

    def flatten(self) -> Dict:
        """Flattens the ChainMap."""
        return {
            "id": self.id,
            "message": self.message,
            "parent": self.parent,
            "children": self.children,
            "references": self.references,
            "relationships": self.relationships,
        }


class MultiChainMaps(BaseModel):
    """
    Represents a collection of ChainMaps.
    chain_maps (List[ChainMap]): List of ChainMaps.
    """

    chain_maps: List[ChainMap] = Field([], description="List of ChainMaps.")

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"<MultiChainMaps chain_maps={self.chain_maps}>"

    def to_dict(self) -> Dict:
        """Converts the MultiChainMaps to a dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict) -> "MultiChainMaps":
        """Converts a dictionary to a MultiChainMaps."""
        return cls(**data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MultiChainMaps":
        """Converts a DataFrame to a MultiChainMaps."""
        chain_maps = ChainMap.from_dataframe(df)
        return cls(chain_maps=chain_maps)

    def flatten(self) -> List[Dict]:
        """Flattens the MultiChainMaps."""
        return [chain_map.flatten() for chain_map in self.chain_maps]
