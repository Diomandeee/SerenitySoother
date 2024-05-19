from typing import List, Tuple, Optional, Dict, Any
from app.response.interface import IChainFactory, IChainTree
from app.chain_tree.schemas import Content, ChainCoordinate


class ChainNode:
    def __init__(self, key: str, value: int, level: int = 0) -> None:
        self.key = key
        self.value = value
        self.level = level
        self.parent: ChainNode = None
        self.children: List[ChainNode] = []

    def add_child(self, child_node: "ChainNode") -> None:
        child_node.parent = self
        self.children.append(child_node)

    def remove_child(self, child_node: "ChainNode") -> None:
        child_node.parent = None
        self.children.remove(child_node)

    def __str__(self):
        return f"Node(id={self.key}, value={self.value})"

    def __repr__(self):
        return self.__str__()


class ChainLink:
    def __init__(self, root: ChainNode) -> None:
        self.root = root

    def __str__(self):
        return f"ChainLink(root={self.root})"

    def __repr__(self):
        return self.__str__()


class ChainTreeLink(IChainTree):
    """
    Represents a ChainTree data structure.
    """

    def __init__(
        self,
        chain_factory: IChainFactory,
        allow_duplicates: bool = False,
    ):
        self.chain_factory = chain_factory
        self.allow_duplicates = allow_duplicates
        self.chains_links = {}
        self.chains = []
        self.nodes = []

    def add_chain(
        self,
        chain_type: str,
        id: str,
        content: Content,
        coordinate: ChainCoordinate,
        parent: Optional[str] = None,
    ) -> None:
        """
        Adds a chain to the Chainchain_tree.tree.
        """
        if not self.allow_duplicates and id in self.chains:
            raise ValueError(f"Duplicate id {id} found in sequence.")

        chain = self.chain_factory.create_chain(
            chain_type, id, content, coordinate, parent
        )
        self.chains.append(chain)

        # Add the chain to the chain with the same id
        if id not in self.chains_links:
            self.chains_links[id] = ChainLink(chain)
        else:
            self.chains_links[id].root.add_child(chain)

        # Add the chain to the chains with values that are prefixes of the chain's id
        for i in range(1, len(id)):
            prefix = id[:i]
            if prefix not in self.chains_links:
                self.chains_links[prefix] = ChainLink(chain)
            else:
                self.chains_links[prefix].root.add_child(chain)

    def remove_chain(self, id: str) -> None:
        """
        Removes a chain from the Chainchain_tree.tree.
        """
        chain_to_remove = None
        for chain in self.chains:
            if chain.id == id:
                chain_to_remove = chain
                break

        if chain_to_remove is None:
            raise ValueError(f"Chain with id {id} not found.")

        # Remove the chain from the chains with values that are prefixes of the chain's id
        for i in range(len(id)):
            prefix = id[:i]
            if prefix in self.chains_links:
                self.chains_links[prefix].root.remove_child(chain_to_remove)

        self.chains.remove(chain_to_remove)

    def update_chain(
        self,
        id: str,
        new_content: Optional[Content] = None,
        new_coordinate: Optional[ChainCoordinate] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
    ):
        for chain in self.chains:
            if chain.id == id:
                if new_content is not None:
                    chain.content = new_content
                if new_coordinate is not None:
                    chain.coordinate = new_coordinate
                if new_metadata is not None:
                    chain.metadata = new_metadata
                break

    def add_node(self, key: str, value: int) -> None:
        """
        Adds a node to the Chainchain_tree.tree.
        """
        if not self.allow_duplicates and key in self.chains:
            raise ValueError(f"Duplicate key {key} found in sequence.")

        node = ChainNode(key, value)
        self.nodes.append(node)

        # Add the node to the chain with the same key
        if key not in self.chains_links:
            self.chains_links[key] = ChainLink(node)
        else:
            self.chains_links[key].root.add_child(node)

        # Add the node to the chains with keys that are prefixes of the node's key
        for i in range(1, len(key)):
            prefix = key[:i]
            if prefix not in self.chains_links:
                self.chains_links[prefix] = ChainLink(node)
            else:
                self.chains_links[prefix].root.add_child(node)

    def remove_node(self, key: str) -> None:
        """
        Removes a node from the Chainchain_tree.tree.
        """
        node_to_remove = None
        for node in self.nodes:
            if node.key == key:
                node_to_remove = node
                break

        if node_to_remove is None:
            raise ValueError(f"Node with key {key} not found.")

        # Remove the node from the chains with keys that are prefixes of the node's key
        for i in range(len(key)):
            prefix = key[:i]
            if prefix in self.chains_links:
                self.chains_links[prefix].root.remove_child(node_to_remove)

        self.nodes.remove(node_to_remove)

    def add_nodes(self, nodes: List[Tuple[str, int]]) -> None:
        """
        Adds multiple nodes to the Chainchain_tree.tree.
        """
        for node in nodes:
            self.add_node(node[0], node[1])

    def remove_nodes(self, keys: List[str]) -> None:
        """
        Removes multiple nodes from the Chainchain_tree.tree.
        """
        for key in keys:
            self.remove_node(key)

    def get_chains_by_coordinate(self, coordinate: ChainCoordinate):
        return [chain for chain in self.chains if chain.coordinate == coordinate]

    def get_chains(self):
        return self.chains

    def clear_chains(self):
        self.chains = []
        self.chains_links = {}

    def remove_last_chain(self):
        self.chains.pop()

    def remove_first_chains(self, n: int):
        self.chains = self.chains[n:]

    def remove_last_chains(self, n: int):
        self.chains = self.chains[:-n]

    def get_chain(self, id: str):
        for chain in self.chains:
            if chain.id == id:
                return chain
        return None

    def get_last_chain(self):
        return self.chains[-1]

    def remove_chain(self, id: str):
        self.chains = [chain for chain in self.chains if chain.id != id]

    def remove_chains(self, ids: List[str]):
        self.chains = [chain for chain in self.chains if chain.id not in ids]

    def get_chains_by_type(self, chain_type: str):
        return [chain for chain in self.chains if chain.chain_type == chain_type]

    def get_nodes(self) -> List[ChainNode]:
        """
        Gets all nodes from the Chainchain_tree.tree.
        """
        return self.nodes

    def get_node(self, key: str) -> ChainNode:
        """
        Gets a node from the Chainchain_tree.tree.
        """
        if key not in self.chains_links:
            raise ValueError(f"Key {key} not found in sequence.")

        return self.chains_links[key].root

    def search_partial_sequence(self, partial_sequence: List[str]) -> List[ChainNode]:
        """
        Searches for nodes in the ChainTree that match the given partial sequence.
        """

        def search_chain(chain_key: str, seq_index: int) -> List[ChainNode]:
            if seq_index >= len(partial_sequence):
                return [self.chains_links[chain_key].root]

            result = []
            for child in self.chains_links[chain_key].root.children:
                if child.key == partial_sequence[seq_index]:
                    result.extend(search_chain(child.key, seq_index + 1))

            return result

        return search_chain(partial_sequence[0], 1)

    def _traverse(self, node: ChainNode, indent: str = "") -> List[str]:
        """
        Helper method for __str__ that recursively traverses the ChainTree and generates a list of strings
        representing the nodes and their relationships in the chain_tree.tree.
        """
        lines = [f"{indent}{node.key}: {node.value}"]
        for child in node.children:
            lines.extend(self._traverse(child, indent + "  "))
        return lines
