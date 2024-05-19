import os
import re
import json
import hashlib
import csv
import shutil
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import init_db, engine, Script, Scene, Element, Session, User, Section
import asyncio
import sys


class DataUploader:
    def __init__(self, directory):
        self.directory = directory
        self.logger = self.create_logger()

    def create_logger(self):
        import logging

        logger = logging.getLogger("DataUploader")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def natural_sort(self, l):
        """Sort the given list in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(l, key=alphanum_key)

    def read_text_file(self, file_path: str) -> str:
        """Reads the content of a text file."""
        with open(file_path, "r") as file:
            return file.read()

    def process_file(
        self, file_path: str, verbose: bool = True, max_retries: int = 3
    ) -> str:
        """
        Processes a file and returns its local path.

        Args:
            file_path (str): The local path of the file.
            verbose (bool): If True, enables verbose output.
            max_retries (int): Maximum number of retries for processing the file.

        Returns:
            str: The local file path.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                if verbose:
                    print(f"Processed {file_path}")
                return file_path

            except (TimeoutError, ConnectionError) as e:
                attempt += 1
                if verbose:
                    self.logger.info(
                        f"Retry {attempt}/{max_retries} for {file_path} due to error: {e}"
                    )
                if attempt == max_retries:
                    self.logger.info(
                        f"Failed to process {file_path} after {max_retries} attempts"
                    )
            except Exception as e:
                if verbose:
                    self.logger.error(f"Error processing {file_path}: {e}")
                break  # Break on other types of exceptions

        return ""

    def process_batch(
        self,
        mode: str,
        batch_files: list,
        conversation_id: str,
        path: str,
        verbose: bool,
    ) -> dict:
        """
        Processes a batch of files and constructs the output data.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            batch_files (list): List of file names to be processed.
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            verbose (bool): If True, enables verbose output.

        Returns:
            dict: A dictionary containing the structured data for the batch.
        """
        data = {"conversation_id": conversation_id, mode: []}
        for file_name in batch_files:
            file_path = os.path.join(path, mode, conversation_id, file_name)
            if mode in ["prompt", "caption"]:
                content = self.read_text_file(file_path)
                data[mode].append(content)
            else:
                processed_file = self.process_file(file_path, verbose)
                if processed_file:
                    data[mode].append(processed_file)
        return data

    def initialize_processing(
        self, mode: str, conversation_id: str, path: str, batch_size: int, verbose: bool
    ) -> list:
        """
        Initializes the processing by creating batches of files.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            batch_size (int): The number of files in each batch.
            verbose (bool): If True, enables verbose output.

        Returns:
            list: A list of tuples, each containing the arguments for processing a batch of files.
        """
        directory = os.path.join(path, mode, conversation_id)
        if not os.path.exists(directory):
            if verbose:
                self.logger.info(f"Directory not found: {directory}")
            return []

        file_list = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        file_list = self.natural_sort(file_list)
        file_list = [title for title in file_list if title != ".DS_Store"]

        return [
            (mode, file_list[i : i + batch_size], conversation_id, path, verbose)
            for i in range(0, len(file_list), batch_size)
        ]

    def construct_media_data(
        self, conversation_id: str, directory=None, **kwargs
    ) -> dict:
        """
        Manages the construction of media data across different modalities.

        Args:
            conversation_id (str): The conversation ID associated with the files.
            **kwargs: Additional keyword arguments including:
                      - path (str): The base path where the files are located.
                      - batch_size (int): The number of files in each batch.
                      - verbose (bool): If True, enables verbose output.

        Returns:
            dict: A dictionary containing the structured data for all modalities.
        """
        modalities = [
            "audio",
            "image",
            "caption",
            "prompt",
        ]
        if directory is None:
            directory = self.directory
        else:
            directory = directory

        path = kwargs.get("path", directory)
        batch_size = kwargs.get("batch_size", 10)
        verbose = kwargs.get("verbose", True)

        media_data = {"prompt": [], "caption": [], "image": [], "audio": []}

        for mode in modalities:
            tasks = self.initialize_processing(
                mode, conversation_id, path, batch_size, verbose
            )
            for task in tasks:
                batch_data = self.process_batch(*task)
                media_data[mode].extend(batch_data[mode])

        return media_data

    def upload_all_media_in_parallel(self, base_path) -> dict:
        """
        Manages the parallel upload of media files across different modalities for all conversations.

        Args:
            base_path (str): The base path where the files are located.

        Returns:
            dict: A dictionary containing the structured data for all conversations.
        """
        all_media_data = {}
        title_list = os.listdir(base_path)
        title_list = [title for title in title_list if title != ".DS_Store"]
        for title in title_list:
            directory = os.path.join(base_path, title)
            all_media_data[title] = {
                "prompt": [],
                "caption": [],
                "image": [],
                "audio": [],
            }
            for folder_id in os.listdir(os.path.join(directory, "image")):
                folder_path = os.path.join(directory, "image", folder_id)
                if os.path.isdir(folder_path):
                    media_data = self.construct_media_data(folder_id, path=directory)
                    all_media_data[title]["prompt"].extend(media_data["prompt"])
                    all_media_data[title]["caption"].extend(media_data["caption"])
                    all_media_data[title]["image"].extend(media_data["image"])
                    all_media_data[title]["audio"].extend(media_data["audio"])
        return all_media_data

    # NFTree Methods
    def writeBigUInt128BE(self, buf, uint, offset):
        upper = (uint >> 64) & 0xFFFFFFFFFFFFFFFF
        lower = uint & 0xFFFFFFFFFFFFFFFF
        buf[offset : offset + 8] = upper.to_bytes(8, byteorder="big")
        buf[offset + 8 : offset + 16] = lower.to_bytes(8, byteorder="big")

    def makeNFTDesc(self, dataHash, size, tickets):
        buf = bytearray(64)
        buf[0:32] = dataHash
        self.writeBigUInt128BE(buf, size, 32)
        self.writeBigUInt128BE(buf, tickets, 48)

        descHasher = hashlib.sha512()
        descHasher.update(buf)
        descHash = descHasher.digest()[:32]

        return {
            "nftDesc": buf,
            "nftDescHash": descHash,
            "dataHash": dataHash.hex(),
            "size": size,
            "tickets": tickets,
        }

    def NFTDescFromFile(self, path, tickets):
        with open(path, "rb") as file:
            fileBuff = file.read()
        fileInfo = os.stat(path)

        hasher = hashlib.sha512()
        hasher.update(fileBuff)
        dataHash = hasher.digest()[:32]

        return self.makeNFTDesc(dataHash, fileInfo.st_size, tickets)

    def loadNFTreeCSV(self, path):
        try:
            with open(path, mode="r") as file:
                reader = csv.DictReader(file)
                if not reader:
                    self.error_out(f"Malformed tickets CSV at {path}: could not parse")

                tickets = {}
                for row in reader:
                    if "name" not in row or "tickets" not in row:
                        self.error_out(
                            f"Malformed tickets CSV at {path}: missing 'name' or 'tickets' column"
                        )
                    name = row["name"]
                    try:
                        tickets[name] = {"tickets": int(row["tickets"])}
                    except ValueError:
                        self.error_out(
                            f"Malformed tickets CSV at {path}: invalid 'tickets' value for '{name}'"
                        )
                return tickets
        except Exception as e:
            self.error_out(f"Failed to load {path}")

    def processOneFile(self, nft_csv, src_root, src_path, dest_root):
        if src_path not in nft_csv:
            raise Exception(f"No tickets defined for {src_path} in {src_root}")
        tickets = nft_csv[src_path]["tickets"]
        srcFullPath = os.path.join(src_root, src_path)
        nftDesc = self.NFTDescFromFile(srcFullPath, tickets)

        nftDestPath = os.path.join(dest_root, nftDesc["dataHash"])
        nftDescPath = f"{nftDestPath}.desc"

        if os.path.exists(nftDestPath):
            self.error_out(f"NFT for {srcFullPath} at {nftDestPath} already exists")

        shutil.copyfile(srcFullPath, nftDestPath)
        with open(nftDescPath, "w") as f:
            f.write(json.dumps(nftDesc["nftDesc"].hex()))

        return {"nftDesc": nftDesc, "nftDestPath": nftDestPath}

    def makeMerkleTree(self, hashes):
        leaves = [hash for hash in hashes]
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        tree = [leaves]
        while len(hashes) > 1:
            next_layer = []
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            for i in range(0, len(hashes), 2):
                h1 = hashes[i]
                h2 = hashes[i + 1]
                hasher = hashlib.sha512()
                hasher.update(h1)
                hasher.update(h2)
                next_layer.append(hasher.digest()[:32])
            hashes = next_layer
            if len(next_layer) % 2 == 1 and len(next_layer) > 1:
                next_layer.append(next_layer[-1])
            tree.append(next_layer)
        self.debug(tree)
        return tree

    def makeMerkleProof(self, merkle_tree, index):
        saveIndex = index
        if index >= len(merkle_tree[0]):
            raise Exception(f"index out of bounds: {index} >= {len(merkle_tree[0])}")
        proof = []
        for i in range(len(merkle_tree) - 1):
            if index % 2 == 0:
                if index + 1 >= len(merkle_tree[i]):
                    raise Exception(
                        f"FATAL: index {index + 1} out of bounds ({len(merkle_tree[i])})"
                    )
                sibling_hash = merkle_tree[i][index + 1]
            else:
                if index <= 0:
                    raise Exception(f"FATAL: index must be positive")
                sibling_hash = merkle_tree[i][index - 1]
            proof.append(sibling_hash)
            index //= 2
        return {"hashes": [h.hex() for h in proof], "index": saveIndex}

    def processOneDirectory(self, src_root, dest_root):
        self.debug(f"Process {src_root} --> {dest_root}")

        tickets_path = os.path.join(src_root, "tickets.csv")
        nft_csv = self.loadNFTreeCSV(tickets_path)

        children = sorted(os.listdir(src_root))

        nft_infos = []
        nft_desc_hashes = []
        total_tickets = 0
        total_size = 0

        for child in children:
            if child == "tickets.csv":
                continue

            child_path = os.path.join(src_root, child)
            nft_info = None

            if os.path.isfile(child_path):
                try:
                    rec = self.processOneFile(nft_csv, src_root, child, dest_root)
                    nft_info = rec["nftDesc"]
                except Exception as e:
                    print(
                        f"WARN: failed to process NFT file at {child_path}: {e}",
                        file=sys.stderr,
                    )
                    continue
            elif os.path.isdir(child_path):
                try:
                    new_src_root = child_path
                    new_dest_root = os.path.join(dest_root, child)
                    rec = self.processOneDirectory(new_src_root, new_dest_root)
                    nft_info = rec["nftDesc"]
                except Exception as e:
                    print(
                        f"WARN: failed to process NFT collection at {child_path}: {e}",
                        file=sys.stderr,
                    )
                    continue

            if nft_info:
                nft_infos.append(nft_info)
                total_tickets += nft_info["tickets"]
                total_size += nft_info["size"]
                nft_desc_hashes.append(nft_info["nftDescHash"])

        if "." in nft_csv and "tickets" in nft_csv["."]:
            total_tickets = nft_csv["."]["tickets"]

        merkle_root = bytes.fromhex(
            "c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a"
        )
        if nft_desc_hashes:
            merkle_tree = self.makeMerkleTree(nft_desc_hashes)
            for i, info in enumerate(nft_infos):
                proof = self.makeMerkleProof(merkle_tree, i)
                with open(
                    os.path.join(dest_root, f"{info['dataHash']}.proof"), "w"
                ) as f:
                    f.write(json.dumps(proof))
            merkle_root = merkle_tree[-1][0]

        with open(os.path.join(dest_root, "root"), "w") as f:
            f.write(json.dumps(merkle_root.hex()))

        dirDesc = self.makeNFTDesc(merkle_root, total_size, total_tickets)
        dirDescPath = os.path.join(dest_root, f"{dirDesc['dataHash']}.desc")
        with open(dirDescPath, "w") as f:
            f.write(json.dumps(dirDesc["nftDesc"].hex()))

        self.debug(f"Merkle root of {src_root} is in {dirDescPath}")

        return {"nftDesc": dirDesc, "nftDestPath": dest_root}

    def verifyProof(self, root_hash, desc_hash, proof):
        idx = proof["index"]
        cur_hash = desc_hash
        self.debug(f"{cur_hash.hex()} at {idx}")
        for proof_hash in proof["hashes"]:
            proof_hash = bytes.fromhex(proof_hash)
            hasher = hashlib.sha512()
            if idx % 2 == 0:
                hasher.update(cur_hash)
                hasher.update(proof_hash)
            else:
                hasher.update(proof_hash)
                hasher.update(cur_hash)
            cur_hash = hasher.digest()[:32]
            idx //= 2
            self.debug(f"{cur_hash.hex()}")
        self.debug(f"{cur_hash.hex()} == {root_hash.hex()}")
        return cur_hash == root_hash


async def insert_data(db: AsyncSession, data: dict):
    try:
        async with db.begin():
            for data_id, content in data.items():
                session = Session(
                    user_id=1,  # assuming user ID 1 for this example
                    session_type="Hypnotherapy",
                    session_status="Completed",
                    session_description=f"Session for script {data_id}",
                )
                db.add(session)
                await db.flush()

                script = Script(
                    session_id=session.id,
                    script_type="Hypnotherapy Script",
                    script_content="",
                    script_description=f"Script for {data_id}",
                )
                db.add(script)
                await db.flush()

                # Create a list to hold section contents
                sections_content = []

                for section in content.get("prompt", []):
                    section_data = section.split("\n", 1)
                    part_title = section_data[0].strip()
                    content_text = (
                        section_data[1].strip() if len(section_data) > 1 else ""
                    )
                    section_entry = Section(
                        script_id=script.id, part_title=part_title, content=content_text
                    )
                    sections_content.append(content_text)
                    db.add(section_entry)
                    await db.flush()

                # Join section contents to form the script content
                script.script_content = "\n".join(sections_content)
                db.add(script)
                await db.flush()

                for index, text in enumerate(content.get("caption", [])):
                    scene = Scene(
                        script_id=script.id,
                        scene_type="Hypnotherapy Scene",
                        scene_description=text,
                    )
                    if "image" in content:
                        scene.scene_image = (
                            content["image"][index]
                            if index < len(content["image"])
                            else None
                        )
                        db.add(scene)
                        await db.flush()

                    if "audio" in content:
                        scene.scene_audio = (
                            content["audio"][index]
                            if index < len(content["audio"])
                            else None
                        )
                        db.add(scene)
                        await db.flush()

    except Exception as e:
        await db.rollback()
        raise e
