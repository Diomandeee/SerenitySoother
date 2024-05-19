#!/usr/bin/env python3

import os
import sys
import json
import hashlib
import csv
import shutil
from pathlib import Path
import argparse

verbose = False


def error_out(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def debug(message):
    if verbose:
        print(message, file=sys.stderr)


def writeBigUInt128BE(buf, uint, offset):
    upper = (uint >> 64) & 0xFFFFFFFFFFFFFFFF
    lower = uint & 0xFFFFFFFFFFFFFFFF
    buf[offset : offset + 8] = upper.to_bytes(8, byteorder="big")
    buf[offset + 8 : offset + 16] = lower.to_bytes(8, byteorder="big")


def makeNFTDesc(dataHash, size, tickets):
    buf = bytearray(64)
    buf[0:32] = dataHash
    writeBigUInt128BE(buf, size, 32)
    writeBigUInt128BE(buf, tickets, 48)

    descHasher = hashlib.sha512()
    descHasher.update(buf)
    descHash = descHasher.digest()[:32]

    # Ensure the data hash length is 64 hexadecimal characters (32 bytes)
    assert len(dataHash.hex()) == 64, "Hash length is incorrect"

    return {
        "nftDesc": buf,
        "nftDescHash": descHash,
        "dataHash": dataHash.hex(),
        "size": size,
        "tickets": tickets,
    }


def NFTDescFromFile(path, tickets):
    with open(path, "rb") as file:
        fileBuff = file.read()
    fileInfo = os.stat(path)

    hasher = hashlib.sha512()
    hasher.update(fileBuff)
    dataHash = hasher.digest()[:32]

    return makeNFTDesc(dataHash, fileInfo.st_size, tickets)


def loadNFTreeCSV(path):
    try:
        with open(path, mode="r") as file:
            reader = csv.DictReader(file)
            if not reader:
                error_out(f"Malformed tickets CSV at {path}: could not parse")

            tickets = {}
            for row in reader:
                if "name" not in row or "tickets" not in row:
                    error_out(
                        f"Malformed tickets CSV at {path}: missing 'name' or 'tickets' column"
                    )
                name = row["name"]
                try:
                    tickets[name] = {"tickets": int(row["tickets"])}
                except ValueError:
                    error_out(
                        f"Malformed tickets CSV at {path}: invalid 'tickets' value for '{name}'"
                    )
            return tickets
    except Exception as e:
        error_out(f"Failed to load {path}")


def processOneFile(nft_csv, src_root, src_path, dest_root):

    if (
        src_path.startswith(".")
        or src_path in ["thumbs.db"]
        or src_path.endswith(".DS_Store")
    ):
        debug(f"Skipping system file {src_path}")
        return None  # Return None or continue depending on your script structure

    if src_path not in nft_csv:
        raise Exception(f"No tickets defined for {src_path} in {src_root}")
    tickets = nft_csv[src_path]["tickets"]
    srcFullPath = os.path.join(src_root, src_path)
    nftDesc = NFTDescFromFile(srcFullPath, tickets)

    file_hash = nftDesc["dataHash"]
    nftDestDir = os.path.join(dest_root, file_hash)
    os.makedirs(nftDestDir, exist_ok=True)
    nftDestPath = os.path.join(nftDestDir, file_hash)
    nftDescPath = f"{nftDestPath}.desc"

    if os.path.exists(nftDestPath):
        error_out(f"NFT for {srcFullPath} at {nftDestPath} already exists")

    shutil.copyfile(srcFullPath, nftDestPath)
    with open(nftDescPath, "w") as f:
        f.write(json.dumps(nftDesc["nftDesc"].hex()))

    return {"nftDesc": nftDesc, "nftDestPath": nftDestPath}


def makeMerkleTree(hashes):
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
    return tree


def makeMerkleProof(merkle_tree, index):
    saveIndex = index
    if index >= len(merkle_tree[0]):
        raise Exception(f"index out of bounds: {index} >= {len(merkle_tree[0])}")
    proof = []
    for i in range(len(merkle_tree) - 1):
        if index % 2 == 0:
            if index + 1 >= len(merkle_tree[i]):
                raise Exception(f"index {index + 1} out of bounds")
            sibling_hash = merkle_tree[i][index + 1]
        else:
            sibling_hash = merkle_tree[i][index - 1]
        proof.append(sibling_hash)
        index //= 2
    return {"hashes": [h.hex() for h in proof], "index": saveIndex}


def processOneDirectory(src_root, dest_root, top_level=True):
    debug(f"Process {src_root} --> {dest_root}")

    tickets_path = os.path.join(src_root, "tickets.csv")
    nft_csv = loadNFTreeCSV(tickets_path)

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
                rec = processOneFile(nft_csv, src_root, child, dest_root)
                if not rec:
                    continue
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
                rec = processOneDirectory(new_src_root, new_dest_root, top_level=False)
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
        merkle_tree = makeMerkleTree(nft_desc_hashes)
        for i, info in enumerate(nft_infos):
            proof = makeMerkleProof(merkle_tree, i)
            proof_dir = os.path.join(dest_root, info["dataHash"])
            proof_path = os.path.join(proof_dir, f"{info['dataHash']}.proof")
            os.makedirs(proof_dir, exist_ok=True)  # Ensure the directory exists
            with open(proof_path, "w") as f:
                f.write(json.dumps(proof))
        merkle_root = merkle_tree[-1][0]

    if top_level:
        root_path = os.path.join(dest_root, "root")
        with open(root_path, "w") as f:
            f.write(json.dumps(merkle_root.hex()))

    dirDesc = makeNFTDesc(merkle_root, total_size, total_tickets)
    dirDescPath = os.path.join(dest_root, f"{dirDesc['dataHash']}.desc")
    with open(dirDescPath, "w") as f:
        f.write(json.dumps(dirDesc["nftDesc"].hex()))

    debug(f"Merkle root of {src_root} is in {dirDescPath}")

    return {"nftDesc": dirDesc, "nftDestPath": dest_root}


def verifyProof(root_hash, desc_hash, proof):
    idx = proof["index"]
    cur_hash = desc_hash
    debug(f"{cur_hash.hex()} at {idx}")
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
        debug(f"{cur_hash.hex()}")
    debug(f"{cur_hash.hex()} == {root_hash.hex()}")
    return cur_hash == root_hash


def usage(progname, command):
    print(f"Usage: {progname} [opts] command ARGS", file=sys.stderr)

    if command == "build":
        print(f"\nCommand usage: {progname} build SRC_DIR DEST_DIR", file=sys.stderr)
        print("Where:", file=sys.stderr)
        print(
            "   SRC_DIR: the path to the directory that contains all the original NFTs",
            file=sys.stderr,
        )
        print(
            "   DEST_DIR: the path to the directory in which to store all the NFTree data to upload",
            file=sys.stderr,
        )
    elif command == "verify":
        print(
            f"\nCommand usage: {progname} verify ROOT_PATH DATA_HASH",
            file=sys.stderr,
        )
        print("Where:", file=sys.stderr)
        print(
            "   ROOT_PATH: the path to the `root` file in the NFTree directory, or the parent NFT desc file for the collection that contains this NFT",
            file=sys.stderr,
        )
        print(
            "   DATA_HASH: the hash of the NFT data",
            file=sys.stderr,
        )
    else:
        print("\nOptions:", file=sys.stderr)
        print("   -v       Verbose debug output", file=sys.stderr)
        print("\nCommands:", file=sys.stderr)
        print("   build SRC_DIR DEST_DIR", file=sys.stderr)
        print("   verify ROOT_PATH DATA_HASH", file=sys.stderr)
        print(f"\nRun `{progname} COMMAND` for command-specific help", file=sys.stderr)

    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="NFTree command-line tool")
    parser.add_argument("-v", action="store_true", help="Enable verbose debug output")
    subparsers = parser.add_subparsers(dest="command")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build NFTree data")
    build_parser.add_argument("src_dir", help="Source directory containing NFTs")
    build_parser.add_argument("dest_dir", help="Destination directory for NFTree data")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify NFTree proof")
    verify_parser.add_argument(
        "root_path", help="Path to the root file or parent NFT descriptor"
    )
    verify_parser.add_argument("data_hash", help="Hash of the NFT data")

    args = parser.parse_args()

    global verbose
    verbose = args.v

    if args.command == "build":
        src_dir = args.src_dir
        dest_dir = args.dest_dir

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        else:
            error_out(f"The path {dest_dir} already exists; aborting")

        collection = processOneDirectory(src_dir, dest_dir)
        print(json.dumps(collection["nftDesc"]["nftDesc"].hex()))

    elif args.command == "verify":
        root_path = args.root_path
        desc_path = args.data_hash

        # get the last part of the path from data_hash
        data_hash = desc_path.split("/")[-1].split(".")[0]
        base_path = os.path.dirname(desc_path)
        desc_path = f"{base_path}/{data_hash}/{data_hash}.desc"

        print(f"Verifying proof for {desc_path} with root {root_path}")
        # Ensure the root_hash is loaded properly
        with open(root_path, "r") as file:
            root_hash_data = json.loads(file.read())
            root_hash = bytes.fromhex(
                root_hash_data if isinstance(root_hash_data, str) else root_hash_data[0]
            )

        # Load the descriptor and proof
        with open(desc_path, "r") as file:
            desc_data = json.loads(file.read())
            desc = bytes.fromhex(
                desc_data if isinstance(desc_data, str) else desc_data[0]
            )

        # Calculate the path for the proof file from the descriptor path
        proof_path = desc_path.replace(".desc", ".proof")

        with open(proof_path, "r") as file:
            proof = json.loads(file.read())

        # Proceed to hash the descriptor
        hasher = hashlib.sha512()
        hasher.update(desc)
        desc_hash = hasher.digest()[:32]

        # Verify the proof
        if verifyProof(root_hash, desc_hash, proof):
            print("Proof verified")
        else:
            print("Proof verification failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
