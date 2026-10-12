import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


def entropy(data):
    if not data:
        return 0.0
    return round(-sum((count / len(data)) * math.log2(count / len(data))
                      for count in Counter(data).values()), 4)


def inspect_artifact(path):
    data = path.read_bytes()
    if data[:2] != b"MZ" or len(data) < 64:
        raise ValueError(f"Not a Windows PE file: {path}")
    pe_offset = struct.unpack_from("<I", data, 60)[0]
    if data[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise ValueError(f"Missing PE signature: {path}")
    machine, count, _, _, _, optional_size, _ = struct.unpack_from("<HHIIIHH", data, pe_offset + 4)
    optional_offset = pe_offset + 24
    magic = struct.unpack_from("<H", data, optional_offset)[0]
    if magic not in (0x10b, 0x20b):
        raise ValueError(f"Unsupported PE optional header: {path}")
    directory_offset = optional_offset + (112 if magic == 0x20b else 96)
    section_offset = optional_offset + optional_size
    sections = []
    for index in range(count):
        header = section_offset + index * 40
        name = data[header:header + 8].rstrip(b"\0").decode("ascii", errors="replace")
        size, offset = struct.unpack_from("<II", data, header + 16)
        flags = struct.unpack_from("<I", data, header + 36)[0]
        sections.append({"name": name, "size": size, "entropy": entropy(data[offset:offset + size]),
                         "executable": bool(flags & 0x20000000)})
    resource_address, resource_size = struct.unpack_from("<II", data, directory_offset + 16)
    signature_offset, signature_size = struct.unpack_from("<II", data, directory_offset + 32)
    return {"file": str(path.resolve()), "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "entropy": entropy(data), "machine": hex(machine), "sections": sections,
            "resources_present": bool(resource_address and resource_size),
            "signature_directory_present": bool(signature_offset and signature_size)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executables", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps([inspect_artifact(path) for path in args.executables], indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
