"""Load the XDP program that hands frames to an AF_XDP socket, and pin its socket map.

    sudo python3 experimental/xdp_load.py attach IFNAME --pin DIR [--mode drv|skb] [--queues N]
    sudo python3 experimental/xdp_load.py detach IFNAME --pin DIR

The program is five BPF instructions, written here as bytes, so no C compiler is on the path of the
thing that ships. It looks up the frame's receive queue in a map of sockets and redirects the frame
there, or passes it to the kernel's stack if no socket has registered for that queue.

    r2 = ctx->rx_queue_index
    r1 = the map (this one takes two slots)
    r3 = XDP_PASS, what to do if nothing is registered
    call bpf_redirect_map
    exit

This is a separate step from serving, because a serving loop that loaded programs would need the
netlink and verifier error handling a loader has. What the runtime needs from here is one file, the
pinned map, which it opens with the bpf system call. If that is missing, it refuses to start. If the
interface already has a program, that is someone else's, and attaching refuses to replace it.

Run this in the network namespace of the interface. The pin directory must be on a mounted bpffs.
"""
import argparse
import contextlib
import ctypes
import os
import socket
import struct
import sys
from typing import Iterator

BPF_SYSCALL = 321  # x86-64
BPF_MAP_CREATE, BPF_PROG_LOAD, BPF_OBJ_PIN = 0, 5, 6
BPF_MAP_TYPE_XSKMAP = 17
BPF_PROG_TYPE_XDP = 6
BPF_PSEUDO_MAP_FD = 1
HELPER_REDIRECT_MAP = 51
XDP_DROP, XDP_PASS = 1, 2
RX_QUEUE_INDEX = 16  # offsetof(struct xdp_md, rx_queue_index)
ATTR_BYTES = 144

RTM_SETLINK = 19
NLM_F_REQUEST, NLM_F_ACK = 1, 4
NLMSG_ERROR = 2
IFLA_XDP = 43
NLA_F_NESTED = 1 << 15
IFLA_XDP_FD, IFLA_XDP_FLAGS = 1, 3
XDP_FLAGS_UPDATE_IF_NOEXIST, XDP_FLAGS_SKB_MODE, XDP_FLAGS_DRV_MODE = 1, 2, 4
MAP_FILE = "xsks"
_libc = ctypes.CDLL(None, use_errno=True)


class LoadError(Exception):
    pass


@contextlib.contextmanager
def in_namespace(path: str) -> Iterator[None]:
    """Run this thread in a network namespace, and what it starts, without the remount of /sys that
    `ip netns exec` does, which would hide the bpffs a pinned map is on."""
    home = os.open("/proc/thread-self/ns/net", os.O_RDONLY)
    there = os.open(path, os.O_RDONLY)
    try:
        os.setns(there, os.CLONE_NEWNET)
        yield
    finally:
        os.setns(home, os.CLONE_NEWNET)
        os.close(there)
        os.close(home)


def _bpf(command: int, attr: bytes, what: str) -> int:
    buffer = ctypes.create_string_buffer(attr.ljust(ATTR_BYTES, b"\0"), ATTR_BYTES)
    result = _libc.syscall(ctypes.c_long(BPF_SYSCALL), ctypes.c_long(command), buffer, ctypes.c_long(ATTR_BYTES))
    if result < 0:
        number = ctypes.get_errno()
        raise LoadError(f"{what}: {os.strerror(number)}")
    return result


def _instruction(code: int, dst: int = 0, src: int = 0, offset: int = 0, immediate: int = 0) -> bytes:
    return struct.pack("<BBhi", code, dst | src << 4, offset, immediate)


def program_bytes(map_fd: int) -> bytes:
    return (_instruction(0x61, dst=2, src=1, offset=RX_QUEUE_INDEX)            # r2 = *(u32 *)(r1 + 16)
            + _instruction(0x18, dst=1, src=BPF_PSEUDO_MAP_FD, immediate=map_fd)  # r1 = the map
            + _instruction(0x00)                                               # its upper half
            + _instruction(0xb7, dst=3, immediate=XDP_PASS)                    # r3 = XDP_PASS
            + _instruction(0x85, immediate=HELPER_REDIRECT_MAP)                # call bpf_redirect_map
            + _instruction(0x95))                                              # exit


def drop_program_bytes() -> bytes:
    """r0 = XDP_DROP, and exit. experimental/speed.py puts this on the port that frames return to, so that their
    return costs the generator next to nothing, and the card has still counted them."""
    return _instruction(0xb7, dst=0, immediate=XDP_DROP) + _instruction(0x95)


def create_map(queues: int) -> int:
    attr = struct.pack("<IIIII", BPF_MAP_TYPE_XSKMAP, 4, 4, queues, 0)
    return _bpf(BPF_MAP_CREATE, attr, "creating the socket map")


def load_program(code: bytes) -> int:
    instructions = ctypes.create_string_buffer(code, len(code))
    license_text = ctypes.create_string_buffer(b"GPL")
    log = ctypes.create_string_buffer(65536)
    attr = struct.pack("<IIQQIIQ", BPF_PROG_TYPE_XDP, len(code) // 8, ctypes.addressof(instructions),
                       ctypes.addressof(license_text), 1, len(log), ctypes.addressof(log))
    try:
        return _bpf(BPF_PROG_LOAD, attr, "loading the XDP program")
    except LoadError as failure:
        raise LoadError(f"{failure}\n{log.value.decode(errors='replace')}") from failure


def pin(fd: int, path: str) -> None:
    name = ctypes.create_string_buffer(path.encode())
    attr = struct.pack("<QI", ctypes.addressof(name), fd)
    _bpf(BPF_OBJ_PIN, attr, f"pinning {path}")


def _align(length: int) -> int:
    return (length + 3) & ~3


def _attribute(kind: int, payload: bytes) -> bytes:
    length = 4 + len(payload)
    return struct.pack("<HH", length, kind) + payload + bytes(_align(length) - length)


def set_xdp(interface: str, program_fd: int, flags: int) -> None:
    """RTM_SETLINK with IFLA_XDP. A program fd of -1 detaches."""
    index = socket.if_nametoindex(interface)
    nested = _attribute(IFLA_XDP_FD, struct.pack("<i", program_fd)) + _attribute(IFLA_XDP_FLAGS, struct.pack("<I", flags))
    payload = struct.pack("<BxHiII", socket.AF_UNSPEC, 0, index, 0, 0) + _attribute(IFLA_XDP | NLA_F_NESTED, nested)
    message = struct.pack("<IHHII", 16 + len(payload), RTM_SETLINK, NLM_F_REQUEST | NLM_F_ACK, 1, 0) + payload
    with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_ROUTE) as sock:
        sock.send(message)
        reply = sock.recv(4096)
    _length, kind, _flags, _sequence, _pid = struct.unpack_from("<IHHII", reply)
    if kind != NLMSG_ERROR:
        raise LoadError(f"netlink answered with message type {kind}")
    code = struct.unpack_from("<i", reply, 16)[0]
    if code:
        raise LoadError(f"attaching XDP to {interface}: {os.strerror(-code)}")


def _mode_flag(mode: str) -> int:
    return XDP_FLAGS_DRV_MODE if mode == "drv" else XDP_FLAGS_SKB_MODE


def attach(interface: str, directory: str, mode: str, queues: int) -> None:
    """The map is pinned first, so that a program is attached only once a runtime could find its map."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, MAP_FILE)
    map_fd = create_map(queues)
    program_fd = -1
    try:
        pin(map_fd, path)
        program_fd = load_program(program_bytes(map_fd))
        set_xdp(interface, program_fd, _mode_flag(mode) | XDP_FLAGS_UPDATE_IF_NOEXIST)
    except (LoadError, OSError):
        if os.path.exists(path):
            os.unlink(path)
        raise
    finally:
        os.close(map_fd)
        if program_fd >= 0:
            os.close(program_fd)


def attach_drop(interface: str, mode: str) -> None:
    program_fd = load_program(drop_program_bytes())
    try:
        set_xdp(interface, program_fd, _mode_flag(mode) | XDP_FLAGS_UPDATE_IF_NOEXIST)
    finally:
        os.close(program_fd)


def detach_program(interface: str, mode: str) -> None:
    set_xdp(interface, -1, _mode_flag(mode))


def detach(interface: str, directory: str, mode: str) -> None:
    detach_program(interface, mode)
    path = os.path.join(directory, MAP_FILE)
    if os.path.exists(path):
        os.unlink(path)
    if os.path.isdir(directory) and not os.listdir(directory):
        os.rmdir(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("attach", "detach"))
    parser.add_argument("interface")
    parser.add_argument("--pin", required=True, help="a directory on a bpffs, for the socket map")
    parser.add_argument("--mode", choices=("drv", "skb"), default="drv")
    parser.add_argument("--queues", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.action == "attach":
            attach(args.interface, args.pin, args.mode, args.queues)
        else:
            detach(args.interface, args.pin, args.mode)
    except (LoadError, OSError) as failure:
        print(f"xdp_load: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
