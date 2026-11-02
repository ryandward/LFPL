"""The fixed workload, as a Python model that shares nothing with what it judges.

experimental/workload.md says what the workload is. This is another opinion on it: examples/ipv4.lfpl is judged by
it on UDP sockets, and the C baseline in experimental/ipv4_forward.c and experimental/ipv4.lfpl are compared with each
other on a virtual pair, and both with this, on every frame. It works in network byte order, as the RFCs write it.
"""
from typing import Callable, Optional

ETHERNET_HEADER = 14
IPV4_HEADER = 20
HEADER_BYTES = ETHERNET_HEADER + IPV4_HEADER
ETHERTYPE_IPV4 = 0x0800
VERSION_AND_LENGTH = 0x45  # version 4, 5 words of header
TTL_AT, CHECKSUM_AT, DESTINATION_AT = ETHERNET_HEADER + 8, ETHERNET_HEADER + 10, ETHERNET_HEADER + 16
MATCH_HOPS = 4

FORWARD, MALFORMED, EXPIRED = "forward", "malformed", "expired"


def macs_of(hop: int) -> tuple[bytes, bytes]:
    """The destination and source addresses of a next hop. Locally administered, and easy to read."""
    return (bytes((0x02, 0, 0, 0, 0, 0x10 + hop)), bytes((0x02, 0, 0, 0, 1, 0x10 + hop)))


def by_top_bits(destination: bytes) -> int:
    """The next hop examples/ipv4.lfpl and experimental/ipv4.lfpl choose: the top two bits of the first octet."""
    return destination[0] >> 6


def one_complement_sum(data: bytes) -> int:
    total = sum(int.from_bytes(data[at:at + 2], "big") for at in range(0, len(data), 2))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return total


def incremental_checksum(old: int, old_word: int, new_word: int) -> int:
    """RFC 1624, equation 3: HC' = ~(~HC + ~m + m')."""
    total = (~old & 0xFFFF) + (~old_word & 0xFFFF) + new_word
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def forward(frame: bytes, next_hop: Callable[[bytes], Optional[int]]) -> tuple[str, Optional[bytes]]:
    """What the workload does with a frame: a verdict, and the frame to send if there is one."""
    if len(frame) < HEADER_BYTES:
        return MALFORMED, None
    if int.from_bytes(frame[12:14], "big") != ETHERTYPE_IPV4:
        return MALFORMED, None
    if frame[ETHERNET_HEADER] != VERSION_AND_LENGTH:
        return MALFORMED, None
    total_length = int.from_bytes(frame[ETHERNET_HEADER + 2:ETHERNET_HEADER + 4], "big")
    if total_length < IPV4_HEADER or total_length > len(frame) - ETHERNET_HEADER:
        return MALFORMED, None
    if one_complement_sum(frame[ETHERNET_HEADER:HEADER_BYTES]) != 0xFFFF:
        return MALFORMED, None
    ttl = frame[TTL_AT]
    if ttl <= 1:
        return EXPIRED, None
    hop = next_hop(frame[DESTINATION_AT:DESTINATION_AT + 4])
    if hop is None:
        return MALFORMED, None
    protocol = frame[TTL_AT + 1]
    checksum = int.from_bytes(frame[CHECKSUM_AT:CHECKSUM_AT + 2], "big")
    patched = incremental_checksum(checksum, ttl << 8 | protocol, (ttl - 1) << 8 | protocol)
    destination_mac, source_mac = macs_of(hop)
    return FORWARD, (destination_mac + source_mac + frame[12:TTL_AT] + bytes((ttl - 1, protocol))
                     + patched.to_bytes(2, "big") + frame[CHECKSUM_AT + 2:])


def checksum_of(header: bytes) -> int:
    """The checksum a header should carry, from the bytes with that field zero."""
    zeroed = header[:10] + bytes(2) + header[12:]
    return ~one_complement_sum(zeroed) & 0xFFFF


def ipv4_frame(destination: bytes = bytes((10, 0, 0, 1)), source: bytes = bytes((192, 168, 0, 1)), *,
               ttl: int = 64, protocol: int = 17, identification: int = 0x1234, payload: bytes = b"",
               total_length: Optional[int] = None, version_and_length: int = VERSION_AND_LENGTH,
               ethertype: int = ETHERTYPE_IPV4, checksum: Optional[int] = None, tos: int = 0,
               flags_and_offset: int = 0x4000, padding: bytes = b"") -> bytes:
    """A frame, valid unless told otherwise. `padding` follows the IP packet, as Ethernet allows."""
    length = IPV4_HEADER + len(payload) if total_length is None else total_length
    header = (bytes((version_and_length, tos)) + length.to_bytes(2, "big") + identification.to_bytes(2, "big")
              + flags_and_offset.to_bytes(2, "big") + bytes((ttl, protocol)) + bytes(2) + source + destination)
    value = checksum_of(header) if checksum is None else checksum
    header = header[:10] + value.to_bytes(2, "big") + header[12:]
    ethernet = bytes.fromhex("aabbccddeeff" "112233445566") + ethertype.to_bytes(2, "big")
    return ethernet + header + payload + padding


def with_checksum(target: int, *, ttl: int = 64, destination: bytes = bytes((10, 0, 0, 1)),
                  payload: bytes = b"") -> bytes:
    """A valid frame whose header checksum is `target`, by choosing the identification field."""
    base = ipv4_frame(destination, ttl=ttl, identification=0, payload=payload)
    others = one_complement_sum(base[ETHERNET_HEADER:ETHERNET_HEADER + 10] + bytes(2) + base[CHECKSUM_AT + 2:HEADER_BYTES])
    wanted = ~target & 0xFFFF
    identification = (wanted - others) % 0xFFFF
    frame = ipv4_frame(destination, ttl=ttl, identification=identification, payload=payload, checksum=target)
    assert one_complement_sum(frame[ETHERNET_HEADER:HEADER_BYTES]) == 0xFFFF, hex(target)
    return frame


def longest_prefix(routes: list) -> Callable[[bytes], Optional[int]]:
    """The answer a DIR-24-8 table has to give, worked out the slow way. (address, length, hop)."""
    parsed = [(int.from_bytes(address, "big"), length, hop) for address, length, hop in routes]

    def look_up(destination: bytes) -> Optional[int]:
        value = int.from_bytes(destination, "big")
        best = None
        for address, length, hop in parsed:
            mask = ~((1 << (32 - length)) - 1) & 0xFFFFFFFF
            if value & mask == address & mask and (best is None or length >= best[0]):
                best = (length, hop)
        return None if best is None else best[1]

    return look_up


def corpus() -> list:
    """Every kind of frame the workload has to tell apart: (name, verdict, frame). The fast tier checks
    these verdicts against the code above, so that a case meant to be refused cannot quietly be valid."""
    import random

    cases = []

    def case(name: str, verdict: str, frame: bytes) -> None:
        cases.append((name, verdict, frame))

    # --- valid frames
    for first in (0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255):
        case(f"to {first}.2.3.4", FORWARD, ipv4_frame(bytes((first, 2, 3, 4))))
    for size in (0, 1, 2, 7, 26, 27, 64, 1000, 1480):
        case(f"{size} bytes of payload", FORWARD, ipv4_frame(payload=(bytes(range(256)) * 6)[:size]))
    case("the shortest there is: no payload, 34 bytes", FORWARD, ipv4_frame())
    case("padded after the packet, as Ethernet does", FORWARD, ipv4_frame(padding=bytes(26)))
    case("padded with what is not zero", FORWARD, ipv4_frame(payload=b"abc", padding=b"\xff" * 23))
    case("a fragment", FORWARD, ipv4_frame(flags_and_offset=0x2000 | 185))
    case("with a type of service", FORWARD, ipv4_frame(tos=0xB8))
    case("protocol 0", FORWARD, ipv4_frame(protocol=0))
    case("protocol 255", FORWARD, ipv4_frame(protocol=255))
    case("total length that needs its bytes swapped", FORWARD, ipv4_frame(payload=bytes(300)))
    case("exactly the length it says", FORWARD, ipv4_frame(payload=bytes(1)))
    case("source and destination all ones", FORWARD, ipv4_frame(b"\xff" * 4, b"\xff" * 4))

    # --- too short to hold the headers
    whole = ipv4_frame(payload=b"payload!")
    for length in (14, 15, 20, 33):
        case(f"{length} bytes", MALFORMED, whole[:length])

    # --- not IPv4 with a 5-word header
    for ethertype, name in ((0x0806, "ARP"), (0x86DD, "IPv6"), (0x8100, "a VLAN tag"), (0x88A8, "two VLAN tags"),
                            (0x0000, "ethertype 0"), (0x0008, "the IPv4 ethertype, bytes swapped"),
                            (0xFFFF, "ethertype all ones"), (0x0801, "one bit away"), (0x0900, "one bit away")):
        case(name, MALFORMED, ipv4_frame(ethertype=ethertype))
    for byte in (0x65, 0x05, 0xF5, 0x35, 0x44, 0x46, 0x4F, 0x40, 0x41, 0x54, 0x00, 0xFF):
        case(f"version and header length {byte:#x}", MALFORMED, ipv4_frame(version_and_length=byte))

    # --- a total length that does not fit, all with good checksums
    for total, name in ((0, "0"), (1, "1"), (19, "19"), (0x1400, "20, bytes swapped, in a short frame")):
        case(f"total length {name}", MALFORMED, ipv4_frame(total_length=total))
    case("total length 1 past the frame", MALFORMED, ipv4_frame(payload=bytes(10), total_length=31))
    case("total length 65535", MALFORMED, ipv4_frame(total_length=65535))

    # --- a bad checksum: every byte of the header, one bit off
    good = ipv4_frame(bytes((10, 9, 8, 7)), payload=bytes(40))
    for position in range(ETHERNET_HEADER, HEADER_BYTES):
        for bit in (0x01, 0x80):
            bad = bytearray(good)
            bad[position] ^= bit
            verdict = MALFORMED
            case(f"byte {position} of the frame, bit {bit:#x}", verdict, bytes(bad))
    case("a checksum of 0 where it is not", MALFORMED, ipv4_frame(checksum=0))
    case("a checksum with its bytes swapped", MALFORMED, ipv4_frame(checksum=int.from_bytes(
        ipv4_frame()[CHECKSUM_AT:CHECKSUM_AT + 2], "little")))

    # --- the TTL
    for ttl, verdict in ((0, EXPIRED), (1, EXPIRED), (2, FORWARD), (3, FORWARD), (127, FORWARD), (128, FORWARD),
                         (129, FORWARD), (255, FORWARD)):
        case(f"TTL {ttl}", verdict, ipv4_frame(ttl=ttl))
    case("TTL 1 and a bad checksum: the checksum comes first", MALFORMED, ipv4_frame(ttl=1, checksum=0x1234))
    case("TTL 0 and a bad total length: the length comes first", MALFORMED, ipv4_frame(ttl=0, total_length=3))

    # --- a header checksum near 0xFFFF, where RFC 1141's update and a naive fold go wrong
    for target in (0xFFFF, 0xFFFE, 0xFFFD, 0xFEFF, 0xFF00, 0xFF01, 0x00FF, 0x0100, 0x0000, 0x0001, 0x0002):
        for ttl in (2, 3, 64, 255):
            case(f"checksum {target:#06x}, TTL {ttl}", FORWARD, with_checksum(target, ttl=ttl))
    case("checksum 0xFFFF at TTL 1", EXPIRED, with_checksum(0xFFFF, ttl=1))

    # --- and whatever chance brings, from a fixed seed: valid headers, and noise
    rng = random.Random(0x4950)
    for number in range(150):
        frame = ipv4_frame(rng.randbytes(4), rng.randbytes(4), ttl=rng.randrange(256), protocol=rng.randrange(256),
                           identification=rng.randrange(1 << 16), flags_and_offset=rng.randrange(1 << 16),
                           tos=rng.randrange(256), payload=rng.randbytes(rng.randrange(1400)),
                           padding=rng.randbytes(rng.randrange(30)))
        case(f"random valid header {number}", forward(frame, by_top_bits)[0], frame)
    for number in range(100):
        frame = rng.randbytes(rng.randrange(14, 200))
        case(f"noise {number}", forward(frame, by_top_bits)[0], frame)
    for number in range(100):
        # noise in the header, which keeps the checksum good: this reaches the checks past it
        frame = bytearray(ipv4_frame(rng.randbytes(4), payload=rng.randbytes(rng.randrange(100))))
        frame[ETHERNET_HEADER + 2:ETHERNET_HEADER + 4] = rng.randrange(1 << 16).to_bytes(2, "big")
        frame[ETHERNET_HEADER + 8] = rng.choice((0, 1, 2, 3, rng.randrange(256)))
        frame[CHECKSUM_AT:CHECKSUM_AT + 2] = checksum_of(bytes(frame[ETHERNET_HEADER:HEADER_BYTES])).to_bytes(2, "big")
        frame = bytes(frame)
        case(f"random length and TTL {number}", forward(frame, by_top_bits)[0], frame)
    return cases
