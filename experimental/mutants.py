"""The forwarder's mutants: tools/mutate_bounds.py's method, on what moved out of main with them.

Most are caught by tests that put an XDP program on a virtual pair, so this needs root:

    sudo env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 python3 -m experimental.mutants [name of a mutant...]
"""
import os
import sys
import tempfile

from tools.mutate_bounds import run

FORWARDER_TESTS = ["experimental.test_ipv4.TheTwoAgree", "experimental.test_ipv4.TheRings",
                   "experimental.test_ipv4.OtherHandlers", "experimental.test_ipv4.WhatItCosts",
                   "experimental.test_ipv4.RefusalsAtStartup", "experimental.test_ipv4.TheLoader",
                   "experimental.test_ipv4.TheTable", "experimental.test_ipv4.TheBaselinesRoutes",
                   "experimental.test_ipv4.WithoutRoot", "experimental.test_ipv4.Binding",
                   "experimental.test_ipv4.TheProgramsBytes", "experimental.test_ipv4.TheHarness"]

MUTANTS = {
    # --- the AF_XDP runtime
    "xdp: a frame too short to read is admitted": ("experimental/runtime_xdp.py",
        'add(f"    cmp ecx, {item.header_bytes}")\n    add("    jb .malformed")', 'pass', FORWARDER_TESTS),
    "xdp: a full ring is not a refusal": ("experimental/runtime_xdp.py",
        'add("    jae .full")', 'pass', FORWARDER_TESTS),
    "xdp: a refusal loses its frame": ("experimental/runtime_xdp.py",
        'add("    mov [rsi+rcx*8], rax")\n    add("    inc edx")', 'pass', FORWARDER_TESTS),
    "xdp: what was sent does not go back on fill": ("experimental/runtime_xdp.py",
        'add("    mov [r9+rdi*8], rsi")\n    add("    inc ecx")\n    add("    inc r10d")',
        'add("    inc ecx")', FORWARDER_TESTS),
    "xdp: the kernel is not told it has rx entries back": ("experimental/runtime_xdp.py",
        'add("    mov rax, [rx_cons_at]")\n    add("    mov edx, [rx_cons]")\n    add("    mov [rax], edx")',
        'pass', FORWARDER_TESTS),
    "xdp: fill is not published after a batch": ("experimental/runtime_xdp.py",
        'add("    mov rax, [fill_prod_at]")\n    add("    mov edx, [fill_prod]")\n    add("    mov [rax], edx")\n    add("    call flush_tx")',
        'add("    call flush_tx")', FORWARDER_TESTS),
    "xdp: tx is not published": ("experimental/runtime_xdp.py",
        'add("    mov rax, [tx_prod_at]")\n    add("    mov edx, [tx_prod]")\n    add("    mov [rax], edx")\n    add("    mov rax, [tx_flags_at]")',
        'add("    mov edx, [tx_prod]")\n    add("    mov rax, [tx_flags_at]")', FORWARDER_TESTS),
    "xdp: the kernel is never woken": ("experimental/runtime_xdp.py",
        f'add(f"    test dword [rax], {{XDP_RING_NEED_WAKEUP}}")\n    add("    jz .done")', 'add("    jmp .done")', FORWARDER_TESTS),
    "xdp: egress leaves its cells out of the pool": ("experimental/runtime_xdp.py",
        'add(f"    add qword [pool_count], {output.cells}")', 'pass', FORWARDER_TESTS),
    "xdp: egress loses a cell": ("experimental/runtime_xdp.py",
        'add("    mov [pool], rax")\n    add(f"    add qword [pool_count]',
        'add("    mov rax, [rax]")\n    add("    mov [pool], rax")\n    add(f"    add qword [pool_count]', FORWARDER_TESTS),
    "xdp: the output's fields are not written back": ("experimental/runtime_xdp.py",
        "_free_cells(add, ctor, True)", "_free_cells(add, ctor, False)", FORWARDER_TESTS),
    "xdp: a field goes back one byte along": ("experimental/runtime_xdp.py",
        "writes.append(Scalar(name, cell, off, size, source.wire))",
        "writes.append(Scalar(name, cell, off, size, source.wire + 1))", FORWARDER_TESTS),
    "xdp: a frame leaves with the wrong length": ("experimental/runtime_xdp.py",
        'add("    shr rbx, 32")', 'add("    shr rbx, 33")', FORWARDER_TESTS),
    "xdp: the frame's length is not given to the program": ("experimental/runtime_xdp.py",
        'add(f"    mov [rax+{item.length[1]}], cx")', 'pass', FORWARDER_TESTS),
    "xdp: a gap in the cell is copied across": ("experimental/runtime_xdp.py",
        'if runs and runs[-1][0] + runs[-1][2] == scalar.wire and runs[-1][1] + runs[-1][2] == scalar.offset:',
        'if runs and runs[-1][0] + runs[-1][2] == scalar.wire:', FORWARDER_TESTS),
    "xdp: length counts as a wire field": ("experimental/runtime_xdp.py",
        'length = (cell, off)\n            continue', 'length = (cell, off)', FORWARDER_TESTS),
    "xdp: P does not pause": ("experimental/runtime_xdp.py",
        'add("    mov qword [egress_paused], 1")', 'pass', FORWARDER_TESTS),
    "xdp: R does not resume": ("experimental/runtime_xdp.py",
        'add("    cmp al, 82")\n    add("    jne .answer")\n    add("    mov qword [egress_paused], 0")',
        'pass', FORWARDER_TESTS),
    "xdp: Q leaves what is queued": ("experimental/runtime_xdp.py",
        'add("    mov qword [egress_paused], 0")\n    add(f"    mov qword [patience], {QUIT_PATIENCE}")',
        'add("    jmp .leave")\n    add(f"    mov qword [patience], {QUIT_PATIENCE}")', FORWARDER_TESTS),
    "xdp: a bad port is sent anyway": ("experimental/runtime_xdp.py",
        'add(f"    jne .bad_port_{position}")', 'pass', FORWARDER_TESTS),
    "xdp: the pool is not counted when asked": ("experimental/runtime_xdp.py",
        'add("    mov [cnt_pool], rcx")', 'pass', FORWARDER_TESTS),
    "xdp: what the kernel dropped is not asked for": ("experimental/runtime_xdp.py",
        'add("    mov [cnt_kernel_dropped], rax")', 'pass', FORWARDER_TESTS),
    "xdp: the reserve is not handed to the program": ("experimental/runtime_xdp.py",
        'add(f"    mov {PARAMS[k + 1]}, r13")', 'add(f"    xor {PARAMS[k + 1]}, {PARAMS[k + 1]}")', FORWARDER_TESTS),
    "xdp: the image is sized without the reserve": ("experimental/runtime_xdp.py",
        "return self.item.cells + self.reserve\n", "return self.item.cells\n", FORWARDER_TESTS),
    "xdp: what is left of the reserve does not come back": ("experimental/runtime_xdp.py",
        'add("    test r13, r13")\n    add("    jz .given_back")', 'add("    jmp .given_back")', FORWARDER_TESTS),
    "xdp: the state is not handed to the program": ("experimental/runtime_xdp.py",
        'add(f"    mov {PARAMS[position]}, [state_{position}]")', 'pass', FORWARDER_TESTS),
    "xdp: an image of any size is built": ("experimental/runtime_xdp.py",
        "if image.bytes > MAX_IMAGE_BYTES:", "if False:", FORWARDER_TESTS),
    "xdp: a constructor without a frame is bound": ("experimental/runtime_xdp.py",
        'if frame is None:\n        raise BindingError', 'if False:\n        raise BindingError', FORWARDER_TESTS),
    "xdp: an output may write what the item lacks": ("experimental/runtime_xdp.py",
        'if source is None:', 'if source is None:\n            continue\n        if False:', FORWARDER_TESTS),
    "xdp: a missing map is not a refusal": ("experimental/runtime_xdp.py",
        'syscall(321, "js .refuse_map", str(BPF_OBJ_GET), "obj_attr", "24")\n'
        '    add("    mov [update_attr], eax")\n'
        '    syscall(321, "jnz .refuse_map", str(BPF_MAP_UPDATE_ELEM), "update_attr", "32")',
        'syscall(321, "", str(BPF_OBJ_GET), "obj_attr", "24")\n'
        '    add("    mov [update_attr], eax")\n'
        '    syscall(321, "", str(BPF_MAP_UPDATE_ELEM), "update_attr", "32")', FORWARDER_TESTS),
    "xdp: the socket is not put in the map": ("experimental/runtime_xdp.py",
        'syscall(321, "jnz .refuse_map", str(BPF_MAP_UPDATE_ELEM), "update_attr", "32")', 'pass', FORWARDER_TESTS),
    "xdp: the packet memory is outside what startup touches": ("experimental/runtime_xdp.py",
        'add(f"umem: resb {frames * CHUNK}")\n    add(f"bss_end_page: resb {PAGE}")\n    add("bss_end:")',
        'add(f"bss_end_page: resb {PAGE}")\n    add("bss_end:")\n    add(f"umem: resb {frames * CHUNK}")', FORWARDER_TESTS),
    "xdp: a frame the kernel dropped is not counted": ("experimental/runtime_xdp.py",
        'for number in (EAGAIN, ENOBUFS):', 'for number in (EAGAIN, ENOBUFS, 16):', FORWARDER_TESTS),
    # --- the harness: only what can be shown without a card
    "speed: a window where the forwarder heard nothing is a result": ("experimental/speed.py",
        'if after["frames"] == before["frames"]:', 'if False:', FORWARDER_TESTS),
    "speed: a virtual port can give a rate": ("experimental/speed.py",
        'if not args.plumbing:\n        for port in (args.generator, args.forwarder):\n            if not has_a_device(port):',
        'if False:\n        for port in (args.generator, args.forwarder):\n            if not has_a_device(port):',
        FORWARDER_TESTS),
    "speed: a port with no carrier is not waited for": ("experimental/speed.py",
        'while "LOWER_UP" not in link(port, namespace)["flags"]:', 'while False:', FORWARDER_TESTS),
    "speed: a signal does not put things back": ("experimental/speed.py",
        "signal.signal(number, _leave)", "pass", FORWARDER_TESTS),
    "speed: a program is left on the generator's port": ("experimental/speed.py",
        "xdp_load.detach_program(args.generator, self.mode)", "pass", FORWARDER_TESTS),
    # --- the loader
    "xdp: a program already there is replaced": ("experimental/xdp_load.py",
        "set_xdp(interface, program_fd, _mode_flag(mode) | XDP_FLAGS_UPDATE_IF_NOEXIST)\n    except (LoadError, OSError):",
        "set_xdp(interface, program_fd, _mode_flag(mode))\n    except (LoadError, OSError):", FORWARDER_TESTS),
    "xdp: a failed attach leaves its map for a runtime to find": ("experimental/xdp_load.py",
        "if os.path.exists(path):\n            os.unlink(path)\n        raise", "raise", FORWARDER_TESTS),
    "xdp: returning frames are passed on": ("experimental/xdp_load.py",
        "_instruction(0xb7, dst=0, immediate=XDP_DROP)", "_instruction(0xb7, dst=0, immediate=XDP_PASS)", FORWARDER_TESTS),
    "xdp: what no socket wants is dropped": ("experimental/xdp_load.py",
        "XDP_DROP, XDP_PASS = 1, 2", "XDP_DROP, XDP_PASS = 1, 1", FORWARDER_TESTS),
    "xdp: the program reads the wrong field for the queue": ("experimental/xdp_load.py",
        "RX_QUEUE_INDEX = 16", "RX_QUEUE_INDEX = 12", FORWARDER_TESTS),
    # --- the handler
    "ipv4: a TTL of 1 is forwarded": ("experimental/ipv4.lfpl", "if <=(ttl, 1) then", "if <(ttl, 1) then", FORWARDER_TESTS),
    "ipv4: a TTL of 2 is refused": ("experimental/ipv4.lfpl", "if <=(ttl, 1) then", "if <=(ttl, 2) then", FORWARDER_TESTS),
    "ipv4: the ethertype is not checked": ("experimental/ipv4.lfpl",
        "if !=(ethertype, 8) then false            # 08 00 on the wire\n  else if", "if", FORWARDER_TESTS),
    "ipv4: the ethertype is read the other way round": ("experimental/ipv4.lfpl",
        "if !=(ethertype, 8) then", "if !=(ethertype, 2048) then", FORWARDER_TESTS),
    "ipv4: only the version is checked": ("experimental/ipv4.lfpl",
        "else if !=(version, 69) then false", "else if !=(&(version, 240), 64) then false", FORWARDER_TESTS),
    "ipv4: only the header length is checked": ("experimental/ipv4.lfpl",
        "else if !=(version, 69) then false", "else if !=(&(version, 15), 5) then false", FORWARDER_TESTS),
    "ipv4: a total of 19 passes": ("experimental/ipv4.lfpl", "else if <(total, 20) then false", "else if <(total, 19) then false", FORWARDER_TESTS),
    "ipv4: a total of 20 is refused": ("experimental/ipv4.lfpl", "else if <(total, 20) then false", "else if <=(total, 20) then false", FORWARDER_TESTS),
    "ipv4: a total past the frame passes": ("experimental/ipv4.lfpl",
        "else if >(+(total, 14), pack<16>(length)) then false", "else if >(+(total, 13), pack<16>(length)) then false", FORWARDER_TESTS),
    "ipv4: a total of exactly the frame is refused": ("experimental/ipv4.lfpl",
        "else if >(+(total, 14), pack<16>(length)) then false", "else if >=(+(total, 14), pack<16>(length)) then false", FORWARDER_TESTS),
    "ipv4: the total is read with its bytes swapped": ("experimental/ipv4.lfpl",
        "pack<8,8>(low8(total), low8(>>(total, 8)))", "pack<8,8>(low8(>>(total, 8)), low8(total))", FORWARDER_TESTS),
    "ipv4: the checksum is not checked": ("experimental/ipv4.lfpl",
        "else ==(fold(sum), 65535);", "else true;", FORWARDER_TESTS),
    "ipv4: the check does not fold": ("experimental/ipv4.lfpl",
        "else ==(fold(sum), 65535);", "else ==(sum, 65535);", FORWARDER_TESTS),
    "ipv4: a word is left out of the sum": ("experimental/ipv4.lfpl",
        "+(pack<16>(total), +(pack<16>(id),", "+(pack<16>(total), +(0,", FORWARDER_TESTS),
    "ipv4: the TTL goes down by 2": ("experimental/ipv4.lfpl", "-(ttl, 1), 2,", "-(ttl, 2), 2,", FORWARDER_TESTS),
    "ipv4: the checksum is not patched": ("experimental/ipv4.lfpl",
        "Tx2@d3(fresh_sum, frame)", "Tx2@d3(sum, frame)", FORWARDER_TESTS),
    "ipv4: the update leaves out the new word": ("experimental/ipv4.lfpl",
        "+(^(m, 65535), -(m, 1))", "+(^(m, 65535), m)", FORWARDER_TESTS),
    "ipv4: the update does not fold": ("experimental/ipv4.lfpl",
        "low16(^(fold(+(^(pack<16>(sum), 65535), +(^(m, 65535), -(m, 1)))), 65535))",
        "low16(^(+(^(pack<16>(sum), 65535), +(^(m, 65535), -(m, 1))), 65535))", FORWARDER_TESTS),
    "ipv4: the update is RFC 1141's": ("experimental/ipv4.lfpl",
        "low16(^(fold(+(^(pack<16>(sum), 65535), +(^(m, 65535), -(m, 1)))), 65535))",
        "low16(fold(+(pack<16>(sum), 1)))", FORWARDER_TESTS),
    "ipv4: the boundary at 64 is off by one": ("experimental/ipv4.lfpl", "if <(first, 64) then 0", "if <=(first, 64) then 0", FORWARDER_TESTS),
    "ipv4: the boundary at 128 is off by one": ("experimental/ipv4.lfpl", "if <(first, 128) then 1", "if <=(first, 128) then 1", FORWARDER_TESTS),
    "ipv4: the boundary at 192 is off by one": ("experimental/ipv4.lfpl", "if <(first, 192) then 2", "if <=(first, 192) then 2", FORWARDER_TESTS),
    "ipv4: the hop comes from the wrong byte": ("experimental/ipv4.lfpl",
        "next_hop(low8(dst_hi))", "next_hop(low8(>>(dst_hi, 8)))", FORWARDER_TESTS),
    "ipv4: the destination MAC is one off": ("experimental/ipv4.lfpl", "low16(<<(mac_end, 8))", "low16(<<(+(mac_end, 1), 8))", FORWARDER_TESTS),
    "ipv4: the source MAC loses its 1": ("experimental/ipv4.lfpl", "+(<<(mac_end, 24), 65536)", "<<(mac_end, 24)", FORWARDER_TESTS),
    "ipv4: an expired frame is called malformed": ("experimental/ipv4.lfpl",
        "(Expired@d1(frame), More@d2", "(Malformed@d1(frame), More@d2", FORWARDER_TESTS),
    # --- the baseline, which is an oracle only if it is right. There is no mutant for its check that a
    # frame has 34 bytes: a shorter one fails the check of its total length anyway, whatever is in it.
    "c: a TTL of 1 is forwarded": ("experimental/ipv4_forward.c", "if (ttl <= 1)", "if (ttl < 1)", FORWARDER_TESTS),
    "c: the ethertype is not checked": ("experimental/ipv4_forward.c",
        "    if (read16(frame + 12) != ETHERTYPE_IPV4_LE)\n        return MALFORMED;\n", "", FORWARDER_TESTS),
    "c: the version byte is not checked": ("experimental/ipv4_forward.c",
        "    if (ip[0] != VERSION_AND_LENGTH)\n        return MALFORMED;\n", "", FORWARDER_TESTS),
    "c: a total past the frame passes": ("experimental/ipv4_forward.c", "total > length - 14", "total > length - 13", FORWARDER_TESTS),
    # (void) keeps each of these three compiling, because -Werror refusing it would say nothing.
    "c: the checksum is not checked": ("experimental/ipv4_forward.c",
        "    if (fold(sum) != 0xFFFF)\n        return MALFORMED;\n", "    (void)sum;\n", FORWARDER_TESTS),
    "c: the checksum is not patched": ("experimental/ipv4_forward.c",
        "    memcpy(frame + 24, &checksum, 2);\n", "    (void)checksum;\n", FORWARDER_TESTS),
    "c: the MACs are not rewritten": ("experimental/ipv4_forward.c", "    memcpy(frame, hop_macs[hop], 12);\n", "", FORWARDER_TESTS),
    "c: routes are not sorted": ("experimental/ipv4_forward.c",
        "    qsort(routes, used, sizeof routes[0], compare_length);\n", "    (void)compare_length;\n", FORWARDER_TESTS),
    "c: a new second level starts empty": ("experimental/ipv4_forward.c",
        "        for (uint32_t i = 0; i < TBL8_GROUP; i++)\n            tbl8[group * TBL8_GROUP + i] = tbl24[top];\n", "", FORWARDER_TESTS),
    "c: the second level is not looked at": ("experimental/ipv4_forward.c",
        "    if (entry & TBL8_FLAG)\n        entry = tbl8[(entry & ~TBL8_FLAG) * TBL8_GROUP + destination[3]];\n",
        "    if (entry & TBL8_FLAG)\n        entry = 0;\n", FORWARDER_TESTS),
    "c: a /24 spills over": ("experimental/ipv4_forward.c",
        "uint32_t cells = 1u << (24 - length);", "uint32_t cells = (1u << (24 - length)) + 1;", FORWARDER_TESTS),
    "c: no route is not refused": ("experimental/ipv4_forward.c",
        "    if (hop == NO_ROUTE)\n        return MALFORMED;\n", "    if (hop == NO_ROUTE)\n        hop = 0;\n", FORWARDER_TESTS),
    "c: match is off at 128": ("experimental/ipv4_forward.c", "first < 128 ? 1", "first <= 128 ? 1", FORWARDER_TESTS),
    "c: a line of the routes that does not parse is skipped": ("experimental/ipv4_forward.c",
        'fail("a line of the routes file does not parse");', 'continue;', FORWARDER_TESTS),
    "c: octets are not checked": ("experimental/ipv4_forward.c",
        "a > 255 || b > 255 || c > 255 || d > 255 || ", "", FORWARDER_TESTS),
}


def main() -> int:
    if os.geteuid() != 0:
        print("these need root, for a network namespace and an XDP program", file=sys.stderr)
        return 2
    survivors = 0
    wanted = sys.argv[1:] or list(MUTANTS)
    with tempfile.TemporaryDirectory() as work:
        for name in wanted:
            verdict, survived = run(work, *MUTANTS[name])
            survivors += survived
            print(f"{name}: {verdict}", flush=True)
    print(f"{len(wanted) - survivors} of {len(wanted)} caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
