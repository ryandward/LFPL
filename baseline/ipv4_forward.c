/* The C baseline: docs/workload.md on AF_XDP, on one core.
 *
 * This is the ceiling LFPL-O is measured against, and the oracle for its outputs. It is plain and
 * direct on purpose: it reads the frame where it lies, batches like any AF_XDP program, and has no
 * cells, no admission table and no linear types. Anything LFPL-O spends beyond this is the cost of
 * what LFPL-O promises, and the speed harness says how much that is.
 *
 *     ipv4_forward --ifname IF --map PINNED --ctl-port PORT [--routes FILE | --match]
 *
 * --routes gives a DIR-24-8 table, built at startup: lines of `a.b.c.d/len hop`. --match replaces the
 * table with what examples/ipv4.lfpl does, the top two bits of the destination's first octet, which
 * is how the two are compared frame by frame.
 *
 * tools/xdp_load.py attaches the XDP program and pins the socket map this registers with. The control
 * port takes `S`, answered with the counters below as 64-bit words, and `Q`, which stops it.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <linux/bpf.h>
#include <linux/if_xdp.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef SOL_XDP
#define SOL_XDP 283
#endif

/* The same shape as src/runtime_xdp.py's defaults, so that the two are compared like for like. */
#define FRAME_SIZE 2048u
#define FRAMES 4096u
#define RING_SIZE FRAMES /* fill, rx and completion: a frame is on one ring at a time */
#define RING_MASK (RING_SIZE - 1)
#define TX_SIZE 2048u
#define TX_MASK (TX_SIZE - 1)
#define BATCH 64u
#define HEADER_BYTES 34u
#define ETHERTYPE_IPV4_LE 0x0008u /* 08 00 on the wire */
#define VERSION_AND_LENGTH 0x45u
#define TBL8_GROUP 256u
#define TBL8_GROUPS 4096u
#define TBL8_FLAG 0x8000u
#define NO_ROUTE 0x7FFFu
#define MATCH_HOPS 4u
#define MAX_HOPS 64u

enum verdict { FORWARD, MALFORMED, EXPIRED };

struct counters {
    uint64_t frames, forwarded, malformed, expired, full, completed, kernel_dropped, zerocopy, send_errors;
};

struct ring {
    uint32_t *producer, *consumer, *flags;
    void *descriptors;
    uint32_t cached_producer, cached_consumer; /* ours, or the kernel's as last seen */
};

static struct counters count;
static uint8_t *umem;
static uint8_t hop_macs[MAX_HOPS][12];
static uint16_t *tbl24, *tbl8;
static uint32_t tbl8_used;
static int use_match;

static void die(const char *what) {
    fprintf(stderr, "ipv4_forward: %s: %s\n", what, strerror(errno));
    exit(1);
}

static void fail(const char *what) {
    fprintf(stderr, "ipv4_forward: %s\n", what);
    exit(1);
}

/* --- the next hop */

static void add_route(uint32_t prefix, unsigned length, unsigned hop) {
    if (length <= 24) {
        uint32_t first = length ? prefix >> 8 & ~((1u << (24 - length)) - 1) : 0;
        uint32_t cells = 1u << (24 - length);
        for (uint32_t i = 0; i < cells; i++)
            tbl24[first + i] = (uint16_t)hop;
        return;
    }
    uint32_t top = prefix >> 8;
    uint32_t group;
    if (tbl24[top] & TBL8_FLAG) {
        group = tbl24[top] & ~TBL8_FLAG;
    } else {
        if (tbl8_used == TBL8_GROUPS)
            fail("too many /25 to /32 groups");
        group = tbl8_used++;
        for (uint32_t i = 0; i < TBL8_GROUP; i++)
            tbl8[group * TBL8_GROUP + i] = tbl24[top];
        tbl24[top] = (uint16_t)(TBL8_FLAG | group);
    }
    uint32_t cells = 1u << (32 - length);
    uint32_t first = prefix & 0xFF & ~(cells - 1);
    for (uint32_t i = 0; i < cells; i++)
        tbl8[group * TBL8_GROUP + first + i] = (uint16_t)hop;
}

static int compare_length(const void *a, const void *b) {
    return (int)(((const uint64_t *)a)[1] - ((const uint64_t *)b)[1]);
}

/* Shorter prefixes first, so that a longer one overwrites what it covers. */
static void load_routes(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file)
        die(path);
    static uint64_t routes[65536][3];
    size_t used = 0;
    unsigned a, b, c, d, length, hop;
    char line[128];
    while (fgets(line, sizeof line, file)) {
        int end = 0;
        /* Refused and not skipped: a table missing routes would call good frames malformed. */
        if (sscanf(line, "%u.%u.%u.%u/%u %u %n", &a, &b, &c, &d, &length, &hop, &end) != 6 || line[end] != '\0')
            fail("a line of the routes file does not parse");
        if (used == 65536 || a > 255 || b > 255 || c > 255 || d > 255 || length > 32 || hop >= MAX_HOPS)
            fail("a route is out of range, or there are too many");
        routes[used][0] = a << 24 | b << 16 | c << 8 | d;
        routes[used][1] = length;
        routes[used][2] = hop;
        used++;
    }
    fclose(file);
    qsort(routes, used, sizeof routes[0], compare_length);
    for (size_t i = 0; i < used; i++)
        add_route((uint32_t)routes[i][0], (unsigned)routes[i][1], (unsigned)routes[i][2]);
}

static void make_table(void) {
    tbl24 = mmap(NULL, sizeof(uint16_t) << 24, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    tbl8 = mmap(NULL, sizeof(uint16_t) * TBL8_GROUP * TBL8_GROUPS, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (tbl24 == MAP_FAILED || tbl8 == MAP_FAILED)
        die("mapping the table");
    madvise(tbl24, sizeof(uint16_t) << 24, MADV_HUGEPAGE); /* a hint: without it, more TLB misses */
    for (uint32_t i = 0; i < 1u << 24; i++)
        tbl24[i] = NO_ROUTE;
}

static inline uint32_t next_hop(const uint8_t *destination) {
    if (use_match) {
        uint8_t first = destination[0];
        return first < 64 ? 0 : first < 128 ? 1 : first < 192 ? 2 : 3;
    }
    uint32_t top = (uint32_t)destination[0] << 16 | (uint32_t)destination[1] << 8 | destination[2];
    uint32_t entry = tbl24[top];
    if (entry & TBL8_FLAG)
        entry = tbl8[(entry & ~TBL8_FLAG) * TBL8_GROUP + destination[3]];
    return entry;
}

/* --- the workload. Words are read little-endian: one's complement sums do not care. */

static inline uint16_t read16(const uint8_t *at) {
    uint16_t value;
    memcpy(&value, at, 2);
    return value;
}

static inline uint32_t fold(uint32_t sum) {
    return (sum & 0xFFFF) + (sum >> 16);
}

static inline enum verdict forward(uint8_t *frame, uint32_t length) {
    if (length < HEADER_BYTES)
        return MALFORMED;
    if (read16(frame + 12) != ETHERTYPE_IPV4_LE)
        return MALFORMED;
    const uint8_t *ip = frame + 14;
    if (ip[0] != VERSION_AND_LENGTH)
        return MALFORMED;
    uint32_t total = (uint32_t)ip[2] << 8 | ip[3];
    if (total < 20 || total > length - 14)
        return MALFORMED;
    uint32_t sum = 0;
    for (unsigned at = 0; at < 20; at += 2)
        sum += read16(ip + at);
    /* At most 0x9FFF6, so one fold gives at most 0x10008, and what is above 0xFFFF cannot fold to it. */
    if (fold(sum) != 0xFFFF)
        return MALFORMED;
    uint32_t ttl = ip[8];
    if (ttl <= 1)
        return EXPIRED;
    uint32_t hop = next_hop(ip + 16);
    if (hop == NO_ROUTE)
        return MALFORMED;
    /* RFC 1624, equation 3, on the word holding the TTL and the protocol. The TTL is its low byte
     * here, so m' = m - 1, and ~m + m' is 0xFFFE whatever m is: at most 0x1FFFD, so one fold will do. */
    uint32_t old_word = read16(ip + 8);
    uint32_t new_word = old_word - 1;
    uint32_t patched = (~(uint32_t)read16(ip + 10) & 0xFFFF) + (~old_word & 0xFFFF) + new_word;
    uint16_t checksum = (uint16_t)~fold(patched);
    memcpy(frame + 24, &checksum, 2);
    frame[22] = (uint8_t)(ttl - 1);
    memcpy(frame, hop_macs[hop], 12);
    return FORWARD;
}

/* --- AF_XDP */

static int bpf_call(int command, union bpf_attr *attr) {
    return (int)syscall(SYS_bpf, command, attr, sizeof *attr);
}

static void map_ring(struct ring *ring, int fd, const struct xdp_ring_offset *offsets, off_t where, size_t bytes) {
    bytes += offsets->desc;
    uint8_t *base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, where);
    if (base == MAP_FAILED)
        die("mapping a ring");
    ring->producer = (uint32_t *)(base + offsets->producer);
    ring->consumer = (uint32_t *)(base + offsets->consumer);
    ring->flags = (uint32_t *)(base + offsets->flags);
    ring->descriptors = base + offsets->desc;
}

static inline void publish(uint32_t *where, uint32_t value) {
    __atomic_store_n(where, value, __ATOMIC_RELEASE);
}

static inline uint32_t observe(const uint32_t *where) {
    return __atomic_load_n(where, __ATOMIC_ACQUIRE);
}

static inline void refill(struct ring *fill, uint64_t address) {
    ((uint64_t *)fill->descriptors)[fill->cached_producer++ & RING_MASK] = address;
}

/* EAGAIN and ENOBUFS leave the rest on tx for the next turn. EBUSY is a frame the kernel took and dropped.
 * In copy mode poll sends from tx as well and cannot say so, so this is a lower bound: see tx_dropped. */
static void wake_tx(int xsk) {
    if (sendto(xsk, NULL, 0, MSG_DONTWAIT, NULL, 0) < 0 && errno != EAGAIN && errno != ENOBUFS)
        count.send_errors++;
}

int main(int argc, char **argv) {
    const char *interface = NULL, *map_path = NULL, *routes = NULL;
    int control_port = 0, queue = 0;
    static const struct option options[] = {
        {"ifname", required_argument, NULL, 'i'}, {"map", required_argument, NULL, 'm'},
        {"ctl-port", required_argument, NULL, 'c'}, {"routes", required_argument, NULL, 'r'},
        {"match", no_argument, NULL, 'M'}, {"queue", required_argument, NULL, 'q'}, {NULL, 0, NULL, 0}};
    for (int option; (option = getopt_long(argc, argv, "", options, NULL)) != -1;) {
        switch (option) {
        case 'i': interface = optarg; break;
        case 'm': map_path = optarg; break;
        case 'c': control_port = atoi(optarg); break;
        case 'r': routes = optarg; break;
        case 'M': use_match = 1; break;
        case 'q': queue = atoi(optarg); break;
        default: return 2;
        }
    }
    if (!interface || !map_path || !control_port || (use_match == (routes != NULL))) {
        fprintf(stderr, "usage: ipv4_forward --ifname IF --map PINNED --ctl-port PORT (--routes FILE | --match)\n");
        return 2;
    }
    for (unsigned hop = 0; hop < MAX_HOPS; hop++) {
        const uint8_t macs[12] = {2, 0, 0, 0, 0, (uint8_t)(0x10 + hop), 2, 0, 0, 0, 1, (uint8_t)(0x10 + hop)};
        memcpy(hop_macs[hop], macs, 12);
    }
    if (routes) {
        make_table();
        load_routes(routes);
    }
    if (mlockall(MCL_CURRENT | MCL_FUTURE))
        die("mlockall");

    umem = mmap(NULL, (size_t)FRAMES * FRAME_SIZE, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (umem == MAP_FAILED)
        die("mapping the packet memory");
    int xsk = socket(AF_XDP, SOCK_RAW, 0);
    if (xsk < 0)
        die("opening the AF_XDP socket");
    struct xdp_umem_reg region = {.addr = (uint64_t)(uintptr_t)umem, .len = (uint64_t)FRAMES * FRAME_SIZE,
                                  .chunk_size = FRAME_SIZE, .headroom = 0};
    if (setsockopt(xsk, SOL_XDP, XDP_UMEM_REG, &region, sizeof region))
        die("registering the packet memory");
    int size = RING_SIZE, tx_size = TX_SIZE;
    if (setsockopt(xsk, SOL_XDP, XDP_UMEM_FILL_RING, &size, sizeof size) ||
        setsockopt(xsk, SOL_XDP, XDP_UMEM_COMPLETION_RING, &size, sizeof size) ||
        setsockopt(xsk, SOL_XDP, XDP_RX_RING, &size, sizeof size) ||
        setsockopt(xsk, SOL_XDP, XDP_TX_RING, &tx_size, sizeof tx_size))
        die("sizing the rings");
    struct xdp_mmap_offsets offsets;
    socklen_t offsets_length = sizeof offsets;
    if (getsockopt(xsk, SOL_XDP, XDP_MMAP_OFFSETS, &offsets, &offsets_length))
        die("asking where the rings are");
    struct ring rx = {0}, tx = {0}, fill = {0}, completion = {0};
    map_ring(&rx, xsk, &offsets.rx, XDP_PGOFF_RX_RING, RING_SIZE * sizeof(struct xdp_desc));
    map_ring(&tx, xsk, &offsets.tx, XDP_PGOFF_TX_RING, TX_SIZE * sizeof(struct xdp_desc));
    map_ring(&fill, xsk, &offsets.fr, XDP_UMEM_PGOFF_FILL_RING, RING_SIZE * sizeof(uint64_t));
    map_ring(&completion, xsk, &offsets.cr, XDP_UMEM_PGOFF_COMPLETION_RING, RING_SIZE * sizeof(uint64_t));
    for (uint32_t i = 0; i < FRAMES; i++)
        refill(&fill, (uint64_t)i * FRAME_SIZE);
    publish(fill.producer, fill.cached_producer);

    struct sockaddr_xdp where = {.sxdp_family = AF_XDP, .sxdp_ifindex = if_nametoindex(interface),
                                 .sxdp_queue_id = (uint32_t)queue, .sxdp_flags = XDP_USE_NEED_WAKEUP};
    if (!where.sxdp_ifindex)
        die(interface);
    if (bind(xsk, (struct sockaddr *)&where, sizeof where))
        die("binding to the interface");
    struct xdp_options bound;
    socklen_t bound_length = sizeof bound;
    if (!getsockopt(xsk, SOL_XDP, XDP_OPTIONS, &bound, &bound_length))
        count.zerocopy = bound.flags & XDP_OPTIONS_ZEROCOPY ? 1 : 0;

    union bpf_attr attribute;
    memset(&attribute, 0, sizeof attribute);
    attribute.pathname = (uint64_t)(uintptr_t)map_path;
    int map = bpf_call(BPF_OBJ_GET, &attribute);
    if (map < 0)
        die(map_path);
    uint32_t key = (uint32_t)queue, value = (uint32_t)xsk;
    memset(&attribute, 0, sizeof attribute);
    attribute.map_fd = (uint32_t)map;
    attribute.key = (uint64_t)(uintptr_t)&key;
    attribute.value = (uint64_t)(uintptr_t)&value;
    if (bpf_call(BPF_MAP_UPDATE_ELEM, &attribute))
        die("registering with the socket map");

    int control = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (control < 0)
        die("opening the control port");
    struct sockaddr_in control_address = {.sin_family = AF_INET, .sin_port = htons((uint16_t)control_port),
                                          .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    if (bind(control, (struct sockaddr *)&control_address, sizeof control_address))
        die("binding the control port");

    struct pollfd waiting = {.fd = xsk, .events = POLLIN};
    uint64_t turn = 0;
    for (;;) {
        int worked = 0;
        uint32_t done = observe(completion.producer);
        if (done != completion.cached_consumer) {
            worked = 1;
            for (; completion.cached_consumer != done; completion.cached_consumer++, count.completed++)
                refill(&fill, ((uint64_t *)completion.descriptors)[completion.cached_consumer & RING_MASK]);
            publish(completion.consumer, completion.cached_consumer);
            publish(fill.producer, fill.cached_producer);
        }
        uint32_t arrived = observe(rx.producer) - rx.cached_consumer;
        if (arrived > BATCH)
            arrived = BATCH;
        if (arrived) {
            worked = 1;
            count.frames += arrived;
            for (uint32_t i = 0; i < arrived; i++, rx.cached_consumer++) {
                const struct xdp_desc *in = &((struct xdp_desc *)rx.descriptors)[rx.cached_consumer & RING_MASK];
                uint64_t address = in->addr;
                uint32_t length = in->len;
                switch (forward(umem + address, length)) {
                case FORWARD:
                    /* The kernel's consumer is a cache line it writes, so look only when out of room. */
                    if (tx.cached_producer - tx.cached_consumer == TX_SIZE)
                        tx.cached_consumer = observe(tx.consumer);
                    if (tx.cached_producer - tx.cached_consumer == TX_SIZE) {
                        count.full++;
                        refill(&fill, address);
                    } else {
                        struct xdp_desc *out = &((struct xdp_desc *)tx.descriptors)[tx.cached_producer++ & TX_MASK];
                        out->addr = address;
                        out->len = length;
                        out->options = 0;
                        count.forwarded++;
                    }
                    break;
                case MALFORMED:
                    count.malformed++;
                    refill(&fill, address);
                    break;
                case EXPIRED:
                    count.expired++;
                    refill(&fill, address);
                    break;
                }
            }
            publish(rx.consumer, rx.cached_consumer);
            publish(fill.producer, fill.cached_producer);
            publish(tx.producer, tx.cached_producer);
            if (observe(tx.flags) & XDP_RING_NEED_WAKEUP)
                wake_tx(xsk);
        } else {
            if (tx.cached_producer != observe(tx.consumer) && observe(tx.flags) & XDP_RING_NEED_WAKEUP)
                wake_tx(xsk); /* copy mode sends only so many at a time */
            if (observe(fill.flags) & XDP_RING_NEED_WAKEUP)
                poll(&waiting, 1, 0); /* a card out of frames stops looking at fill until it is woken */
        }
        if (!worked)
            poll(&waiting, 1, 1);
        if (worked && ++turn % 1024)
            continue;
        char command[16];
        struct sockaddr_in peer;
        socklen_t peer_length = sizeof peer;
        if (recvfrom(control, command, sizeof command, 0, (struct sockaddr *)&peer, &peer_length) >= 1) {
            if (command[0] == 'Q')
                return 0;
            struct xdp_statistics statistics;
            socklen_t statistics_length = sizeof statistics;
            if (!getsockopt(xsk, SOL_XDP, XDP_STATISTICS, &statistics, &statistics_length))
                count.kernel_dropped = statistics.rx_dropped + statistics.rx_ring_full;
            sendto(control, &count, sizeof count, 0, (struct sockaddr *)&peer, peer_length);
        }
    }
}
