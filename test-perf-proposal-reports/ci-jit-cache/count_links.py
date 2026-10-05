# Usage: python3 count_links.py CACHE_DIR
# Counts the nvJitLink links stored in a CUDA JIT cache directory, by kernel family (and data type
# for CAGRA). Each link stores one entry keyed by the linked NVVM IR, and usually a second one keyed
# by its PTX. The linked fragments are named in the IR by _INTERNAL_<crc>_<n>_<file>_cu_<crc> symbols.
import collections, os, re, struct, sys

FAMILIES = [b"cagra_search_single_cta_mp", b"cagra_search_single_cta", b"cagra_search_multi_cta_mp",
            b"cagra_search_multi_cta", b"cagra_random_pickup", b"cagra_compute_distance_to_child_nodes",
            b"cagra_apply_filter", b"ivf_pq_compute_similarity", b"ivf_flat_interleaved_scan",
            b"ivf_sq_scan", b"ivf_rabitq", b"pairwise_matrix"]
counts = collections.Counter()
for root, _, files in os.walk(sys.argv[1]):
    for name in files:
        if name == "index":
            continue
        data = open(os.path.join(root, name), "rb").read()
        key = data[58 : 58 + struct.unpack_from("<Q", data, 4)[0] - 30]  # u32, u64 key size + 30, ...
        if key[:4] != b"BC\xc0\xde":  # keyed by PTX: the second entry of a link
            continue
        frags = set(re.findall(rb"_INTERNAL_[0-9a-f]{8}_\d+_(\w+?)_cu_[0-9a-f]{8}", key))
        family = next((f for f in FAMILIES if any(x.startswith(f) for x in frags)), b"other")
        dtype = re.search(rb"_data_(f|h|i8|u8)_", b" ".join(sorted(frags)) + b" ")
        dtype = dtype.group(1) if dtype and family.startswith(b"cagra_search") else b""
        counts[(family.decode(), dtype.decode())] += 1
for (family, dtype), n in sorted(counts.items()):
    print(f"{n:5d}  {family} {dtype}")
print(f"{sum(counts.values()):5d}  links")
