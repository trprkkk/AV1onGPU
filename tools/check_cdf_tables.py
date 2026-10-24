#!/usr/bin/env python3
"""Check every default probability table in the decoder against the specification.

``gen_cdf_tables.py`` generates the inter tables straight from the
specification's annex and can therefore only be stale or right. The intra and
coefficient tables in ``cdf_tables.hpp`` were transcribed once and are held in
place by decoded frame hashes, which pins them for the frames the tests use but
says nothing about the rows those frames never touched - an entire quantiser
class, a whole transform size, a context only a rare block shape reaches.

This tool reads the same annex, re-does the forward-to-inverse conversion, and
compares the result with both headers row by row, in order. Where a header
deliberately differs from the specification's shape the difference is listed in
``SHAPE_ADJUSTMENTS`` with the reason, and the comparison is made on the
adjusted table, so a second, accidental difference cannot slip through as
"just padding".

    tools/check_cdf_tables.py --spec /path/to/av1-spec/10.additional.tables.md
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_cdf_tables as gen  # noqa: E402 - the parser lives with the generator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INTRA = os.path.join(ROOT, "src", "frontend", "cdf_tables.hpp")
INTER = os.path.join(ROOT, "src", "frontend", "cdf_tables_inter.hpp")

# Our array -> the specification tables whose rows it holds, outermost
# dimension first. The four partition tables are stored largest block first
# (level 0 is 128x128), the specification prints them smallest first.
INTRA_TABLES = {
    "kfym": ["Default_Intra_Frame_Y_Mode_Cdf"],
    "partition": ["Default_Partition_W128_Cdf", "Default_Partition_W64_Cdf",
                  "Default_Partition_W32_Cdf", "Default_Partition_W16_Cdf",
                  "Default_Partition_W8_Cdf"],
    "skip": ["Default_Skip_Cdf"],
    "intrabc": ["Default_Intrabc_Cdf"],
    "seg_id": ["Default_Segment_Id_Cdf"],
    "seg_pred": ["Default_Segment_Id_Predicted_Cdf"],
    "delta_q": ["Default_Delta_Q_Cdf"],
    "delta_lf": ["Default_Delta_Lf_Cdf"],
    "uv_mode": ["Default_Uv_Mode_Cfl_Not_Allowed_Cdf", "Default_Uv_Mode_Cfl_Allowed_Cdf"],
    "angle_delta": ["Default_Angle_Delta_Cdf"],
    "filter_intra": ["Default_Filter_Intra_Mode_Cdf"],
    "use_filter_intra": ["Default_Filter_Intra_Cdf"],
    "pal_sz": ["Default_Palette_Y_Size_Cdf", "Default_Palette_Uv_Size_Cdf"],
    "pal_y": ["Default_Palette_Y_Mode_Cdf"],
    "pal_uv": ["Default_Palette_Uv_Mode_Cdf"],
    "cfl_sign": ["Default_Cfl_Sign_Cdf"],
    "cfl_alpha": ["Default_Cfl_Alpha_Cdf"],
    "txtp_intra1": ["Default_Intra_Tx_Type_Set1_Cdf"],
    "txtp_intra2": ["Default_Intra_Tx_Type_Set2_Cdf"],
    "txsz": ["Default_Tx_8x8_Cdf", "Default_Tx_16x16_Cdf",
             "Default_Tx_32x32_Cdf", "Default_Tx_64x64_Cdf"],
    "txpart": ["Default_Txfm_Split_Cdf"],
    "eob_bin_16": ["Default_Eob_Pt_16_Cdf"],
    "eob_bin_32": ["Default_Eob_Pt_32_Cdf"],
    "eob_bin_64": ["Default_Eob_Pt_64_Cdf"],
    "eob_bin_128": ["Default_Eob_Pt_128_Cdf"],
    "eob_bin_256": ["Default_Eob_Pt_256_Cdf"],
    "eob_bin_512": ["Default_Eob_Pt_512_Cdf"],
    "eob_bin_1024": ["Default_Eob_Pt_1024_Cdf"],
    "eob_base_tok": ["Default_Coeff_Base_Eob_Cdf"],
    "base_tok": ["Default_Coeff_Base_Cdf"],
    "br_tok": ["Default_Coeff_Br_Cdf"],
    "eob_hi_bit": ["Default_Eob_Extra_Cdf"],
    "coef_skip": ["Default_Txb_Skip_Cdf"],
    "dc_sign": ["Default_Dc_Sign_Cdf"],
}

# The inter header's arrays are the generator's own table list: one
# specification table each, some of them broadcast over the two motion vector
# components.
INTER_TABLES = {name: [spec] for name, spec in gen.TABLES}

# The two shapes that deliberately differ from the specification: a dimension
# the header does not store, with the reason. The comparison drops the same
# slice from the specification's table, so any *other* difference still fails.
SHAPE_ADJUSTMENTS = {
    # SIG_COEF_CONTEXTS is 42 but the token context is clamped to 40, so the
    # specification's last row is never indexed and is not stored.
    "base_tok": ("drop", 3, 41, "the 42nd sign context is unused"),
    # The base range token is only read below TX_32X32, so the fifth transform
    # size slice is dead. dav1d stores four slices too.
    "br_tok": ("drop", 1, 4, "the largest transform size never reads it"),
    # One table for the whole superblock, but the five delta LF buffers adapt
    # separately, so the header keeps a copy per buffer.
    "delta_lf": ("broadcast", 5, 0, "one table per delta LF buffer"),
}

# Tables the decoder does not use yet, and so does not carry.
UNUSED = ["Default_Use_Wiener_Cdf", "Default_Use_Sgrproj_Cdf", "Default_Restoration_Type_Cdf"]


def parse_header(path):
    """Every ``inline const uint16_t name[dims] = { ... }`` in a header."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    tables = {}
    for match in re.finditer(r"inline const uint16_t (\w+)((?:\[\d+\])+)\s*=\s*\{", text):
        dims = [int(d) for d in re.findall(r"\d+", match.group(2))]
        start = text.index("{", match.end() - 1)
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        tables[match.group(1)] = (tuple(dims), nest(text[start:i + 1]))
    return tables


def nest(body):
    """A C initialiser into nested lists of ints."""
    tokens = re.findall(r"[{}]|\d+", body)
    stack, current = [], []
    for token in tokens:
        if token == "{":
            stack.append(current)
            current = []
        elif token == "}":
            done, current = current, stack.pop()
            current.append(done)
        else:
            current.append(int(token))
    return current[0]


def flatten(table):
    """Nested lists to the list of leaf rows, in order."""
    if table and isinstance(table[0], list):
        out = []
        for row in table:
            out.extend(flatten(row))
        return out
    return [table]


def drop_axis(table, axis, index):
    """Remove one entry along ``axis``, in every parent group."""
    if axis == 0:
        del table[index]
    else:
        for child in table:
            drop_axis(child, axis - 1, index)
    return table


def compare(name, ours, spec_rows, path, failures):
    """Row by row, ours against the specification, after stripping padding."""
    ours_rows = flatten(ours)
    if len(ours_rows) != len(spec_rows):
        failures.append("%s: %d rows in the header, %d in the specification"
                        % (name, len(ours_rows), len(spec_rows)))
        return
    for index, (row, expected) in enumerate(zip(ours_rows, spec_rows)):
        got = list(row)
        while got and got[-1] == 0:
            got.pop()
        want = list(expected)
        while want and want[-1] == 0:
            want.pop()
        if got != want:
            failures.append("%s: row %d is %r, the specification has %r"
                            % (name, index, row, expected))
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True,
                        help="path to the specification's 10.additional.tables.md")
    args = parser.parse_args()

    spec = gen.parse_spec_tables(args.spec)
    failures = []
    checked = 0
    adjusted = []

    for path, mapping in ((INTRA, INTRA_TABLES), (INTER, INTER_TABLES)):
        header = parse_header(path)
        for name, spec_names in mapping.items():
            if name not in header:
                failures.append("%s: the header has no %s" % (os.path.basename(path), name))
                continue
            missing = [s for s in spec_names if s not in spec]
            if missing:
                failures.append("%s: the specification has no %s" % (name, ", ".join(missing)))
                continue
            table = spec[spec_names[0]] if len(spec_names) == 1 else \
                [spec[s] for s in spec_names]
            adjustment = SHAPE_ADJUSTMENTS.get(name)
            if adjustment:
                kind, first, second, reason = adjustment
                if kind == "drop":
                    drop_axis(table, first, second)
                else:
                    table = [table for _ in range(first)]
                adjusted.append("%s: %s" % (name, reason))
            rows = flatten(gen.to_inverse(table, spec_names[0]))
            compare(name, header[name][1], rows, path, failures)
            checked += 1

    for name in UNUSED:
        if name not in spec:
            failures.append("the specification has no %s" % name)

    if failures:
        for failure in failures:
            sys.stderr.write(failure + "\n")
        return 1
    print("%d default tables match the specification" % checked)
    for line in adjusted:
        print("  %s" % line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
