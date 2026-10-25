#!/usr/bin/env python3
"""Emit the inter frame probability tables from the specification.

AV1's default CDFs live in the specification's annex (``10.additional.tables.md``
in the markdown edition of the spec) as ordinary C initialisers. Transcribing
them by hand is the kind of work that produces one wrong * 128 somewhere, and a
wrong probability does not crash - it decodes a different symbol and the picture
comes out subtly wrong thousands of symbols later. So they are generated.

Two conversions happen here:

* The specification prints *forward* cumulative frequencies (they increase to
  32768); the range decoder wants the *inverse* form dav1d and libaom keep
  (entry i is the probability that the symbol is below i, so it decreases),
  which is 32768 - x. The trailing 32768 becomes 0 and is dropped.

* A row of n symbols needs n - 1 stored probabilities plus the adaptation
  counter, which ``read_symbol`` reads past the last probability. The
  specification ends every row with 32768, 0, so a row of the table has exactly
  one more entry than the number of probabilities it prints.

The output is ``src/frontend/cdf_tables_inter.hpp``. Only the tables an inter
frame needs are emitted; the intra and coefficient tables in ``cdf_tables.hpp``
were transcribed earlier and are pinned by the frame hashes in
``tests/test_frontend.cpp``.

    tools/gen_cdf_tables.py --spec /path/to/av1-spec/10.additional.tables.md
    tools/gen_cdf_tables.py --spec ... --check    # exit 1 if the file is stale
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "src", "frontend", "cdf_tables_inter.hpp")

# C++ name -> the specification's table name. Sizes are taken from the
# specification's own initialiser, so a table that grows or shrinks upstream
# changes the generated array instead of being silently truncated.
TABLES = [
    ("y_mode", "Default_Y_Mode_Cdf"),
    ("is_inter", "Default_Is_Inter_Cdf"),
    ("skip_mode", "Default_Skip_Mode_Cdf"),
    ("comp_mode", "Default_Comp_Mode_Cdf"),
    ("comp_ref_type", "Default_Comp_Ref_Type_Cdf"),
    ("uni_comp_ref", "Default_Uni_Comp_Ref_Cdf"),
    ("comp_ref", "Default_Comp_Ref_Cdf"),
    ("comp_bwd_ref", "Default_Comp_Bwd_Ref_Cdf"),
    ("single_ref", "Default_Single_Ref_Cdf"),
    ("compound_mode", "Default_Compound_Mode_Cdf"),
    ("compound_idx", "Default_Compound_Idx_Cdf"),
    ("comp_group_idx", "Default_Comp_Group_Idx_Cdf"),
    ("compound_type", "Default_Compound_Type_Cdf"),
    ("wedge_index", "Default_Wedge_Index_Cdf"),
    ("inter_intra", "Default_Inter_Intra_Cdf"),
    ("inter_intra_mode", "Default_Inter_Intra_Mode_Cdf"),
    ("wedge_inter_intra", "Default_Wedge_Inter_Intra_Cdf"),
    ("use_obmc", "Default_Use_Obmc_Cdf"),
    ("motion_mode", "Default_Motion_Mode_Cdf"),
    ("interp_filter", "Default_Interp_Filter_Cdf"),
    ("mv_joint", "Default_Mv_Joint_Cdf"),
    ("mv_sign", "Default_Mv_Sign_Cdf"),
    ("mv_class", "Default_Mv_Class_Cdf"),
    ("mv_class0_bit", "Default_Mv_Class0_Bit_Cdf"),
    ("mv_class0_fr", "Default_Mv_Class0_Fr_Cdf"),
    ("mv_class0_hp", "Default_Mv_Class0_Hp_Cdf"),
    ("mv_hp", "Default_Mv_Hp_Cdf"),
    ("mv_bit", "Default_Mv_Bit_Cdf"),
    ("mv_fr", "Default_Mv_Fr_Cdf"),
    ("new_mv", "Default_New_Mv_Cdf"),
    ("zero_mv", "Default_Zero_Mv_Cdf"),
    ("ref_mv", "Default_Ref_Mv_Cdf"),
    ("drl_mode", "Default_Drl_Mode_Cdf"),
    ("tx_type_inter1", "Default_Inter_Tx_Type_Set1_Cdf"),
    ("tx_type_inter2", "Default_Inter_Tx_Type_Set2_Cdf"),
    ("tx_type_inter3", "Default_Inter_Tx_Type_Set3_Cdf"),
]

# The context shape each table is indexed by, for the comment in front of it.
NOTES = {
    "y_mode": "intra mode of an intra block found in an inter frame, per size group",
    "is_inter": "neighbours that were inter predicted",
    "skip_mode": "neighbours that used skip mode",
    "comp_mode": "how the neighbours were referenced",
    "comp_ref_type": "compound reference type context",
    "uni_comp_ref": "unidirectional compound reference",
    "comp_ref": "forward reference of a compound block",
    "comp_bwd_ref": "backward reference of a compound block",
    "single_ref": "reference of a single reference block",
    "compound_mode": "the compound mode symbol",
    "compound_idx": "distance weighted compound index",
    "comp_group_idx": "wedge or difference weighted compound",
    "compound_type": "which compound blend",
    "wedge_index": "wedge index, per block size",
    "inter_intra": "inter/intra compound, per size group",
    "inter_intra_mode": "the intra mode of an inter/intra block",
    "wedge_inter_intra": "wedge blend for inter/intra, per block size",
    "use_obmc": "overlapped block motion compensation, per block size",
    "motion_mode": "simple, OBMC or local warped motion",
    "interp_filter": "the subpel filter of an inter block",
    "mv_joint": "which motion vector components are coded",
    "mv_sign": "the sign of a motion vector component",
    "mv_class": "[horizontal] motion vector class",
    "mv_class0_bit": "the bit below class 0",
    "mv_class0_fr": "[class0 bit] class 0 fractional part",
    "mv_class0_hp": "[horizontal] class 0 high precision bit",
    "mv_hp": "[horizontal] high precision bit",
    "mv_bit": "[offset] the bits above the class base",
    "mv_fr": "[joint] fractional part of a class 0 component",
    "new_mv": "new motion vector context",
    "zero_mv": "zero motion vector context",
    "ref_mv": "reference motion vector context",
    "drl_mode": "dynamic reference list index",
    "tx_type_inter1": "transform type, inter, set 1",
    "tx_type_inter2": "transform type, inter, set 2",
    "tx_type_inter3": "transform type, inter, set 3",
}


def parse_spec_tables(path):
    """Every ``Name[ dims ] = { ... }`` CDF initialiser in the annex."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    tables = {}
    for match in re.finditer(r"^(Default_\w+_Cdf)\s*((?:\[[^\]]*\])+)\s*=\s*\{", text,
                             re.MULTILINE):
        name = match.group(1)
        dims = len(re.findall(r"\[", match.group(2)))
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
        body = text[start:i + 1]
        tables[name] = shape(body, dims)
    return tables


def shape(body, dims):
    """Turns a C initialiser into nested lists, evaluating the products."""
    body = body.replace("*", " * ")

    def value(token):
        token = token.strip()
        if not token:
            return None
        if not re.fullmatch(r"[0-9 *+()]+", token):
            raise ValueError("unexpected initialiser element %r" % token)
        return int(eval(token, {"__builtins__": {}}, {}))  # noqa: S307 - digits only

    if dims == 1:
        return [value(t) for t in body.strip("{} \n").split(",") if t.strip()]
    rows = []
    depth, start = 0, None
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
            if depth == 2:
                start = i
        elif ch == "}":
            if depth == 2:
                rows.append(shape(body[start:i + 1], dims - 1))
            depth -= 1
    return rows


def to_inverse(rows, name):
    """Forward cumulative (32768-terminated) to the decoder's inverse layout."""
    if isinstance(rows[0], list):
        return [to_inverse(row, name) for row in rows]
    probabilities = rows[:-1] if rows[-1] != 0 else rows[:-2]
    if not probabilities or rows[len(probabilities)] != 32768:
        raise ValueError("%s: a row does not end in 32768, 0: %r" % (name, rows))
    out = [32768 - p for p in probabilities] + [0]
    if any(b < 0 or a <= b for a, b in zip(out, out[1:]) if b or a):
        # the layout must decrease towards the last symbol; equal neighbours
        # only happen at the 32768/0 tail
        for a, b in zip(out, out[1:]):
            if b and a <= b:
                raise ValueError("%s: row does not decrease: %r" % (name, rows))
    return out


PAD_WIDTH = 16
PAD_THRESHOLD = 8


def pad_rows(table):
    """Pads long rows to sixteen entries, the alignment cdf_tables.hpp uses."""
    if isinstance(table[0], list):
        return [pad_rows(row) for row in table]
    if len(table) > PAD_THRESHOLD:
        return table + [0] * (PAD_WIDTH - len(table))
    return table


def palette_color_table(spec):
    """The fourteen palette colour index tables as one array.

    ``Default_Palette_Size_N_{Y,Uv}_Color_Cdf[ctx][N + 1]``: the table for a
    palette of N colours codes N - 1 probabilities, so the rows of the fourteen
    tables have different lengths. They are padded to eight entries here so one
    array can hold all of them; ``read_symbol`` only reads the entries its
    symbol count needs, and the padding sits past the adaptation counter.
    """
    planes = []
    for suffix in ("Y", "Uv"):
        sizes = []
        for size in range(2, 9):
            name = "Default_Palette_Size_%d_%s_Color_Cdf" % (size, suffix)
            if name not in spec:
                raise KeyError(name)
            rows = to_inverse(spec[name], name)
            sizes.append([row + [0] * (8 - len(row)) for row in rows])
        planes.append(sizes)
    return planes


def dims_of(table):
    dims = []
    node = table
    while isinstance(node, list):
        dims.append(len(node))
        node = node[0]
    return tuple(dims)


def emit_rows_body(table):
    """The body of a nested uint16_t initialiser, one row per line."""
    dims = dims_of(table)

    def rows(node, indent, top=False):
        """Rows, one per line; the outermost braces come from the declaration."""
        if isinstance(node[0], list):
            text = ",\n".join(rows(child, indent + 1) for child in node)
            if top:
                return text
            pad = "  " * indent
            return "%s{\n%s\n%s}" % (pad, text, pad)
        line = ", ".join(str(v) for v in node)
        # a single context table has no inner braces; indent its one row anyway
        return "%s%s" % ("  " * indent, line if top else "{ %s }" % line)

    return rows(table, 1, top=True)


def emit_array(name, table, spec_name, alphabet):
    dims = dims_of(table)
    suffix = "".join("[%d]" % d for d in dims)
    note = NOTES.get(name)
    shapes = " x ".join(str(d) for d in dims[:-1]) if len(dims) > 1 else "one context"
    comment = "/* %s\n   * %s */\n" % (
        "%s: %d symbols, %s." % (spec_name, alphabet, shapes), note) if note else ""
    return "%sinline const uint16_t %s%s = {\n%s,\n};\n\n" % (
        comment, name, suffix, emit_rows_body(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True,
                        help="path to the specification's 10.additional.tables.md")
    parser.add_argument("--check", action="store_true",
                        help="fail if the generated file is out of date")
    args = parser.parse_args()

    spec = parse_spec_tables(args.spec)
    body = []
    missing = []
    for name, spec_name in TABLES:
        if spec_name not in spec:
            missing.append(spec_name)
            continue
        rows = to_inverse(spec[spec_name], spec_name)
        first = rows
        while isinstance(first[0], list):
            first = first[0]
        alphabet = len(first) - 1
        body.append(emit_array(name, pad_rows(rows), spec_name, alphabet))

    # The palette colour index tables, one per palette size, in one array. The
    # symbol count is the palette size minus one, which is why they could not go
    # through TABLES: their rows are not all the same length.
    try:
        pal = palette_color_table(spec)
    except KeyError as missing_table:
        missing.append(str(missing_table).strip("'"))
    else:
        body.append(
            "/* palette colour index, [plane][palette size - 2][colour context].\n"
            "   * The palette size - 1 symbol index is coded as a symbol with\n"
            "   * palette size values, so the row for a palette of N colours codes\n"
            "   * N - 1 probabilities and is padded to eight entries. */\n"
            "inline const uint16_t pal_color[2][7][5][8] = {\n%s,\n};\n\n"
            % emit_rows_body(pal))
    if missing:
        sys.stderr.write("the specification has no %s\n" % ", ".join(missing))
        return 1

    text = (
        "/*\n"
        " * AV1onGPU - default probability tables for inter frames (generated)\n"
        " *\n"
        " * Generated by tools/gen_cdf_tables.py from the specification's annex of\n"
        " * default CDFs; do not edit. The tables are in the decoder's layout -\n"
        " * inverse cumulative, decreasing towards the last symbol, with the\n"
        " * adaptation counter after the last probability - which is the layout\n"
        " * RangeDecoder::read_symbol consumes and the one cdf_tables.hpp uses.\n"
        " *\n"
        " * The sizes are the specification's, so a row holds one entry more than\n"
        " * the number of probabilities it prints: the counter. A table's comment\n"
        " * names the contexts its dimensions index.\n"
        " */\n"
        "#ifndef AV1GPU_FRONTEND_CDF_TABLES_INTER_HPP\n"
        "#define AV1GPU_FRONTEND_CDF_TABLES_INTER_HPP\n"
        "\n"
        "#include <cstdint>\n"
        "\n"
        "namespace av1gpu {\n"
        "namespace frontend {\n"
        "namespace cdf_tables_inter {\n"
        "\n" + "".join(body) +
        "} /* namespace cdf_tables_inter */\n"
        "} /* namespace frontend */\n"
        "} /* namespace av1gpu */\n"
        "\n"
        "#endif /* AV1GPU_FRONTEND_CDF_TABLES_INTER_HPP */\n")

    if args.check:
        try:
            with open(OUTPUT, encoding="utf-8") as handle:
                if handle.read() != text:
                    sys.stderr.write("%s is out of date\n" % OUTPUT)
                    return 1
        except FileNotFoundError:
            sys.stderr.write("%s does not exist\n" % OUTPUT)
            return 1
        print("%s is up to date" % os.path.relpath(OUTPUT, ROOT))
        return 0

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("wrote %s (%d tables)" % (os.path.relpath(OUTPUT, ROOT), len(TABLES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
