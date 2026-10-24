"""Emit src/frontend/quant_tables.hpp from the specification's quantiser tables.

The spec ships them as three 256 entry rows (8, 10 and 12 bit), indexed by a
quantiser index; this project's bit depth selection is `(bit_depth - 8) >> 1`,
exactly like the spec's `Dc_Qlookup[ (BitDepth-8) >> 1 ]`.

Run with the specification checkout at /tmp/av1-spec-master:

    python3 tools/gen_quant_tables.py /tmp/av1-spec-master

The tables are read out of the markdown rather than retyped, because a single
wrong entry would only show up as a slightly wrong pixel after dequantisation -
the hardest kind of bug to find.
"""
import re
import sys

SPEC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/av1-spec-master'
text = open(SPEC + '/08.decoding.process.md').read()

HEADER = '''/*
 * AV1onGPU - dequantisation tables (generated, do not edit by hand)
 *
 * Generated from the specification by tools/gen_quant_tables.py. One row per
 * bit depth (8, 10, 12), 256 entries each, indexed by a quantiser index in
 * 0..255; the dequantisation step for a coefficient is this value multiplied by
 * its level, with the DC coefficient using the DC row and everything else the
 * AC row.
 */
#ifndef AV1GPU_FRONTEND_QUANT_TABLES_HPP
#define AV1GPU_FRONTEND_QUANT_TABLES_HPP

#include <cstdint>

namespace av1gpu {
namespace frontend {
namespace quant_tables {

'''


def extract(name):
    m = re.search(r'\b' + name + r'\[ 3 \]\[ 256 \] = \{(.*?)\n~~~~~', text, re.S)
    if not m:
        raise SystemExit('table not found: ' + name)
    rows = re.findall(r'\{([^{}]*)\}', m.group(1))
    out = []
    for row in rows:
        values = [int(v) for v in re.findall(r'-?\d+', row)]
        if len(values) != 256:
            raise SystemExit('%s row has %d entries' % (name, len(values)))
        out.append(values)
    if len(out) != 3:
        raise SystemExit('%s has %d rows' % (name, len(out)))
    return out


def emit(name, rows, comment):
    lines = [comment, 'inline const uint16_t %s[3][256] = {' % name]
    for row in rows:
        lines.append('    {')
        for i in range(0, 256, 8):
            lines.append('        ' + ', '.join(str(v) for v in row[i:i + 8]) + ',')
        lines.append('    },')
    lines.append('};')
    lines.append('')
    return '\n'.join(lines)


body = []
body.append(emit('kDcQlookup', extract('Dc_Qlookup'),
                 '/* Step size for the DC coefficient. */'))
body.append(emit('kAcQlookup', extract('Ac_Qlookup'),
                 '/* Step size for the other coefficients. */'))
body.append('} /* namespace quant_tables */\n} /* namespace frontend */\n'
            '} /* namespace av1gpu */\n\n#endif /* AV1GPU_FRONTEND_QUANT_TABLES_HPP */\n')

path = 'src/frontend/quant_tables.hpp'
open(path, 'w').write(HEADER + '\n'.join(body))
print('wrote', path)
