/*
 * AV1onGPU - OBU framing
 *
 * An AV1 stream is a sequence of Open Bitstream Units. Each one starts with a
 * one byte header (type, extension flag, size-present flag), an optional
 * extension byte (temporal/spatial id), and - in the "low overhead" form used
 * by .ivf and .obu files - a leb128 payload size.
 *
 * The layer here splits a temporal unit into OBUs and hands the payload to the
 * header parsers. It deliberately does not interpret any payload: sequence and
 * frame headers live in headers.cpp.
 */
#ifndef AV1GPU_FRONTEND_OBU_HPP
#define AV1GPU_FRONTEND_OBU_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av1gpu {
namespace frontend {

enum class ObuType : int {
  Reserved0 = 0,
  SequenceHeader = 1,
  TemporalDelimiter = 2,
  FrameHeader = 3,
  TileGroup = 4,
  Metadata = 5,
  Frame = 6,
  RedundantFrameHeader = 7,
  TileList = 8,
  Padding = 15,
};

struct ObuHeader {
  ObuType type = ObuType::Reserved0;
  bool has_extension = false;
  bool has_size = false;
  int temporal_id = 0;
  int spatial_id = 0;
};

struct Obu {
  ObuType type = ObuType::Reserved0;
  int temporal_id = 0;
  int spatial_id = 0;
  const uint8_t* data = nullptr; /* payload, size bytes */
  size_t size = 0;
  size_t header_size = 0; /* bytes before the payload, including the size field */
};

/* The four bit type field as a printable name, for tools and error messages. */
const char* obu_name(ObuType type);
bool obu_type_from_int(int value, ObuType* out);

/* Reads a leb128 from `data` at `*pos`, advancing `*pos`. */
bool read_leb128(const uint8_t* data, size_t size, size_t* pos, uint64_t* value,
                 std::string* error);

/* Parses one OBU header at `data`. `*pos` is advanced past the header (but not
 * past the size field) and the payload size is returned in `*payload_size`.
 * When the size field is absent the payload runs to the end of `size`. */
bool parse_obu_header(const uint8_t* data, size_t size, ObuHeader* header, size_t* pos,
                      size_t* payload_size, std::string* error);

/* Splits one temporal unit (the payload of one .ivf frame) into OBUs. */
bool split_obus(const uint8_t* data, size_t size, std::vector<Obu>* obus,
                std::string* error);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_OBU_HPP */
