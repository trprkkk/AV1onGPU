#include "frontend/obu.hpp"

namespace av1gpu {
namespace frontend {
namespace {

/* The largest payload an OBU may describe; also keeps the size arithmetic in a
 * range that cannot overflow size_t on 32 bit hosts. */
constexpr uint64_t kMaxObuSize = 1ull << 32;

} /* namespace */

const char* obu_name(ObuType type) {
  switch (type) {
    case ObuType::SequenceHeader: return "sequence_header";
    case ObuType::TemporalDelimiter: return "temporal_delimiter";
    case ObuType::FrameHeader: return "frame_header";
    case ObuType::TileGroup: return "tile_group";
    case ObuType::Metadata: return "metadata";
    case ObuType::Frame: return "frame";
    case ObuType::RedundantFrameHeader: return "redundant_frame_header";
    case ObuType::TileList: return "tile_list";
    case ObuType::Padding: return "padding";
    case ObuType::Reserved0: break;
  }
  return "reserved";
}

bool obu_type_from_int(int value, ObuType* out) {
  switch (value) {
    case 1: *out = ObuType::SequenceHeader; return true;
    case 2: *out = ObuType::TemporalDelimiter; return true;
    case 3: *out = ObuType::FrameHeader; return true;
    case 4: *out = ObuType::TileGroup; return true;
    case 5: *out = ObuType::Metadata; return true;
    case 6: *out = ObuType::Frame; return true;
    case 7: *out = ObuType::RedundantFrameHeader; return true;
    case 8: *out = ObuType::TileList; return true;
    case 15: *out = ObuType::Padding; return true;
    default: return false; /* types 9..14 are reserved and must not appear */
  }
}

bool read_leb128(const uint8_t* data, size_t size, size_t* pos, uint64_t* value,
                 std::string* error) {
  uint64_t result = 0;
  for (int i = 0; i < 8; ++i) {
    if (*pos >= size) {
      if (error) *error = "leb128 runs past the end of the buffer";
      return false;
    }
    const uint8_t byte = data[(*pos)++];
    result |= static_cast<uint64_t>(byte & 0x7f) << (7 * i);
    if (!(byte & 0x80)) {
      *value = result;
      return true;
    }
  }
  if (error) *error = "leb128 longer than eight bytes";
  return false;
}

bool parse_obu_header(const uint8_t* data, size_t size, ObuHeader* header, size_t* pos,
                      size_t* payload_size, std::string* error) {
  if (*pos >= size) {
    if (error) *error = "truncated OBU header";
    return false;
  }
  const uint8_t byte = data[(*pos)++];
  if (byte & 0x80) {
    if (error) *error = "OBU forbidden bit set (not an AV1 stream?)";
    return false;
  }
  if (!obu_type_from_int((byte >> 3) & 0x0f, &header->type)) {
    if (error) *error = "reserved OBU type " + std::to_string((byte >> 3) & 0x0f);
    return false;
  }
  header->has_extension = (byte >> 2) & 1;
  header->has_size = (byte >> 1) & 1;
  if ((byte & 1) != 0) {
    if (error) *error = "OBU header reserved bit is set";
    return false;
  }

  if (header->has_extension) {
    if (*pos >= size) {
      if (error) *error = "truncated OBU extension header";
      return false;
    }
    const uint8_t extension = data[(*pos)++];
    header->temporal_id = (extension >> 5) & 0x07;
    header->spatial_id = (extension >> 3) & 0x03;
    if ((extension & 0x07) != 0) {
      if (error) *error = "OBU extension reserved bits are set";
      return false;
    }
  }

  if (header->has_size) {
    uint64_t declared = 0;
    if (!read_leb128(data, size, pos, &declared, error)) return false;
    if (declared > kMaxObuSize || declared > size - *pos) {
      if (error) {
        *error = "OBU size " + std::to_string(declared) + " does not fit in the " +
                 std::to_string(size - *pos) + " bytes that are left";
      }
      return false;
    }
    *payload_size = static_cast<size_t>(declared);
  } else {
    /* Low overhead bitstream format: the OBU runs to the end of the temporal
     * unit. Only legal for the last OBU, which is what "size fits" means. */
    *payload_size = size - *pos;
  }
  return true;
}

bool split_obus(const uint8_t* data, size_t size, std::vector<Obu>* obus,
                std::string* error) {
  obus->clear();
  size_t pos = 0;
  while (pos < size) {
    Obu obu;
    ObuHeader header;
    size_t payload_size = 0;
    const size_t start = pos;
    if (!parse_obu_header(data, size, &header, &pos, &payload_size, error)) return false;
    obu.type = header.type;
    obu.temporal_id = header.temporal_id;
    obu.spatial_id = header.spatial_id;
    obu.data = data + pos;
    obu.size = payload_size;
    obu.header_size = pos - start;
    obus->push_back(obu);
    pos += payload_size;
  }
  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */
