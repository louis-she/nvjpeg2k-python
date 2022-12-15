#include <cuda_runtime.h>
#include <nvjpeg2k.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iostream>
#include <string>

struct Decoder {
  Decoder() {
    nvjpeg2kCreateSimple(&jpeg2k_handle);
    nvjpeg2kDecodeStateCreate(jpeg2k_handle, &jpeg2k_decode_state);
    nvjpeg2kStreamCreate(&jpeg2k_stream);
    nvjpeg2kDecodeParamsCreate(&decode_params);
    nvjpeg2kDecodeParamsSetRGBOutput(decode_params, 0);
  };

  ~Decoder() {
    nvjpeg2kDecodeParamsDestroy(decode_params);
    nvjpeg2kDecodeStateDestroy(jpeg2k_decode_state);
    nvjpeg2kDestroy(jpeg2k_handle);
    nvjpeg2kStreamDestroy(jpeg2k_stream);
  };

  pybind11::array_t<uint16_t> decode(std::string path) {
    std::ostringstream sstream;
    std::ifstream fs(path.c_str());
    sstream << fs.rdbuf();
    const std::string str(sstream.str());
    auto length = str.length();
    auto bitstream_buffer = (const unsigned char *)str.c_str();

    nvjpeg2kImageInfo_t image_info;
    nvjpeg2kImage_t output_image;
    int bytes_per_element = 1;

    std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info;
    std::vector<unsigned short *> decode_output_u16;
    std::vector<unsigned char *> decode_output_u8;
    std::vector<size_t> decode_output_pitch;

    nvjpeg2kStreamParse(jpeg2k_handle, bitstream_buffer, length, 0, 0,
                        jpeg2k_stream);
    nvjpeg2kStreamGetImageInfo(jpeg2k_stream, &image_info);
    image_comp_info.resize(image_info.num_components);
    for (uint32_t c = 0; c < image_info.num_components; c++) {
      nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &image_comp_info[c],
                                          c);
    }

    decode_output_pitch.resize(image_info.num_components);
    output_image.pitch_in_bytes = decode_output_pitch.data();
    if (image_comp_info[0].precision > 8 &&
        image_comp_info[0].precision <= 16) {
      decode_output_u16.resize(image_info.num_components);
      output_image.pixel_data = (void **)decode_output_u16.data();
      output_image.pixel_type = NVJPEG2K_UINT16;
      bytes_per_element = 2;
    } else {
      decode_output_u8.resize(image_info.num_components);
      output_image.pixel_data = (void **)decode_output_u8.data();
      output_image.pixel_type = NVJPEG2K_UINT8;
      bytes_per_element = 1;
    }

    output_image.num_components = image_info.num_components;
    for (uint32_t c = 0; c < image_info.num_components; c++) {
      cudaMallocPitch(
          &output_image.pixel_data[c], &output_image.pitch_in_bytes[c],
          image_info.image_width * bytes_per_element, image_info.image_height);
    }
    nvjpeg2kDecodeImage(jpeg2k_handle, jpeg2k_decode_state, jpeg2k_stream,
                        decode_params, &output_image, 0);

    auto pixels_length = image_info.image_height * image_info.image_width *
                         image_info.num_components;
    auto result = pybind11::array_t<uint16_t>(pixels_length);

    cudaMemcpy2D(result.request().ptr,
                 image_info.image_width * bytes_per_element,
                 output_image.pixel_data[0], output_image.pitch_in_bytes[0],
                 image_info.image_width * bytes_per_element,
                 image_info.image_height, cudaMemcpyDeviceToHost);
    for (uint32_t c = 0; c < image_info.num_components; c++) {
      cudaFree(output_image.pixel_data[c]);
    }

    return result.reshape({image_info.image_height, image_info.image_width});
  };

  nvjpeg2kHandle_t jpeg2k_handle;
  nvjpeg2kStream_t jpeg2k_stream;
  nvjpeg2kDecodeState_t jpeg2k_decode_state;
  nvjpeg2kDecodeParams_t decode_params;
};

PYBIND11_MODULE(nvjpeg2k, m) {
  m.doc() = "pybind11 nvjpeg2k plugin";

  pybind11::class_<Decoder>(m, "Decoder")
      .def(pybind11::init<>())
      .def("decode", &Decoder::decode);
}

int main() {
  pybind11::initialize_interpreter();
  pybind11::module nvjpeg2k = pybind11::module::import("nvjpeg2k");
  // 2776 x 2082
  auto decoder = nvjpeg2k.attr("Decoder")();
  pybind11::array_t<double> result =
      decoder.attr("decode")("/home/featurize/output_2k/1031443799.dcm.jp2");
  pybind11::finalize_interpreter();
}
