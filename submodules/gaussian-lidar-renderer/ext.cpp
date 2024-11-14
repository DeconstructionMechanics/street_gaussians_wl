#include "lidar_render.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_beam", &render_beam);
}
