# nvjpeg2k-python

NVJPEG2K Python binding, this is mainly used for the Kaggle competition [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) so the functionality may not completed, for now it only support `decode` and lack of test.

See this [kaggle notebook](https://www.kaggle.com/code/snaker/easy-load-the-image-with-nvjpeg2000) that use this library to boost the image loading speed.

## Steps compile


**Requirements**

* Linux
* cmake
* CUDA 11.x
* Python 3.6 or later
* libnvjpeg_2k, you can download it from [here](https://developer.nvidia.com/nvjpeg), be sure to download the nvJPEG2000.

**Build**

```bash
# get the source code
git clone --recursive https://github.com/louis-she/nvjpeg2k-python.git
cd nvjpeg2k-python

# build the extension
mkdir build
cd build

# **ABSOLUTE PATH** of libnvjpeg2k path, modified this in your own case
PATH_OF_THE_LIBNVJPEG_2K="/....../libnvjpeg_2k-linux-x86_64-0.6.0.28-archive"

# can get rid of the `-DCMAKE_BUILD_TYPE=Debug` option
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DNVJPEG2K_PATH=${PATH_OF_THE_LIBNVJPEG_2K} \
  -DNVJPEG2K_LIB=${PATH_OF_THE_LIBNVJPEG_2K}/lib/libnvjpeg2k_static.a

make
```

## Usage

```python
import nvjpeg2k
from pathlib import Path

decoder = nvjpeg2k.Decoder()
res = decoder.decode(Path("../assets/demo.dcm.jp2").read_bytes())
print(res.shape)
```
