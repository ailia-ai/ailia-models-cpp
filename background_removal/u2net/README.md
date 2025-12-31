# u2net

## Setup

[Check ailia-models-cpp/README.md](https://github.com/ailia-ai/ailia-models-cpp)

## Input

- input shape: (1, 3, 320, 320)  
- Preprocessing

```
max_pixel_value = max(image)
B = (B/max_pixel_value - 0.485) / 0.229
G = (G/max_pixel_value - 0.456) / 0.224
R = (R/max_pixel_value - 0.406) / 0.225
```

## Output

- output shape: (1, 1, 320, 320)
- Postprocessing

```
pred = ailia.predict(input)
ma = max(pred)
mi = min(pred)
pred = (pred - mi) / (ma - mi)
```

## Build

```
export AILIA_LIBRARY_PATH=../ailia/library
cmake .
make
```

## Run

```
./u2net.sh -o 11 -v
```

## Arguments

[Check Python version](https://github.com/ailia-ai/ailia-models/tree/master/image_segmentation/u2net)
