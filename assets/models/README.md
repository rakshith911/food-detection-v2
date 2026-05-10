# Food precheck ONNX model

Place the production model at:

```txt
assets/models/food_precheck.onnx
```

Current model:

- Source: `mrdbourke/food-not-food-classifier-nextvit-v2`
- Weights: `model_best_fv.safetensors`
- License: Apache-2.0
- Exported architecture: `nextvit_small.bd_ssld_6m_in1k_384`

Expected contract:

- Task: binary image classification
- Classes: `food_or_drink`, `not_food_or_drink`
- Input: RGB float32 tensor, NCHW, shape `1x3x384x384`
- Preprocessing: resize to 384x384, scale to `[0, 1]`, normalize with ImageNet mean/std
- Output: two logits or probabilities in class order `[food, not_food]`

The app rejects images when this model is missing or cannot run.
