# Alzheimer's Image Classification Demo

Browser-based TensorFlow.js prototype for classifying brain scan imagery into four dementia-related labels: `MildDemented`, `ModerateDemented`, `NonDemented`, and `VeryMildDemented`.

This project is a compact example of taking a trained image model, exporting it to TensorFlow.js, and wrapping it in a simple client-side interface that can run without a backend inference server.

## What It Shows

- TensorFlow.js graph model served directly from static files
- Four-class image classification UI with local image upload
- Model weights split into browser-loadable shard files
- Lightweight HTML/JavaScript inference flow
- Early ML experimentation around healthcare image classification

## Model Summary

| Item | Detail |
| --- | --- |
| Runtime | TensorFlow.js in the browser |
| Model format | `graph-model` |
| Input | RGB image tensor, dynamic height and width, 3 channels |
| Output | 4 class scores |
| Classes | MildDemented, ModerateDemented, NonDemented, VeryMildDemented |
| Weight shards | 4 local `.bin` files |

The original project notes reported an average detection result around 90 percent in project testing, with class-level precision notes:

| Class | Reported precision |
| --- | ---: |
| NonDemented | 65% |
| MildDemented | 94.40% |
| VeryMildDemented | 97.93% |
| ModerateDemented | 100.00% |

## Run Locally

The model must be served through HTTP so TensorFlow.js can load `model.json` and the weight shards.

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

Use the bundled sample image or upload a similar top-down brain image, then click `predict`.

## Repository Layout

```text
index.html              Browser UI and inference code
model.json              TensorFlow.js graph model manifest
classes.json            Class-to-index mapping
group1-shard*.bin       Model weight shards
image.jpg               Sample input image
```

## Verification

Current automated checks performed:

- Parsed `model.json` and `classes.json`
- Confirmed all 4 weight shards referenced by `model.json` exist
- Served `index.html`, `model.json`, `classes.json`, and all weight shards over local HTTP

## Limitations

This is a research and portfolio prototype, not a clinical system or medical device. The repository does not include the original training dataset, a reproducible training pipeline, or independent validation artifacts. Any real healthcare use would require dataset documentation, privacy review, external validation, bias evaluation, and clinician oversight.

## Next Improvements

- Add a model card with dataset source, training split, preprocessing, and evaluation method
- Add confusion matrix, recall, F1, and per-class support counts
- Normalize input preprocessing so uploaded images match training conditions
- Add confidence bars instead of returning only the top class
- Add a small test harness that verifies TensorFlow.js inference against the sample image
