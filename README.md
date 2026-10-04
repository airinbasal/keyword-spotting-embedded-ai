# Keyword Spotting for Edge Deployment

A complete keyword spotting pipeline 
built for ARM Cortex-M deployment using 
TensorFlow Lite, designed around embedded 
constraints.

## What it does
Detects spoken keywords (yes, no, stop, 
right, silence) from audio in real time, 
optimized to run on microcontrollers with 
minimal memory and power.

## Pipeline
1. **Data** — Google Speech Commands dataset
   (85,000+ audio samples)
2. **Preprocessing** — Audio converted to 
   Mel spectrograms (treats sound as images)
3. **Model** — Small CNN with 
   GlobalAveragePooling (60K parameters)
4. **Compression** — TFLite quantization 
   (int8)
5. **Deployment target** — ARM Cortex-M 
   via TFLite Micro

## Results
| Version | Size | Accuracy |
|---------|------|----------|
| Original (Flatten CNN) | 12.0 MB | - |
| Optimized CNN (GAP) | 235.5 KB | 85.5% |
| TFLite Quantized | 67.5 KB | 62.0% |

**182x size reduction** from initial 
architecture to final quantized model.

Note: Accuracy drop from quantization 
is expected with basic post-training 
quantization — quantization-aware training 
(QAT) would reduce this to ~1-3% drop.

## Key Engineering Decisions
- **GlobalAveragePooling** instead of 
  Flatten — reduced parameters from 
  3.1M to 60K (52x smaller)
- **Int8 quantization** — 3.5x additional 
  size reduction
- **Mel spectrograms** — converts 1D audio 
  to 2D images for CNN processing
- **6-class design** — includes unknown 
  class to reduce false triggers on 
  always-on devices

## Tech Stack
- Python
- TensorFlow / Keras
- TensorFlow Lite
- Google Colab

## Target Hardware
ARM Cortex-M microcontrollers via 
TFLite Micro runtime

## Status
- Data pipeline complete  
- Model trained (85.5% accuracy)  
- TFLite conversion complete  
- Quantization applied  
- Working on Quantization-aware training  


## Related Work
This project extends my undergraduate 
IEEE-published research on real-time 
EEG-based emotion detection on 
microcontrollers (IEEE RAICS 2025).
