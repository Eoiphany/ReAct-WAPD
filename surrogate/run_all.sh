#!/bin/bash

echo "===== USC: U-Net ====="
python autodl-tmp/surrogate/train_usc_surrogate.py \
  --model-type unet \
  --data-root autodl-tmp/usc/ \
  --output-root autodl-tmp/surrogate/runs

echo "===== USC: TransUNet ====="
python autodl-tmp/surrogate/train_usc_surrogate.py \
  --model-type transunet \
  --data-root autodl-tmp/usc/ \
  --output-root autodl-tmp/surrogate/runs

echo "===== USC: RadioUNet ====="
python autodl-tmp/surrogate/train_usc_surrogate.py \
  --model-type radiounet \
  --data-root autodl-tmp/usc/ \
  --output-root autodl-tmp/surrogate/runs

echo "===== USC: PMNet ====="
python autodl-tmp/surrogate/train_usc_surrogate.py \
  --model-type pmnet \
  --data-root autodl-tmp/usc/ \
  --output-root autodl-tmp/surrogate/runs

echo "===== USC: RMNet ====="
python autodl-tmp/surrogate/train_usc_surrogate.py \
  --model-type rmnet \
  --data-root autodl-tmp/usc/ \
  --output-root autodl-tmp/surrogate/runs

echo "All training finished. Shutting down..."
shutdown -h now
