#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer.max_epochs=5

python src/train.py trainer.max_epochs=10

python src/train.py trainer.max_epochs=15

python src/train.py trainer.max_epochs=20

python src/train.py trainer.max_epochs=25

python src/train.py trainer.max_epochs=30
