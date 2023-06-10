python -m hw2.experiments run-exp -K 32 -L 8 -P 8 -H 100 -M resnet --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4
python -m hw2.experiments run-exp -K 32 -L 16 -P 8 -H 100 -M resnet --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4
python -m hw2.experiments run-exp -K 32 -L 32 -P 8 -H 100 -M resnet --bs-train 50 --batches 10 --early-stopping 5 --run-name exp1_4