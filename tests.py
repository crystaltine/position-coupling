from src.data.addition import AdditionDatasetWithCoupledPositions
from src.data.varied_arithmetic import VariedDatasetWithCoupledPositions

gen = AdditionDatasetWithCoupledPositions(1, 5, 10, padding=True, pad_token='_')
vargen = VariedDatasetWithCoupledPositions(1, 5, 10, pattern='+-+', padding=True, pad_token='_', reverse_output=True)

inputs, labels, input_positions, label_positions = gen[0]
print(f"inputs: {inputs} (type: {type(inputs)})")
print(f"labels: {labels} (type: {type(labels)})")
print(f"input_positions: {input_positions} (type: {type(input_positions)})")
print(f"label_positions: {label_positions} (type: {type(label_positions)})")

print("\n" + "="*50 + "\n")
inputs, labels, input_positions, label_positions = vargen[0]
print(f"inputs: {inputs} (type: {type(inputs)})")
print(f"labels: {labels} (type: {type(labels)})")
print(f"input_positions: {input_positions} (type: {type(input_positions)})")
print(f"label_positions: {label_positions} (type: {type(label_positions)})")