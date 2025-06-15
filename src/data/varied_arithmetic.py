import torch
import re
from src.data.common import ArithmeticDataset

class VariedDatasetWithCoupledPositions(ArithmeticDataset):    
	"""
	generates datasets with expressions like
	1234 + 4567 * 3491 - 2834.

	No division for now since it might result in non-integer outputs.
	Also no chained multiplication since we cant have outputs getting super long.
	"""

	def __init__(self,
	n_data,
	min_n_digits,
	max_n_digits,
	pattern: str, # eg '+*-*' will generate something like 324 + 78 * 456 - 123 * 90
	reverse_input=False,
	reverse_output=False,
	padding=False,
	pad_token='0',
	randomize=True,
	max_position=200,
	**kwargs):  

		super().__init__()      
		
		# validate pattern
		last_was_mult = False
		for c in pattern:
			if c not in ['+', '-', '*']:
				raise ValueError(f"Invalid character '{c}' in pattern. Only '+', '-', '*', '/' are allowed.")
			if c == '*':
				if last_was_mult:
					raise ValueError("Consecutive '*' operators are not allowed in the pattern.")
				last_was_mult = True
			else:
				last_was_mult = False

		self.min_n_digits = min_n_digits
		self.max_n_digits = max_n_digits
		self.reverse_input = reverse_input
		self.reverse_output = reverse_output
		self.randomize = randomize
		self.max_position = max_position
		self.pattern = pattern
		self.pad_token = pad_token
		print(f"\x1b[33mnote: padding={padding}")
		self.padding = True
		
		# generate inputs and labels
		for _ in range(n_data):
			first_num = self._generate_n_digit_num(self.min_n_digits, self.max_n_digits) if pattern[0] != "*" else self._generate_n_digit_num(1, 2)
			first_num_padded_maybe = first_num.rjust(self.max_n_digits, self.pad_token) if self.padding else first_num
			
			evalable_expr = str(first_num) # same as expression but without padding - allows for using eval() func
			expression = str(first_num_padded_maybe)
			
			for i, op in enumerate(pattern):
				expression += op
				evalable_expr += op
				
				# generate another number
				# if its involved with any multiplication at all, make sure it only has 1 or 2 digits
				new_num = self._generate_n_digit_num(self.min_n_digits, self.max_n_digits)
				
				if op == '*' or (i != len(pattern) - 1 and pattern[i+1] == '*'):
					new_num = self._generate_n_digit_num(1, 2)

				evalable_expr += new_num

				if self.padding:
					new_num = new_num.rjust(self.max_n_digits, self.pad_token)

				expression += new_num

			self.inputs.append(expression)
			self.labels.append(evalable_expr)

		# solve the expressions (also add plus in case of positive, helps w/ padding/position consistency)
		# self.labels = [f"{eval(evalable_expr):+}" for evalable_expr in self.inputs]

	def _generate_n_digit_num(self, min_n_digits: int, max_n_digits: int) -> str:
		n_digits = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(1,)).item()
		# uniform sampling of a number
		if n_digits == 1:
				num = str(torch.randint(0, 10, size=(1,)).item())
		else:
				num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
				# num = int(''.join(map(str, num_arr)))
				num = ''.join(map(str, num_arr))
		return num

	def __getitem__(self, index):
		inp = self.inputs[index]
		label = self.labels[index]

		input_nums = re.split(r'[+\-*]', inp)
		max_digits_in_input = max(len(num) for num in input_nums)
	
		input_num_positions = []

		# no ned for this anytmore, padding is done upon object construction
		# if self.padding:
		# 	# pad each input number to max_digits
		# 	for j in range(len(input_nums)):
		# 		if len(input_nums[j]) < max_digits_in_input:
		# 			input_nums[j] = input_nums[j].rjust(max_digits_in_input, self.pad_token)

		# generate positions
		start_pos_offset = max(max_digits_in_input, len(label))
		start = 1 if not self.randomize else torch.randint(1, self.max_position-start_pos_offset+(1 if self.reverse_output else 0), size=(1,)).item()
		
		# print(f"Varied.getitem: input nums are: {input_nums}, patterm: {self.pattern}")

		for n in input_nums:
			# generate positions for each number in the input expr
			num_positions = list(range(start+1, start+1 + len(n)))
			input_num_positions.append(num_positions)

		# print(f"Varied.getitem: input_num_positions are: {input_num_positions}")

		if not self.reverse_input: # reverse all the positions
			# refer to addition.py - the positions are already reversed so re-reverse them if reverse_input==False
			input_num_positions = [list(reversed(positions)) for positions in input_num_positions]
		
		# print("Varied.getitem: after reverse_input check, input_num_positions are: ", input_num_positions)

		# piece all input positions (and positions of the operators) into one long arr
		final_input_positions = []
		for positions in input_num_positions:
			final_input_positions.extend(positions)
			final_input_positions.append(start) # the operator

		# remove extra operator position
		final_input_positions.pop()

		ans_positions = list(range(start+1, start+1+len(label)))
		if self.reverse_output:
				eq_position = start
		else:
				ans_positions = ans_positions[::-1]
				eq_position = start+len(label)+1
		label_positions = [eq_position] + ans_positions

		return " ".join(inp), " ".join(label), final_input_positions, label_positions