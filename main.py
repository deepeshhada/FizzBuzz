from model.model import *
import argparse
import torch


def generate_test_input(path):
	test_input = []
	f = open(path, "r")
	for line in f:
		test_input.append(int(line.rstrip('\n')))
	f.close()
	return test_input


def encode_binary(num):
	binary_rep = [int(i) for i in list('{0:0b}'.format(num))]
	for i in range(len(binary_rep) + 1, 11):
		binary_rep.insert(0, 0)
	return binary_rep


def software1():
	output_list = []

	for i in input_list:
		if i % 3 == 0 and i % 5 == 0:
			output_list.append('fizzbuzz')
		elif i % 3 == 0:
			output_list.append('fizz')
		elif i % 5 == 0:
			output_list.append('buzz')
		else:
			output_list.append(str(i))

	with open('Software1.txt', 'w') as f:
		for i in range(len(output_list)):
			if i != len(output_list) - 1:
				f.write("%s\n" % output_list[i])
			else:
				f.write("%s" % output_list[i])

	return output_list


def software2():
	output_list = []

	for i in input_list:
		test_x = torch.tensor(encode_binary(i))
		test_x = test_x.view(1, -1)
		y_hat = model_output(test_x, weight1.float(), bias1, weight2.float(), bias2)
		max_index = torch.argmax(y_hat, dim=1)

		if max_index == 0:
			output_list.append('fizz')
		elif max_index == 1:
			output_list.append('buzz')
		elif max_index == 2:
			output_list.append('fizzbuzz')
		else:
			output_list.append(str(i))

	with open('Software2.txt', 'w') as f:
		for i in range(len(output_list)):
			if i != len(output_list) - 1:
				f.write("%s\n" % output_list[i])
			else:
				f.write("%s" % output_list[i])

	return output_list


parser = argparse.ArgumentParser()
parser.add_argument("-td", "--test-data", help="Test file path")
args = parser.parse_args()

input_list = generate_test_input(args.test_data)

software1()
software2()
