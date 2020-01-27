import torch
import numpy as np
import torch.nn.functional as F


def encode_binary(num):
	binary_rep = [int(i) for i in list('{0:0b}'.format(num))]
	for i in range(len(binary_rep) + 1, 11):
		binary_rep.insert(0, 0)
	return binary_rep


def generate_training_set():
	train_X = []
	train_Y = []

	for i in range(101, 1001):
		train_X.append(encode_binary(i))
		if i % 3 == 0 and i % 5 == 0:
			train_Y.append([0, 0, 1, 0])
		elif i % 3 == 0:
			train_Y.append([1, 0, 0, 0])
		elif i % 5 == 0:
			train_Y.append([0, 1, 0, 0])
		else:
			train_Y.append([0, 0, 0, 1])

	return torch.tensor(train_X), torch.tensor(train_Y)


def model(x, weight1, bias1, weight2, bias2):
	a1 = torch.matmul(x.float(), weight1) + bias1
	h1 = a1.sigmoid()
	a2 = torch.matmul(h1, weight2) + bias2
	h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)
	return h2


def accuracy(y_hat, y):
	acc = (torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).float().mean().item()
	return acc


def train_model():
	torch.manual_seed(0)

	weight1 = np.random.randn(10, 10)
	weight1 = torch.from_numpy(weight1)
	weight1.requires_grad_()
	bias1 = torch.zeros(10, requires_grad=True)

	weight2 = np.random.randn(10, 4)
	weight2 = torch.from_numpy(weight2)
	weight2.requires_grad_()
	bias2 = torch.zeros(4, requires_grad=True)

	x, y = generate_training_set()

	learning_rate = 8
	loss_arr = []
	accuracy_arr = []
	loss = 1000
	epochs = 0

	while loss > 0.045:
		epochs += 1
		if epochs > 4000:
			learning_rate = 3

		y_hat = model(x, weight1.float(), bias1, weight2.float(), bias2)
		accuracy_arr.append(accuracy(y_hat, y))
		loss = F.mse_loss(y_hat.float(), y.float())
		loss_arr.append(loss.item())
		loss.backward()
		with torch.no_grad():
			weight1 -= learning_rate * weight1.grad
			bias1 -= learning_rate * bias1.grad
			weight2 -= learning_rate * weight2.grad
			bias2 -= learning_rate * bias2.grad
			weight1.grad.zero_()
			bias1.grad.zero_()
			weight2.grad.zero_()
			bias2.grad.zero_()

	print("Epochs: ", epochs)
	print("Accuracy before training: ", accuracy_arr[0]*100, '%')
	print("Accuracy after training: ", accuracy_arr[-1], '%')


train_model()
