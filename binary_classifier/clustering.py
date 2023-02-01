import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from clustering_ae import Clustering, utils
from torchvision.models import vgg16, VGG16_Weights

def create_cluster_model(num_clusters, epochs=500):
	cae = Clustering(num_clusters=num_clusters,
					 n_init=5,
					 epochs=epochs,
					 tol=1e-5,
					 initialization="k-means++"
					)
	return cae

def retrieve_flat_data(poisoned_train_loader, train_loader, test_loader, save_file):
	if os.path.exists(save_file):
		print("Loading flattened data from saved location...")
		data = np.load(save_file)
		return data["flat_train_imgs"], data["flat_train_lbls"], data["flat_test_imgs"], data["flat_test_lbls"], data["flat_true_lbls"]
	train_imgs, train_lbls = [], []
	test_imgs, test_lbls = [], []
	true_lbls = []
	for imgs, lbls in tqdm(poisoned_train_loader, desc="Flattening poisoned training image"):
		train_imgs.extend(imgs.view(imgs.shape[0], -1).numpy())
		train_lbls.extend(lbls.cpu().numpy())
	for images, labels in tqdm(train_loader, desc="Retrieve true training labels"):
		true_lbls.extend(labels.cpu().numpy())
	for imgs, lbls in tqdm(test_loader, desc="Flattening testing image"):
		test_imgs.extend(imgs.view(imgs.shape[0], -1).numpy())
		test_lbls.extend(lbls.cpu().numpy())
	# Save the flattened data
	print("Saving flattened data to disk...")
	np.savez(save_file,
			 flat_train_imgs=np.array(train_imgs),
			 flat_train_lbls=np.array(train_lbls),
			 flat_test_imgs=np.array(test_imgs),
			 flat_test_lbls=np.array(test_lbls),
			 flat_true_lbls=np.array(true_lbls))
	return np.array(train_imgs), np.array(train_lbls), np.array(test_imgs), np.array(test_lbls), np.array(true_lbls)

def retrieve_feature_data(poisoned_train_loader, train_loader, test_loader, device, save_file):
	if os.path.exists(save_file):
		print("Loading feature data from saved location...")
		data = np.load(save_file)
		return data["feature_train_imgs"], data["feature_train_lbls"], data["feature_test_imgs"], data["feature_test_lbls"], data["feature_true_lbls"]
	
	# Initialize the model
	model = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).to(device)
	# Set the model to evaluation mode
	model.eval()
	# Initialize the lists to store the feature data
	train_imgs, train_lbls = [], []
	test_imgs, test_lbls = [], []
	true_lbls = []
	with torch.no_grad():
		for images, labels in tqdm(poisoned_train_loader, desc="Retrieve poisoned training image features"):
			# Convert the single channel images to 3 channel
			images = torch.repeat_interleave(images.to(device), 3, dim=1)
			images = F.interpolate(images, size=(32, 32), mode='nearest')
			features_batch = model.features(images).squeeze()
			train_imgs.append(features_batch.cpu().detach().numpy())
			train_lbls.extend(labels.cpu().numpy())
	for images, labels in tqdm(train_loader, desc="Retrieve true training labels"):
		true_lbls.extend(labels.cpu().numpy())
	with torch.no_grad():
		for images, labels in tqdm(test_loader, desc="Retrieve testing image features"):
			# Convert the single channel images to 3 channel
			images = torch.repeat_interleave(images.to(device), 3, dim=1)
			images = F.interpolate(images, size=(32, 32), mode='nearest')
			features_batch = model.features(images).squeeze()
			test_imgs.append(features_batch.cpu().detach().numpy())
			test_lbls.extend(labels.cpu().numpy())
	# Save the feature data
	print("Saving feature data to disk...")
	np.savez(save_file,
			 feature_train_imgs=np.concatenate(train_imgs),
			 feature_train_lbls=np.array(train_lbls),
			 feature_test_imgs=np.concatenate(test_imgs),
			 feature_test_lbls=np.array(test_lbls),
			 feature_true_lbls=np.array(true_lbls))
	return np.concatenate(train_imgs), np.array(train_lbls), np.concatenate(test_imgs), np.array(test_lbls), np.array(true_lbls)

def train_cluster(poisoned_train_loader, train_loader, test_loader, classes, model_type, device, save_dir, display_cluster=True, n_components=10):
	if model_type == "flat":
		train_imgs, train_lbls, test_imgs, test_lbls, true_lbls = retrieve_flat_data(poisoned_train_loader, train_loader, test_loader, os.path.join(save_dir, "flat_features.npz"))
	elif model_type == "feature":
		train_imgs, train_lbls, test_imgs, test_lbls, true_lbls = retrieve_feature_data(poisoned_train_loader, train_loader, test_loader, device, os.path.join(save_dir, "vgg16_features.npz"))
	else:
		raise ValueError(f"Unknown model_type argument: {model_type}. Has to be either \"flat\" or \"feature\".")

	print("Clustering the data...")
	# Create the cluster model
	cae = create_cluster_model(num_clusters=len(classes))
	train_imgs_PCA, test_imgs_PCA = utils.encodePCA(train_imgs, test_imgs, n_components-1)
	cae.train(train_imgs_PCA)
	train_benchmark = cae.benchmark(f'Fashion MNIST - {model_type} {n_components} Components - Train Data', train_imgs_PCA, train_lbls)
	test_benchmark = cae.benchmark(f'Fashion MNIST - {model_type} {n_components} Components - Test Data', test_imgs_PCA, test_lbls)
	print(f'Fashion MNIST - {model_type} {n_components} Components - ACCURACY: {utils.clustering_accuracy(true_lbls, cae.model.labels_)}')
	if display_cluster:
		utils.plot(train_imgs_PCA, cae.model.labels_, "2d")
	return cae, train_benchmark, test_benchmark