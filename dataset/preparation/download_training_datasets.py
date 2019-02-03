import os
import subprocess

def download_demon_datasets():
	DATASET_NAMES = ["sun3d", "rgbd", "mvs", "scenes11"]
	for dataset_name in DATASET_NAMES:
		if not os.path.exists("{:}_train.tgz".format(dataset_name)):
			print("Downloading {:} dataset...".format(dataset_name))
			subprocess.call(
					"wget https://lmb.informatik.uni-freiburg.de/data/demon/traindata/{:}_train.tgz ;".format(dataset_name) +
					"tar -xvzf {:}_train.tgz ;".format(dataset_name),
					shell = True
				)

if __name__ == "__main__":
	download_demon_datasets()
