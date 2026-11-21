import kagglehub

# Download dataset from Kaggle API
path = kagglehub.dataset_download(
    "masoudnickparvar/brain-tumor-mri-dataset"
)

print("Dataset downloaded successfully!")
print("Dataset Path:", path)