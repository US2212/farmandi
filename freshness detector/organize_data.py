import os
import shutil


source_path = r"C:\Users\dhaarna goyal\Downloads\archive (5)\Dataset\train"
dest_path = r"C:\Users\dhaarna goyal\OneDrive\Desktop\FreshnessDetector"

os.makedirs(dest_path, exist_ok=True)
classes = {'fresh': ['fresh', 'good'], 'rotten': ['rotten'], 'medium': []}

for freshness in classes:
    os.makedirs(os.path.join(dest_path, freshness), exist_ok=True)

counter = {'fresh': 0, 'medium': 0, 'rotten': 0}

for folder in os.listdir(source_path):
    folder_path = os.path.join(source_path, folder)
    if not os.path.isdir(folder_path):
        continue

    category = None
    for freshness, keywords in classes.items():
        if any(keyword in folder.lower() for keyword in keywords):
            category = freshness
            break
    if category is None:
        category = 'medium'  # Default to medium if not found

    for filename in os.listdir(folder_path):
        src = os.path.join(folder_path, filename)
        dst = os.path.join(dest_path, category, f"{category}_{counter[category]}.jpg")
        shutil.copy(src, dst)
        counter[category] += 1

print("Dataset organized into Fresh, Medium, and Rotten.")

dest_path = r"C:\Users\dhaarna goyal\OneDrive\Desktop\FreshnessDetector"

os.makedirs(dest_path, exist_ok=True)
classes = {'fresh': ['fresh', 'good'], 'rotten': ['rotten'], 'medium': []}

for freshness in classes:
    os.makedirs(os.path.join(dest_path, freshness), exist_ok=True)

counter = {'fresh': 0, 'medium': 0, 'rotten': 0}

for folder in os.listdir(source_path):
    folder_path = os.path.join(source_path, folder)
    if not os.path.isdir(folder_path):
        continue

    category = None
    for freshness, keywords in classes.items():
        if any(keyword in folder.lower() for keyword in keywords):
            category = freshness
            break
    if category is None:
        category = 'medium'  # Default to medium if not found

    for filename in os.listdir(folder_path):
        src = os.path.join(folder_path, filename)
        dst = os.path.join(dest_path, category, f"{category}_{counter[category]}.jpg")
        shutil.copy(src, dst)
        counter[category] += 1

print("Dataset organized into Fresh, Medium, and Rotten.")
