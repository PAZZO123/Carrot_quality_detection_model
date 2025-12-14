import matplotlib.pyplot as plt

# Counts
original_d = 1077
duplicates = 1003
cleaned = 74
augmented = 74 * 3  # 3 augmentations per image

categories = ['Original Data', 'Duplicates Data', 'Cleaned Data', 'Augmented Data']
values = [original_d, duplicates, cleaned, augmented]

# Bar chart
plt.figure(figsize=(8,5))
bars = plt.bar(categories, values, color=['green', 'red', 'orange', 'blue'])
plt.title("Dataset Variation Across Stages")
plt.ylabel("Number of Images")

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, yval, ha='center', va='bottom')

# Save as image
plt.savefig("dataset_variation.png", dpi=300, bbox_inches='tight')

# Show plot
plt.show()
