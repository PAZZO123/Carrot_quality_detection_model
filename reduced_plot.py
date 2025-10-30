import matplotlib.pyplot as plt

# Counts
original = 1003
cleaned = 74
augmented = 74 * 3  # 3 augmentations per image

categories = ['Original Data', 'Cleaned Data', 'Augmented Data']
values = [original, cleaned, augmented]

# Bar chart
plt.figure(figsize=(8,5))
bars = plt.bar(categories, values, color=['skyblue', 'orange', 'green'])
plt.title("Dataset Variation Across Stages")
plt.ylabel("Number of Images")

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, yval, ha='center', va='bottom')

plt.show()
