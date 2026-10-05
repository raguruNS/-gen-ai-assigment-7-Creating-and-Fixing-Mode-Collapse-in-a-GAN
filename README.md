# Assignment 7 — Creating and Fixing Mode Collapse in a GAN

## 📌 Project Overview

This project demonstrates **mode collapse in Generative Adversarial Networks (GANs)** using a DCGAN trained on the **Fashion-MNIST dataset**.

The experiment intentionally makes the GAN unstable by modifying the generator architecture and training parameters. The resulting output is analyzed for mode collapse. A technique called **Label Smoothing** is then applied to improve training stability and increase the diversity of generated images.

---

## 🎯 Objective

The main objectives of this assignment are:

- Understand the concept of GAN mode collapse.
- Modify a working DCGAN to make its training unstable.
- Observe the effect of unstable training on generated images.
- Identify similar or nearly identical generated outputs.
- Apply a technique to reduce mode collapse.
- Compare the results before and after improvement.

---

## 🧠 What is Mode Collapse?

**Mode collapse** is a common problem in GAN training where the generator produces a limited variety of outputs instead of learning the complete distribution of the training data.

For example, when training on Fashion-MNIST, the generator should produce different types of clothing such as:

- Shirts
- Shoes
- Bags
- Trousers
- Coats
- Dresses

During mode collapse, the generator may repeatedly produce very similar images.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab
- Fashion-MNIST Dataset

---

## 📊 Dataset

The **Fashion-MNIST** dataset was used for this experiment.

It contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Grayscale images
- 10 clothing categories

The images were normalized to the range:

```text
[-1, 1]
```

This matches the `tanh` activation used in the generator's output layer.

---

# 🔴 Part 1 — Intentionally Creating Mode Collapse

The working DCGAN was deliberately modified to make training unstable.

### Modifications

The following changes were made:

1. Batch Normalization was removed from the generator.
2. Generator capacity was reduced.
3. The learning rate was increased significantly.
4. The model was trained for approximately 20 epochs.

### Learning Rate

The unstable model used:

```text
0.01
```

This is significantly higher than the typical learning rate used for stable DCGAN training.

### Expected Result

The unstable model may produce images with:

- Low diversity
- Similar shapes
- Repeated patterns
- Nearly identical outputs

This behavior demonstrates **mode collapse**.

---

# 🟢 Part 2 — Fixing Mode Collapse

To improve the unstable model, **Label Smoothing** was used.

Instead of assigning real images a label of:

```text
1.0
```

the real label was changed to:

```text
0.9
```

The labels used were:

```text
Real = 0.9
Fake = 0.0
```

This prevents the discriminator from becoming excessively confident and can provide more useful gradients to the generator.

---

## Additional Improvements

The following improvements were also made:

### 1. Batch Normalization

Batch Normalization was restored in the generator to improve training stability.

### 2. Learning Rate

The learning rate was reduced from:

```text
0.01
```

to:

```text
0.0002
```

### 3. Label Smoothing

Real labels were changed from:

```text
1.0
```

to:

```text
0.9
```

---

# 📈 Results

## Before Fixing

The intentionally unstable GAN showed reduced diversity in its generated images.

Some images were very similar to each other, demonstrating the effect of mode collapse.

### Screenshot

Save the screenshot as:

```text
mode_collapse_before.png
```

**Figure 1: Generated images before reducing mode collapse.**

---

## After Fixing

After applying label smoothing, restoring Batch Normalization, and using a lower learning rate, the generator produced more diverse outputs.

### Screenshot

Save the screenshot as:

```text
mode_collapse_after.png
```

**Figure 2: Generated images after reducing mode collapse.**

---

# 🔍 Comparison

| Feature | Before Fix | After Fix |
|---|---|---|
| Batch Normalization | Removed | Restored |
| Generator Capacity | Reduced | Increased |
| Learning Rate | 0.01 | 0.0002 |
| Label Smoothing | No | Yes |
| Training Stability | Low | Improved |
| Image Diversity | Low | Higher |
| Mode Collapse | More noticeable | Reduced |

---

# 💡 Explanation

The unstable GAN had difficulty learning the full distribution of the Fashion-MNIST dataset because the generator was deliberately weakened and the learning rate was made excessively high.

As training progressed, the generator could find a limited set of outputs that were effective at fooling the discriminator. Instead of generating many different types of clothing, it repeatedly generated similar patterns.

This is known as **mode collapse**.

Label smoothing was then introduced by changing the real label from `1.0` to `0.9`. This reduced the discriminator's tendency to become overly confident and helped provide more useful gradients to the generator.

Restoring Batch Normalization and reducing the learning rate further improved the stability of the GAN.

---

# 🧪 Training Configuration

### Unstable Model

```text
Dataset: Fashion-MNIST
Epochs: 20
Batch Size: 128
Latent Dimension: 100
Learning Rate: 0.01
Batch Normalization: Removed
Label Smoothing: No
```

### Improved Model

```text
Dataset: Fashion-MNIST
Epochs: 20
Batch Size: 128
Latent Dimension: 100
Learning Rate: 0.0002
Batch Normalization: Enabled
Label Smoothing: 0.9
```

---

# 📁 Project Structure

```text
Assignment-7-Mode-Collapse-GAN/
│
├── Assignment_7_Mode_Collapse_GAN.ipynb
│
├── mode_collapse_before.png
│
├── mode_collapse_after.png
│
├── broken_outputs/
│   ├── epoch_1.png
│   ├── epoch_2.png
│   ├── ...
│   └── epoch_20.png
│
├── fixed_outputs/
│   ├── epoch_1.png
│   ├── epoch_2.png
│   ├── ...
│   └── epoch_20.png
│
└── README.md
```

---

# ▶️ How to Run

### Step 1

Open the notebook in **Google Colab**.

### Step 2

Run the cells in order.

### Step 3

Load the Fashion-MNIST dataset.

### Step 4

Train the intentionally unstable GAN for approximately 20 epochs.

### Step 5

Observe and save the mode-collapse output.

### Step 6

Apply label smoothing and restore stable training settings.

### Step 7

Retrain the GAN.

### Step 8

Compare the generated images before and after the improvement.

---

# ✅ Conclusion

This experiment demonstrated the effect of **mode collapse in GANs** and showed how unstable architecture and optimization settings can negatively affect GAN training.

The unstable model produced less diverse outputs. After applying **label smoothing**, restoring Batch Normalization, and reducing the learning rate, the generated images became more diverse and the training became more stable.

Therefore, appropriate architecture and optimization techniques are important for achieving stable GAN training and reducing mode collapse.

---

## 👨‍💻 Assignment

**Assignment:** 7 — Creating and Fixing Mode Collapse in a GAN

**Model:** DCGAN

**Dataset:** Fashion-MNIST

**Framework:** TensorFlow / Keras

**Platform:** Google Colab

**Technique Used:** Label Smoothing# Assignment 7 — Creating and Fixing Mode Collapse in a GAN

## 📌 Project Overview

This project demonstrates **mode collapse in Generative Adversarial Networks (GANs)** using a DCGAN trained on the **Fashion-MNIST dataset**.

The experiment intentionally makes the GAN unstable by modifying the generator architecture and training parameters. The resulting output is analyzed for mode collapse. A technique called **Label Smoothing** is then applied to improve training stability and increase the diversity of generated images.

---

## 🎯 Objective

The main objectives of this assignment are:

- Understand the concept of GAN mode collapse.
- Modify a working DCGAN to make its training unstable.
- Observe the effect of unstable training on generated images.
- Identify similar or nearly identical generated outputs.
- Apply a technique to reduce mode collapse.
- Compare the results before and after improvement.

---

## 🧠 What is Mode Collapse?

**Mode collapse** is a common problem in GAN training where the generator produces a limited variety of outputs instead of learning the complete distribution of the training data.

For example, when training on Fashion-MNIST, the generator should produce different types of clothing such as:

- Shirts
- Shoes
- Bags
- Trousers
- Coats
- Dresses

During mode collapse, the generator may repeatedly produce very similar images.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab
- Fashion-MNIST Dataset

---

## 📊 Dataset

The **Fashion-MNIST** dataset was used for this experiment.

It contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Grayscale images
- 10 clothing categories

The images were normalized to the range:

```text
[-1, 1]
```

This matches the `tanh` activation used in the generator's output layer.

---

# 🔴 Part 1 — Intentionally Creating Mode Collapse

The working DCGAN was deliberately modified to make training unstable.

### Modifications

The following changes were made:

1. Batch Normalization was removed from the generator.
2. Generator capacity was reduced.
3. The learning rate was increased significantly.
4. The model was trained for approximately 20 epochs.

### Learning Rate

The unstable model used:

```text
0.01
```

This is significantly higher than the typical learning rate used for stable DCGAN training.

### Expected Result

The unstable model may produce images with:

- Low diversity
- Similar shapes
- Repeated patterns
- Nearly identical outputs

This behavior demonstrates **mode collapse**.

---

# 🟢 Part 2 — Fixing Mode Collapse

To improve the unstable model, **Label Smoothing** was used.

Instead of assigning real images a label of:

```text
1.0
```

the real label was changed to:

```text
0.9
```

The labels used were:

```text
Real = 0.9
Fake = 0.0
```

This prevents the discriminator from becoming excessively confident and can provide more useful gradients to the generator.

---

## Additional Improvements

The following improvements were also made:

### 1. Batch Normalization

Batch Normalization was restored in the generator to improve training stability.

### 2. Learning Rate

The learning rate was reduced from:

```text
0.01
```

to:

```text
0.0002
```

### 3. Label Smoothing

Real labels were changed from:

```text
1.0
```

to:

```text
0.9
```

---

# 📈 Results

## Before Fixing

The intentionally unstable GAN showed reduced diversity in its generated images.

Some images were very similar to each other, demonstrating the effect of mode collapse.

### Screenshot

Save the screenshot as:

```text
mode_collapse_before.png
```

**Figure 1: Generated images before reducing mode collapse.**

---

## After Fixing

After applying label smoothing, restoring Batch Normalization, and using a lower learning rate, the generator produced more diverse outputs.

### Screenshot

Save the screenshot as:

```text
mode_collapse_after.png
```

**Figure 2: Generated images after reducing mode collapse.**

---

# 🔍 Comparison

| Feature | Before Fix | After Fix |
|---|---|---|
| Batch Normalization | Removed | Restored |
| Generator Capacity | Reduced | Increased |
| Learning Rate | 0.01 | 0.0002 |
| Label Smoothing | No | Yes |
| Training Stability | Low | Improved |
| Image Diversity | Low | Higher |
| Mode Collapse | More noticeable | Reduced |

---

# 💡 Explanation

The unstable GAN had difficulty learning the full distribution of the Fashion-MNIST dataset because the generator was deliberately weakened and the learning rate was made excessively high.

As training progressed, the generator could find a limited set of outputs that were effective at fooling the discriminator. Instead of generating many different types of clothing, it repeatedly generated similar patterns.

This is known as **mode collapse**.

Label smoothing was then introduced by changing the real label from `1.0` to `0.9`. This reduced the discriminator's tendency to become overly confident and helped provide more useful gradients to the generator.

Restoring Batch Normalization and reducing the learning rate further improved the stability of the GAN.

---

# 🧪 Training Configuration

### Unstable Model

```text
Dataset: Fashion-MNIST
Epochs: 20
Batch Size: 128
Latent Dimension: 100
Learning Rate: 0.01
Batch Normalization: Removed
Label Smoothing: No
```

### Improved Model

```text
Dataset: Fashion-MNIST
Epochs: 20
Batch Size: 128
Latent Dimension: 100
Learning Rate: 0.0002
Batch Normalization: Enabled
Label Smoothing: 0.9
```

---

# 📁 Project Structure

```text
Assignment-7-Mode-Collapse-GAN/
│
├── Assignment_7_Mode_Collapse_GAN.ipynb
│
├── mode_collapse_before.png
│
├── mode_collapse_after.png
│
├── broken_outputs/
│   ├── epoch_1.png
│   ├── epoch_2.png
│   ├── ...
│   └── epoch_20.png
│
├── fixed_outputs/
│   ├── epoch_1.png
│   ├── epoch_2.png
│   ├── ...
│   └── epoch_20.png
│
└── README.md
```

---

# ▶️ How to Run

### Step 1

Open the notebook in **Google Colab**.

### Step 2

Run the cells in order.

### Step 3

Load the Fashion-MNIST dataset.

### Step 4

Train the intentionally unstable GAN for approximately 20 epochs.

### Step 5

Observe and save the mode-collapse output.

### Step 6

Apply label smoothing and restore stable training settings.

### Step 7

Retrain the GAN.

### Step 8

Compare the generated images before and after the improvement.

---

# ✅ Conclusion

This experiment demonstrated the effect of **mode collapse in GANs** and showed how unstable architecture and optimization settings can negatively affect GAN training.

The unstable model produced less diverse outputs. After applying **label smoothing**, restoring Batch Normalization, and reducing the learning rate, the generated images became more diverse and the training became more stable.

Therefore, appropriate architecture and optimization techniques are important for achieving stable GAN training and reducing mode collapse.

---

## 👨‍💻 Assignment

**Assignment:** 7 — Creating and Fixing Mode Collapse in a GAN

**Model:** DCGAN

**Dataset:** Fashion-MNIST

**Framework:** TensorFlow / Keras

**Platform:** Google Colab

**Technique Used:** Label Smoothing
