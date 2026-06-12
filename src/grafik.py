"""#Augmentation etkisinin model performansına etkisi için grafik
import matplotlib.pyplot as plt

# veriler
labels = ["Orijinal", "Blur", "Noise20", "Noise40", "Noise60"]
probabilities = [0.82, 0.63, 0.51, 0.24, 0.19]

plt.figure(figsize=(8,5))
plt.plot(labels, probabilities, marker='o')

plt.title("Augmentation Etkisinin Model Performansına Etkisi")
plt.xlabel("Görüntü Türü")
plt.ylabel("Probability")

plt.grid(True)
plt.show() 
"""

#Augmentation etkisi için grafik
import matplotlib.pyplot as plt

labels = ["Augmentation Yok", "Augmentation Var"]
word_accuracy = [48, 61]
cer = [27, 18.7]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.bar(labels, word_accuracy)
plt.title("Word Accuracy (%)")

plt.subplot(1,2,2)
plt.bar(labels, cer)
plt.title("Character Error Rate (%)")

plt.show()