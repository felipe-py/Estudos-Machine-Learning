# Importações básicas
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Carregar o dataset Cats vs Dogs (já vem no TFDS)
# Ele já baixa e organiza automaticamente
(raw_train, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

# Verificar classes
print(metadata.features['label'].names)  # ['cat', 'dog']

# Função para redimensionar imagens
IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# Preparar dados
train = raw_train.map(format_example).batch(32).shuffle(1000)
test = raw_test.map(format_example).batch(32)

# Ver algumas imagens
for image, label in train.take(1):
    plt.figure(figsize=(10,10))
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(image[i])
        plt.title(metadata.features['label'].int2str(label[i]))
        plt.axis("off")


# Carregar modelo pré-treinado (sem a última camada)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Congelar camadas para não treinar tudo de novo
base_model.trainable = False

# Construir modelo final
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: cat e dog
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar
history = model.fit(train, epochs=3, validation_data=test)

# Avaliar modelo
loss, acc = model.evaluate(test)
print(f"Acurácia no teste: {acc:.2f}")

# Plotar curva de treino
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
