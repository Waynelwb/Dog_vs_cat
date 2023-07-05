import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('../model/dogs_and_cats.h5')

# 设置数据集路径
test_dir = '../data/new/test'

# 加载数据
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=128,
                                                  class_mode='binary')

# 评估模型准确度
results = model.evaluate(test_generator, verbose=1)

# 输出结果
print(f'Test Loss: {results[0]}')
print(f'Test Accuracy: {results[1]}')
