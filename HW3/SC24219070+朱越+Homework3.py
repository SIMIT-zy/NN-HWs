import os
os.environ['KERAS_HOME'] = './datas/keras'  # 设置数据集路径
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 加载数据集
num_words = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words, seed=42)
class_names = ["Negative", "Positive"]

epochs = 20
batch_size = 128
sequences_len = 500
# 将序列填充/截断为固定长度（例如 500 单词）
train_sequences = sequence.pad_sequences(train_data, maxlen=sequences_len)
test_sequences = sequence.pad_sequences(test_data, maxlen=sequences_len)
model = Sequential([
    # 嵌入层：将整数索引映射为 32 维向量
    Embedding(input_dim=num_words, output_dim=32),
    # LSTM 层：32 个隐藏单元
    LSTM(32, dropout=0.2, recurrent_dropout=0.2),  # 使用 LSTM 单元
    # 输出层：二分类（sigmoid）
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.build(input_shape=(batch_size, sequences_len))  # (batch_size, sequence_length)
model.summary()

history = model.fit(train_sequences, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
# 绘制训练集和验证集的损失
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.xticks(range(0, epochs+1, 5))
plt.ylabel('Loss')
plt.legend()
plt.show()
# 绘制训练集和验证集的准确率
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.xticks(range(0, epochs+1, 5))
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('./models/HW3_RNN.keras')  # 保存为 keras 格式
# 评估测试集
net = load_model('./models/HW3_RNN.keras')  # 加载 keras 文件
test_loss, test_acc = net.evaluate(test_sequences, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")