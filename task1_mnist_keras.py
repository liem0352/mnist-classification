"""
任务一：使用Keras高层API搭建神经网络实现手写数字分类

本模块包含：
- MNIST数据集加载与预处理
- 神经网络模型搭建（Sequential API）
- 模型编译、训练、评估
- 模型保存与加载
- 预测与可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data():
    """
    加载MNIST数据集并进行预处理
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) 预处理后的数据集
    """
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 归一化：将像素值从 [0, 255] 缩放到 [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 将图像从 28x28 展平为 784 维向量
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))
    
    # 将标签转换为one-hot编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"训练集形状: {x_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {x_test.shape}, 标签形状: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test


def build_model():
    """
    使用Keras Sequential API构建神经网络模型
    
    模型结构：
    - 输入层：784个神经元（对应28x28图像展平）
    - 隐藏层1：256个神经元，ReLU激活
    - Dropout层：防止过拟合
    - 隐藏层2：128个神经元，ReLU激活
    - 输出层：10个神经元，Softmax激活（对应10个数字类别）
    
    Returns:
        keras.Model: 编译好的神经网络模型
    """
    model = models.Sequential([
        # 输入层
        layers.Input(shape=(784,)),
        
        # 第一隐藏层
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        
        # 第二隐藏层
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        
        # 输出层
        layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("模型结构：")
    model.summary()
    
    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """
    训练神经网络模型
    
    Args:
        model: Keras模型
        x_train: 训练数据
        y_train: 训练标签
        x_test: 测试数据
        y_test: 测试标签
        epochs: 训练轮数
        batch_size: 批次大小
        
    Returns:
        History: 训练历史记录
    """
    print(f"\n开始训练模型（epochs={epochs}, batch_size={batch_size}）...")
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """
    评估模型性能
    
    Args:
        model: 训练好的Keras模型
        x_test: 测试数据
        y_test: 测试标签
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    print("\n评估模型性能...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy


def save_model(model, model_path='mnist_model.keras'):
    """
    保存训练好的模型
    
    Args:
        model: 训练好的Keras模型
        model_path: 模型保存路径
    """
    model.save(model_path)
    print(f"\n模型已保存到: {model_path}")


def load_saved_model(model_path='mnist_model.keras'):
    """
    加载已保存的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        keras.Model: 加载的模型
    """
    model = keras.models.load_model(model_path)
    print(f"模型已从 {model_path} 加载")
    return model


def visualize_predictions(model, x_test, y_test, num_samples=10):
    """
    可视化预测结果
    
    Args:
        model: 训练好的模型
        x_test: 测试数据
        y_test: 测试标签
        num_samples: 展示样本数量
    """
    # 随机选择样本
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        # 预测
        sample = x_test[idx:idx+1]
        prediction = model.predict(sample, verbose=0)
        predicted_label = np.argmax(prediction)
        true_label = np.argmax(y_test[idx])
        
        # 显示图像
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        color = 'green' if predicted_label == true_label else 'red'
        plt.title(f'预测: {predicted_label}\n真实: {true_label}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('task1_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("预测可视化已保存到 task1_predictions.png")


def plot_training_history(history):
    """
    绘制训练历史曲线
    
    Args:
        history: 模型训练返回的History对象
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(history.history['loss'], label='训练损失')
    axes[0].plot(history.history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练与验证损失')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history.history['accuracy'], label='训练准确率')
    axes[1].plot(history.history['val_accuracy'], label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('训练与验证准确率')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('task1_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("训练历史曲线已保存到 task1_training_history.png")


def demonstrate_forward_backward_propagation(model, x_train, y_train):
    """
    演示前向传播和反向传播过程
    
    Args:
        model: Keras模型
        x_train: 训练数据
        y_train: 训练标签
    """
    print("\n" + "="*60)
    print("前向传播与反向传播演示")
    print("="*60)
    
    # 取一个样本
    sample_idx = 0
    x_sample = x_train[sample_idx:sample_idx+1]
    y_sample = y_train[sample_idx:sample_idx+1]
    
    print(f"\n选取样本索引: {sample_idx}")
    print(f"输入形状: {x_sample.shape}")
    print(f"标签形状: {y_sample.shape}")
    print(f"真实标签: {np.argmax(y_sample)}")
    
    # 前向传播：获取各层输出
    print("\n--- 前向传播过程 ---")
    
    # 先进行一次预测，建立模型连接
    _ = model.predict(x_sample, verbose=0)
    
    # 创建逐层输出模型，使用Input层作为输入
    from tensorflow.keras.layers import Input
    input_layer = Input(shape=(784,))
    x = input_layer
    for layer in model.layers:
        x = layer(x)
    activation_model = models.Model(inputs=input_layer, outputs=x)
    
    # 获取各层输出
    layer_outputs = []
    x = input_layer
    for layer in model.layers:
        x = layer(x)
        layer_outputs.append(x)
    activation_model = models.Model(inputs=input_layer, outputs=layer_outputs)
    
    activations = activation_model.predict(x_sample, verbose=0)
    
    for i, (layer, activation) in enumerate(zip(model.layers, activations)):
        print(f"层 {i+1} ({layer.name}): 输出形状 = {activation.shape}")
        if i == len(model.layers) - 1:
            print(f"  输出概率分布: {activation[0].round(4)}")
            print(f"  预测类别: {np.argmax(activation[0])}")
    
    # 反向传播：计算梯度
    print("\n--- 反向传播过程（梯度计算） ---")
    
    x_tensor = tf.convert_to_tensor(x_sample, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_sample, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor)
        loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)
    
    # 获取可训练变量的梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    
    print(f"损失值: {loss.numpy()[0]:.6f}")
    print("\n各层参数梯度统计:")
    
    for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
        if grad is not None:
            print(f"  参数 {i+1} ({var.name}):")
            print(f"    形状: {grad.shape}")
            print(f"    梯度均值: {tf.reduce_mean(tf.abs(grad)).numpy():.6f}")
            print(f"    梯度最大值: {tf.reduce_max(tf.abs(grad)).numpy():.6f}")
    
    print("\n" + "="*60)


def main():
    """
    任务一主函数：完整的MNIST分类流程
    """
    print("="*60)
    print("任务一：Keras高层API实现手写数字分类")
    print("="*60)
    
    # 1. 加载并预处理数据
    print("\n【步骤1】加载并预处理MNIST数据集")
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # 2. 构建模型
    print("\n【步骤2】构建神经网络模型")
    model = build_model()
    
    # 3. 演示前向传播和反向传播
    print("\n【步骤3】演示前向传播与反向传播")
    demonstrate_forward_backward_propagation(model, x_train, y_train)
    
    # 4. 训练模型
    print("\n【步骤4】训练模型")
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128)
    
    # 5. 评估模型
    print("\n【步骤5】评估模型")
    evaluate_model(model, x_test, y_test)
    
    # 6. 保存模型
    print("\n【步骤6】保存模型")
    save_model(model, 'mnist_model.keras')
    
    # 7. 加载模型并验证
    print("\n【步骤7】加载模型并验证")
    loaded_model = load_saved_model('mnist_model.keras')
    evaluate_model(loaded_model, x_test, y_test)
    
    # 8. 可视化训练历史
    print("\n【步骤8】可视化训练过程")
    plot_training_history(history)
    
    # 9. 可视化预测结果
    print("\n【步骤9】可视化预测结果")
    visualize_predictions(model, x_test, y_test, num_samples=10)
    
    print("\n" + "="*60)
    print("任务一完成！")
    print("="*60)


if __name__ == '__main__':
    main()
