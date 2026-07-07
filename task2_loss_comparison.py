"""
任务二：对比交叉熵与均方误差损失对手写数字分类的影响

本模块包含：
- 使用相同网络结构，分别使用交叉熵损失和均方误差损失训练模型
- 对比两种损失函数的训练效果
- 输出梯度更新示例
- 可视化对比结果
"""

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
    加载并预处理MNIST数据集
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) 预处理后的数据集
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 展平
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))
    
    # One-hot编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test


def build_model(loss_function='categorical_crossentropy'):
    """
    构建神经网络模型
    
    Args:
        loss_function: 损失函数类型，用于标识模型
        
    Returns:
        keras.Model: 未编译的神经网络模型
    """
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    return model


def train_with_loss(model, loss_name, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """
    使用指定损失函数训练模型
    
    Args:
        model: Keras模型
        loss_name: 损失函数名称 ('categorical_crossentropy' 或 'mse')
        x_train: 训练数据
        y_train: 训练标签
        x_test: 测试数据
        y_test: 测试标签
        epochs: 训练轮数
        batch_size: 批次大小
        
    Returns:
        tuple: (训练好的模型, 训练历史)
    """
    print(f"\n{'='*60}")
    print(f"使用损失函数: {loss_name}")
    print(f"{'='*60}")
    
    # 根据损失函数名称选择损失函数
    if loss_name == 'categorical_crossentropy':
        loss = 'categorical_crossentropy'
    elif loss_name == 'mse':
        loss = 'mse'
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    print(f"开始训练（epochs={epochs}, batch_size={batch_size}）...")
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    return model, history


def demonstrate_gradient_update(model, loss_name, x_train, y_train):
    """
    演示梯度更新过程
    
    Args:
        model: Keras模型
        loss_name: 损失函数名称
        x_train: 训练数据
        y_train: 训练标签
    """
    print(f"\n{'='*60}")
    print(f"梯度更新示例 - {loss_name}")
    print(f"{'='*60}")
    
    # 取一个样本
    sample_idx = 0
    x_sample = x_train[sample_idx:sample_idx+1]
    y_sample = y_train[sample_idx:sample_idx+1]
    
    # 转换为TensorFlow张量
    x_tensor = tf.convert_to_tensor(x_sample, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_sample, dtype=tf.float32)
    
    # 选择损失函数
    if loss_name == 'categorical_crossentropy':
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()
    
    # 计算梯度
    with tf.GradientTape() as tape:
        predictions = model(x_tensor)
        loss = loss_fn(y_tensor, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    print(f"\n样本索引: {sample_idx}")
    print(f"真实标签: {np.argmax(y_sample)}")
    print(f"预测概率: {predictions.numpy()[0].round(4)}")
    print(f"预测类别: {np.argmax(predictions.numpy()[0])}")
    print(f"\n损失值 ({loss_name}): {loss.numpy():.6f}")
    
    print("\n各层参数梯度统计:")
    for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
        if grad is not None:
            grad_flat = tf.reshape(grad, [-1])
            print(f"\n  参数 {i+1} ({var.name}):")
            print(f"    形状: {grad.shape}")
            print(f"    梯度均值: {tf.reduce_mean(tf.abs(grad)).numpy():.8f}")
            print(f"    梯度标准差: {tf.math.reduce_std(grad).numpy():.8f}")
            print(f"    梯度最大值: {tf.reduce_max(tf.abs(grad)).numpy():.8f}")
            print(f"    梯度最小值: {tf.reduce_min(tf.abs(grad)).numpy():.8f}")
            
            # 显示前5个梯度值
            print(f"    前5个梯度值: {grad_flat[:5].numpy().round(8)}")


def compare_results(history_ce, history_mse, epochs=10):
    """
    对比两种损失函数的训练结果
    
    Args:
        history_ce: 交叉熵损失的训练历史
        history_mse: 均方误差损失的训练历史
        epochs: 训练轮数
    """
    print(f"\n{'='*60}")
    print("训练结果对比")
    print(f"{'='*60}")
    
    # 最终训练结果
    print("\n【最终训练结果对比】")
    print(f"{'指标':<25} {'交叉熵':<15} {'均方误差':<15}")
    print("-" * 60)
    print(f"{'最终训练损失':<25} {history_ce.history['loss'][-1]:<15.4f} {history_mse.history['loss'][-1]:<15.4f}")
    print(f"{'最终训练准确率':<25} {history_ce.history['accuracy'][-1]:<15.4f} {history_mse.history['accuracy'][-1]:<15.4f}")
    print(f"{'最终验证损失':<25} {history_ce.history['val_loss'][-1]:<15.4f} {history_mse.history['val_loss'][-1]:<15.4f}")
    print(f"{'最终验证准确率':<25} {history_ce.history['val_accuracy'][-1]:<15.4f} {history_mse.history['val_accuracy'][-1]:<15.4f}")
    
    # 收敛速度对比
    print("\n【收敛速度对比】")
    ce_best_epoch = np.argmin(history_ce.history['val_loss']) + 1
    mse_best_epoch = np.argmin(history_mse.history['val_loss']) + 1
    print(f"交叉熵最佳验证损失轮次: {ce_best_epoch}")
    print(f"均方误差最佳验证损失轮次: {mse_best_epoch}")


def plot_comparison(history_ce, history_mse, epochs=10):
    """
    绘制两种损失函数的对比图
    
    Args:
        history_ce: 交叉熵损失的训练历史
        history_mse: 均方误差损失的训练历史
        epochs: 训练轮数
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epoch_range = range(1, epochs + 1)
    
    # 训练损失对比
    axes[0, 0].plot(epoch_range, history_ce.history['loss'], 'b-', label='交叉熵', linewidth=2)
    axes[0, 0].plot(epoch_range, history_mse.history['loss'], 'r-', label='均方误差', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练损失对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 验证损失对比
    axes[0, 1].plot(epoch_range, history_ce.history['val_loss'], 'b-', label='交叉熵', linewidth=2)
    axes[0, 1].plot(epoch_range, history_mse.history['val_loss'], 'r-', label='均方误差', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('验证损失对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 训练准确率对比
    axes[1, 0].plot(epoch_range, history_ce.history['accuracy'], 'b-', label='交叉熵', linewidth=2)
    axes[1, 0].plot(epoch_range, history_mse.history['accuracy'], 'r-', label='均方误差', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('训练准确率对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 验证准确率对比
    axes[1, 1].plot(epoch_range, history_ce.history['val_accuracy'], 'b-', label='交叉熵', linewidth=2)
    axes[1, 1].plot(epoch_range, history_mse.history['val_accuracy'], 'r-', label='均方误差', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('验证准确率对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('task2_loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n对比图已保存到 task2_loss_comparison.png")


def analyze_loss_characteristics():
    """
    分析两种损失函数的特性差异
    """
    print(f"\n{'='*60}")
    print("损失函数特性分析")
    print(f"{'='*60}")
    
    print("""
【交叉熵损失 (Categorical Crossentropy)】
1. 数学公式: L = -Σ(y_true * log(y_pred))
2. 特点:
   - 专门针对分类任务设计
   - 对预测概率与真实标签的差异敏感
   - 梯度在误差大时较大，收敛速度快
   - 与Softmax激活函数配合效果最佳
3. 适用场景: 多分类任务的首选损失函数

【均方误差损失 (Mean Squared Error)】
1. 数学公式: L = (1/n) * Σ(y_true - y_pred)²
2. 特点:
   - 通用损失函数， originally 用于回归任务
   - 梯度随误差线性变化
   - 在分类任务中可能导致梯度消失问题
   - 对离群值敏感
3. 适用场景: 回归任务，不推荐用于分类任务

【关键差异】
1. 梯度特性:
   - 交叉熵: 梯度与误差成正比，学习信号强
   - MSE: 梯度受Softmax导数影响，可能过小

2. 收敛速度:
   - 交叉熵: 收敛更快，通常需要更少epoch
   - MSE: 收敛较慢，可能需要更多epoch

3. 最终性能:
   - 交叉熵: 在分类任务上通常达到更高准确率
   - MSE: 在分类任务上性能通常较差
""")


def main():
    """
    任务二主函数：对比交叉熵与均方误差损失
    """
    print("="*60)
    print("任务二：对比交叉熵与均方误差损失对手写数字分类的影响")
    print("="*60)
    
    # 加载数据
    print("\n【步骤1】加载并预处理MNIST数据集")
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # 分析损失函数特性
    print("\n【步骤2】分析损失函数特性")
    analyze_loss_characteristics()
    
    # 使用交叉熵损失训练
    print("\n【步骤3】使用交叉熵损失训练模型")
    model_ce = build_model('categorical_crossentropy')
    model_ce, history_ce = train_with_loss(
        model_ce, 'categorical_crossentropy',
        x_train, y_train, x_test, y_test,
        epochs=10, batch_size=128
    )
    
    # 使用均方误差损失训练
    print("\n【步骤4】使用均方误差损失训练模型")
    model_mse = build_model('mse')
    model_mse, history_mse = train_with_loss(
        model_mse, 'mse',
        x_train, y_train, x_test, y_test,
        epochs=10, batch_size=128
    )
    
    # 对比训练结果
    print("\n【步骤5】对比训练结果")
    compare_results(history_ce, history_mse, epochs=10)
    
    # 梯度更新示例 - 交叉熵
    print("\n【步骤6】交叉熵损失梯度更新示例")
    demonstrate_gradient_update(model_ce, 'categorical_crossentropy', x_train, y_train)
    
    # 梯度更新示例 - MSE
    print("\n【步骤7】均方误差损失梯度更新示例")
    demonstrate_gradient_update(model_mse, 'mse', x_train, y_train)
    
    # 可视化对比
    print("\n【步骤8】可视化对比结果")
    plot_comparison(history_ce, history_mse, epochs=10)
    
    print("\n" + "="*60)
    print("任务二完成！")
    print("="*60)


if __name__ == '__main__':
    main()
