"""
测试脚本 - 验证模型功能和性能

本脚本包含：
- 测试数据加载和预处理
- 测试模型构建
- 测试训练过程
- 测试模型保存和加载
- 测试预测功能
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import task1_mnist_keras as task1
import task2_loss_comparison as task2


def test_data_loading():
    """测试数据加载和预处理"""
    print("\n" + "="*60)
    print("测试1: 数据加载和预处理")
    print("="*60)
    
    x_train, y_train, x_test, y_test = task1.load_and_preprocess_data()
    
    # 验证数据形状
    assert x_train.shape == (60000, 784), f"训练数据形状错误: {x_train.shape}"
    assert y_train.shape == (60000, 10), f"训练标签形状错误: {y_train.shape}"
    assert x_test.shape == (10000, 784), f"测试数据形状错误: {x_test.shape}"
    assert y_test.shape == (10000, 10), f"测试标签形状错误: {y_test.shape}"
    
    # 验证数据范围
    assert np.all(x_train >= 0) and np.all(x_train <= 1), "训练数据归一化错误"
    assert np.all(x_test >= 0) and np.all(x_test <= 1), "测试数据归一化错误"
    
    # 验证one-hot编码
    assert np.all(np.sum(y_train, axis=1) == 1), "训练标签one-hot编码错误"
    assert np.all(np.sum(y_test, axis=1) == 1), "测试标签one-hot编码错误"
    
    print("[OK] 数据加载和预处理测试通过")
    return x_train, y_train, x_test, y_test


def test_model_building():
    """测试模型构建"""
    print("\n" + "="*60)
    print("测试2: 模型构建")
    print("="*60)
    
    model = task1.build_model()
    
    # 验证模型类型
    assert isinstance(model, keras.Model), "模型类型错误"
    
    # 验证输入输出形状
    assert model.input_shape == (None, 784), f"输入形状错误: {model.input_shape}"
    assert model.output_shape == (None, 10), f"输出形状错误: {model.output_shape}"
    
    # 验证层数
    assert len(model.layers) == 5, f"层数错误: {len(model.layers)}"
    
    # 验证编译
    assert model.loss == 'categorical_crossentropy', "损失函数错误"
    assert model.optimizer is not None, "优化器未设置"
    
    print("[OK] 模型构建测试通过")
    return model


def test_training():
    """测试训练过程"""
    print("\n" + "="*60)
    print("测试3: 训练过程")
    print("="*60)
    
    # 使用少量数据进行快速测试
    x_train, y_train, x_test, y_test = task1.load_and_preprocess_data()
    
    # 只使用1000个样本进行快速测试
    x_train_small = x_train[:1000]
    y_train_small = y_train[:1000]
    
    model = task1.build_model()
    
    # 训练1个epoch
    history = model.fit(
        x_train_small, y_train_small,
        epochs=1,
        batch_size=32,
        verbose=0
    )
    
    # 验证训练历史
    assert 'loss' in history.history, "训练历史缺少损失记录"
    assert 'accuracy' in history.history, "训练历史缺少准确率记录"
    assert len(history.history['loss']) == 1, "训练轮数错误"
    
    # 验证损失值合理
    assert history.history['loss'][0] > 0, "损失值异常"
    assert history.history['accuracy'][0] >= 0, "准确率异常"
    
    print(f"[OK] 训练过程测试通过 (损失: {history.history['loss'][0]:.4f}, 准确率: {history.history['accuracy'][0]:.4f})")
    return model


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n" + "="*60)
    print("测试4: 模型保存和加载")
    print("="*60)
    
    # 创建并训练一个简单模型
    x_train, y_train, x_test, y_test = task1.load_and_preprocess_data()
    x_train_small = x_train[:1000]
    y_train_small = y_train[:1000]
    
    model = task1.build_model()
    model.fit(x_train_small, y_train_small, epochs=1, batch_size=32, verbose=0)
    
    # 保存前预测
    sample = x_test[:5]
    pred_before = model.predict(sample, verbose=0)
    
    # 保存模型
    test_model_path = 'test_model.keras'
    task1.save_model(model, test_model_path)
    
    # 验证文件存在
    import os
    assert os.path.exists(test_model_path), "模型文件未保存成功"
    
    # 加载模型
    loaded_model = task1.load_saved_model(test_model_path)
    
    # 加载后预测
    pred_after = loaded_model.predict(sample, verbose=0)
    
    # 验证预测结果一致
    assert np.allclose(pred_before, pred_after, atol=1e-6), "加载后预测结果不一致"
    
    # 清理测试文件
    os.remove(test_model_path)
    
    print("[OK] 模型保存和加载测试通过")


def test_prediction():
    """测试预测功能"""
    print("\n" + "="*60)
    print("测试5: 预测功能")
    print("="*60)
    
    x_train, y_train, x_test, y_test = task1.load_and_preprocess_data()
    x_train_small = x_train[:1000]
    y_train_small = y_train[:1000]
    
    model = task1.build_model()
    model.fit(x_train_small, y_train_small, epochs=1, batch_size=32, verbose=0)
    
    # 预测
    predictions = model.predict(x_test[:10], verbose=0)
    
    # 验证预测结果形状
    assert predictions.shape == (10, 10), f"预测结果形状错误: {predictions.shape}"
    
    # 验证概率和为1
    assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5), "预测概率和不等于1"
    
    # 验证概率在0-1之间
    assert np.all(predictions >= 0) and np.all(predictions <= 1), "预测概率超出范围"
    
    # 验证预测类别
    predicted_classes = np.argmax(predictions, axis=1)
    assert len(predicted_classes) == 10, "预测类别数量错误"
    assert np.all((predicted_classes >= 0) & (predicted_classes <= 9)), "预测类别超出范围"
    
    print(f"[OK] 预测功能测试通过 (预测类别: {predicted_classes})")


def test_loss_comparison():
    """测试损失函数对比"""
    print("\n" + "="*60)
    print("测试6: 损失函数对比")
    print("="*60)
    
    x_train, y_train, x_test, y_test = task2.load_and_preprocess_data()
    
    # 使用少量数据快速测试
    x_train_small = x_train[:1000]
    y_train_small = y_train[:1000]
    x_test_small = x_test[:200]
    y_test_small = y_test[:200]
    
    # 测试交叉熵
    model_ce = task2.build_model('categorical_crossentropy')
    model_ce.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_ce = model_ce.fit(x_train_small, y_train_small, epochs=1, batch_size=32, verbose=0)
    
    # 测试MSE
    model_mse = task2.build_model('mse')
    model_mse.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    history_mse = model_mse.fit(x_train_small, y_train_small, epochs=1, batch_size=32, verbose=0)
    
    # 验证两种损失都能正常训练
    assert 'loss' in history_ce.history, "交叉熵训练历史错误"
    assert 'loss' in history_mse.history, "MSE训练历史错误"
    
    # 评估
    loss_ce, acc_ce = model_ce.evaluate(x_test_small, y_test_small, verbose=0)
    loss_mse, acc_mse = model_mse.evaluate(x_test_small, y_test_small, verbose=0)
    
    print(f"[OK] 损失函数对比测试通过")
    print(f"  交叉熵 - 损失: {loss_ce:.4f}, 准确率: {acc_ce:.4f}")
    print(f"  MSE    - 损失: {loss_mse:.4f}, 准确率: {acc_mse:.4f}")


def test_gradient_computation():
    """测试梯度计算"""
    print("\n" + "="*60)
    print("测试7: 梯度计算")
    print("="*60)
    
    x_train, y_train, x_test, y_test = task1.load_and_preprocess_data()
    
    model = task1.build_model()
    
    # 取一个样本
    x_sample = x_train[:1]
    y_sample = y_train[:1]
    
    x_tensor = tf.convert_to_tensor(x_sample, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_sample, dtype=tf.float32)
    
    # 计算梯度
    with tf.GradientTape() as tape:
        predictions = model(x_tensor)
        loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 验证梯度
    assert gradients is not None, "梯度计算失败"
    assert len(gradients) == len(model.trainable_variables), "梯度数量不匹配"
    
    # 验证梯度不为零
    for i, grad in enumerate(gradients):
        if grad is not None:
            assert not np.allclose(grad.numpy(), 0), f"参数 {i+1} 的梯度全为零"
    
    print("[OK] 梯度计算测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始运行测试")
    print("="*60)
    
    tests = [
        ("数据加载和预处理", test_data_loading),
        ("模型构建", test_model_building),
        ("训练过程", test_training),
        ("模型保存和加载", test_model_save_load),
        ("预测功能", test_prediction),
        ("损失函数对比", test_loss_comparison),
        ("梯度计算", test_gradient_computation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_name}测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n[OK] 所有测试通过！")
    else:
        print(f"\n[FAIL] {failed}个测试失败，请检查代码。")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
