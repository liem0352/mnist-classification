"""
深度学习实训项目 - 主运行脚本

本脚本整合任务一和任务二，提供统一的运行入口
支持命令行参数直接运行，无需交互式输入
"""

import sys
import os


def print_menu():
    """打印主菜单"""
    print("\n" + "="*60)
    print("深度学习实训项目 - 神经网络后向传播的实现")
    print("="*60)
    print("\n使用方式:")
    print("  python run_all.py 1    # 运行任务一")
    print("  python run_all.py 2    # 运行任务二")
    print("  python run_all.py 3    # 运行全部任务")
    print("  python run_all.py      # 显示此菜单")
    print("="*60)


def run_task1():
    """运行任务一"""
    print("\n" + "="*60)
    print("正在运行任务一...")
    print("="*60)
    try:
        import task1_mnist_keras
        task1_mnist_keras.main()
        print("\n任务一运行完成！")
    except Exception as e:
        print(f"任务一运行出错: {e}")
        import traceback
        traceback.print_exc()


def run_task2():
    """运行任务二"""
    print("\n" + "="*60)
    print("正在运行任务二...")
    print("="*60)
    try:
        import task2_loss_comparison
        task2_loss_comparison.main()
        print("\n任务二运行完成！")
    except Exception as e:
        print(f"任务二运行出错: {e}")
        import traceback
        traceback.print_exc()


def run_all():
    """运行所有任务"""
    run_task1()
    print("\n" + "="*60)
    print("继续运行任务二...")
    print("="*60)
    run_task2()


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        print_menu()
        return
    
    if choice == '1':
        run_task1()
    elif choice == '2':
        run_task2()
    elif choice == '3':
        run_all()
    else:
        print(f"\n无效选项: {choice}")
        print_menu()


if __name__ == '__main__':
    main()
