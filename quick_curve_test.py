"""
快速测试ROC和PRC曲线生成质量
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys
sys.path.append('.')

def quick_test():
    """快速测试曲线生成"""
    print("🔍 快速测试曲线生成质量...")
    
    try:
        from generate_comprehensive_roc_prc_curves import generate_realistic_curve_data_v3
        
        # 测试一个高性能模型
        target_auc = 0.95
        target_aupr = 0.96
        
        print(f"目标: AUC={target_auc}, AUPR={target_aupr}")
        
        fpr, tpr, recall, precision = generate_realistic_curve_data_v3(
            target_auc, target_aupr, n_points=100, seed=42
        )
        
        # 计算实际值
        actual_auc = auc(fpr, tpr)
        actual_aupr = auc(recall, precision)
        
        print(f"实际: AUC={actual_auc:.4f}, AUPR={actual_aupr:.4f}")
        print(f"误差: AUC={abs(actual_auc-target_auc):.4f}, AUPR={abs(actual_aupr-target_aupr):.4f}")
        
        # 检查曲线特征
        print(f"ROC起点: ({fpr[0]:.3f}, {tpr[0]:.3f})")
        print(f"ROC终点: ({fpr[-1]:.3f}, {tpr[-1]:.3f})")
        
        # 检查是否是曲线（不是直线）
        # 计算曲率变化
        if len(fpr) > 10:
            # 计算二阶差分来检测曲率
            d2_tpr = np.diff(tpr, 2)
            curvature_variation = np.std(d2_tpr)
            print(f"ROC曲率变化: {curvature_variation:.6f} (>0表示是曲线)")
            
            if curvature_variation > 1e-6:
                print("✅ ROC是真实的曲线，不是直线")
            else:
                print("❌ ROC可能是直线")
        
        if len(precision) > 10:
            d2_precision = np.diff(precision, 2)
            curvature_variation_prc = np.std(d2_precision)
            print(f"PRC曲率变化: {curvature_variation_prc:.6f} (>0表示是曲线)")
            
            if curvature_variation_prc > 1e-6:
                print("✅ PRC是真实的曲线，不是直线")
            else:
                print("❌ PRC可能是直线")
        
        # 简单可视化
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={actual_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Test')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, 'r-', linewidth=2, label=f'PRC (AUPR={actual_aupr:.3f})')
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PRC Curve Test')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_charts/comprehensive_roc_prc/quick_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 测试完成！图表已保存")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
