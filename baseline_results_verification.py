"""
基线结果真实性验证脚本
检查所有基线方法的结果是否真实可信
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_baseline_results():
    """加载基线结果"""
    print("📊 加载基线实验结果...")
    
    with open('comprehensive_results/comprehensive_comparison_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def analyze_result_credibility(results):
    """分析结果可信度"""
    print("\n🔍 分析基线结果可信度...")
    print("=" * 60)
    
    credibility_analysis = {}
    
    for method_name, method_data in results.items():
        if method_name == 'our_model':
            continue
            
        avg_results = method_data['average_results']
        fold_results = method_data['fold_results']
        
        # 提取关键指标
        auc_mean = avg_results['auc']['mean']
        auc_std = avg_results['auc']['std']
        aupr_mean = avg_results['aupr']['mean']
        aupr_std = avg_results['aupr']['std']
        
        # 检查各种可疑指标
        credibility_issues = []
        credibility_score = 100  # 满分100
        
        # 1. 检查完美性能（最可疑）
        if auc_mean >= 0.999 and auc_std == 0.0:
            credibility_issues.append("🚨 完美AUC性能 (1.0000±0.0000) - 极度可疑")
            credibility_score -= 80
        elif auc_mean >= 0.995:
            credibility_issues.append("⚠️ 异常高AUC性能 (≥0.995) - 可疑")
            credibility_score -= 30
        
        # 2. 检查标准差异常
        if auc_std == 0.0 and auc_mean < 1.0:
            credibility_issues.append("⚠️ AUC标准差为0 - 不现实")
            credibility_score -= 20
        elif auc_std < 0.001:
            credibility_issues.append("⚠️ AUC标准差过小 - 可疑")
            credibility_score -= 10
        
        # 3. 检查recall异常
        recall_values = [fold['recall'] for fold in fold_results]
        if all(r == 1.0 for r in recall_values):
            credibility_issues.append("🚨 所有折的Recall都是1.0 - 极度可疑")
            credibility_score -= 40
        
        # 4. 检查accuracy与其他指标的一致性
        accuracy_values = [fold['accuracy'] for fold in fold_results]
        precision_values = [fold['precision'] for fold in fold_results]
        
        # 检查precision和accuracy的关系
        avg_accuracy = np.mean(accuracy_values)
        avg_precision = np.mean(precision_values)
        
        if avg_precision > 0.9 and avg_accuracy < 0.6:
            credibility_issues.append("⚠️ 高Precision但低Accuracy - 不合理")
            credibility_score -= 15
        
        # 5. 检查AUC和AUPR的关系
        if auc_mean > 0.99 and aupr_mean > 0.99:
            credibility_issues.append("🚨 AUC和AUPR都接近完美 - 极度可疑")
            credibility_score -= 50
        
        # 6. 检查折间变异性
        auc_fold_values = [fold['auc'] for fold in fold_results]
        auc_range = max(auc_fold_values) - min(auc_fold_values)
        
        if auc_range == 0.0:
            credibility_issues.append("🚨 所有折的AUC完全相同 - 极度可疑")
            credibility_score -= 30
        elif auc_range < 0.001:
            credibility_issues.append("⚠️ 折间AUC变异极小 - 可疑")
            credibility_score -= 15
        
        # 确定可信度等级
        if credibility_score >= 80:
            credibility_level = "✅ 高可信度"
        elif credibility_score >= 60:
            credibility_level = "⚠️ 中等可信度"
        elif credibility_score >= 40:
            credibility_level = "❌ 低可信度"
        else:
            credibility_level = "🚨 极低可信度"
        
        credibility_analysis[method_name] = {
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'aupr_mean': aupr_mean,
            'aupr_std': aupr_std,
            'credibility_score': credibility_score,
            'credibility_level': credibility_level,
            'issues': credibility_issues,
            'fold_results': fold_results
        }
        
        # 显示分析结果
        print(f"\n📋 {method_name.upper()}:")
        print(f"   AUC: {auc_mean:.4f}±{auc_std:.4f}")
        print(f"   AUPR: {aupr_mean:.4f}±{aupr_std:.4f}")
        print(f"   可信度评分: {credibility_score}/100")
        print(f"   可信度等级: {credibility_level}")
        
        if credibility_issues:
            print("   发现的问题:")
            for issue in credibility_issues:
                print(f"     • {issue}")
        else:
            print("   ✅ 未发现明显问题")
    
    return credibility_analysis

def identify_suspicious_patterns(credibility_analysis):
    """识别可疑模式"""
    print("\n🕵️ 识别可疑模式...")
    print("=" * 60)
    
    suspicious_methods = []
    reliable_methods = []
    
    for method_name, analysis in credibility_analysis.items():
        if analysis['credibility_score'] < 40:
            suspicious_methods.append((method_name, analysis))
        elif analysis['credibility_score'] >= 80:
            reliable_methods.append((method_name, analysis))
    
    print(f"🚨 高度可疑的方法 ({len(suspicious_methods)}个):")
    for method_name, analysis in suspicious_methods:
        print(f"   • {method_name}: {analysis['credibility_level']} (评分: {analysis['credibility_score']}/100)")
        print(f"     AUC: {analysis['auc_mean']:.4f}, AUPR: {analysis['aupr_mean']:.4f}")
    
    print(f"\n✅ 相对可靠的方法 ({len(reliable_methods)}个):")
    for method_name, analysis in reliable_methods:
        print(f"   • {method_name}: {analysis['credibility_level']} (评分: {analysis['credibility_score']}/100)")
        print(f"     AUC: {analysis['auc_mean']:.4f}, AUPR: {analysis['aupr_mean']:.4f}")
    
    return suspicious_methods, reliable_methods

def generate_credibility_report(credibility_analysis, suspicious_methods, reliable_methods):
    """生成可信度报告"""
    print("\n📝 生成可信度分析报告...")
    
    report_path = "training_charts/baseline_comparison/credibility_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("基线方法结果可信度分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("分析目的: 检查基线方法实验结果的真实性和可信度\n")
        f.write("分析方法: 多维度可疑指标检测\n\n")
        
        # 总体统计
        total_methods = len(credibility_analysis)
        suspicious_count = len(suspicious_methods)
        reliable_count = len(reliable_methods)
        
        f.write("总体统计:\n")
        f.write(f"  总方法数: {total_methods}\n")
        f.write(f"  高度可疑: {suspicious_count} ({suspicious_count/total_methods*100:.1f}%)\n")
        f.write(f"  相对可靠: {reliable_count} ({reliable_count/total_methods*100:.1f}%)\n\n")
        
        # 详细分析
        f.write("详细分析结果:\n")
        f.write("-" * 30 + "\n")
        
        # 按可信度评分排序
        sorted_methods = sorted(credibility_analysis.items(), 
                              key=lambda x: x[1]['credibility_score'], reverse=True)
        
        for method_name, analysis in sorted_methods:
            f.write(f"\n{method_name.upper()}:\n")
            f.write(f"  AUC: {analysis['auc_mean']:.4f}±{analysis['auc_std']:.4f}\n")
            f.write(f"  AUPR: {analysis['aupr_mean']:.4f}±{analysis['aupr_std']:.4f}\n")
            f.write(f"  可信度评分: {analysis['credibility_score']}/100\n")
            f.write(f"  可信度等级: {analysis['credibility_level']}\n")
            
            if analysis['issues']:
                f.write("  发现的问题:\n")
                for issue in analysis['issues']:
                    f.write(f"    • {issue}\n")
            else:
                f.write("  ✅ 未发现明显问题\n")
        
        # 结论和建议
        f.write(f"\n结论和建议:\n")
        f.write("-" * 30 + "\n")
        
        if suspicious_methods:
            f.write("🚨 发现高度可疑的基线结果:\n")
            for method_name, analysis in suspicious_methods:
                f.write(f"  • {method_name}: 可能存在数据泄露、过拟合或实现错误\n")
            
            f.write("\n建议:\n")
            f.write("1. 重新实现可疑的基线方法\n")
            f.write("2. 使用相同的数据划分和预处理\n")
            f.write("3. 仔细检查评估方法的正确性\n")
            f.write("4. 考虑使用原始论文的官方代码\n")
        
        if reliable_methods:
            f.write(f"\n✅ 相对可靠的基线方法可以作为有效对比:\n")
            for method_name, analysis in reliable_methods:
                f.write(f"  • {method_name}: AUC={analysis['auc_mean']:.4f}\n")
        
        # 对我们模型的影响
        f.write(f"\n对我们模型的影响:\n")
        f.write("1. 应该主要与可靠的基线方法进行对比\n")
        f.write("2. 可疑的完美性能(如iPiDi-PUL)不应作为真实基准\n")
        f.write("3. 我们的目标应该是超越可靠基线中的最佳性能\n")
    
    print(f"✅ 可信度报告已保存到: {report_path}")

def create_credibility_visualization(credibility_analysis):
    """创建可信度可视化"""
    print("\n📊 创建可信度可视化...")
    
    # 准备数据
    methods = list(credibility_analysis.keys())
    scores = [credibility_analysis[m]['credibility_score'] for m in methods]
    aucs = [credibility_analysis[m]['auc_mean'] for m in methods]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 可信度评分图
    colors = ['red' if s < 40 else 'orange' if s < 60 else 'yellow' if s < 80 else 'green' for s in scores]
    bars = ax1.bar(range(len(methods)), scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('基线方法')
    ax1.set_ylabel('可信度评分')
    ax1.set_title('基线方法可信度评分')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加评分标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    # AUC vs 可信度散点图
    ax2.scatter(aucs, scores, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('AUC性能')
    ax2.set_ylabel('可信度评分')
    ax2.set_title('AUC性能 vs 可信度评分')
    ax2.grid(True, alpha=0.3)
    
    # 添加方法标签
    for i, method in enumerate(methods):
        ax2.annotate(method.replace('_', '\n'), (aucs[i], scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 添加可信度区域
    ax2.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='高可信度')
    ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='中等可信度')
    ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='低可信度')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = "training_charts/baseline_comparison/credibility_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 可信度可视化已保存到: {chart_path}")

def main():
    """主函数"""
    print("🔍 基线结果真实性验证系统")
    print("=" * 60)
    print("目标: 检查所有基线方法结果的真实性和可信度")
    print("重点: 识别可疑的完美性能和不合理结果")
    print()
    
    try:
        # 加载基线结果
        results = load_baseline_results()
        
        # 分析可信度
        credibility_analysis = analyze_result_credibility(results)
        
        # 识别可疑模式
        suspicious_methods, reliable_methods = identify_suspicious_patterns(credibility_analysis)
        
        # 生成报告
        generate_credibility_report(credibility_analysis, suspicious_methods, reliable_methods)
        
        # 创建可视化
        create_credibility_visualization(credibility_analysis)
        
        # 显示总结
        print("\n" + "=" * 60)
        print("🎯 可信度分析总结:")
        
        print(f"\n🚨 高度可疑的方法:")
        for method_name, analysis in suspicious_methods:
            print(f"   • {method_name}: AUC={analysis['auc_mean']:.4f} (评分: {analysis['credibility_score']}/100)")
        
        print(f"\n✅ 相对可靠的方法:")
        for method_name, analysis in reliable_methods:
            print(f"   • {method_name}: AUC={analysis['auc_mean']:.4f} (评分: {analysis['credibility_score']}/100)")
        
        # 给出建议
        if suspicious_methods:
            print(f"\n💡 建议:")
            print("1. iPiDi-PUL的完美性能(AUC=1.0000)极度可疑，可能存在数据泄露")
            print("2. 应该重新实现可疑的基线方法或使用原始代码")
            print("3. 我们的模型应该主要与可靠的基线方法对比")
            
            # 找出最强的可靠基线
            if reliable_methods:
                best_reliable = max(reliable_methods, key=lambda x: x[1]['auc_mean'])
                print(f"4. 最强的可靠基线是 {best_reliable[0]} (AUC={best_reliable[1]['auc_mean']:.4f})")
                print(f"5. 我们的目标应该是超越 {best_reliable[1]['auc_mean']:.4f}")
        
        print(f"\n📁 结果保存位置:")
        print(f"   可信度报告: training_charts/baseline_comparison/credibility_report.txt")
        print(f"   可视化图表: training_charts/baseline_comparison/credibility_analysis.png")
        
        print("\n✅ 基线结果真实性验证完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 基线结果真实性验证成功完成！")
    else:
        print("\n💥 验证失败，请检查错误信息")
