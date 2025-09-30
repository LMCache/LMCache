#!/usr/bin/env python3
"""
检查 LMCache 文档的中文翻译进度
"""
import os
import re
from pathlib import Path
from collections import defaultdict

def parse_po_file(po_file_path):
    """解析 .po 文件，返回翻译统计信息"""
    with open(po_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 统计所有 msgid/msgstr 对
    pattern = r'msgid\s+"([^"]*)"\nmsgstr\s+"([^"]*)"'
    matches = re.findall(pattern, content, re.MULTILINE)
    
    total = 0
    translated = 0
    untranslated_items = []
    
    for msgid, msgstr in matches:
        if msgid:  # 跳过空的 msgid
            total += 1
            if msgstr:
                translated += 1
            else:
                untranslated_items.append(msgid)
    
    return {
        'total': total,
        'translated': translated,
        'untranslated': total - translated,
        'percentage': (translated / total * 100) if total > 0 else 0,
        'untranslated_items': untranslated_items
    }

def main():
    locale_dir = Path(__file__).parent / 'source' / 'locale' / 'zh_CN' / 'LC_MESSAGES'
    
    if not locale_dir.exists():
        print("错误：找不到翻译文件目录")
        return
    
    print("=" * 80)
    print("LMCache 文档中文翻译进度报告")
    print("=" * 80)
    print()
    
    stats_by_dir = defaultdict(lambda: {'total': 0, 'translated': 0})
    all_stats = {'total': 0, 'translated': 0}
    files_needing_work = []
    
    # 遍历所有 .po 文件
    for po_file in sorted(locale_dir.rglob('*.po')):
        stats = parse_po_file(po_file)
        
        if stats['total'] == 0:
            continue
        
        relative_path = po_file.relative_to(locale_dir)
        dir_name = str(relative_path.parent) if relative_path.parent != Path('.') else '根目录'
        
        # 按目录分类统计
        stats_by_dir[dir_name]['total'] += stats['total']
        stats_by_dir[dir_name]['translated'] += stats['translated']
        
        # 总体统计
        all_stats['total'] += stats['total']
        all_stats['translated'] += stats['translated']
        
        # 记录需要翻译的文件
        if stats['untranslated'] > 0:
            files_needing_work.append({
                'file': relative_path,
                'stats': stats
            })
    
    # 打印总体进度
    overall_percentage = (all_stats['translated'] / all_stats['total'] * 100) if all_stats['total'] > 0 else 0
    print(f"📊 总体进度: {all_stats['translated']}/{all_stats['total']} ({overall_percentage:.1f}%)")
    print(f"   ✅ 已翻译: {all_stats['translated']}")
    print(f"   ⏳ 待翻译: {all_stats['total'] - all_stats['translated']}")
    print()
    
    # 打印各目录的进度
    print("📁 各目录翻译进度:")
    print("-" * 80)
    for dir_name in sorted(stats_by_dir.keys()):
        stats = stats_by_dir[dir_name]
        percentage = (stats['translated'] / stats['total'] * 100) if stats['total'] > 0 else 0
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"{dir_name:40} {bar} {percentage:5.1f}% ({stats['translated']}/{stats['total']})")
    print()
    
    # 打印需要翻译的文件（按待翻译数量排序）
    if files_needing_work:
        print("📝 需要翻译的文件（按待翻译条目数排序）:")
        print("-" * 80)
        files_needing_work.sort(key=lambda x: x['stats']['untranslated'], reverse=True)
        
        for i, item in enumerate(files_needing_work[:20], 1):  # 只显示前20个
            file_path = item['file']
            stats = item['stats']
            print(f"{i:2}. {file_path}")
            print(f"    待翻译: {stats['untranslated']} 条 ({stats['percentage']:.1f}% 已完成)")
        
        if len(files_needing_work) > 20:
            print(f"\n... 还有 {len(files_needing_work) - 20} 个文件需要翻译")
    else:
        print("🎉 恭喜！所有文件都已翻译完成！")
    
    print()
    print("=" * 80)
    print("💡 提示:")
    print("   - 运行 'make i18n-update' 更新翻译文件")
    print("   - 运行 'make html-zh' 构建中文文档")
    print("   - 编辑 source/locale/zh_CN/LC_MESSAGES/ 下的 .po 文件进行翻译")
    print("=" * 80)

if __name__ == '__main__':
    main() 