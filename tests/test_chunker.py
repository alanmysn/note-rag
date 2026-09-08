import tempfile
import unittest
from pathlib import Path

from app.chunker import build_chunks, chunk_text, split_long


class ChunkTests(unittest.TestCase):
    def test_short_paragraphs_stop_at_target_without_slicing(self):
        parts = ['甲' * 160, '乙' * 180, '丙' * 146, '丁' * 100]
        chunks = chunk_text('\n\n'.join(parts))
        self.assertEqual([c.body for c in chunks], ['\n\n'.join(parts[:3]), parts[3]])
        self.assertEqual(len(chunks[0].body), 490)

    def test_300_and_250_are_not_merged(self):
        self.assertEqual(len(chunk_text('甲' * 300 + '\n\n' + '乙' * 250)), 2)

    def test_complete_650_character_paragraph_stays_intact(self):
        self.assertEqual([c.body for c in chunk_text('甲' * 650)], ['甲' * 650])

    def test_heading_and_separator_boundaries(self):
        chunks = chunk_text('# 上级\n介绍\n## 甲\n第一段\n---\n第二段\n## 乙\n第三段')
        self.assertEqual([c.title_chain for c in chunks], ['上级', '上级 > 甲', '上级 > 甲', '上级 > 乙'])
        self.assertEqual(len({c.parent_id for c in chunks}), 4)

    def test_bold_heading_retains_markdown_parent(self):
        chunks = chunk_text('# 上级\n## 下级\n**加粗一**\n甲\n**加粗二**\n乙')
        self.assertEqual([c.title_chain for c in chunks], ['上级 > 下级 > 加粗一', '上级 > 下级 > 加粗二'])

    def test_skipped_heading_levels(self):
        chunks = chunk_text('# 上级\n### 甲\n一\n### 乙\n二')
        self.assertEqual(chunks[1].title_chain, '上级 > 乙')

    def test_long_paragraph_preserves_original_and_sentence_boundary(self):
        original = '甲' * 449 + '。' + '乙' * 449 + '。' + '丙' * 200
        chunks = chunk_text(original)
        self.assertEqual(''.join(c.body for c in chunks), original)
        self.assertEqual(chunks[0].body[-1], '。')
        self.assertEqual({c.context for c in chunks}, {original})
        self.assertEqual(len({c.parent_id for c in chunks}), 1)
        self.assertTrue(all(len(c.body) <= 800 for c in chunks))

    def test_long_sentence_uses_clause_boundary_then_hard_split(self):
        text = '甲' * 499 + '，' + '乙' * 900
        parts = split_long(text, 500, 800)
        self.assertEqual(parts[0][-1], '，')
        self.assertEqual(''.join(parts), text)
        self.assertTrue(all(len(p) <= 800 for p in parts))

    def test_list_intro_and_nested_items_stay_together(self):
        text = '原因如下：\n\n1. 耗材贵\n2. 连接不好\n   - 安卓\n   - 苹果\n3. 维修难'
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].context, text)

    def test_long_list_splits_at_top_level_items(self):
        intro = '原因如下：'
        first = '1. ' + '甲' * 300 + '\n   - 子项目'
        second = '2. ' + '乙' * 300
        third = '3. ' + '丙' * 300
        text = intro + '\n\n' + '\n'.join([first, second, third])
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].body, intro + '\n\n' + first)
        self.assertIn('2. ', chunks[1].body)
        self.assertEqual({c.context for c in chunks}, {text})
        self.assertEqual(len({c.parent_id for c in chunks}), 1)

    def test_blank_lines_inside_list_and_following_paragraph(self):
        text = '提示：\n\n- 甲\n\n- 乙\n\n这是另一个段落'
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].context, '提示：\n\n- 甲\n\n- 乙')
        self.assertEqual(chunks[1].body, '这是另一个段落')

    def test_oversized_list_item_and_intro_are_bounded(self):
        text = '引' * 850 + '：\n\n1. ' + '甲' * 1800 + '\n2. 乙'
        chunks = chunk_text(text)
        self.assertTrue(all(0 < len(c.body) <= 800 for c in chunks))
        self.assertEqual({c.context for c in chunks}, {text})
        self.assertIn('2. 乙', chunks[-1].body)

    def test_code_fence_does_not_create_headings(self):
        chunks = chunk_text('# 真实\n```python\n# 注释\n---\n```')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title_chain, '真实')
        self.assertIn('---', chunks[0].body)

    def test_file_name_is_in_embedding_and_source_is_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / 'folder' / 'note.md'
            note.parent.mkdir()
            note.write_text('# 标题\n正文', encoding='utf-8')
            items = build_chunks({'vault_path': tmp, 'chunk_overlap': 50})
            self.assertEqual(note.read_text(encoding='utf-8'), '# 标题\n正文')
        self.assertEqual(items[0][0], Path('folder/note.md'))
        self.assertEqual(items[0][1][0].text, '[note.md > 标题]\n正文')

    def test_invalid_sizes(self):
        for target, maximum in [(0, 800), (900, 800), (500, 0)]:
            with self.assertRaises(ValueError):
                chunk_text('文字', target, maximum)


if __name__ == '__main__':
    unittest.main()
