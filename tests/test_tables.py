import unittest
from app.chunker import chunk_text


class TableAndMetadataTests(unittest.TestCase):
    def test_frontmatter_skipped_and_body_separator_kept(self):
        chunks = chunk_text('---\ntags:\n  - 标签\naliases: [别名]\n---\n甲\n---\n乙')
        self.assertEqual([c.body for c in chunks], ['甲', '乙'])

    def test_unclosed_frontmatter_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, '属性区'):
            chunk_text('---\ntags: [标签]')

    def test_short_explanation_without_colon_attached(self):
        text = '等价于动力\n\n- 第一项\n- 第二项'
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].context, text)

    def test_table_row_boundaries_and_repeated_header(self):
        header = '| 名称 | 说明 |\n| --- | --- |'
        rows = ['| 甲 | ' + '甲' * 300 + ' |', '| 乙 | ' + '乙' * 300 + ' |', '| 丙 | ' + '丙' * 300 + ' |']
        intro = '比较结果'
        text = intro + '\n\n' + header + '\n' + '\n'.join(rows)
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 3)
        for row, chunk in zip(rows, chunks):
            self.assertEqual(chunk.body, intro + '\n\n' + header + '\n' + row)
            self.assertEqual(chunk.context, text)
        self.assertEqual(len({c.parent_id for c in chunks}), 1)

    def test_single_oversized_row_is_not_cut(self):
        text = '| 列 | 内容 |\n| --- | --- |\n| 甲 | ' + '长' * 900 + ' |'
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].body, text)

    def test_continuous_steps_bridge_table_and_restore_all(self):
        table = '| 项目 | 内容 |\n| --- | --- |\n| 甲 | ' + '长' * 900 + ' |'
        text = '1. 一\n2. 二\n3. 三\n\n' + table + '\n\n4. 四\n5. 五'
        chunks = chunk_text(text)
        self.assertEqual(len({c.parent_id for c in chunks}), 1)
        self.assertTrue(all(c.context == text for c in chunks))
        self.assertTrue(any(c.body.startswith('| 项目') for c in chunks))
        self.assertTrue(any('4. 四' in c.body for c in chunks))

    def test_noncontinuous_steps_do_not_bridge(self):
        table = '| 项目 | 内容 |\n| --- | --- |\n| 甲 | 乙 |'
        for after in ['1. 重启', '- 无序', '插入正文\n\n4. 四', '---\n4. 四', '## 新标题\n4. 四']:
            chunks = chunk_text('3. 三\n\n' + table + '\n\n' + after)
            self.assertGreater(len({c.parent_id for c in chunks}), 1)
            self.assertFalse(any('3. 三' in c.context and after in c.context for c in chunks))

    def test_long_intro_stays_in_context_without_repeating(self):
        text = '引' * 700 + '：\n\n| 名称 | 说明 |\n| --- | --- |\n| 甲 | ' + '长' * 900 + ' |'
        chunks = chunk_text(text)
        self.assertTrue(all(c.context == text for c in chunks))
        self.assertTrue(chunks[-1].body.startswith('| 名称'))

    def test_table_inside_fence_is_not_a_table_unit(self):
        text = '```text\n| A | B |\n|---|---|\n|1|2|\n```'
        chunks = chunk_text(text)
        self.assertEqual(chunks[0].body, text)


if __name__ == '__main__':
    unittest.main()
