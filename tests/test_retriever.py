import unittest
from app.chunker import CHUNK_VERSION, chunk_text
from app.retriever import restore_hits


def results_for(chunks, scores, sources=None):
    return {
        'documents': [[c.text for c in chunks]],
        'metadatas': [[{'source': sources[i] if sources else 'note.md',
                       'title_chain': c.title_chain, 'parent_id': c.parent_id,
                       'context': c.context, 'chunk_version': CHUNK_VERSION}
                      for i, c in enumerate(chunks)]],
        'distances': [[1 - s for s in scores]],
    }


class RestoreTests(unittest.TestCase):
    def test_restores_whole_paragraph_without_neighbors(self):
        paragraph = '甲' * 1800
        chunks = chunk_text(paragraph + '\n\n不应自动补充的邻段')
        hits = restore_hits(results_for([chunks[1]], [0.8]), 0.6)
        self.assertEqual(hits[0]['text'], paragraph)
        self.assertEqual(hits[0]['matched_chunks'], [chunks[1].text])

    def test_multiple_hits_deduplicated_with_highest_score(self):
        chunks = chunk_text('甲' * 1800)
        hits = restore_hits(results_for(chunks, [0.7, 0.9, 0.8]), 0.6)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['score'], 0.9)
        self.assertEqual(len(hits[0]['matched_chunks']), 3)

    def test_threshold_applied_before_context_expansion(self):
        chunks = chunk_text('甲' * 1800)
        hits = restore_hits(results_for(chunks[:2], [0.7, 0.5]), 0.6)
        self.assertEqual(len(hits[0]['matched_chunks']), 1)
        self.assertEqual(len(hits[0]['text']), 1800)
        self.assertEqual(restore_hits(results_for(chunks[:1], [0.5]), 0.6), [])

    def test_different_files_not_deduplicated(self):
        chunk = chunk_text('相同文字')[0]
        hits = restore_hits(results_for([chunk, chunk], [0.7, 0.8], ['a.md', 'b.md']), 0.6)
        self.assertEqual([h['file'] for h in hits], ['b.md', 'a.md'])

    def test_long_list_restored_in_full(self):
        text = '注意：\n\n1. ' + '甲' * 600 + '\n2. ' + '乙' * 600
        chunks = chunk_text(text)
        hits = restore_hits(results_for([chunks[-1]], [0.8]), 0.6)
        self.assertEqual(hits[0]['text'], text)

    def test_old_index_rejected(self):
        result = results_for(chunk_text('正文'), [0.8])
        del result['metadatas'][0][0]['chunk_version']
        with self.assertRaisesRegex(ValueError, '重建索引'):
            restore_hits(result, 0.6)


if __name__ == '__main__':
    unittest.main()
