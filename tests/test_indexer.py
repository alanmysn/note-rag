import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.chunker import CHUNK_VERSION, chunk_text
from app.indexer import build_index, validate_embedding_lengths
from app.retriever import restore_hits


class IndexContractTests(unittest.TestCase):
    def test_index_stores_full_context_but_embeds_small_chunks(self):
        original = '甲' * 1800
        chunks = chunk_text(original, file_name='note.md')
        collection = MagicMock()
        collection.get.return_value = {'ids': []}
        vectors = MagicMock()
        vectors.tolist.return_value = [[1.0, 0.0]] * len(chunks)
        embedder = MagicMock()
        embedder.max_seq_length = 8192
        embedder.tokenizer.return_value = {'input_ids': [[1, 2, 3]] * len(chunks)}
        embedder.encode.return_value = vectors
        with patch('app.indexer.load_config', return_value=({}, Path('.'), Path('data'))), \
             patch('app.indexer.build_chunks', return_value=[(Path('folder/note.md'), chunks)]), \
             patch('app.indexer._get_collection', return_value=collection), \
             patch('app.indexer._get_embedder', return_value=embedder):
            stats = build_index()
        payload = collection.add.call_args.kwargs
        self.assertEqual(stats, {'files': 1, 'blocks': 3})
        self.assertEqual(embedder.encode.call_args.args[0], [c.text for c in chunks])
        self.assertTrue(all(m['context'] == original for m in payload['metadatas']))
        self.assertTrue(all(m['source'] == 'folder/note.md' for m in payload['metadatas']))
        hits = restore_hits({'documents': [payload['documents']],
                             'metadatas': [payload['metadatas']],
                             'distances': [[0.2] * len(chunks)]}, 0.6)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['text'], original)
        collection.modify.assert_called_once_with(
            metadata={'hnsw:space': 'cosine', 'chunk_version': CHUNK_VERSION})

    def test_oversized_input_fails_without_truncation(self):
        embedder = MagicMock()
        embedder.max_seq_length = 3
        embedder.tokenizer.return_value = {'input_ids': [[1, 2, 3, 4]]}
        with self.assertRaisesRegex(ValueError, '超过向量模型容量'):
            validate_embedding_lengths(embedder, ['table'])
        self.assertFalse(embedder.tokenizer.call_args.kwargs['truncation'])


if __name__ == '__main__':
    unittest.main()
