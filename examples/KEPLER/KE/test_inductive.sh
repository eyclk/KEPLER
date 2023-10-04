#!/bin/bash

python evaluate_transe_inductive.py --entity_embeddings files_for_inductive/EntityEmb_dummy.npy \
		--relation_embeddings files_for_inductive/RelEmb_dummy.npy \
		--dim 768 \
		--entity2id files_for_inductive/entity2id_dummy.json \
		--relation2id files_for_inductive/relation2id.json \
		--dataset files_for_inductive/wikidata5m_test_dummy.txt

