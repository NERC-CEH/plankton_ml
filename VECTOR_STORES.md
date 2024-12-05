# Vector stores

Investigation of alternative vector stores for image model embeddings.

## ChromaDB

* "Simplest useful thing", default in the LangChain examples for LLM rapid prototyping
* Idiosyncratic, not standards-oriented
* Evolving quickly (a couple of back-incompatible API changes since starting with it)

## SQLite-vec

* Lightweight and helpful examples, quick to start with?
* Single process
* "_expect breaking changes!_"

https://til.simonwillison.net/sqlite/sqlite-vec

https://github.com/asg017/sqlite-vec

https://github.com/asg017/sqlite-vec/releases

```
pip install sqlite-utils
sqlite-utils install sqlite-utils-sqlite-vec
```

Main use is in the `streamlit` app which is _really_ tied to the internal logic of `chromadb` :/

Queries are

* get all identifiers (need `LIMIT` for large collection) - URLs were used directly as IDs
* get embeddings vector for one ID
* get N closest results to one set of embeddings by cosine similarity




