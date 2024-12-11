# TODO manage this better elsewhere, once we settle on a storage option
SQLITE_SCHEMA = """
    create virtual table embeddings using vec0(
    id integer primary key,
    url text not null,
    classification text not null,
    embedding float[{}]);
"""

# Options passed as keyword arguments when setting a db connection
OPTIONS = {"sqlite": {"embedding_len": 512, "check_same_thread": False}, "chromadb": {}}
