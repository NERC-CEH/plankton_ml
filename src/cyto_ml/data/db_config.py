# TODO manage this better elsewhere, once we settle on a storage option
SQLITE_SCHEMA = """
    create virtual table embeddings using vec0(
    id integer primary key,
    url text not null,
    embedding float[{}]);
"""
