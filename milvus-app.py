from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from embedding_util import generate_embeddings


def connect_milvus():
    try:
        connections.connect('default', host='localhost', port='19530')
        print("Connected to Milvus")
    except Exception as e:
        print(f"Failed to connect to milvus - ", e)
        raise


def create_collection(name, fields, description, consistency_level='Strong'):
    schema = CollectionSchema(fields, description)
    collection = Collection(name, schema, consistency_level=consistency_level)
    return collection


def insert_data(collection, entities):
    insert_result = collection.insert(entities)
    collection.flush()
    print(f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")
    return insert_result


def create_index(collection, filed_name, index_type, metric_type, params):
    index = {"index_type": index_type, "metric_type": metric_type, "params": params}
    collection.create_index(field_name, index)
    print(f"Index '{index_type}' created for field: '{field_name}'.")


def search_and_query(collection, search_vectors, search_field, search_params):
    collection.load()
    result = collection.search(search_vectors, search_field, search_params, limit=3, output_fields=["source"])


def print_results(results, message):
    print(message)
    for hits in results:
        print(f"Hit: {hit}, source field: {hit.entity.get('source')}")


def delete_entities(collection, expr):
    collection.delete(expr)
    print(f"deleted entities where exp = '{expr}'")


def drop_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped collection name - '{collection_name}'")


dim = 768
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="source", dtype=DataType.VARCHAR, is_primary=True, max_length=500),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, is_primary=True, dim=dim)
]
collection = create_collection("Hello Milvus", fields, "Collection for demo purposes")

documents = [
    "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures,",
    "The mathematician found solace in numbers, deciphering the hidden patterns of the universe",
    "The robot, with its intricate circuitry and precise movements, assembles the device swiftly",
    "the chef, with a sprinkle of spices and a dash of love, creates cullinary masterpieces.",
    "The ancient tree, with its gnarled branches and deep roots,whispers secrets of the past",
    "The detective with keen observation and logical reasoning, unravels the web of truths",
    "The sunset paints the sky with shades of orange, pink, purple, reflecting on the calm sea",
    "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word"
]

embeddings = [generate_embeddings(doc) for doc in documents]
entities = [
    [str(i) for i in range(len(documents))],
    [str(doc) for doc in documents],
    embeddings
]
insert_result = insert_data(collection, entities)
create_index(collection, "embeddings", "IVF_FLAT", "L2", {"nlist": 128})
query = "Give me some oontent about the ocean"
query_vector = generate_embeddings(query)
search_and_query(collection, [query_vector], "embeddings", {"metric": "L2", "params": {"nprobe": 10}})
delete_entities(collection, f'pk in ["{insert_result.primary_keys[0]}", "{insert_result.primary_keys[1]}')
drop_collection("hello milvus")