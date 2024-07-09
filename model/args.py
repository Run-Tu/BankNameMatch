from dataclasses import dataclass, field

@dataclass
class Args:
    encoder: str = field(
        default="model/models--TidalTail--FinQA-FlagEmbedding/snapshots/272edad9ab6cd0160d2baf3597f881c8d49f1de1",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=128,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=1,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    save_path: str = field(
        default="20240509_embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )