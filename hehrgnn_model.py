import torch

from graph_encoder import GraphEncoder
from graph_decoder import GraphDecoder


class hehrgnn_model(torch.nn.Module):
    """Encoder (GNN) + decoder (DistMult/HypE) wrapped as a single module so that
    training, evaluation, optimizer and checkpointing all operate on one object."""

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.graph_encoder = GraphEncoder(dataset, config)
        # Node classification reads class logits straight off the encoder output.
        self.graph_decoder = GraphDecoder(dataset, config) if config.task == "link_prediction" else None

    def load_model(self, saved_model):
        self.load_state_dict(torch.load(saved_model, map_location=self.config.device))
        print(f"Saved model loaded from {saved_model}")

    def forward(self, graph_data, task, input_for_pred):
        ent_embed, rel_embed = self.graph_encoder(graph_data, task)
        if task == "node_classification":
            return ent_embed
        return self.graph_decoder(ent_embed, rel_embed, input_for_pred)
