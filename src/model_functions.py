import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv

def build_graph_from_frame(frame_df, runner_player_id, player_to_team):
    """
    Build a fully connected, edge-attributed graph for a single tracking frame.

    Each player and the ball are represented as nodes with kinematic features
    (position, velocity, speed, acceleration) and binary indicators for
    runner identity and ball identity. Directed edges are created between all
    node pairs (excluding self-loops) and encode spatial and relational context,
    including inter-node distance, same-team membership, ball involvement, and
    relative speed along the interaction direction.

    The function also identifies and stores the node index corresponding to
    the designated runner, enabling runner-centric representations in
    downstream graph neural network models.

    Args:
        frame_df (pd.DataFrame): Tracking data for a single frame, with one row
            per player and shared ball information.
        runner_player_id (int): Player ID of the runner of interest.
        player_to_team (pd.DataFrame): Mapping from player IDs to team IDs.

    Returns:
        torch_geometric.data.Data: A graph object containing node features,
        edge indices, edge attributes, runner index, team IDs, and ball mask.
    """
    frame_df = frame_df.sort_values("player").reset_index(drop=True)

    player_ids = frame_df["player"].values
    Np = len(player_ids)

    #player nodes

    player_feats = torch.tensor(
        frame_df[[
            "x", "y",
            "dx", "dy",
            "s",
            "d",
            "acceleration",
            "acc_direction"
        ]].values,
        dtype=torch.float
    )
    # player team ids
    team_ids = torch.tensor(
        player_to_team.loc[player_ids, "team_id"].values,
        dtype=torch.long
    )

    # runner flag for players
    is_runner_player = torch.tensor(
        (player_ids == runner_player_id).astype(float),
        dtype=torch.float
    ).unsqueeze(1)

    is_runner = torch.cat(
        [is_runner_player, torch.zeros(1, 1)],
        dim=0
    )
    # is the node the ball
    is_ball = torch.zeros(Np + 1, dtype=torch.bool)

    #ball node

    ball_feats = torch.tensor(
        [[
            frame_df["ball_x"].iloc[0],
            frame_df["ball_y"].iloc[0],
            frame_df["ball_dx"].iloc[0],
            frame_df["ball_dy"].iloc[0],
            frame_df["ball_speed"].iloc[0],
            frame_df["ball_speed_direction"].iloc[0],
            frame_df["ball_acceleration"].iloc[0],
            frame_df["ball_acc_direction"].iloc[0],
        ]],
        dtype=torch.float
    )

    #node features

    x = torch.cat([
        player_feats,
        ball_feats
    ], dim=0)

    is_ball[-1] = True
    is_ball_feat = is_ball.float().unsqueeze(1)

    team_ids = torch.cat([
        team_ids,
        torch.tensor([-1])
    ])

    x = torch.cat([
        x,
        is_runner,
        is_ball_feat
    ], dim=1)

    pos = x[:, :2]
    vel = x[:, 2:4]
    N = pos.size(0)

    # Edges and edge attributes
    device = pos.device
    row, col = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        indexing="ij"
    )

    mask = row != col
    row = row[mask]
    col = col[mask]

    edge_index = torch.stack([row, col], dim=0)

    rel_pos = pos[col] - pos[row]              
    dist = torch.norm(rel_pos, dim=1)           

    direction = rel_pos / (dist.unsqueeze(1) + 1e-6)
    # speed difference between the two players/ball
    rel_vel = vel[col] - vel[row]  
    rel_speed = torch.sum(rel_vel * direction, dim=1)
    # are the two players on the same team
    same_team = (
        (team_ids[row] == team_ids[col]) &
        (team_ids[row] != -1)
    ).float()
    # is one of the nodes the ball
    ball_edge = (is_ball[row] | is_ball[col]).float()

    edge_attr = torch.stack(
        [dist, same_team, ball_edge, rel_speed],
        dim=1
    )
    # what node index is the runner
    runner_idx = int(
        torch.where(
            torch.tensor(player_ids) == runner_player_id
        )[0].item()
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        runner_idx=torch.tensor(runner_idx, dtype=torch.long),
        team_ids=team_ids,
        is_ball=is_ball
    )

class TemporalRunnerDataset(Dataset):
    """
    Dataset for modeling off-ball runner behavior using spatio-temporal
    player–ball interaction graphs.

    Each dataset item corresponds to a single annotated run event and returns
    a sequence of graph-structured frames spanning a short pre-run context
    and the full run duration. The first 10 frames precede the run onset and
    provide contextual information for intent and run-type inference, while
    subsequent frames correspond to the executed run trajectory.

    The target trajectory is expressed in runner-centric coordinates, anchored
    at the runner's position at run start, and includes only frames during
    the run itself (excluding pre-run context). Additional supervision signals
    include binary labels indicating whether the possession led to a shot or
    goal, as well as a categorical run-type label.

    Args:
        tracking_df (dict-like): Mapping from (match_id, event_id) to per-frame
            tracking DataFrames containing player and ball states.
        run_features (pd.DataFrame): Run-level metadata including timing,
            runner identity, run subtype, and outcome labels.
        player_to_team (pd.DataFrame): Mapping from player IDs to team IDs.
        run_type_vocab (dict): Mapping from run subtype strings to integer IDs.

    Returns (per __getitem__):
        graphs (List[torch_geometric.data.Data]):
            Sequence of graph objects, one per frame, including 10 pre-run
            context frames followed by run frames.
        target_path (torch.FloatTensor):
            Tensor of shape (T_run, 2) containing the runner's ground-truth
            displacement trajectory during the run.
        shot_label (torch.FloatTensor):
            Binary indicator for whether the possession resulted in a shot.
        goal_label (torch.FloatTensor):
            Binary indicator for whether the possession resulted in a goal.
        run_type_id (int):
            Integer-encoded run subtype label.
    """
    def __init__(self, tracking_df, run_features, player_to_team, run_type_vocab):
        self.df = tracking_df
        self.run_features = run_features
        self.player_to_team = player_to_team
        self.run_vocab = run_type_vocab

    def __len__(self):
        return len(self.run_features)

    def __getitem__(self, idx):
        run = self.run_features.iloc[idx]

        match_id = run["match_id"]
        run_id = run["event_id"]
        runner_id = run["player_id"]
        run_type = run["event_subtype"]
        run_type_id = self.run_vocab[run_type]

        frame_start = int(run["frame_start_run"])
        frame_end   = int(run["frame_end_run"])

        run_tracking = self.df[(match_id, run_id)]
        run_tracking = run_tracking.sort_values("frame_id")

        graphs = []
        runner_positions = []

        # --- anchor position ---
        start_row = run_tracking[
            (run_tracking["frame_id"] == frame_start) &
            (run_tracking["player"] == runner_id)
        ].iloc[0]

        x0, y0 = start_row["x"], start_row["y"]

        for t in range(frame_start - 10, frame_end + 1):
            frame_df = run_tracking[run_tracking["frame_id"] == t]

            g = build_graph_from_frame(
                frame_df=frame_df,
                runner_player_id=runner_id,
                player_to_team=self.player_to_team
            )
            graphs.append(g)
            if t>=frame_start:
                runner_row = frame_df[frame_df["player"] == runner_id].iloc[0]
                runner_positions.append([
                    runner_row["x"] - x0,
                    runner_row["y"] - y0
                ])

        target_path = torch.tensor(runner_positions, dtype=torch.float) 
        
        lead_to_shot_val = self.run_features.loc[self.run_features['id'] == run_id, 'possession_lead_to_shot'].values
        if len(lead_to_shot_val) == 0:
            shot_label = torch.tensor(0.0) 
        else:
            shot_label = torch.tensor(float(lead_to_shot_val[0]))
        
        lead_to_goal_val = self.run_features.loc[self.run_features['id'] == run_id, 'possession_lead_to_goal'].values
        if len(lead_to_goal_val) == 0:
            goal_label = torch.tensor(0.0) 
        else:
            goal_label = torch.tensor(float(lead_to_goal_val[0]))
        

        return graphs, target_path, shot_label, goal_label, run_type_id

def collate_fn(batch, max_len=100):
    """
    Collate function to combine individual dataset samples into a batch.

    Args:
        batch (list): List of tuples from TemporalRunnerDataset __getitem__,
            each containing (graphs, target_path, shot_label, goal_label, run_type_id).
        max_len (int): Maximum sequence length for padding target trajectories.

    Returns:
        batch_graphs (torch_geometric.data.Batch): Batched graph data combining
            all frame graphs from the batch samples.
        padded_targets (torch.FloatTensor): Tensor of shape (batch_size, max_len, 2)
            with zero-padded runner trajectories.
        lengths (list[int]): List of original sequence lengths (capped at max_len).
        shot_labels (torch.FloatTensor): Tensor of shot outcome labels.
        goal_labels (torch.FloatTensor): Tensor of goal outcome labels.
        run_type_labels (torch.LongTensor): Tensor of integer run-type labels.
    """
    graphs_list, targets_list, shot_labels_list, goal_labels_list, run_type_id_list = zip(*batch)

    batch_graphs = Batch.from_data_list([g for graphs in graphs_list for g in graphs])

    lengths = [len(target) for target in targets_list]

    padded_targets = torch.zeros(len(targets_list), max_len, 2) 

    for i, target in enumerate(targets_list):
        length = min(len(target), max_len)
        padded_targets[i, :length] = target[:length]

    # Adjust lengths to max_len if longer sequences exist
    lengths = [min(l, max_len) for l in lengths]
    shot_labels = torch.tensor(shot_labels_list, dtype=torch.float)
    goal_labels = torch.tensor(goal_labels_list, dtype=torch.float)
    run_type_labels = torch.tensor(run_type_id_list, dtype=torch.long)


    return batch_graphs, padded_targets, lengths, shot_labels, goal_labels, run_type_labels

class TemporalRunnerGNN(nn.Module):
    """
    Temporal graph neural network for predicting runner intents, shot probabilities,
    and future trajectories in soccer event data.

    Architecture:
    - Two-layer Transformer-based GNN for spatial encoding of player and ball features.
    - Temporal Transformer encoder for intent prediction from the first 10 frames.
    - Temporal Transformer encoder on all frames for shot probability estimation and path decoding.
    - Embedding-based conditioning on run types for counterfactual trajectory prediction.

    Inputs:
    - graphs_batch: Batched graph data for all frames in a run.
    - lengths: List of sequence lengths per run.

    Outputs:
    - run_type_logits: Raw logits for run type classification.
    - run_type_probs: Softmax probabilities for run types.
    - shot_logits_per_type: Raw shot probability logits per run type.
    - shot_probs_per_type: Softmax shot probabilities per run type.
    - run_paths_per_type: Predicted future trajectories per run type.
    """
    def __init__(self, node_feat_dim, edge_dim, gnn_hidden_dim, K):
        super().__init__()

        self.gnn1 = TransformerConv(
            in_channels=node_feat_dim,
            out_channels=gnn_hidden_dim,
            heads=4,
            concat=False,
            edge_dim=edge_dim,
            dropout = .1
        )

        self.gnn2 = TransformerConv(
            in_channels=gnn_hidden_dim,
            out_channels=gnn_hidden_dim,
            heads=4,
            concat=False,
            edge_dim=edge_dim,
            dropout=.1
        )
        self.run_type_dim = 16

        self.temporal_proj = nn.Linear(
            gnn_hidden_dim + self.run_type_dim,
            gnn_hidden_dim
        )


        self.temporal_pe = nn.Parameter(
            torch.randn(1, 100, gnn_hidden_dim) * 0.01 
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden_dim,
            nhead=4,
            dim_feedforward=4 * gnn_hidden_dim,
            dropout=0.1,
            batch_first=True
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        num_run_types = K
        run_type_dim = 16

        self.run_type_head = nn.Linear(gnn_hidden_dim, num_run_types)
        self.run_type_emb = nn.Embedding(num_run_types, run_type_dim)
        
        self.pos_head = nn.Linear(gnn_hidden_dim, 2)
        self.shot_head = nn.Linear(gnn_hidden_dim, num_run_types) 
        
    def forward(self, graphs_batch, lengths):
        device = next(self.parameters()).device
        graphs_batch = graphs_batch.to(device)

        x, edge_index, edge_attr = (
            graphs_batch.x,
            graphs_batch.edge_index,
            graphs_batch.edge_attr,
        )

        edge_attr = edge_attr / edge_attr.std(dim=0, keepdim=True).clamp(min=1e-6)

        # GNN encoder
        x = F.relu(self.gnn1(x, edge_index, edge_attr))
        x = F.relu(self.gnn2(x, edge_index, edge_attr))

        batch_vec = graphs_batch.batch
        runner_idx = graphs_batch.runner_idx
        total_graphs = runner_idx.size(0)

        B = len(lengths)
        max_len = max(lengths)

        # Extract runner embedding per graph
        runner_embeds_per_graph = []
        for t in range(total_graphs):
            node_mask = batch_vec == t
            node_indices = node_mask.nonzero(as_tuple=False).view(-1)
            global_runner_idx = node_indices[runner_idx[t]]
            runner_embeds_per_graph.append(x[global_runner_idx])

        runner_embeds_per_run = []
        start = 0
        for length in lengths:
            run_embeds = runner_embeds_per_graph[start : start + length]
            start += length

            if length < max_len:
                pad = [torch.zeros_like(run_embeds[0])] * (max_len - length)
                run_embeds.extend(pad)

            runner_embeds_per_run.append(torch.stack(run_embeds))

        runner_embeds_batch = torch.stack(runner_embeds_per_run) 

        B, T, H = runner_embeds_batch.shape
        T_intent = 10

        # 1) INTENT ENCODER (FIRST 10 FRAMES ONLY)
        x_intent = runner_embeds_batch[:, :T_intent]
        x_intent = x_intent + self.temporal_pe[:, :T_intent]

        intent_mask = torch.triu(
            torch.ones(T_intent, T_intent, device=device), diagonal=1
        ).bool()

        intent_out = self.temporal_encoder(
            x_intent,
            mask=intent_mask,
        )

        intent_state = intent_out[:, -1]

        # Run type prediction (early intent)
        run_type_logits = self.run_type_head(intent_state)
        run_type_probs = F.softmax(run_type_logits, dim=-1)

        # 2) FULL-RUN ENCODER (ALL FRAMES)
        x_full = runner_embeds_batch + self.temporal_pe[:, :T]

        full_mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()

        padding_mask = (
            torch.arange(T, device=device)[None, :]
            >= torch.tensor(lengths, device=device)[:, None]
        )

        full_out = self.temporal_encoder(
            x_full,
            mask=full_mask,
            src_key_padding_mask=padding_mask,
        )

        full_state = full_out[:, -1]  

        # Shot probability per run type 
        shot_logits_per_type = self.shot_head(full_state) 
        shot_probs_per_type = F.softmax(shot_logits_per_type, dim=-1)

        # 3) PATH DECODER (ALL FRAMES, PER RUN TYPE)
        run_paths = []

        for k in range(self.run_type_emb.num_embeddings):
            run_type_embed = self.run_type_emb.weight[k]
            run_type_embed = (
                run_type_embed.unsqueeze(0)
                .unsqueeze(0)
                .expand(B, T, -1)
            )

            x_path = torch.cat([runner_embeds_batch, run_type_embed], dim=-1)
            x_path = self.temporal_proj(x_path)
            x_path = x_path + self.temporal_pe[:, :T]

            path_out = self.temporal_encoder(
                x_path,
                mask=full_mask,
                src_key_padding_mask=padding_mask,
            )

            pred_path = self.pos_head(path_out) 
            run_paths.append(pred_path)

        run_paths = torch.stack(run_paths, dim=1) 

        return {
            "run_type_logits": run_type_logits,
            "run_type_probs": run_type_probs,
            "shot_logits_per_type": shot_logits_per_type,
            "shot_probs_per_type": shot_probs_per_type,
            "run_paths_per_type": run_paths,
        }

def trajectory_diversity_loss(run_paths, lengths, eps=1e-6):
    """
    Computes a diversity loss to encourage predicted trajectories for different run types
    to be distinct within each sample in the batch.

    The loss is based on the average pairwise cosine similarity between flattened run paths,
    penalizing similar trajectories across run types.

    Args:
        run_paths (torch.Tensor): Predicted trajectories of shape (B, K, T, 2), where
            B = batch size,
            K = number of run types,
            T = trajectory length (time steps),
            2 = (x, y) position coordinates.
        lengths (list[int]): List of actual trajectory lengths per batch sample (<= T).
        eps (float, optional): Small constant for numerical stability (not used here but included for extensibility).

    Returns:
        torch.Tensor: Scalar tensor representing the average diversity loss over the batch.
                      Lower values mean less similarity (more diversity) among run type trajectories.
    """
    B, K, T, _ = run_paths.shape
    loss = 0.0
    count = 0

    for i in range(B):
        T_i = lengths[i]
        paths = run_paths[i, :, :T_i]  
        paths_flat = paths.reshape(K, -1)  
        paths_flat = F.normalize(paths_flat, dim=1)
        sim = paths_flat @ paths_flat.T  

        loss += (sim.sum() - torch.trace(sim)) / (K * (K - 1))
        count += 1

    return loss / count

def endpoint_diversity_loss(run_paths, lengths):
    """
    Computes a diversity loss that encourages the predicted trajectory endpoints
    for different run types within each batch sample to be distinct.

    The loss is defined as the negative mean pairwise Euclidean distance between
    the endpoints of predicted trajectories across run types, averaged over the batch.
    By minimizing this loss, the model is encouraged to predict more diverse endpoints.

    Args:
        run_paths (torch.Tensor): Predicted trajectories of shape (B, K, T, 2), where
            B = batch size,
            K = number of run types,
            T = trajectory length (time steps),
            2 = (x, y) position coordinates.
        lengths (list[int]): List of actual trajectory lengths per batch sample (<= T).

    Returns:
        torch.Tensor: Scalar tensor representing the average endpoint diversity loss over the batch.
                      Lower values correspond to less endpoint similarity (more diverse endpoints).
    """
    B, K, _, _ = run_paths.shape
    loss = 0.0

    for i in range(B):
        T_i = lengths[i]
        endpoints = run_paths[i, :, T_i-1]  # (K, 2)

        dists = torch.cdist(endpoints, endpoints, p=2)
        loss += -dists.mean()

    return loss / B

def train_model(model, device, dataloader, num_epochs=10, output_file="temporal_runner_gnn_v4.pth"):
    """
    Trains the TemporalRunnerGNN model with curriculum learning over multiple epochs.

    The training process uses a combination of losses including:
      - Run-type classification loss
      - Shot probability loss (only for the true run type)
      - Trajectory position, velocity, speed, and acceleration losses
      - Trajectory diversity loss to encourage distinct predicted paths across run types
      - Endpoint diversity loss to encourage distinct final positions across run types

    Curriculum learning is implemented by gradually increasing the weights of different loss
    components over epochs to stabilize training and encourage better performance.

    Args:
        model (nn.Module): The TemporalRunnerGNN model to train.
        device (torch.device): Device to perform training on (e.g., 'cuda' or 'cpu').
        dataloader (DataLoader): DataLoader providing batches of training data.
        num_epochs (int, optional): Number of training epochs. Default is 10.
        output_file (str, optional): File path to save the trained model weights. Default is "temporal_runner_gnn_v4.pth".

    Training loop details:
        - Uses AdamW optimizer with learning rate 1e-3 and weight decay 1e-4.
        - Applies gradient clipping with max norm 1.0.
        - For the first 10 epochs, only position and speed losses are active.
        - Gradually adds run-type classification, shot loss, and diversity losses in later epochs.
        - Losses are weighted by hyperparameters that change per epoch range to implement curriculum.

    Returns:
        Model. Return The trained model and the model weights are saved to the specified output_file.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    lambda_vel = 1.0

    v_max = 9.5
    dt = 0.1
    T_intent = 10

    shot_weight = 5.0
    goal_weight = 10.0
    base_weight = 1.0

    ce_loss = nn.CrossEntropyLoss()

    model.train()

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        
        if epoch < 10:
            lambda_type = 0.0
            lambda_shot = 0.0
            lambda_div = 0.0
            lambda_end = 0.0
            lambda_acc = 0.0
            lambda_speed = 0.2
        elif epoch < 20:
            lambda_type = 1.0
            lambda_shot = 0.0
            lambda_div = 0.0
            lambda_end = 0.0
            lambda_acc = 0.05
            lambda_speed = 0.5
        elif epoch < 30:
            lambda_type = 1.0
            lambda_shot = 1.0
            lambda_div = 0.05
            lambda_end = 0.05
        elif epoch < 40:
            lambda_div = 0.2
            lambda_end = 0.1
            lambda_acc = 0.1
        else:
            lambda_div = 0.3
            lambda_end = 0.15
            lambda_acc = 0.15
            lambda_speed = 0.75
        
        for (batch_graphs, padded_targets, lengths, shot_labels, goal_labels, run_type_labels) in dataloader:

            batch_graphs = batch_graphs.to(device)
            padded_targets = padded_targets.to(device)
            shot_labels = shot_labels.to(device)
            goal_labels = goal_labels.to(device)
            run_type_labels = run_type_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_graphs, lengths)

            run_type_logits = outputs["run_type_logits"]      
            shot_logits_per_type = outputs["shot_logits_per_type"] 
            run_paths_per_type = outputs["run_paths_per_type"]    

            # Run-type classification loss
            loss_type = ce_loss(run_type_logits, run_type_labels)

            # Shot probability loss (only for true run type)
            batch_size = run_type_labels.size(0)
            batch_indices = torch.arange(batch_size, device=device)

            shot_logits_true_type = shot_logits_per_type[batch_indices, run_type_labels] 
            shot_loss = F.binary_cross_entropy_with_logits(shot_logits_true_type, shot_labels)

            # Trajectory loss on predicted path for true run type
            loss = 0.0

            for i, length in enumerate(lengths):
                if length <= T_intent + 1:
                    continue

                # Select predicted path for true run type i
                pred_seq = run_paths_per_type[i, run_type_labels[i], :length]
                target_seq = padded_targets[i, :length] 

                # Position loss
                loss_pos = F.mse_loss(pred_seq, target_seq)

                # Velocity loss
                pred_vel = (pred_seq[1:] - pred_seq[:-1]) / dt
                target_vel = (target_seq[1:] - target_seq[:-1]) / dt
                loss_vel = F.mse_loss(pred_vel, target_vel)

                # Speed penalty
                speed = torch.norm(pred_vel, dim=-1)
                excess_speed = torch.relu(speed - v_max)
                loss_speed = torch.mean(excess_speed ** 2)

                # Acceleration loss
                if pred_vel.size(0) > 1:
                    pred_acc = (pred_vel[1:] - pred_vel[:-1]) / dt
                    target_acc = (target_vel[1:] - target_vel[:-1]) / dt
                    loss_acc = F.mse_loss(pred_acc, target_acc)
                else:
                    loss_acc = 0.0

                sample_loss = (
                    loss_pos
                    + lambda_vel * loss_vel
                    + lambda_acc * loss_acc
                    + lambda_speed * loss_speed
                )

                sample_weight = (
                    base_weight
                    + shot_weight * shot_labels[i]
                    + goal_weight * goal_labels[i]
                )

                loss += sample_loss * sample_weight

            loss = loss / len(lengths)

            total_loss_val = loss + lambda_type * loss_type + lambda_shot * shot_loss + lambda_div * trajectory_diversity_loss(run_paths_per_type, lengths) + lambda_end * endpoint_diversity_loss(run_paths_per_type, lengths)

            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += total_loss_val.item()

        # print(f"Epoch {epoch+1}: loss = {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), output_file)
    
    return model
    
def predict_optimal_run(run,model,tracking_frame_groups,device,player_to_team,run_type_vocab):
    """
    Predict optimal run types, shot probabilities, and future run paths for a given run.

    Args:
        run (pd.Series): A row from dynamic event data off ball run dataframe
        model (nn.Module): Trained TemporalRunnerGNN model.
        tracking_frame_groups (dict): Grouped tracking DataFrame indexed by (match_id, event_id).
        device (torch.device): Device to run inference on.
        player_to_team (pd.DataFrame): Mapping from player IDs to their team IDs.
        run_type_vocab (dict): Mapping from run type string to integer index.

    Returns:
        dict: Keys are run types (str), values are dicts with:
              - "run_type_prob": predicted probability of that run type,
              - "shot_prob": predicted shot probability for that run type,
              - "run_path": predicted future run path (numpy array).
    """
    match_id = run['match_id']
    run_id = run['event_id']
    runner_id = run['player_id']
    team_id = run["team_id"]
    frame_start = int(run['frame_start_run'])
    frame_end = int(run['frame_end_run'])

    run_tracking = tracking_frame_groups[(match_id, run_id)].sort_values('frame_id')

    graphs = []
    for t in range(frame_start, frame_end + 1):
        frame_df = run_tracking[run_tracking['frame_id'] == t]
        g = build_graph_from_frame(
            frame_df=frame_df,
            runner_player_id=runner_id,
            player_to_team=player_to_team
        )
        graphs.append(g)
        
    graphs_batch = Batch.from_data_list(graphs)
    length = len(graphs)
        
    lengths = [length]
    graphs_batch = graphs_batch.to(device)

    with torch.no_grad():
        output = model(graphs_batch, lengths)  
    run_type_probs = output["run_type_probs"].squeeze(0)
    shot_probs_per_type = output["shot_probs_per_type"].squeeze(0)
    run_paths_per_type = output["run_paths_per_type"].squeeze(0)
    shot_probs_per_type = shot_probs_per_type.numpy()
    run_type_probs = run_type_probs.numpy()
    run_paths_per_type = run_paths_per_type.numpy()

    x0, y0 = run[["x_start","y_start"]]

    run_type_dict = {
        run_type: {
            "run_type_prob": run_type_probs[idx],
            "shot_prob": shot_probs_per_type[idx],
            "run_path": run_paths_per_type[idx] + [x0,y0],
        }
        for run_type, idx in run_type_vocab.items()
    }
    return run_type_dict