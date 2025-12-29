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
    Constructs a graph representation of player and ball states for a single frame in a soccer match.

    Args:
        frame_df (pd.DataFrame): DataFrame containing tracking data for all players and the ball at one specific frame.
                                 Must include columns: ["player", "x", "y", "dx", "dy", "s", "d", "acceleration", "acc_direction",
                                 "ball_x", "ball_y", "ball_dx", "ball_dy", "ball_speed", "ball_speed_direction",
                                 "ball_acceleration", "ball_acc_direction"].
        runner_player_id (int): The player ID of the runner (target player) in this frame.
        player_to_team (pd.DataFrame): DataFrame mapping player IDs to their team IDs.

    Returns:
        torch_geometric.data.Data: A graph data object with the following fields:
            - x (Tensor): Node features matrix of shape [num_nodes, feature_dim]. Features include player and ball spatial,
                          velocity, acceleration, runner and ball flags.
            - edge_index (LongTensor): Edge connectivity matrix of shape [2, num_edges], fully connected excluding self-loops.
            - edge_attr (Tensor): Edge attributes matrix of shape [num_edges, 4], containing relative distance, same-team flag,
                                  ball edge flag, and relative speed along edge direction.
            - runner_idx (LongTensor): Index of the runner player node within the node list.
            - team_ids (Tensor): Team IDs for each node, with -1 for the ball node.
            - is_ball (Tensor): Boolean mask indicating which node corresponds to the ball.

    Description:
        - Creates player nodes with positional, velocity, and acceleration features.
        - Adds the ball as an additional node with its own features.
        - Marks the runner player node with a dedicated flag.
        - Constructs edges connecting all nodes except self-connections.
        - Computes edge attributes capturing spatial and team relationship dynamics.

    This graph structure serves as input to graph neural networks for spatial-temporal modeling of player and ball interactions.
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

    # Edges an edge attributes
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

    rel_vel = vel[col] - vel[row]  
    rel_speed = torch.sum(rel_vel * direction, dim=1)

    same_team = (
        (team_ids[row] == team_ids[col]) &
        (team_ids[row] != -1)
    ).float()

    ball_edge = (is_ball[row] | is_ball[col]).float()

    edge_attr = torch.stack(
        [dist, same_team, ball_edge, rel_speed],
        dim=1
    )

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
    PyTorch Dataset for preparing graph-based sequential data of player runs in soccer matches.

    Args:
        tracking_df (dict or pd.DataFrameGroupBy): Dictionary or grouped DataFrame where keys are (match_id, run_id) tuples,
            and values are DataFrames containing tracking data for all players and ball across frames within that run.
        run_features (pd.DataFrame): DataFrame containing metadata for each run, including:
            - "match_id": match identifier
            - "event_id": run event identifier
            - "player_id": runner player ID
            - "frame_start_run": start frame index of the run
            - "frame_end_run": end frame index of the run
            - "possession_lead_to_shot": binary label indicating if run led to a shot
            - "possession_lead_to_goal": binary label indicating if run led to a goal
            - "id": run event ID (used for label lookup)
        player_to_team (pd.DataFrame): DataFrame mapping player IDs to their team IDs.

    Methods:
        __len__():
            Returns the number of runs available.

        __getitem__(idx):
            For a given index, returns:
                - graphs: List of graph objects representing each frame in the run, created by `build_graph_from_frame`.
                - target_path: Tensor of shape [run_length, 2] containing runner's x,y positions relative to run start.
                - shot_label: Tensor scalar (0.0 or 1.0) indicating if the run led to a shot.
                - goal_label: Tensor scalar (0.0 or 1.0) indicating if the run led to a goal.

    Description:
        For each run, this dataset extracts the sequence of frames, converts each frame into a graph of players and ball,
        normalizes runner positions relative to the run start frame, and retrieves binary outcome labels for shot and goal.
        The dataset facilitates training spatial-temporal GNN models for trajectory and event prediction tasks.
    """
    def __init__(self, tracking_df, run_features, player_to_team):
        self.df = tracking_df
        self.run_features = run_features
        self.player_to_team = player_to_team

    def __len__(self):
        return len(self.run_features)

    def __getitem__(self, idx):
        run = self.run_features.iloc[idx]

        match_id = run["match_id"]
        run_id = run["event_id"]
        runner_id = run["player_id"]

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

        for t in range(frame_start, frame_end + 1):
            frame_df = run_tracking[run_tracking["frame_id"] == t]

            g = build_graph_from_frame(
                frame_df=frame_df,
                runner_player_id=runner_id,
                player_to_team=self.player_to_team
            )
            graphs.append(g)

            runner_row = frame_df[frame_df["player"] == runner_id].iloc[0]
            runner_positions.append([
                runner_row["x"] - x0,
                runner_row["y"] - y0
            ])

        target_path = torch.tensor(runner_positions, dtype=torch.float)  # [L,2]
        
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

        return graphs, target_path, shot_label, goal_label

def collate_fn(batch, max_len=100):
    """
    Custom collate function to prepare batches of graph sequences with padded target trajectories
    and associated binary labels for model training.

    Args:
        batch (list): List of tuples returned by TemporalRunnerDataset.__getitem__, each containing:
            - graphs (list of Data): List of graph objects for each frame in a run.
            - target_path (Tensor): Runner's relative x,y positions, shape [run_length, 2].
            - shot_label (Tensor): Binary label indicating if run led to a shot.
            - goal_label (Tensor): Binary label indicating if run led to a goal.
        max_len (int, optional): Maximum length to pad or truncate target trajectories. Defaults to 100.

    Returns:
        batch_graphs (Batch): Batched graph object combining all graphs in the batch (across all runs and frames).
        padded_targets (Tensor): Tensor of shape [batch_size, max_len, 2] containing zero-padded or truncated target trajectories.
        lengths (list[int]): List of actual run lengths (capped at max_len) for each example in the batch.
        shot_labels (Tensor): Tensor of shape [batch_size] with binary shot labels.
        goal_labels (Tensor): Tensor of shape [batch_size] with binary goal labels.

    Notes:
        - The function flattens the list of graph sequences into a single list for batching.
        - Target trajectories shorter than max_len are zero-padded; longer ones are truncated.
        - This collate function enables variable-length sequences to be batched efficiently for GNN + temporal models.
    """
    graphs_list, targets_list, shot_labels_list, goal_labels_list = zip(*batch)

    batch_graphs = Batch.from_data_list([g for graphs in graphs_list for g in graphs])

    lengths = [len(target) for target in targets_list]

    padded_targets = torch.zeros(len(targets_list), max_len, 2) 

    for i, target in enumerate(targets_list):
        length = min(len(target), max_len)
        padded_targets[i, :length] = target[:length]

    lengths = [min(l, max_len) for l in lengths]
    shot_labels = torch.tensor(shot_labels_list, dtype=torch.float)
    goal_labels = torch.tensor(goal_labels_list, dtype=torch.float)

    return batch_graphs, padded_targets, lengths, shot_labels, goal_labels



class TemporalRunnerGNN(nn.Module):
    """
    TemporalRunnerGNN predicts the trajectory of a soccer player (runner) over time using spatial-temporal graph neural networks.

    Architecture:
    - Two-layer Transformer-based Graph Neural Network (TransformerConv) to encode spatial relationships between players and the ball per frame.
    - Learned temporal positional encoding added to the runner node embeddings sequence.
    - Transformer Encoder layers to model temporal dependencies across frames.
    - Final linear layer to predict 2D runner positions (x, y) for each frame.

    Args:
        node_feat_dim (int): Number of input features per node.
        edge_dim (int): Number of features per edge.
        gnn_hidden_dim (int): Hidden dimension size for GNN layers and temporal encoder.

    Forward Input:
        graphs_batch (torch_geometric.data.Batch): Batched graph data containing all frames in the batch.
        lengths (list[int]): List containing the length (number of frames) of each run in the batch.

    Forward Output:
        pred_path (Tensor): Predicted runner trajectories of shape (batch_size, max_length, 2), 
            where max_length is the longest run length in the batch.

    Notes:
    - The model extracts embeddings specifically for the runner node from each frame graph.
    - Runner embeddings are padded to the maximum sequence length in the batch for efficient temporal encoding.
    - Causal masking prevents the Transformer from attending to future frames.
    - Padding mask ensures frames beyond run length are ignored in the Transformer.
    """
    
    def __init__(self, node_feat_dim, edge_dim, gnn_hidden_dim):
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

        self.temporal_pe = nn.Parameter(
            torch.randn(1, 100, gnn_hidden_dim) * 0.01  # max 100 frames
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
        
        self.pos_head = nn.Linear(gnn_hidden_dim, 2)

    def forward(self, graphs_batch, lengths):
        device = next(self.parameters()).device
        graphs_batch = graphs_batch.to(device)

        x, edge_index, edge_attr = graphs_batch.x, graphs_batch.edge_index,graphs_batch.edge_attr
        edge_attr = edge_attr / edge_attr.std(dim=0, keepdim=True).clamp(min=1e-6)

        x = F.relu(self.gnn1(x, edge_index, edge_attr))
        x = F.relu(self.gnn2(x, edge_index, edge_attr))

        batch_vec = graphs_batch.batch 
        runner_idx = graphs_batch.runner_idx  

        total_graphs = runner_idx.size(0)   

        B = len(lengths)
        max_len = max(lengths)

        runner_embeds_per_graph = []

        for t in range(total_graphs):
            node_mask = (batch_vec == t)
            node_indices = node_mask.nonzero(as_tuple=False).view(-1)
            global_runner_idx = node_indices[runner_idx[t]]
            runner_embeds_per_graph.append(x[global_runner_idx])

        runner_embeds_per_run = []
        start = 0
        for length in lengths:
            run_embeds = runner_embeds_per_graph[start:start+length]
            start += length

            if length < max_len:
                padding = [torch.zeros_like(run_embeds[0])]*(max_len - length)
                run_embeds.extend(padding)

            run_embeds = torch.stack(run_embeds)
            runner_embeds_per_run.append(run_embeds)

        runner_embeds_batch = torch.stack(runner_embeds_per_run)
        
        T = runner_embeds_batch.size(1)

        x = runner_embeds_batch + self.temporal_pe[:, :T]

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        # padding mask
        padding_mask = torch.arange(T, device=x.device)[None, :] >= torch.tensor(lengths, device=x.device)[:, None]

        out = self.temporal_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        pred_path = self.pos_head(out)
        return pred_path
    
    
def train_model(model, device, dataloader,num_epochs = 10,output_file = "temporal_runner_gnn_v2.pth"):
    """
    Train the TemporalRunnerGNN model on run trajectory data with weighted loss components.

    Args:
        model (nn.Module): The TemporalRunnerGNN model instance to train.
        device (torch.device): Device (CPU/GPU) for computation.
        dataloader (DataLoader): PyTorch DataLoader providing batches of graph data, targets, and labels.
        num_epochs (int, optional): Number of training epochs. Default is 10.
        output_file (str, optional): File path to save the trained model weights. Default is 'temporal_runner_gnn_v2.pth'.

    Training Details:
        - Uses Adam optimizer with learning rate 1e-3.
        - Loss is a weighted sum of:
            - Position MSE loss
            - Velocity MSE loss (first-order difference)
            - Acceleration MSE loss (second-order difference)
            - Speed penalty for exceeding maximum speed (v_max=9.5)
        - Sample weights increase loss contribution for runs leading to shots and goals.
        - Uses exponential moving average (EMA) for stable loss tracking.
        - Loss components lambda weights:
            - Velocity loss: 1.0
            - Acceleration loss: 0.1
            - Speed penalty: 0.5
            - Shot weight: 2.0
            - Goal weight: 3.0
            - Base weight: 1.0

    Returns:
        model (nn.Module): Trained model instance with updated weights.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lambda_vel = 1.0
    lambda_acc = .1
    lambda_speed = .5
    
    v_max = 9.5

    shot_weight = 2.0
    goal_weight = 3.0
    base_weight = 1.0
        
    model.train()
    
    for epoch in range(num_epochs):
        
        total_loss = 0
        for batch_graphs, padded_targets, lengths, shot_labels, goal_labels in tqdm(dataloader):
            batch_graphs = batch_graphs.to(device)
            padded_targets = padded_targets.to(device)
            shot_labels = shot_labels.to(device)
            goal_labels = goal_labels.to(device)
            
            optimizer.zero_grad()

            pred_path = model(batch_graphs, lengths) 

            loss = 0
            for i, length in enumerate(lengths):
                pred_seq = pred_path[i, :length]
                target_seq = padded_targets[i, :length]

                loss_pos = F.mse_loss(pred_seq, target_seq)

                if length > 1:
                    pred_vel = pred_seq[1:] - pred_seq[:-1]
                    target_vel = target_seq[1:] - target_seq[:-1]
                    loss_vel = F.mse_loss(pred_vel, target_vel)
                    
                    pred_vel = (pred_seq[1:] - pred_seq[:-1]) / .1
                    speed = torch.norm(pred_vel, dim=-1)

                    excess_speed = torch.relu(speed - v_max)
                    loss_speed = torch.mean(excess_speed ** 2)
                else:
                    loss_vel = 0

                if length > 2:
                    pred_acc = (pred_vel[1:] - pred_vel[:-1])
                    target_acc = target_vel[1:] - target_vel[:-1]
                    loss_acc = F.mse_loss(pred_acc, target_acc)
                    
                else:
                    loss_acc = 0


                sample_loss = loss_pos + lambda_vel * loss_vel + lambda_acc * loss_acc + lambda_speed * loss_speed

                # Calculate sample weight
                sample_weight = base_weight + shot_weight * shot_labels[i] + goal_weight * goal_labels[i]

                loss += sample_loss * sample_weight

            loss /= len(lengths)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), output_file)
    
    return model

def predict_optimal_run(run,model,tracking_frame_groups,device,player_to_team):
    """
    Predict the optimal run trajectory for a given run using the trained TemporalRunnerGNN model.

    Args:
        run (pd.Series): A single run event containing metadata such as match_id, event_id, player_id,
                         team_id, frame start/end, and start position (x_start, y_start).
        model (nn.Module): Trained TemporalRunnerGNN model used to predict trajectories.
        tracking_frame_groups (pd.DataFrameGroupBy): Grouped tracking data indexed by (match_id, event_id).
        device (torch.device): Device (CPU/GPU) on which to run the model.
        player_to_team (pd.DataFrame): Mapping from player IDs to team IDs.

    Returns:
        torch.Tensor: Predicted absolute runner positions over time (tensor shape: [run_length, 2]),
                      shifted by the run's starting position to absolute field coordinates.

    Process:
        - Extracts tracking frames for the run from start to end frame.
        - Converts each frame to a graph with player and ball features.
        - Batches the graphs and feeds through the model for trajectory prediction.
        - Converts predicted relative positions back to absolute coordinates by adding start position.
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
        pred_path = model(graphs_batch, lengths)  

    pred_path = pred_path.squeeze(0)  
    run_length = lengths[0]
    pred_path = pred_path[:run_length]

    x0, y0 = run[["x_start","y_start"]] 
    absolute_path = pred_path + torch.tensor([x0, y0], device=pred_path.device)
    return absolute_path

def compute_losses(true_path, pred_path):
    """
    true_path, pred_path: tensors of shape (T, 2) [x, y]
    Calculate position, velocity, and acceleration losses.
    """

    # Position loss (MSE)
    loss_pos = F.mse_loss(pred_path, true_path)

    # Velocity (first difference)
    true_vel = true_path[1:] - true_path[:-1]
    pred_vel = pred_path[1:] - pred_path[:-1]
    loss_vel = F.mse_loss(pred_vel, true_vel)

    # Acceleration (second difference)
    true_acc = true_vel[1:] - true_vel[:-1]
    pred_acc = pred_vel[1:] - pred_vel[:-1]
    loss_acc = F.mse_loss(pred_acc, true_acc)

    return loss_pos.item(), loss_vel.item(), loss_acc.item()

def evaluate_all_runs(runs_to_predict, model, tracking_frame_groups, device, player_to_team):
    """
    Evaluate the TemporalRunnerGNN model's predictions across multiple runs.

    Args:
        runs_to_predict (pd.DataFrame): DataFrame containing runs metadata to be evaluated.
        model (nn.Module): Trained TemporalRunnerGNN model used for predictions.
        tracking_frame_groups (pd.DataFrameGroupBy): Grouped tracking data indexed by (match_id, event_id).
        device (torch.device): Device (CPU/GPU) to perform computations on.
        player_to_team (pd.DataFrame): Mapping from player IDs to their team IDs.

    Returns:
        list of dict: A list containing dictionaries for each run with the following keys:
            - 'match_id': Match identifier.
            - 'run_id': Run event identifier.
            - 'loss_pos': Position mean squared error loss.
            - 'loss_vel': Velocity mean squared error loss.
            - 'loss_acc': Acceleration mean squared error loss.
            - 'possession_lead_to_shot': Indicator if run led to a shot (0 or 1).
            - 'possession_lead_to_goal': Indicator if run led to a goal (0 or 1).

    Process:
        - Sets the model to evaluation mode.
        - Iterates over each run in the dataset.
        - Predicts the run trajectory using the model.
        - Retrieves the true trajectory from tracking data.
        - Computes losses between predicted and true trajectories.
        - Collects and returns results for all runs.
    """
    model.eval()
    results = []

    for idx, run in runs_to_predict.iterrows():
        # Predict run path
        try:
            absolute_path = predict_optimal_run(run, model, tracking_frame_groups, device, player_to_team)
        except:
            pass
        
        # Extract true trajectory from tracking data
        match_id = run['match_id']
        run_id = run['event_id']
        runner_id = run['player_id']
        frame_start = int(run['frame_start_run'])
        frame_end = int(run['frame_end_run'])

        run_tracking = tracking_frame_groups[(match_id, run_id)].sort_values('frame_id')
        true_points_df = run_tracking[(run_tracking['player'] == runner_id) & 
                                     (run_tracking['frame_id'] >= frame_start) & 
                                     (run_tracking['frame_id'] <= frame_end)]

        true_path = torch.tensor(true_points_df[['x', 'y']].values, device=device, dtype=torch.float32)
        
        min_len = min(len(true_path), len(absolute_path))
        true_path = true_path[:min_len]
        pred_path = absolute_path[:min_len]

        # Compute losses
        loss_pos, loss_vel, loss_acc = compute_losses(true_path, pred_path)

        shot_flag = int(run.get("possession_lead_to_shot", 0))
        goal_flag = int(run.get("possession_lead_to_goal", 0))

        results.append({
            'match_id': match_id,
            'run_id': run_id,
            'loss_pos': loss_pos,
            'loss_vel': loss_vel,
            'loss_acc': loss_acc,
            'possession_lead_to_shot': shot_flag,
            'possession_lead_to_goal': goal_flag
        })

    return results