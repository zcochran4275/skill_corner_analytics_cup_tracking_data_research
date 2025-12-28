import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.data import Dataset

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TransformerConv

def build_graph_from_frame(frame_df, runner_player_id, player_to_team):
    """
    frame_df: rows = players at ONE frame_id
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

        graphs_batch = Batch.from_data_list(graphs)

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
        
        x_threat_run = self.run_features.loc[self.run_features['id'] == run_id, 'xthreat'].values
        if len(x_threat_run) == 0:
            x_threat_run = torch.tensor(0.0) 
        else:
            x_threat_run = torch.tensor(float(x_threat_run[0]))
        
        x_pass_run = self.run_features.loc[self.run_features['id'] == run_id, 'xpass_completion'].values
        if len(x_pass_run) == 0:
            x_pass_run = torch.tensor(0.0) 
        else:
            x_pass_run = torch.tensor(float(x_pass_run[0]))

        return graphs, target_path, shot_label, goal_label, x_threat_run, x_pass_run

def collate_fn(batch, max_len=100):
    graphs_list, targets_list, shot_labels_list, goal_labels_list, x_threat_list, x_pass_list = zip(*batch)

    batch_graphs = Batch.from_data_list([g for graphs in graphs_list for g in graphs])

    lengths = [len(target) for target in targets_list]

    padded_targets = torch.zeros(len(targets_list), max_len, 2) 

    for i, target in enumerate(targets_list):
        length = min(len(target), max_len)
        padded_targets[i, :length] = target[:length]

    lengths = [min(l, max_len) for l in lengths]
    shot_labels = torch.tensor(shot_labels_list, dtype=torch.float)
    goal_labels = torch.tensor(goal_labels_list, dtype=torch.float)
    x_t_labels = torch.tensor(x_threat_list, dtype=torch.float)
    x_pass_labels = torch.tensor(x_pass_list, dtype=torch.float)


    return batch_graphs, padded_targets, lengths, shot_labels, goal_labels, x_t_labels,x_pass_labels



class TemporalRunnerGNN(nn.Module):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lambda_vel = 1.0
    lambda_acc = .1
    lambda_speed = .5
    
    v_max = 9.5

    shot_weight = 2.0
    goal_weight = 3.0
    base_weight = 1.0

    ema_decay = 0.995

    def update_ema(old, new):
        if old is None:
            return new
        return ema_decay * old + (1 - ema_decay) * new
        
    model.train()
    
    for epoch in range(num_epochs):
        
        total_loss = 0
        for batch_graphs, padded_targets, lengths, shot_labels, goal_labels, x_threat_labels, x_pass_labels in tqdm(dataloader):
            batch_graphs = batch_graphs.to(device)
            padded_targets = padded_targets.to(device)
            shot_labels = shot_labels.to(device)
            goal_labels = goal_labels.to(device)
            x_threat_labels = x_threat_labels.to(device)
            x_pass_labels = x_pass_labels.to(device)
            
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