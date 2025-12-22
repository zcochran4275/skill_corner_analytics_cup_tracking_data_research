import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_soccer_pitch(
    field_dimen: tuple[float, float] = (106.0, 68.0),
    pitch_color: str = "mediumseagreen",
    linewidth: int = 2,
    markersize: int = 20,
    fig: plt.figure = None,
    ax: plt.axes = None,
) -> tuple[plt.figure, plt.axes]:
    """A function to plot a soccer pitch
    Note: relies heavily on https://github.com/Friends-of-Tracking-Data-FoTD/
    LaurieOnTracking/blob/master/Metrica_Viz.py

    Args:
        field_dimen (tuple, optional): x and y length of pitch in meters. Defaults to
            (106.0, 68.0).
        pitch_color (str, optional): Color of the pitch. Defaults to "mediumseagreen".
        linewidth (int, optional): Width of the lines on the pitch. Defaults to 2.
        markersize (int, optional): Size of the dots on the pitch. Defaults to 20.
        fig (plt.figure, optional): Figure to plot the pitch on. Defaults to None.
        ax (plt.axes, optional): Axes to plot the pitch on. Defaults to None.

    Returns:
        Tuple[plt.figure, plt.axes]: figure and axes with the pitch depicted on it
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Set pitch and line colors
    ax.set_facecolor(pitch_color)
    if pitch_color not in ["white", "w"]:
        lc = "whitesmoke"  # line color
        pc = "w"  # 'spot' colors
    else:
        lc = "k"
        pc = "k"

    # All dimensions in meters
    border_dimen = (3, 3)  # include a border arround of the field of width 3m
    half_pitch_length = field_dimen[0] / 2.0  # length of half pitch
    half_pitch_width = field_dimen[1] / 2.0  # width of half pitch

    # Soccer field dimensions are in yards, so we need to convert them to meters
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    box_circle_length = 8 * meters_per_yard
    box_circle_radius = 10 * meters_per_yard
    box_circle_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard

    zorder = -2

    # Plot half way line
    ax.plot(
        [0, 0],
        [-half_pitch_width, half_pitch_width],
        lc,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.scatter(
        0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize, zorder=zorder
    )
    # Plot center circle
    y = np.linspace(-1, 1, 150) * centre_circle_radius
    x = np.sqrt(centre_circle_radius**2 - y**2)
    ax.plot(x, y, lc, linewidth=linewidth, zorder=zorder)
    ax.plot(-x, y, lc, linewidth=linewidth, zorder=zorder)

    signs = [-1, 1]
    for s in signs:  # plots each line seperately
        # Plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Goal posts & line
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-goal_line_width / 2.0, goal_line_width / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth,
            zorder=zorder - 1,
        )

        # 6 yard box
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [-box_width / 2.0, -box_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [
                s * half_pitch_length - s * box_length,
                s * half_pitch_length - s * box_length,
            ],
            [-box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Penalty area
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [-area_width / 2.0, -area_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            [
                s * half_pitch_length - s * area_length,
                s * half_pitch_length - s * area_length,
            ],
            [-area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Penalty spot
        ax.scatter(
            s * half_pitch_length - s * penalty_spot,
            0.0,
            marker="o",
            facecolor=lc,
            linewidth=0,
            s=markersize,
            zorder=zorder,
        )

        # Corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(
            s * half_pitch_length - s * x,
            -half_pitch_width + y,
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.plot(
            s * half_pitch_length - s * x,
            half_pitch_width - y,
            lc,
            linewidth=linewidth,
            zorder=zorder,
        )

        # Draw the half circles by the box: the D
        y = (
            np.linspace(-1, 1, 50) * box_circle_length
        )  # D_length is the chord of the circle that defines the D
        x = np.sqrt(box_circle_radius**2 - y**2) + box_circle_pos
        ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth, zorder=zorder)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2.0 + border_dimen[0]
    ymax = field_dimen[1] / 2.0 + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)

    return fig, ax

def tracking_long_to_wide(tracking_data):
    
    tracking_data = tracking_data.pivot_table(
        index=["period_id", "timestamp", "frame_id","match_id","run_active","run_id","ball_x",'ball_y','ball_speed'],
        columns="player",
        values=["x", "y", "d", "s"],
        fill_value=None
    )
    tracking_data.columns = [f"{player}_{val}" for val, player in tracking_data.columns]

    tracking_data = tracking_data.reset_index()
    return tracking_data

def plot_run(run,tracking_data,player_to_team,ax=None, plot_ball=True, plot_defense=True, plot_offense=False, title=""):
    player_id = run["player_id"]
    team_id = run["team_id"]

    tracking_data_filtered = tracking_data[(tracking_data["run_id"]==run["event_id"]) & (tracking_data["match_id"]==run["match_id"])]

    run_tracking = tracking_long_to_wide(tracking_data_filtered)

    # Normalize sizes based on timestamp
    timestamps = run_tracking["timestamp"].apply(pd.Timedelta).dt.total_seconds().values
    # Normalize between 20 and 150 (you can adjust these)
    sizes = 1 + 130 * (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-9)

    fig = None
    if ax is None:
        fig, ax = plot_soccer_pitch(pitch_color="white")
    else:
        plot_soccer_pitch(pitch_color="white", ax=ax)
    legend_handles = []

    if plot_ball:
        ball_coords = run_tracking.iloc[:, ["ball" in col for col in run_tracking.columns]]
        ball_x = ball_coords.loc[:, "ball_x"].values
        ball_y = ball_coords.loc[:, "ball_y"].values
        # Use sizes here too
        ax.scatter(ball_x, ball_y, s=sizes, color="black", alpha=0.5, label="Ball Trajectory")
        legend_handles.append(
            plt.Line2D([0], [0], color="black", marker="o", linestyle="", label="Ball Trajectory")
        )

    if plot_defense:
        def_players = player_to_team[player_to_team["team_id"] != team_id].reset_index()["id"].values
        for player in def_players:
            player_coords = run_tracking.iloc[:, [str(player) in col for col in run_tracking.columns]]
            if player_coords.shape[1] == 0:
                continue
            player_x = player_coords.loc[:, f"{player}_x"].values
            player_y = player_coords.loc[:, f"{player}_y"].values

            if (player_x == 0).all() and (player_y == 0).all():
                continue

            ax.scatter(player_x, player_y, s=sizes, color="red", alpha=0.2, label=f"Defensive Player Trajectory")
        legend_handles.append(
            plt.Line2D([0], [0], color="red", marker="o", linestyle="", alpha=0.4, label="Defensive Trajectories")
        )

    if plot_offense:
        off_players = player_to_team[player_to_team["team_id"] == team_id].reset_index()["id"].values
        for player in off_players:
            player_coords = run_tracking.iloc[:, [str(player) in col for col in run_tracking.columns]]
            if player_coords.shape[1] == 0:
                continue
            player_x = player_coords.loc[:, f"{player}_x"].values
            player_y = player_coords.loc[:, f"{player}_y"].values

            ax.scatter(player_x, player_y, s=sizes, color="blue", alpha=0.2, label=f"Offensive Player Trajectory")
        legend_handles.append(
            plt.Line2D([0], [0], color="blue", marker="o", linestyle="", alpha=0.4, label="Offensive Trajectories")
        )

    run_coords = run_tracking.iloc[:, [str(player_id) in col for col in run_tracking.columns]]
    run_x = run_coords.loc[:, f"{player_id}_x"].values
    run_y = run_coords.loc[:, f"{player_id}_y"].values
    active_mask = run_tracking["run_active"].values.astype(bool)

    ax.scatter(run_x[~active_mask], run_y[~active_mask], s=sizes[~active_mask], color="gray", alpha=0.5, label="Before/After Run")
    ax.scatter(run_x[active_mask], run_y[active_mask], s=sizes[active_mask], color="green", alpha=0.8, label="During Run")

    legend_handles.append(
        plt.Line2D([0], [0], color="green", marker="o", linestyle="", label="During Run")
    )
    legend_handles.append(
        plt.Line2D([0], [0], color="gray", marker="o", linestyle="", label="Before/After Run")
    )

    if title == "":
        title = f'Match: {run["match_id"]}, id: {run["event_id"]}, Player: {player_id}'

    ax.set_title(title)
    plt.legend(handles=legend_handles, loc="upper left", frameon=True, title="Legend", fontsize=9, title_fontsize=10)
    #plt.show()
    
def animate_run(run,tracking_data,player_to_team,title=""):
    player_id = run["player_id"]
    team_id = run["team_id"]

    tracking_data_filtered = tracking_data[(tracking_data["run_id"]==run["event_id"]) & (tracking_data["match_id"]==run["match_id"])]

    run_tracking = tracking_long_to_wide(tracking_data_filtered)

    player_ids = [col.split("_")[0] for col in run_tracking.columns if col.endswith("_x")]
    player_ids = sorted(set(player_ids) - {'ball'})  # exclude 'ball' if needed
    valid_player_ids = []

    for pid in player_ids:
        x_col = f"{pid}_x"
        y_col = f"{pid}_y"
        
        # Check if player columns exist (just in case)
        if x_col in run_tracking.columns and y_col in run_tracking.columns:
            all_zero_x = (run_tracking[x_col] == 0).all()
            all_zero_y = (run_tracking[y_col] == 0).all()
            
            # Keep player only if NOT all zeros for both x and y
            if not (all_zero_x and all_zero_y):
                valid_player_ids.append(pid)
    fig, ax = plot_soccer_pitch(pitch_color="white")
    player_to_color = {pid: "g" if int(pid) == player_id else "b" if player_to_team.loc[int(pid)].iloc[0] == team_id else "r" for pid in valid_player_ids}
    all_colors = list(player_to_color.values()) + ['black']
    dummy_coords = np.zeros((len(player_to_color)+1, 2))
    scat = ax.scatter(dummy_coords[:, 0], dummy_coords[:, 1], c=all_colors, s=50)
    #dummy arrows 
    arrow_starts = np.zeros((len(valid_player_ids), 2))
    arrow_dirs = np.zeros((len(valid_player_ids), 2))
    quiv = ax.quiver(
        arrow_starts[:, 0], arrow_starts[:, 1],
        arrow_dirs[:, 0], arrow_dirs[:, 1],
        color=all_colors[:-1], scale=1, scale_units='xy', angles='xy',
        width=0.005/2, headwidth=3/2, headlength=5/2
    )
    #legend
    legend_handles = []
    legend_handles.append(
            plt.Line2D([0], [0], color="black", marker="o", linestyle="", label="Ball"))
    legend_handles.append(
            plt.Line2D([0], [0], color="red", marker="o", linestyle="", alpha=0.4, label="Defensive Players"))
    legend_handles.append(
            plt.Line2D([0], [0], color="blue", marker="o", linestyle="", alpha=0.4, label="Offensive Players"))
    legend_handles.append(
            plt.Line2D([0], [0], color="green", marker="o", linestyle="", label="Player Making the Run"))
    plt.legend(handles=legend_handles, loc="upper left", frameon=True, title="Legend", fontsize=9, title_fontsize=10)
    #title
    if title == "":
        title = f'Match: {run["match_id"]}, id: {run["event_id"]}, Player: {player_id}'
    ax.set_title(title)

    def init():
        current = run_tracking.iloc[0]
        coords = [[current[f"{pid}_x"], current[f"{pid}_y"]] for pid in valid_player_ids]
        coords.append([current["ball_x"], current["ball_y"]])
        scat.set_offsets(coords)
        starts = np.array(coords[:-1])
        directions = np.array([
            [
                current[f"{pid}_s"] * np.cos(current[f"{pid}_d"]) if pd.notna(current[f"{pid}_s"]) and pd.notna(current[f"{pid}_d"]) else 0,
                current[f"{pid}_s"] * np.sin(current[f"{pid}_d"]) if pd.notna(current[f"{pid}_s"]) and pd.notna(current[f"{pid}_d"]) else 0
            ] for pid in valid_player_ids
        ])
        quiv.set_offsets(starts)
        quiv.set_UVC(directions[:, 0], directions[:, 1])
        return scat, quiv

    def update(frame):
        current = run_tracking.iloc[frame]
        coords = [[current[f"{pid}_x"], current[f"{pid}_y"]] for pid in valid_player_ids]
        coords.append([current["ball_x"], current["ball_y"]])
        scat.set_offsets(coords)
        sizes = [50] * len(valid_player_ids) + [50]
        scat.set_sizes(sizes)
        # Update quiver arrows
        starts = np.array([[current[f"{pid}_x"], current[f"{pid}_y"]] for pid in valid_player_ids])
        directions = np.array([
            [
                current[f"{pid}_s"] * np.cos(current[f"{pid}_d"]) if pd.notna(current[f"{pid}_s"]) and pd.notna(current[f"{pid}_d"]) else 0,
                current[f"{pid}_s"] * np.sin(current[f"{pid}_d"]) if pd.notna(current[f"{pid}_s"]) and pd.notna(current[f"{pid}_d"]) else 0
            ] for pid in valid_player_ids
        ])

        quiv.set_offsets(starts)
        quiv.set_UVC(directions[:, 0], directions[:, 1])

        return scat, quiv
    ani = animation.FuncAnimation(fig, update, frames=len(run_tracking), init_func=init, blit=False, interval=100, repeat=True)
    plt.show()
    return ani