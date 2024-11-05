from dataclasses import dataclass


@dataclass
class PlottingOptions:
    angle_rotate_yaxis_labels: int = 60
    angle_rotate_xaxis_labels: int = 10
    plot_axis_font_size: int = 8
    plot_axis_tick_size: int = 8
    plot_title_font_size: int = 20
    plot_colour_scheme: str = "classic"
