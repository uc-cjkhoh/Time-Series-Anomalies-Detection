import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque, defaultdict
from datetime import datetime

class RealtimePlotter:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.category_data = defaultdict(lambda: {
            'times': deque(maxlen=max_points),
            'buy_prices': deque(maxlen=max_points),
            'buy_quantities': deque(maxlen=max_points),
            'total_assets': deque(maxlen=max_points)
        })
        
        # Setup figure with 3 subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Real-Time Market Statistics (by Category)')
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.plot_index = 0
        self.plot_order = [self.ax1, self.ax2, self.ax3]
        
    def update(self, stats_report):
        """Update plots with new stats report data"""
        for category, data in stats_report.items():
            self.category_data[category]['times'].append(data['time'])
            self.category_data[category]['buy_prices'].append(data['buy_price_stats']['mean'])
            self.category_data[category]['buy_quantities'].append(data['buy_quantity_stats']['mean'])
            self.category_data[category]['total_assets'].append(data['total_asset_stats']['mean'])
        
        self._draw_plots()
    
    def _draw_plots(self):
        """Draw scatter points one at a time across plots"""
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot all historical data for each category
        plot_configs = [
            (self.ax1, 'buy_prices', 'Buy Price (Mean)', 'Buy Price Over Time'),
            (self.ax2, 'buy_quantities', 'Buy Quantity (Mean)', 'Buy Quantity Over Time'),
            (self.ax3, 'total_assets', 'Total Asset (Mean)', 'Total Asset Over Time')
        ]
        
        for ax, data_key, ylabel, title in plot_configs:
            for idx, (category, cat_data) in enumerate(self.category_data.items()):
                times = list(cat_data['times'])
                values = list(cat_data[data_key])
                ax.scatter(times, values, label=category, color=self.colors[idx % len(self.colors)], s=30)
                # Add labels only to the last point
                if len(times) > 0:
                    ax.annotate(f'{values[-1]:.2f}', (times[-1], values[-1]), 
                               fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
            
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

# Usage example:
# plotter = RealtimePlotter(max_points=100)
# plotter.update(stats_report)

