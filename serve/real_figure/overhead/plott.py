#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt

def main(output_path):
    # Hard-coded results
    qps = [0.1, 1, 10, 100, 1000]
    latency_ms = [0.02, 0.10, 0.88, 2.77, 2.82]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the single line
    ax.plot(
        qps,
        latency_ms,
        marker='o',
        markersize=10,
        linewidth=5,
        label='Avg inform_new latency'
    )

    # Log-scale on the x-axis
    ax.set_xscale('log')

    # Labels and title
    ax.set_xlabel('QPS (query/sec)', fontsize=16)
    ax.set_ylabel('Latency (ms)', fontsize=16)
    ax.set_title('Policy Optimizer Runtime Overhead vs QPS', fontsize=16)

    # Tick label sizes
    ax.tick_params(axis='both', labelsize=14)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Tight layout
    fig.tight_layout()

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Save average inform_new latency vs QPS plot'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output PNG path (required)',
        required=True
    )
    args = parser.parse_args()
    main(args.output)
