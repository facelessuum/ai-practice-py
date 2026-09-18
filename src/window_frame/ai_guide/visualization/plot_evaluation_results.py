"""A browser-readable accuracy and confidence report, with no extra dependency."""
from html import escape


def plot_evaluation_results(metrics, path):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="750">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="14">',
           '<text x="30" y="30" font-size="20">Window-frame evaluation</text>',
           '<text x="30" y="60">Class overlap (IoU): higher is better; gray = no examples/predictions</text>']
    for index, (name, value) in enumerate(metrics['class_iou'].items()):
        y = 85 + index * 34
        svg.append(f'<text x="30" y="{y+16}">{escape(name)}</text>')
        svg.append(f'<rect x="210" y="{y}" width="500" height="22" fill="#eee"/>')
        if value is not None:
            svg.append(f'<rect x="210" y="{y}" width="{500*value:.1f}" height="22" fill="#257cac"/>')
        text = 'N/A' if value is None else f'{value:.3f}'
        svg.append(f'<text x="730" y="{y+16}">{text}</text>')
    svg.extend([
        '<text x="30" y="335">Confidence reliability: points closer to the diagonal are better.</text>',
        '<path d="M70 660V370M70 660H360" stroke="black" fill="none"/>',
        '<path d="M70 660L360 370" stroke="#aaa" stroke-dasharray="5 5"/>',
        '<text x="80" y="690">Mean confidence: 0 to 1 (left to right)</text>',
        '<text x="30" y="355">Correct fraction: 0 to 1 (bottom to top)</text>',
    ])
    for count, confidence, correct in metrics['confidence_bins']:
        if count:
            x, y = 70 + 290 * confidence / count, 660 - 290 * correct / count
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#257cac"/>')
    keys = [('frame_iou', 'Frame overlap'), ('boundary_precision', 'Boundary precision'),
            ('boundary_recall', 'Boundary recall'), ('frame_mae', 'Enhanced frame error'),
            ('unchanged_input_frame_mae', 'Unchanged-input error'), ('frame_improvement', 'Improvement over no edit'),
            ('outside_mae', 'Background change'), ('protected_mae', 'Protected-region change'),
            ('confidence_calibration_error', 'Confidence mismatch')]
    for index, (key, label) in enumerate(keys):
        value = metrics[key]
        text = 'N/A' if value is None else f'{value:.5f}'
        svg.append(f'<text x="420" y="{380+index*30}">{escape(label)}: {text}</text>')
    svg.extend(['<text x="30" y="730">Confidence describes class certainty, not whether an enhancement looks attractive.</text>', '</g></svg>'])
    path.write_text('\n'.join(svg))
