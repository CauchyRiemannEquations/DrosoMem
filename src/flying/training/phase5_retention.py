"""Preserve the first-32 loss share while expanding the weighted prefix."""
import argparse
import sys

from flying.training import phase5_curriculum as engine
from flying.training.phase5_prefix import NonlinearReadout, sample_weights
from flying.training.phase5_readout import nonlinear_seed

ARMS = ('fixed', 'curriculum', 'anchored')


def validate(cfg):
    if cfg.get('experiment') != 'prefix_share_retention':
        raise ValueError('Expected prefix_share_retention experiment')
    engine.validate({k:v for k,v in cfg.items() if k != 'experiment'})
    if cfg['offsets'] != [0]:
        raise ValueError('This discovery protocol uses only the game starting segment')


def anchored_weights(count, prompt_length, window, anchor_window=32, multiplier=4.):
    """Keep the normalized anchor mass equal to fixed-window weighting.

    Non-anchor weights retain the ordinary curriculum ratios. This is a loss
    allocation constraint, not a guarantee of retaining correct predictions.
    """
    if type(anchor_window) is not int or anchor_window < 1 or window < anchor_window or count <= anchor_window:
        raise ValueError('Invalid anchor window')
    weights = sample_weights(count, prompt_length, window, multiplier)
    start = prompt_length - 1
    outside_mass = weights.sum() - multiplier*anchor_window
    weights[start:start+anchor_window] = multiplier*outside_mass/(count-anchor_window)
    return weights


def fit_head(states, labels, mbon, key, initialization, arm, cfg):
    if arm in ('fixed', 'curriculum'):
        return engine.fit_head(states, labels, mbon, key, initialization, arm, cfg)
    if arm != 'anchored':
        raise ValueError('Unknown retention treatment')
    schedule = [(end, anchored_weights(len(labels), cfg['prompt_length'], window,
                                      cfg['prefix_window'], cfg['prefix_weight']))
                for end,window in zip(engine.endpoints(cfg), cfg['windows'])]
    head = NonlinearReadout(mbon, cfg['hidden_units'], nonlinear_seed(key['seed'], initialization))
    return head.fit(states, labels, epochs=engine.endpoints(cfg)[-1], checkpoints=engine.endpoints(cfg),
                    learning_rate=cfg['learning_rate'], l2=cfg['l2'], sample_weight_schedule=schedule)


def summary(frame, cfg):
    shared = engine.summary(frame, cfg, intervention='anchored', controls=('fixed','curriculum'))
    real = shared['by_offset']['0']['fly']
    recall = real['conditions'] == 6 and all(
        c['paired_mean_delta'] > 0 and c['wins'] >= 4 for c in real['comparisons'].values())
    retention = (real['early_retention']['anchored']['lost_at_final'] <= real['early_retention']['fixed']['lost_at_final']
                 and real['completion_counts']['anchored']['32'] >= real['completion_counts']['fixed']['32'])
    later = real['mean_metrics']['anchored']['later_accuracy'] >= real['mean_metrics']['fixed']['later_accuracy']
    return dict(scope='offset_0_discovery', by_offset=shared['by_offset'],
                criteria=dict(recall=bool(recall), retention=bool(retention), later_accuracy=bool(later)),
                primary_success=bool(recall and retention), joint_success=bool(recall and retention and later))


def run(config_path, output, resume=False, max_conditions=None):
    return engine.run(config_path, output, resume, max_conditions, design=sys.modules[__name__])


def verify(output):
    return engine.verify(output, design=sys.modules[__name__])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase5_retention.json')
    parser.add_argument('--output', default='outputs/phase5_retention')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--verify', action='store_true')
    parser.add_argument('--max-conditions', type=int)
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.config, args.output, args.resume, args.max_conditions)
